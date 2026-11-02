"""Runtime device and model resolution.

SPEC §3 pins Qwen2.5-VL-7B and SPEC §11 *assumes* a 24 GB GPU. That assumption
does not hold on every machine this project runs on, so the runtime decides at
startup what can actually be served here — in the order the user specified:

    CUDA  ->  MPS  ->  CPU

Two rules from ADR 0003 make this safe rather than merely convenient:

* A step-down is **never silent**. `ResolvedModel.reasons` is a human-readable
  audit trail, the decision is logged at WARNING, and the model that actually
  answered is written into every `AnswerPayload.provenance` and every MLflow run.
* Nothing here imports torch at module scope. Detection degrades to `nvidia-smi`
  and finally to CPU, so the package stays importable — and the test suite stays
  runnable — on a machine with no ML stack installed at all.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum

from maintidoc.observability import get_logger

logger = get_logger(__name__)


class DeviceKind(StrEnum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


#: Approximate stored bytes per parameter. 4-bit formats carry more than 0.5
#: because scales and zero-points are stored alongside the packed weights.
_BYTES_PER_PARAM: dict[str, float] = {
    "none": 2.1,
    "bf16": 2.1,
    "fp16": 2.1,
    "int8": 1.1,
    "awq": 0.62,
    "bnb4": 0.62,
    "gptq": 0.62,
}

#: Vision tower + KV cache + allocator fragmentation. Qwen2.5-VL keeps a ViT
# alongside the LLM, and CUDA's caching allocator rarely hands back the last
#: few hundred MB, so a model that *just* fits on paper will still OOM.
_HEADROOM_GB = 1.5

#: Matches the size token in a HF model id: "7B", "3b", "72B", "0.5B".
_SIZE_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*([bB])(?![a-zA-Z])")

#: Nominal parameter counts for the models this project configures, taken from
#: their model cards.
#:
#: Why this table exists: parsing the size token out of an id ("Qwen2.5-VL-7B"
#: -> 7e9) *undercounts* vision-language models, because the vision tower is not
#: included in the "7B" figure. Undercounting makes the ladder optimistic, and an
#: optimistic ladder OOMs at load time instead of declining cleanly.
#:
#: These are planning figures, not measurements. `bench-vlm` records the measured
#: peak VRAM and that measurement supersedes this table (ADR 0003).
_NOMINAL_PARAMETERS: dict[str, float] = {
    "qwen2.5-vl-7b-instruct": 8.29e9,
    "qwen2.5-vl-7b-instruct-awq": 8.29e9,
    "qwen2.5-vl-3b-instruct": 3.75e9,
    "qwen2.5-vl-3b-instruct-awq": 3.75e9,
    "qwen2.5-vl-32b-instruct": 32.8e9,
}


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """A device this process could run inference on."""

    kind: DeviceKind
    name: str
    index: int | None = None
    total_vram_gb: float | None = None
    free_vram_gb: float | None = None
    #: False when torch is not installed, i.e. detection used nvidia-smi only.
    torch_visible: bool = False
    #: Can the *installed* runtime actually execute on this device?
    #:
    #: Hardware existing is not the same as torch being able to drive it: a
    #: CUDA-capable GPU plus a CPU-only torch build looks present but raises
    #: "Torch not compiled with CUDA enabled" the moment you move a tensor to it.
    #: Selection must therefore skip unusable devices while `doctor` still
    #: reports them, so the hardware is visible without being chosen.
    usable: bool = True

    @property
    def usable_vram_gb(self) -> float | None:
        return self.free_vram_gb if self.free_vram_gb is not None else self.total_vram_gb

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "index": self.index,
            "total_vram_gb": self.total_vram_gb,
            "free_vram_gb": self.free_vram_gb,
            "torch_visible": self.torch_visible,
            "usable": self.usable,
        }


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """The outcome of the ladder: what to load, where, and why."""

    model_id: str
    device: DeviceInfo
    quantization: str
    requested_model: str
    fallback_applied: bool
    estimated_vram_gb: float
    #: Ordered audit trail. Surfaced in logs, the API response and MLflow.
    reasons: tuple[str, ...] = field(default_factory=tuple)
    #: True when even the last resort is expected to be tight or unusable.
    degraded: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "requested_model": self.requested_model,
            "fallback_applied": self.fallback_applied,
            "quantization": self.quantization,
            "estimated_vram_gb": round(self.estimated_vram_gb, 2),
            "degraded": self.degraded,
            "reasons": list(self.reasons),
            "device": self.device.as_dict(),
        }


def estimate_parameters(model_id: str) -> float | None:
    """Best available parameter count for `model_id`, or None if unknowable.

    Prefers a known nominal figure, then falls back to parsing the size token
    out of the id. Returns None when neither works, so callers treat the
    requirement as unknown instead of assuming the model fits.
    """
    normalized = model_id.strip().lower()
    if normalized in _NOMINAL_PARAMETERS:
        return _NOMINAL_PARAMETERS[normalized]
    basename = normalized.rsplit("/", 1)[-1]
    if basename in _NOMINAL_PARAMETERS:
        return _NOMINAL_PARAMETERS[basename]

    match = _SIZE_TOKEN.search(model_id)
    if match is None:
        return None
    return float(match.group(1)) * 1e9


def _candidate_quantizations(quantization: str) -> tuple[str, ...]:
    """Quantizations to try for a *fallback* model, best quality first.

    A smaller model run unquantized usually beats a smaller model quantized, so
    try full precision first and only fall back to the configured quantization
    when full precision does not fit either.
    """
    ordered = ["none", quantization]
    deduped: list[str] = []
    for candidate in ordered:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def estimate_vram_gb(
    model_id: str,
    *,
    quantization: str,
    parameters: float | None = None,
    headroom_gb: float = _HEADROOM_GB,
) -> float | None:
    """Rough peak VRAM for `model_id` at `quantization`, or None if unknown.

    Deliberately a planning heuristic, not a measurement: it decides *whether to
    attempt* a load. The real numbers come from `bench-vlm` and are recorded in
    an ADR (SPEC §11).
    """
    params = parameters if parameters is not None else estimate_parameters(model_id)
    if params is None:
        return None
    per_param = _BYTES_PER_PARAM.get(quantization, _BYTES_PER_PARAM["none"])
    return (params * per_param) / (1024**3) + headroom_gb


def _query_nvidia_smi() -> list[tuple[str, float, float]]:
    """Read (name, total_gb, free_gb) per GPU from nvidia-smi, if present.

    `shutil.which` is used so we never execute a partial path from PATH blindly,
    and any failure is non-fatal: this is a probe, not a requirement.
    """
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    try:
        completed = subprocess.run(  # noqa: S603 - fixed binary, fixed args, no shell
            [
                binary,
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("nvidia-smi probe failed: %s", exc)
        return []
    if completed.returncode != 0:
        return []

    devices: list[tuple[str, float, float]] = []
    for line in completed.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            devices.append((parts[0], float(parts[1]) / 1024, float(parts[2]) / 1024))
        except ValueError:
            continue
    return devices


def detect_devices() -> tuple[DeviceInfo, ...]:
    """Every device available to this process, best first, CPU always last.

    Works with or without torch installed.
    """
    found: list[DeviceInfo] = []
    torch_cuda = False
    try:
        import torch

        torch_installed = True
        if torch.cuda.is_available():
            torch_cuda = True
            for index in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(index)
                found.append(
                    DeviceInfo(
                        kind=DeviceKind.CUDA,
                        name=torch.cuda.get_device_name(index),
                        index=index,
                        total_vram_gb=total / (1024**3),
                        free_vram_gb=free / (1024**3),
                        torch_visible=True,
                    )
                )
        if torch.backends.mps.is_available():
            found.append(
                DeviceInfo(
                    kind=DeviceKind.MPS,
                    name="Apple Silicon (MPS)",
                    torch_visible=True,
                )
            )
    except ImportError:
        torch_installed = False

    if not torch_cuda:
        # Either torch is absent, or it is a CPU-only build. nvidia-smi still
        # tells us whether CUDA hardware exists, which `doctor` should report.
        for smi_index, (smi_name, smi_total, smi_free) in enumerate(_query_nvidia_smi()):
            if any(d.kind is DeviceKind.CUDA and d.index == smi_index for d in found):
                continue
            found.append(
                DeviceInfo(
                    kind=DeviceKind.CUDA,
                    name=smi_name,
                    index=smi_index,
                    total_vram_gb=smi_total,
                    free_vram_gb=smi_free,
                    torch_visible=torch_installed,
                    # Reached only when torch cannot drive CUDA: either torch is
                    # absent or it is a CPU-only build. The hardware exists, so
                    # report it, but never select it.
                    usable=False,
                )
            )

    found.append(DeviceInfo(kind=DeviceKind.CPU, name="CPU"))
    return tuple(found)


def select_device(
    preference: str = "auto",
    *,
    devices: tuple[DeviceInfo, ...] | None = None,
) -> DeviceInfo:
    """Pick a device honouring `preference` (auto | cuda | mps | cpu).

    An explicit preference that cannot be satisfied is an error, not a silent
    substitution: if you pinned CUDA you want to know CUDA is missing. Devices
    the installed runtime cannot execute on are skipped - a CUDA GPU next to a
    CPU-only torch build is visible to `doctor` but must never be selected.
    """
    available = devices if devices is not None else detect_devices()
    runnable = tuple(device for device in available if device.usable) or (available[-1],)

    if preference == "auto":
        return runnable[0]

    wanted = DeviceKind(preference)
    for device in runnable:
        if device.kind is wanted:
            return device

    present = ", ".join(f"{d.kind.value}{'' if d.usable else ' (unusable)'}" for d in available)
    msg = f"requested device '{preference}' is not available; detected: {present}"
    raise RuntimeError(msg)


def resolve_model(
    *,
    requested_model: str,
    fallback_model: str | None,
    quantization: str,
    device: DeviceInfo,
    allow_fallback: bool = True,
    min_free_vram_gb: float = 0.0,
    parameters: float | None = None,
) -> ResolvedModel:
    """Walk the ladder: requested model -> fallback model -> CPU.

    On a CUDA/MPS device the decision is driven by estimated VRAM against the
    VRAM actually free. On CPU there is nothing to compare against, so the
    requested model is kept but flagged `degraded` — a 7B on CPU is minutes per
    answer, and the caller must be told that rather than discover it.
    """
    reasons: list[str] = []
    accelerator = device.kind in (DeviceKind.CUDA, DeviceKind.MPS)

    if not accelerator:
        reasons.append(f"no accelerator available; running {requested_model} on CPU")
        reasons.append("expect minutes per answer; this is a portability fallback, not a target")
        return ResolvedModel(
            model_id=requested_model,
            device=device,
            quantization=quantization,
            requested_model=requested_model,
            fallback_applied=False,
            estimated_vram_gb=0.0,
            reasons=tuple(reasons),
            degraded=True,
        )

    free = device.usable_vram_gb
    if free is None:
        reasons.append(f"{device.kind.value} reported no VRAM figure; attempting {requested_model}")
        return ResolvedModel(
            model_id=requested_model,
            device=device,
            quantization=quantization,
            requested_model=requested_model,
            fallback_applied=False,
            estimated_vram_gb=0.0,
            reasons=tuple(reasons),
        )

    need = estimate_vram_gb(requested_model, quantization=quantization, parameters=parameters)
    if need is None:
        reasons.append(
            f"cannot infer the size of '{requested_model}' from its id; attempting the load"
        )
        return ResolvedModel(
            model_id=requested_model,
            device=device,
            quantization=quantization,
            requested_model=requested_model,
            fallback_applied=False,
            estimated_vram_gb=0.0,
            reasons=tuple(reasons),
        )

    budget = max(free - min_free_vram_gb, 0.0)
    if need <= budget:
        reasons.append(
            f"{requested_model} at {quantization} needs ~{need:.1f} GB; {free:.1f} GB free — fits"
        )
        return ResolvedModel(
            model_id=requested_model,
            device=device,
            quantization=quantization,
            requested_model=requested_model,
            fallback_applied=False,
            estimated_vram_gb=need,
            reasons=tuple(reasons),
        )

    reasons.append(
        f"{requested_model} at {quantization} needs ~{need:.1f} GB but only {free:.1f} GB is free"
    )

    if not allow_fallback:
        reasons.append("fallback disabled by configuration; refusing to substitute a model")
        return ResolvedModel(
            model_id=requested_model,
            device=device,
            quantization=quantization,
            requested_model=requested_model,
            fallback_applied=False,
            estimated_vram_gb=need,
            reasons=tuple(reasons),
            degraded=True,
        )

    if fallback_model and fallback_model != requested_model:
        for candidate_quant in _candidate_quantizations(quantization):
            fb_need = estimate_vram_gb(fallback_model, quantization=candidate_quant)
            if fb_need is None:
                continue
            if fb_need <= budget:
                reasons.append(
                    f"stepping down to {fallback_model} at {candidate_quant} (~{fb_need:.1f} GB) "
                    "— SPEC §3 names the 7B, so this deviation is recorded in ADR 0003 "
                    "and in every response's provenance"
                )
                return ResolvedModel(
                    model_id=fallback_model,
                    device=device,
                    quantization=candidate_quant,
                    requested_model=requested_model,
                    fallback_applied=True,
                    estimated_vram_gb=fb_need,
                    reasons=tuple(reasons),
                )
            reasons.append(
                f"{fallback_model} at {candidate_quant} needs ~{fb_need:.1f} GB; still too large"
            )

    # Nothing fits on the accelerator. On CPU, prefer the smaller model: AWQ
    # kernels are CUDA-only, so a 7B would load unquantized and need roughly
    # 16 GB of host RAM — more than a typical container is given — while the
    # fallback model has a real chance of completing.
    cpu = DeviceInfo(kind=DeviceKind.CPU, name="CPU")
    cpu_model = fallback_model or requested_model
    if cpu_model != requested_model:
        reasons.append(
            f"no candidate fits on the accelerator; falling back to CPU with {cpu_model}, "
            "which is small enough to actually complete there"
        )
    else:
        reasons.append(
            f"no candidate fits on the accelerator; falling back to CPU with {cpu_model}"
        )
    return ResolvedModel(
        model_id=cpu_model,
        device=cpu,
        quantization="none",
        requested_model=requested_model,
        fallback_applied=True,
        estimated_vram_gb=0.0,
        reasons=tuple(reasons),
        degraded=True,
    )


def resolve_runtime(
    *,
    device_preference: str = "auto",
    requested_model: str,
    fallback_model: str | None,
    quantization: str,
    allow_fallback: bool = True,
    min_free_vram_gb: float = 0.0,
) -> ResolvedModel:
    """Convenience entrypoint: detect the device, then resolve the model."""
    device = select_device(device_preference)
    resolved = resolve_model(
        requested_model=requested_model,
        fallback_model=fallback_model,
        quantization=quantization,
        device=device,
        allow_fallback=allow_fallback,
        min_free_vram_gb=min_free_vram_gb,
    )
    if resolved.fallback_applied or resolved.degraded:
        # ADR 0003: the step-down must be impossible to miss.
        logger.warning(
            "inference degraded: serving %s on %s (requested %s)",
            resolved.model_id,
            resolved.device.kind.value,
            resolved.requested_model,
            extra={
                "resolved_model": resolved.model_id,
                "requested_model": resolved.requested_model,
                "device": resolved.device.kind.value,
                "reasons": "; ".join(resolved.reasons),
            },
        )
    else:
        logger.info("inference ready: %s on %s", resolved.model_id, resolved.device.kind.value)
    return resolved


__all__ = [
    "DeviceInfo",
    "DeviceKind",
    "ResolvedModel",
    "detect_devices",
    "estimate_parameters",
    "estimate_vram_gb",
    "resolve_model",
    "resolve_runtime",
    "select_device",
]
