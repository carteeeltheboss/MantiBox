"""Device detection and the model fallback ladder (ADR 0003).

Pure logic: devices are injected, so these tests do not depend on the host
having a GPU, on torch being installed, or on nvidia-smi being present.
"""

from __future__ import annotations

import json

import pytest

from maintidoc.runtime.devices import (
    DeviceInfo,
    DeviceKind,
    ResolvedModel,
    _candidate_quantizations,
    detect_devices,
    estimate_parameters,
    estimate_vram_gb,
    resolve_model,
    select_device,
)

QWEN_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_7B_AWQ = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
QWEN_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
CPU_ONLY = DeviceInfo(kind=DeviceKind.CPU, name="CPU")


def gpu(free_gb: float, total_gb: float = 8.0) -> DeviceInfo:
    return DeviceInfo(
        kind=DeviceKind.CUDA,
        name="Test GPU",
        index=0,
        total_vram_gb=total_gb,
        free_vram_gb=free_gb,
    )


class TestEstimateParameters:
    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            (QWEN_7B, 8.29e9),
            (QWEN_7B_AWQ, 8.29e9),
            (QWEN_3B, 3.75e9),
            ("qwen/qwen2.5-vl-3b-instruct", 3.75e9),  # case-insensitive
        ],
    )
    def test_known_models_use_the_nominal_table(self, model_id: str, expected: float) -> None:
        assert estimate_parameters(model_id) == expected

    def test_vision_tower_is_counted(self) -> None:
        """The id says 7B but the model is larger; undercounting causes OOM."""
        assert estimate_parameters(QWEN_7B_AWQ) is not None
        assert estimate_parameters(QWEN_7B_AWQ) > 7e9

    def test_unknown_falls_back_to_the_id_token(self) -> None:
        assert estimate_parameters("some-org/Model-13B-Chat") == 13e9
        assert estimate_parameters("some-org/model-0.5b") == 0.5e9

    def test_no_size_token_is_unknown(self) -> None:
        assert estimate_parameters("some-org/tiny-vision-model") is None
        assert estimate_parameters("") is None

    def test_a_local_path_basename_still_matches(self) -> None:
        assert estimate_parameters("/models/cache/Qwen2.5-VL-3B-Instruct") == 3.75e9


class TestEstimateVram:
    def test_quantized_needs_less_than_full_precision(self) -> None:
        full = estimate_vram_gb(QWEN_7B, quantization="none")
        packed = estimate_vram_gb(QWEN_7B, quantization="awq")
        assert full is not None and packed is not None
        assert packed < full

    def test_headroom_is_included(self) -> None:
        without = estimate_vram_gb(QWEN_3B, quantization="none", headroom_gb=0.0)
        with_default = estimate_vram_gb(QWEN_3B, quantization="none")
        assert without is not None and with_default is not None
        assert with_default > without

    def test_explicit_parameter_count_overrides_the_id(self) -> None:
        measured = estimate_vram_gb(QWEN_7B, quantization="awq", parameters=1e9)
        nominal = estimate_vram_gb(QWEN_7B, quantization="awq")
        assert measured is not None and nominal is not None
        assert measured < nominal

    def test_unknown_model_is_unknown(self) -> None:
        assert estimate_vram_gb("org/nameless-model", quantization="awq") is None


class TestCandidateQuantizations:
    def test_prefers_full_precision_then_configured(self) -> None:
        assert _candidate_quantizations("awq") == ("none", "awq")

    def test_deduplicates_when_already_full_precision(self) -> None:
        assert _candidate_quantizations("none") == ("none",)


class TestDeviceInfo:
    def test_free_vram_preferred_over_total(self) -> None:
        assert gpu(free_gb=5.0, total_gb=8.0).usable_vram_gb == 5.0

    def test_falls_back_to_total_when_free_unknown(self) -> None:
        device = DeviceInfo(kind=DeviceKind.CUDA, name="g", total_vram_gb=8.0)
        assert device.usable_vram_gb == 8.0

    def test_cpu_has_no_vram(self) -> None:
        assert CPU_ONLY.usable_vram_gb is None


class TestDetectDevices:
    def test_cpu_is_always_present_and_last(self) -> None:
        devices = detect_devices()
        assert devices, "detect_devices must always return at least the CPU"
        assert devices[-1].kind is DeviceKind.CPU

    def test_accelerators_are_ordered_before_cpu(self) -> None:
        for device in detect_devices()[:-1]:
            assert device.kind in (DeviceKind.CUDA, DeviceKind.MPS)

    def test_serializes_to_json(self) -> None:
        json.dumps([d.as_dict() for d in detect_devices()])


class TestSelectDevice:
    def test_auto_picks_the_first(self) -> None:
        devices = (gpu(6.0), CPU_ONLY)
        assert select_device("auto", devices=devices).kind is DeviceKind.CUDA

    def test_cpu_only_machine_selects_cpu(self) -> None:
        assert select_device("auto", devices=(CPU_ONLY,)).kind is DeviceKind.CPU

    def test_explicit_preference_is_honoured(self) -> None:
        devices = (gpu(6.0), CPU_ONLY)
        assert select_device("cpu", devices=devices).kind is DeviceKind.CPU

    def test_unsatisfiable_explicit_preference_raises(self) -> None:
        """Pinning CUDA on a machine without one must fail loudly, not silently
        run something else."""
        with pytest.raises(RuntimeError, match="not available"):
            select_device("cuda", devices=(CPU_ONLY,))

    def test_unknown_preference_raises(self) -> None:
        with pytest.raises(ValueError, match="tpu"):
            select_device("tpu", devices=(CPU_ONLY,))


class TestResolveModelFits:
    def test_requested_model_is_kept_when_it_fits(self) -> None:
        resolved = resolve_model(
            requested_model=QWEN_3B,
            fallback_model=None,
            quantization="none",
            device=gpu(free_gb=24.0),
        )
        assert resolved.model_id == QWEN_3B
        assert resolved.fallback_applied is False
        assert resolved.degraded is False

    def test_every_decision_carries_a_reason(self) -> None:
        """ADR 0003: no path through the ladder may be silent."""
        resolved = resolve_model(
            requested_model=QWEN_3B,
            fallback_model=None,
            quantization="none",
            device=gpu(free_gb=24.0),
        )
        assert resolved.reasons
        assert all(isinstance(r, str) and r for r in resolved.reasons)


class TestResolveModelStepDown:
    def test_steps_down_to_the_smaller_model_on_the_gpu(self) -> None:
        """The real 8 GB laptop case: 7B-AWQ does not fit, 3B-AWQ does."""
        resolved = resolve_model(
            requested_model=QWEN_7B_AWQ,
            fallback_model=QWEN_3B,
            quantization="awq",
            device=gpu(free_gb=5.5),
        )
        assert resolved.model_id == QWEN_3B
        assert resolved.device.kind is DeviceKind.CUDA
        assert resolved.fallback_applied is True
        assert resolved.requested_model == QWEN_7B_AWQ
        assert resolved.degraded is False

    def test_step_down_prefers_full_precision_when_it_fits(self) -> None:
        """A smaller model is better served unquantized, if it fits that way.

        Uses synthetic ids because with the real 7B/3B pair this branch is
        unreachable: 3B unquantized (~8.8 GB) needs more VRAM than 7B-AWQ
        (~6.3 GB), so any budget that admits the former already admits the
        latter and no step-down happens at all.
        """
        resolved = resolve_model(
            requested_model="org/Big-70B-Instruct",
            fallback_model="org/Small-1B-Instruct",
            quantization="awq",
            device=gpu(free_gb=10.0, total_gb=10.0),
        )
        assert resolved.model_id == "org/Small-1B-Instruct"
        assert resolved.quantization == "none"
        assert resolved.fallback_applied is True

    def test_step_down_uses_quantization_when_full_precision_does_not_fit(
        self,
    ) -> None:
        resolved = resolve_model(
            requested_model="org/Big-70B-Instruct",
            fallback_model="org/Mid-3B-Instruct",
            quantization="awq",
            device=gpu(free_gb=5.0, total_gb=5.0),
        )
        assert resolved.model_id == "org/Mid-3B-Instruct"
        assert resolved.quantization == "awq"

    def test_fallback_is_never_larger_than_the_requested_model(self) -> None:
        """Regression: the ladder once offered an unquantized 3B (~7.4 GB) as a
        step-down from a 4-bit 7B (~5.5 GB), which is not a step down at all."""
        resolved = resolve_model(
            requested_model=QWEN_7B_AWQ,
            fallback_model=QWEN_3B,
            quantization="awq",
            device=gpu(free_gb=5.5),
        )
        requested_need = estimate_vram_gb(QWEN_7B_AWQ, quantization="awq")
        chosen_need = estimate_vram_gb(resolved.model_id, quantization=resolved.quantization)
        assert requested_need is not None and chosen_need is not None
        assert chosen_need <= requested_need

    def test_reasons_mention_the_spec_deviation(self) -> None:
        resolved = resolve_model(
            requested_model=QWEN_7B_AWQ,
            fallback_model=QWEN_3B,
            quantization="awq",
            device=gpu(free_gb=5.5),
        )
        assert any("ADR 0003" in reason for reason in resolved.reasons)

    def test_reserved_headroom_is_respected(self) -> None:
        """2 GB is reserved for the desktop, so a model that only just fits on
        paper is refused rather than allowed to OOM the display driver."""
        resolved = resolve_model(
            requested_model=QWEN_3B,
            fallback_model=None,
            quantization="none",
            device=gpu(free_gb=9.0),
            min_free_vram_gb=2.0,
        )
        # 3B unquantized needs ~8.8 GB; only 7.0 GB is budgeted.
        assert resolved.device.kind is DeviceKind.CPU
        assert resolved.degraded is True
        assert any("9.0 GB" in reason for reason in resolved.reasons)


class TestResolveModelNoFallback:
    def test_refuses_to_substitute_when_disabled(self) -> None:
        resolved = resolve_model(
            requested_model=QWEN_7B_AWQ,
            fallback_model=QWEN_3B,
            quantization="awq",
            device=gpu(free_gb=5.5),
            allow_fallback=False,
        )
        assert resolved.model_id == QWEN_7B_AWQ
        assert resolved.fallback_applied is False
        assert resolved.degraded is True
        assert any("fallback disabled" in r for r in resolved.reasons)

    def test_falls_back_to_cpu_with_the_smaller_model(self) -> None:
        """When nothing fits, CPU gets the model that can actually complete."""
        resolved = resolve_model(
            requested_model=QWEN_7B_AWQ,
            fallback_model=QWEN_3B,
            quantization="awq",
            device=gpu(free_gb=1.0),
        )
        assert resolved.device.kind is DeviceKind.CPU
        assert resolved.model_id == QWEN_3B
        assert resolved.degraded is True

    def test_cpu_fallback_keeps_requested_model_when_no_fallback_configured(self) -> None:
        resolved = resolve_model(
            requested_model=QWEN_7B,
            fallback_model=None,
            quantization="none",
            device=gpu(free_gb=1.0),
        )
        assert resolved.device.kind is DeviceKind.CPU
        assert resolved.model_id == QWEN_7B


class TestResolveModelUnknowns:
    def test_unknown_model_size_is_attempted_not_refused(self) -> None:
        resolved = resolve_model(
            requested_model="org/some-unlisted-vlm",
            fallback_model=QWEN_3B,
            quantization="awq",
            device=gpu(free_gb=5.5),
        )
        assert resolved.model_id == "org/some-unlisted-vlm"
        assert resolved.fallback_applied is False
        assert any("cannot infer" in r for r in resolved.reasons)

    def test_device_reporting_no_vram_is_attempted(self) -> None:
        device = DeviceInfo(kind=DeviceKind.MPS, name="Apple Silicon (MPS)")
        resolved = resolve_model(
            requested_model=QWEN_7B,
            fallback_model=QWEN_3B,
            quantization="none",
            device=device,
        )
        assert resolved.model_id == QWEN_7B
        assert resolved.fallback_applied is False

    def test_pure_cpu_device_is_flagged_degraded(self) -> None:
        resolved = resolve_model(
            requested_model=QWEN_7B,
            fallback_model=QWEN_3B,
            quantization="none",
            device=CPU_ONLY,
        )
        assert resolved.device.kind is DeviceKind.CPU
        assert resolved.degraded is True
        assert any("CPU" in r for r in resolved.reasons)


class TestResolvedModelSerialization:
    def test_as_dict_is_json_serializable(self) -> None:
        resolved: ResolvedModel = resolve_model(
            requested_model=QWEN_7B_AWQ,
            fallback_model=QWEN_3B,
            quantization="awq",
            device=gpu(free_gb=5.5),
        )
        payload = json.loads(json.dumps(resolved.as_dict()))
        assert payload["model_id"] == QWEN_3B
        assert payload["requested_model"] == QWEN_7B_AWQ
        assert payload["fallback_applied"] is True
        assert isinstance(payload["reasons"], list)
        assert payload["device"]["kind"] == "cuda"
