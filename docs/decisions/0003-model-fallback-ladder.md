# 0003 — Probe CUDA -> MPS -> CPU and step the model down loudly

- **Date:** 2026-11-01
- **Status:** accepted
- **Issue/PR:** runtime device selection (W1/W3)

## Context
SPEC §3 pins **Qwen2.5-VL-7B-Instruct** and SPEC §11 assumes a **24 GB** GPU.
The development machine is an RTX 4070 Laptop with **8 GB total VRAM**, of which
roughly 4–6 GB is free depending on what else is running. A 7B checkpoint is
~16 GB in bf16 and ~5.5 GB at 4-bit (AWQ), so the 7B is at best marginal here.

AGENTS.md NEVER #8 forbids changing the model without an ADR, and NEVER #6
forbids inventing numbers. Both rules would be broken by quietly serving a
smaller model, or by reporting a fabricated confidence.

The user's explicit instruction: attempt the spec'd 7B first, detect the device
automatically (CUDA, then MPS, then CPU), and accept a step-down only if it is
visible.

## Options considered
1. **Hard-fail when the 7B does not fit** — maximum fidelity to SPEC §3, but the
   project would not answer anything on the machine it is developed on.
2. **Ship the 3B and call it done** — simple, and silently breaks NEVER #8.
3. **Capability-probed ladder with an audit trail** — try the configured model,
   step down when the estimate does not fit, and record every reason.

## Decision
Option 3. `runtime/devices.py`:

- probes devices in the order **CUDA -> MPS -> CPU**; nothing hardcodes a device;
- estimates VRAM from nominal parameter counts (VL models undercount if you
  trust the "7B" in the id, because the vision tower is not in that number);
- walks the ladder: requested model -> fallback model (preferring full precision
  when it fits) -> CPU with the smaller model;
- returns `ResolvedModel.reasons`, an ordered audit trail that is logged at
  WARNING and written into every `AnswerPayload.provenance`, together with
  `requested_model` and `fallback_applied`.

`MAINTIDOC_ALLOW_MODEL_FALLBACK=false` disables substitution: the requested
model is attempted and either loads or fails, which is how the real 7B gets
benchmarked. `bench-vlm` attempts the configured model regardless of the
estimate.

## Consequences
- A downgrade can never be invisible: it is in the log, in the response and in
  MLflow.
- The estimate is a planning heuristic, **not a measurement**. The measured
  peak VRAM from `bench-vlm` supersedes `_NOMINAL_PARAMETERS` and must be
  recorded here when it lands.
- Measured on the development host: 8.0 GB total, free VRAM varying 3.8–5.5 GB;
  7B-AWQ estimated 6.3 GB (does not fit), 3B-AWQ estimated 3.7 GB (fits).
- Revisit when the benchmark has run on real hardware, and if a 24 GB target
  machine becomes available.

## Amendment — substitution is disabled by default (owner decision)

The original decision above allowed the ladder to step down 7B -> 3B -> CPU
automatically. The project owner subsequently ruled that **Qwen2.5-VL-7B is the
model and a smaller one must never be substituted**: if the 7B does not fit, the
system reports the situation and waits for a human instead of serving something
else.

So the ladder is retained as a *mechanism* but its fallback is now off by
default (`MAINTIDOC_ALLOW_MODEL_FALLBACK=false`):

- `resolve_model` still computes whether the 7B fits and still records the full
  audit trail, so `doctor` explains exactly why a load would fail.
- With fallback disabled it returns the requested model with `degraded=True`
  rather than a different model, and the caller must decide: attempt the load
  with AWQ plus tight `max_new_tokens` / `max_pixels`, or stop.
- Stepping down is now an explicit, recorded deviation, never a default.

Consequence: on hardware where the 7B genuinely cannot fit, the correct outcome
is a clear failure, not a working 3B. That is the owner's call and is recorded
here rather than being quietly overridden by a default.

## Amendment 2 — `usable` distinguishes hardware from a runtime that can drive it

Found by executing the stack: `detect_devices` listed the RTX 4070 because
`nvidia-smi` reports it, even though the installed torch was a **CPU-only
build**. `select_device("auto")` therefore returned `cuda`, and the first tensor
move failed with *"Torch not compiled with CUDA enabled"*.

`DeviceInfo.usable` now records whether the *installed runtime* can execute on a
device:

- CUDA found only through `nvidia-smi` (torch absent, or a CPU-only build) is
  reported by `doctor` with `usable=false` and is never selected;
- `select_device` skips unusable devices, so `auto` lands on CPU instead of
  failing at the first tensor;
- pinning `cuda` on unusable hardware still raises, and now says `unusable` so
  the cause is obvious.

This matters for #16: the 7B cannot run on a CPU-only torch regardless of VRAM,
so the GPU path needs a CUDA build of torch, not just free memory.