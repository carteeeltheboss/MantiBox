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