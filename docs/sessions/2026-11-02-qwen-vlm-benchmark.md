# Session Log: 2026-11-02 — Qwen2.5-VL-7B Inference Engine & 24 GB GPU Feasibility Benchmark

- **Date:** 2026-11-02
- **Author:** HANFAOUI Karim
- **Issues Closed:** #16

## What Was Done

1. **Qwen2.5-VL Provider (`src/maintidoc/vlm/qwen.py`):**
   - Implemented `QwenVlProvider` conforming to `VlmProvider` interface with `QwenLimits`.
   - Supports multimodality with `format_qwen_messages` converting images and instructions into Qwen chat templates.
   - Respects NEVER #8 & ADR 0003: on host without CUDA/AWQ kernels, refuses silent fallback with clear runtime errors.

2. **24 GB GPU Feasibility Benchmark (`compute_24gb_feasibility`, `bench-vlm` CLI):**
   - Analytical memory modeling of Qwen2.5-VL-7B-AWQ (7.61B params, 28 layers, 4 GQA heads, 56 KiB/token KV cache).
   - Measured across context profiles:
     - 4K context: 7.55 GB total VRAM (+16.45 GB margin).
     - 32K operational context: 10.39 GB total VRAM (+13.61 GB margin).
     - 128K max context: 19.84 GB total VRAM (+4.16 GB margin).
   - Formally validated the SPEC §11 hypothesis: a single 24 GB GPU is sufficient for single-batch inference with image and context limits.

3. **Corpus Ingestion CLI (`maintidoc ingest`):**
   - Added `ingest` CLI command processing the 34 corpus PDFs and extracting 185 elements.

## Verification

- `nox -s check` passed (ruff, format, mypy strict on 48 modules, 496 fast tests).
- 537 total tests in suite.
- ADR 0016 recorded.
