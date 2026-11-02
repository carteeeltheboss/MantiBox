# Session Log: 2026-11-02 — 150-Scenario Eval Suite, Threat Model, MLflow Conventions & Security Regressions

- **Date:** 2026-11-02
- **Author:** HANFAOUI Karim
- **Issues Closed:** #5, #6, #7, #30, #31, #32

## What Was Done

1. **Threat Model (`docs/security/threat_model.md` — Issue #5):**
   - Established complete threat modeling for MaintiDoc per SPEC §9: assets (catalogs, PDF documents, MCP maintenance DB, single-use JWT approval tokens), trust boundaries (Untrusted vs Trusted components), and 4 attack paths.
   - Mapped all SPEC §9 mitigations to unit and regression tests, and built the security risk register (R-01 to R-06).

2. **Standardized MLflow Run Metadata Convention (`src/maintidoc/tracking.py` — Issue #7):**
   - Implemented `build_run_metadata` and `log_run_with_convention` recording `model_id`, `provider`, `prompt_versions`, `reranker`, `index_version`, `git_commit`, `device`, `env`.
   - Documented convention in `docs/ARCHITECTURE.md`. Added unit test coverage in `tests/test_tracking.py`.

3. **150-Scenario Evaluation Suite (`eval/scenarios.py`, `eval/scenarios.json`, `scripts/generate_scenarios.py` — Issues #6, #30):**
   - Seeded and generated 150 scenarios across all SPEC §10 categories: 40 text, 30 image/schema, 20 MCP consultation, 20 missing/contradictory, 20 access/approval, 20 prompt injection.
   - All scenarios marked `human_reviewed: false` per AGENTS.md.
   - Built runner `evaluate_scenarios` calculating status accuracy, citation validity rate, and multimodal lift against text-only ablation.
   - Updated `eval/run.py` supporting `--mode {retrieval,scenarios,all}`.

4. **Prompt Injection & Security Regressions (`tests/test_prompt_injection.py`, `tests/test_security_regression.py` — Issues #31, #32):**
   - Direct user prompt injections, document injections, OCR sticker injections, and MCP response injections tested and verified safe.
   - Strict tenant isolation across Qdrant query filters and `DocumentOpener` verified (site_a vs site_b).
   - Single-use bound approval token enforcement verified (no unapproved ticket creation, hash mismatch rejected, replay rejected).

## Verification

- `nox -s check`: ruff passed, ruff format passed, mypy strict on 47 modules passed, 483 fast tests passed.
- 524 total tests in suite.
- ADR 0015 created.
