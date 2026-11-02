# ADR 0015: 150-Scenario Evaluation Suite, Threat Model, MLflow Conventions & Security Regressions

- **Status:** accepted
- **Date:** 2026-11-02
- **Author:** HANFAOUI Karim
- **Issues:** #5, #6, #7, #30, #31, #32
- **Deciders:** HANFAOUI Karim

## Context

1. **Threat Model & Security Boundary (Issue #5, SPEC §9):** A comprehensive threat model is required covering trust boundaries, assets (catalogs, PDF documents, MCP maintenance database, approval tokens), attack vectors (indirect prompt injection via documents/OCR/tools, ticket forgery, privilege escalation), and a concrete risk register.
2. **Standardized MLflow Run Metadata (Issue #7, SPEC §7, §11):** Tracking runs in MLflow requires a consistent convention capturing model ID, provider, prompt versions, reranker arm, index version, git commit SHA, device, environment, and evaluation metrics.
3. **150 Evaluation Scenarios (Issues #6, #30, SPEC §10):** A realistic test suite of 150 scenarios must be established covering all 6 SPEC §10 categories: 40 text, 30 image/schema, 20 MCP, 20 missing/contradictory sources, 20 access control/approval, and 20 prompt injection. Scenarios must measure status accuracy, citation validity, and multimodal vs. text-only quality lift. Per AGENTS.md, generated scenarios must be marked `human_reviewed: false` until human review.
4. **Prompt Injection & Security Regressions (Issues #31, #32, SPEC §9):** Automated regression suites must verify that document text, OCR stickers, and MCP outputs are strictly treated as DATA, preventing instruction overrides, unapproved ticket generation, or cross-group data leakage.

## Decisions

1. **Threat Model (`docs/security/threat_model.md`):**
   - Formalized architecture trust boundaries between Untrusted Inputs (user questions, uploaded images, raw PDF/OCR content, external MCP responses) and Trusted Systems (Access Control, Ticket Approval Service, LangGraph Controller).
   - Documented attack paths and mapped every mitigation to unit and regression tests.
   - Defined risk register R-01 through R-06 with corresponding residual risk assessments.

2. **MLflow Tracking Convention (`src/maintidoc/tracking.py`):**
   - Implemented `build_run_metadata(...)` enforcing standardized parameter keys: `model_id`, `provider`, `prompt_versions`, `reranker`, `index_version`, `git_commit`, `device`, `env`.
   - Implemented `log_run_with_convention(...)` logging parameters, metrics, and git SHA gracefully into MLflow or local tracking when server is not configured.

3. **150-Scenario Suite & Runner (`eval/scenarios.py`, `eval/scenarios.json`, `scripts/generate_scenarios.py`):**
   - Seeded 150 diverse scenarios matching SPEC §10 breakdown across French industrial maintenance domains (pumps, electrical cabinets, valves).
   - Marked every scenario `human_reviewed: false` to respect AGENTS.md requirements.
   - Built `evaluate_scenarios(...)` executing queries against the `AnswerPipeline`, verifying status matches, calculating citation validity rates, and computing multimodal lift against text-only ablation.
   - Integrated scenario evaluation mode (`--mode {retrieval,scenarios,all}`) into `eval/run.py`.

4. **Security Regression & Injection Test Suites (`tests/test_prompt_injection.py`, `tests/test_security_regression.py`):**
   - Tested direct user prompt injections, document injections, OCR sticker injections, and MCP response injections. Confirmed zero instruction hijacking due to XML delimitation and prompt contracts.
   - Validated that Qdrant query filters and DocumentOpener enforce strict isolation across tenant access groups (`site_a` vs `site_b`).
   - Validated that ticket creation without valid approval token, with modified content hash, or with replayed JTI fails unconditionally.

## Consequences

- Comprehensive automated security and evaluation verification is established across all SPEC §9 and §10 requirements.
- 524 total tests in suite, dev gate runs 483 fast tests in under 6 seconds cleanly.
- Issues #5, #6, #7, #30, #31, and #32 are fully completed and closed.
