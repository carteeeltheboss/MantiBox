# ADR 0014: Structured Output Validation with Bounded Retry, Citation Verification & Uncertainty Logic

- **Status:** accepted
- **Date:** 2026-11-02
- **Author:** HANFAOUI Karim
- **Issues:** #18, #19, #20
- **Deciders:** HANFAOUI Karim

## Context

SPEC §4.2, §7, §8, §9 and §10 mandate:
1. **Structured output validation with bounded retry (SPEC §7, §9):** Model responses must satisfy Pydantic contracts strictly. Retries on invalid JSON or schema errors must be bounded and strictly counted against the 6-model-call budget. Exhausted retries must safely fall back to abstention without crashing.
2. **Citation verification against retrieved elements and evidence (SPEC §4.2, AGENTS.md NEVER #5):** Every citation returned must match an actually retrieved element and be verified to support the claim. Unsupported citations must be stripped, and answers with no surviving citations must be downgraded to abstained. The citation validity rate metric ($\frac{\text{supported}}{\text{total}}$) must be measured.
3. **Uncertainty and abstention handling (SPEC §8, AGENTS.md NEVER #6):** Clarification or abstention must be returned when photos are unreadable, sources are missing or contradictory, or out-of-role risks occur. No fabricated confidence scores or percentages can ever be emitted.

## Decisions

1. **Structured Validation Seam (`src/maintidoc/agent/validation.py`):**
   - Implemented `generate_validated_output[T](vlm, schema, instruction, ...)` which executes up to `min(max_retries + 1, budget_remaining)` calls.
   - On parse/validation failure, formats a retry instruction using a versioned prompt (`prompts/retry_format_v1.yaml`).
   - If retries are exhausted or budget is depleted, returns `ValidatedOutputResult(output=None)` which triggers safe abstention via `create_validation_fallback_payload`.
2. **Evidence-Based Citation Verification (`src/maintidoc/agent/citations.py`):**
   - Validates existence in `retrieved_elements` first (`(document_id, element_id, page)`).
   - Evaluates support using the versioned `evidence_check_v1.yaml` prompt through `OutputSchema.EVIDENCE_VERDICT` when model calls remain in budget.
   - Rejects unretrieved or unsupported citations. Downgrades `status=answered` to `status=abstained` if no verified citations remain.
   - Implemented `calculate_citation_validity(payload, retrieved_elements)` returning `CitationVerificationReport` with `validity_rate`.
3. **Uncertainty & Abstention Module (`src/maintidoc/agent/uncertainty.py`):**
   - Implemented `detect_uncertainty`, `build_clarification_payload`, and `build_abstention_payload` covering the four SPEC §8 cases (`UNREADABLE_PHOTO`, `OUT_OF_ROLE_RISK`, `NO_SOURCES`, `CONTRADICTORY_SOURCES`).
   - Implemented `verify_no_confidence_emitted(payload)` scanning for confidence tokens and regexes to enforce AGENTS.md NEVER #6.
4. **Integration into RAG Pipeline & LangGraph Workflow:**
   - Integrated into `AnswerPipeline` (`src/maintidoc/agent/pipeline.py`) and `run_workflow` / `build_workflow_graph` (`src/maintidoc/agent/workflow.py`).
   - Updated LangGraph state machine with conditional routing after `analyze_request` to route directly to `finalize` upon out-of-role or unreadable photo detection.

## Consequences

- Model call budget accounting is strictly respected across answer generation, retries, and citation evidence checks.
- Zero unretrieved or unverified citations can escape to the user.
- Complete test suite passes with 512 total tests (471 fast tests, 41 integration tests).
