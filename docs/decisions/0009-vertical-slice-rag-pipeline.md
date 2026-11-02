# 0009 — Vertical slice: end-to-end question answering pipeline

- **Date:** 2026-11-02
- **Status:** accepted
- **Issue/PR:** #9 (Vertical slice: question -> retrieve -> cited answer)

## Context
SPEC §4.2 specifies the complete RAG sequence:
1. Verify user identity and permissions;
2. Hybrid retrieval over text and visual collections;
3. Reciprocal Rank Fusion;
4. Candidate reranking;
5. Delivery of top cited passages and figures to the VLM;
6. Generation and citation verification producing `AnswerPayload` (SPEC §7).

Issue #9 is the first vertical slice connecting these components together into a runnable, end-to-end question answering flow (`AnswerPipeline`).

## Decisions

1. **Pipeline coordination (`src/maintidoc/agent/pipeline.py`):**
   - Implemented `AnswerPipeline` taking `HybridRetriever`, `VlmProvider`, and prompt configuration.
   - Sequence:
     - `retriever.search(query, principal=principal, top_n=top_k)` enforces access filtering *inside* the vector search.
     - `render_sources(hits)` serializes retrieved elements into delimited `<sources>` format.
     - `prompt.format(question=question, sources=sources_text, visual_analysis=...)` loads versioned `answer_v1.yaml` enforcing NEVER #4 (untrusted data is delimited).
     - `vlm_provider.generate(instruction=..., expects=OutputSchema.ANSWER)` generates raw model JSON.
     - `ModelAnswerDraft.model_validate_json()` leniently parses model output into typed draft.
     - `verify_citations(draft.citations, hits)` validates citations against actually-retrieved elements (NEVER #5).
     - Construction of strict `AnswerPayload` with full provenance tracking.

2. **Citation verification & abstention (SPEC §7, §8, NEVER #5):**
   - If a citation in the draft points to an element that was not in the retrieved candidate set, it is stripped.
   - If the draft status was `answered` but no valid citations remain, status is automatically downgraded to `abstained` with explicit uncertainties.
   - If no sources were retrieved (e.g. access denied or zero matches), the pipeline returns an `abstained` payload.

3. **Known shortcuts recorded in STATUS.md:**
   - Visual analysis branch is plumbed with empty visual observation when text-only;
   - LangGraph state machine and human approval interrupt are handled in #24-#25;
   - VLM provider uses deterministic `StubVlmProvider` until CUDA environment is ready for the 7B (#16).

## Consequences
- End-to-end question answering is functional and verifiable through `AnswerPayload`.
- Clean integration layer ready for LangGraph workflow and FastAPI endpoints.
