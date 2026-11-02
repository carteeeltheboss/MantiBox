# Session 2026-11-02 — Vertical slice RAG and MCP maintenance server (Issues #9 & #22)

- **LLM/tool:** Gemini 3.8 Flash (High) / Antigravity (Windows host)
- **Issues closed:** #9 (Vertical slice: question -> retrieve -> render_sources -> stub VLM -> citation verification -> AnswerPayload), #22 (MCP maintenance server with server-side authorization)
- **Goal:** implement the end-to-end question answering pipeline with citation verification and abstention (Issue #9) and build the MCP maintenance service with strict server-side authorization on every tool call (Issue #22).

## What was done

- **Vertical Slice RAG Pipeline (`src/maintidoc/agent/pipeline.py`):**
  - Built `AnswerPipeline` connecting `HybridRetriever` -> `render_sources` -> `StubVlmProvider` -> citation verification -> `AnswerPayload`.
  - Enforced SPEC §7 contract: every cited element in the candidate answer is verified against actually retrieved elements.
  - Implemented automatic abstention (`AnswerStatus.ABSTAINED`) when retrieval produces no results, when access control filters out all sources, or when all citations are ungrounded.
  - Added unit and scenario tests (`tests/test_agent_pipeline.py`).
  - Added ADR 0009 (`docs/decisions/0009-vertical-slice-rag-pipeline.md`).
- **MCP Maintenance Server (`src/maintidoc/mcp/server.py`):**
  - Implemented `MaintenanceDatabase` seeded with fictional history and open tickets for pump and electrical cabinet equipment families (`pompe_centrifuge_01`, `pompe_centrifuge_02`, `armoire_electrique_01`, `armoire_electrique_02`).
  - Implemented `MaintenanceService` with strict server-side authorization:
    - Active principal check (`is_active`).
    - API key constant-time comparison (`hmac.compare_digest`).
    - Budget remaining check (`budget_remaining <= 0` denied per SPEC §9).
    - Equipment access check with no enumeration oracle (unknown equipment raises `PermissionDeniedError` identical to unauthorized equipment).
    - Row-level access filtering on equipment history (`truncated_by_access=True` when entries are withheld).
    - Strict ticket creation verification: canonical content hash match, derived idempotency key, and single-use approval token verification via `ApprovalService.verify(token, spend=True)` (SPEC §6, AGENTS.md NEVER #2, NEVER #3).
  - Added 10 tests in `tests/test_mcp_server.py`.
  - Added ADR 0010 (`docs/decisions/0010-mcp-maintenance-server.md`).
- **Dev gate & test suite:**
  - `nox -s check`: 434 tests passed, ruff clean, format clean, mypy strict clean on 37 source files.
  - Full suite with live Qdrant (`pytest -q`): 475 tests passed.

## Decisions (link ADRs)

- **ADR 0009** — Vertical slice RAG pipeline with citation verification and abstention.
- **ADR 0010** — MCP maintenance server with server-side authorization on every call.

## What was NOT verified

- Real BGE cross-encoder reranker inference: model weights (~2.3 GB) not downloaded; interface verified.
- The 7B VLM has never been loaded (CPU torch build, no CUDA).
- Transport mounting for MCP (FastMCP / stdio transport) — service layer tested directly in-process; client/transport wired in #23.
- No OCR has run (tesseract absent on host).
