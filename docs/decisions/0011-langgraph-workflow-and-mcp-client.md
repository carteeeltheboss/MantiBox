# 0011. LangGraph workflow state machine and MCP client

Date: 2026-11-02  
Status: accepted  
Issues: #23, #24, #25

## Context

SPEC §6 defines the agent orchestration sequence:
1. Verify access
2. Analyze request & image
3. Retrieve sources (multimodal hybrid search)
4. Consult history via MCP if needed
5. Build and verify response
6. Route outcome (answer / needs clarification / ticket)
7. Prepare ticket draft
8. Wait for human approval
9. Create ticket via MCP

SPEC §9 defines strict execution budgets for the prototype:
- Maximum 3 MCP tool calls per request.
- Maximum 6 model calls per request.
- Maximum 60 seconds automatic processing time.
- Maximum image size (10 MB).

AGENTS.md NEVER #2 and NEVER #3 mandate:
- Authorization must be verified on EVERY call (both client-side context tracking and server-side authorization).
- Sensitive actions (ticket creation) must require an explicit, un-tampered, single-use human approval token bound to user and canonical content hash.

## Decision

1. **MCP Client (`src/maintidoc/mcp/client.py`):**
   - Implemented `McpClient` wrapping the three tools (`get_equipment_history`, `get_open_tickets`, `create_ticket`).
   - Tracks `calls_made` and `budget_remaining` per request.
   - Enforces the client-side budget: raises `BudgetExceededError` if more than 3 calls are attempted.
   - Injects `ToolCallContext` with active `Principal`, API key, request ID, and remaining budget to the server.

2. **LangGraph Workflow (`src/maintidoc/agent/workflow.py`):**
   - Implemented `StateGraph(WorkflowState)` with explicit typed state (`WorkflowState`).
   - Nodes faithfully map the SPEC §6 flow:
     - `verify_access`: Checks principal validity and image size limits before any processing.
     - `handle_access_denied`: Immediate short-circuit to `AnswerStatus.ABSTAINED` with 0 model calls on unauthenticated or group-less callers.
     - `analyze_request`: Performs visual inspection and equipment identification when an image is supplied.
     - `retrieve_sources`: Access-filtered hybrid retrieval (`HybridRetriever`).
     - `consult_history`: Consults equipment history only when equipment is identified and failure history / ticket request is present.
     - `build_and_verify_response`: Model inference, draft parsing, and real citation verification (ungrounded citations removed; downgrades to `ABSTAINED` if unsourced).
     - `draft_ticket`: Prepares `TicketProposal` with SHA-256 canonical content hash and derived idempotency key.
     - `wait_human_approval`: Interrupt state awaiting approval token.
     - `create_ticket`: Submits approved ticket request to MCP client.
     - `finalize`: Assembles strict `AnswerPayload` ensuring all status invariants are met.
   - Enforced SPEC §9 budgets: max 6 model calls, max 3 MCP calls, 60s wall-clock limit, 10 MB image size.
   - Exported graph architecture diagram to `docs/architecture/workflow_graph.md`.

## Consequences

- End-to-end orchestration is now driven by a deterministic state machine rather than open-ended tool loops.
- Budgets and loop bounds are guaranteed in code before and during execution.
- Human approval is structurally required before ticket creation: without a token, the workflow halts at `AnswerStatus.NEEDS_APPROVAL`.
