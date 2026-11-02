# 0010 — MCP maintenance service and server-side authorization

- **Date:** 2026-11-02
- **Status:** accepted
- **Issue/PR:** #22 (MCP maintenance server with server-side authorization)

## Context
SPEC §6 and §9 define the external maintenance service exposed via Model Context Protocol (MCP).
Three tools:
- `get_equipment_history(equipment_id)`
- `get_open_tickets(equipment_id)`
- `create_ticket(equipment_id, title, description, approval_token)`

Key requirements & constraints:
1. AGENTS.md NEVER #2: The server must re-check authorization on **every** call. Caller-supplied context cannot be trusted blindly.
2. AGENTS.md NEVER #3: No ticket is created without an explicit human approval token bound to user + content sha256 + idempotency key.
3. SPEC §9: MCP tool call budget (maximum 3 calls per request).
4. No existence oracle: Non-existent equipment must be denied identically to unauthorized equipment.

## Decisions

1. **Service architecture (`src/maintidoc/mcp/server.py`):**
   - `MaintenanceDatabase`: thread-safe in-memory store initialized with fictional equipment, interventions, and tickets covering the two families (`pompes` on Site A, `armoires` on Site B) plus restricted supervisor operations.
   - `MaintenanceService`: core service logic with complete authorization enforcement on every call.

2. **Server-side authorization pipeline:**
   - Active principal check: inactive accounts are denied immediately.
   - API key verification: constant-time (`hmac.compare_digest`) against server secret.
   - Budget enforcement: calls rejected when `budget_remaining <= 0`.
   - Access control on equipment: unknown equipment produces `AccessDeniedError` identical to unauthorized equipment (enumeration protection).
   - Row-level filtering: `EquipmentHistoryEntry` and `Ticket` lists filter out rows whose `access_groups` do not intersect caller groups. When rows are withheld, `truncated_by_access=True` is returned.

3. **Ticket creation & idempotency:**
   - Content hash: `canonical_content_hash(equipment_id, title, description)`.
   - Idempotency key verification: derived deterministically via `derive_idempotency_key(user_id, content_hash)`.
   - Deduplication: previously processed idempotency keys return `CreateTicketResult(created=False, deduplicated=True)` without re-spending approval tokens.
   - Single-use token: verified and spent atomically via `ApprovalService.verify(spend=True)`.

4. **Security logging:**
   - Every authorization decision (allow and deny) emits an `access_event` log with structured technical metadata only (no raw ticket contents).

## Consequences
- Server-side authorization is strict and verified on every call.
- The service can run in-process or be exposed over standard MCP transports.
- Ready for LangGraph MCP client integration in #23.
