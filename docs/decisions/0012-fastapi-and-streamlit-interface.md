# 0012. FastAPI web backend and Streamlit interface

Date: 2026-11-02  
Status: accepted  
Issues: #21, #26

## Context

SPEC §3 and §14 require an interactive web interface and API for technicians:
- FastAPI backend providing authenticated endpoints for question answering, approval token issuance, ticket creation, and equipment history.
- Streamlit UI allowing photo uploads, technical questions, display of strict citations and uncertainties, and interactive human-in-the-loop approval of maintenance tickets.
- Support for the end-to-end scenario from SPEC §1:
  « Voici une photo du panneau de commande. Que signifie ce voyant orange ? Consulte les dernières interventions et prépare un ticket si nécessaire. »

AGENTS.md NEVER #2 and NEVER #8 require:
- Per-call authentication and authorization on every endpoint.
- Any dependency addition (`streamlit>=1.40`) requires an ADR.

## Decision

1. **Dependency Addition:**
   - Added `streamlit>=1.40` to project dependencies in `pyproject.toml`.
   - Ignored `B008` in ruff (standard FastAPI dependency injection pattern using `Depends()`, `Form()`, `File()`).

2. **FastAPI Backend (`src/maintidoc/api/app.py`):**
   - Factory function `create_app(...)` wiring `retriever`, `vlm_provider`, `maintenance_service`, and `approval_service`.
   - Constant-time API key verification (`hmac.compare_digest`) and header-based `Principal` injection (`X-User-Id`, `X-User-Groups`, `X-API-Key`).
   - Image upload validation enforcing the SPEC §9 10 MB limit.
   - Endpoints:
     - `GET /health`: Technical readiness probe.
     - `POST /ask`: Multimodal question answering running `run_workflow(...)` and returning strict `AnswerPayload`.
     - `POST /approval-token`: Computes `canonical_content_hash` and issues a short-lived HS256 approval token bound to user + content.
     - `POST /tickets/approve`: Creates ticket via MCP client after verifying approval token and single-use idempotency.
     - `GET /equipment/{equipment_id}/history`: Returns row-level filtered equipment history.
     - `GET /equipment/{equipment_id}/tickets`: Returns open maintenance tickets.

3. **Streamlit Frontend (`src/maintidoc/ui/app.py`):**
   - Two-tab layout: "💬 Question & Photo" and "📋 Historique & Tickets MCP".
   - Sourced answer display with visual status badges, collapsible verified citations, and missing information uncertainties.
   - Interactive human approval screen: displays candidate ticket details, content SHA-256 fingerprint, and an approval action button that triggers token generation and ticket creation via MCP.

4. **End-to-End Scenario Verification (`tests/test_api.py`):**
   - Verified the complete SPEC §1 ticket creation workflow: question proposing ticket -> `needs_approval` response with `TicketProposal` -> approval token generation -> ticket creation via MCP -> replay rejection.

## Consequences

- Full technician user experience is functional both via programmatic API and interactive browser UI.
- All access-control, approval-binding, and budget limits are preserved across the HTTP boundary.
