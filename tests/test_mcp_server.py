"""Tests for the MCP maintenance server with server-side authorization (SPEC §6, §9, Issue #22).

AGENTS.md NEVER #2: Authorization must be re-checked on EVERY call.
AGENTS.md NEVER #3: No ticket without explicit human approval token bound to user + content.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from maintidoc.errors import AccessDeniedError, ApprovalInvalidError
from maintidoc.mcp.contracts import (
    CreateTicketRequest,
    TicketState,
    ToolCallContext,
)
from maintidoc.mcp.server import MaintenanceDatabase, MaintenanceService
from maintidoc.security.access import Principal
from maintidoc.security.approval import (
    ApprovalService,
    InMemoryApprovalStore,
    canonical_content_hash,
    derive_idempotency_key,
)


@pytest.fixture
def approval_service(signing_key: str) -> ApprovalService:
    return ApprovalService(secret_key=signing_key, store=InMemoryApprovalStore())


@pytest.fixture
def mcp_service(api_key: SecretStr, approval_service: ApprovalService) -> MaintenanceService:
    return MaintenanceService(
        api_key=api_key,
        approval_service=approval_service,
        database=MaintenanceDatabase.seed_default(),
    )


def _context(
    principal: Principal,
    api_key: SecretStr,
    budget_remaining: int = 3,
) -> ToolCallContext:
    return ToolCallContext(
        principal=principal,
        api_key=api_key,
        request_id="req-123",
        budget_remaining=budget_remaining,
    )


class TestMcpAuthorization:
    def test_invalid_api_key_rejected(
        self,
        mcp_service: MaintenanceService,
        technician: Principal,
    ) -> None:
        ctx = _context(technician, SecretStr("wrong-api-key"))
        with pytest.raises(AccessDeniedError, match="invalid API key"):
            mcp_service.get_equipment_history("pompe_centrifuge_01", ctx)

    def test_inactive_principal_rejected(
        self,
        mcp_service: MaintenanceService,
        deactivated: Principal,
        api_key: SecretStr,
    ) -> None:
        ctx = _context(deactivated, api_key)
        with pytest.raises(AccessDeniedError, match="inactive principal"):
            mcp_service.get_equipment_history("pompe_centrifuge_01", ctx)

    def test_exhausted_budget_rejected(
        self,
        mcp_service: MaintenanceService,
        technician: Principal,
        api_key: SecretStr,
    ) -> None:
        ctx = _context(technician, api_key, budget_remaining=0)
        with pytest.raises(AccessDeniedError, match="budget exceeded"):
            mcp_service.get_equipment_history("pompe_centrifuge_01", ctx)

    def test_unauthorized_equipment_rejected_without_existence_leak(
        self,
        mcp_service: MaintenanceService,
        outsider: Principal,
        api_key: SecretStr,
    ) -> None:
        # Outsider has site_b, cannot access pompe_centrifuge_01 on site_a
        ctx = _context(outsider, api_key)
        with pytest.raises(AccessDeniedError, match="access denied"):
            mcp_service.get_equipment_history("pompe_centrifuge_01", ctx)

        # Non-existent equipment raises the same error (no enumeration oracle)
        with pytest.raises(AccessDeniedError, match="access denied"):
            mcp_service.get_equipment_history("equipement_fantome", ctx)


class TestEquipmentHistory:
    def test_authorized_technician_sees_accessible_rows(
        self,
        mcp_service: MaintenanceService,
        technician: Principal,
        api_key: SecretStr,
    ) -> None:
        ctx = _context(technician, api_key)
        history = mcp_service.get_equipment_history("pompe_centrifuge_01", ctx)

        assert history.equipment_id == "pompe_centrifuge_01"
        assert len(history.entries) == 2  # Has 2 standard maintenance entries
        # Supervisor-only entry was filtered out
        assert history.truncated_by_access is True

    def test_supervisor_sees_all_rows(
        self,
        mcp_service: MaintenanceService,
        supervisor: Principal,
        api_key: SecretStr,
    ) -> None:
        ctx = _context(supervisor, api_key)
        history = mcp_service.get_equipment_history("pompe_centrifuge_01", ctx)

        assert history.equipment_id == "pompe_centrifuge_01"
        assert len(history.entries) == 3  # 2 standard + 1 supervisor entry
        assert history.truncated_by_access is False


class TestOpenTickets:
    def test_returns_only_open_tickets(
        self,
        mcp_service: MaintenanceService,
        technician: Principal,
        api_key: SecretStr,
    ) -> None:
        ctx = _context(technician, api_key)
        ticket_list = mcp_service.get_open_tickets("pompe_centrifuge_01", ctx)

        assert ticket_list.equipment_id == "pompe_centrifuge_01"
        assert len(ticket_list.tickets) >= 1
        assert all(t.state is TicketState.OPEN for t in ticket_list.tickets)


class TestCreateTicket:
    def test_create_ticket_with_valid_token(
        self,
        mcp_service: MaintenanceService,
        approval_service: ApprovalService,
        technician: Principal,
        api_key: SecretStr,
    ) -> None:
        eq_id = "pompe_centrifuge_01"
        title = "Fuite garniture"
        desc = "Remplacement requis"

        content_hash = canonical_content_hash(equipment_id=eq_id, title=title, description=desc)
        idem_key = derive_idempotency_key(user_id=technician.user_id, content_hash=content_hash)

        grant = approval_service.issue(
            user_id=technician.user_id,
            content_hash=content_hash,
            idempotency_key=idem_key,
        )

        req = CreateTicketRequest(
            equipment_id=eq_id,
            title=title,
            description=desc,
            approval_token=grant.token,
            idempotency_key=idem_key,
        )
        ctx = _context(technician, api_key)

        res = mcp_service.create_ticket(req, ctx)
        assert res.created is True
        assert res.deduplicated is False
        assert res.equipment_id == eq_id
        assert len(res.ticket_id) > 0

    def test_tampered_content_rejected(
        self,
        mcp_service: MaintenanceService,
        approval_service: ApprovalService,
        technician: Principal,
        api_key: SecretStr,
    ) -> None:
        eq_id = "pompe_centrifuge_01"
        title = "Titre original"
        desc = "Description"

        content_hash = canonical_content_hash(equipment_id=eq_id, title=title, description=desc)
        idem_key = derive_idempotency_key(user_id=technician.user_id, content_hash=content_hash)

        grant = approval_service.issue(
            user_id=technician.user_id,
            content_hash=content_hash,
            idempotency_key=idem_key,
        )

        # Alter title after approval token was granted
        req = CreateTicketRequest(
            equipment_id=eq_id,
            title="Titre modifié sans accord",
            description=desc,
            approval_token=grant.token,
            idempotency_key=idem_key,
        )
        ctx = _context(technician, api_key)

        with pytest.raises(ApprovalInvalidError):
            mcp_service.create_ticket(req, ctx)

    def test_idempotent_retry_deduplicates(
        self,
        mcp_service: MaintenanceService,
        approval_service: ApprovalService,
        technician: Principal,
        api_key: SecretStr,
    ) -> None:
        eq_id = "pompe_centrifuge_01"
        title = "Ticket réémis"
        desc = "Description"

        content_hash = canonical_content_hash(equipment_id=eq_id, title=title, description=desc)
        idem_key = derive_idempotency_key(user_id=technician.user_id, content_hash=content_hash)

        grant = approval_service.issue(
            user_id=technician.user_id,
            content_hash=content_hash,
            idempotency_key=idem_key,
        )

        req = CreateTicketRequest(
            equipment_id=eq_id,
            title=title,
            description=desc,
            approval_token=grant.token,
            idempotency_key=idem_key,
        )
        ctx = _context(technician, api_key)

        # First call creates ticket
        res1 = mcp_service.create_ticket(req, ctx)
        assert res1.created is True
        assert res1.deduplicated is False

        # Retry with same idempotency key deduplicates without re-spending token
        res2 = mcp_service.create_ticket(req, ctx)
        assert res2.created is False
        assert res2.deduplicated is True
        assert res2.ticket_id == res1.ticket_id
