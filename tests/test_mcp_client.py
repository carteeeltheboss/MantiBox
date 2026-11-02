"""Tests for MCP maintenance client (Issue #23, SPEC §6, §9)."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from maintidoc.errors import AccessDeniedError, BudgetExceededError
from maintidoc.mcp.client import McpClient
from maintidoc.mcp.contracts import CreateTicketRequest, TicketState
from maintidoc.mcp.server import MaintenanceDatabase, MaintenanceService
from maintidoc.security.access import Principal
from maintidoc.security.approval import (
    ApprovalService,
    InMemoryApprovalStore,
    canonical_content_hash,
    derive_idempotency_key,
)


@pytest.fixture
def test_principal() -> Principal:
    return Principal(
        user_id="user_pompe",
        groups=frozenset({"maintenance", "site_a"}),
        is_active=True,
    )


@pytest.fixture
def approval_service() -> ApprovalService:
    return ApprovalService(
        secret_key="super-secret-key-32-bytes-minimum!",
        store=InMemoryApprovalStore(),
    )


@pytest.fixture
def maintenance_service(approval_service: ApprovalService) -> MaintenanceService:
    db = MaintenanceDatabase.seed_default()
    return MaintenanceService(
        api_key=SecretStr("valid-test-mcp-key"),
        approval_service=approval_service,
        database=db,
    )


def test_mcp_client_get_equipment_history(
    maintenance_service: MaintenanceService, test_principal: Principal
) -> None:
    client = McpClient(
        service=maintenance_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-123",
    )
    history = client.get_equipment_history("pompe_centrifuge_01")
    assert history.equipment_id == "pompe_centrifuge_01"
    assert len(history.entries) == 2
    assert history.truncated_by_access  # 3rd entry restricted to supervisors
    assert client.calls_made == 1
    assert client.budget_remaining == 2


def test_mcp_client_get_open_tickets(
    maintenance_service: MaintenanceService, test_principal: Principal
) -> None:
    client = McpClient(
        service=maintenance_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-123",
    )
    ticket_list = client.get_open_tickets("pompe_centrifuge_01")
    assert ticket_list.equipment_id == "pompe_centrifuge_01"
    assert len(ticket_list.tickets) == 1
    assert ticket_list.tickets[0].state == TicketState.OPEN
    assert client.calls_made == 1


def test_mcp_client_create_ticket_with_approval(
    maintenance_service: MaintenanceService,
    approval_service: ApprovalService,
    test_principal: Principal,
) -> None:
    client = McpClient(
        service=maintenance_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-123",
    )
    content_hash = canonical_content_hash(
        equipment_id="pompe_centrifuge_01",
        title="Fuite garniture mécanique",
        description="Remplacement nécessaire de la garniture suite à fuite.",
    )
    idempotency_key = derive_idempotency_key(
        user_id=test_principal.user_id, content_hash=content_hash
    )
    grant = approval_service.issue(
        user_id=test_principal.user_id,
        content_hash=content_hash,
        idempotency_key=idempotency_key,
    )

    result = client.create_ticket(
        CreateTicketRequest(
            equipment_id="pompe_centrifuge_01",
            title="Fuite garniture mécanique",
            description="Remplacement nécessaire de la garniture suite à fuite.",
            approval_token=grant.token,
            idempotency_key=idempotency_key,
        )
    )
    assert result.created is True
    assert result.equipment_id == "pompe_centrifuge_01"
    assert client.calls_made == 1


def test_mcp_client_enforces_budget_limit_of_three_calls(
    maintenance_service: MaintenanceService, test_principal: Principal
) -> None:
    client = McpClient(
        service=maintenance_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-123",
        max_calls=3,
    )
    # 1st call
    client.get_equipment_history("pompe_centrifuge_01")
    # 2nd call
    client.get_open_tickets("pompe_centrifuge_01")
    # 3rd call
    client.get_equipment_history("pompe_centrifuge_01")
    assert client.calls_made == 3
    assert client.budget_remaining == 0

    # 4th call must raise BudgetExceededError
    with pytest.raises(BudgetExceededError):
        client.get_equipment_history("pompe_centrifuge_01")


def test_mcp_client_propagates_access_denied(
    maintenance_service: MaintenanceService,
) -> None:
    # Principal without access to site_a
    unauth_principal = Principal(
        user_id="other_user",
        groups=frozenset({"other_group"}),
        is_active=True,
    )
    client = McpClient(
        service=maintenance_service,
        principal=unauth_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-123",
    )
    with pytest.raises(AccessDeniedError):
        client.get_equipment_history("pompe_centrifuge_01")
