"""Tests for FastAPI backend (Issue #21 & #26, SPEC §3, §6, §9)."""

from __future__ import annotations

import io
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from maintidoc.api.app import create_app
from maintidoc.domain import ElementKind, RetrievedElement
from maintidoc.mcp.server import MaintenanceDatabase, MaintenanceService
from maintidoc.retrieval.reranker import NoReranker
from maintidoc.schemas import AnswerStatus
from maintidoc.security.approval import (
    ApprovalService,
    InMemoryApprovalStore,
)
from maintidoc.vlm.stub import StubVlmProvider
from tests.factories import make_element, make_retrieved


class FakeRetriever:
    reranker = NoReranker()

    def __init__(self, hits: list[RetrievedElement] | None = None) -> None:
        self.hits = hits or []

    def search(self, *args: Any, **kwargs: Any) -> list[RetrievedElement]:
        return self.hits


@pytest.fixture
def approval_service() -> ApprovalService:
    return ApprovalService(
        secret_key="super-secret-key-32-bytes-minimum!",
        store=InMemoryApprovalStore(),
    )


@pytest.fixture
def mcp_service(approval_service: ApprovalService) -> MaintenanceService:
    return MaintenanceService(
        api_key=SecretStr("valid-test-mcp-key"),
        approval_service=approval_service,
        database=MaintenanceDatabase.seed_default(),
    )


@pytest.fixture
def sample_hits() -> list[RetrievedElement]:
    el = make_element(
        document_id="doc_pompe_01",
        version="v1",
        page=2,
        element_id="p_1",
        kind=ElementKind.PARAGRAPH,
        text="Voyant orange : alarme température palier, seuil 85 °C.",
        access_groups=frozenset({"maintenance", "site_a"}),
    )
    return [make_retrieved(el, score=0.95)]


@pytest.fixture
def client(
    mcp_service: MaintenanceService,
    approval_service: ApprovalService,
    sample_hits: list[RetrievedElement],
) -> TestClient:
    app = create_app(
        retriever=FakeRetriever(hits=sample_hits),
        vlm_provider=StubVlmProvider(),
        maintenance_service=mcp_service,
        approval_service=approval_service,
        api_key=SecretStr("valid-test-mcp-key"),
    )
    return TestClient(app)


def test_health_check(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ask_without_auth_header_rejected(client: TestClient) -> None:
    resp = client.post("/ask", data={"question": "Test question"})
    assert resp.status_code == 401


def test_ask_with_valid_auth_answers_question(client: TestClient) -> None:
    headers = {
        "X-User-Id": "tech_1",
        "X-User-Groups": "maintenance,site_a",
        "X-API-Key": "valid-test-mcp-key",
    }
    resp = client.post(
        "/ask",
        headers=headers,
        data={"question": "Que signifie le voyant orange ?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == AnswerStatus.ANSWERED
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["document_id"] == "doc_pompe_01"


def test_ask_with_oversized_image_rejected(client: TestClient) -> None:
    headers = {
        "X-User-Id": "tech_1",
        "X-User-Groups": "maintenance,site_a",
        "X-API-Key": "valid-test-mcp-key",
    }
    oversized = io.BytesIO(b"x" * (11 * 1024 * 1024))
    resp = client.post(
        "/ask",
        headers=headers,
        data={"question": "Photo question"},
        files={"image": ("test.png", oversized, "image/png")},
    )
    assert resp.status_code == 422


def test_end_to_end_ticket_scenario(
    client: TestClient,
    approval_service: ApprovalService,
) -> None:
    """SPEC §1 & §14 scenario: question -> draft ticket -> approve -> create."""
    headers = {
        "X-User-Id": "tech_1",
        "X-User-Groups": "maintenance,site_a",
        "X-API-Key": "valid-test-mcp-key",
    }

    # 1. Ask question proposing ticket creation on pompe_centrifuge_01
    resp = client.post(
        "/ask",
        headers=headers,
        data={
            "question": "Fuite importante constatée, préparer un ticket pour pompe_centrifuge_01",
            "equipment_id": "pompe_centrifuge_01",
            "request_ticket": "true",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == AnswerStatus.NEEDS_APPROVAL
    assert data["proposed_action"] is not None
    proposal = data["proposed_action"]
    assert proposal["equipment_id"] == "pompe_centrifuge_01"
    assert proposal["content_hash"] != ""

    # 2. Issue approval token for exact proposal content
    token_resp = client.post(
        "/approval-token",
        headers=headers,
        json={
            "equipment_id": proposal["equipment_id"],
            "title": proposal["title"],
            "description": proposal["description"],
        },
    )
    assert token_resp.status_code == 200
    token_data = token_resp.json()
    approval_token = token_data["approval_token"]

    # 3. Submit approval to create the ticket
    create_resp = client.post(
        "/tickets/approve",
        headers=headers,
        json={
            "equipment_id": proposal["equipment_id"],
            "title": proposal["title"],
            "description": proposal["description"],
            "approval_token": approval_token,
            "idempotency_key": proposal["idempotency_key"],
        },
    )
    assert create_resp.status_code == 200
    result = create_resp.json()
    assert result["created"] is True
    assert result["equipment_id"] == "pompe_centrifuge_01"
    assert result["ticket_id"] != ""

    # 4. Replay attempt with same token fails (single use)
    replay_resp = client.post(
        "/tickets/approve",
        headers=headers,
        json={
            "equipment_id": proposal["equipment_id"],
            "title": proposal["title"],
            "description": proposal["description"],
            "approval_token": approval_token,
            "idempotency_key": "different-key-attempt",
        },
    )
    assert replay_resp.status_code == 403


def test_equipment_history_endpoint(client: TestClient) -> None:
    headers = {
        "X-User-Id": "tech_1",
        "X-User-Groups": "maintenance,site_a",
        "X-API-Key": "valid-test-mcp-key",
    }
    resp = client.get("/equipment/pompe_centrifuge_01/history", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["equipment_id"] == "pompe_centrifuge_01"
    assert len(data["entries"]) >= 2
