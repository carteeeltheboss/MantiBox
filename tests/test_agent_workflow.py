"""Tests for LangGraph workflow state machine and budgets (Issues #23 & #24, SPEC §6, §9)."""

from __future__ import annotations

import time
from typing import Any

import pytest
from pydantic import SecretStr

from maintidoc.agent.workflow import (
    WorkflowLimits,
    build_workflow_graph,
    run_workflow,
)
from maintidoc.domain import RetrievedElement
from maintidoc.errors import BudgetExceededError, ValidationError
from maintidoc.mcp.client import McpClient
from maintidoc.mcp.server import MaintenanceDatabase, MaintenanceService
from maintidoc.retrieval.reranker import NoReranker
from maintidoc.schemas import AnswerStatus, ElementKind
from maintidoc.security.access import Principal
from maintidoc.security.approval import (
    ApprovalService,
    InMemoryApprovalStore,
    canonical_content_hash,
    derive_idempotency_key,
)
from maintidoc.vlm.stub import StubVlmProvider
from tests.factories import make_element, make_retrieved


class FakeHybridRetriever:
    """Fast retriever test double returning controllable hits."""

    def __init__(self, hits: list[RetrievedElement] | None = None) -> None:
        self.hits = hits or []
        self.reranker = NoReranker()

    def search(
        self,
        query: str,
        principal: Principal,
        query_image: bytes | None = None,
        top_n: int = 8,
    ) -> list[RetrievedElement]:
        if not principal.is_active or not principal.groups:
            return []
        return self.hits[:top_n]


@pytest.fixture
def test_principal() -> Principal:
    return Principal(
        user_id="tech_user",
        groups=frozenset({"maintenance", "site_a"}),
        is_active=True,
    )


@pytest.fixture
def inactive_principal() -> Principal:
    return Principal(
        user_id="bad_user",
        groups=frozenset({"maintenance", "site_a"}),
        is_active=False,
    )


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
def sample_retrieved_elements() -> list[RetrievedElement]:
    el = make_element(
        document_id="doc_pompe_01",
        version="v1",
        page=2,
        element_id="p_1",
        kind=ElementKind.PARAGRAPH,
        text="La pression maximale de service de la pompe centrifuge 01 est de 16 bars.",
        access_groups=frozenset({"maintenance", "site_a"}),
    )
    return [make_retrieved(el, score=0.92)]


def test_workflow_inactive_principal_denied(
    inactive_principal: Principal,
    mcp_service: MaintenanceService,
) -> None:
    retriever = FakeHybridRetriever()
    vlm = StubVlmProvider()
    client = McpClient(
        service=mcp_service,
        principal=inactive_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-1",
    )

    payload = run_workflow(
        question="Quelle est la pression max ?",
        principal=inactive_principal,
        retriever=retriever,
        vlm_provider=vlm,
        mcp_client=client,
    )
    assert payload.status is AnswerStatus.ABSTAINED
    assert "Accès refusé" in payload.answer
    assert payload.provenance.model_calls_used == 0


def test_workflow_answers_question_with_verified_citations(
    test_principal: Principal,
    sample_retrieved_elements: list[RetrievedElement],
    mcp_service: MaintenanceService,
) -> None:
    retriever = FakeHybridRetriever(hits=sample_retrieved_elements)
    vlm = StubVlmProvider()
    client = McpClient(
        service=mcp_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-2",
    )

    payload = run_workflow(
        question="Quelle est la pression maximale ?",
        principal=test_principal,
        retriever=retriever,
        vlm_provider=vlm,
        mcp_client=client,
    )
    assert payload.status is AnswerStatus.ANSWERED
    assert len(payload.citations) >= 1
    assert payload.citations[0].document_id == "doc_pompe_01"
    assert payload.citations[0].element_id == "p_1"
    assert payload.provenance.model_calls_used >= 1


def test_workflow_consults_mcp_history_when_equipment_and_panne_mentioned(
    test_principal: Principal,
    sample_retrieved_elements: list[RetrievedElement],
    mcp_service: MaintenanceService,
) -> None:
    retriever = FakeHybridRetriever(hits=sample_retrieved_elements)
    vlm = StubVlmProvider()
    client = McpClient(
        service=mcp_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-3",
    )

    payload = run_workflow(
        question="Quel est l'historique des pannes sur la pompe_centrifuge_01 ?",
        principal=test_principal,
        retriever=retriever,
        vlm_provider=vlm,
        mcp_client=client,
        equipment_id="pompe_centrifuge_01",
    )
    assert payload.status in (AnswerStatus.ANSWERED, AnswerStatus.NEEDS_CLARIFICATION)
    # MCP client should have been called for equipment history and tickets
    assert client.calls_made >= 1


def test_workflow_skips_mcp_when_history_not_needed(
    test_principal: Principal,
    sample_retrieved_elements: list[RetrievedElement],
    mcp_service: MaintenanceService,
) -> None:
    retriever = FakeHybridRetriever(hits=sample_retrieved_elements)
    vlm = StubVlmProvider()
    client = McpClient(
        service=mcp_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-4",
    )

    payload = run_workflow(
        question="Expliquez le principe physique de fonctionnement.",
        principal=test_principal,
        retriever=retriever,
        vlm_provider=vlm,
        mcp_client=client,
    )
    assert client.calls_made == 0
    assert payload.status in (AnswerStatus.ANSWERED, AnswerStatus.ABSTAINED)


def test_workflow_drafts_ticket_and_stops_for_approval(
    test_principal: Principal,
    sample_retrieved_elements: list[RetrievedElement],
    mcp_service: MaintenanceService,
) -> None:
    retriever = FakeHybridRetriever(hits=sample_retrieved_elements)
    vlm = StubVlmProvider()
    client = McpClient(
        service=mcp_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-5",
    )

    # Ask to create a ticket on pompe_centrifuge_01 without providing approval token yet
    payload = run_workflow(
        question="Créer un ticket pour fuite importante sur pompe_centrifuge_01",
        principal=test_principal,
        retriever=retriever,
        vlm_provider=vlm,
        mcp_client=client,
        equipment_id="pompe_centrifuge_01",
        request_ticket=True,
    )
    assert payload.status is AnswerStatus.NEEDS_APPROVAL
    assert payload.proposed_action is not None
    assert payload.proposed_action.equipment_id == "pompe_centrifuge_01"
    assert payload.proposed_action.content_hash != ""


def test_workflow_executes_ticket_with_approval_token(
    test_principal: Principal,
    sample_retrieved_elements: list[RetrievedElement],
    mcp_service: MaintenanceService,
    approval_service: ApprovalService,
) -> None:
    retriever = FakeHybridRetriever(hits=sample_retrieved_elements)
    vlm = StubVlmProvider()
    client = McpClient(
        service=mcp_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-6",
    )

    title = "Remplacement garniture mécanique"
    desc = "Fuite constatée sur pompe_centrifuge_01 lors de la ronde."
    ch = canonical_content_hash(
        equipment_id="pompe_centrifuge_01",
        title=title,
        description=desc,
    )
    idem = derive_idempotency_key(user_id=test_principal.user_id, content_hash=ch)
    grant = approval_service.issue(
        user_id=test_principal.user_id,
        content_hash=ch,
        idempotency_key=idem,
    )

    payload = run_workflow(
        question="Créer le ticket validé",
        principal=test_principal,
        retriever=retriever,
        vlm_provider=vlm,
        mcp_client=client,
        equipment_id="pompe_centrifuge_01",
        request_ticket=True,
        ticket_title=title,
        ticket_description=desc,
        approval_token=grant.token,
    )
    assert payload.status is AnswerStatus.ANSWERED
    assert "Ticket créé avec succès" in payload.answer
    assert client.calls_made >= 1


def test_workflow_enforces_budget_limits_model_calls(
    test_principal: Principal,
    sample_retrieved_elements: list[RetrievedElement],
    mcp_service: MaintenanceService,
) -> None:
    retriever = FakeHybridRetriever(hits=sample_retrieved_elements)
    vlm = StubVlmProvider()
    client = McpClient(
        service=mcp_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-7",
    )
    # Set limit of 0 model calls to trigger BudgetExceededError
    limits = WorkflowLimits(max_model_calls=0)
    with pytest.raises(BudgetExceededError):
        run_workflow(
            question="Question",
            principal=test_principal,
            retriever=retriever,
            vlm_provider=vlm,
            mcp_client=client,
            limits=limits,
        )


def test_workflow_enforces_time_limit(
    test_principal: Principal,
    sample_retrieved_elements: list[RetrievedElement],
    mcp_service: MaintenanceService,
) -> None:
    class SlowRetriever(FakeHybridRetriever):
        def search(self, *args: Any, **kwargs: Any) -> list[RetrievedElement]:
            time.sleep(0.05)
            return super().search(*args, **kwargs)

    retriever = SlowRetriever(hits=sample_retrieved_elements)
    vlm = StubVlmProvider()
    client = McpClient(
        service=mcp_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-9",
    )
    # Set limit to 0.01s while retriever takes 0.05s
    limits = WorkflowLimits(max_processing_time_s=0.01)
    with pytest.raises(BudgetExceededError):
        run_workflow(
            question="Question",
            principal=test_principal,
            retriever=retriever,
            vlm_provider=vlm,
            mcp_client=client,
            limits=limits,
        )


def test_workflow_enforces_image_size_limit(
    test_principal: Principal,
    sample_retrieved_elements: list[RetrievedElement],
    mcp_service: MaintenanceService,
) -> None:
    retriever = FakeHybridRetriever(hits=sample_retrieved_elements)
    vlm = StubVlmProvider()
    client = McpClient(
        service=mcp_service,
        principal=test_principal,
        api_key=SecretStr("valid-test-mcp-key"),
        request_id="req-8",
    )
    oversized_image = b"x" * (11 * 1024 * 1024)  # 11 MB > 10 MB limit
    with pytest.raises(ValidationError):
        run_workflow(
            question="Photo",
            principal=test_principal,
            retriever=retriever,
            vlm_provider=vlm,
            mcp_client=client,
            image_bytes=oversized_image,
        )


def test_workflow_export_mermaid_diagram() -> None:
    from maintidoc.agent.workflow import export_workflow_graph_mermaid
    from maintidoc.retrieval.reranker import NoReranker

    class DummyRetriever:
        reranker = NoReranker()

        def search(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

    dummy_service = MaintenanceService(
        api_key=SecretStr("key"),
        approval_service=ApprovalService(
            secret_key="super-secret-key-32-bytes-minimum!",
            store=InMemoryApprovalStore(),
        ),
        database=MaintenanceDatabase(),
    )
    dummy_client = McpClient(
        service=dummy_service,
        principal=Principal(user_id="u", groups=frozenset({"g"}), is_active=True),
        api_key=SecretStr("key"),
        request_id="req",
    )
    graph = build_workflow_graph(
        retriever=DummyRetriever(),
        vlm_provider=StubVlmProvider(),
        mcp_client=dummy_client,
    )
    mermaid = export_workflow_graph_mermaid(graph)
    assert "verify_access" in mermaid
    assert "create_ticket" in mermaid
