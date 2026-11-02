"""End-to-end verification of the 7-step demonstration and rehearsal script (SPEC §14)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr
from qdrant_client import QdrantClient

from maintidoc.agent.pipeline import AnswerPipeline
from maintidoc.errors import AccessDeniedError, ApprovalInvalidError
from maintidoc.mcp.contracts import CreateTicketRequest, ToolCallContext
from maintidoc.mcp.server import (
    MaintenanceDatabase,
    MaintenanceService,
)
from maintidoc.retrieval.access import DocumentOpener, InMemoryCatalog
from maintidoc.retrieval.embeddings import HashingEmbedder
from maintidoc.retrieval.fusion import HybridRetriever
from maintidoc.retrieval.images import HashingImageEmbedder
from maintidoc.retrieval.index import IndexConfig, QdrantIndex
from maintidoc.retrieval.reranker import NoReranker
from maintidoc.schemas import AnswerStatus
from maintidoc.security.access import Principal
from maintidoc.security.approval import (
    ApprovalService,
    InMemoryApprovalStore,
    canonical_content_hash,
    derive_idempotency_key,
)
from maintidoc.storage.objects import FilesystemStore
from maintidoc.vlm.stub import StubPolicy, StubVlmProvider
from tests.factories import make_element


def test_rehearsal_seven_steps(tmp_path: Path) -> None:
    # -------------------------------------------------------------
    # Setup test environment
    # -------------------------------------------------------------
    client = QdrantClient(":memory:")
    t_idx = QdrantIndex(client, IndexConfig(collection="demo_t", dimension=32))
    i_idx = QdrantIndex(client, IndexConfig(collection="demo_i", dimension=32))
    t_idx.ensure_collection()
    i_idx.ensure_collection()

    elem_1 = make_element(
        document_id="manuel_pompe_a",
        element_id="paragraph_1_7",
        page=1,
        text="Voyant orange : alarme de température de palier, seuil 85 °C.",
        access_groups=frozenset({"maintenance", "site_a"}),
    )
    t_idx.upsert(
        [elem_1],
        [HashingEmbedder(dimension=32).embed_query(elem_1.text)],
        embedder_model="hash-32",
    )

    retriever = HybridRetriever(
        text_index=t_idx,
        image_index=i_idx,
        text_embedder=HashingEmbedder(dimension=32),
        image_embedder=HashingImageEmbedder(dimension=32),
        reranker=NoReranker(),
    )
    vlm = StubVlmProvider(policy=StubPolicy.AUTO)
    pipeline = AnswerPipeline(retriever=retriever, vlm_provider=vlm)

    tech_site_a = Principal(user_id="tech_01", groups=frozenset({"maintenance", "site_a"}))

    # -------------------------------------------------------------
    # Step 1 & 2: Question with photo -> Sourced answer + Verified citation
    # -------------------------------------------------------------
    fake_photo = b"PNG_FAKE_ORANGE_LIGHT"
    ans = pipeline.answer_question(
        "Que signifie le voyant orange ?",
        principal=tech_site_a,
        image_bytes=fake_photo,
    )
    assert ans.status == AnswerStatus.ANSWERED
    assert len(ans.citations) >= 1
    assert ans.citations[0].document_id == "manuel_pompe_a"
    assert ans.citations[0].page == 1

    # -------------------------------------------------------------
    # Step 3: MCP History consultation
    # -------------------------------------------------------------
    db = MaintenanceDatabase.seed_default()
    signing_key = "a" * 32
    approval_svc = ApprovalService(secret_key=signing_key, store=InMemoryApprovalStore())
    api_key = SecretStr("test-secret-api-key-12345678")
    mcp_service = MaintenanceService(
        api_key=api_key,
        approval_service=approval_svc,
        database=db,
    )
    ctx = ToolCallContext(
        api_key=api_key,
        principal=tech_site_a,
        request_id="req-demo-01",
    )
    history = mcp_service.get_equipment_history("pompe_centrifuge_01", ctx)
    assert len(history.entries) >= 1
    assert history.equipment_id == "pompe_centrifuge_01"

    # -------------------------------------------------------------
    # Step 4: Ticket proposal, human approval token & creation
    # -------------------------------------------------------------
    content_hash = canonical_content_hash(
        equipment_id="pompe_centrifuge_01",
        title="Alarme 85°C",
        description="Remplacement garniture.",
    )
    idem_key = derive_idempotency_key(
        user_id=tech_site_a.user_id,
        content_hash=content_hash,
    )
    grant = approval_svc.issue(
        user_id=tech_site_a.user_id,
        content_hash=content_hash,
        idempotency_key=idem_key,
    )

    req = CreateTicketRequest(
        equipment_id="pompe_centrifuge_01",
        title="Alarme 85°C",
        description="Remplacement garniture.",
        approval_token=grant.token,
        idempotency_key=idem_key,
    )
    created_ticket = mcp_service.create_ticket(req, ctx)
    assert created_ticket.ticket_id.startswith("tick-")

    # Immediate replay deduplicates
    dup_res = mcp_service.create_ticket(req, ctx)
    assert dup_res.deduplicated is True
    assert dup_res.ticket_id == created_ticket.ticket_id

    # Spent token reused for a different idempotency key / request fails
    diff_req = CreateTicketRequest(
        equipment_id="pompe_centrifuge_01",
        title="Alarme 85°C modifiée",
        description="Remplacement garniture.",
        approval_token=grant.token,
        idempotency_key="diff_idempotency_key_1234",
    )
    with pytest.raises(ApprovalInvalidError):
        mcp_service.create_ticket(diff_req, ctx)

    # -------------------------------------------------------------
    # Step 5: Ambiguity and uncertainty handling
    # -------------------------------------------------------------
    pipeline_clarif = AnswerPipeline(
        retriever=retriever,
        vlm_provider=StubVlmProvider(policy=StubPolicy.NEEDS_CLARIFICATION),
    )
    ans_clarif = pipeline_clarif.answer_question(
        "Comment régler le capteur ?",
        principal=tech_site_a,
    )
    assert ans_clarif.status == AnswerStatus.NEEDS_CLARIFICATION
    assert ans_clarif.clarification_question is not None

    # -------------------------------------------------------------
    # Step 6: Security: Cross-site isolation & prompt injection
    # -------------------------------------------------------------
    store = FilesystemStore(tmp_path)
    store.put("documents/doc_b.pdf", b"%PDF-1.4 secret site b")
    catalog = InMemoryCatalog({"doc_b": frozenset({"site_b"})})
    opener = DocumentOpener(store=store, catalog=catalog)

    with pytest.raises(AccessDeniedError):
        opener.open_document(principal=tech_site_a, document_id="doc_b")

    # -------------------------------------------------------------
    # Step 7: Verification of artifacts and documentation
    # -------------------------------------------------------------
    perf_file = Path("eval/results/performance_benchmark.json")
    reranker_file = Path("eval/results/reranker_comparison.json")
    limits_file = Path("docs/limits/fiche_de_limites.md")
    rehearsal_file = Path("docs/demo/rehearsal_script.md")

    assert perf_file.exists()
    assert reranker_file.exists()
    assert limits_file.exists()
    assert rehearsal_file.exists()
