"""Access-control and approval security regression suite (Issue #32, SPEC §9, §10).

Acceptance criteria:
- Zero cross-group leaks in retrieval and document access
- Zero ticket creations without valid, user-bound approval token in tests
- Tampering and replay protection verified under attack conditions
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr
from qdrant_client import QdrantClient

from maintidoc.errors import AccessDeniedError, ApprovalInvalidError
from maintidoc.mcp.contracts import CreateTicketRequest, ToolCallContext
from maintidoc.mcp.server import (
    MaintenanceDatabase,
    MaintenanceService,
)
from maintidoc.retrieval.access import DocumentOpener, InMemoryCatalog
from maintidoc.retrieval.embeddings import HashingEmbedder
from maintidoc.retrieval.index import IndexConfig, QdrantIndex
from maintidoc.security.access import Principal
from maintidoc.security.approval import (
    ApprovalService,
    InMemoryApprovalStore,
    canonical_content_hash,
    derive_idempotency_key,
)
from maintidoc.storage.objects import FilesystemStore
from tests.factories import make_element


class TestAccessControlZeroLeaks:
    def test_zero_cross_group_leaks_in_qdrant_query(self) -> None:
        client = QdrantClient(":memory:")
        cfg = IndexConfig(collection="sec_reg_text", dimension=16)
        idx = QdrantIndex(client, cfg)
        idx.ensure_collection()

        elem_site_a = make_element(
            document_id="doc_site_a",
            element_id="p1",
            text="Secret procédure Site A",
            access_groups=frozenset({"site_a"}),
        )
        elem_site_b = make_element(
            document_id="doc_site_b",
            element_id="p2",
            text="Secret procédure Site B",
            access_groups=frozenset({"site_b"}),
        )
        elem_restricted = make_element(
            document_id="doc_supervisors",
            element_id="p3",
            text="Secret de consignation restreint",
            access_groups=frozenset({"supervisors"}),
        )

        emb = HashingEmbedder(dimension=16)
        all_elems = [elem_site_a, elem_site_b, elem_restricted]
        idx.upsert(
            all_elems,
            emb.embed_documents([e.text for e in all_elems]),
            embedder_model=emb.model_id,
        )

        # Tech A (site_a only)
        tech_a = Principal(user_id="tech_a", groups=frozenset({"site_a", "maintenance"}))
        hits_a = idx.search(emb.embed_query("Secret"), principal=tech_a, limit=10)
        assert len(hits_a) == 1
        assert hits_a[0].document_id == "doc_site_a"

        # Tech B (site_b only)
        tech_b = Principal(user_id="tech_b", groups=frozenset({"site_b", "maintenance"}))
        hits_b = idx.search(emb.embed_query("Secret"), principal=tech_b, limit=10)
        assert len(hits_b) == 1
        assert hits_b[0].document_id == "doc_site_b"

        # Outsider with no matching group
        outsider = Principal(user_id="visitor", groups=frozenset({"public"}))
        hits_out = idx.search(emb.embed_query("Secret"), principal=outsider, limit=10)
        assert len(hits_out) == 0

        # Supervisor has supervisors group
        supervisor = Principal(
            user_id="sup", groups=frozenset({"supervisors", "site_a", "maintenance"})
        )
        hits_sup = idx.search(emb.embed_query("Secret"), principal=supervisor, limit=10)
        doc_ids = {h.document_id for h in hits_sup}
        assert "doc_site_a" in doc_ids
        assert "doc_supervisors" in doc_ids
        assert "doc_site_b" not in doc_ids  # Still no access to site_b!

    def test_document_opener_blocks_unauthorized_fetches(self, tmp_path: Path) -> None:
        path = tmp_path / "reg_store"
        store = FilesystemStore(path)

        # Store 2 documents
        store.put("documents/doc_a.pdf", b"%PDF-1.4 site a content")
        store.put("documents/doc_b.pdf", b"%PDF-1.4 site b content")

        catalog = InMemoryCatalog({"doc_a": frozenset({"site_a"}), "doc_b": frozenset({"site_b"})})
        opener = DocumentOpener(store=store, catalog=catalog)

        tech_a = Principal(user_id="tech_a", groups=frozenset({"site_a"}))
        # Tech A can open doc A
        bytes_a = opener.open_document(principal=tech_a, document_id="doc_a")
        assert b"site a content" in bytes_a

        # Tech A CANNOT open doc B
        with pytest.raises(AccessDeniedError):
            opener.open_document(principal=tech_a, document_id="doc_b")


class TestApprovalZeroUnapprovedTickets:
    def test_zero_tickets_without_valid_approval_token(self) -> None:
        secret = "secret-key-32-bytes-minimum-required-here"
        store = InMemoryApprovalStore()
        approval_svc = ApprovalService(secret_key=secret, store=store)
        db = MaintenanceDatabase.seed_default()
        mcp_svc = MaintenanceService(
            database=db,
            api_key=SecretStr("mcp-key"),
            approval_service=approval_svc,
        )

        tech = Principal(user_id="tech_1", groups=frozenset({"site_a", "maintenance"}))
        content_hash = canonical_content_hash(
            equipment_id="pompe_centrifuge_01",
            title="Intervention critique",
            description="Changement du roulement défectueux",
        )
        idempotency_key = derive_idempotency_key(user_id=tech.user_id, content_hash=content_hash)

        ctx_tech = ToolCallContext(
            principal=tech,
            api_key=SecretStr("mcp-key"),
            request_id="req-sec-1",
        )

        # Attempt 1: Call without token fails Pydantic schema validation
        with pytest.raises(Exception):  # noqa: B017
            CreateTicketRequest(  # type: ignore[call-arg]
                equipment_id="pompe_centrifuge_01",
                title="Intervention critique",
                description="Changement du roulement défectueux",
                idempotency_key=idempotency_key,
            )

        # Attempt 2: Call with forged / bogus token
        req_bad_token = CreateTicketRequest(
            equipment_id="pompe_centrifuge_01",
            title="Intervention critique",
            description="Changement du roulement défectueux",
            approval_token="forged-approval-token-invalid-jwt-here",
            idempotency_key=idempotency_key,
        )
        with pytest.raises(ApprovalInvalidError):
            mcp_svc.create_ticket(
                request=req_bad_token,
                context=ctx_tech,
            )

        # Issue legitimate token for user tech_1
        grant = approval_svc.issue(
            user_id="tech_1",
            content_hash=content_hash,
            idempotency_key=idempotency_key,
        )
        valid_token = grant.token

        # Attempt 3: Another user (tech_2) tries to use tech_1's token
        imposter = Principal(user_id="tech_2", groups=frozenset({"site_a", "maintenance"}))
        ctx_imposter = ToolCallContext(
            principal=imposter,
            api_key=SecretStr("mcp-key"),
            request_id="req-sec-2",
        )
        req_valid = CreateTicketRequest(
            equipment_id="pompe_centrifuge_01",
            title="Intervention critique",
            description="Changement du roulement défectueux",
            approval_token=valid_token,
            idempotency_key=idempotency_key,
        )
        with pytest.raises(ApprovalInvalidError):
            mcp_svc.create_ticket(
                request=req_valid,
                context=ctx_imposter,
            )

        # Attempt 4: Tampered content with valid token (e.g. title altered)
        alt_hash = canonical_content_hash(
            equipment_id="pompe_centrifuge_01",
            title="Intervention critique [ALTERED]",
            description="Changement du roulement défectueux",
        )
        alt_idem = derive_idempotency_key(user_id=tech.user_id, content_hash=alt_hash)
        req_tampered = CreateTicketRequest(
            equipment_id="pompe_centrifuge_01",
            title="Intervention critique [ALTERED]",
            description="Changement du roulement défectueux",
            approval_token=valid_token,
            idempotency_key=alt_idem,
        )
        with pytest.raises(ApprovalInvalidError):
            mcp_svc.create_ticket(
                request=req_tampered,
                context=ctx_tech,
            )

        # Attempt 5: Legitimate creation succeeds
        res = mcp_svc.create_ticket(
            request=req_valid,
            context=ctx_tech,
        )
        assert res.ticket_id.startswith("tick-")

        # Attempt 6: Replay attack (re-submitting the exact same token)
        # Change idempotency key so deduplication check doesn't fire before token verification
        req_replay = CreateTicketRequest(
            equipment_id="pompe_centrifuge_01",
            title="Intervention critique",
            description="Changement du roulement défectueux",
            approval_token=valid_token,
            idempotency_key=idempotency_key + "-replay",
        )
        with pytest.raises(ApprovalInvalidError):
            mcp_svc.create_ticket(
                request=req_replay,
                context=ctx_tech,
            )
