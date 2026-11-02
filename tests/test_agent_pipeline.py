"""Tests for the vertical slice RAG pipeline (Issue #9, SPEC §4.2, §7, §8).

Wired flow:
Question -> HybridRetriever -> render_sources -> VlmProvider -> citation check -> AnswerPayload.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pytest
from eval.benchmark_data import SEED_QUERIES
from qdrant_client import QdrantClient

from maintidoc.agent.pipeline import AnswerPipeline
from maintidoc.domain import DocumentRecord
from maintidoc.ingestion.pdf import extract_pdf
from maintidoc.retrieval.embeddings import HashingEmbedder
from maintidoc.retrieval.fusion import HybridRetriever
from maintidoc.retrieval.images import HashingImageEmbedder
from maintidoc.retrieval.index import IndexConfig, QdrantIndex
from maintidoc.retrieval.reranker import NoReranker
from maintidoc.schemas import AnswerPayload, AnswerStatus, ElementKind
from maintidoc.security.access import Principal
from maintidoc.storage.objects import FilesystemStore
from maintidoc.vlm.base import ImageInput
from maintidoc.vlm.stub import StubPolicy, StubVlmProvider
from tests.factories import make_element


@pytest.fixture
def memory_retriever() -> HybridRetriever:
    client = QdrantClient(":memory:")
    text_cfg = IndexConfig(collection="test_text", dimension=32)
    img_cfg = IndexConfig(collection="test_img", dimension=32)
    text_idx = QdrantIndex(client, text_cfg)
    img_idx = QdrantIndex(client, img_cfg)
    text_idx.ensure_collection()
    img_idx.ensure_collection()

    elem = make_element(
        document_id="manuel_pompe_a",
        version="2024-03",
        page=1,
        element_id="paragraph_1_7",
        kind=ElementKind.PARAGRAPH,
        text="Le voyant orange signale une alarme de température.",
        access_groups=frozenset({"maintenance", "site_a"}),
    )

    text_emb = HashingEmbedder(dimension=32)
    img_emb = HashingImageEmbedder(dimension=32)

    text_idx.upsert([elem], text_emb.embed_documents([elem.text]), embedder_model=text_emb.model_id)

    return HybridRetriever(
        text_index=text_idx,
        image_index=img_idx,
        text_embedder=text_emb,
        image_embedder=img_emb,
        reranker=NoReranker(),
    )


class TestAnswerPipeline:
    def test_text_question_returns_answered_with_real_citation(
        self,
        memory_retriever: HybridRetriever,
        technician: Principal,
    ) -> None:
        vlm = StubVlmProvider(policy=StubPolicy.AUTO)
        pipeline = AnswerPipeline(retriever=memory_retriever, vlm_provider=vlm)

        payload = pipeline.answer_question(
            "Que signifie le voyant orange ?",
            principal=technician,
        )

        assert isinstance(payload, AnswerPayload)
        assert payload.status is AnswerStatus.ANSWERED
        assert len(payload.citations) >= 1

        first_cite = payload.citations[0]
        assert first_cite.document_id == "manuel_pompe_a"
        assert first_cite.element_id == "paragraph_1_7"
        assert first_cite.page == 1

        assert payload.provenance is not None
        assert payload.provenance.vlm_provider == "stub"
        assert "answer" in payload.provenance.prompt_versions

    def test_no_retrieved_evidence_returns_abstained(
        self,
        memory_retriever: HybridRetriever,
        outsider: Principal,
    ) -> None:
        # Outsider has site_b, so access control blocks all elements
        vlm = StubVlmProvider(policy=StubPolicy.AUTO)
        pipeline = AnswerPipeline(retriever=memory_retriever, vlm_provider=vlm)

        payload = pipeline.answer_question(
            "Que signifie le voyant orange ?",
            principal=outsider,
        )

        assert isinstance(payload, AnswerPayload)
        assert payload.status is AnswerStatus.ABSTAINED
        assert len(payload.citations) == 0
        assert len(payload.uncertainties) >= 1

    def test_citation_verification_strips_hallucinated_citations(
        self,
        memory_retriever: HybridRetriever,
        technician: Principal,
    ) -> None:
        # Responder that attempts to cite an unretrieved document
        def fake_responder(instruction: str, images: Sequence[ImageInput]) -> str:
            return json.dumps(
                {
                    "status": "answered",
                    "answer": "Réponse inventée",
                    "citations": [
                        {
                            "document_id": "document_imaginaire",
                            "page": 99,
                            "element_id": "p_fake",
                        }
                    ],
                    "uncertainties": [],
                    "clarification_question": None,
                    "proposed_action": None,
                }
            )

        vlm = StubVlmProvider(responder=fake_responder)
        pipeline = AnswerPipeline(retriever=memory_retriever, vlm_provider=vlm)

        payload = pipeline.answer_question(
            "Question test",
            principal=technician,
        )

        # Because the only citation was unretrieved, it must downgrade to abstained
        assert payload.status is AnswerStatus.ABSTAINED
        assert len(payload.citations) == 0
        assert any("vérifiée" in u.lower() or "source" in u.lower() for u in payload.uncertainties)

    def test_end_to_end_corpus_scenario(self) -> None:
        # End to end scenario on actual corpus
        data = json.loads(Path("corpus/manifest.json").read_text(encoding="utf-8"))
        doc_dict = next(d for d in data["documents"] if d["document_id"] == "manuel_pompe_a")
        doc = DocumentRecord.model_validate(
            {k: v for k, v in doc_dict.items() if k in DocumentRecord.model_fields}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            store = FilesystemStore(Path(tmp_dir))
            pdf_bytes = (Path("corpus") / doc.object_key).read_bytes()
            res = extract_pdf(data=pdf_bytes, document=doc, store=store)

            client = QdrantClient(":memory:")
            text_cfg = IndexConfig(collection="e2e_text", dimension=32)
            img_cfg = IndexConfig(collection="e2e_img", dimension=32)
            t_idx = QdrantIndex(client, text_cfg)
            i_idx = QdrantIndex(client, img_cfg)
            t_idx.ensure_collection()
            i_idx.ensure_collection()

            text_emb = HashingEmbedder(dimension=32)
            img_emb = HashingImageEmbedder(dimension=32)

            text_elems = [el for el in res.elements if el.text.strip()]
            t_idx.upsert(
                text_elems,
                text_emb.embed_documents([e.text for e in text_elems]),
                embedder_model=text_emb.model_id,
            )

            retriever = HybridRetriever(
                text_index=t_idx,
                image_index=i_idx,
                text_embedder=text_emb,
                image_embedder=img_emb,
                reranker=NoReranker(),
            )

            pipeline = AnswerPipeline(
                retriever=retriever, vlm_provider=StubVlmProvider(policy=StubPolicy.AUTO)
            )

            scenario = SEED_QUERIES[0]  # voyant orange pompe
            payload = pipeline.answer_question(scenario.query, principal=scenario.principal)

            assert isinstance(payload, AnswerPayload)
            assert payload.status is AnswerStatus.ANSWERED
            assert len(payload.citations) > 0
            assert payload.citations[0].document_id == "manuel_pompe_a"
