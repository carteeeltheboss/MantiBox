"""Tests for HybridRetriever combining text, image, RRF and reranker with access control."""

from __future__ import annotations

from maintidoc.domain import Element
from maintidoc.retrieval.embeddings import HashingEmbedder
from maintidoc.retrieval.fusion import HybridRetriever
from maintidoc.retrieval.images import HashingImageEmbedder
from maintidoc.retrieval.index import SearchHit
from maintidoc.retrieval.reranker import DeterministicReranker, NoReranker
from maintidoc.security.access import Principal
from tests.factories import make_element


class FakeIndex:
    """Mock Qdrant index that records search calls and filters by principal."""

    def __init__(self, elements: list[Element], name: str = "index") -> None:
        self.elements = elements
        self.name = name
        self.last_principal: Principal | None = None
        self.last_query_vector: list[float] | None = None

    def search(
        self,
        query_vector: list[float],
        *,
        principal: Principal,
        limit: int = 20,
    ) -> list[SearchHit]:
        self.last_principal = principal
        self.last_query_vector = query_vector

        # Filter elements according to principal's allowed access groups
        hits: list[SearchHit] = []
        rank = 1
        for el in self.elements:
            if not principal.is_active or not principal.groups:
                continue
            # Must have non-empty intersection
            if el.access_groups.intersection(principal.groups):
                hits.append(SearchHit(element=el, score=1.0 - (rank * 0.05), rank=rank))
                rank += 1
                if len(hits) >= limit:
                    break
        return hits


class TestHybridRetriever:
    def test_combines_text_and_image_channels(self, technician: Principal) -> None:
        text_el = make_element(
            document_id="doc_a",
            element_id="p_1",
            text="Manuel de pompe",
            access_groups=technician.groups,
        )
        img_el = make_element(
            document_id="doc_a",
            element_id="fig_1",
            text="Schema de commande",
            image_key="fig1.png",
            access_groups=technician.groups,
        )

        text_idx = FakeIndex([text_el], name="text")
        img_idx = FakeIndex([img_el], name="image")

        text_emb = HashingEmbedder(dimension=32)
        img_emb = HashingImageEmbedder(dimension=32)

        retriever = HybridRetriever(
            text_index=text_idx,  # type: ignore[arg-type]
            image_index=img_idx,  # type: ignore[arg-type]
            text_embedder=text_emb,
            image_embedder=img_emb,
            reranker=NoReranker(),
        )

        results = retriever.search("pompe", principal=technician, top_n=5)
        assert len(results) == 2
        element_ids = {r.element_id for r in results}
        assert element_ids == {"p_1", "fig_1"}
        assert all(r.rrf_score is not None for r in results)

    def test_enforces_access_control(self, outsider: Principal, technician: Principal) -> None:
        secret_el = make_element(
            document_id="doc_restricted",
            element_id="p_secret",
            text="Données confidentielles",
            access_groups=technician.groups,  # only site_a
        )

        text_idx = FakeIndex([secret_el])
        img_idx = FakeIndex([])

        retriever = HybridRetriever(
            text_index=text_idx,  # type: ignore[arg-type]
            image_index=img_idx,  # type: ignore[arg-type]
            text_embedder=HashingEmbedder(dimension=16),
            image_embedder=HashingImageEmbedder(dimension=16),
        )

        # Outsider has site_b; must receive 0 elements
        results = retriever.search("secret", principal=outsider)
        assert len(results) == 0

    def test_reranker_reorders_fused_results(self, technician: Principal) -> None:
        el1 = make_element(element_id="e1", text="Généralités", access_groups=technician.groups)
        el2 = make_element(
            element_id="e2",
            text="Alarme voyant orange",
            access_groups=technician.groups,
        )

        text_idx = FakeIndex([el1, el2])  # el1 is rank 1 in fake index
        img_idx = FakeIndex([])

        retriever = HybridRetriever(
            text_index=text_idx,  # type: ignore[arg-type]
            image_index=img_idx,  # type: ignore[arg-type]
            text_embedder=HashingEmbedder(dimension=16),
            image_embedder=HashingImageEmbedder(dimension=16),
            reranker=DeterministicReranker(),
        )

        results = retriever.search("voyant orange", principal=technician)
        assert len(results) == 2
        # Deterministic reranker prefers "Alarme voyant orange"
        assert results[0].element_id == "e2"
