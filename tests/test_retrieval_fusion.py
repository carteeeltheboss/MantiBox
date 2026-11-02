"""Tests for Reciprocal Rank Fusion (SPEC §4.2).

Formula: RRF(d) = sum_{c in Channels} 1 / (k + rank_c(d))
Default k = 60 (RetrievalSettings.rrf_k).
"""

from __future__ import annotations

import pytest

from maintidoc.domain import Element
from maintidoc.retrieval.fusion import reciprocal_rank_fusion
from maintidoc.retrieval.index import SearchHit
from tests.factories import make_element, make_retrieved


def _hit(elem: Element, score: float, rank: int) -> SearchHit:
    return SearchHit(element=elem, score=score, rank=rank)


class TestReciprocalRankFusion:
    def test_single_channel_text(self) -> None:
        elem1 = make_element(element_id="el_1", text="Text 1")
        elem2 = make_element(element_id="el_2", text="Text 2")
        text_hits = [_hit(elem1, score=0.9, rank=1), _hit(elem2, score=0.8, rank=2)]

        fused = reciprocal_rank_fusion(text_hits=text_hits, image_hits=[], k=60)

        assert len(fused) == 2
        assert fused[0].element_id == "el_1"
        assert fused[0].rank == 1
        assert fused[0].channel == "text"
        assert fused[0].rrf_score == pytest.approx(1.0 / (60 + 1))

        assert fused[1].element_id == "el_2"
        assert fused[1].rank == 2
        assert fused[1].channel == "text"
        assert fused[1].rrf_score == pytest.approx(1.0 / (60 + 2))

    def test_single_channel_image(self) -> None:
        elem1 = make_element(element_id="fig_1", image_key="f1.png", text="Fig 1")
        image_hits = [_hit(elem1, score=0.85, rank=1)]

        fused = reciprocal_rank_fusion(text_hits=[], image_hits=image_hits, k=60)

        assert len(fused) == 1
        assert fused[0].element_id == "fig_1"
        assert fused[0].rank == 1
        assert fused[0].channel == "image"
        assert fused[0].rrf_score == pytest.approx(1.0 / 61)

    def test_fusion_of_both_channels(self) -> None:
        # elem_shared appears in text rank 1 and image rank 2
        # elem_text appears only in text rank 2
        # elem_image appears only in image rank 1
        elem_shared = make_element(
            document_id="doc_a",
            element_id="shared_1",
            text="Shared element",
            image_key="shared.png",
        )
        elem_text = make_element(document_id="doc_a", element_id="text_only", text="Text element")
        elem_image = make_element(
            document_id="doc_b",
            element_id="image_only",
            text="Image element",
            image_key="img.png",
        )

        text_hits = [
            _hit(elem_shared, score=0.95, rank=1),
            _hit(elem_text, score=0.80, rank=2),
        ]
        image_hits = [
            _hit(elem_image, score=0.90, rank=1),
            _hit(elem_shared, score=0.75, rank=2),
        ]

        # RRF scores (k=60):
        # shared: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.016393 + 0.016129 = 0.032522
        # image_only: 1/(60+1) = 1/61 = 0.016393
        # text_only: 1/(60+2) = 1/62 = 0.016129

        fused = reciprocal_rank_fusion(text_hits=text_hits, image_hits=image_hits, k=60)

        assert len(fused) == 3
        # Rank 1: shared element
        assert fused[0].element_id == "shared_1"
        assert fused[0].rank == 1
        assert fused[0].channel == "fused"
        assert fused[0].rrf_score == pytest.approx((1.0 / 61) + (1.0 / 62))

        # Rank 2: image_only element
        assert fused[1].element_id == "image_only"
        assert fused[1].rank == 2
        assert fused[1].channel == "image"
        assert fused[1].rrf_score == pytest.approx(1.0 / 61)

        # Rank 3: text_only element
        assert fused[2].element_id == "text_only"
        assert fused[2].rank == 3
        assert fused[2].channel == "text"
        assert fused[2].rrf_score == pytest.approx(1.0 / 62)

    def test_top_n_truncation(self) -> None:
        elems = [make_element(element_id=f"el_{i}", text=f"T {i}") for i in range(10)]
        text_hits = [_hit(el, score=1.0 - (i * 0.05), rank=i + 1) for i, el in enumerate(elems)]

        fused = reciprocal_rank_fusion(text_hits=text_hits, image_hits=[], k=60, top_n=5)
        assert len(fused) == 5
        assert [f.element_id for f in fused] == [f"el_{i}" for i in range(5)]

    def test_custom_k(self) -> None:
        elem = make_element(element_id="el_1", text="T 1")
        text_hits = [_hit(elem, score=0.9, rank=1)]
        fused = reciprocal_rank_fusion(text_hits=text_hits, image_hits=[], k=20)
        assert fused[0].rrf_score == pytest.approx(1.0 / 21)

    def test_rejects_negative_or_zero_k(self) -> None:
        elem = make_element(element_id="el_1", text="T 1")
        text_hits = [_hit(elem, score=0.9, rank=1)]
        with pytest.raises(ValueError, match="k must be >= 1"):
            reciprocal_rank_fusion(text_hits=text_hits, image_hits=[], k=0)

    def test_empty_both_channels(self) -> None:
        assert reciprocal_rank_fusion(text_hits=[], image_hits=[]) == []

    def test_accepts_retrieved_elements(self) -> None:
        elem = make_element(element_id="el_1", text="T 1")
        retrieved = [make_retrieved(elem, rank=1, score=0.8, channel="text")]
        fused = reciprocal_rank_fusion(text_hits=retrieved, image_hits=[])
        assert len(fused) == 1
        assert fused[0].element_id == "el_1"
