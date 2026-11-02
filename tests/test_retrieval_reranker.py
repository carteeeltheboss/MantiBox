"""Tests for the Reranker seam and arms (SPEC §5).

SPEC §5 specifies comparisons:
- sans reranker ("none")
- avec reranker d'origine ("bge" -> BAAI/bge-reranker-v2-m3)
- avec reranker adapté ("adapter")
plus "deterministic" for fast torch-free tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maintidoc.errors import ConfigurationError
from maintidoc.retrieval.reranker import (
    BgeReranker,
    DeterministicReranker,
    NoReranker,
    Reranker,
    make_reranker,
)
from tests.factories import make_element, make_retrieved


class TestNoReranker:
    def test_satisfies_protocol(self) -> None:
        reranker = NoReranker()
        assert isinstance(reranker, Reranker)
        assert reranker.arm == "none"

    def test_preserves_order(self) -> None:
        elem1 = make_element(element_id="el_1", text="Text 1")
        elem2 = make_element(element_id="el_2", text="Text 2")
        retrieved = [
            make_retrieved(elem1, rank=1, score=0.9),
            make_retrieved(elem2, rank=2, score=0.8),
        ]
        reranked = NoReranker().rerank("voyant orange", retrieved)
        assert len(reranked) == 2
        assert reranked[0].element_id == "el_1"
        assert reranked[0].rank == 1
        assert reranked[1].element_id == "el_2"
        assert reranked[1].rank == 2

    def test_top_n_truncation(self) -> None:
        elem1 = make_element(element_id="el_1", text="Text 1")
        elem2 = make_element(element_id="el_2", text="Text 2")
        retrieved = [
            make_retrieved(elem1, rank=1, score=0.9),
            make_retrieved(elem2, rank=2, score=0.8),
        ]
        reranked = NoReranker().rerank("voyant orange", retrieved, top_n=1)
        assert len(reranked) == 1
        assert reranked[0].element_id == "el_1"

    def test_empty_candidates(self) -> None:
        assert NoReranker().rerank("query", []) == []


class TestDeterministicReranker:
    def test_satisfies_protocol(self) -> None:
        reranker = DeterministicReranker()
        assert isinstance(reranker, Reranker)
        assert reranker.arm == "deterministic"

    def test_reranks_by_keyword_overlap(self) -> None:
        elem_match = make_element(element_id="match", text="Alarme voyant orange de température.")
        elem_other = make_element(element_id="other", text="Pression hydraulique nominale.")
        # elem_other starts at rank 1, match at rank 2
        retrieved = [
            make_retrieved(elem_other, rank=1, score=0.9),
            make_retrieved(elem_match, rank=2, score=0.5),
        ]

        reranked = DeterministicReranker().rerank("voyant orange", retrieved)
        assert len(reranked) == 2
        # elem_match should now be rank 1
        assert reranked[0].element_id == "match"
        assert reranked[0].rank == 1
        assert reranked[0].score > reranked[1].score
        assert reranked[1].element_id == "other"
        assert reranked[1].rank == 2

    def test_deterministic_and_reproducible(self) -> None:
        elem1 = make_element(element_id="e1", text="Panneau de commande")
        elem2 = make_element(element_id="e2", text="Armoire electrique")
        retrieved = [make_retrieved(elem1, rank=1), make_retrieved(elem2, rank=2)]

        r1 = DeterministicReranker().rerank("commande", retrieved)
        r2 = DeterministicReranker().rerank("commande", retrieved)
        assert [x.score for x in r1] == [x.score for x in r2]

    def test_handles_empty_text_gracefully(self) -> None:
        elem = make_element(element_id="fig", text="", image_key="f.png")
        retrieved = [make_retrieved(elem, rank=1, score=0.5)]
        reranked = DeterministicReranker().rerank("query", retrieved)
        assert len(reranked) == 1
        assert reranked[0].element_id == "fig"


class TestBgeRerankerSeam:
    def test_lazy_loading(self) -> None:
        reranker = BgeReranker(model_name_or_path="BAAI/bge-reranker-v2-m3")
        assert reranker._model is None
        assert reranker.arm == "bge"

    def test_adapter_arm(self) -> None:
        reranker = BgeReranker(
            model_name_or_path="BAAI/bge-reranker-v2-m3",
            adapter_path="adapters/reranker_v1",
        )
        assert reranker.arm == "adapter"

    def test_missing_weights_path_raises_configuration_error(self, tmp_path: Path) -> None:
        # Use an empty local directory: must raise ConfigurationError without network call
        reranker = BgeReranker(model_name_or_path=str(tmp_path))
        elem = make_element(element_id="el_1", text="Text")
        retrieved = [make_retrieved(elem)]
        with pytest.raises(ConfigurationError):
            reranker.rerank("query", retrieved)


class TestFactory:
    def test_make_none(self) -> None:
        reranker = make_reranker(backend="none")
        assert isinstance(reranker, NoReranker)
        assert reranker.arm == "none"

    def test_make_deterministic(self) -> None:
        reranker = make_reranker(backend="deterministic")
        assert isinstance(reranker, DeterministicReranker)
        assert reranker.arm == "deterministic"

    def test_make_bge(self) -> None:
        reranker = make_reranker(backend="bge")
        assert isinstance(reranker, BgeReranker)
        assert reranker.arm == "bge"

    def test_make_adapter(self) -> None:
        reranker = make_reranker(backend="adapter", adapter_path="models/adapter")
        assert isinstance(reranker, BgeReranker)
        assert reranker.arm == "adapter"

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown reranker backend"):
            make_reranker(backend="unknown")  # type: ignore[arg-type]
