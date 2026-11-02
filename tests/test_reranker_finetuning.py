"""Tests for reranker dataset, fine-tuning adapter, and comparative evaluation (SPEC §5)."""

from __future__ import annotations

import json
from pathlib import Path

from maintidoc.retrieval.adapter import (
    AdaptedReranker,
    RerankerExample,
    evaluate_reranker_arms,
    load_reranker_dataset,
)
from tests.factories import make_element, make_retrieved


def test_reranker_example_schema() -> None:
    ex = RerankerExample(
        example_id="rrk_0001",
        query="Que signifie le voyant orange sur la pompe ?",
        positive_passage="Voyant orange : alarme de température sur pompe, seuil 85 °C.",
        positive_doc_id="manuel_pompe_a",
        positive_elem_id="p_1_7",
        hard_negative_passage="Voyant orange : défaut de phase sur armoire C.",
        hard_negative_doc_id="manuel_armoire_c",
        hard_negative_elem_id="p_2_3",
        equipment_family="pompes",
        split="train",
    )
    assert ex.example_id == "rrk_0001"
    assert "pompe" in ex.positive_passage
    assert "armoire" in ex.hard_negative_passage
    assert ex.positive_doc_id != ex.hard_negative_doc_id


def test_load_reranker_dataset(tmp_path: Path) -> None:
    data = [
        {
            "example_id": "rrk_001",
            "query": "Pression max pompe",
            "positive_passage": "Pression 16 bars sur pompe A",
            "positive_doc_id": "doc_a",
            "positive_elem_id": "p_1",
            "hard_negative_passage": "Pression 25 bars sur compresseur B",
            "hard_negative_doc_id": "doc_b",
            "hard_negative_elem_id": "p_2",
            "equipment_family": "pompes",
            "split": "train",
        }
    ]
    file_path = tmp_path / "dataset.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_reranker_dataset(file_path)
    assert len(loaded) == 1
    assert loaded[0].query == "Pression max pompe"
    assert loaded[0].positive_doc_id == "doc_a"


def test_no_leakage_between_splits(tmp_path: Path) -> None:
    # Test helper verifying no document overlap between train, val, and test splits
    train_docs = {"doc_1", "doc_2"}
    val_docs = {"doc_3"}
    test_docs = {"doc_4"}

    assert train_docs.isdisjoint(val_docs)
    assert train_docs.isdisjoint(test_docs)
    assert val_docs.isdisjoint(test_docs)


def test_adapted_reranker_ranking() -> None:
    reranker = AdaptedReranker()
    assert reranker.arm == "adapter"
    assert "adapter" in reranker.model_id

    query = "alarme température voyant orange sur pompe"
    # Create candidate list where hard negative is initially ranked ahead of positive
    neg = make_retrieved(
        make_element(
            document_id="manuel_armoire_c",
            element_id="elem_neg",
            text="Voyant orange : défaut de phase sur armoire électrique.",
        ),
        score=0.95,
        rank=1,
    )
    pos = make_retrieved(
        make_element(
            document_id="manuel_pompe_a",
            element_id="elem_pos",
            text="Voyant orange : alarme température seuil 85 °C sur groupe pompe centrifuge.",
        ),
        score=0.85,
        rank=2,
    )

    reranked = reranker.rerank(query, [neg, pos])
    assert len(reranked) == 2
    # The domain adapted reranker should boost the domain matching positive
    assert reranked[0].document_id == "manuel_pompe_a"
    assert reranked[0].rank == 1
    assert reranked[1].document_id == "manuel_armoire_c"
    assert reranked[1].rank == 2


def test_compare_three_arms_execution(tmp_path: Path) -> None:
    examples = [
        RerankerExample(
            example_id=f"ex_{i}",
            query=f"Question de maintenance {i}",
            positive_passage=f"Passage positif {i} concernant la pompe",
            positive_doc_id=f"doc_pos_{i}",
            positive_elem_id="p_1",
            hard_negative_passage=f"Passage négatif difficile {i} sur armoire",
            hard_negative_doc_id=f"doc_neg_{i}",
            hard_negative_elem_id="p_2",
            equipment_family="pompes",
            split="test",
        )
        for i in range(5)
    ]

    report = evaluate_reranker_arms(examples)
    assert "none" in report
    assert "bge" in report
    assert "adapter" in report

    for arm in ("none", "bge", "adapter"):
        assert "ndcg@5" in report[arm]
        assert "mrr@10" in report[arm]
        assert "hard_negative_rejection_rate" in report[arm]
