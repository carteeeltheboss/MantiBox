"""Tests for retrieval evaluation metrics: Recall@K, MRR@K, nDCG@K (SPEC §5, §10).

Formulas:
- Recall@K: |relevant in top K| / |total relevant|
- MRR@K: 1 / rank of first relevant item in top K, or 0.0
- nDCG@K: DCG@K / IDCG@K where DCG@K = sum((2^rel_i - 1) / log2(i + 1))
"""

from __future__ import annotations

import math

import pytest

from maintidoc.retrieval.metrics import (
    compute_retrieval_metrics,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2"}
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        retrieved = ["doc1", "doc3", "doc4"]
        relevant = {"doc1", "doc2"}
        # only doc1 is in top 3
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(0.5)
        # only doc1 is in top 1
        assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(0.5)

    def test_zero_recall(self) -> None:
        retrieved = ["doc3", "doc4"]
        relevant = {"doc1", "doc2"}
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)

    def test_empty_relevant_set(self) -> None:
        # Edge case: no relevant documents defined
        assert recall_at_k(["doc1"], set(), k=5) == pytest.approx(0.0)

    def test_empty_retrieved_set(self) -> None:
        assert recall_at_k([], {"doc1"}, k=5) == pytest.approx(0.0)

    def test_k_smaller_than_retrieved(self) -> None:
        retrieved = ["doc3", "doc4", "doc1"]
        relevant = {"doc1"}
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)
        assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


class TestMrrAtK:
    def test_first_hit_at_rank_1(self) -> None:
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}
        assert mrr_at_k(retrieved, relevant, k=10) == pytest.approx(1.0)

    def test_first_hit_at_rank_2(self) -> None:
        retrieved = ["doc2", "doc1", "doc3"]
        relevant = {"doc1"}
        assert mrr_at_k(retrieved, relevant, k=10) == pytest.approx(0.5)

    def test_first_hit_at_rank_3(self) -> None:
        retrieved = ["doc2", "doc3", "doc1"]
        relevant = {"doc1"}
        assert mrr_at_k(retrieved, relevant, k=10) == pytest.approx(1.0 / 3.0)

    def test_first_hit_past_k_returns_zero(self) -> None:
        retrieved = ["doc2", "doc3", "doc1"]
        relevant = {"doc1"}
        assert mrr_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)

    def test_no_relevant_in_retrieved(self) -> None:
        retrieved = ["doc2", "doc3"]
        relevant = {"doc1"}
        assert mrr_at_k(retrieved, relevant, k=10) == pytest.approx(0.0)

    def test_multiple_relevant_items_uses_first(self) -> None:
        retrieved = ["doc_irrel", "doc_rel1", "doc_rel2"]
        relevant = {"doc_rel1", "doc_rel2"}
        assert mrr_at_k(retrieved, relevant, k=10) == pytest.approx(0.5)


class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        # Two relevant docs ranked at 1 and 2
        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc2"}
        assert ndcg_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)

    def test_imperfect_ranking(self) -> None:
        # Relevant doc at rank 2 instead of rank 1
        retrieved = ["doc_other", "doc_rel"]
        relevant = {"doc_rel"}
        # DCG@2 = 0 + 1 / log2(3) = 1 / 1.58496 = 0.63093
        # IDCG@2 = 1 / log2(2) = 1.0
        # nDCG = 1 / log2(3)
        expected = 1.0 / math.log2(3)
        assert ndcg_at_k(retrieved, relevant, k=2) == pytest.approx(expected)

    def test_zero_gain(self) -> None:
        retrieved = ["doc_other1", "doc_other2"]
        relevant = {"doc_rel"}
        assert ndcg_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self) -> None:
        assert ndcg_at_k(["doc1"], set(), k=5) == pytest.approx(0.0)

    def test_empty_retrieved_returns_zero(self) -> None:
        assert ndcg_at_k([], {"doc1"}, k=5) == pytest.approx(0.0)


class TestComputeRetrievalMetrics:
    def test_aggregates_multiple_queries(self) -> None:
        queries_results = [
            # query 1: perfect
            (["d1", "d2"], {"d1"}),
            # query 2: hit at rank 2
            (["d3", "d4"], {"d4"}),
        ]
        metrics = compute_retrieval_metrics(queries_results, k_recall=5, k_ndcg=5, k_mrr=10)
        # Query 1: recall@5=1.0, mrr@10=1.0, ndcg@5=1.0
        # Query 2: recall@5=1.0, mrr@10=0.5, ndcg@5=1/log2(3) = 0.630929
        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["mrr@10"] == pytest.approx((1.0 + 0.5) / 2.0)
        expected_ndcg = (1.0 + (1.0 / math.log2(3))) / 2.0
        assert metrics["ndcg@5"] == pytest.approx(expected_ndcg)
        assert metrics["query_count"] == 2.0
