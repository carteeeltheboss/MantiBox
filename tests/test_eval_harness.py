"""Tests for the eval harness and seed benchmark (SPEC §5, §10)."""

from __future__ import annotations

from eval.benchmark_data import SEED_QUERIES
from eval.run import evaluate_system, load_corpus_elements

from maintidoc.retrieval.embeddings import HashingEmbedder
from maintidoc.retrieval.images import HashingImageEmbedder
from maintidoc.retrieval.reranker import NoReranker


def test_seed_queries_structure() -> None:
    assert len(SEED_QUERIES) >= 5
    for q in SEED_QUERIES:
        assert q.query.strip()
        assert len(q.relevant_targets) > 0
        assert len(q.principal.groups) > 0


def test_load_corpus_elements() -> None:
    elements, figure_images = load_corpus_elements()
    assert len(elements) > 100
    assert len(figure_images) > 0


def test_evaluate_system_with_hash_doubles() -> None:
    elements, figure_images = load_corpus_elements()
    metrics = evaluate_system(
        name="test_hash_run",
        text_embedder=HashingEmbedder(dimension=32),
        image_embedder=HashingImageEmbedder(dimension=32),
        reranker=NoReranker(),
        elements=elements,
        figure_images=figure_images,
        queries=SEED_QUERIES[:2],
    )
    assert "recall@5" in metrics
    assert "ndcg@5" in metrics
    assert "mrr@10" in metrics
    assert metrics["query_count"] == 2.0
