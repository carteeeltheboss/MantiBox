"""Tests for performance benchmark: median and p95 latency, memory, GPU time (SPEC §10)."""

from __future__ import annotations

from eval.benchmark_performance import (
    PerformanceReport,
    measure_pipeline_performance,
)
from qdrant_client import QdrantClient

from maintidoc.agent.pipeline import AnswerPipeline
from maintidoc.retrieval.embeddings import HashingEmbedder
from maintidoc.retrieval.fusion import HybridRetriever
from maintidoc.retrieval.images import HashingImageEmbedder
from maintidoc.retrieval.index import IndexConfig, QdrantIndex
from maintidoc.retrieval.reranker import NoReranker
from maintidoc.vlm.stub import StubPolicy, StubVlmProvider


def test_measure_pipeline_performance() -> None:
    client = QdrantClient(":memory:")
    t_cfg = IndexConfig(collection="perf_text", dimension=32)
    i_cfg = IndexConfig(collection="perf_img", dimension=32)
    t_idx = QdrantIndex(client, t_cfg)
    i_idx = QdrantIndex(client, i_cfg)
    t_idx.ensure_collection()
    i_idx.ensure_collection()

    retriever = HybridRetriever(
        text_index=t_idx,
        image_index=i_idx,
        text_embedder=HashingEmbedder(dimension=32),
        image_embedder=HashingImageEmbedder(dimension=32),
        reranker=NoReranker(),
    )
    vlm = StubVlmProvider(policy=StubPolicy.AUTO)
    pipeline = AnswerPipeline(retriever=retriever, vlm_provider=vlm)

    queries = [
        "Quelle est la vitesse de rotation ?",
        "Pression maximale au refoulement",
        "Signification du voyant orange",
    ]

    report = measure_pipeline_performance(pipeline, queries=queries, repetitions=3)
    assert isinstance(report, PerformanceReport)
    assert report.total_requests == 9
    assert report.latency.median_ms > 0
    assert report.latency.p95_ms >= report.latency.median_ms
    assert report.memory_rss_mb > 0
    # On stub pipeline and small queries, p95 must comfortably meet SPEC §10 30s target (30,000 ms)
    assert report.latency.p95_ms < 30000.0
    assert report.spec_target_met is True
