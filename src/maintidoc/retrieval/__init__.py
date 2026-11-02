"""Retrieval: embeddings, indexing, fusion, reranking and prompt rendering."""

from __future__ import annotations

from maintidoc.retrieval.embeddings import (
    BGE_M3_DIMENSION,
    BgeM3Embedder,
    Embedder,
    HashingEmbedder,
    make_embedder,
)
from maintidoc.retrieval.fusion import (
    DEFAULT_RRF_K,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from maintidoc.retrieval.images import (
    CLIP_DIMENSION,
    ClipImageEmbedder,
    HashingImageEmbedder,
    ImageEmbedder,
    make_image_embedder,
)
from maintidoc.retrieval.index import (
    INDEX_VERSION,
    IndexConfig,
    QdrantIndex,
    SearchHit,
    element_from_payload,
    element_payload,
    make_client,
    point_id,
)
from maintidoc.retrieval.metrics import (
    compute_retrieval_metrics,
    dcg_at_k,
    idcg_at_k,
    log_metrics_to_mlflow,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from maintidoc.retrieval.render import (
    SourceRef,
    extract_section,
    parse_source_refs,
    render_sources,
)
from maintidoc.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    BgeReranker,
    DeterministicReranker,
    NoReranker,
    Reranker,
    make_reranker,
)

__all__ = [
    "BGE_M3_DIMENSION",
    "CLIP_DIMENSION",
    "DEFAULT_RERANKER_MODEL",
    "DEFAULT_RRF_K",
    "INDEX_VERSION",
    "BgeM3Embedder",
    "BgeReranker",
    "ClipImageEmbedder",
    "DeterministicReranker",
    "Embedder",
    "HashingEmbedder",
    "HashingImageEmbedder",
    "HybridRetriever",
    "ImageEmbedder",
    "IndexConfig",
    "NoReranker",
    "QdrantIndex",
    "Reranker",
    "SearchHit",
    "SourceRef",
    "compute_retrieval_metrics",
    "dcg_at_k",
    "element_from_payload",
    "element_payload",
    "extract_section",
    "idcg_at_k",
    "log_metrics_to_mlflow",
    "make_client",
    "make_embedder",
    "make_image_embedder",
    "make_reranker",
    "mrr_at_k",
    "ndcg_at_k",
    "parse_source_refs",
    "point_id",
    "recall_at_k",
    "reciprocal_rank_fusion",
    "render_sources",
]
