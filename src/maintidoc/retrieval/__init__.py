"""Retrieval: embeddings, indexing, fusion, reranking and prompt rendering."""

from __future__ import annotations

from maintidoc.retrieval.embeddings import (
    BGE_M3_DIMENSION,
    BgeM3Embedder,
    Embedder,
    HashingEmbedder,
    make_embedder,
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
from maintidoc.retrieval.render import (
    SourceRef,
    extract_section,
    parse_source_refs,
    render_sources,
)

__all__ = [
    "BGE_M3_DIMENSION",
    "INDEX_VERSION",
    "BgeM3Embedder",
    "Embedder",
    "HashingEmbedder",
    "IndexConfig",
    "QdrantIndex",
    "SearchHit",
    "SourceRef",
    "element_from_payload",
    "element_payload",
    "extract_section",
    "make_client",
    "make_embedder",
    "parse_source_refs",
    "point_id",
    "render_sources",
]
