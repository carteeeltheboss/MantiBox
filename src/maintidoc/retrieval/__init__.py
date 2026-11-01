"""Retrieval: embeddings, indexing, fusion, reranking and prompt rendering."""

from __future__ import annotations

from maintidoc.retrieval.render import (
    SourceRef,
    extract_section,
    parse_source_refs,
    render_sources,
)

__all__ = ["SourceRef", "extract_section", "parse_source_refs", "render_sources"]
