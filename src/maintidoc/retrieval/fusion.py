"""Reciprocal Rank Fusion and Hybrid Retriever (SPEC §4.2).

Combines text search (BGE-M3 1024-d) and visual search (CLIP 512-d) over the
authorized corpus, merging ranked candidates via Reciprocal Rank Fusion (RRF):

    RRF(d) = sum_{c in Channels} 1 / (k + rank_c(d))

where k is a smoothing constant defaulting to 60 (RetrievalSettings.rrf_k).
Fused results are then passed through the Reranker seam (SPEC §5).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from maintidoc.domain import RetrievalChannel, RetrievedElement
from maintidoc.errors import ConfigurationError
from maintidoc.observability import get_logger
from maintidoc.retrieval.embeddings import Embedder
from maintidoc.retrieval.images import ImageEmbedder
from maintidoc.retrieval.index import QdrantIndex, SearchHit
from maintidoc.retrieval.reranker import NoReranker, Reranker
from maintidoc.security.access import Principal

logger = get_logger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    text_hits: Sequence[SearchHit | RetrievedElement],
    image_hits: Sequence[SearchHit | RetrievedElement],
    *,
    k: int = DEFAULT_RRF_K,
    top_n: int | None = None,
) -> list[RetrievedElement]:
    """Fuse ranked lists from text and image channels using Reciprocal Rank Fusion.

    Parameters:
        text_hits: Ranked results from the text collection.
        image_hits: Ranked results from the image collection.
        k: Smoothing constant, defaults to 60. Must be >= 1.
        top_n: Optional cut-off for the number of returned results.

    Returns:
        List of `RetrievedElement` sorted by RRF score descending.
    """
    if k < 1:
        msg = f"k must be >= 1, got {k}"
        raise ValueError(msg)

    # Key: (document_id, element_id) -> metadata dict
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    def _process_hits(hits: Sequence[SearchHit | RetrievedElement], channel_name: str) -> None:
        for hit in hits:
            elem = hit.element
            key = (elem.document_id, elem.element_id)
            rank = hit.rank
            rrf_contrib = 1.0 / (k + rank)

            if key in merged:
                merged[key]["rrf_score"] += rrf_contrib
                merged[key]["channels"].add(channel_name)
                # Keep figure image_key if available
                if elem.image_key and not merged[key]["element"].image_key:
                    merged[key]["element"] = elem
            else:
                merged[key] = {
                    "element": elem,
                    "rrf_score": rrf_contrib,
                    "channels": {channel_name},
                }

    _process_hits(text_hits, "text")
    _process_hits(image_hits, "image")

    # Deterministic sorting: by rrf_score descending, then doc_id, elem_id
    sorted_items = sorted(
        merged.values(),
        key=lambda entry: (
            entry["rrf_score"],
            entry["element"].document_id,
            entry["element"].element_id,
        ),
        reverse=True,
    )

    fused_elements: list[RetrievedElement] = []
    for rank_0, entry in enumerate(sorted_items):
        channels: set[str] = entry["channels"]
        channel: RetrievalChannel
        if len(channels) > 1:
            channel = "fused"
        elif "text" in channels:
            channel = "text"
        else:
            channel = "image"

        score = float(entry["rrf_score"])
        fused_elements.append(
            RetrievedElement(
                element=entry["element"],
                score=score,
                channel=channel,
                rank=rank_0 + 1,
                rrf_score=score,
            )
        )

    if top_n is not None and top_n > 0:
        fused_elements = fused_elements[:top_n]

    return fused_elements


class HybridRetriever:
    """Coordinates multimodal retrieval: text search, visual search, RRF fusion, and reranking."""

    def __init__(
        self,
        *,
        text_index: QdrantIndex,
        image_index: QdrantIndex,
        text_embedder: Embedder,
        image_embedder: ImageEmbedder,
        reranker: Reranker | None = None,
    ) -> None:
        self.text_index = text_index
        self.image_index = image_index
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.reranker = reranker or NoReranker()

    def search(
        self,
        query: str,
        *,
        principal: Principal,
        query_image: bytes | None = None,
        top_k_text: int = 20,
        top_k_image: int = 20,
        top_n: int = 8,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[RetrievedElement]:
        """Search text and visual indexes with access filtering, fuse with RRF, and rerank.

        Access control is enforced *inside* Qdrant queries via `principal`.
        """
        # 1. Text search
        text_vector = self.text_embedder.embed_query(query)
        text_hits = self.text_index.search(
            text_vector,
            principal=principal,
            limit=top_k_text,
        )

        # 2. Visual search
        if query_image is not None:
            image_vectors = self.image_embedder.embed_images([query_image])
            if not image_vectors:
                msg = "image embedder returned no vectors for query image"
                raise ConfigurationError(msg)
            visual_vector = image_vectors[0]
        else:
            visual_vector = self.image_embedder.embed_text(query)

        image_hits = self.image_index.search(
            visual_vector,
            principal=principal,
            limit=top_k_image,
        )

        # 3. Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            text_hits=text_hits,
            image_hits=image_hits,
            k=rrf_k,
            top_n=None,
        )

        # 4. Reranking
        return self.reranker.rerank(query, fused, top_n=top_n)


__all__ = [
    "DEFAULT_RRF_K",
    "HybridRetriever",
    "reciprocal_rank_fusion",
]
