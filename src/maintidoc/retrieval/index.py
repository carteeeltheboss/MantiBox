"""Qdrant indexing and access-filtered search (SPEC §4.2).

Two properties this module exists to guarantee:

1. **The access filter is part of the query.** `search` builds the filter from
   `security.access.build_access_filter` and passes it to Qdrant, so the client
   never receives vectors the caller may not read. A post-filter would already
   have leaked them into the process.
2. **Point IDs are derived, not random.** `point_id` is a UUIDv5 over
   (document_id, element_id), so re-ingesting a document overwrites its points
   instead of duplicating them, and a retried ingestion is idempotent.

The payload carries the SPEC §4.1 metadata plus the embedder model and index
version, so a stale index can be detected rather than silently trusted.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from qdrant_client import QdrantClient, models

from maintidoc.domain import Element
from maintidoc.errors import ConfigurationError
from maintidoc.observability import get_logger
from maintidoc.schemas import ElementKind
from maintidoc.security.access import ACCESS_GROUPS_FIELD, Principal, build_access_filter

logger = get_logger(__name__)

#: Bump when the payload shape or vector semantics change, so an old collection
#: is recognisably old rather than quietly mismatched.
INDEX_VERSION = "v1"

_POINT_NAMESPACE = uuid.UUID("0f5c1d38-5f4b-4a3e-9c2f-1b7a5e9d0c14")

_DOCUMENT_FIELD = "document_id"
_KIND_FIELD = "kind"

#: The condition union `Filter.must` accepts. Spelled out because qdrant-client
#: annotates `Filter.must` itself more widely than its own constructor accepts
#: (the attribute alias also admits a `tuple[str, Any]` variant), so reading a
#: filter back and re-building one needs one narrowing cast.
Condition = (
    models.FieldCondition
    | models.IsEmptyCondition
    | models.IsNullCondition
    | models.HasIdCondition
    | models.HasVectorCondition
    | models.SliceCondition
    | models.NestedCondition
    | models.Filter
)


@dataclass(frozen=True, slots=True)
class IndexConfig:
    collection: str = "maintidoc_text"
    dimension: int = 1024
    distance: models.Distance = models.Distance.COSINE
    index_version: str = INDEX_VERSION


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One ranked result, reconstructed into a typed `Element`."""

    element: Element
    score: float
    rank: int

    @property
    def element_id(self) -> str:
        return self.element.element_id

    @property
    def document_id(self) -> str:
        return self.element.document_id


def point_id(document_id: str, element_id: str) -> str:
    """Stable UUID for a point, so upserts are idempotent across re-ingestion."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{document_id}:{element_id}"))


def element_payload(
    element: Element,
    *,
    embedder_model: str,
    index_version: str = INDEX_VERSION,
) -> dict[str, Any]:
    """Everything `element_from_payload` needs, plus provenance for staleness checks."""
    return {
        "element_id": element.element_id,
        "document_id": element.document_id,
        "document_version": element.document_version,
        "page": element.page,
        _KIND_FIELD: element.kind.value,
        "text": element.text,
        "equipment_family": element.equipment_family,
        ACCESS_GROUPS_FIELD: sorted(element.access_groups),
        "image_key": element.image_key,
        "from_ocr": element.from_ocr,
        "embedder_model": embedder_model,
        "index_version": index_version,
    }


def element_from_payload(payload: Mapping[str, Any]) -> Element:
    """Rebuild an `Element` from a stored payload.

    Raises:
        ConfigurationError: a required field is missing or unparsable, which
            means the collection was written by an incompatible version.
    """
    try:
        return Element(
            element_id=str(payload["element_id"]),
            document_id=str(payload["document_id"]),
            document_version=str(payload["document_version"]),
            page=int(payload["page"]),
            kind=ElementKind(str(payload[_KIND_FIELD])),
            text=str(payload.get("text", "")),
            equipment_family=str(payload["equipment_family"]),
            access_groups=frozenset(str(group) for group in payload.get(ACCESS_GROUPS_FIELD, [])),
            image_key=_optional_str(payload.get("image_key")),
            from_ocr=bool(payload.get("from_ocr", False)),
        )
    except (KeyError, ValueError) as exc:
        raise ConfigurationError(f"stored payload is not a valid element: {exc}") from exc


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


class QdrantIndex:
    """A Qdrant collection holding one vector space."""

    def __init__(self, client: QdrantClient, config: IndexConfig) -> None:
        self._client = client
        self._config = config

    @property
    def config(self) -> IndexConfig:
        return self._config

    # ------------------------------------------------------------------ lifecycle

    def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create the collection and its payload indexes if needed.

        Payload indexes on `access_groups` and `document_id` are not optional
        polish: the access filter runs on every search, and without an index
        Qdrant scans the payload.
        """
        collection = self._config.collection

        if recreate and self._client.collection_exists(collection):
            self._client.delete_collection(collection)
            logger.info("dropped collection %s for recreation", collection)

        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=self._config.dimension,
                    distance=self._config.distance,
                ),
            )
            logger.info(
                "created collection %s (dim=%d, version=%s)",
                collection,
                self._config.dimension,
                self._config.index_version,
            )

        for field_name in (ACCESS_GROUPS_FIELD, _DOCUMENT_FIELD):
            self._client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def count(self) -> int:
        if not self._client.collection_exists(self._config.collection):
            return 0
        return int(self._client.count(self._config.collection, exact=True).count)

    # --------------------------------------------------------------------- writes

    def upsert(
        self,
        elements: Sequence[Element],
        vectors: Sequence[Sequence[float]],
        *,
        embedder_model: str,
    ) -> int:
        """Index elements with their vectors. Returns the number written.

        Raises:
            ConfigurationError: the vector count or width does not match.
        """
        if len(elements) != len(vectors):
            raise ConfigurationError(f"{len(elements)} elements but {len(vectors)} vectors")
        if not elements:
            return 0

        points: list[models.PointStruct] = []
        for element, vector in zip(elements, vectors, strict=True):
            if len(vector) != self._config.dimension:
                raise ConfigurationError(
                    f"vector for {element.element_id} has {len(vector)} dimensions, "
                    f"collection expects {self._config.dimension}"
                )
            points.append(
                models.PointStruct(
                    id=point_id(element.document_id, element.element_id),
                    vector=[float(value) for value in vector],
                    payload=element_payload(
                        element,
                        embedder_model=embedder_model,
                        index_version=self._config.index_version,
                    ),
                )
            )

        self._client.upsert(self._config.collection, points=points, wait=True)
        return len(points)

    def delete_document(self, document_id: str) -> None:
        """Remove every point of one document, e.g. before re-ingesting it."""
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key=_DOCUMENT_FIELD,
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        )
        self._client.delete(self._config.collection, points_selector=selector, wait=True)

    # -------------------------------------------------------------------- reading

    def search(
        self,
        vector: Sequence[float],
        *,
        principal: Principal,
        limit: int = 20,
        kind: ElementKind | None = None,
    ) -> list[SearchHit]:
        """Search within what `principal` may read.

        Raises:
            AccessDeniedError: the principal has no groups. Deny-by-default: an
                empty `MatchAny` would otherwise match everything.
            ConfigurationError: the query vector has the wrong width.
        """
        if len(vector) != self._config.dimension:
            raise ConfigurationError(
                f"query vector has {len(vector)} dimensions, collection expects "
                f"{self._config.dimension}"
            )

        access_filter = build_access_filter(principal)
        if not isinstance(access_filter, models.Filter):  # pragma: no cover - contract guard
            raise ConfigurationError("access filter is not a Qdrant Filter")

        # The cast only narrows qdrant-client's wider attribute alias; runtime shapes
        # are identical. Conditions are never dropped here - discarding an access
        # condition to satisfy a type checker would weaken the filter.
        conditions = cast("list[Condition]", list(access_filter.must or []))
        if kind is not None:
            conditions.append(
                models.FieldCondition(key=_KIND_FIELD, match=models.MatchValue(value=kind.value))
            )

        response = self._client.query_points(
            self._config.collection,
            query=[float(value) for value in vector],
            query_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=True,
        )

        hits: list[SearchHit] = []
        for rank, scored in enumerate(response.points, start=1):
            payload = scored.payload or {}
            hits.append(
                SearchHit(
                    element=element_from_payload(payload),
                    score=float(scored.score),
                    rank=rank,
                )
            )
        return hits


def make_client(*, host: str, port: int, api_key: str | None = None) -> QdrantClient:
    """Build a client. Kept here so callers do not import qdrant-client directly."""
    return QdrantClient(host=host, port=port, api_key=api_key, timeout=10)


__all__ = [
    "INDEX_VERSION",
    "IndexConfig",
    "QdrantIndex",
    "SearchHit",
    "element_from_payload",
    "element_payload",
    "make_client",
    "point_id",
]
