"""Core domain models for documents and retrievable elements (SPEC §4.1).

SPEC §4.1 requires every element to carry: the document id and version, the
page, the element id, the equipment type, and the authorized access groups.
Those five are not optional metadata — the access filter, the citation contract
and the reranker dataset all depend on them, so they are required fields here
rather than something ingestion may or may not fill in.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maintidoc.schemas import Citation, ElementKind

#: Which search surfaced an element. Kept so RRF and the eval metrics can
#: attribute quality to the text index, the image index, or their fusion.
RetrievalChannel = Literal["text", "image", "fused"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DocumentRecord(_Model):
    """A document in the corpus, as tracked in PostgreSQL."""

    document_id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=500)
    #: One of the two equipment families in scope (SPEC §2).
    equipment_family: str = Field(min_length=1, max_length=128)
    #: Empty means nobody may read it — deny by default, see security/access.py.
    access_groups: frozenset[str] = frozenset()
    #: MinIO object key. The bytes never live in PostgreSQL or in git.
    object_key: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    language: str = Field(default="fr", max_length=16)
    page_count: int = Field(default=0, ge=0)
    #: True when the document was produced by scripts/generate_corpus.py.
    #: Synthetic provenance is recorded so no one mistakes it for a real manual.
    synthetic: bool = True
    license_note: str = Field(default="", max_length=500)
    ingested_at: datetime | None = None


class Element(_Model):
    """One retrievable unit: a paragraph, table, page, figure or schema."""

    element_id: str = Field(min_length=1, max_length=200)
    document_id: str = Field(min_length=1, max_length=200)
    document_version: str = Field(min_length=1, max_length=64)
    page: int = Field(ge=1)
    kind: ElementKind
    #: Extracted or OCR'd text. Empty for a purely visual element, whose meaning
    #: then comes from the image embedding and the VLM.
    text: str = ""
    equipment_family: str = Field(min_length=1, max_length=128)
    access_groups: frozenset[str] = frozenset()
    #: MinIO key of the rendered crop, when this element has a visual form.
    image_key: str | None = Field(default=None, max_length=500)
    #: Page-space bounding box (x0, y0, x1, y1) in PDF points.
    bbox: tuple[float, float, float, float] | None = None
    #: True when `text` came from OCR rather than the PDF text layer, so a lower
    #: trust can be attached to it downstream.
    from_ocr: bool = False

    @model_validator(mode="after")
    def _text_or_image_required(self) -> Element:
        """An element with neither text nor an image cannot be retrieved by any
        channel and should never have been produced."""
        if not self.text.strip() and not self.image_key:
            msg = f"element {self.element_id} has neither text nor an image"
            raise ValueError(msg)
        return self

    def to_citation(self, *, quote: str | None = None) -> Citation:
        """Build the citation this element supports.

        The citation can only ever point at an element that was actually
        retrieved, because it is constructed from the retrieved element itself
        (AGENTS.md NEVER #5).
        """
        return Citation(
            document_id=self.document_id,
            page=self.page,
            element_id=self.element_id,
            document_version=self.document_version,
            kind=self.kind,
            quote=quote,
        )


class RetrievedElement(_Model):
    """An `Element` plus how and why it was retrieved."""

    element: Element
    #: Similarity or reranker score. Only comparable within one channel.
    score: float
    channel: RetrievalChannel
    #: Rank within `channel`, 1-based, as returned by the index.
    rank: int = Field(ge=1)
    #: Reciprocal Rank Fusion contribution, when this element came from fusion.
    rrf_score: float | None = None

    @property
    def document_id(self) -> str:
        return self.element.document_id

    @property
    def page(self) -> int:
        return self.element.page

    @property
    def element_id(self) -> str:
        return self.element.element_id

    def to_citation(self, *, quote: str | None = None) -> Citation:
        return self.element.to_citation(quote=quote)


__all__ = [
    "DocumentRecord",
    "Element",
    "RetrievalChannel",
    "RetrievedElement",
]
