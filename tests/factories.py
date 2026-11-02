"""Shared builders for domain objects used across tests.

Kept out of conftest so they can be imported explicitly by name, and out of the
test modules so two files cannot drift into different notions of "an element".
"""

from __future__ import annotations

from maintidoc.domain import DocumentRecord, Element, RetrievedElement
from maintidoc.schemas import ElementKind

DEFAULT_TEXT = "Voyant orange : alarme de température, seuil 85 °C."


def make_document(
    *,
    document_id: str = "manuel_pompe_a",
    version: str = "2024-03",
    title: str = "Manuel Pompe A",
    equipment_family: str = "pompes",
    access_groups: frozenset[str] = frozenset({"maintenance"}),
    object_key: str = "documents/manuel_pompe_a.pdf",
    sha256: str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        version=version,
        title=title,
        equipment_family=equipment_family,
        access_groups=access_groups,
        object_key=object_key,
        sha256=sha256,
    )


def make_element(
    *,
    document_id: str = "manuel_pompe_a",
    version: str = "2024-03",
    page: int = 18,
    element_id: str = "figure_18_2",
    kind: ElementKind = ElementKind.FIGURE,
    text: str = DEFAULT_TEXT,
    equipment_family: str = "pompes",
    access_groups: frozenset[str] = frozenset({"maintenance"}),
    from_ocr: bool = False,
    image_key: str | None = None,
) -> Element:
    return Element(
        element_id=element_id,
        document_id=document_id,
        document_version=version,
        page=page,
        kind=kind,
        text=text,
        equipment_family=equipment_family,
        access_groups=access_groups,
        image_key=image_key,
        from_ocr=from_ocr,
    )


def make_retrieved(
    element: Element,
    *,
    rank: int = 1,
    score: float = 0.9,
    channel: str = "text",
) -> RetrievedElement:
    return RetrievedElement(
        element=element,
        score=score,
        channel=channel,  # type: ignore[arg-type]
        rank=rank,
    )
