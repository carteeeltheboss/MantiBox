"""Rendering retrieved elements into the prompt's `<sources>` block.

This module owns one format used from two directions:

* the workflow renders retrieved elements with `render_sources()` before calling
  the `answer` prompt;
* the deterministic stub provider parses the same block with
  `parse_source_refs()` so that it can only ever cite an element that was
  actually retrieved.

Keeping both directions in one place is what stops them drifting apart. Without
it, a stub (or a model) could emit a citation for a document that was never in
the context, which AGENTS.md NEVER #5 forbids.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from maintidoc.domain import RetrievedElement

#: Header line preceding each element's text. Extra trailing fields are allowed
#: so new metadata can be added without breaking the parser.
_SOURCE_HEADER = re.compile(
    r"^\[(?P<index>\d+)\]\s+"
    r"document_id=(?P<document_id>\S+)\s+"
    r"version=(?P<version>\S+)\s+"
    r"page=(?P<page>\d+)\s+"
    r"element_id=(?P<element_id>\S+)\s+"
    r"kind=(?P<kind>\S+)",
    re.MULTILINE,
)

_ELEMENT_SEPARATOR = "\n---\n"

#: Placeholder for an element that has a visual form but no extracted text. The
#: VLM sees the image itself; the text channel has nothing to offer.
_NO_TEXT = "(élément visuel sans texte extrait)"


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A parsed `<sources>` header: enough to build a valid Citation."""

    index: int
    document_id: str
    document_version: str
    page: int
    element_id: str
    kind: str


def render_sources(
    elements: Sequence[RetrievedElement],
    *,
    max_chars_per_element: int = 900,
) -> str:
    """Render retrieved elements as the text handed to the `answer` prompt.

    Text is truncated per element so a single long page cannot consume the whole
    context budget (SPEC §9 imposes limits on context size).
    """
    if not elements:
        return "(aucune source récupérée)"

    blocks: list[str] = []
    for index, retrieved in enumerate(elements, start=1):
        element = retrieved.element
        header = (
            f"[{index}] document_id={element.document_id} "
            f"version={element.document_version} page={element.page} "
            f"element_id={element.element_id} kind={element.kind.value} "
            f"channel={retrieved.channel}"
        )
        text = element.text.strip() or _NO_TEXT
        if len(text) > max_chars_per_element:
            text = f"{text[:max_chars_per_element].rstrip()} […tronqué]"
        if element.from_ocr:
            header += " ocr=true"
        blocks.append(f"{header}\n{text}")
    return _ELEMENT_SEPARATOR.join(blocks)


def parse_source_refs(text: str) -> tuple[SourceRef, ...]:
    """Parse every `<sources>` header out of a rendered block."""
    refs: list[SourceRef] = []
    for match in _SOURCE_HEADER.finditer(text):
        refs.append(
            SourceRef(
                index=int(match.group("index")),
                document_id=match.group("document_id"),
                document_version=match.group("version"),
                page=int(match.group("page")),
                element_id=match.group("element_id"),
                kind=match.group("kind"),
            )
        )
    return tuple(refs)


def extract_section(text: str, tag: str) -> str:
    """Return the content between `<tag>` and `</tag>`, or "" if absent.

    Used to pull the sources block back out of a fully rendered prompt.
    """
    pattern = re.compile(rf"<{re.escape(tag)}>\n?(.*?)\n?</{re.escape(tag)}>", re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else ""


__all__ = [
    "SourceRef",
    "extract_section",
    "parse_source_refs",
    "render_sources",
]
