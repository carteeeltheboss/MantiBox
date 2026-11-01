"""PDF extraction into `Element`s (SPEC §4.1).

What SPEC §4.1 requires per element: document id and version, page, element id,
equipment type, and the authorized access groups. All five come from the
`DocumentRecord` plus the page geometry, so they can never be forgotten.

Design notes, each the result of probing the real PyMuPDF 2.0 API:

* `insert_textbox`-style assumptions do not apply to reading, but `find_tables()`
  does **not** detect the vector-drawn tables in this corpus, so tables are
  reconstructed from their caption plus the short text blocks beneath it. The
  acceptance criterion is that table content becomes *retrievable text*, which
  this achieves without depending on table detection.
* Figure regions are found by taking the drawings above a figure caption. The
  corpus figures are vector drawings, not embedded images, so `get_images()`
  alone would find nothing.
* A page whose text layer is empty is treated as scanned: OCR if an engine is
  available, and a stored page render either way, so the visual channel still
  works when OCR is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pymupdf

from maintidoc.domain import DocumentRecord, Element
from maintidoc.errors import IngestionError
from maintidoc.ingestion.ocr import OcrEngine, OcrUnavailableError, TesseractOcr
from maintidoc.observability import content_fingerprint, get_logger
from maintidoc.schemas import ElementKind
from maintidoc.storage.objects import ObjectStore, ObjectStoreError

logger = get_logger(__name__)

#: Caption prefixes used to locate figures and tables. Corpus-specific by
#: necessity; a real corpus would carry these in the document manifest.
DEFAULT_FIGURE_CAPTIONS = ("Figure -", "Figure :", "Fig.")
DEFAULT_TABLE_CAPTIONS = ("Tableau -", "Tableau :")

#: A block longer than this is treated as prose, which terminates a table region.
_PROSE_LENGTH = 80

_BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Tuning for one extraction run."""

    #: Passages are capped so one long page cannot fill the retrieval context
    #: (SPEC §9 limits context size).
    max_chars: int = 1600
    #: Shorter blocks are usually page furniture, not content.
    min_block_chars: int = 15
    header_band: float = 55.0
    #: The corpus renders its page number at y=790..800 on an 842pt page, so a
    #: 45pt band would leave it in as a paragraph. 60pt clears it.
    footer_band: float = 60.0

    ocr_enabled: bool = True
    ocr_dpi: int = 200
    #: A page with less than this much text is treated as scanned.
    ocr_page_text_threshold: int = 40

    figure_dpi: int = 150
    figure_caption_prefixes: tuple[str, ...] = DEFAULT_FIGURE_CAPTIONS
    table_caption_prefixes: tuple[str, ...] = DEFAULT_TABLE_CAPTIONS


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Elements plus what had to be skipped, so nothing fails silently."""

    elements: tuple[Element, ...]
    page_count: int
    ocr_pages: tuple[int, ...] = ()
    warnings: tuple[str, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class _Block:
    """A text block with its page geometry, in PDF points."""

    bbox: _BBox
    text: str

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


def _content_blocks(page: pymupdf.Page, config: ExtractionConfig) -> list[_Block]:
    """Text blocks minus header/footer furniture, in reading order."""
    page_rect = page.rect
    height = float(page_rect.height)
    blocks: list[_Block] = []

    for raw in page.get_text("blocks"):
        x0, y0, x1, y1, text, _number, block_type = raw
        if int(block_type) != 0:
            continue
        cleaned = " ".join(str(text).split())
        if not cleaned:
            continue
        if float(y1) <= config.header_band or float(y0) >= height - config.footer_band:
            continue
        blocks.append(
            _Block(
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                text=cleaned,
            )
        )
    blocks.sort(key=lambda block: (block.y0, block.bbox[0]))
    return blocks


def _is_caption(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def _chunks(text: str, max_chars: int) -> list[str]:
    """Split on word boundaries, never mid-word, never above `max_chars`."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in text.split():
        addition = len(word) + (1 if current else 0)
        if length + addition > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += addition
    if current:
        chunks.append(" ".join(current))
    return chunks


def _group_rows(blocks: list[_Block]) -> list[list[_Block]]:
    """Group blocks into visual rows by vertical position."""
    rows: list[list[_Block]] = []
    for block in sorted(blocks, key=lambda b: (round(b.y0, 1), b.bbox[0])):
        if rows and abs(rows[-1][0].y0 - block.y0) <= 3.0:
            rows[-1].append(block)
        else:
            rows.append([block])
    for row in rows:
        row.sort(key=lambda b: b.bbox[0])
    return rows


def _table_text(blocks: list[_Block]) -> str:
    rows = _group_rows(blocks)
    return "\n".join(" | ".join(block.text for block in row) for row in rows)


def _figure_clip(page: pymupdf.Page, caption: _Block, page_width: float) -> _BBox | None:
    """Region containing the drawings that sit above a figure caption."""
    above = [
        drawing for drawing in page.get_drawings() if float(drawing["rect"].y1) <= caption.y0 + 1.0
    ]
    if not above:
        return None
    top = min(float(drawing["rect"].y0) for drawing in above)
    return (0.0, max(top - 4.0, 0.0), page_width, caption.y0 - 2.0)


def _render_png(page: pymupdf.Page, clip: _BBox, dpi: int) -> bytes:
    rect = pymupdf.Rect(*clip)
    pixmap = page.get_pixmap(dpi=dpi, clip=rect)
    data: bytes = pixmap.tobytes("png")
    return data


def _store_image(store: ObjectStore, key: str, data: bytes) -> str:
    try:
        return store.put(key, data, content_type="image/png")
    except ObjectStoreError as exc:
        raise IngestionError(f"could not store rendered image: {exc}") from exc


def extract_pdf(
    *,
    data: bytes,
    document: DocumentRecord,
    store: ObjectStore,
    config: ExtractionConfig | None = None,
    ocr: OcrEngine | None = None,
) -> ExtractionResult:
    """Extract every retrievable element from one PDF.

    Raises:
        IngestionError: the bytes are not a readable PDF.
    """
    cfg = config or ExtractionConfig()
    engine = ocr if ocr is not None else TesseractOcr()

    if not data:
        raise IngestionError("document bytes are empty")

    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError(f"PDF could not be opened: {exc.__class__.__name__}") from exc

    elements: list[Element] = []
    warnings: list[str] = []
    ocr_pages: list[int] = []

    try:
        page_count = int(pdf.page_count)
        for page_number in range(1, page_count + 1):
            page = pdf.load_page(page_number - 1)
            counters = {ElementKind.PARAGRAPH: 0, ElementKind.TABLE: 0, ElementKind.FIGURE: 0}
            page_width = float(page.rect.width)
            blocks = _content_blocks(page, cfg)

            consumed: set[int] = set()
            page_text_length = sum(len(block.text) for block in blocks)

            # --- tables: caption followed by short blocks, terminated by prose
            for index, block in enumerate(blocks):
                if not _is_caption(block.text, cfg.table_caption_prefixes):
                    continue
                cells: list[_Block] = []
                for follower in blocks[index + 1 :]:
                    if len(follower.text) >= _PROSE_LENGTH:
                        break
                    if _is_caption(follower.text, cfg.figure_caption_prefixes):
                        break
                    cells.append(follower)
                    consumed.add(id(follower))
                consumed.add(id(block))
                body = _table_text(cells)
                text = block.text if not body else f"{block.text}\n{body}"
                counters[ElementKind.TABLE] += 1
                elements.append(
                    _element(
                        document=document,
                        page_number=page_number,
                        kind=ElementKind.TABLE,
                        index=counters[ElementKind.TABLE],
                        text=text,
                        bbox=block.bbox,
                    )
                )

            # --- figures: caption plus the drawings above it
            for block in blocks:
                if not _is_caption(block.text, cfg.figure_caption_prefixes):
                    continue
                consumed.add(id(block))
                clip = _figure_clip(page, block, page_width)
                if clip is None:
                    warnings.append(f"page {page_number}: figure caption with no drawings above it")
                    continue
                labels = [
                    other.text
                    for other in blocks
                    if other is not block
                    and clip[1] <= other.y0 <= clip[3]
                    and len(other.text) <= 60
                ]
                counters[ElementKind.FIGURE] += 1
                prefix = _image_prefix(document)
                figure_number = counters[ElementKind.FIGURE]
                image_key = f"{prefix}_p{page_number}_figure_{figure_number}.png"
                _store_image(store, image_key, _render_png(page, clip, cfg.figure_dpi))
                text = " ".join([block.text, *labels]).strip()
                elements.append(
                    _element(
                        document=document,
                        page_number=page_number,
                        kind=ElementKind.FIGURE,
                        index=counters[ElementKind.FIGURE],
                        text=text,
                        bbox=block.bbox,
                        image_key=image_key,
                    )
                )

            # --- remaining blocks are paragraphs
            for block in blocks:
                if id(block) in consumed:
                    continue
                for chunk_index, chunk in enumerate(_chunks(block.text, cfg.max_chars)):
                    counters[ElementKind.PARAGRAPH] += 1
                    suffix = "" if chunk_index == 0 else "b"
                    elements.append(
                        _element(
                            document=document,
                            page_number=page_number,
                            kind=ElementKind.PARAGRAPH,
                            index=counters[ElementKind.PARAGRAPH],
                            text=chunk,
                            bbox=block.bbox,
                            suffix=suffix,
                        )
                    )

            # --- no usable text layer: treat as scanned
            if page_text_length < cfg.ocr_page_text_threshold:
                elements.extend(
                    _ingest_scanned_page(
                        page=page,
                        page_number=page_number,
                        page_width=page_width,
                        document=document,
                        store=store,
                        engine=engine,
                        config=cfg,
                        ocr_pages=ocr_pages,
                        warnings=warnings,
                        counter_start=len(elements),
                    )
                )
    finally:
        pdf.close()

    logger.info(
        "extracted document elements",
        extra={
            "document_id": document.document_id,
            "pages": page_count,
            "elements": len(elements),
            "ocr_pages": len(ocr_pages),
            "warnings": len(warnings),
            "fingerprint": content_fingerprint(data),
        },
    )

    return ExtractionResult(
        elements=tuple(elements),
        page_count=page_count,
        ocr_pages=tuple(ocr_pages),
        warnings=tuple(warnings),
    )


def _image_prefix(document: DocumentRecord) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", document.document_id.lower())


def _element(
    *,
    document: DocumentRecord,
    page_number: int,
    kind: ElementKind,
    index: int,
    text: str,
    bbox: _BBox,
    image_key: str | None = None,
    from_ocr: bool = False,
    suffix: str = "",
) -> Element:
    return Element(
        element_id=f"{kind.value}_{page_number}_{index}{suffix}",
        document_id=document.document_id,
        document_version=document.version,
        page=page_number,
        kind=kind,
        text=text,
        equipment_family=document.equipment_family,
        access_groups=document.access_groups,
        image_key=image_key,
        bbox=bbox,
        from_ocr=from_ocr,
    )


def _ingest_scanned_page(
    *,
    page: pymupdf.Page,
    page_number: int,
    page_width: float,
    document: DocumentRecord,
    store: ObjectStore,
    engine: OcrEngine,
    config: ExtractionConfig,
    ocr_pages: list[int],
    warnings: list[str],
    counter_start: int,
) -> list[Element]:
    """OCR a page with no text layer, and always keep its render.

    The render is kept even when OCR is unavailable so the image channel can
    still retrieve the page; the warning records that its text is missing.
    """
    page_height = float(page.rect.height)
    clip = (0.0, 0.0, page_width, page_height)
    image_key = f"{_image_prefix(document)}_p{page_number}_page.png"
    _store_image(store, image_key, _render_png(page, clip, config.figure_dpi))

    if not config.ocr_enabled:
        warnings.append(f"page {page_number}: no text layer and OCR is disabled")
        return [
            _element(
                document=document,
                page_number=page_number,
                kind=ElementKind.PAGE,
                index=1,
                text="",
                bbox=clip,
                image_key=image_key,
            )
        ]

    if not engine.is_available():
        warnings.append(
            f"page {page_number}: no text layer and tesseract is unavailable; "
            "page stored as an image only"
        )
        return [
            _element(
                document=document,
                page_number=page_number,
                kind=ElementKind.PAGE,
                index=1,
                text="",
                bbox=clip,
                image_key=image_key,
            )
        ]

    try:
        text = engine.text_from_png(_render_png(page, clip, config.ocr_dpi))
    except OcrUnavailableError as exc:
        warnings.append(f"page {page_number}: OCR unavailable ({exc.code})")
        text = ""

    ocr_pages.append(page_number)

    produced: list[Element] = []
    if text.strip():
        normalized = " ".join(text.split())
        for chunk_index, chunk in enumerate(_chunks(normalized, config.max_chars)):
            produced.append(
                _element(
                    document=document,
                    page_number=page_number,
                    kind=ElementKind.PARAGRAPH,
                    index=counter_start + chunk_index + 1,
                    text=chunk,
                    bbox=clip,
                    image_key=image_key,
                    from_ocr=True,
                )
            )
    else:
        warnings.append(f"page {page_number}: OCR produced no text")
        produced.append(
            _element(
                document=document,
                page_number=page_number,
                kind=ElementKind.PAGE,
                index=1,
                text="",
                bbox=clip,
                image_key=image_key,
            )
        )
    return produced


__all__ = [
    "ExtractionConfig",
    "ExtractionResult",
    "extract_pdf",
]
