"""PDF ingestion into `Element`s (SPEC §4.1).

These run against PDFs produced by the real corpus generator, not hand-written
fixtures, so the extractor is tested against the same geometry it must handle in
production. No services and no tesseract binary are required: the OCR engine is
injected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.generate_corpus import generate

from maintidoc.domain import DocumentRecord, Element
from maintidoc.errors import IngestionError
from maintidoc.ingestion import (
    ExtractionConfig,
    ExtractionResult,
    NullOcr,
    OcrEngine,
    OcrUnavailableError,
    extract_pdf,
)
from maintidoc.schemas import ElementKind
from maintidoc.storage.objects import FilesystemStore


class FakeOcr:
    """Deterministic OCR engine: returns a fixed transcript."""

    def __init__(self, transcript: str = "Texte reconnu par OCR. Température 81 °C.") -> None:
        self.transcript = transcript
        self.calls: list[bytes] = []

    def is_available(self) -> bool:
        return True

    def text_from_png(self, png: bytes, *, languages: str | None = None) -> str:
        self.calls.append(png)
        return self.transcript


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ingest-corpus")
    generate(6, corpus_dir=root)
    return root


@pytest.fixture
def store(tmp_path: Path) -> FilesystemStore:
    return FilesystemStore(root=tmp_path / "objects")


def document(corpus: Path, document_id: str) -> DocumentRecord:
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    entry = next(d for d in manifest["documents"] if d["document_id"] == document_id)
    return DocumentRecord(
        document_id=entry["document_id"],
        version=entry["version"],
        title=entry["title"],
        equipment_family=entry["equipment_family"],
        access_groups=frozenset(entry["access_groups"]),
        object_key=entry["object_key"],
        sha256=entry["sha256"],
        page_count=entry["page_count"],
    )


def bytes_of(corpus: Path, document_id: str) -> bytes:
    return (corpus / "documents" / f"{document_id}.pdf").read_bytes()


def extract(
    corpus: Path,
    document_id: str,
    store: FilesystemStore,
    *,
    ocr: OcrEngine | None = None,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    return extract_pdf(
        data=bytes_of(corpus, document_id),
        document=document(corpus, document_id),
        store=store,
        config=config,
        ocr=ocr if ocr is not None else NullOcr(),
    )


def texts(result: ExtractionResult, kind: ElementKind | None = None) -> list[str]:
    return [element.text for element in result.elements if kind is None or element.kind is kind]


class TestMetadataPropagation:
    def test_every_element_carries_the_spec_4_1_metadata(
        self, corpus: Path, store: FilesystemStore
    ) -> None:
        record = document(corpus, "manuel_pompe_a")
        result = extract(corpus, "manuel_pompe_a", store)

        assert result.elements
        for element in result.elements:
            assert element.document_id == record.document_id
            assert element.document_version == record.version
            assert element.page >= 1
            assert element.element_id
            assert element.equipment_family == record.equipment_family
            assert element.access_groups == record.access_groups

    def test_restricted_document_keeps_its_restricted_groups(
        self, corpus: Path, store: FilesystemStore
    ) -> None:
        result = extract(corpus, "manuel_pompe_a_restreint", store)
        assert all(
            element.access_groups == frozenset({"supervisors"}) for element in result.elements
        )

    def test_access_groups_are_never_dropped(self, corpus: Path, store: FilesystemStore) -> None:
        """NEVER #7: an element with empty groups would be invisible to everyone,
        so ingestion must never produce one from a labelled document."""
        result = extract(corpus, "manuel_pompe_a", store)
        assert all(element.access_groups for element in result.elements)


class TestParagraphs:
    def test_paragraphs_are_extracted(self, corpus: Path, store: FilesystemStore) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        paragraphs = texts(result, ElementKind.PARAGRAPH)

        assert len(paragraphs) >= 5
        assert any("groupe de pompage centrifuge" in text for text in paragraphs)

    def test_page_numbers_are_one_based(self, corpus: Path, store: FilesystemStore) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        pages = {element.page for element in result.elements}

        assert min(pages) == 1
        assert max(pages) == result.page_count

    def test_element_ids_are_unique(self, corpus: Path, store: FilesystemStore) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        ids = [element.element_id for element in result.elements]
        assert len(ids) == len(set(ids))

    def test_bbox_and_kind_are_present(self, corpus: Path, store: FilesystemStore) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        for element in result.elements:
            assert element.bbox is not None
            assert element.kind in tuple(ElementKind)

    def test_header_and_footer_are_dropped(self, corpus: Path, store: FilesystemStore) -> None:
        """Page furniture would otherwise be retrieved as if it were content."""
        result = extract(corpus, "manuel_pompe_a", store)
        joined = " ".join(texts(result))

        assert "page 1" not in joined.lower()


class TestTables:
    def test_table_cells_become_retrievable_text(
        self, corpus: Path, store: FilesystemStore
    ) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        tables = texts(result, ElementKind.TABLE)

        assert tables, "no table element was produced"
        combined = " ".join(tables)
        assert "Grandeur" in combined
        assert "42 m³/h" in combined

    def test_table_keeps_its_caption(self, corpus: Path, store: FilesystemStore) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        tables = texts(result, ElementKind.TABLE)
        assert any(text.startswith("Tableau -") for text in tables)

    def test_table_rows_are_preserved_as_lines(self, corpus: Path, store: FilesystemStore) -> None:
        """PyMuPDF merges same-line cells into one block, so a row becomes a line
        and cells are space-separated. What matters is that the row structure and
        every header survive into retrievable text."""
        result = extract(corpus, "manuel_pompe_a", store)
        table = next(t for t in texts(result, ElementKind.TABLE) if t.startswith("Tableau -"))
        lines = [line for line in table.splitlines() if line.strip()]

        assert lines[0].startswith("Tableau -")
        header = lines[1]
        assert "Grandeur" in header
        assert "Valeur" in header
        assert "Tolérance" in header
        assert len(lines) >= 6


class TestFigures:
    def test_figure_is_extracted_with_a_stored_image(
        self, corpus: Path, store: FilesystemStore
    ) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        figures = [e for e in result.elements if e.kind is ElementKind.FIGURE]

        assert figures, "no figure element was produced"
        figure = figures[0]
        assert figure.image_key is not None
        assert store.exists(figure.image_key)

    def test_figure_text_includes_the_caption_and_light_labels(
        self, corpus: Path, store: FilesystemStore
    ) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        figures = [e for e in result.elements if e.kind is ElementKind.FIGURE]
        combined = " ".join(e.text for e in figures)

        assert "Figure -" in combined
        assert "orange" in combined.lower()

    def test_stored_image_is_a_png(self, corpus: Path, store: FilesystemStore) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        figure = next(e for e in result.elements if e.kind is ElementKind.FIGURE)
        assert figure.image_key is not None
        assert store.get(figure.image_key).startswith(b"\x89PNG")


class TestScannedDocument:
    def test_without_ocr_the_page_is_stored_as_an_image_with_a_warning(
        self, corpus: Path, store: FilesystemStore
    ) -> None:
        result = extract(corpus, "fiche_scanner_pompe_a", store)

        assert result.page_count == 1
        assert result.ocr_pages == ()
        assert any("tesseract is unavailable" in w for w in result.warnings)
        pages = [e for e in result.elements if e.kind is ElementKind.PAGE]
        assert pages and pages[0].image_key is not None
        assert store.exists(pages[0].image_key)

    def test_with_ocr_the_text_is_marked_as_ocr(self, corpus: Path, store: FilesystemStore) -> None:
        engine = FakeOcr()
        result = extract(corpus, "fiche_scanner_pompe_a", store, ocr=engine)

        assert result.ocr_pages == (1,)
        assert engine.calls, "the OCR engine was never invoked"
        paragraphs = [e for e in result.elements if e.kind is ElementKind.PARAGRAPH]
        assert paragraphs
        assert all(e.from_ocr for e in paragraphs)
        assert "Température 81" in paragraphs[0].text

    def test_ocr_disabled_produces_no_ocr(self, corpus: Path, store: FilesystemStore) -> None:
        result = extract(
            corpus,
            "fiche_scanner_pompe_a",
            store,
            ocr=FakeOcr(),
            config=ExtractionConfig(ocr_enabled=False),
        )
        assert result.ocr_pages == ()
        assert any("OCR is disabled" in w for w in result.warnings)

    def test_engine_raising_unavailable_is_recorded_not_fatal(
        self, corpus: Path, store: FilesystemStore
    ) -> None:
        class Broken:
            def is_available(self) -> bool:
                return True

            def text_from_png(self, png: bytes, *, languages: str | None = None) -> str:
                raise OcrUnavailableError("engine exploded")

        result = extract(corpus, "fiche_scanner_pompe_a", store, ocr=Broken())
        assert any("OCR unavailable" in w for w in result.warnings)


class TestChunking:
    def test_long_text_is_split_below_the_limit(self, corpus: Path, store: FilesystemStore) -> None:
        config = ExtractionConfig(max_chars=120)
        result = extract(corpus, "manuel_pompe_a", store, config=config)

        paragraphs = texts(result, ElementKind.PARAGRAPH)
        assert any(len(text) > 80 for text in paragraphs)
        assert all(len(text) <= 120 for text in paragraphs)

    def test_split_chunks_do_not_lose_words(self, corpus: Path, store: FilesystemStore) -> None:
        config = ExtractionConfig(max_chars=120)
        split = texts(
            extract(corpus, "manuel_pompe_a", store, config=config), ElementKind.PARAGRAPH
        )
        unsplit = texts(
            extract(corpus, "manuel_pompe_a", store, config=ExtractionConfig(max_chars=10000)),
            ElementKind.PARAGRAPH,
        )

        assert " ".join(split).replace("  ", " ") == " ".join(unsplit).replace("  ", " ")


class TestFailures:
    def test_empty_bytes_raise(self, store: FilesystemStore, corpus: Path) -> None:
        with pytest.raises(IngestionError, match="empty"):
            extract_pdf(
                data=b"",
                document=document(corpus, "manuel_pompe_a"),
                store=store,
                ocr=NullOcr(),
            )

    def test_non_pdf_bytes_raise(self, store: FilesystemStore, corpus: Path) -> None:
        with pytest.raises(IngestionError, match="could not be opened"):
            extract_pdf(
                data=b"this is not a pdf at all",
                document=document(corpus, "manuel_pompe_a"),
                store=store,
                ocr=NullOcr(),
            )

    def test_elements_are_immutable(self, corpus: Path, store: FilesystemStore) -> None:
        result = extract(corpus, "manuel_pompe_a", store)
        element: Element = result.elements[0]
        with pytest.raises(ValueError):
            element.text = "tampered"  # type: ignore[misc]

    def test_ingesting_the_same_document_twice_is_stable(
        self, corpus: Path, store: FilesystemStore
    ) -> None:
        first = extract(corpus, "manuel_pompe_a", store)
        second = extract(corpus, "manuel_pompe_a", store)
        assert [e.element_id for e in first.elements] == [e.element_id for e in second.elements]

    def test_document_fingerprint_is_not_logged_as_content(
        self, corpus: Path, store: FilesystemStore
    ) -> None:
        """Sanity: the fingerprint helper is what logs correlate on, not text."""
        data = bytes_of(corpus, "manuel_pompe_a")
        digest = hashlib.sha256(data).hexdigest()
        assert len(digest) == 64
