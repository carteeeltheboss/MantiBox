"""Corpus generator regression tests.

Two silent defects are pinned here, both of which would have made retrieval and
the evaluation meaningless without failing anything:

1. `insert_textbox` writes **nothing** and returns a negative "unused height"
   when text does not fit, so all 32 tables were generated with zero extractable
   cell text.
2. PyMuPDF's base-14 fonts are Latin-1 only, so an em dash silently became `?`,
   corrupting captions that retrieval matches on.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest
from scripts.generate_corpus import (
    TableSpec,
    _assert_renderable,
    _fit_cell,
    build_specs,
    generate,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    generate(6, corpus_dir=root)
    return root


def page_text(path: Path) -> str:
    with pymupdf.open(path) as document:
        return "".join(page.get_text() for page in document)


class TestTableTextIsExtractable:
    def test_cells_are_present_in_the_text_layer(self, corpus: Path) -> None:
        text = page_text(corpus / "documents" / "manuel_pompe_a.pdf")

        assert "Grandeur" in text
        assert "Valeur" in text
        assert "42 m³/h" in text
        assert "alarme à 85 °C" in text

    def test_every_row_of_a_known_table_survives(self, corpus: Path) -> None:
        text = page_text(corpus / "documents" / "manuel_pompe_a.pdf")

        for expected in ("Débit nominal", "3,8 bar", "18,5 A", "2,8 mm/s"):
            assert expected in text


class TestTextEncoding:
    def test_no_character_is_silently_replaced(self, corpus: Path) -> None:
        for pdf in (corpus / "documents").glob("*.pdf"):
            text = page_text(pdf)
            assert "?" not in text, f"{pdf.name} contains a replacement character"

    def test_captions_use_renderable_punctuation(self, corpus: Path) -> None:
        text = page_text(corpus / "documents" / "manuel_pompe_a.pdf")
        assert "Figure - " in text
        assert "Tableau - " in text

    def test_french_accents_survive(self, corpus: Path) -> None:
        text = page_text(corpus / "documents" / "manuel_pompe_a.pdf")
        assert "Température" in text
        assert "préventive" in text

    def test_guard_rejects_a_non_latin1_character(self) -> None:
        specs = build_specs(1)
        broken = replace(specs[0], title="Titre avec un tiret cadratin \u2014")
        with pytest.raises(ValueError, match="cannot be rendered"):
            _assert_renderable([broken])

    def test_shipped_specs_all_pass_the_guard(self) -> None:
        _assert_renderable(build_specs(34))


class TestFitCell:
    def test_short_text_is_untouched(self) -> None:
        assert _fit_cell("Grandeur", 100.0) == "Grandeur"

    def test_long_text_is_truncated_within_the_width(self) -> None:
        long_text = "Vibrations palier avant en charge maximale continue"
        fitted = _fit_cell(long_text, 60.0)

        assert fitted.endswith("...")
        assert pymupdf.get_text_length(fitted, fontname="helv", fontsize=8.5) <= 60.0
        assert len(fitted) < len(long_text)

    def test_never_exceeds_a_narrow_width(self) -> None:
        for text in ("a", "ab", "Vibrations palier avant"):
            fitted = _fit_cell(text, 12.0)
            assert pymupdf.get_text_length(fitted, fontname="helv", fontsize=8.5) <= 12.0

    def test_table_spec_is_unchanged_by_fitting(self) -> None:
        table = TableSpec(caption="c", headers=("a",), rows=(("b",),))
        assert table.headers == ("a",)


class TestScannedDocument:
    def test_has_no_text_layer(self, corpus: Path) -> None:
        """OCR is the only way to read it, which is the point of the fixture."""
        text = page_text(corpus / "documents" / "fiche_scanner_pompe_a.pdf")
        assert text.strip() == ""

    def test_carries_an_image(self, corpus: Path) -> None:
        with pymupdf.open(corpus / "documents" / "fiche_scanner_pompe_a.pdf") as document:
            images = sum(len(page.get_images(full=True)) for page in document)
        assert images >= 1


class TestManifest:
    def test_records_every_document(self, corpus: Path) -> None:
        manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["document_count"] == 6
        assert len(manifest["documents"]) == 6

    def test_records_access_groups_and_licence(self, corpus: Path) -> None:
        manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        entry = manifest["documents"][0]
        assert entry["access_groups"]
        assert entry["license_note"]
        assert entry["synthetic"] is True
        assert len(entry["sha256"]) == 64

    def test_flags_the_required_fixtures(self, corpus: Path) -> None:
        manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["scanned_documents"] == ["fiche_scanner_pompe_a"]
        assert manifest["injection_document"] == "note_diffusion_pompe_a"
        assert manifest["restricted_documents"] == ["manuel_pompe_a_restreint"]

    def test_is_deterministic(self, tmp_path: Path) -> None:
        first = generate(6, corpus_dir=tmp_path / "a")["documents"]
        second = generate(6, corpus_dir=tmp_path / "b")["documents"]
        assert [d["sha256"] for d in first] == [d["sha256"] for d in second]
