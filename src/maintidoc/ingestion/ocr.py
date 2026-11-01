"""Local OCR for scanned documents (SPEC §3: "PyMuPDF + OCR local").

Local means no cloud OCR service — the text never leaves the machine, which is
what SPEC §9 requires for document content.

The tesseract binary is **not** present on the Windows development host, so this
module is written to be absent-safe: `TesseractOcr.is_available()` lets the
extractor decide whether to attempt OCR, and a failure raises a typed error the
caller records as a warning rather than a crash. This is why ingestion can run
on the dev host (text-layer documents only) while the container, which installs
tesseract with French language data, handles the scanned corpus document.
"""

from __future__ import annotations

import io
import shutil
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from maintidoc.errors import MaintiDocError
from maintidoc.observability import get_logger

logger = get_logger(__name__)

#: French first, English as a fallback for part numbers and units that OCR
#: handles better with the Latin model.
DEFAULT_LANGUAGES = "fra+eng"


class OcrUnavailableError(MaintiDocError):
    """The OCR engine cannot run here (binary missing or init failed)."""

    code = "ocr_unavailable"
    http_status = 503


class OcrFailedError(MaintiDocError):
    """OCR was attempted on an image and produced no usable text."""

    code = "ocr_failed"
    http_status = 500


@runtime_checkable
class OcrEngine(Protocol):
    """Minimal OCR contract so tests can substitute a deterministic engine."""

    def is_available(self) -> bool:
        """True when this engine can actually run on this machine."""
        ...

    def text_from_png(self, png: bytes, *, languages: str | None = None) -> str:
        """Extract text from PNG bytes. Empty string means "no text found"."""
        ...


@dataclass(frozen=True, slots=True)
class TesseractOcr:
    """Tesseract via pytesseract. The binary must be on PATH."""

    languages: str = DEFAULT_LANGUAGES

    def is_available(self) -> bool:
        return shutil.which("tesseract") is not None

    def text_from_png(self, png: bytes, *, languages: str | None = None) -> str:
        if not self.is_available():
            raise OcrUnavailableError("tesseract binary is not on PATH")

        import pytesseract
        from PIL import Image, UnidentifiedImageError

        # pytesseract is untyped; annotate so mypy strict does not see Any here.
        text: str = ""
        try:
            with Image.open(io.BytesIO(png)) as image:
                # Convert early: tesseract rejects palette/alpha modes in some builds.
                text = str(
                    pytesseract.image_to_string(
                        image.convert("RGB"),
                        lang=languages or self.languages,
                    )
                )
        except (UnidentifiedImageError, OSError) as exc:
            raise OcrFailedError(
                f"image could not be read for OCR: {exc.__class__.__name__}"
            ) from exc
        except Exception as exc:  # pytesseract raises TesseractNotFoundError, Error, ...
            raise OcrFailedError(f"OCR failed: {exc.__class__.__name__}") from exc

        return text.strip()


@dataclass(frozen=True, slots=True)
class NullOcr:
    """An engine that is never available. Used when OCR is disabled by config."""

    def is_available(self) -> bool:
        return False

    def text_from_png(self, png: bytes, *, languages: str | None = None) -> str:
        raise OcrUnavailableError("OCR is disabled by configuration")


__all__ = [
    "DEFAULT_LANGUAGES",
    "NullOcr",
    "OcrEngine",
    "OcrFailedError",
    "OcrUnavailableError",
    "TesseractOcr",
]
