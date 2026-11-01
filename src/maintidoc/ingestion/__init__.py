"""Ingestion: turn authorized documents into retrievable elements."""

from __future__ import annotations

from maintidoc.ingestion.ocr import (
    DEFAULT_LANGUAGES,
    NullOcr,
    OcrEngine,
    OcrFailedError,
    OcrUnavailableError,
    TesseractOcr,
)
from maintidoc.ingestion.pdf import (
    ExtractionConfig,
    ExtractionResult,
    extract_pdf,
)

__all__ = [
    "DEFAULT_LANGUAGES",
    "ExtractionConfig",
    "ExtractionResult",
    "NullOcr",
    "OcrEngine",
    "OcrFailedError",
    "OcrUnavailableError",
    "TesseractOcr",
    "extract_pdf",
]
