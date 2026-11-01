"""Vision-language inference behind a provider seam.

`base` holds the contract and `stub` a deterministic implementation used by
tests and by the portable CPU deployment. The workflow depends only on the
`VlmProvider` protocol, so torch stays optional at import time.

The real Qwen2.5-VL provider lives in `maintidoc.vlm.qwen` and is imported
lazily by the provider factory, never at package import: it needs torch,
transformers and several GB of weights.
"""

from __future__ import annotations

from maintidoc.vlm.base import (
    EvidenceVerdict,
    ImageInput,
    OutputSchema,
    VisualObservation,
    VlmProvider,
    VlmResponse,
)
from maintidoc.vlm.stub import STUB_MODEL_ID, StubPolicy, StubVlmProvider

__all__ = [
    "STUB_MODEL_ID",
    "EvidenceVerdict",
    "ImageInput",
    "OutputSchema",
    "StubPolicy",
    "StubVlmProvider",
    "VisualObservation",
    "VlmProvider",
    "VlmResponse",
]
