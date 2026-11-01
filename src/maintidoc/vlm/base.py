"""Vision-language model provider interface (SPEC §3, §8).

Two model-output contracts live here because both are validated before anything
downstream trusts them (SPEC §7: "Les réponses sont validées avec un schéma
Pydantic"):

* `VisualObservation` — what is *observable* in a photograph. Deliberately has
  no diagnosis field: SPEC §8 states a visual resemblance is not enough to
  confirm a diagnosis, so the shape itself refuses to express one.
* `EvidenceVerdict` — whether a retrieved element actually supports a claim
  (AGENTS.md NEVER #5).

`VlmProvider` is the seam that keeps torch optional: the deterministic
`StubVlmProvider` satisfies the whole pipeline and test suite, while
`QwenVlProvider` performs real inference. Both return the same `VlmResponse`,
which carries the model that actually served the call so a fallback can never be
silent (ADR 0003).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class OutputSchema(StrEnum):
    """Which validated contract a `generate` call must satisfy.

    The caller declares this rather than the provider inferring it from the
    prompt text. Inference was tried first and is fragile: the `answer` prompt
    embeds a rendered `visual_analysis` payload, so its text legitimately
    contains another schema's field names and any content-sniffing heuristic
    picks the wrong contract.

    Real providers ignore it — they generate text either way. It exists so the
    deterministic stub can produce output that validates, and so a call site
    documents what it expects.
    """

    TEXT = "text"
    ANSWER = "answer"
    VISUAL_OBSERVATION = "visual_observation"
    EVIDENCE_VERDICT = "evidence_verdict"
    TICKET_DRAFT = "ticket_draft"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VisualObservation(_Strict):
    """Observable facts from an image — no diagnosis, no confidence.

    SPEC §8: the system states the available evidence and the missing
    information rather than emitting a fabricated certainty.
    """

    #: False when the photo is too blurred, dark or cropped to read.
    image_readable: bool
    #: Only set when a nameplate, label or model reference is actually legible.
    equipment_identified: str | None = Field(default=None, max_length=200)
    #: What can be seen: indicator lights, labels, gauges, damage, corrosion.
    observations: tuple[str, ...] = ()
    #: What cannot be determined from this image.
    limitations: tuple[str, ...] = ()
    #: True when the image shows something outside the documentation role, e.g.
    #: a person at risk or an unsafe condition — the workflow must then abstain.
    out_of_role: bool = False


class EvidenceVerdict(_Strict):
    """Does the retrieved element actually support the claim?"""

    supported: bool
    #: Short justification, quoted from the evidence where possible.
    reason: str = Field(max_length=1000)


@dataclass(frozen=True, slots=True)
class VlmResponse:
    """Raw provider output plus the provenance needed to report it honestly."""

    text: str
    #: The model that actually produced this text — after any fallback.
    model_id: str
    device: str
    provider: str
    #: True when the ladder stepped down from the configured model.
    fallback_applied: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    finish_reason: str | None = None
    #: Ordered audit trail from the resolver; surfaced in logs and MLflow.
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ImageInput:
    """An image handed to the model.

    Carries bytes rather than a path so the provider never touches the
    filesystem, and so the access check has already happened upstream.
    """

    data: bytes
    media_type: str = "image/jpeg"
    #: Identifier for logging and citation only — never the content itself.
    reference: str = ""


@runtime_checkable
class VlmProvider(Protocol):
    """The seam between the workflow and whatever inference is available."""

    @property
    def model_id(self) -> str:
        """The model that will actually serve requests."""
        ...

    @property
    def device(self) -> str: ...

    @property
    def fallback_applied(self) -> bool: ...

    def describe_image(
        self,
        *,
        instruction: str,
        images: Sequence[ImageInput] = (),
    ) -> VlmResponse:
        """Analyse one or more images under `instruction`."""
        ...

    def generate(
        self,
        *,
        instruction: str,
        images: Sequence[ImageInput] = (),
        max_new_tokens: int | None = None,
        expects: OutputSchema = OutputSchema.TEXT,
    ) -> VlmResponse:
        """Text (optionally image-conditioned) generation.

        `expects` declares the contract the caller will validate the output
        against. Inference providers may ignore it; the stub honours it.
        """
        ...

    def close(self) -> None:
        """Release weights and device memory."""
        ...


__all__ = [
    "EvidenceVerdict",
    "ImageInput",
    "OutputSchema",
    "VisualObservation",
    "VlmProvider",
    "VlmResponse",
]
