"""Structured output contract (SPEC §7).

Every model answer is validated against `AnswerPayload`. Nothing leaves the
system in any other shape.

Three invariants are enforced here rather than trusted to the model:

1. **`extra="forbid"` everywhere.** A model cannot smuggle in a field the
   contract does not define. This is what makes AGENTS.md NEVER #6 — never
   display a model-invented confidence percentage — a structural property
   instead of a code-review hope. See `_FORBIDDEN_FIELD_TOKENS` and the test
   that asserts no schema field ever matches one.
2. **Status-specific requirements.** An `answered` payload must cite something;
   `needs_clarification` must actually ask the question; `abstained` must say
   what is missing; `needs_approval` must carry the exact ticket content the
   human is about to approve.
3. **Provenance is part of the answer.** SPEC §7 requires every run to be tied
   to the model, prompt, reranker and index versions. Because the inference
   layer may step down to a smaller model on constrained hardware (ADR 0003),
   the model that *actually* answered is recorded separately from the one that
   was requested, together with `fallback_applied`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: lowercase sha256 hex. Used for every content hash in the contract so a
#: truncated or uppercase digest fails at the boundary rather than downstream.
_SHA256_HEX = r"^[0-9a-f]{64}$"

#: Field-name fragments that would reintroduce a fabricated confidence score.
#: `extra="forbid"` already blocks unknown fields at runtime; this list exists
#: so a test can fail the build if anyone *adds* one to the contract itself.
_FORBIDDEN_FIELD_TOKENS: frozenset[str] = frozenset(
    {"confidence", "certainty", "probability", "likelihood", "score_pct", "percent"}
)


class AnswerStatus(StrEnum):
    """SPEC §7 + backlog issue #2."""

    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    ABSTAINED = "abstained"
    NEEDS_APPROVAL = "needs_approval"


class ElementKind(StrEnum):
    """The retrievable unit types produced by ingestion (SPEC §4.1)."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    PAGE = "page"
    FIGURE = "figure"
    SCHEMA = "schema"
    IMAGE = "image"


class _Strict(BaseModel):
    """Base for every contract model: unknown fields are a hard error."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Citation(_Strict):
    """A pointer to an element that was *actually retrieved*.

    AGENTS.md NEVER #5: a citation that was not retrieved, or that does not
    support the claim, must never be returned. `quote` holds the verified
    excerpt so the evidence check and the UI can show the support; it is
    stripped for callers who lack access to the document.
    """

    document_id: str = Field(min_length=1, max_length=200)
    page: int = Field(ge=1)
    element_id: str = Field(min_length=1, max_length=200)
    document_version: str | None = Field(default=None, max_length=64)
    kind: ElementKind | None = None
    quote: str | None = Field(default=None, max_length=4000)


class RunProvenance(_Strict):
    """What produced this answer. Required by SPEC §7 and by ADR 0003."""

    #: The model that actually generated the answer, after any fallback.
    vlm_model: str = Field(min_length=1)
    #: What was configured, so a step-down is visible by comparison.
    requested_model: str | None = None
    vlm_provider: str = Field(min_length=1)
    vlm_device: str = Field(min_length=1)
    fallback_applied: bool = False
    #: e.g. {"answer": "v1", "visual_analysis": "v1"}
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    reranker: str | None = None
    index_version: str | None = None
    git_commit: str | None = Field(default=None, max_length=64)
    model_calls_used: int = Field(default=0, ge=0)
    mcp_calls_used: int = Field(default=0, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)


class TicketProposal(_Strict):
    """Exact ticket content awaiting human approval (SPEC §6).

    `content_hash` is what the approval token binds to: change one character of
    the title or description and the hash changes, so a previously granted
    approval no longer validates. `idempotency_key` prevents duplicate creation
    on retry.
    """

    equipment_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=8000)
    content_hash: str = Field(pattern=_SHA256_HEX)
    idempotency_key: str = Field(min_length=16, max_length=128)
    #: Evidence the draft was built from; carried into the approval screen.
    source_citations: tuple[Citation, ...] = ()


class AnswerPayload(_Strict):
    """The one response shape. Mirrors the SPEC §7 example exactly."""

    status: AnswerStatus
    answer: str = Field(max_length=8000)
    citations: tuple[Citation, ...] = ()
    uncertainties: tuple[str, ...] = ()
    clarification_question: str | None = Field(default=None, max_length=1000)
    proposed_action: TicketProposal | None = None
    provenance: RunProvenance | None = None

    @model_validator(mode="after")
    def _status_invariants(self) -> Self:
        status = self.status

        if status is AnswerStatus.ANSWERED:
            # SPEC §2: outputs are *sourced* answers. An answer with no citation
            # is not a sourced answer, so it must be downgraded instead.
            if not self.answer:
                msg = "status=answered requires a non-empty answer"
                raise ValueError(msg)
            if not self.citations:
                msg = (
                    "status=answered requires at least one citation; "
                    "use needs_clarification or abstained when nothing supports the claim"
                )
                raise ValueError(msg)

        elif status is AnswerStatus.NEEDS_CLARIFICATION:
            if not self.clarification_question:
                msg = "status=needs_clarification requires clarification_question"
                raise ValueError(msg)

        elif status is AnswerStatus.ABSTAINED:
            # SPEC §8: state the available evidence and the missing information
            # instead of guessing or emitting a confidence number.
            if not self.uncertainties:
                msg = "status=abstained requires at least one entry in uncertainties"
                raise ValueError(msg)

        elif status is AnswerStatus.NEEDS_APPROVAL:
            if self.proposed_action is None:
                msg = "status=needs_approval requires proposed_action"
                raise ValueError(msg)

        if self.proposed_action is not None and status is not AnswerStatus.NEEDS_APPROVAL:
            msg = "proposed_action is only permitted with status=needs_approval"
            raise ValueError(msg)

        return self


def assert_no_forbidden_fields(model: type[BaseModel]) -> None:
    """Fail loudly if a schema (re)introduces a fabricated-confidence field.

    Called from the test suite over every model in this module.
    """
    for name in model.model_fields:
        lowered = name.lower()
        for token in _FORBIDDEN_FIELD_TOKENS:
            if token in lowered:
                msg = (
                    f"{model.__name__}.{name} matches forbidden token '{token}': "
                    "AGENTS.md NEVER #6 forbids model-invented confidence values"
                )
                raise AssertionError(msg)


class ApprovalDecision(_Strict):
    """The human's verdict on a `TicketProposal` (SPEC §6)."""

    approved: bool
    #: Must equal the proposal's content_hash; a mismatch means the content the
    #: user saw is not the content being created.
    content_hash: str = Field(pattern=_SHA256_HEX)
    idempotency_key: str = Field(min_length=16, max_length=128)
    note: str | None = Field(default=None, max_length=1000)


class TicketCreationResult(_Strict):
    ticket_id: str = Field(min_length=1, max_length=200)
    equipment_id: str = Field(min_length=1, max_length=200)
    created: bool
    #: True when the idempotency key matched an existing ticket, i.e. this call
    #: was a replay and nothing new was created.
    deduplicated: bool = False


def filter_citations_by_access(
    payload: AnswerPayload, *, allowed_document_ids: set[str]
) -> AnswerPayload:
    """Drop citations to documents the caller may not open.

    Defence in depth for SPEC §4.2: access is filtered inside the vector query
    *and* re-checked on display.

    The result is rebuilt through `model_validate` on purpose. If filtering
    empties the citations of an `answered` payload, that payload is no longer a
    sourced answer and the status invariants must reject it — silently returning
    an uncited "answered" would violate AGENTS.md NEVER #5.
    """
    kept = [c.model_dump() for c in payload.citations if c.document_id in allowed_document_ids]
    data = payload.model_dump()
    data["citations"] = kept
    return AnswerPayload.model_validate(data)


__all__ = [
    "AnswerPayload",
    "AnswerStatus",
    "ApprovalDecision",
    "Citation",
    "ElementKind",
    "RunProvenance",
    "TicketCreationResult",
    "TicketProposal",
    "assert_no_forbidden_fields",
    "filter_citations_by_access",
]
