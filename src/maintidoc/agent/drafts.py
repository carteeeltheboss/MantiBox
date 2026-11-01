"""Lenient parse of raw model output, before the workflow builds the contract.

Two shapes, deliberately distinct:

* `ModelAnswerDraft` — what the model is allowed to return. Lenient about
  cross-field invariants, and it may *not* fill `proposed_action`.
* `AnswerPayload` (schemas.py) — what the user receives. Strict.

Separating them is not ceremony. A `needs_approval` payload must carry the exact
ticket content, its sha256 and an idempotency key (SPEC §6). Only the workflow
can compute those, because the hash has to cover the content the human is shown
and the key has to be derived from the acting user. A model that filled them in
would be inventing security material.

So the model signals intent, and the workflow constructs the binding proposal.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maintidoc.schemas import AnswerStatus, Citation


class ModelAnswerDraft(BaseModel):
    """Raw output of the `answer` prompt, validated but not yet trustworthy."""

    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus
    answer: str = Field(max_length=8000)
    citations: list[Citation] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    clarification_question: str | None = Field(default=None, max_length=1000)
    #: Must stay null coming from the model; see the module docstring.
    proposed_action: None = None

    @model_validator(mode="after")
    def _minimum_coherence(self) -> Self:
        """Only the checks that make the draft *interpretable*.

        The stronger invariants (an answered payload must cite, an abstention
        must state what is missing) are enforced on `AnswerPayload`, after the
        workflow has verified the citations. Checking them here would reject a
        draft the workflow could still legitimately repair by downgrading it.
        """
        if self.status is AnswerStatus.NEEDS_CLARIFICATION and not self.clarification_question:
            msg = "status=needs_clarification requires clarification_question"
            raise ValueError(msg)
        return self

    @property
    def wants_ticket(self) -> bool:
        """True when the model signalled that a sensitive action was requested."""
        return self.status is AnswerStatus.NEEDS_APPROVAL


class ModelTicketDraft(BaseModel):
    """Raw output of the `ticket_draft` prompt: text only, no security material."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=8000)


class ModelEvidenceVerdict(BaseModel):
    """Raw output of the `evidence_check` prompt.

    Mirrors `vlm.base.EvidenceVerdict`; kept here so the agent layer has one
    place to look for every model-output shape it parses.
    """

    model_config = ConfigDict(extra="forbid")

    supported: bool
    reason: str = Field(default="", max_length=1000)


__all__ = ["ModelAnswerDraft", "ModelEvidenceVerdict", "ModelTicketDraft"]
