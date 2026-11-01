"""The model-output draft layer (agent/drafts.py).

The point of these tests is the boundary itself: a model may *signal* that a
ticket is wanted, but it may not supply the security material that makes one
creatable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from maintidoc.agent.drafts import ModelAnswerDraft, ModelEvidenceVerdict, ModelTicketDraft
from maintidoc.schemas import AnswerPayload, AnswerStatus, Citation
from maintidoc.security.approval import canonical_content_hash, derive_idempotency_key

ANSWERED = {
    "status": "answered",
    "answer": "Le voyant orange signale une alarme de température.",
    "citations": [{"document_id": "manuel_pompe_a", "page": 18, "element_id": "figure_18_2"}],
    "uncertainties": [],
    "clarification_question": None,
    "proposed_action": None,
}


class TestModelAnswerDraft:
    def test_accepts_a_well_formed_answer(self) -> None:
        draft = ModelAnswerDraft.model_validate(ANSWERED)
        assert draft.status is AnswerStatus.ANSWERED
        assert draft.citations[0].page == 18

    def test_needs_approval_may_signal_intent_without_a_proposal(self) -> None:
        """The model signals; the workflow builds the binding proposal."""
        draft = ModelAnswerDraft.model_validate(
            {**ANSWERED, "status": "needs_approval", "citations": []}
        )
        assert draft.wants_ticket is True
        assert draft.proposed_action is None

    def test_a_model_supplied_proposal_is_rejected(self) -> None:
        """NEVER #3: content_hash and idempotency_key are workflow-computed."""
        with pytest.raises(ValidationError):
            ModelAnswerDraft.model_validate(
                {
                    **ANSWERED,
                    "status": "needs_approval",
                    "proposed_action": {
                        "equipment_id": "pompe-a-01",
                        "title": "t",
                        "description": "d",
                        "content_hash": "a" * 64,
                        "idempotency_key": "idem-0123456789abcdef",
                    },
                }
            )

    def test_unknown_field_is_rejected(self) -> None:
        """NEVER #6: a confidence value cannot be smuggled in."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ModelAnswerDraft.model_validate({**ANSWERED, "confidence": 0.9})

    def test_needs_clarification_requires_a_question(self) -> None:
        with pytest.raises(ValidationError, match="clarification_question"):
            ModelAnswerDraft.model_validate({**ANSWERED, "status": "needs_clarification"})

    def test_an_uncited_answered_draft_is_accepted_at_this_layer(self) -> None:
        """The draft is lenient on purpose: the workflow may still downgrade it
        to abstained after citation verification. The strict rule lives on
        AnswerPayload."""
        draft = ModelAnswerDraft.model_validate({**ANSWERED, "citations": []})
        assert draft.status is AnswerStatus.ANSWERED
        with pytest.raises(ValidationError, match="at least one citation"):
            AnswerPayload.model_validate(draft.model_dump())

    def test_wants_ticket_is_false_for_other_statuses(self) -> None:
        assert ModelAnswerDraft.model_validate(ANSWERED).wants_ticket is False


class TestPromotionToContract:
    def test_an_answered_draft_promotes_to_a_valid_payload(self) -> None:
        draft = ModelAnswerDraft.model_validate(ANSWERED)
        payload = AnswerPayload.model_validate(draft.model_dump())
        assert payload.status is AnswerStatus.ANSWERED
        assert isinstance(payload.citations[0], Citation)

    def test_the_workflow_can_attach_a_ticket_proposal(self) -> None:
        draft = ModelAnswerDraft.model_validate(
            {**ANSWERED, "status": "needs_approval", "citations": []}
        )
        content_hash = canonical_content_hash(
            equipment_id="pompe-a-01", title="Voyant orange", description="Alarme température."
        )
        payload = AnswerPayload.model_validate(
            {
                **draft.model_dump(),
                "proposed_action": {
                    "equipment_id": "pompe-a-01",
                    "title": "Voyant orange",
                    "description": "Alarme température.",
                    "content_hash": content_hash,
                    "idempotency_key": derive_idempotency_key(
                        user_id="u-tech-1", content_hash=content_hash
                    ),
                },
            }
        )
        assert payload.status is AnswerStatus.NEEDS_APPROVAL
        assert payload.proposed_action is not None
        assert payload.proposed_action.content_hash == content_hash


class TestTicketAndVerdictDrafts:
    def test_ticket_draft_is_text_only(self) -> None:
        draft = ModelTicketDraft(title="Voyant orange", description="Alarme température.")
        assert draft.title == "Voyant orange"

    def test_ticket_draft_rejects_security_material(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ModelTicketDraft.model_validate(
                {"title": "t", "description": "d", "content_hash": "a" * 64}
            )

    def test_ticket_draft_requires_both_fields(self) -> None:
        with pytest.raises(ValidationError):
            ModelTicketDraft(title="", description="d")

    def test_evidence_verdict(self) -> None:
        verdict = ModelEvidenceVerdict(supported=False, reason="hors sujet")
        assert verdict.supported is False

    def test_evidence_verdict_rejects_a_score(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ModelEvidenceVerdict.model_validate(
                {"supported": True, "reason": "r", "confidence": 0.99}
            )
