"""SPEC §7 output contract.

These tests are the executable form of AGENTS.md NEVER #5 and #6: no citation
that was not retrieved, and no model-invented confidence value.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from maintidoc.schemas import (
    _FORBIDDEN_FIELD_TOKENS,
    AnswerPayload,
    AnswerStatus,
    Citation,
    ElementKind,
    RunProvenance,
    TicketProposal,
    assert_no_forbidden_fields,
    filter_citations_by_access,
)
from maintidoc.security.approval import canonical_content_hash, derive_idempotency_key

# The example from SPEC §7, verbatim, including the French text.
SPEC_7_EXAMPLE: dict[str, object] = {
    "status": "needs_clarification",
    "answer": "Le modèle de l'équipement doit être confirmé.",
    "citations": [
        {
            "document_id": "manuel_pompe_a",
            "page": 18,
            "element_id": "figure_18_2",
        }
    ],
    "uncertainties": ["La plaque signalétique est illisible."],
    "clarification_question": "Pouvez-vous photographier la plaque signalétique ?",
    "proposed_action": None,
}

_CITATION = Citation(document_id="manuel_pompe_a", page=18, element_id="figure_18_2")


def _proposal() -> TicketProposal:
    content_hash = canonical_content_hash(
        equipment_id="pompe-a-01",
        title="Voyant orange au panneau de commande",
        description="Alarme température relevée sur site.",
    )
    return TicketProposal(
        equipment_id="pompe-a-01",
        title="Voyant orange au panneau de commande",
        description="Alarme température relevée sur site.",
        content_hash=content_hash,
        idempotency_key=derive_idempotency_key(user_id="u-tech-1", content_hash=content_hash),
    )


def _provenance(**overrides: object) -> RunProvenance:
    base: dict[str, object] = {
        "vlm_model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "vlm_provider": "qwen",
        "vlm_device": "cuda",
        "prompt_versions": {"answer": "v1"},
    }
    base.update(overrides)
    return RunProvenance.model_validate(base)


class TestSpecExample:
    def test_spec_section7_example_parses(self) -> None:
        payload = AnswerPayload.model_validate(SPEC_7_EXAMPLE)

        assert payload.status is AnswerStatus.NEEDS_CLARIFICATION
        assert payload.answer == "Le modèle de l'équipement doit être confirmé."
        assert len(payload.citations) == 1
        assert payload.citations[0].document_id == "manuel_pompe_a"
        assert payload.citations[0].page == 18
        assert payload.citations[0].element_id == "figure_18_2"
        assert payload.uncertainties == ("La plaque signalétique est illisible.",)
        assert (
            payload.clarification_question == "Pouvez-vous photographier la plaque signalétique ?"
        )
        assert payload.proposed_action is None

    def test_spec_example_roundtrips_through_json(self) -> None:
        payload = AnswerPayload.model_validate(SPEC_7_EXAMPLE)
        assert AnswerPayload.model_validate_json(payload.model_dump_json()) == payload

    def test_status_enum_covers_the_four_required_values(self) -> None:
        assert {s.value for s in AnswerStatus} == {
            "answered",
            "needs_clarification",
            "abstained",
            "needs_approval",
        }


class TestCitationRequirements:
    def test_citation_requires_the_three_spec_fields(self) -> None:
        for missing in ("document_id", "page", "element_id"):
            data = {"document_id": "d", "page": 1, "element_id": "e"}
            del data[missing]
            with pytest.raises(ValidationError, match=missing):
                Citation.model_validate(data)

    @pytest.mark.parametrize("page", [0, -1, -100])
    def test_page_is_one_indexed_and_positive(self, page: int) -> None:
        with pytest.raises(ValidationError):
            Citation(document_id="d", page=page, element_id="e")

    def test_blank_identifiers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Citation(document_id="   ", page=1, element_id="e")

    def test_optional_metadata_accepted(self) -> None:
        citation = Citation(
            document_id="manuel_pompe_a",
            page=18,
            element_id="figure_18_2",
            document_version="2024-03",
            kind=ElementKind.FIGURE,
            quote="Voyant orange : alarme de température.",
        )
        assert citation.kind is ElementKind.FIGURE
        assert citation.document_version == "2024-03"


class TestStatusInvariants:
    def test_answered_requires_at_least_one_citation(self) -> None:
        with pytest.raises(ValidationError, match="at least one citation"):
            AnswerPayload(status=AnswerStatus.ANSWERED, answer="Le voyant indique une surchauffe.")

    def test_answered_requires_a_non_empty_answer(self) -> None:
        with pytest.raises(ValidationError, match="non-empty answer"):
            AnswerPayload(status=AnswerStatus.ANSWERED, answer="   ", citations=[_CITATION])

    def test_answered_with_citation_is_valid(self) -> None:
        payload = AnswerPayload(
            status=AnswerStatus.ANSWERED,
            answer="Le voyant orange signale une alarme de température.",
            citations=[_CITATION],
        )
        assert payload.status is AnswerStatus.ANSWERED

    def test_needs_clarification_requires_the_question(self) -> None:
        with pytest.raises(ValidationError, match="clarification_question"):
            AnswerPayload(status=AnswerStatus.NEEDS_CLARIFICATION, answer="Précisez le modèle.")

    def test_abstained_requires_stated_uncertainties(self) -> None:
        """SPEC §8: abstain by stating what is missing, not by guessing."""
        with pytest.raises(ValidationError, match="uncertainties"):
            AnswerPayload(status=AnswerStatus.ABSTAINED, answer="Je ne peux pas répondre.")

    def test_abstained_with_uncertainties_is_valid(self) -> None:
        payload = AnswerPayload(
            status=AnswerStatus.ABSTAINED,
            answer="Aucune source accessible ne couvre cet équipement.",
            uncertainties=["Photo illisible", "Aucun manuel pour cette famille"],
        )
        assert payload.status is AnswerStatus.ABSTAINED

    def test_needs_approval_requires_the_proposal(self) -> None:
        with pytest.raises(ValidationError, match="proposed_action"):
            AnswerPayload(status=AnswerStatus.NEEDS_APPROVAL, answer="Ticket à valider.")

    def test_proposal_is_rejected_without_needs_approval(self) -> None:
        """A ticket draft must never ride along on an ordinary answer."""
        with pytest.raises(ValidationError, match="only permitted with status=needs_approval"):
            AnswerPayload(
                status=AnswerStatus.ANSWERED,
                answer="Réponse.",
                citations=[_CITATION],
                proposed_action=_proposal(),
            )

    def test_needs_approval_carries_the_exact_content(self) -> None:
        proposal = _proposal()
        payload = AnswerPayload(
            status=AnswerStatus.NEEDS_APPROVAL,
            answer="Voici le ticket proposé.",
            proposed_action=proposal,
        )
        assert payload.proposed_action is not None
        assert payload.proposed_action.content_hash == proposal.content_hash


class TestNoFabricatedConfidence:
    """AGENTS.md NEVER #6."""

    @pytest.mark.parametrize(
        "field",
        ["confidence", "confidence_score", "certainty", "probability", "answer_percent"],
    )
    def test_unknown_fields_are_rejected_not_ignored(self, field: str) -> None:
        data = dict(SPEC_7_EXAMPLE)
        data[field] = 0.87
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            AnswerPayload.model_validate(data)

    def test_the_contract_itself_defines_no_confidence_field(self) -> None:
        """Guards against someone *adding* one to the schema later."""
        import maintidoc.schemas as schemas_module

        models = [
            value
            for value in vars(schemas_module).values()
            if isinstance(value, type) and issubclass(value, BaseModel)
        ]
        assert models, "expected to find schema models to audit"
        for model in models:
            assert_no_forbidden_fields(model)

    def test_forbidden_token_list_is_not_empty(self) -> None:
        assert "confidence" in _FORBIDDEN_FIELD_TOKENS


class TestTicketProposal:
    @pytest.mark.parametrize(
        "bad_hash",
        ["not-a-hash", "z" * 64, "A" * 64, "abcd1234" * 7 + "xy!!"],
        ids=["too-short", "non-hex", "uppercase", "invalid-chars"],
    )
    def test_content_hash_must_be_lowercase_sha256_hex(self, bad_hash: str) -> None:
        with pytest.raises(ValidationError, match="content_hash"):
            TicketProposal(
                equipment_id="pompe-a-01",
                title="t",
                description="d",
                content_hash=bad_hash,
                idempotency_key="idem-0123456789abcdef",
            )

    def test_a_real_hash_is_accepted(self) -> None:
        real = canonical_content_hash(equipment_id="e", title="t", description="d")
        proposal = TicketProposal(
            equipment_id="e",
            title="t",
            description="d",
            content_hash=real,
            idempotency_key=derive_idempotency_key(user_id="u-tech-1", content_hash=real),
        )
        assert proposal.content_hash == real

    def test_short_idempotency_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TicketProposal(
                equipment_id="e",
                title="t",
                description="d",
                content_hash=canonical_content_hash(equipment_id="e", title="t", description="d"),
                idempotency_key="short",
            )


class TestProvenance:
    def test_records_the_model_that_actually_answered(self) -> None:
        """ADR 0003: a fallback must never be invisible."""
        provenance = _provenance(
            vlm_model="Qwen/Qwen2.5-VL-3B-Instruct",
            requested_model="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            fallback_applied=True,
            vlm_device="cpu",
        )
        assert provenance.fallback_applied is True
        assert provenance.vlm_model != provenance.requested_model

    def test_budget_counters_cannot_go_negative(self) -> None:
        with pytest.raises(ValidationError):
            _provenance(model_calls_used=-1)


class TestAccessFiltering:
    def test_inaccessible_citations_are_dropped(self) -> None:
        payload = AnswerPayload(
            status=AnswerStatus.ANSWERED,
            answer="Réponse sourcée.",
            citations=[
                Citation(document_id="public_doc", page=1, element_id="e1", quote="visible"),
                Citation(document_id="restricted_doc", page=2, element_id="e2", quote="secret"),
            ],
        )
        filtered = filter_citations_by_access(payload, allowed_document_ids={"public_doc"})
        assert [c.document_id for c in filtered.citations] == ["public_doc"]

    def test_emptying_an_answered_payload_raises(self) -> None:
        """Silently returning an uncited 'answered' would break NEVER #5."""
        payload = AnswerPayload(
            status=AnswerStatus.ANSWERED,
            answer="Réponse.",
            citations=[Citation(document_id="restricted_doc", page=2, element_id="e2")],
        )
        with pytest.raises(ValidationError, match="at least one citation"):
            filter_citations_by_access(payload, allowed_document_ids={"other_doc"})


class TestImmutability:
    def test_payload_is_frozen(self) -> None:
        payload = AnswerPayload.model_validate(SPEC_7_EXAMPLE)
        with pytest.raises(ValidationError):
            payload.status = AnswerStatus.ANSWERED  # type: ignore[misc]

    def test_citation_lists_are_tuples_not_lists(self) -> None:
        payload = AnswerPayload(status=AnswerStatus.ANSWERED, answer="r", citations=[_CITATION])
        assert isinstance(payload.citations, tuple)
        assert isinstance(payload.uncertainties, tuple)
