"""Tests for uncertainty and abstention logic (Issue #20, SPEC §8).

Acceptance criteria:
- Clarification or abstention for unreadable photo, no source, contradictory sources,
  or out-of-role risk
- No model-generated confidence percentage anywhere (AGENTS.md NEVER #6)
- Evidence and missing info shown instead
- Eval scenarios for each case
"""

from __future__ import annotations

import pytest

from maintidoc.agent.uncertainty import (
    UncertaintyCase,
    build_abstention_payload,
    build_clarification_payload,
    detect_uncertainty,
    verify_no_confidence_emitted,
)
from maintidoc.domain import RetrievedElement
from maintidoc.schemas import AnswerPayload, AnswerStatus, ElementKind
from maintidoc.vlm.base import VisualObservation
from tests.factories import make_element


class TestUncertaintyAndAbstention:
    def test_unreadable_photo_yields_clarification(self) -> None:
        obs = VisualObservation(
            image_readable=False,
            equipment_identified=None,
            observations=(),
            limitations=("Image floue",),
            out_of_role=False,
        )

        case = detect_uncertainty(observation=obs, retrieved_elements=[])
        assert case is UncertaintyCase.UNREADABLE_PHOTO

        payload = build_clarification_payload(case=case, missing_info="plaque signalétique")
        assert payload.status is AnswerStatus.NEEDS_CLARIFICATION
        assert payload.clarification_question is not None
        assert (
            "photo" in payload.clarification_question.lower()
            or "plaque" in payload.clarification_question.lower()
        )
        assert len(payload.uncertainties) >= 1
        verify_no_confidence_emitted(payload)

    def test_out_of_role_risk_yields_abstention(self) -> None:
        obs = VisualObservation(
            image_readable=True,
            equipment_identified="pompe_01",
            observations=("Fuite importante de produit corrosif",),
            limitations=(),
            out_of_role=True,
        )

        case = detect_uncertainty(observation=obs, retrieved_elements=[])
        assert case is UncertaintyCase.OUT_OF_ROLE_RISK

        payload = build_abstention_payload(case=case)
        assert payload.status is AnswerStatus.ABSTAINED
        assert "sécurité" in payload.answer.lower() or "documentaire" in payload.answer.lower()
        assert len(payload.uncertainties) >= 1
        verify_no_confidence_emitted(payload)

    def test_no_sources_yields_abstention(self) -> None:
        case = detect_uncertainty(observation=None, retrieved_elements=[])
        assert case is UncertaintyCase.NO_SOURCES

        payload = build_abstention_payload(case=case)
        assert payload.status is AnswerStatus.ABSTAINED
        assert any("source" in u.lower() or "document" in u.lower() for u in payload.uncertainties)
        verify_no_confidence_emitted(payload)

    def test_contradictory_sources_yields_clarification(self) -> None:
        e1 = make_element(
            document_id="manuel_a",
            version="2023",
            page=1,
            element_id="p1",
            kind=ElementKind.PARAGRAPH,
            text="La température maximale de sécurité est de 80 °C.",
        )
        e2 = make_element(
            document_id="procedure_b",
            version="2024",
            page=2,
            element_id="p2",
            kind=ElementKind.PARAGRAPH,
            text="La température maximale admissible est de 95 °C.",
        )
        hits = [
            RetrievedElement(element=e1, score=0.9, channel="text", rank=1),
            RetrievedElement(element=e2, score=0.88, channel="text", rank=2),
        ]

        case = detect_uncertainty(
            observation=None,
            retrieved_elements=hits,
            detected_contradiction="Seuil 80 °C (manuel_a) vs 95 °C (procedure_b)",
        )
        assert case is UncertaintyCase.CONTRADICTORY_SOURCES

        payload = build_clarification_payload(
            case=case,
            contradiction_detail="manuel_a p.1 (80 °C) vs procedure_b p.2 (95 °C)",
        )
        assert payload.status is AnswerStatus.NEEDS_CLARIFICATION
        assert payload.clarification_question is not None
        assert len(payload.uncertainties) >= 1
        verify_no_confidence_emitted(payload)

    def test_never_emit_confidence_percentage(self) -> None:
        # A payload containing a fabricated confidence string in uncertainties must raise
        bad_payload = AnswerPayload(
            status=AnswerStatus.ABSTAINED,
            answer="Je ne sais pas.",
            uncertainties=("Indice de confiance du modèle : 87 %",),
        )

        with pytest.raises(ValueError, match="confidence"):
            verify_no_confidence_emitted(bad_payload)
