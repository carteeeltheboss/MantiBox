"""Uncertainty management and abstention logic (Issue #20, SPEC §8).

Enforces:
- SPEC §8: Clarification or abstention for unreadable photos, absent sources,
  contradictory sources, or out-of-role risks.
- AGENTS.md NEVER #6: Never display a model-invented confidence percentage.
  Explicit evidence and missing information are returned instead.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

from maintidoc.domain import RetrievedElement
from maintidoc.observability import get_logger
from maintidoc.schemas import AnswerPayload, AnswerStatus, RunProvenance
from maintidoc.vlm.base import VisualObservation

logger = get_logger(__name__)

_CONFIDENCE_PATTERN = re.compile(
    r"\b(indice|taux|pourcentage|degré|score)\s+de\s+confiance\b|\b\d+[\s]*%",
    re.IGNORECASE,
)


class UncertaintyCase(StrEnum):
    """Categories requiring abstention or clarification (SPEC §8)."""

    NONE = "none"
    UNREADABLE_PHOTO = "unreadable_photo"
    OUT_OF_ROLE_RISK = "out_of_role_risk"
    NO_SOURCES = "no_sources"
    CONTRADICTORY_SOURCES = "contradictory_sources"
    UNVERIFIED_EVIDENCE = "unverified_evidence"


def detect_uncertainty(
    *,
    observation: VisualObservation | None = None,
    retrieved_elements: Sequence[RetrievedElement] = (),
    detected_contradiction: str | None = None,
) -> UncertaintyCase:
    """Evaluate whether the request triggers one of the SPEC §8 uncertainty or risk conditions."""
    if observation is not None:
        if observation.out_of_role:
            return UncertaintyCase.OUT_OF_ROLE_RISK
        if not observation.image_readable:
            return UncertaintyCase.UNREADABLE_PHOTO

    if detected_contradiction:
        return UncertaintyCase.CONTRADICTORY_SOURCES

    if not retrieved_elements:
        return UncertaintyCase.NO_SOURCES

    return UncertaintyCase.NONE


def build_clarification_payload(
    *,
    case: UncertaintyCase,
    missing_info: str | None = None,
    contradiction_detail: str | None = None,
    provenance: RunProvenance | None = None,
) -> AnswerPayload:
    """Construct an AnswerPayload with status=needs_clarification."""
    uncertainties: tuple[str, ...]
    if case is UncertaintyCase.UNREADABLE_PHOTO:
        answer = (
            "La photographie transmise est illisible ou ne permet pas "
            "d'identifier l'équipement avec certitude."
        )
        target_info = missing_info or "plaque signalétique"
        clarification_q = (
            f"Pouvez-vous fournir une photographie plus nette de l'équipement "
            f"(notamment la {target_info}) ?"
        )
        uncertainties = (
            "Photographie floue ou illisible",
            "Équipement non identifiable visuellement",
        )
    elif case is UncertaintyCase.CONTRADICTORY_SOURCES:
        answer = "Les documents techniques consultés contiennent des informations divergentes."
        detail = contradiction_detail or "valeurs divergentes"
        clarification_q = (
            f"Les sources présentent une contradiction : {detail}. "
            "Quelle version ou procédure doit faire référence ?"
        )
        uncertainties = (
            "Contradiction entre les documents de référence",
            f"Détail : {detail}",
        )
    else:
        answer = "Des informations complémentaires sont requises pour traiter cette demande."
        clarification_q = (
            f"Précisez l'élément suivant : {missing_info or 'référence de l équipement'}."
        )
        uncertainties = ("Informations incomplètes",)

    return AnswerPayload(
        status=AnswerStatus.NEEDS_CLARIFICATION,
        answer=answer,
        citations=(),
        uncertainties=uncertainties,
        clarification_question=clarification_q,
        proposed_action=None,
        provenance=provenance,
    )


def build_abstention_payload(
    *,
    case: UncertaintyCase,
    reason: str | None = None,
    provenance: RunProvenance | None = None,
) -> AnswerPayload:
    """Construct an AnswerPayload with status=abstained."""
    uncertainties: tuple[str, ...]
    if case is UncertaintyCase.OUT_OF_ROLE_RISK:
        answer = (
            "Cette situation présente un risque pour la sécurité ou dépasse le rôle "
            "documentaire de l'assistant. MaintiDoc est une aide documentaire et ne "
            "réalise aucun diagnostic garanti ni commande d'équipement."
        )
        uncertainties = (
            "Risque de sécurité ou intervention hors périmètre documentaire",
            "Intervention requérant un technicien habilité selon les consignes de sécurité",
        )
    elif case is UncertaintyCase.NO_SOURCES:
        answer = "Aucune source documentaire accessible ne permet de répondre à cette demande."
        uncertainties = ("Absence de documentation technique accessible ou autorisée",)
    else:
        answer = (
            reason
            or "Les informations disponibles sont insuffisantes pour répondre avec certitude."
        )
        uncertainties = ("Sources documentaires insuffisantes pour étayer la réponse",)

    return AnswerPayload(
        status=AnswerStatus.ABSTAINED,
        answer=answer,
        citations=(),
        uncertainties=uncertainties,
        clarification_question=None,
        proposed_action=None,
        provenance=provenance,
    )


def verify_no_confidence_emitted(payload: AnswerPayload) -> None:
    """Verify that no fabricated confidence metric is present in the payload (NEVER #6)."""
    texts_to_check = [payload.answer]
    texts_to_check.extend(payload.uncertainties)
    if payload.clarification_question:
        texts_to_check.append(payload.clarification_question)

    for text in texts_to_check:
        if match := _CONFIDENCE_PATTERN.search(text):
            msg = (
                f"Model response contains prohibited confidence token '{match.group(0)}': "
                "AGENTS.md NEVER #6 forbids model-invented confidence numbers."
            )
            raise ValueError(msg)


__all__ = [
    "UncertaintyCase",
    "build_abstention_payload",
    "build_clarification_payload",
    "detect_uncertainty",
    "verify_no_confidence_emitted",
]
