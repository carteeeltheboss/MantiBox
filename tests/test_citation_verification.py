"""Tests for citation verification against retrieved evidence (Issue #19, SPEC §4.2, §10).

Acceptance criteria:
- Every citation matches a retrieved page/element
- Evidence check verifies the cited element supports the claim
- Unsupported citations removed or answer downgraded
- Citation validity metric implemented
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from maintidoc.agent.citations import (
    CitationVerificationReport,
    calculate_citation_validity,
    verify_citations_against_evidence,
)
from maintidoc.domain import RetrievedElement
from maintidoc.schemas import AnswerPayload, AnswerStatus, Citation, ElementKind
from maintidoc.vlm.base import ImageInput
from maintidoc.vlm.stub import StubPolicy, StubVlmProvider
from tests.factories import make_element


@pytest.fixture
def retrieved_docs() -> list[RetrievedElement]:
    e1 = make_element(
        document_id="manuel_pompe_a",
        version="v1",
        page=2,
        element_id="p_alarm",
        kind=ElementKind.PARAGRAPH,
        text="Le voyant orange indique une surchauffe au-delà de 85 °C.",
    )
    e2 = make_element(
        document_id="manuel_pompe_a",
        version="v1",
        page=5,
        element_id="p_press",
        kind=ElementKind.PARAGRAPH,
        text="La pression nominale de fonctionnement est de 6 bars.",
    )
    return [
        RetrievedElement(element=e1, score=0.9, channel="text", rank=1),
        RetrievedElement(element=e2, score=0.8, channel="text", rank=2),
    ]


class TestCitationVerification:
    def test_unretrieved_citations_are_rejected(
        self, retrieved_docs: list[RetrievedElement]
    ) -> None:
        citations = [
            Citation(document_id="manuel_pompe_a", page=2, element_id="p_alarm"),
            Citation(document_id="doc_inexistant", page=1, element_id="p_ghost"),
        ]

        report = verify_citations_against_evidence(
            citations=citations,
            retrieved_elements=retrieved_docs,
            claim="Le voyant orange indique une surchauffe.",
            vlm_provider=None,
        )

        assert isinstance(report, CitationVerificationReport)
        assert len(report.verified_citations) == 1
        assert report.verified_citations[0].element_id == "p_alarm"
        assert len(report.rejected_citations) == 1
        assert report.rejected_citations[0].element_id == "p_ghost"
        assert report.validity_rate == 0.5

    def test_evidence_verdict_supported_preserves_citation(
        self, retrieved_docs: list[RetrievedElement]
    ) -> None:
        def responder(instruction: str, images: Sequence[ImageInput]) -> str:
            return json.dumps(
                {
                    "supported": True,
                    "reason": "La preuve confirme explicitement la surchauffe.",
                }
            )

        vlm = StubVlmProvider(responder=responder)
        citations = [
            Citation(document_id="manuel_pompe_a", page=2, element_id="p_alarm"),
        ]

        report = verify_citations_against_evidence(
            citations=citations,
            retrieved_elements=retrieved_docs,
            claim="Le voyant orange indique une surchauffe à 85 °C.",
            vlm_provider=vlm,
        )

        assert len(report.verified_citations) == 1
        assert report.validity_rate == 1.0
        assert report.checks[0].supported is True
        assert "surchauffe" in report.checks[0].reason

    def test_evidence_verdict_unsupported_rejects_citation(
        self, retrieved_docs: list[RetrievedElement]
    ) -> None:
        def responder(instruction: str, images: Sequence[ImageInput]) -> str:
            return json.dumps(
                {
                    "supported": False,
                    "reason": "La preuve mentionne la pression et non la température.",
                }
            )

        vlm = StubVlmProvider(responder=responder)
        citations = [
            Citation(document_id="manuel_pompe_a", page=5, element_id="p_press"),
        ]

        report = verify_citations_against_evidence(
            citations=citations,
            retrieved_elements=retrieved_docs,
            claim="La température critique est de 100 °C.",
            vlm_provider=vlm,
        )

        assert len(report.verified_citations) == 0
        assert len(report.rejected_citations) == 1
        assert report.validity_rate == 0.0
        assert report.checks[0].supported is False

    def test_citation_validity_metric_on_answer_payload(
        self, retrieved_docs: list[RetrievedElement]
    ) -> None:
        payload = AnswerPayload(
            status=AnswerStatus.ANSWERED,
            answer="La pression nominale est de 6 bars.",
            citations=(Citation(document_id="manuel_pompe_a", page=5, element_id="p_press"),),
            uncertainties=(),
        )

        vlm = StubVlmProvider(policy=StubPolicy.AUTO)
        report = calculate_citation_validity(payload, retrieved_docs, vlm_provider=vlm)

        assert report.total_citations == 1
        assert report.validity_rate == 1.0
        assert len(report.verified_citations) == 1

    def test_empty_citations_yields_validity_rate_one(
        self, retrieved_docs: list[RetrievedElement]
    ) -> None:
        payload = AnswerPayload(
            status=AnswerStatus.ABSTAINED,
            answer="Je ne sais pas.",
            citations=(),
            uncertainties=("Manque de données",),
        )

        report = calculate_citation_validity(payload, retrieved_docs)
        assert report.total_citations == 0
        assert report.validity_rate == 1.0
