"""Citation verification against retrieved elements and evidence (Issue #19, SPEC §4.2, §10).

Enforces:
- AGENTS.md NEVER #5: Citations must come from actually-retrieved elements
  and must support the claim.
- SPEC §4.2: Evidence check evaluates candidate citations against retrieved document text.
- SPEC §10: Citation validity metric calculation for evaluation and monitoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from maintidoc.agent.validation import generate_validated_output
from maintidoc.domain import RetrievedElement
from maintidoc.observability import get_logger
from maintidoc.prompts.loader import load_prompt
from maintidoc.schemas import AnswerPayload, Citation
from maintidoc.vlm.base import EvidenceVerdict, OutputSchema, VlmProvider

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """Detailed verification verdict for one candidate citation."""

    citation: Citation
    matched_element: bool
    supported: bool
    reason: str
    element: RetrievedElement | None = None


@dataclass(frozen=True, slots=True)
class CitationVerificationReport:
    """Summary of citation verification and validity metrics."""

    total_citations: int
    verified_citations: tuple[Citation, ...]
    rejected_citations: tuple[Citation, ...]
    checks: tuple[CitationCheck, ...]
    validity_rate: float
    model_calls_used: int = 0


def verify_citations_against_evidence(
    *,
    citations: Sequence[Citation],
    retrieved_elements: Sequence[RetrievedElement],
    claim: str,
    vlm_provider: VlmProvider | None = None,
    budget_remaining: int = 6,
    prompt_name: str = "evidence_check",
    prompt_version: int = 1,
) -> CitationVerificationReport:
    """Verify citations against retrieved elements and evaluate whether they support the claim.

    Any unretrieved or unsupported citations are rejected.
    """
    if not citations:
        return CitationVerificationReport(
            total_citations=0,
            verified_citations=(),
            rejected_citations=(),
            checks=(),
            validity_rate=1.0,
            model_calls_used=0,
        )

    # Build lookup by (document_id, element_id, page)
    lookup: dict[tuple[str, str, int], RetrievedElement] = {
        (re.document_id, re.element_id, re.page): re for re in retrieved_elements
    }

    verified: list[Citation] = []
    rejected: list[Citation] = []
    checks: list[CitationCheck] = []
    calls_used = 0
    remaining_budget = budget_remaining

    evidence_prompt = None
    if vlm_provider is not None:
        try:
            evidence_prompt = load_prompt(prompt_name, version=prompt_version)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load evidence check prompt: %s", exc)

    for c in citations:
        key = (c.document_id, c.element_id, c.page)
        retrieved = lookup.get(key)

        if retrieved is None:
            logger.warning(
                "Citation rejected: element not present in retrieved set",
                extra={
                    "document_id": c.document_id,
                    "element_id": c.element_id,
                    "page": c.page,
                },
            )
            rejected.append(c)
            checks.append(
                CitationCheck(
                    citation=c,
                    matched_element=False,
                    supported=False,
                    reason="Element not found in retrieved sources",
                    element=None,
                )
            )
            continue

        # Element was retrieved. If VLM provider is available and budget allows, run evidence check
        supported = True
        reason = "Element exists in retrieved sources"

        if vlm_provider is not None and evidence_prompt is not None and remaining_budget > 0:
            evidence_text = retrieved.element.text
            instruction = evidence_prompt.format(claim=claim, evidence=evidence_text)

            res = generate_validated_output(
                vlm=vlm_provider,
                schema=EvidenceVerdict,
                instruction=instruction,
                expects=OutputSchema.EVIDENCE_VERDICT,
                max_retries=1,
                budget_remaining=remaining_budget,
            )
            calls_used += res.calls_made
            remaining_budget = max(0, remaining_budget - res.calls_made)

            if res.output is not None:
                supported = res.output.supported
                reason = res.output.reason
            else:
                # If validation failed after retry, be conservative: require claim/quote match
                supported = bool(c.quote and c.quote.lower() in retrieved.element.text.lower())
                reason = (
                    "Vérification par modèle indisponible; validation par correspondance exacte"
                )

        if supported:
            enriched = Citation(
                document_id=c.document_id,
                page=c.page,
                element_id=c.element_id,
                document_version=retrieved.element.document_version,
                kind=retrieved.element.kind,
                quote=c.quote or retrieved.element.text[:400],
            )
            verified.append(enriched)
            checks.append(
                CitationCheck(
                    citation=enriched,
                    matched_element=True,
                    supported=True,
                    reason=reason,
                    element=retrieved,
                )
            )
        else:
            logger.warning(
                "Citation rejected: retrieved element does not support the claim",
                extra={
                    "document_id": c.document_id,
                    "element_id": c.element_id,
                    "page": c.page,
                },
            )
            rejected.append(c)
            checks.append(
                CitationCheck(
                    citation=c,
                    matched_element=True,
                    supported=False,
                    reason=reason,
                    element=retrieved,
                )
            )

    total = len(citations)
    validity_rate = len(verified) / total if total > 0 else 1.0

    return CitationVerificationReport(
        total_citations=total,
        verified_citations=tuple(verified),
        rejected_citations=tuple(rejected),
        checks=tuple(checks),
        validity_rate=validity_rate,
        model_calls_used=calls_used,
    )


def calculate_citation_validity(
    payload: AnswerPayload,
    retrieved_elements: Sequence[RetrievedElement],
    *,
    vlm_provider: VlmProvider | None = None,
) -> CitationVerificationReport:
    """Calculate the citation validity metric for an AnswerPayload against retrieved evidence."""
    return verify_citations_against_evidence(
        citations=payload.citations,
        retrieved_elements=retrieved_elements,
        claim=payload.answer,
        vlm_provider=vlm_provider,
    )


__all__ = [
    "CitationCheck",
    "CitationVerificationReport",
    "calculate_citation_validity",
    "verify_citations_against_evidence",
]
