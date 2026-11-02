"""Vertical slice RAG pipeline (Issue #9, SPEC §4.2, §7, §8).

End-to-end question answering pipeline wiring:
Question -> HybridRetriever -> render_sources -> VlmProvider -> citation check -> AnswerPayload.

Enforces:
- AGENTS.md NEVER #5: Citations must come from actually-retrieved elements.
- SPEC §7: Result is validated strictly against AnswerPayload.
- SPEC §8: Abstention when no retrieved evidence supports the claim.
"""

from __future__ import annotations

from maintidoc.agent.citations import verify_citations_against_evidence
from maintidoc.agent.drafts import ModelAnswerDraft
from maintidoc.agent.uncertainty import (
    UncertaintyCase,
    build_abstention_payload,
    verify_no_confidence_emitted,
)
from maintidoc.agent.validation import (
    create_validation_fallback_payload,
    generate_validated_output,
)
from maintidoc.domain import RetrievedElement
from maintidoc.observability import get_logger
from maintidoc.prompts.loader import load_prompt
from maintidoc.retrieval.fusion import HybridRetriever
from maintidoc.retrieval.index import INDEX_VERSION
from maintidoc.retrieval.render import render_sources
from maintidoc.schemas import AnswerPayload, AnswerStatus, Citation, RunProvenance
from maintidoc.security.access import Principal
from maintidoc.vlm.base import ImageInput, OutputSchema, VlmProvider

logger = get_logger(__name__)


def verify_citations(
    draft_citations: list[Citation],
    retrieved_elements: list[RetrievedElement],
) -> list[Citation]:
    """Filter and verify citations against actually-retrieved elements (AGENTS.md NEVER #5).

    Only citations matching a retrieved (document_id, element_id, page) are preserved.
    Metadata (version, kind) is enriched from the retrieved element.
    """
    report = verify_citations_against_evidence(
        citations=draft_citations,
        retrieved_elements=retrieved_elements,
        claim="",
        vlm_provider=None,
    )
    return list(report.verified_citations)


class AnswerPipeline:
    """Vertical slice RAG pipeline producing strictly validated AnswerPayloads."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        vlm_provider: VlmProvider,
        prompt_name: str = "answer",
        prompt_version: int = 1,
    ) -> None:
        self.retriever = retriever
        self.vlm_provider = vlm_provider
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version

    def answer_question(
        self,
        question: str,
        *,
        principal: Principal,
        image_bytes: bytes | None = None,
        top_k: int = 8,
        budget_remaining: int = 6,
    ) -> AnswerPayload:
        """Run the end-to-end question answering pipeline with validated bounded retry."""
        # 1. Access-filtered hybrid retrieval (SPEC §4.2)
        hits = self.retriever.search(
            query=question,
            principal=principal,
            query_image=image_bytes,
            top_n=top_k,
        )

        # Check uncertainty condition: no sources found (SPEC §8)
        if not hits:
            provenance = RunProvenance(
                vlm_model=self.vlm_provider.model_id,
                vlm_provider=getattr(self.vlm_provider, "provider", "vlm"),
                vlm_device=self.vlm_provider.device,
                prompt_versions={self.prompt_name: f"v{self.prompt_version}"},
                index_version=INDEX_VERSION,
                model_calls_used=0,
            )
            return build_abstention_payload(
                case=UncertaintyCase.NO_SOURCES,
                provenance=provenance,
            )

        # 2. Render retrieved sources into the untrusted prompt section
        sources_text = render_sources(hits)

        # 3. Format versioned prompt
        prompt = load_prompt(self.prompt_name, version=self.prompt_version)
        rendered_prompt = prompt.format(
            question=question,
            sources=sources_text,
            visual_analysis="",
        )

        # 4. Generate answer through VLM provider with bounded validation retry (SPEC §7, §9)
        images = (ImageInput(data=image_bytes, media_type="image/png"),) if image_bytes else ()
        validation_res = generate_validated_output(
            vlm=self.vlm_provider,
            schema=ModelAnswerDraft,
            instruction=rendered_prompt,
            images=images,
            expects=OutputSchema.ANSWER,
            max_retries=2,
            budget_remaining=budget_remaining,
        )

        model_calls_used = validation_res.calls_made
        remaining_budget = max(0, budget_remaining - model_calls_used)

        provenance = RunProvenance(
            vlm_model=getattr(validation_res.last_response, "model_id", self.vlm_provider.model_id),
            vlm_provider=getattr(self.vlm_provider, "provider", "vlm"),
            vlm_device=getattr(validation_res.last_response, "device", self.vlm_provider.device),
            fallback_applied=getattr(validation_res.last_response, "fallback_applied", False),
            prompt_versions={self.prompt_name: f"v{prompt.version}"},
            reranker=getattr(self.retriever.reranker, "model_id", None),
            index_version=INDEX_VERSION,
            latency_ms=getattr(validation_res.last_response, "latency_ms", None),
            model_calls_used=model_calls_used,
        )

        # If validation completely failed after retries, return safe abstention
        if validation_res.output is None:
            return create_validation_fallback_payload(
                validation_res.validation_errors,
                provenance=provenance,
            )

        draft = validation_res.output

        # 5. Verify citations against evidence (SPEC §4.2, AGENTS.md NEVER #5)
        # We verify evidence using the VLM if remaining budget allows
        citation_report = verify_citations_against_evidence(
            citations=draft.citations,
            retrieved_elements=hits,
            claim=draft.answer,
            vlm_provider=self.vlm_provider if remaining_budget > 0 else None,
            budget_remaining=remaining_budget,
        )
        model_calls_used += citation_report.model_calls_used
        provenance = RunProvenance(
            vlm_model=provenance.vlm_model,
            vlm_provider=provenance.vlm_provider,
            vlm_device=provenance.vlm_device,
            fallback_applied=provenance.fallback_applied,
            prompt_versions=provenance.prompt_versions,
            reranker=provenance.reranker,
            index_version=provenance.index_version,
            latency_ms=provenance.latency_ms,
            model_calls_used=model_calls_used,
        )

        verified_citations = list(citation_report.verified_citations)
        status = draft.status
        answer = draft.answer
        uncertainties = list(draft.uncertainties)

        # Invariant: answered requires at least one verified citation
        if status is AnswerStatus.ANSWERED and not verified_citations:
            logger.info(
                "Draft was answered but had no verified citations; downgrading to abstained"
            )
            status = AnswerStatus.ABSTAINED
            answer = "Aucune source vérifiée ne permet de répondre à cette question."
            uncertainties.append("Absence de référence documentaire vérifiée")

        # Invariant: abstained requires at least one uncertainty
        if status is AnswerStatus.ABSTAINED and not uncertainties:
            uncertainties.append("Sources documentaires insuffisantes")

        payload = AnswerPayload(
            status=status,
            answer=answer,
            citations=tuple(verified_citations),
            uncertainties=tuple(uncertainties),
            clarification_question=draft.clarification_question,
            proposed_action=None,
            provenance=provenance,
        )

        # Enforce NEVER #6: no model-invented confidence
        verify_no_confidence_emitted(payload)

        return payload


__all__ = [
    "AnswerPipeline",
    "verify_citations",
]
