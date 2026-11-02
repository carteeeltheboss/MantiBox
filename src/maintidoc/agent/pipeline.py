"""Vertical slice RAG pipeline (Issue #9, SPEC §4.2, §7, §8).

End-to-end question answering pipeline wiring:
Question -> HybridRetriever -> render_sources -> VlmProvider -> citation check -> AnswerPayload.

Enforces:
- AGENTS.md NEVER #5: Citations must come from actually-retrieved elements.
- SPEC §7: Result is validated strictly against AnswerPayload.
- SPEC §8: Abstention when no retrieved evidence supports the claim.
"""

from __future__ import annotations

from maintidoc.agent.drafts import ModelAnswerDraft
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
    lookup: dict[tuple[str, str, int], RetrievedElement] = {
        (re.document_id, re.element_id, re.page): re for re in retrieved_elements
    }

    verified: list[Citation] = []
    for c in draft_citations:
        key = (c.document_id, c.element_id, c.page)
        if key in lookup:
            retrieved = lookup[key]
            verified.append(
                Citation(
                    document_id=c.document_id,
                    page=c.page,
                    element_id=c.element_id,
                    document_version=retrieved.element.document_version,
                    kind=retrieved.element.kind,
                    quote=c.quote,
                )
            )
        else:
            logger.warning(
                "Stripped unretrieved citation from draft",
                extra={
                    "document_id": c.document_id,
                    "element_id": c.element_id,
                    "page": c.page,
                },
            )
    return verified


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
    ) -> AnswerPayload:
        """Run the end-to-end question answering pipeline."""
        # 1. Access-filtered hybrid retrieval (SPEC §4.2)
        hits = self.retriever.search(
            query=question,
            principal=principal,
            query_image=image_bytes,
            top_n=top_k,
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

        # 4. Generate answer through VLM provider
        images = (ImageInput(data=image_bytes, media_type="image/png"),) if image_bytes else ()
        vlm_response = self.vlm_provider.generate(
            instruction=rendered_prompt,
            images=images,
            expects=OutputSchema.ANSWER,
        )

        # 5. Parse lenient model draft
        try:
            draft = ModelAnswerDraft.model_validate_json(vlm_response.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model draft parsing failed, abstaining: %s", exc)
            draft = ModelAnswerDraft(
                status=AnswerStatus.ABSTAINED,
                answer="Erreur de format de réponse du modèle.",
                uncertainties=["Format JSON de sortie non valide"],
            )

        # 6. Verify citations (AGENTS.md NEVER #5) and enforce status invariants (SPEC §7, §8)
        verified_citations = verify_citations(draft.citations, hits)

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

        # 7. Construct Provenance
        provenance = RunProvenance(
            vlm_model=vlm_response.model_id,
            vlm_provider=vlm_response.provider,
            vlm_device=vlm_response.device,
            fallback_applied=vlm_response.fallback_applied,
            prompt_versions={self.prompt_name: f"v{prompt.version}"},
            reranker=getattr(self.retriever.reranker, "model_id", None),
            index_version=INDEX_VERSION,
            latency_ms=vlm_response.latency_ms,
            model_calls_used=1,
        )

        # 8. Return strict AnswerPayload
        return AnswerPayload(
            status=status,
            answer=answer,
            citations=tuple(verified_citations),
            uncertainties=tuple(uncertainties),
            clarification_question=draft.clarification_question,
            proposed_action=None,
            provenance=provenance,
        )


__all__ = [
    "AnswerPipeline",
    "verify_citations",
]
