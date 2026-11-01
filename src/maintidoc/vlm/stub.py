"""Deterministic VLM provider for tests, CI and the portable CPU deployment.

**This is not a model.** It performs no inference and understands no image. It
exists so that:

* the full pipeline — retrieval, fusion, citation verification, budgets, MCP,
  approval, API — is testable and runnable with no torch, no GPU and no weights;
* `docker compose up` works on a machine with no accelerator at all.

Three rules keep it honest:

1. Every answer it produces is prefixed `[STUB]`, and `provider` is `"stub"` in
   the response and in `AnswerPayload.provenance`, so a stub answer can never be
   mistaken for a model answer downstream or in a demo.
2. It may only cite sources that were actually present in the rendered context.
   It parses them back out with `parse_source_refs` instead of inventing
   identifiers (AGENTS.md NEVER #5). With no sources available it abstains,
   exactly as the real path must.
3. It dispatches on the caller-declared `expects` schema, never on the content of
   the prompt. Content sniffing was tried first and is unsound: the `answer`
   prompt embeds a rendered `visual_analysis` payload, so its text legitimately
   contains another schema's field names and the heuristic picks the wrong
   contract.

`create_vlm_provider` refuses to build a stub when `MAINTIDOC_ENV=prod`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from maintidoc.retrieval.render import SourceRef, extract_section, parse_source_refs
from maintidoc.vlm.base import ImageInput, OutputSchema, VlmResponse

STUB_MODEL_ID = "maintidoc-stub-vlm"

_STUB_PREFIX = "[STUB] "

#: Tags that may hold a rendered source block, tried in order. `answer` uses
#: <sources>; `evidence_check` and `ticket_draft` use <evidence>.
_SOURCE_TAGS = ("sources", "evidence")

Responder = Callable[[str, Sequence[ImageInput]], str]


class StubPolicy(StrEnum):
    """Which answer status the stub should produce.

    `AUTO` decides from the context: with retrievable sources it answers, without
    them it abstains. The explicit members let the SPEC §14 demo show each branch
    — nominal answer, clarification, abstention and ticket approval — without
    needing a real model.
    """

    AUTO = "auto"
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    ABSTAINED = "abstained"
    NEEDS_APPROVAL = "needs_approval"


@dataclass(slots=True)
class StubVlmProvider:
    """Deterministic stand-in implementing `VlmProvider`."""

    policy: StubPolicy = StubPolicy.AUTO
    #: Optional injected responder, for tests that need an exact model output.
    responder: Responder | None = None
    #: Method names in call order, so tests can assert the budget accounting.
    calls: list[str] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return STUB_MODEL_ID

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def provider(self) -> str:
        return "stub"

    @property
    def fallback_applied(self) -> bool:
        return False

    def describe_image(
        self,
        *,
        instruction: str,
        images: Sequence[ImageInput] = (),
    ) -> VlmResponse:
        return self._timed("describe_image", lambda: _dumps(self._describe(images)))

    def generate(
        self,
        *,
        instruction: str,
        images: Sequence[ImageInput] = (),
        max_new_tokens: int | None = None,
        expects: OutputSchema = OutputSchema.TEXT,
    ) -> VlmResponse:
        if self.responder is not None:
            responder = self.responder
            return self._timed("generate", lambda: responder(instruction, images))
        return self._timed("generate", lambda: self._render(expects, instruction, images))

    def close(self) -> None:
        """Nothing to release."""

    # ------------------------------------------------------------------ internals

    def _timed(self, method: str, produce: Callable[[], str]) -> VlmResponse:
        self.calls.append(method)
        started = time.monotonic()
        text = produce()
        elapsed_ms = max(int((time.monotonic() - started) * 1000), 0)
        return VlmResponse(
            text=text,
            model_id=STUB_MODEL_ID,
            device="cpu",
            provider="stub",
            fallback_applied=False,
            latency_ms=elapsed_ms,
            finish_reason="stub",
            reasons=("deterministic stub provider: no inference performed",),
        )

    def _render(
        self,
        expects: OutputSchema,
        instruction: str,
        images: Sequence[ImageInput],
    ) -> str:
        match expects:
            case OutputSchema.VISUAL_OBSERVATION:
                return _dumps(self._describe(images))
            case OutputSchema.EVIDENCE_VERDICT:
                return _dumps(
                    {
                        "supported": self.policy is not StubPolicy.ABSTAINED,
                        "reason": (
                            f"{_STUB_PREFIX}vérification de référence non effectuée : "
                            "fournisseur déterministe"
                        ),
                    }
                )
            case OutputSchema.TICKET_DRAFT:
                return _dumps(self._ticket_draft(_refs_from(instruction)))
            case OutputSchema.ANSWER:
                return _dumps(self._answer(_refs_from(instruction)))
            case OutputSchema.TEXT:
                return f"{_STUB_PREFIX}texte déterministe, aucune inférence effectuée."

    def _describe(self, images: Sequence[ImageInput]) -> dict[str, object]:
        if not images:
            return {
                "image_readable": False,
                "equipment_identified": None,
                "observations": [],
                "limitations": [f"{_STUB_PREFIX}aucune image fournie avec la demande d'analyse"],
                "out_of_role": False,
            }
        total = sum(len(image.data) for image in images)
        return {
            "image_readable": True,
            # Never a guessed nameplate reference: SPEC §8 forbids treating a
            # visual resemblance as an identification.
            "equipment_identified": None,
            "observations": [
                f"{_STUB_PREFIX}{len(images)} image(s) reçue(s), {total} octets au total"
            ],
            "limitations": [
                f"{_STUB_PREFIX}ce fournisseur n'effectue aucune analyse visuelle réelle"
            ],
            "out_of_role": False,
        }

    def _ticket_draft(self, refs: tuple[SourceRef, ...]) -> dict[str, object]:
        reference = (
            f" Références : {refs[0].document_id} p.{refs[0].page} {refs[0].element_id}."
            if refs
            else " Aucune référence documentaire disponible."
        )
        return {
            "title": f"{_STUB_PREFIX}Ticket à valider",
            "description": (
                f"{_STUB_PREFIX}Brouillon produit par le fournisseur déterministe."
                f"{reference} Ce contenu doit être approuvé par un humain avant création."
            ),
        }

    def _answer(self, refs: tuple[SourceRef, ...]) -> dict[str, object]:
        policy = self.policy
        if policy is StubPolicy.AUTO:
            policy = StubPolicy.ANSWERED if refs else StubPolicy.ABSTAINED

        # Even when ANSWERED was requested explicitly, citing nothing is not an
        # option: an "answered" payload without a citation fails the contract.
        if policy is StubPolicy.ABSTAINED or not refs:
            return {
                "status": "abstained",
                "answer": f"{_STUB_PREFIX}aucune source récupérée ne permet de répondre.",
                "citations": [],
                "uncertainties": [
                    f"{_STUB_PREFIX}aucun élément récupéré disponible dans le contexte"
                ],
                "clarification_question": None,
                "proposed_action": None,
            }

        if policy is StubPolicy.NEEDS_CLARIFICATION:
            return {
                "status": "needs_clarification",
                "answer": f"{_STUB_PREFIX}la demande doit être précisée.",
                "citations": [],
                "uncertainties": [f"{_STUB_PREFIX}équipement non identifié"],
                "clarification_question": (
                    f"{_STUB_PREFIX}pouvez-vous photographier la plaque signalétique ?"
                ),
                "proposed_action": None,
            }

        if policy is StubPolicy.NEEDS_APPROVAL:
            # The content hash and idempotency key are computed by the workflow
            # from the approved content, never by the provider, so the stub only
            # signals intent and supplies the supporting evidence.
            return {
                "status": "needs_approval",
                "answer": (
                    f"{_STUB_PREFIX}une action sensible est proposée et attend votre approbation."
                ),
                "citations": [_citation(refs[0])],
                "uncertainties": [f"{_STUB_PREFIX}brouillon non produit par un modèle"],
                "clarification_question": None,
                "proposed_action": None,
            }

        first = refs[0]
        return {
            "status": "answered",
            "answer": (
                f"{_STUB_PREFIX}réponse déterministe construite à partir de la source "
                f"[{first.index}] ({first.document_id} p.{first.page}). Aucun modèle de "
                "vision-langage n'a été consulté."
            ),
            "citations": [_citation(first)],
            "uncertainties": [],
            "clarification_question": None,
            "proposed_action": None,
        }


def _refs_from(instruction: str) -> tuple[SourceRef, ...]:
    """Parse source references out of whichever tag the prompt used."""
    for tag in _SOURCE_TAGS:
        refs = parse_source_refs(extract_section(instruction, tag))
        if refs:
            return refs
    return ()


def _citation(ref: SourceRef) -> dict[str, object]:
    return {
        "document_id": ref.document_id,
        "page": ref.page,
        "element_id": ref.element_id,
        "document_version": ref.document_version,
        "kind": ref.kind,
    }


def _dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False)


__all__ = ["STUB_MODEL_ID", "Responder", "StubPolicy", "StubVlmProvider"]
