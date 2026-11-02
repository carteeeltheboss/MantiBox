"""Structured output validation with bounded retry (Issue #18, SPEC §7, §9).

Enforces:
- SPEC §7: Model outputs are strictly validated against Pydantic schemas.
- SPEC §9: Bounded retries strictly counted against the 6-model-call budget.
- AGENTS.md NEVER #1: Logging captures technical metadata and schema errors only,
  never raw document or model text.
- Fallback safe abstention when outputs fail validation after exhausting retries.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from maintidoc.observability import get_logger
from maintidoc.prompts.loader import load_prompt
from maintidoc.schemas import AnswerPayload, AnswerStatus, RunProvenance
from maintidoc.vlm.base import ImageInput, OutputSchema, VlmProvider, VlmResponse

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ValidatedOutputResult(Generic[T]):
    """Result of attempting to generate and validate a structured schema."""

    output: T | None
    last_response: VlmResponse | None
    calls_made: int
    validation_errors: tuple[str, ...]


def generate_validated_output(
    *,
    vlm: VlmProvider,
    schema: type[T],
    instruction: str,
    images: Sequence[ImageInput] = (),
    expects: OutputSchema = OutputSchema.TEXT,
    max_retries: int = 2,
    budget_remaining: int = 6,
    retry_prompt_name: str = "retry_format",
    retry_prompt_version: int = 1,
) -> ValidatedOutputResult[T]:
    """Generate model response with bounded retries against schema validation errors.

    The total number of calls made will never exceed min(max_retries + 1, budget_remaining).
    """
    if budget_remaining <= 0:
        logger.warning("Structured output generation halted: 0 model calls remaining in budget")
        return ValidatedOutputResult(
            output=None,
            last_response=None,
            calls_made=0,
            validation_errors=("Budget d'appels modèle épuisé (0 restant)",),
        )

    current_instruction = instruction
    calls_made = 0
    errors: list[str] = []
    last_resp: VlmResponse | None = None
    max_attempts = min(max_retries + 1, budget_remaining)

    prompt = load_prompt(retry_prompt_name, version=retry_prompt_version)

    for attempt in range(max_attempts):
        calls_made += 1
        logger.info(
            "Calling VLM for structured output",
            extra={
                "attempt": attempt + 1,
                "max_attempts": max_attempts,
                "schema": schema.__name__,
                "expects": expects.value,
            },
        )

        vlm_resp = vlm.generate(
            instruction=current_instruction,
            images=images,
            expects=expects,
        )
        last_resp = vlm_resp
        raw_text = vlm_resp.text.strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if len(lines) >= 2 and lines[-1].strip() == "```":
                raw_text = "\n".join(lines[1:-1]).strip()

        try:
            parsed = schema.model_validate_json(raw_text)
            logger.info(
                "Structured output successfully validated",
                extra={"attempt": attempt + 1, "schema": schema.__name__},
            )
            return ValidatedOutputResult(
                output=parsed,
                last_response=last_resp,
                calls_made=calls_made,
                validation_errors=tuple(errors),
            )
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            err_msg = f"Erreur de validation {schema.__name__}: {exc.__class__.__name__}"
            if isinstance(exc, ValidationError):
                # Include sanitized field error summary (no user data)
                field_errors = [
                    f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                    for err in exc.errors()
                ]
                err_msg += f" ({'; '.join(field_errors)})"
            errors.append(err_msg)
            logger.warning(
                "Structured output validation failed",
                extra={
                    "attempt": attempt + 1,
                    "schema": schema.__name__,
                    "error_type": exc.__class__.__name__,
                },
            )

            # If more attempts remain, format retry prompt using versioned template
            if attempt + 1 < max_attempts:
                current_instruction = prompt.format(
                    original_instruction=instruction,
                    previous_output=raw_text,
                    error_message=err_msg,
                )

    return ValidatedOutputResult(
        output=None,
        last_response=last_resp,
        calls_made=calls_made,
        validation_errors=tuple(errors),
    )


def create_validation_fallback_payload(
    errors: Sequence[str],
    *,
    provenance: RunProvenance | None = None,
) -> AnswerPayload:
    """Construct a safe abstention AnswerPayload when validation fails after retries."""
    uncertainties = tuple(errors) if errors else ("Format de réponse du modèle non valide",)
    return AnswerPayload(
        status=AnswerStatus.ABSTAINED,
        answer=(
            "Le modèle n'a pas pu produire une réponse au format requis après plusieurs tentatives."
        ),
        citations=(),
        uncertainties=uncertainties,
        clarification_question=None,
        proposed_action=None,
        provenance=provenance,
    )


__all__ = [
    "ValidatedOutputResult",
    "create_validation_fallback_payload",
    "generate_validated_output",
]
