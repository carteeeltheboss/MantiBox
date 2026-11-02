"""Tests for structured output validation with bounded retry (Issue #18, SPEC §7, §9).

Acceptance criteria:
- All model outputs validated against the schema
- Retries bounded and counted against the 6-model-call budget
- Invalid output after retries returns a safe abstention
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, Field

from maintidoc.agent.validation import generate_validated_output
from maintidoc.vlm.base import ImageInput, OutputSchema
from maintidoc.vlm.stub import StubVlmProvider


class SampleContract(BaseModel):
    title: str = Field(min_length=1)
    count: int = Field(ge=0)


class TestStructuredOutputValidation:
    def test_first_attempt_valid_uses_single_call(self) -> None:
        calls = 0

        def responder(instruction: str, images: Sequence[ImageInput]) -> str:
            nonlocal calls
            calls += 1
            return json.dumps({"title": "pompe", "count": 2})

        vlm = StubVlmProvider(responder=responder)
        result = generate_validated_output(
            vlm=vlm,
            schema=SampleContract,
            instruction="Donne le résultat",
            expects=OutputSchema.TEXT,
            max_retries=2,
            budget_remaining=6,
        )

        assert result.output is not None
        assert result.output.title == "pompe"
        assert result.output.count == 2
        assert result.calls_made == 1
        assert len(result.validation_errors) == 0

    def test_malformed_json_retries_and_succeeds(self) -> None:
        calls = 0

        def responder(instruction: str, images: Sequence[ImageInput]) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                return "Not a valid JSON at all!"
            return json.dumps({"title": "pompe", "count": 1})

        vlm = StubVlmProvider(responder=responder)
        result = generate_validated_output(
            vlm=vlm,
            schema=SampleContract,
            instruction="Donne le résultat",
            expects=OutputSchema.TEXT,
            max_retries=2,
            budget_remaining=6,
        )

        assert result.output is not None
        assert result.output.title == "pompe"
        assert result.calls_made == 2
        assert len(result.validation_errors) == 1

    def test_schema_error_retries_and_succeeds(self) -> None:
        calls = 0

        def responder(instruction: str, images: Sequence[ImageInput]) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                # count is negative, violating ge=0
                return json.dumps({"title": "pompe", "count": -5})
            return json.dumps({"title": "pompe", "count": 5})

        vlm = StubVlmProvider(responder=responder)
        result = generate_validated_output(
            vlm=vlm,
            schema=SampleContract,
            instruction="Donne le résultat",
            expects=OutputSchema.TEXT,
            max_retries=2,
            budget_remaining=6,
        )

        assert result.output is not None
        assert result.output.count == 5
        assert result.calls_made == 2
        assert len(result.validation_errors) == 1

    def test_exhausted_retries_returns_none(self) -> None:
        calls = 0

        def responder(instruction: str, images: Sequence[ImageInput]) -> str:
            nonlocal calls
            calls += 1
            return "Persistent garbage output"

        vlm = StubVlmProvider(responder=responder)
        result = generate_validated_output(
            vlm=vlm,
            schema=SampleContract,
            instruction="Donne le résultat",
            expects=OutputSchema.TEXT,
            max_retries=2,
            budget_remaining=6,
        )

        assert result.output is None
        assert result.calls_made == 3  # initial + 2 retries
        assert len(result.validation_errors) == 3

    def test_retries_strictly_bounded_by_budget(self) -> None:
        calls = 0

        def responder(instruction: str, images: Sequence[ImageInput]) -> str:
            nonlocal calls
            calls += 1
            return "Bad JSON"

        vlm = StubVlmProvider(responder=responder)
        # Budget only has 1 call remaining even if max_retries is 3
        result = generate_validated_output(
            vlm=vlm,
            schema=SampleContract,
            instruction="Donne le résultat",
            expects=OutputSchema.TEXT,
            max_retries=3,
            budget_remaining=1,
        )

        assert result.output is None
        assert result.calls_made == 1
        assert len(result.validation_errors) == 1

    def test_zero_budget_makes_no_calls(self) -> None:
        calls = 0

        def responder(instruction: str, images: Sequence[ImageInput]) -> str:
            nonlocal calls
            calls += 1
            return "{}"

        vlm = StubVlmProvider(responder=responder)
        result = generate_validated_output(
            vlm=vlm,
            schema=SampleContract,
            instruction="Donne le résultat",
            expects=OutputSchema.TEXT,
            max_retries=2,
            budget_remaining=0,
        )

        assert result.output is None
        assert result.calls_made == 0
        assert "Budget" in result.validation_errors[0]
