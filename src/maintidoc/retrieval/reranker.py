"""Reranker seam and arms (SPEC §5).

SPEC §5 specifies three comparative arms for the evaluation:
1. sans reranker ("none") — identity reranker preserving candidate order;
2. avec reranker d'origine ("bge") — BAAI/bge-reranker-v2-m3 base model;
3. avec reranker adapté ("adapter") — fine-tuned adapter on the domain data.

Plus a "deterministic" arm for torch-free fast tests and portable gates.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from maintidoc.domain import RetrievedElement
from maintidoc.errors import ConfigurationError
from maintidoc.observability import get_logger

logger = get_logger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

RerankerBackend = Literal["none", "deterministic", "bge", "adapter", "auto"]


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking candidate elements given a query."""

    @property
    def model_id(self) -> str: ...

    @property
    def arm(self) -> str: ...

    def rerank(
        self,
        query: str,
        elements: Sequence[RetrievedElement],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedElement]: ...


class NoReranker:
    """The 'sans reranker' baseline arm required by SPEC §5.

    Preserves candidate order exactly as retrieved from fusion or single-channel search.
    """

    @property
    def model_id(self) -> str:
        return "none"

    @property
    def arm(self) -> str:
        return "none"

    def rerank(
        self,
        query: str,
        elements: Sequence[RetrievedElement],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedElement]:
        candidates = list(elements)
        if top_n is not None and top_n > 0:
            candidates = candidates[:top_n]
        return candidates


class DeterministicReranker:
    """Deterministic, non-semantic keyword-overlap reranker for tests and offline runs."""

    def __init__(self) -> None:
        pass

    @property
    def model_id(self) -> str:
        return "deterministic-overlap"

    @property
    def arm(self) -> str:
        return "deterministic"

    def _score(self, query: str, text: str) -> float:
        if not text.strip() or not query.strip():
            return 0.0
        q_tokens = set(re.findall(r"\w+", query.lower()))
        t_tokens = set(re.findall(r"\w+", text.lower()))
        if not q_tokens:
            return 0.0
        overlap = len(q_tokens.intersection(t_tokens))
        return overlap / len(q_tokens)

    def rerank(
        self,
        query: str,
        elements: Sequence[RetrievedElement],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedElement]:
        if not elements:
            return []

        scored: list[tuple[float, RetrievedElement]] = []
        for el in elements:
            overlap = self._score(query, el.element.text)
            # Combine overlap as primary key, original score/rrf as secondary tie-breaker
            secondary = el.rrf_score if el.rrf_score is not None else el.score
            combined = overlap * 10.0 + secondary
            scored.append((combined, el))

        # Sort descending by score, tie-break by document_id and element_id
        scored.sort(
            key=lambda pair: (pair[0], pair[1].document_id, pair[1].element_id),
            reverse=True,
        )

        reranked: list[RetrievedElement] = []
        for rank_0, (score, orig) in enumerate(scored):
            updated = orig.model_copy(
                update={
                    "score": score,
                    "rank": rank_0 + 1,
                }
            )
            reranked.append(updated)

        if top_n is not None and top_n > 0:
            reranked = reranked[:top_n]
        return reranked


class BgeReranker:
    """Cross-encoder reranker wrapping BAAI/bge-reranker-v2-m3 and optional adapter.

    Covers both SPEC §5 arms:
    - 'bge' (base model)
    - 'adapter' (fine-tuned adapter applied on top)
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RERANKER_MODEL,
        *,
        adapter_path: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self._adapter_path = adapter_path
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        if self._adapter_path:
            return f"{self.model_name_or_path}+{self._adapter_path}"
        return self.model_name_or_path

    @property
    def arm(self) -> str:
        return "adapter" if self._adapter_path else "bge"

    def _ensure_loaded(self) -> tuple[Any, Any]:
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return self._tokenizer, self._model

            path = Path(self.model_name_or_path)
            # If path points to a local directory, verify it contains files
            if path.exists() and path.is_dir() and not any(path.iterdir()):
                msg = f"reranker model directory '{self.model_name_or_path}' is empty"
                raise ConfigurationError(msg)

            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                msg = "the 'ml' extra (torch, transformers) is required for BgeReranker"
                raise ConfigurationError(msg) from exc

            # Record device placement
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            try:
                logger.info(
                    "Loading reranker model",
                    extra={"model": self.model_name_or_path, "arm": self.arm},
                )
                tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)

                if self._adapter_path:
                    try:
                        from peft import PeftModel  # type: ignore[import-not-found]

                        model = PeftModel.from_pretrained(model, self._adapter_path)
                        logger.info("Loaded PEFT adapter", extra={"adapter": self._adapter_path})
                    except ImportError as exc:
                        msg = "the 'peft' package is required for adapter reranker"
                        raise ConfigurationError(msg) from exc

                model.eval()
                self._tokenizer = tokenizer
                self._model = model
                return tokenizer, model
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, ConfigurationError):
                    raise
                msg = f"failed to load reranker '{self.model_name_or_path}': {exc}"
                raise ConfigurationError(msg) from exc

    def rerank(
        self,
        query: str,
        elements: Sequence[RetrievedElement],
        *,
        top_n: int | None = None,
    ) -> list[RetrievedElement]:
        if not elements:
            return []

        tokenizer, model = self._ensure_loaded()
        import torch

        pairs = [[query, el.element.text if el.element.text.strip() else ""] for el in elements]

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            scores = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
                .tolist()
            )

        scored = list(zip(scores, elements, strict=True))
        scored.sort(
            key=lambda pair: (pair[0], pair[1].document_id, pair[1].element_id),
            reverse=True,
        )

        reranked: list[RetrievedElement] = []
        for rank_0, (score, orig) in enumerate(scored):
            reranked.append(
                orig.model_copy(
                    update={
                        "score": float(score),
                        "rank": rank_0 + 1,
                    }
                )
            )

        if top_n is not None and top_n > 0:
            reranked = reranked[:top_n]
        return reranked


def make_reranker(
    backend: RerankerBackend | str = "none",
    *,
    model_name_or_path: str | None = None,
    adapter_path: str | None = None,
) -> Reranker:
    """Factory for selecting the reranker arm."""
    if backend == "none":
        return NoReranker()
    if backend == "deterministic":
        return DeterministicReranker()
    if backend == "bge":
        return BgeReranker(model_name_or_path=model_name_or_path or DEFAULT_RERANKER_MODEL)
    if backend == "adapter":
        return BgeReranker(
            model_name_or_path=model_name_or_path or DEFAULT_RERANKER_MODEL,
            adapter_path=adapter_path,
        )
    if backend == "auto":
        from maintidoc.config import get_settings

        settings = get_settings()
        configured_model = settings.inference.reranker_model
        configured_adapter = settings.inference.reranker_adapter
        if not configured_model or configured_model == "none":
            return NoReranker()
        return BgeReranker(
            model_name_or_path=configured_model,
            adapter_path=configured_adapter,
        )
    msg = f"unknown reranker backend: {backend!r}"
    raise ConfigurationError(msg)


__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "BgeReranker",
    "DeterministicReranker",
    "NoReranker",
    "Reranker",
    "RerankerBackend",
    "make_reranker",
]
