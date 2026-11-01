"""Text embeddings behind a provider seam (SPEC §3: BGE-M3).

Two implementations, and the difference between them is not cosmetic:

* `BgeM3Embedder` is the real multilingual model named by SPEC §3. It is
  semantic: similar meanings land near each other.
* `HashingEmbedder` is **not semantic**. It hashes tokens into a fixed-width
  vector so the pipeline, the index and the tests can run with no weights, no
  torch and no network. Retrieval *quality* measured with it is meaningless and
  must never be reported as a result — only ordering, filter correctness and
  payload shape can be verified through it.

`make_embedder` will not silently fall back to the hashing double. An unavailable
real model raises, because a quiet downgrade from a semantic index to a hashed
one would corrupt every retrieval number the project reports.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from maintidoc.errors import ConfigurationError

#: BGE-M3 dense embedding width. Fixed by the model, not a free parameter.
BGE_M3_DIMENSION = 1024

EmbedderBackend = Literal["auto", "bge-m3", "hash"]

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@runtime_checkable
class Embedder(Protocol):
    """Embeds documents and queries into the same space."""

    @property
    def model_id(self) -> str:
        """Identifier recorded in the index payload and in MLflow."""
        ...

    @property
    def dimension(self) -> int:
        """Vector width. The index collection is created from this."""
        ...

    @property
    def semantic(self) -> bool:
        """False for the deterministic double, so callers can refuse to treat
        its similarity scores as meaningful."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages for indexing."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query. Must use the same space as `embed_documents`."""
        ...


class HashingEmbedder:
    """Deterministic, non-semantic embedder for tests and offline runs.

    Uses BLAKE2b rather than `hash()` because Python randomises string hashing
    per process, which would make vectors differ between runs and quietly break
    reproducibility.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ConfigurationError("hashing embedder dimension must be at least 8")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"deterministic-hashing-v1-d{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def semantic(self) -> bool:
        return False

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self._dimension
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            # Cosine distance is undefined against a zero vector; pin a unit
            # vector on one axis so the index never receives a degenerate point.
            vector[0] = 1.0
            return vector
        return [component / norm for component in vector]


class BgeM3Embedder:
    """The SPEC §3 multilingual text embedder, loaded lazily.

    Lazy because importing torch and loading ~2 GB of weights at module import
    would make every command that merely mentions embeddings pay the cost.
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        *,
        device: str = "auto",
        batch_size: int = 16,
        dimension: int = BGE_M3_DIMENSION,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._batch_size = batch_size
        self._dimension = dimension
        self._model: object | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def semantic(self) -> bool:
        return True

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        vectors = self._encode([text])
        return vectors[0]

    # ------------------------------------------------------------------ internals

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        encode = getattr(model, "encode", None)
        if encode is None:  # pragma: no cover - guarded by _load
            raise ConfigurationError("embedding model does not expose encode()")

        raw = encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vectors: list[list[float]] = [[float(value) for value in row] for row in raw]

        for vector in vectors:
            if len(vector) != self._dimension:
                raise ConfigurationError(
                    f"{self._model_id} returned {len(vector)} dimensions, "
                    f"expected {self._dimension}"
                )
        return vectors

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ConfigurationError(
                        "sentence-transformers is not installed; install the 'ml' extra "
                        "or select the deterministic embedder explicitly"
                    ) from exc

                device = self._resolve_device()
                try:
                    self._model = SentenceTransformer(self._model_id, device=device)
                except Exception as exc:
                    # A missing, corrupt or unresolvable model must be a typed
                    # error, not a raw OSError or a Hub traceback. The message
                    # carries only the exception class, never a path with data.
                    raise ConfigurationError(
                        f"could not load embedding model {self._model_id!r} on {device}: "
                        f"{exc.__class__.__name__}"
                    ) from exc
        return self._model

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        from maintidoc.runtime.devices import select_device

        return select_device("auto").kind.value


def make_embedder(
    *,
    backend: EmbedderBackend = "auto",
    model_id: str = "BAAI/bge-m3",
    device: str = "auto",
    hashing_dimension: int = 256,
    allow_hash: bool = False,
) -> Embedder:
    """Build an embedder, refusing to downgrade silently.

    Args:
        backend: `auto` picks the real model and fails if it is unavailable;
            `bge-m3` forces it; `hash` explicitly opts into the double.
        allow_hash: must be True, with `backend="hash"`, to get the double. It
            is a second flag on purpose so no configuration change can turn a
            semantic index into a hashed one by accident.

    Raises:
        ConfigurationError: the requested backend cannot be constructed.
    """
    if backend == "hash":
        if not allow_hash:
            raise ConfigurationError(
                "the hashing embedder is non-semantic and must be requested with allow_hash=True"
            )
        return HashingEmbedder(dimension=hashing_dimension)

    return BgeM3Embedder(model_id, device=device)


__all__ = [
    "BGE_M3_DIMENSION",
    "BgeM3Embedder",
    "Embedder",
    "EmbedderBackend",
    "HashingEmbedder",
    "make_embedder",
]
