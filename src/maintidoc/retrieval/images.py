"""Image and text encoders for the visual index (SPEC §3, §4.2; ADR 0007).

Two towers, one shared space, because SPEC §3 asks for a multilingual
CLIP-compatible encoder and no single permissively-licensed model provides both
halves:

* the **image tower** (`openai/clip-vit-base-patch32`) embeds rendered figures;
* the **text tower** (`sentence-transformers/clip-ViT-B-32-multilingual-v1`)
  embeds a French query into that same space.

`jinaai/jina-clip-v2` would have covered both in one model but is CC-BY-NC-4.0,
i.e. non-commercial, so it is excluded (ADR 0007).

Both towers are **512-dimensional** (ViT-B/32), which is why the image
collection is created at 512 while the text collection is created at BGE-M3's
1024. Mixing the two would silently return nonsense, so the width is part of
each index config and checked on every upsert and search.

As with text embeddings there is a deterministic double, and `make_image_embedder`
will not fall back to it silently.
"""

from __future__ import annotations

import hashlib
import io
import math
import threading
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from maintidoc.errors import ConfigurationError
from maintidoc.observability import get_logger

logger = get_logger(__name__)

#: ViT-B/32 width, shared by both towers in ADR 0007's pairing.
CLIP_DIMENSION = 512

ImageEmbedderBackend = Literal["auto", "clip", "hash"]

DEFAULT_IMAGE_TOWER = "openai/clip-vit-base-patch32"
DEFAULT_TEXT_TOWER = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


@runtime_checkable
class ImageEmbedder(Protocol):
    """Embeds figures and text queries into one comparable space."""

    @property
    def model_id(self) -> str:
        """Identifies the *space*, so a payload records which encoder built it."""
        ...

    @property
    def dimension(self) -> int: ...

    @property
    def semantic(self) -> bool: ...

    def embed_images(self, pngs: Sequence[bytes]) -> list[list[float]]:
        """Embed figure images for indexing."""
        ...

    def embed_text(self, text: str) -> list[float]:
        """Embed a query into the same space as `embed_images`."""
        ...


class HashingImageEmbedder:
    """Deterministic, non-semantic image embedder for tests and offline runs.

    Hashes the bytes, so identical images give identical vectors and different
    images give different ones — enough to test plumbing and ordering, useless
    for judging retrieval quality.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension < 8:
            raise ConfigurationError("hashing image embedder dimension must be at least 8")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"deterministic-hashing-image-v1-d{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def semantic(self) -> bool:
        return False

    def embed_images(self, pngs: Sequence[bytes]) -> list[list[float]]:
        return [self._vector(png) for png in pngs]

    def embed_text(self, text: str) -> list[float]:
        return self._vector(text.encode("utf-8"))

    def _vector(self, payload: bytes) -> list[float]:
        vector = [0.0] * self._dimension
        for offset in range(0, max(len(payload), 1), 32):
            chunk = payload[offset : offset + 32] or payload
            digest = hashlib.blake2b(chunk, digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self._dimension] += 1.0 if (value >> 63) & 1 else -1.0

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [component / norm for component in vector]


class ClipImageEmbedder:
    """ADR 0007's two-tower pairing, loaded lazily.

    Not verified at runtime yet: `transformers` and `sentence-transformers` are
    in the optional `ml` extra and the weights were prefetched separately. The
    first real load happens when `torch` is installed for this issue's
    verification step, and until then the deterministic double is what the tests
    exercise.
    """

    def __init__(
        self,
        image_model: str = DEFAULT_IMAGE_TOWER,
        text_model: str = DEFAULT_TEXT_TOWER,
        *,
        device: str = "auto",
        dimension: int = CLIP_DIMENSION,
        batch_size: int = 8,
    ) -> None:
        self._image_model_id = image_model
        self._text_model_id = text_model
        self._device = device
        self._dimension = dimension
        self._batch_size = batch_size
        # torch/transformers objects are untyped at this boundary: the package is
        # in the optional extra and ignored by mypy, so Any is the honest type.
        self._image_model: Any | None = None
        self._processor: Any | None = None
        self._text_model: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return f"clip:{self._image_model_id}|{self._text_model_id}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def semantic(self) -> bool:
        return True

    def embed_images(self, pngs: Sequence[bytes]) -> list[list[float]]:
        if not pngs:
            return []
        images = [self._to_pil(png) for png in pngs]
        model, processor, device = self._load_image_tower()

        import torch

        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model.get_image_features(**inputs)
            # transformers >=5 returns a BaseModelOutputWithPooling here rather
            # than a tensor; the 512-d embedding is on .pooler_output. getattr
            # keeps this working on 4.x too, where the call already returns the
            # tensor. Verified against transformers 5.22.
            features = getattr(output, "pooler_output", output)
            features = features / features.norm(dim=-1, keepdim=True)

        vectors: list[list[float]] = [
            [float(value) for value in row] for row in features.cpu().tolist()
        ]
        self._check_width(vectors)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        model = self._load_text_tower()
        encode = getattr(model, "encode", None)
        if encode is None:  # pragma: no cover - guarded by _load_text_tower
            raise ConfigurationError("text tower does not expose encode()")

        raw = encode([text], normalize_embeddings=True, convert_to_numpy=True)
        vector = [float(value) for value in raw[0]]
        if len(vector) != self._dimension:
            raise ConfigurationError(
                f"text tower returned {len(vector)} dimensions, expected {self._dimension}; "
                "the two towers are not aligned"
            )
        return vector

    # ------------------------------------------------------------------ internals

    def _check_width(self, vectors: Sequence[Sequence[float]]) -> None:
        for vector in vectors:
            if len(vector) != self._dimension:
                raise ConfigurationError(
                    f"image tower returned {len(vector)} dimensions, expected {self._dimension}"
                )

    @staticmethod
    def _to_pil(png: bytes) -> object:
        from PIL import Image, UnidentifiedImageError

        try:
            return Image.open(io.BytesIO(png)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ConfigurationError(
                f"image could not be decoded: {exc.__class__.__name__}"
            ) from exc

    def _resolve_device(self) -> str:
        if self._device != "auto":
            return self._device
        from maintidoc.runtime.devices import select_device

        return select_device("auto").kind.value

    def _load_image_tower(self) -> tuple[Any, Any, str]:
        with self._lock:
            if self._image_model is None or self._processor is None:
                try:
                    from transformers import CLIPModel, CLIPProcessor
                except ImportError as exc:
                    raise ConfigurationError(
                        "transformers is not installed; install the 'ml' extra or select "
                        "the deterministic image embedder explicitly"
                    ) from exc

                device = self._resolve_device()
                try:
                    model = CLIPModel.from_pretrained(self._image_model_id)
                    processor = CLIPProcessor.from_pretrained(self._image_model_id)
                except Exception as exc:
                    # Unloadable model must be a typed error, not a Hub traceback.
                    raise ConfigurationError(
                        f"could not load CLIP image tower {self._image_model_id!r}: "
                        f"{exc.__class__.__name__}"
                    ) from exc
                self._image_model = model.to(device).eval()
                self._processor = processor
                logger.info("loaded CLIP image tower %s on %s", self._image_model_id, device)
        return self._image_model, self._processor, self._resolve_device()

    def _load_text_tower(self) -> Any:
        with self._lock:
            if self._text_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ConfigurationError(
                        "sentence-transformers is not installed; install the 'ml' extra "
                        "or select the deterministic image embedder explicitly"
                    ) from exc

                device = self._resolve_device()
                try:
                    self._text_model = SentenceTransformer(self._text_model_id, device=device)
                except Exception as exc:
                    raise ConfigurationError(
                        f"could not load CLIP text tower {self._text_model_id!r}: "
                        f"{exc.__class__.__name__}"
                    ) from exc
                logger.info("loaded CLIP text tower %s on %s", self._text_model_id, device)
        return self._text_model


def make_image_embedder(
    *,
    backend: ImageEmbedderBackend = "auto",
    image_model: str = DEFAULT_IMAGE_TOWER,
    text_model: str = DEFAULT_TEXT_TOWER,
    device: str = "auto",
    hashing_dimension: int = 64,
    allow_hash: bool = False,
) -> ImageEmbedder:
    """Build an image embedder, refusing to downgrade silently.

    Raises:
        ConfigurationError: the requested backend cannot be constructed.
    """
    if backend == "hash":
        if not allow_hash:
            raise ConfigurationError(
                "the hashing image embedder is non-semantic and must be requested "
                "with allow_hash=True"
            )
        return HashingImageEmbedder(dimension=hashing_dimension)

    return ClipImageEmbedder(image_model, text_model, device=device)


__all__ = [
    "CLIP_DIMENSION",
    "ClipImageEmbedder",
    "DEFAULT_IMAGE_TOWER",
    "DEFAULT_TEXT_TOWER",
    "HashingImageEmbedder",
    "ImageEmbedder",
    "ImageEmbedderBackend",
    "make_image_embedder",
]
