"""Image embedder seam and the 512-vs-1024 width hazard (ADR 0007)."""

from __future__ import annotations

import base64
import math
from pathlib import Path

import pytest

from maintidoc.errors import ConfigurationError
from maintidoc.retrieval.embeddings import BGE_M3_DIMENSION
from maintidoc.retrieval.images import (
    CLIP_DIMENSION,
    ClipImageEmbedder,
    HashingImageEmbedder,
    ImageEmbedder,
    make_image_embedder,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"figure-bytes" * 8
OTHER_PNG = b"\x89PNG\r\n\x1a\n" + b"different-figure" * 8

#: A genuinely decodable 1x1 PNG. The fake bytes above are fine for the hashing
#: double, but decoding happens before the import check, so the missing-'ml'
#: path can only be reached with an image PIL can actually open.
REAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AF+9lVgAAAAAElFTkSuQmCC"
)


class TestWidthHazard:
    def test_clip_is_512_and_bge_is_1024(self) -> None:
        """The two collections are created at different widths on purpose; mixing
        them would return nonsense rather than an error."""
        assert CLIP_DIMENSION == 512
        assert BGE_M3_DIMENSION == 1024
        assert CLIP_DIMENSION != BGE_M3_DIMENSION


class TestHashingImageEmbedder:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(HashingImageEmbedder(), ImageEmbedder)

    def test_declares_itself_non_semantic(self) -> None:
        assert HashingImageEmbedder().semantic is False

    def test_dimension_is_respected(self) -> None:
        embedder = HashingImageEmbedder(dimension=32)
        assert embedder.dimension == 32
        assert len(embedder.embed_text("voyant orange")) == 32

    def test_rejects_a_degenerate_dimension(self) -> None:
        with pytest.raises(ConfigurationError, match="at least 8"):
            HashingImageEmbedder(dimension=4)

    def test_identical_images_give_identical_vectors(self) -> None:
        embedder = HashingImageEmbedder(dimension=32)
        assert embedder.embed_images([PNG]) == embedder.embed_images([PNG])

    def test_different_images_give_different_vectors(self) -> None:
        embedder = HashingImageEmbedder(dimension=32)
        assert embedder.embed_images([PNG]) != embedder.embed_images([OTHER_PNG])

    def test_vectors_are_unit_length(self) -> None:
        vector = HashingImageEmbedder(dimension=32).embed_images([PNG])[0]
        norm = math.sqrt(sum(component * component for component in vector))
        assert norm == pytest.approx(1.0)

    def test_text_embeddings_are_also_unit_length(self) -> None:
        vector = HashingImageEmbedder(dimension=32).embed_text("panneau de commande")
        norm = math.sqrt(sum(component * component for component in vector))
        assert norm == pytest.approx(1.0)

    def test_empty_batch_is_allowed(self) -> None:
        assert HashingImageEmbedder().embed_images([]) == []

    def test_is_deterministic_across_instances(self) -> None:
        assert HashingImageEmbedder(dimension=16).embed_images([PNG]) == HashingImageEmbedder(
            dimension=16
        ).embed_images([PNG])


class TestClipImageEmbedder:
    def test_declares_itself_semantic(self) -> None:
        assert ClipImageEmbedder().semantic is True

    def test_model_id_names_both_towers(self) -> None:
        """A single id cannot express the pairing, so the id records both."""
        model_id = ClipImageEmbedder().model_id
        assert model_id.startswith("clip:")
        assert "clip-vit-base-patch32" in model_id
        assert "multilingual" in model_id

    def test_dimension_is_vit_b32(self) -> None:
        assert ClipImageEmbedder().dimension == 512

    def test_loading_is_lazy(self) -> None:
        embedder = ClipImageEmbedder()
        assert embedder._image_model is None
        assert embedder._text_model is None

    def test_empty_batch_does_not_load_the_image_tower(self) -> None:
        embedder = ClipImageEmbedder()
        assert embedder.embed_images([]) == []
        assert embedder._image_model is None

    def test_unloadable_image_tower_raises(self, tmp_path: Path) -> None:
        """Environment-independent: with the 'ml' extra absent the import fails,
        with it present the load fails. Both raise ConfigurationError, and an
        empty local directory means this can never become a network request."""
        with pytest.raises(ConfigurationError):
            ClipImageEmbedder(str(tmp_path), str(tmp_path)).embed_images([REAL_PNG])

    def test_unloadable_text_tower_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError):
            ClipImageEmbedder(str(tmp_path), str(tmp_path)).embed_text("voyant orange")

    def test_undecodable_image_raises_before_loading(self) -> None:
        with pytest.raises(ConfigurationError, match="could not be decoded"):
            ClipImageEmbedder().embed_images([b"not an image"])


class TestFactory:
    def test_hash_backend_needs_an_explicit_opt_in(self) -> None:
        with pytest.raises(ConfigurationError, match="allow_hash=True"):
            make_image_embedder(backend="hash")

    def test_hash_backend_with_opt_in_works(self) -> None:
        embedder = make_image_embedder(backend="hash", allow_hash=True, hashing_dimension=16)
        assert isinstance(embedder, HashingImageEmbedder)
        assert embedder.dimension == 16

    def test_auto_selects_the_real_pair(self) -> None:
        assert isinstance(make_image_embedder(backend="auto"), ClipImageEmbedder)

    def test_auto_does_not_fall_back_to_hashing(self) -> None:
        """No embedding call here: with the 'ml' extra installed it would reach
        the network, and a test must never do that."""
        assert make_image_embedder(backend="auto").semantic is True
