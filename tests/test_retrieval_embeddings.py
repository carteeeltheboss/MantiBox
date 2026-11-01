"""Embedder seam: the deterministic double and the refusal to downgrade."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from maintidoc.errors import ConfigurationError
from maintidoc.retrieval.embeddings import (
    BGE_M3_DIMENSION,
    BgeM3Embedder,
    Embedder,
    HashingEmbedder,
    make_embedder,
)


class TestHashingEmbedder:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(HashingEmbedder(), Embedder)

    def test_declares_itself_non_semantic(self) -> None:
        """Callers must be able to refuse to report quality metrics from this."""
        assert HashingEmbedder().semantic is False

    def test_dimension_is_respected(self) -> None:
        assert HashingEmbedder(dimension=64).dimension == 64
        assert len(HashingEmbedder(dimension=64).embed_query("texte")) == 64

    def test_model_id_encodes_the_dimension(self) -> None:
        assert "d128" in HashingEmbedder(dimension=128).model_id

    def test_rejects_a_degenerate_dimension(self) -> None:
        with pytest.raises(ConfigurationError, match="at least 8"):
            HashingEmbedder(dimension=4)

    def test_is_deterministic_across_instances(self) -> None:
        """BLAKE2b, not Python's randomised hash(), so vectors survive a restart."""
        first = HashingEmbedder(dimension=32).embed_query("voyant orange")
        second = HashingEmbedder(dimension=32).embed_query("voyant orange")
        assert first == second

    def test_query_and_document_agree_for_the_same_text(self) -> None:
        embedder = HashingEmbedder(dimension=32)
        assert (
            embedder.embed_query("alarme température")
            == embedder.embed_documents(["alarme température"])[0]
        )

    def test_vectors_are_unit_length(self) -> None:
        vector = HashingEmbedder(dimension=32).embed_query("groupe de pompage centrifuge")
        norm = math.sqrt(sum(component * component for component in vector))
        assert norm == pytest.approx(1.0)

    def test_different_texts_give_different_vectors(self) -> None:
        embedder = HashingEmbedder(dimension=64)
        assert embedder.embed_query("voyant orange") != embedder.embed_query("voyant rouge")

    def test_empty_text_still_yields_a_unit_vector(self) -> None:
        """Cosine distance against a zero vector is undefined, so never emit one."""
        vector = HashingEmbedder(dimension=16).embed_query("")
        norm = math.sqrt(sum(component * component for component in vector))
        assert norm == pytest.approx(1.0)

    def test_documents_are_embedded_in_order(self) -> None:
        embedder = HashingEmbedder(dimension=16)
        texts = ["un", "deux", "trois"]
        assert embedder.embed_documents(texts) == [embedder.embed_query(t) for t in texts]

    def test_empty_batch_is_allowed(self) -> None:
        assert HashingEmbedder(dimension=16).embed_documents([]) == []

    def test_french_accents_are_tokenised_not_dropped(self) -> None:
        embedder = HashingEmbedder(dimension=64)
        assert embedder.embed_query("température") != embedder.embed_query("temperature")


class TestBgeM3Embedder:
    def test_declares_itself_semantic(self) -> None:
        assert BgeM3Embedder().semantic is True

    def test_uses_the_spec_model_and_dimension(self) -> None:
        embedder = BgeM3Embedder()
        assert embedder.model_id == "BAAI/bge-m3"
        assert embedder.dimension == BGE_M3_DIMENSION == 1024

    def test_loading_is_lazy(self) -> None:
        """Constructing the embedder must not import torch or load weights."""
        embedder = BgeM3Embedder()
        assert embedder._model is None

    def test_unloadable_model_raises_rather_than_degrading(self, tmp_path: Path) -> None:
        """Whether or not the 'ml' extra is installed, an unloadable model must
        raise a typed error instead of silently hashing. An empty local directory
        is used so this can never become a network request: with the extra absent
        the import fails, with it present the load fails, and both paths raise
        ConfigurationError."""
        with pytest.raises(ConfigurationError):
            BgeM3Embedder(str(tmp_path)).embed_query("voyant orange")

    def test_empty_batch_does_not_load_the_model(self) -> None:
        embedder = BgeM3Embedder()
        assert embedder.embed_documents([]) == []
        assert embedder._model is None


class TestFactory:
    def test_hash_backend_needs_an_explicit_opt_in(self) -> None:
        """One config flag must never be able to turn a semantic index into a
        hashed one by accident."""
        with pytest.raises(ConfigurationError, match="allow_hash=True"):
            make_embedder(backend="hash")

    def test_hash_backend_with_opt_in_works(self) -> None:
        embedder = make_embedder(backend="hash", allow_hash=True, hashing_dimension=32)
        assert isinstance(embedder, HashingEmbedder)
        assert embedder.dimension == 32

    def test_auto_selects_the_real_model(self) -> None:
        assert isinstance(make_embedder(backend="auto"), BgeM3Embedder)

    def test_auto_does_not_fall_back_to_hashing(self) -> None:
        """No embedding call here: with the 'ml' extra installed it would hit the
        network, and a test must never do that."""
        assert make_embedder(backend="auto").semantic is True
