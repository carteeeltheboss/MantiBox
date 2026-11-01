"""Image index against a live Qdrant: 512-dim collection and the width guard.

Marked `integration`. Skips cleanly when no server is reachable.
"""

from __future__ import annotations

import socket
import uuid

import pytest
from qdrant_client import QdrantClient

from maintidoc.errors import ConfigurationError
from maintidoc.retrieval.images import CLIP_DIMENSION, HashingImageEmbedder
from maintidoc.retrieval.index import IndexConfig, QdrantIndex
from maintidoc.schemas import ElementKind
from maintidoc.security.access import Principal
from tests.factories import make_element

HOST = "localhost"
PORT = 6333


def qdrant_available() -> bool:
    try:
        with socket.create_connection((HOST, PORT), timeout=1.0):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not qdrant_available(), reason="Qdrant is not reachable on localhost:6333"),
]

SITE_A = frozenset({"maintenance", "site_a"})

FIGURES = (
    b"\x89PNG\r\n\x1a\nfigure-panel-a",
    b"\x89PNG\r\n\x1a\nfigure-panel-b",
    b"\x89PNG\r\n\x1a\nrestricted-panel",
)


@pytest.fixture
def client() -> QdrantClient:
    qdrant = QdrantClient(host=HOST, port=PORT, timeout=10)
    yield qdrant
    qdrant.close()


@pytest.fixture
def image_index(client: QdrantClient) -> QdrantIndex:
    collection = f"maintidoc_images_{uuid.uuid4().hex[:12]}"
    built = QdrantIndex(
        client,
        IndexConfig(collection=collection, dimension=CLIP_DIMENSION),
    )
    built.ensure_collection()
    yield built
    client.delete_collection(collection)


@pytest.fixture
def embedder() -> HashingImageEmbedder:
    """Hashing double sized to CLIP's true width, so the collection shape is real."""
    return HashingImageEmbedder(dimension=CLIP_DIMENSION)


def figure_elements() -> list:
    return [
        make_element(
            document_id="manuel_pompe_a",
            element_id="figure_1_1",
            page=1,
            kind=ElementKind.FIGURE,
            text="Figure - Voyants du panneau de commande du groupe de pompage",
            access_groups=SITE_A,
            image_key="manuel_pompe_a_p1_figure_1.png",
        ),
        make_element(
            document_id="manuel_armoire_c",
            element_id="figure_1_1",
            page=1,
            kind=ElementKind.FIGURE,
            text="Figure - Voyants de l'armoire de commande electrique",
            access_groups=frozenset({"maintenance", "site_b"}),
            image_key="manuel_armoire_c_p1_figure_1.png",
        ),
        make_element(
            document_id="manuel_pompe_a_restreint",
            element_id="figure_2_1",
            page=2,
            kind=ElementKind.FIGURE,
            text="Figure - Consignation, document restreint",
            access_groups=frozenset({"supervisors"}),
            image_key="manuel_pompe_a_restreint_p2_figure_1.png",
        ),
    ]


class TestImageCollectionWidth:
    def test_collection_is_created_at_clip_width(self, image_index: QdrantIndex) -> None:
        assert image_index.config.dimension == CLIP_DIMENSION == 512

    def test_text_width_vectors_are_rejected(self, image_index: QdrantIndex) -> None:
        """A 1024-d BGE vector must never enter the 512-d image collection; that
        mismatch is silent nonsense rather than an error at query time."""
        with pytest.raises(ConfigurationError, match="dimensions"):
            image_index.upsert(
                figure_elements()[:1],
                [[0.1] * 1024],
                embedder_model="bge-m3",
            )

    def test_search_with_wrong_width_is_rejected(
        self, image_index: QdrantIndex, technician: Principal
    ) -> None:
        with pytest.raises(ConfigurationError, match="query vector"):
            image_index.search([0.1] * 1024, principal=technician)


class TestImageIndexing:
    def test_figures_are_indexed_with_their_image_keys(
        self, image_index: QdrantIndex, embedder: HashingImageEmbedder
    ) -> None:
        elements = figure_elements()
        vectors = embedder.embed_images(list(FIGURES))
        image_index.upsert(elements, vectors, embedder_model=embedder.model_id)

        assert image_index.count() == 3

    def test_image_vectors_are_accepted_at_clip_width(
        self, image_index: QdrantIndex, embedder: HashingImageEmbedder
    ) -> None:
        vectors = embedder.embed_images(list(FIGURES))
        assert all(len(vector) == CLIP_DIMENSION for vector in vectors)


class TestImageSearchAccessFiltering:
    def seed(self, index: QdrantIndex, embedder: HashingImageEmbedder) -> None:
        elements = figure_elements()
        vectors = embedder.embed_images(list(FIGURES))
        index.upsert(elements, vectors, embedder_model=embedder.model_id)

    def test_text_query_finds_figures_within_the_callers_groups(
        self, image_index: QdrantIndex, embedder: HashingImageEmbedder, technician: Principal
    ) -> None:
        self.seed(image_index, embedder)
        hits = image_index.search(
            embedder.embed_text("voyant orange"), principal=technician, limit=5
        )

        assert hits
        documents = {hit.document_id for hit in hits}
        assert "manuel_pompe_a_restreint" not in documents

    def test_other_site_sees_no_figures(
        self, image_index: QdrantIndex, embedder: HashingImageEmbedder
    ) -> None:
        self.seed(image_index, embedder)
        outsider = Principal(user_id="u-out", groups=frozenset({"site_z"}))
        assert image_index.search(embedder.embed_text("voyant"), principal=outsider, limit=5) == []

    def test_principal_without_groups_is_refused(
        self, image_index: QdrantIndex, embedder: HashingImageEmbedder, no_groups: Principal
    ) -> None:
        self.seed(image_index, embedder)
        with pytest.raises(Exception, match="no access groups"):
            image_index.search(embedder.embed_text("voyant"), principal=no_groups, limit=5)

    def test_figure_hits_carry_their_image_key(
        self, image_index: QdrantIndex, embedder: HashingImageEmbedder, technician: Principal
    ) -> None:
        """The VLM needs the original figure; losing the image key would make a
        hit uncitable in the multimodal sense."""
        self.seed(image_index, embedder)
        hits = image_index.search(embedder.embed_text("panneau"), principal=technician, limit=5)

        assert hits
        assert all(hit.element.image_key for hit in hits)
        assert all(hit.element.kind is ElementKind.FIGURE for hit in hits)
