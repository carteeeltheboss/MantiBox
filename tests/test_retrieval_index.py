"""Qdrant indexing and access-filtered search, against a live Qdrant.

Marked `integration`: these need a running Qdrant and are excluded from the fast
gate. They skip cleanly when no server is reachable so `make check` stays green
on a machine without Docker.

Start one with:
    docker run -d --name maintidoc-qdrant -p 6333:6333 qdrant/qdrant:v1.20.0
"""

from __future__ import annotations

import socket
import uuid

import pytest
from qdrant_client import QdrantClient, models

from maintidoc.errors import AccessDeniedError, ConfigurationError
from maintidoc.retrieval.embeddings import HashingEmbedder
from maintidoc.retrieval.index import (
    INDEX_VERSION,
    IndexConfig,
    QdrantIndex,
    element_from_payload,
    element_payload,
    point_id,
)
from maintidoc.schemas import ElementKind
from maintidoc.security.access import Principal
from tests.factories import make_element

HOST = "localhost"
PORT = 6333
DIMENSION = 64


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


@pytest.fixture
def collection() -> str:
    """A unique collection per test, so tests cannot interfere."""
    return f"maintidoc_test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def client() -> QdrantClient:
    qdrant = QdrantClient(host=HOST, port=PORT, timeout=10.0)
    yield qdrant
    qdrant.close()


@pytest.fixture
def index(client: QdrantClient, collection: str) -> QdrantIndex:
    built = QdrantIndex(client, IndexConfig(collection=collection, dimension=DIMENSION))
    built.ensure_collection()
    yield built
    client.delete_collection(collection)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=DIMENSION)


def sample_elements() -> list:
    return [
        make_element(
            document_id="manuel_pompe_a",
            element_id="paragraph_1_1",
            page=1,
            kind=ElementKind.PARAGRAPH,
            text="Le voyant orange signale une alarme de température du groupe froid.",
            access_groups=SITE_A,
        ),
        make_element(
            document_id="manuel_pompe_a",
            element_id="table_1_1",
            page=1,
            kind=ElementKind.TABLE,
            text="Tableau - Seuils: Température 85 °C alarme",
            access_groups=SITE_A,
        ),
        make_element(
            document_id="manuel_pompe_a_restreint",
            element_id="paragraph_2_1",
            page=2,
            kind=ElementKind.PARAGRAPH,
            text="Procédure restreinte de consignation du groupe.",
            access_groups=frozenset({"supervisors"}),
        ),
        make_element(
            document_id="manuel_armoire_c",
            element_id="paragraph_1_1",
            page=1,
            kind=ElementKind.PARAGRAPH,
            text="L'armoire de commande signale un défaut de tension.",
            access_groups=frozenset({"maintenance", "site_b"}),
        ),
    ]


def seed(index: QdrantIndex, embedder: HashingEmbedder) -> list:
    elements = sample_elements()
    vectors = embedder.embed_documents([element.text for element in elements])
    index.upsert(elements, vectors, embedder_model=embedder.model_id)
    return elements


class TestCollectionLifecycle:
    def test_ensure_collection_is_idempotent(self, client: QdrantClient, collection: str) -> None:
        built = QdrantIndex(client, IndexConfig(collection=collection, dimension=DIMENSION))
        try:
            built.ensure_collection()
            built.ensure_collection()
            assert client.collection_exists(collection)
        finally:
            client.delete_collection(collection)

    def test_recreate_empties_the_collection(
        self, index: QdrantIndex, embedder: HashingEmbedder
    ) -> None:
        seed(index, embedder)
        assert index.count() == 4
        index.ensure_collection(recreate=True)
        assert index.count() == 0

    def test_count_on_a_missing_collection_is_zero(self, client: QdrantClient) -> None:
        missing = QdrantIndex(
            client, IndexConfig(collection="maintidoc_never_created", dimension=DIMENSION)
        )
        assert missing.count() == 0


class TestUpsert:
    def test_indexes_all_elements(self, index: QdrantIndex, embedder: HashingEmbedder) -> None:
        assert (
            index.upsert(
                sample_elements(),
                embedder.embed_documents([e.text for e in sample_elements()]),
                embedder_model=embedder.model_id,
            )
            == 4
        )
        assert index.count() == 4

    def test_reingesting_the_same_elements_does_not_duplicate(
        self, index: QdrantIndex, embedder: HashingEmbedder
    ) -> None:
        """Point ids are derived, so a retried ingestion overwrites."""
        seed(index, embedder)
        seed(index, embedder)
        assert index.count() == 4

    def test_mismatched_counts_raise(self, index: QdrantIndex) -> None:
        with pytest.raises(ConfigurationError, match="elements but"):
            index.upsert(sample_elements(), [[0.1] * DIMENSION], embedder_model="x")

    def test_wrong_vector_width_raises(self, index: QdrantIndex) -> None:
        with pytest.raises(ConfigurationError, match="dimensions"):
            index.upsert(sample_elements()[:1], [[0.1] * 8], embedder_model="x")

    def test_empty_upsert_is_a_no_op(self, index: QdrantIndex) -> None:
        assert index.upsert([], [], embedder_model="x") == 0


class TestAccessFilteredSearch:
    def test_technician_cannot_reach_the_supervisors_only_manual(
        self, index: QdrantIndex, embedder: HashingEmbedder, technician: Principal
    ) -> None:
        """Policy encoded by the corpus: `maintenance` is a role group shared by
        both equipment families, so a maintenance technician may read both; the
        supervisors-only manual is the restricted case and must be excluded."""
        seed(index, embedder)
        hits = index.search(embedder.embed_query("voyant orange"), principal=technician, limit=10)

        assert hits
        documents = {hit.document_id for hit in hits}
        assert "manuel_pompe_a" in documents
        assert "manuel_pompe_a_restreint" not in documents

    def test_site_group_alone_isolates_the_families(
        self, index: QdrantIndex, embedder: HashingEmbedder
    ) -> None:
        """Without the shared role group a site principal sees only its own
        family, which is what proves site isolation rather than role overlap."""
        seed(index, embedder)
        site_a_only = Principal(user_id="u-site-a", groups=frozenset({"site_a"}))

        hits = index.search(embedder.embed_query("voyant"), principal=site_a_only, limit=10)
        assert {hit.document_id for hit in hits} == {"manuel_pompe_a"}

    def test_restricted_document_is_visible_to_a_supervisor(
        self, index: QdrantIndex, embedder: HashingEmbedder, supervisor: Principal
    ) -> None:
        seed(index, embedder)
        hits = index.search(embedder.embed_query("consignation"), principal=supervisor, limit=10)
        assert "manuel_pompe_a_restreint" in {hit.document_id for hit in hits}

    def test_other_site_sees_nothing(self, index: QdrantIndex, embedder: HashingEmbedder) -> None:
        seed(index, embedder)
        outsider = Principal(user_id="u-out", groups=frozenset({"site_c"}))
        hits = index.search(embedder.embed_query("voyant"), principal=outsider, limit=10)
        assert hits == []

    def test_principal_without_groups_is_refused(
        self, index: QdrantIndex, embedder: HashingEmbedder, no_groups: Principal
    ) -> None:
        """NEVER #7: no groups must mean denied, not "no filter"."""
        seed(index, embedder)
        with pytest.raises(AccessDeniedError):
            index.search(embedder.embed_query("voyant"), principal=no_groups, limit=10)

    def test_inactive_principal_is_refused(
        self, index: QdrantIndex, embedder: HashingEmbedder, deactivated: Principal
    ) -> None:
        seed(index, embedder)
        with pytest.raises(AccessDeniedError):
            index.search(embedder.embed_query("voyant"), principal=deactivated, limit=10)

    def test_kind_filter_narrows_results(
        self, index: QdrantIndex, embedder: HashingEmbedder, technician: Principal
    ) -> None:
        seed(index, embedder)
        hits = index.search(
            embedder.embed_query("tableau seuils"),
            principal=technician,
            limit=10,
            kind=ElementKind.TABLE,
        )
        assert hits
        assert all(hit.element.kind is ElementKind.TABLE for hit in hits)

    def test_limit_is_respected(
        self, index: QdrantIndex, embedder: HashingEmbedder, technician: Principal
    ) -> None:
        seed(index, embedder)
        hits = index.search(embedder.embed_query("voyant"), principal=technician, limit=1)
        assert len(hits) == 1
        assert hits[0].rank == 1

    def test_ranks_are_one_based_and_ascending(
        self, index: QdrantIndex, embedder: HashingEmbedder, technician: Principal
    ) -> None:
        seed(index, embedder)
        hits = index.search(embedder.embed_query("voyant"), principal=technician, limit=10)
        assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))

    def test_wrong_query_width_raises(self, index: QdrantIndex, technician: Principal) -> None:
        with pytest.raises(ConfigurationError, match="query vector"):
            index.search([0.1] * 8, principal=technician)

    def test_empty_collection_returns_no_hits(
        self, index: QdrantIndex, embedder: HashingEmbedder, technician: Principal
    ) -> None:
        assert index.search(embedder.embed_query("voyant"), principal=technician) == []


class TestDeletion:
    def test_delete_document_removes_only_that_document(
        self, index: QdrantIndex, embedder: HashingEmbedder
    ) -> None:
        seed(index, embedder)
        index.delete_document("manuel_pompe_a")
        assert index.count() == 2

    def test_delete_is_idempotent(self, index: QdrantIndex, embedder: HashingEmbedder) -> None:
        seed(index, embedder)
        index.delete_document("manuel_pompe_a")
        index.delete_document("manuel_pompe_a")
        assert index.count() == 2


class TestPayloadRoundTrip:
    def test_payload_carries_the_spec_4_1_metadata(self) -> None:
        element = make_element()
        payload = element_payload(element, embedder_model="test")

        assert payload["document_id"] == element.document_id
        assert payload["document_version"] == element.document_version
        assert payload["page"] == element.page
        assert payload["element_id"] == element.element_id
        assert payload["equipment_family"] == element.equipment_family
        assert set(payload["access_groups"]) == set(element.access_groups)

    def test_payload_records_provenance_for_staleness(self) -> None:
        payload = element_payload(make_element(), embedder_model="BAAI/bge-m3")
        assert payload["embedder_model"] == "BAAI/bge-m3"
        assert payload["index_version"] == INDEX_VERSION

    def test_round_trip_reconstructs_the_element(self) -> None:
        element = make_element()
        assert element_from_payload(element_payload(element, embedder_model="x")) == element

    def test_access_groups_are_stored_as_an_ordered_list(self) -> None:
        """Qdrant keyword matching needs a deterministic list, not a set."""
        payload = element_payload(
            make_element(access_groups=frozenset({"b", "a"})), embedder_model="x"
        )
        assert payload["access_groups"] == ["a", "b"]

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="not a valid element"):
            element_from_payload({"element_id": "only-this"})

    def test_unknown_kind_raises(self) -> None:
        payload = element_payload(make_element(), embedder_model="x")
        payload["kind"] = "hologram"
        with pytest.raises(ConfigurationError, match="not a valid element"):
            element_from_payload(payload)


class TestPointIds:
    def test_are_deterministic(self) -> None:
        assert point_id("doc", "el") == point_id("doc", "el")

    def test_differ_by_element_and_document(self) -> None:
        assert point_id("doc", "a") != point_id("doc", "b")
        assert point_id("doc1", "a") != point_id("doc2", "a")

    def test_are_valid_uuids(self) -> None:
        uuid.UUID(point_id("doc", "el"))

    def test_are_accepted_by_qdrant(self, index: QdrantIndex, embedder: HashingEmbedder) -> None:
        seed(index, embedder)
        stored = index._client.scroll(index.config.collection, limit=10)[0]
        assert len(stored) == 4
        assert all(isinstance(point.id, (str, uuid.UUID)) for point in stored)


class TestSchemaGuard:
    def test_collection_reports_its_vector_size(
        self, client: QdrantClient, collection: str
    ) -> None:
        built = QdrantIndex(client, IndexConfig(collection=collection, dimension=DIMENSION))
        try:
            built.ensure_collection()
            info = client.get_collection(collection)
            vectors = info.config.params.vectors
            assert isinstance(vectors, models.VectorParams)
            assert vectors.size == DIMENSION
            assert vectors.distance == models.Distance.COSINE
        finally:
            client.delete_collection(collection)
