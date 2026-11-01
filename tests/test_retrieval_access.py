"""Opening documents and figures: deny before fetch, and no existence oracle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from maintidoc.errors import AccessDeniedError, ConfigurationError
from maintidoc.retrieval.access import (
    DocumentOpener,
    InMemoryCatalog,
    catalog_from_manifest,
    image_prefix,
    key_belongs_to,
)
from maintidoc.security.access import Principal
from maintidoc.storage.objects import FilesystemStore, StoredObject

CORPUS_MANIFEST = Path("corpus/manifest.json")

SITE_A = frozenset({"maintenance", "site_a"})
SITE_B = frozenset({"maintenance", "site_b"})
RESTRICTED = frozenset({"supervisors"})

CATALOG = InMemoryCatalog(
    documents={
        "manuel_pompe_a": SITE_A,
        "manuel_armoire_c": SITE_B,
        "manuel_pompe_a_restreint": RESTRICTED,
    }
)


@dataclass
class RecordingStore:
    """An ObjectStore that records every read, so denial can be proven."""

    inner: FilesystemStore
    reads: list[str] = field(default_factory=list)

    def get(self, key: str) -> bytes:
        self.reads.append(key)
        return self.inner.get(key)

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        return self.inner.put(key, data, content_type=content_type)

    def exists(self, key: str) -> bool:
        return self.inner.exists(key)

    def delete(self, key: str) -> None:
        self.inner.delete(key)

    def stat(self, key: str) -> StoredObject:
        return self.inner.stat(key)


@pytest.fixture
def store(tmp_path: Path) -> RecordingStore:
    inner = FilesystemStore(root=tmp_path / "objects")
    inner.put("documents/manuel_pompe_a.pdf", b"%PDF pompe")
    inner.put("documents/manuel_armoire_c.pdf", b"%PDF armoire")
    inner.put("documents/manuel_pompe_a_restreint.pdf", b"%PDF secret")
    inner.put("manuel_pompe_a_p1_figure_1.png", b"\x89PNG pompe")
    inner.put("manuel_pompe_a_restreint_p2_figure_1.png", b"\x89PNG secret")
    return RecordingStore(inner=inner)


@pytest.fixture
def opener(store: RecordingStore) -> DocumentOpener:
    return DocumentOpener(catalog=CATALOG, store=store)


class TestDenyBeforeFetch:
    def test_denied_document_is_never_read(
        self, opener: DocumentOpener, store: RecordingStore, outsider: Principal
    ) -> None:
        """The point of the check is that no bytes move. Assert the store was
        untouched, not merely that an exception was raised."""
        with pytest.raises(AccessDeniedError):
            opener.open_document(principal=outsider, document_id="manuel_pompe_a")

        assert store.reads == []

    def test_denied_image_is_never_read(
        self, opener: DocumentOpener, store: RecordingStore, outsider: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            opener.open_image(
                principal=outsider,
                document_id="manuel_pompe_a",
                key="manuel_pompe_a_p1_figure_1.png",
            )

        assert store.reads == []

    def test_restricted_document_is_never_read_by_a_technician(
        self, opener: DocumentOpener, store: RecordingStore, technician: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            opener.open_document(principal=technician, document_id="manuel_pompe_a_restreint")

        assert store.reads == []


class TestNoExistenceOracle:
    def test_unknown_document_is_denied_not_notfound(
        self, opener: DocumentOpener, technician: Principal
    ) -> None:
        """Distinguishing "missing" from "not yours" would let a caller probe ids."""
        with pytest.raises(AccessDeniedError):
            opener.open_document(principal=technician, document_id="does_not_exist")

    def test_unknown_and_forbidden_raise_the_same_error_type(
        self, opener: DocumentOpener, outsider: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError) as unknown:
            opener.open_document(principal=outsider, document_id="does_not_exist")
        with pytest.raises(AccessDeniedError) as forbidden:
            opener.open_document(principal=outsider, document_id="manuel_pompe_a")

        assert type(unknown.value) is type(forbidden.value)
        assert unknown.value.code == forbidden.value.code


class TestAllowedReads:
    def test_technician_reads_their_family(
        self, opener: DocumentOpener, technician: Principal
    ) -> None:
        assert (
            opener.open_document(principal=technician, document_id="manuel_pompe_a")
            == b"%PDF pompe"
        )

    def test_supervisor_reads_the_restricted_document(
        self, opener: DocumentOpener, supervisor: Principal
    ) -> None:
        assert (
            opener.open_document(principal=supervisor, document_id="manuel_pompe_a_restreint")
            == b"%PDF secret"
        )

    def test_site_only_principal_cannot_cross_families(self, opener: DocumentOpener) -> None:
        site_a_only = Principal(user_id="u-site-a", groups=frozenset({"site_a"}))
        assert (
            opener.open_document(principal=site_a_only, document_id="manuel_pompe_a")
            == b"%PDF pompe"
        )
        with pytest.raises(AccessDeniedError):
            opener.open_document(principal=site_a_only, document_id="manuel_armoire_c")

    def test_inactive_principal_is_denied(
        self, opener: DocumentOpener, deactivated: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            opener.open_document(principal=deactivated, document_id="manuel_pompe_a")

    def test_principal_without_groups_is_denied(
        self, opener: DocumentOpener, no_groups: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            opener.open_document(principal=no_groups, document_id="manuel_pompe_a")


class TestImageKeyBinding:
    def test_key_from_the_same_document_is_allowed(
        self, opener: DocumentOpener, technician: Principal
    ) -> None:
        assert (
            opener.open_image(
                principal=technician,
                document_id="manuel_pompe_a",
                key="manuel_pompe_a_p1_figure_1.png",
            )
            == b"\x89PNG pompe"
        )

    def test_a_permitted_id_cannot_be_paired_with_another_documents_key(
        self, opener: DocumentOpener, store: RecordingStore, technician: Principal
    ) -> None:
        """Technician may read manuel_pompe_a, but not by borrowing the
        restricted document's figure key."""
        with pytest.raises(AccessDeniedError, match="does not belong"):
            opener.open_image(
                principal=technician,
                document_id="manuel_pompe_a",
                key="manuel_pompe_a_restreint_p2_figure_1.png",
            )

        assert store.reads == []

    def test_denied_before_fetch_even_for_a_matching_key(
        self, opener: DocumentOpener, store: RecordingStore, outsider: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            opener.open_image(
                principal=outsider,
                document_id="manuel_pompe_a",
                key="manuel_pompe_a_p1_figure_1.png",
            )

        assert store.reads == []


class TestKeyHelpers:
    def test_prefix_lowercases_and_replaces_unsafe_characters(self) -> None:
        assert image_prefix("Manuel-Pompe A") == "manuel_pompe_a"

    def test_key_belongs_to_its_document(self) -> None:
        assert key_belongs_to("manuel_pompe_a_p1_figure_1.png", "manuel_pompe_a")

    def test_key_of_another_document_does_not_belong(self) -> None:
        assert not key_belongs_to("manuel_armoire_c_p1_figure_1.png", "manuel_pompe_a")

    def test_directory_style_keys_are_recognised(self) -> None:
        assert key_belongs_to("manuel_pompe_a/figures/1.png", "manuel_pompe_a")


class TestManifestCatalog:
    def test_encodes_the_corpus_access_policy(self) -> None:
        catalog = catalog_from_manifest(CORPUS_MANIFEST)

        assert catalog.access_groups("manuel_pompe_a_restreint") == frozenset({"supervisors"})
        assert "site_a" in (catalog.access_groups("manuel_pompe_a") or frozenset())
        assert catalog.access_groups("not_a_document") is None

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            catalog_from_manifest(tmp_path / "absent.json")

    def test_malformed_manifest_raises(self, tmp_path: Path) -> None:
        broken = tmp_path / "manifest.json"
        broken.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="not valid JSON"):
            catalog_from_manifest(broken)

    def test_manifest_without_documents_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "manifest.json"
        empty.write_text(json.dumps({"documents": []}), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="no usable documents"):
            catalog_from_manifest(empty)


class TestMissingObject:
    def test_catalogued_but_absent_object_is_a_configuration_error(
        self, tmp_path: Path, technician: Principal
    ) -> None:
        """A catalogued document whose bytes are gone is an operational bug, not
        a permission denial, and must not be reported as one."""
        opener = DocumentOpener(
            catalog=InMemoryCatalog(documents={"ghost": SITE_A}),
            store=FilesystemStore(root=tmp_path / "empty"),
        )
        with pytest.raises(ConfigurationError, match="object missing"):
            opener.open_document(principal=technician, document_id="ghost")
