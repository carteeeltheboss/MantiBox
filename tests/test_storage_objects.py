"""Object storage: key safety, atomicity and the backend-neutral contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from maintidoc.storage.objects import (
    FilesystemStore,
    ObjectStore,
    ObjectStoreError,
    StoredObject,
    _guess_content_type,
    validate_key,
)


@pytest.fixture
def store(tmp_path: Path) -> FilesystemStore:
    return FilesystemStore(root=tmp_path / "objects")


class TestValidateKey:
    @pytest.mark.parametrize(
        "key",
        [
            "documents/manuel_pompe_a.pdf",
            "images/manuel_pompe_a_p1_figure_3.png",
            "a",
            "a/b/c/d.json",
            "documents/é-1.pdf",
        ],
    )
    def test_accepts_normal_keys(self, key: str) -> None:
        assert validate_key(key) == key

    def test_rejects_empty(self) -> None:
        with pytest.raises(ObjectStoreError, match="must not be empty"):
            validate_key("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ObjectStoreError):
            validate_key("   ")

    @pytest.mark.parametrize(
        "key",
        [
            "../escape.pdf",
            "documents/../../etc/passwd",
            "a/./b.pdf",
            "documents//x.pdf",
            "documents/",
        ],
        ids=["dotdot", "deep-dotdot", "dot", "double-slash", "trailing-slash"],
    )
    def test_rejects_traversal_and_empty_segments(self, key: str) -> None:
        with pytest.raises(ObjectStoreError, match="invalid segment"):
            validate_key(key)

    @pytest.mark.parametrize("key", ["/absolute.pdf", "C:/windows.pdf"])
    def test_rejects_absolute_paths(self, key: str) -> None:
        with pytest.raises(ObjectStoreError, match="relative"):
            validate_key(key)

    def test_rejects_backslashes(self) -> None:
        """A Windows separator would bypass the '/' based traversal checks."""
        with pytest.raises(ObjectStoreError, match="separators"):
            validate_key("documents\\..\\escape.pdf")

    def test_rejects_overlong_segment(self) -> None:
        with pytest.raises(ObjectStoreError, match="too long"):
            validate_key("d/" + "x" * 201)


class TestFilesystemStore:
    def test_satisfies_the_protocol(self, store: FilesystemStore) -> None:
        assert isinstance(store, ObjectStore)

    def test_put_get_round_trip(self, store: FilesystemStore) -> None:
        key = store.put("documents/a.pdf", b"%PDF-1.7 payload", content_type="application/pdf")
        assert key == "documents/a.pdf"
        assert store.get("documents/a.pdf") == b"%PDF-1.7 payload"

    def test_nested_key_creates_directories(self, store: FilesystemStore) -> None:
        store.put("a/b/c/file.png", b"png")
        assert (store.root / "a" / "b" / "c" / "file.png").is_file()

    def test_binary_and_unicode_payloads_survive(self, store: FilesystemStore) -> None:
        payload = "Voyant orange — 85 °C ✓".encode()
        store.put("documents/é-1.pdf", payload)
        assert store.get("documents/é-1.pdf") == payload

    def test_exists(self, store: FilesystemStore) -> None:
        assert store.exists("documents/missing.pdf") is False
        store.put("documents/present.pdf", b"x")
        assert store.exists("documents/present.pdf") is True

    def test_get_missing_raises(self, store: FilesystemStore) -> None:
        with pytest.raises(ObjectStoreError, match="not found"):
            store.get("documents/missing.pdf")

    def test_delete_is_idempotent(self, store: FilesystemStore) -> None:
        store.put("documents/x.pdf", b"x")
        store.delete("documents/x.pdf")
        store.delete("documents/x.pdf")  # must not raise
        assert store.exists("documents/x.pdf") is False

    def test_overwrite_replaces_content(self, store: FilesystemStore) -> None:
        store.put("documents/x.pdf", b"first")
        store.put("documents/x.pdf", b"second")
        assert store.get("documents/x.pdf") == b"second"

    def test_stat_reports_size_and_content_type(self, store: FilesystemStore) -> None:
        store.put("images/figure.png", b"0123456789")
        info = store.stat("images/figure.png")
        assert isinstance(info, StoredObject)
        assert info.size == 10
        assert info.content_type == "image/png"
        assert info.key == "images/figure.png"

    def test_stat_missing_raises(self, store: FilesystemStore) -> None:
        with pytest.raises(ObjectStoreError, match="not found"):
            store.stat("images/missing.png")

    def test_writes_are_atomic_leaving_no_temp_files(self, store: FilesystemStore) -> None:
        for index in range(5):
            store.put(f"documents/d{index}.pdf", b"x" * 100)
        leftovers = [p.name for p in store.root.rglob(".tmp-*")]
        assert leftovers == []

    def test_traversal_is_refused_by_the_store(self, store: FilesystemStore) -> None:
        with pytest.raises(ObjectStoreError):
            store.put("../escaped.pdf", b"nope")
        assert not (store.root.parent / "escaped.pdf").exists()

    def test_store_creates_its_root(self, tmp_path: Path) -> None:
        fresh = FilesystemStore(root=tmp_path / "deep" / "root")
        assert fresh.root.is_dir()

    def test_empty_payload_is_allowed(self, store: FilesystemStore) -> None:
        store.put("documents/empty.txt", b"")
        assert store.get("documents/empty.txt") == b""


class TestContentTypes:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("a/b.pdf", "application/pdf"),
            ("a/b.PNG", "image/png"),
            ("a/b.jpg", "image/jpeg"),
            ("a/b.json", "application/json"),
            ("a/b.txt", "text/plain"),
            ("a/b.unknown", "application/octet-stream"),
            ("a/b", "application/octet-stream"),
        ],
    )
    def test_guessed_from_extension(self, key: str, expected: str) -> None:
        assert _guess_content_type(key) == expected
