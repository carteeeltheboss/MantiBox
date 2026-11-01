"""Object storage behind an interface (SPEC §3: "MinIO pour les fichiers").

Why an interface rather than a direct MinIO client:

MinIO's public images are currently unpullable (Docker Hub denies access,
quay.io has no manifest, and the binary host returns 410), so the object-store
choice is still open — see docs/STATUS.md. Everything above this module depends
on `ObjectStore`, not on a vendor, so the deployment can land on MinIO,
SeaweedFS or real S3 without touching ingestion, retrieval or the API.

Two backends:

* `FilesystemStore` — the default. No container, no credentials, works on any
  machine. Keys are validated so a crafted document id cannot escape the root.
* `MinioObjectStore` — the S3 backend, used for any S3-compatible endpoint.
  Imported lazily so the `minio` client is only loaded when selected.

Documents and images never pass through logs. Callers log object *keys* and
content fingerprints, never bytes (SPEC §9).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from maintidoc.observability import get_logger

logger = get_logger(__name__)


class ObjectStoreError(Exception):
    """Storage failed. Carries a message safe to log (no content)."""


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Metadata about a stored object. Never contains the bytes."""

    key: str
    size: int
    content_type: str


@runtime_checkable
class ObjectStore(Protocol):
    """Minimal object-store contract: put, get, exists, delete."""

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        """Store `data` under `key`; return the key. Overwrites."""
        ...

    def get(self, key: str) -> bytes:
        """Return the bytes for `key`. Raises ObjectStoreError if missing."""
        ...

    def exists(self, key: str) -> bool:
        """True if `key` is present."""
        ...

    def delete(self, key: str) -> None:
        """Remove `key`. Absent keys are not an error."""
        ...

    def stat(self, key: str) -> StoredObject:
        """Metadata for `key`. Raises ObjectStoreError if missing."""
        ...


_REJECTED_PARTS = frozenset({"", ".", ".."})


def validate_key(key: str) -> str:
    """Normalise and validate an object key.

    Rejects anything that could escape the store root or confuse a path join:
    absolute paths, drive letters, backslashes, empty segments, `.`/`..`, and
    trailing separators. Keys are always forward-slash separated (S3 style),
    which is also what keeps `FilesystemStore` safe on Windows.
    """
    if not key or not key.strip():
        raise ObjectStoreError("object key must not be empty")
    if "\\" in key:
        raise ObjectStoreError("object key must use '/' separators, not backslashes")
    if key.startswith("/") or (len(key) > 1 and key[1] == ":"):
        raise ObjectStoreError("object key must be relative")
    parts = key.split("/")
    for part in parts:
        if part in _REJECTED_PARTS:
            raise ObjectStoreError(f"object key contains an invalid segment: {part!r}")
    if any(len(part) > 200 for part in parts):
        raise ObjectStoreError("object key segment is too long")
    return key


@dataclass(frozen=True, slots=True)
class FilesystemStore:
    """Store objects as files under `root`.

    Writes are atomic (temp file in the same directory, then `os.replace`) so a
    crash mid-write cannot leave a truncated PDF that later fails to parse.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        validate_key(key)
        candidate = (self.root / key).resolve()
        root = self.root.resolve()
        if not candidate.is_relative_to(root):
            # Defence in depth: validate_key already rejects traversal, but a
            # symlink inside the root could still redirect outside it.
            raise ObjectStoreError("resolved object path escapes the store root")
        return candidate

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
            Path(temp_name).replace(target)
        except OSError as exc:
            Path(temp_name).unlink(missing_ok=True)
            raise ObjectStoreError(f"failed to write object: {exc.__class__.__name__}") from exc
        return key

    def get(self, key: str) -> bytes:
        target = self._path(key)
        try:
            return target.read_bytes()
        except FileNotFoundError as exc:
            raise ObjectStoreError("object not found") from exc
        except OSError as exc:
            raise ObjectStoreError(f"failed to read object: {exc.__class__.__name__}") from exc

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def stat(self, key: str) -> StoredObject:
        target = self._path(key)
        try:
            size = target.stat().st_size
        except FileNotFoundError as exc:
            raise ObjectStoreError("object not found") from exc
        return StoredObject(key=key, size=size, content_type=_guess_content_type(key))


@dataclass(frozen=True, slots=True)
class MinioObjectStore:
    """S3-compatible backend (MinIO, SeaweedFS, real S3).

    Constructed with `make_minio_store` so the client is imported lazily and a
    missing dependency only fails when this backend is actually selected.
    """

    client: object
    bucket: str

    def _ensure_bucket(self) -> None:
        from minio import Minio
        from minio.error import S3Error

        assert isinstance(self.client, Minio)  # noqa: S101 - invariant of the factory
        if not self.client.bucket_exists(self.bucket):
            try:
                self.client.make_bucket(self.bucket)
            except S3Error as exc:
                raise ObjectStoreError(f"cannot create bucket: {exc.code}") from exc

    def put(self, key: str, data: bytes, *, content_type: str = "application/octet-stream") -> str:
        import io

        validate_key(key)
        self._ensure_bucket()
        from minio import Minio

        assert isinstance(self.client, Minio)  # noqa: S101
        try:
            self.client.put_object(
                self.bucket,
                key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
        except Exception as exc:  # minio raises a broad set of transport errors
            raise ObjectStoreError(f"failed to put object: {exc.__class__.__name__}") from exc
        return key

    def get(self, key: str) -> bytes:
        from minio import Minio
        from minio.error import S3Error

        validate_key(key)
        assert isinstance(self.client, Minio)  # noqa: S101
        response = None
        try:
            response = self.client.get_object(self.bucket, key)
            return response.read()
        except S3Error as exc:
            raise ObjectStoreError(f"object not readable: {exc.code}") from exc
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def exists(self, key: str) -> bool:
        from minio import Minio
        from minio.error import S3Error

        validate_key(key)
        assert isinstance(self.client, Minio)  # noqa: S101
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except S3Error:
            return False

    def delete(self, key: str) -> None:
        from minio import Minio

        validate_key(key)
        assert isinstance(self.client, Minio)  # noqa: S101
        self.client.remove_object(self.bucket, key)

    def stat(self, key: str) -> StoredObject:
        from minio import Minio
        from minio.error import S3Error

        validate_key(key)
        assert isinstance(self.client, Minio)  # noqa: S101
        try:
            result = self.client.stat_object(self.bucket, key)
        except S3Error as exc:
            raise ObjectStoreError(f"object not found: {exc.code}") from exc
        return StoredObject(
            key=key,
            size=int(result.size or 0),
            content_type=str(result.content_type or _guess_content_type(key)),
        )


def make_minio_store(
    *,
    endpoint: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    secure: bool = False,
) -> MinioObjectStore:
    """Build an S3 store. Fails with ObjectStoreError if the client is absent."""
    try:
        from minio import Minio
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ObjectStoreError("the 'minio' client is not installed") from exc

    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )
    return MinioObjectStore(client=client, bucket=bucket)


_EXTENSION_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".txt": "text/plain",
}


def _guess_content_type(key: str) -> str:
    return _EXTENSION_TYPES.get(Path(key).suffix.lower(), "application/octet-stream")


__all__ = [
    "FilesystemStore",
    "MinioObjectStore",
    "ObjectStore",
    "ObjectStoreError",
    "StoredObject",
    "make_minio_store",
    "validate_key",
]
