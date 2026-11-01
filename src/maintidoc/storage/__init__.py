"""Storage: objects (documents and figures) and metadata repositories."""

from __future__ import annotations

from maintidoc.storage.objects import (
    FilesystemStore,
    MinioObjectStore,
    ObjectStore,
    ObjectStoreError,
    StoredObject,
    make_minio_store,
    validate_key,
)

__all__ = [
    "FilesystemStore",
    "MinioObjectStore",
    "ObjectStore",
    "ObjectStoreError",
    "StoredObject",
    "make_minio_store",
    "validate_key",
]
