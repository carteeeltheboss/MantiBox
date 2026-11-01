"""Opening documents and figures only for callers allowed to read them (SPEC §4.2, §9).

Retrieval already filters inside the vector query, but a filter on a query cannot
protect a direct fetch by id: `/documents/<id>` with a guessed id would bypass
everything. This module is the second half of AGENTS.md NEVER #7.

Two rules that shape the design:

1. **Deny before fetch.** The permission check happens before the object store is
   touched, so a denied request cannot read bytes even transiently. The test for
   this uses a spy store and asserts *no* read occurred, not merely that an error
   was raised.
2. **Unknown is denied, not "not found".** Answering differently for "does not
   exist" and "not yours" is an enumeration oracle: a caller could probe ids to
   learn which documents exist. Both cases raise `AccessDeniedError`.

Keys are also bound to their document, so a caller cannot pair a permitted
document id with an image key belonging to a restricted one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from maintidoc.errors import AccessDeniedError, ConfigurationError
from maintidoc.observability import access_event, get_logger
from maintidoc.security.access import Principal, require_access
from maintidoc.storage.objects import ObjectStore, ObjectStoreError

logger = get_logger(__name__)

_PREFIX_UNSAFE = re.compile(r"[^a-z0-9_]+")


def image_prefix(document_id: str) -> str:
    """The key prefix a document's rendered images use.

    Must match the convention ingestion writes with, otherwise the key-binding
    check would reject legitimate reads.
    """
    return _PREFIX_UNSAFE.sub("_", document_id.lower())


def key_belongs_to(key: str, document_id: str) -> bool:
    """True when `key` was written for `document_id`."""
    prefix = image_prefix(document_id)
    return key.startswith(f"{prefix}_p") or key.startswith(f"{prefix}/")


@runtime_checkable
class DocumentCatalog(Protocol):
    """Looks up which groups may read a document."""

    def access_groups(self, document_id: str) -> frozenset[str] | None:
        """Groups allowed to read the document, or None if it is unknown.

        None means "not found"; the caller must treat it as a denial so that
        existence is not revealed.
        """
        ...


@dataclass(frozen=True, slots=True)
class InMemoryCatalog:
    """A catalog backed by a plain mapping. Used by tests and small deployments."""

    documents: Mapping[str, frozenset[str]]

    def access_groups(self, document_id: str) -> frozenset[str] | None:
        return self.documents.get(document_id)


def catalog_from_manifest(path: Path) -> InMemoryCatalog:
    """Build a catalog from the corpus manifest written by the generator.

    Raises:
        ConfigurationError: the manifest is missing or malformed.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"corpus manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"corpus manifest is not valid JSON: {path}") from exc

    documents = raw.get("documents")
    if not isinstance(documents, list):
        raise ConfigurationError(f"{path} has no 'documents' list")

    mapping: dict[str, frozenset[str]] = {}
    for entry in documents:
        if not isinstance(entry, dict):
            continue
        document_id = entry.get("document_id")
        groups = entry.get("access_groups", [])
        if not isinstance(document_id, str) or not isinstance(groups, list):
            continue
        mapping[document_id] = frozenset(str(group) for group in groups)

    if not mapping:
        raise ConfigurationError(f"{path} lists no usable documents")
    return InMemoryCatalog(documents=mapping)


@dataclass(frozen=True, slots=True)
class DocumentOpener:
    """Fetches document and figure bytes for an authorized principal only."""

    catalog: DocumentCatalog
    store: ObjectStore

    def open_document(self, *, principal: Principal, document_id: str) -> bytes:
        """Return the PDF bytes for `document_id`.

        Raises:
            AccessDeniedError: unknown document, or the principal lacks a group.
        """
        self._authorize(principal, document_id, resource_type="document", action="open")

        key = f"documents/{document_id}.pdf"
        try:
            data = self.store.get(key)
        except ObjectStoreError as exc:
            # The catalog said this document exists but the object is gone; that
            # is an operational error, not a permission one.
            raise ConfigurationError(f"object missing for document {document_id}") from exc

        access_event(
            logger,
            action="read_document",
            subject_id=principal.user_id,
            resource_type="document",
            resource_id=document_id,
            outcome="allow",
            bytes=len(data),
        )
        return data

    def open_image(self, *, principal: Principal, document_id: str, key: str) -> bytes:
        """Return the bytes of a figure belonging to `document_id`.

        Raises:
            AccessDeniedError: the principal lacks a group, the document is
                unknown, or `key` does not belong to that document.
        """
        if not key_belongs_to(key, document_id):
            # Refuse the pairing outright rather than resolving it: otherwise a
            # permitted id could be used to read a restricted document's figure.
            raise AccessDeniedError("image key does not belong to this document")

        self._authorize(principal, document_id, resource_type="image", action="open")
        try:
            data = self.store.get(key)
        except ObjectStoreError as exc:
            raise ConfigurationError(f"object missing for image {key}") from exc

        access_event(
            logger,
            action="read_image",
            subject_id=principal.user_id,
            resource_type="image",
            resource_id=key,
            outcome="allow",
            document_id=document_id,
            bytes=len(data),
        )
        return data

    # ------------------------------------------------------------------ internals

    def _authorize(
        self,
        principal: Principal,
        document_id: str,
        *,
        resource_type: str,
        action: str,
    ) -> None:
        groups = self.catalog.access_groups(document_id)
        if groups is None:
            access_event(
                logger,
                action=action,
                subject_id=principal.user_id,
                resource_type=resource_type,
                resource_id=document_id,
                outcome="deny",
                reason="unknown_document",
            )
            raise AccessDeniedError(f"access denied to {resource_type}")

        require_access(
            principal,
            groups,
            resource_type=resource_type,
            resource_id=document_id,
            logger=logger,
            action=action,
        )


__all__ = [
    "DocumentCatalog",
    "DocumentOpener",
    "InMemoryCatalog",
    "catalog_from_manifest",
    "image_prefix",
    "key_belongs_to",
]
