"""Identity and access control (SPEC §4.1, §4.2, §9).

**Deny by default.** An element whose `access_groups` is empty is visible to
nobody; ingestion must assign groups explicitly. AGENTS.md NEVER #7 forbids
retrieving or displaying a document or image without the access filter, so the
filter is applied in two places and neither is optional:

1. inside the vector query itself (`build_access_filter`) — the model never even
   sees vectors the caller may not read;
2. again when a document or image is opened (`require_access`) — because a
   retrieval filter cannot protect a direct fetch by id.

A principal with no groups cannot be given a "match nothing" Qdrant filter
safely: an empty `MatchAny` is not reliably a no-match across versions. The
builder therefore raises instead, so the failure is loud rather than permissive.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Set

from pydantic import BaseModel, ConfigDict, Field

from maintidoc.errors import AccessDeniedError
from maintidoc.observability import access_event

#: Payload key under which the allowed groups are stored on every point.
ACCESS_GROUPS_FIELD = "access_groups"


class Principal(BaseModel):
    """An authenticated caller: a user plus the groups they belong to.

    Users and groups are fictional in this project (SPEC §2) but the checks are
    real, because the whole point of the exercise is that they hold.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    groups: frozenset[str] = frozenset()
    display_name: str | None = Field(default=None, max_length=200)
    is_active: bool = True

    @property
    def sorted_groups(self) -> tuple[str, ...]:
        return tuple(sorted(self.groups))

    def can_access(self, allowed_groups: Iterable[str]) -> bool:
        """True only if the principal is active and shares a group."""
        if not self.is_active:
            return False
        allowed: Set[str] = set(allowed_groups)
        if not allowed:
            # Empty means "nobody", not "everybody".
            return False
        return bool(allowed & self.groups)


def build_access_filter(principal: Principal) -> object:
    """Return the Qdrant filter that restricts a search to `principal`.

    Typed as `object` so this module does not force a qdrant-client import on
    every caller (and so unit tests run without the dependency); the retrieval
    layer narrows it to `qdrant_client.models.Filter`.

    Raises:
        AccessDeniedError: if the principal is inactive or belongs to no group.
    """
    from qdrant_client import models as qm

    if not principal.is_active:
        msg = "principal is inactive"
        raise AccessDeniedError(msg)
    if not principal.groups:
        msg = "principal has no access groups; refusing to build a permissive filter"
        raise AccessDeniedError(msg)

    return qm.Filter(
        must=[
            qm.FieldCondition(
                key=ACCESS_GROUPS_FIELD,
                # MatchAny is typed for list, not tuple; sorted so the filter is
                # byte-stable and comparable in tests.
                match=qm.MatchAny(any=list(principal.sorted_groups)),
            )
        ]
    )


def require_access(
    principal: Principal,
    allowed_groups: Iterable[str],
    *,
    resource_type: str,
    resource_id: str,
    logger: logging.Logger | None = None,
    action: str = "open",
) -> None:
    """Gate a direct open of a document, page or image.

    Logs an access event either way — SPEC §9 requires access events, and a
    denial that leaves no trace cannot be audited or tested.

    Raises:
        AccessDeniedError: when access is refused.
    """
    allowed = set(allowed_groups)
    granted = principal.can_access(allowed)
    if logger is not None:
        access_event(
            logger,
            action=action,
            subject_id=principal.user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome="allow" if granted else "deny",
            reason=None if granted else "no_shared_group",
        )
    if not granted:
        msg = f"access denied to {resource_type}"
        raise AccessDeniedError(msg)


__all__ = [
    "ACCESS_GROUPS_FIELD",
    "Principal",
    "build_access_filter",
    "require_access",
]
