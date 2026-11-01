"""Security: identity, access control, authentication and action approval.

These modules are security-critical. AGENTS.md requires a human to review every
diff that touches them.
"""

from __future__ import annotations

from maintidoc.security.access import (
    ACCESS_GROUPS_FIELD,
    Principal,
    build_access_filter,
    require_access,
)
from maintidoc.security.approval import (
    ApprovalGrant,
    ApprovalService,
    ApprovalStore,
    InMemoryApprovalStore,
    canonical_content_hash,
    derive_idempotency_key,
)

__all__ = [
    "ACCESS_GROUPS_FIELD",
    "ApprovalGrant",
    "ApprovalService",
    "ApprovalStore",
    "InMemoryApprovalStore",
    "Principal",
    "build_access_filter",
    "canonical_content_hash",
    "derive_idempotency_key",
    "require_access",
]
