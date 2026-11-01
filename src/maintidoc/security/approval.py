"""Human-approval tokens for sensitive actions (SPEC §6).

AGENTS.md NEVER #3: no ticket is created without explicit human approval bound
to *the user* and *the exact content*, plus an idempotency key.

How the binding works:

* `content_hash` is a sha256 over a canonical JSON encoding of the ticket
  fields. Editing one character changes it, so an approval granted for the old
  content no longer validates against the new content.
* The approval token is a short-lived HS256 JWT carrying `sub` (user), `ch`
  (content hash) and `idem`. Verification requires all three to match the
  content and user being submitted *now*.
* `jti` plus an `ApprovalStore` gives replay protection: a token can be spent
  exactly once, so a double-click or a retried request cannot create two
  tickets.
* `idempotency_key` is derived deterministically from (user, content), so a
  legitimate retry of the same approved content presents the same key and the
  MCP server deduplicates it.

The decoding step pins `algorithms=["HS256"]` — without that, a token signed
with a different algorithm could be accepted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Protocol

import jwt
from pydantic import BaseModel, ConfigDict, Field

from maintidoc.errors import ApprovalInvalidError

_ALGORITHM = "HS256"
_DEFAULT_TTL_SECONDS = 900  # 15 minutes: long enough to read, short enough to expire


def canonical_content_hash(
    *,
    equipment_id: str,
    title: str,
    description: str,
) -> str:
    """sha256 over a canonical JSON encoding of the exact ticket content.

    `sort_keys` and `separators` make the encoding byte-stable across Python
    versions and dict orderings, which is what lets a hash computed at draft
    time be compared against one computed at approval time.
    """
    payload = json.dumps(
        {"description": description, "equipment_id": equipment_id, "title": title},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_idempotency_key(*, user_id: str, content_hash: str) -> str:
    """Deterministic key so a retry of the same approved content deduplicates.

    A random key would defeat the purpose: every retry would look like a new
    request to the maintenance service.
    """
    digest = hashlib.sha256(f"{user_id}|{content_hash}".encode()).hexdigest()
    return f"idem-{digest[:40]}"


class ApprovalGrant(BaseModel):
    """A token that has been issued and verified."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: str
    token_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=64, max_length=64)
    idempotency_key: str = Field(min_length=16)
    issued_at: datetime
    expires_at: datetime


class ApprovalStore(Protocol):
    """Single-use tracking for approval tokens.

    `mark_used` must be atomic: two concurrent requests holding the same token
    must not both succeed.
    """

    def mark_used(self, token_id: str, *, ttl_seconds: int) -> bool:
        """Record a spend. Returns False if the token was already used."""
        ...

    def is_used(self, token_id: str) -> bool: ...


class InMemoryApprovalStore:
    """Process-local store. Correct for one worker; use the Postgres store when
    the API runs with several replicas."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._used: dict[str, datetime] = {}

    def mark_used(self, token_id: str, *, ttl_seconds: int) -> bool:
        now = datetime.now(UTC)
        with self._lock:
            self._evict_expired(now)
            if token_id in self._used:
                return False
            self._used[token_id] = now + timedelta(seconds=ttl_seconds)
            return True

    def is_used(self, token_id: str) -> bool:
        with self._lock:
            self._evict_expired(datetime.now(UTC))
            return token_id in self._used

    def _evict_expired(self, now: datetime) -> None:
        expired = [k for k, until in self._used.items() if until <= now]
        for key in expired:
            del self._used[key]


class ApprovalService:
    """Issues and verifies approval tokens."""

    def __init__(
        self,
        *,
        secret_key: str,
        store: ApprovalStore,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        if len(secret_key) < 32:
            msg = "approval signing key must be at least 32 characters"
            raise ApprovalInvalidError(msg)
        self._secret = secret_key
        self._store = store
        self._ttl = ttl_seconds

    def issue(
        self,
        *,
        user_id: str,
        content_hash: str,
        idempotency_key: str,
    ) -> ApprovalGrant:
        """Create a token bound to this user and this exact content."""
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=self._ttl)
        token_id = uuid.uuid4().hex
        claims = {
            "sub": user_id,
            "ch": content_hash,
            "idem": idempotency_key,
            "jti": token_id,
            "iat": now,
            "exp": expires,
            "purpose": "ticket_creation",
        }
        token = jwt.encode(claims, self._secret, algorithm=_ALGORITHM)
        return ApprovalGrant(
            token=token,
            token_id=token_id,
            user_id=user_id,
            content_hash=content_hash,
            idempotency_key=idempotency_key,
            issued_at=now,
            expires_at=expires,
        )

    def verify(
        self,
        token: str,
        *,
        user_id: str,
        content_hash: str,
        idempotency_key: str,
        spend: bool = True,
    ) -> ApprovalGrant:
        """Validate a token against the user and content presented *now*.

        Args:
            spend: consume the token. Pass False only for a dry-run check; the
                creation path must spend it.

        Raises:
            ApprovalInvalidError: on any mismatch, expiry, bad signature,
                unexpected purpose, or replay.
        """
        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=[_ALGORITHM],
                options={"require": ["exp", "iat", "sub", "jti"]},
            )
        except jwt.PyJWTError as exc:
            raise ApprovalInvalidError("approval token rejected") from exc

        if claims.get("purpose") != "ticket_creation":
            raise ApprovalInvalidError("approval token has the wrong purpose")

        # Constant-time comparisons: the hash is a bearer secret for this action.
        if not hmac.compare_digest(str(claims.get("sub", "")), user_id):
            raise ApprovalInvalidError("approval token was issued to a different user")
        if not hmac.compare_digest(str(claims.get("ch", "")), content_hash):
            raise ApprovalInvalidError(
                "ticket content changed after approval; re-approval is required"
            )
        if not hmac.compare_digest(str(claims.get("idem", "")), idempotency_key):
            raise ApprovalInvalidError("idempotency key does not match the approval")

        token_id = str(claims["jti"])
        expires_at = datetime.fromtimestamp(int(claims["exp"]), tz=UTC)
        issued_at = datetime.fromtimestamp(int(claims["iat"]), tz=UTC)

        if spend and not self._store.mark_used(token_id, ttl_seconds=self._ttl):
            raise ApprovalInvalidError("approval token was already used")

        return ApprovalGrant(
            token=token,
            token_id=token_id,
            user_id=user_id,
            content_hash=content_hash,
            idempotency_key=idempotency_key,
            issued_at=issued_at,
            expires_at=expires_at,
        )


def hash_allowed_groups(groups: Iterable[str]) -> str:
    """Stable digest of a group set, for logging without enumerating groups."""
    joined = ",".join(sorted(set(groups)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ApprovalGrant",
    "ApprovalService",
    "ApprovalStore",
    "InMemoryApprovalStore",
    "canonical_content_hash",
    "derive_idempotency_key",
    "hash_allowed_groups",
]
