"""Approval tokens and idempotency (SPEC §6) — AGENTS.md NEVER #3.

Security-critical: a human reviews every diff here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from maintidoc.errors import ApprovalInvalidError
from maintidoc.security.approval import (
    ApprovalService,
    InMemoryApprovalStore,
    canonical_content_hash,
    derive_idempotency_key,
    hash_allowed_groups,
)

EQUIPMENT = "pompe-a-01"
TITLE = "Voyant orange au panneau de commande"
DESCRIPTION = "Alarme de température relevée sur site, intervention à planifier."


@pytest.fixture
def content_hash() -> str:
    return canonical_content_hash(equipment_id=EQUIPMENT, title=TITLE, description=DESCRIPTION)


@pytest.fixture
def idempotency_key(content_hash: str) -> str:
    return derive_idempotency_key(user_id="u-tech-1", content_hash=content_hash)


@pytest.fixture
def service(signing_key: str) -> ApprovalService:
    return ApprovalService(secret_key=signing_key, store=InMemoryApprovalStore())


class TestContentHash:
    def test_is_deterministic(self, content_hash: str) -> None:
        again = canonical_content_hash(equipment_id=EQUIPMENT, title=TITLE, description=DESCRIPTION)
        assert again == content_hash

    def test_is_64_lowercase_hex(self, content_hash: str) -> None:
        assert len(content_hash) == 64
        assert all(c in "0123456789abcdef" for c in content_hash)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("equipment_id", "pompe-a-02"),
            ("title", TITLE + " "),
            ("title", "voyant orange au panneau de commande"),
            ("description", DESCRIPTION + "."),
        ],
    )
    def test_any_edit_changes_the_hash(self, content_hash: str, field: str, value: str) -> None:
        """SPEC §6: any modification forces a new approval."""
        kwargs = {"equipment_id": EQUIPMENT, "title": TITLE, "description": DESCRIPTION}
        kwargs[field] = value
        assert canonical_content_hash(**kwargs) != content_hash

    def test_unicode_content_hashes_stably(self) -> None:
        a = canonical_content_hash(equipment_id="é-1", title="Ångström", description="café")
        b = canonical_content_hash(equipment_id="é-1", title="Ångström", description="café")
        assert a == b
        assert a != canonical_content_hash(equipment_id="e-1", title="Ångström", description="café")


class TestIdempotencyKey:
    def test_is_deterministic_for_same_user_and_content(
        self, content_hash: str, idempotency_key: str
    ) -> None:
        assert (
            derive_idempotency_key(user_id="u-tech-1", content_hash=content_hash) == idempotency_key
        )

    def test_differs_per_user(self, content_hash: str, idempotency_key: str) -> None:
        other = derive_idempotency_key(user_id="u-tech-2", content_hash=content_hash)
        assert other != idempotency_key

    def test_differs_per_content(self, idempotency_key: str) -> None:
        other_hash = canonical_content_hash(
            equipment_id=EQUIPMENT, title="Autre titre", description=DESCRIPTION
        )
        assert (
            derive_idempotency_key(user_id="u-tech-1", content_hash=other_hash) != idempotency_key
        )

    def test_meets_the_contract_minimum_length(self, idempotency_key: str) -> None:
        assert len(idempotency_key) >= 16


class TestIssueAndVerify:
    def test_round_trip(
        self, service: ApprovalService, content_hash: str, idempotency_key: str
    ) -> None:
        grant = service.issue(
            user_id="u-tech-1", content_hash=content_hash, idempotency_key=idempotency_key
        )
        verified = service.verify(
            grant.token,
            user_id="u-tech-1",
            content_hash=content_hash,
            idempotency_key=idempotency_key,
        )
        assert verified.token_id == grant.token_id
        assert verified.content_hash == content_hash

    def test_token_binds_the_user(
        self, service: ApprovalService, content_hash: str, idempotency_key: str
    ) -> None:
        grant = service.issue(
            user_id="u-tech-1", content_hash=content_hash, idempotency_key=idempotency_key
        )
        with pytest.raises(ApprovalInvalidError, match="different user"):
            service.verify(
                grant.token,
                user_id="u-tech-2",
                content_hash=content_hash,
                idempotency_key=idempotency_key,
            )

    def test_edited_content_invalidates_the_approval(
        self, service: ApprovalService, content_hash: str, idempotency_key: str
    ) -> None:
        grant = service.issue(
            user_id="u-tech-1", content_hash=content_hash, idempotency_key=idempotency_key
        )
        edited = canonical_content_hash(
            equipment_id=EQUIPMENT, title="Titre modifié après approbation", description=DESCRIPTION
        )
        with pytest.raises(ApprovalInvalidError, match="re-approval is required"):
            service.verify(
                grant.token,
                user_id="u-tech-1",
                content_hash=edited,
                idempotency_key=idempotency_key,
            )

    def test_mismatched_idempotency_key_rejected(
        self, service: ApprovalService, content_hash: str, idempotency_key: str
    ) -> None:
        grant = service.issue(
            user_id="u-tech-1", content_hash=content_hash, idempotency_key=idempotency_key
        )
        with pytest.raises(ApprovalInvalidError, match="idempotency"):
            service.verify(
                grant.token,
                user_id="u-tech-1",
                content_hash=content_hash,
                idempotency_key=derive_idempotency_key(user_id="u-tech-1", content_hash="0" * 64),
            )


class TestReplayProtection:
    def test_a_spent_token_cannot_be_reused(
        self, service: ApprovalService, content_hash: str, idempotency_key: str
    ) -> None:
        grant = service.issue(
            user_id="u-tech-1", content_hash=content_hash, idempotency_key=idempotency_key
        )
        service.verify(
            grant.token,
            user_id="u-tech-1",
            content_hash=content_hash,
            idempotency_key=idempotency_key,
        )
        with pytest.raises(ApprovalInvalidError, match="already used"):
            service.verify(
                grant.token,
                user_id="u-tech-1",
                content_hash=content_hash,
                idempotency_key=idempotency_key,
            )

    def test_dry_run_does_not_consume(
        self, service: ApprovalService, content_hash: str, idempotency_key: str
    ) -> None:
        grant = service.issue(
            user_id="u-tech-1", content_hash=content_hash, idempotency_key=idempotency_key
        )
        for _ in range(3):
            service.verify(
                grant.token,
                user_id="u-tech-1",
                content_hash=content_hash,
                idempotency_key=idempotency_key,
                spend=False,
            )
        # Still spendable exactly once afterwards.
        service.verify(
            grant.token,
            user_id="u-tech-1",
            content_hash=content_hash,
            idempotency_key=idempotency_key,
        )
        with pytest.raises(ApprovalInvalidError):
            service.verify(
                grant.token,
                user_id="u-tech-1",
                content_hash=content_hash,
                idempotency_key=idempotency_key,
            )


class TestTokenForgery:
    def test_expired_token_rejected(
        self,
        service: ApprovalService,
        signing_key: str,
        content_hash: str,
        idempotency_key: str,
    ) -> None:
        grant = service.issue(
            user_id="u-tech-1", content_hash=content_hash, idempotency_key=idempotency_key
        )
        claims = jwt.decode(grant.token, signing_key, algorithms=["HS256"])
        past = int(datetime.now(UTC).timestamp()) - 3600
        claims["exp"] = past
        claims["iat"] = past - 900
        expired = jwt.encode(claims, signing_key, algorithm="HS256")
        with pytest.raises(ApprovalInvalidError, match="rejected"):
            service.verify(
                expired,
                user_id="u-tech-1",
                content_hash=content_hash,
                idempotency_key=idempotency_key,
            )

    def test_token_signed_with_a_different_algorithm_is_rejected(
        self,
        service: ApprovalService,
        signing_key: str,
        content_hash: str,
        idempotency_key: str,
    ) -> None:
        """Algorithm pinning: an HS512 token must not verify against an HS256 key."""
        claims = {
            "sub": "u-tech-1",
            "ch": content_hash,
            "idem": idempotency_key,
            "jti": "deadbeef" * 4,
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC),
            "purpose": "ticket_creation",
        }
        forged = jwt.encode(claims, signing_key, algorithm="HS512")
        with pytest.raises(ApprovalInvalidError):
            service.verify(
                forged,
                user_id="u-tech-1",
                content_hash=content_hash,
                idempotency_key=idempotency_key,
            )

    def test_tampered_signature_rejected(
        self, service: ApprovalService, content_hash: str, idempotency_key: str
    ) -> None:
        grant = service.issue(
            user_id="u-tech-1", content_hash=content_hash, idempotency_key=idempotency_key
        )
        head, _, _ = grant.token.partition(".")
        tampered = f"{head}.eyJzdWIiOiJ1LXRlY2gtMSJ9.badsignature"
        with pytest.raises(ApprovalInvalidError):
            service.verify(
                tampered,
                user_id="u-tech-1",
                content_hash=content_hash,
                idempotency_key=idempotency_key,
            )

    def test_garbage_token_rejected(self, service: ApprovalService, content_hash: str) -> None:
        with pytest.raises(ApprovalInvalidError):
            service.verify(
                "not.a.jwt",
                user_id="u-tech-1",
                content_hash=content_hash,
                idempotency_key="idem-0123456789abcdef",
            )

    def test_wrong_purpose_rejected(
        self,
        service: ApprovalService,
        signing_key: str,
        content_hash: str,
        idempotency_key: str,
    ) -> None:
        now = datetime.now(UTC)
        claims = {
            "sub": "u-tech-1",
            "ch": content_hash,
            "idem": idempotency_key,
            "jti": "cafe" * 8,
            "iat": now,
            # Must be in the future, otherwise the token is rejected as expired
            # and this test would pass without exercising the purpose check.
            "exp": now + timedelta(minutes=10),
            "purpose": "something_else",
        }
        token = jwt.encode(claims, signing_key, algorithm="HS256")
        with pytest.raises(ApprovalInvalidError, match="wrong purpose"):
            service.verify(
                token,
                user_id="u-tech-1",
                content_hash=content_hash,
                idempotency_key=idempotency_key,
            )


class TestServiceConstruction:
    def test_short_signing_key_refused(self) -> None:
        with pytest.raises(ApprovalInvalidError, match="at least 32"):
            ApprovalService(secret_key="too-short", store=InMemoryApprovalStore())


class TestInMemoryStore:
    def test_mark_used_is_single_shot(self) -> None:
        store = InMemoryApprovalStore()
        assert store.mark_used("t1", ttl_seconds=60) is True
        assert store.mark_used("t1", ttl_seconds=60) is False
        assert store.is_used("t1") is True

    def test_expired_entries_are_evicted(self) -> None:
        """ttl_seconds=0 makes the entry immediately evictable on the next call."""
        store = InMemoryApprovalStore()
        assert store.mark_used("t2", ttl_seconds=0) is True
        assert store.mark_used("t2", ttl_seconds=0) is True

    def test_distinct_tokens_do_not_collide(self) -> None:
        store = InMemoryApprovalStore()
        assert store.mark_used("a", ttl_seconds=60) is True
        assert store.mark_used("b", ttl_seconds=60) is True


class TestGroupHash:
    def test_is_order_independent(self) -> None:
        assert hash_allowed_groups(["b", "a"]) == hash_allowed_groups(["a", "b"])

    def test_differs_for_different_groups(self) -> None:
        assert hash_allowed_groups(["a"]) != hash_allowed_groups(["b"])
