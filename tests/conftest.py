"""Shared fixtures.

Nothing here touches the network, a GPU or a model: the fast gate must run on a
laptop and in CI with no services up.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from maintidoc.security.access import Principal

#: 64+ chars, clearly a test value. Never used outside tests. Long enough that
#: PyJWT does not warn about a short HMAC key when a test forges an HS512 token.
TEST_SIGNING_KEY = "unit-test-signing-key-0123456789-abcdefghijklmnopqrstuvwxyz-0123456789"


@pytest.fixture
def signing_key() -> str:
    return TEST_SIGNING_KEY


@pytest.fixture
def api_key() -> SecretStr:
    return SecretStr("test-mcp-api-key-0123456789-abcdefghijklmno")


@pytest.fixture
def technician() -> Principal:
    """Ordinary maintenance technician on site A."""
    return Principal(user_id="u-tech-1", groups=frozenset({"maintenance", "site_a"}))


@pytest.fixture
def supervisor() -> Principal:
    """Supervisor: sees everything the technician sees, plus restricted docs."""
    return Principal(
        user_id="u-sup-1",
        groups=frozenset({"maintenance", "supervisors", "site_a"}),
    )


@pytest.fixture
def outsider() -> Principal:
    """A different site. Must never see site_a material."""
    return Principal(user_id="u-out-1", groups=frozenset({"site_b"}))


@pytest.fixture
def deactivated(technician: Principal) -> Principal:
    """Same groups as the technician, but the account is disabled."""
    return technician.model_copy(update={"is_active": False})


@pytest.fixture
def no_groups() -> Principal:
    """A principal with no groups at all — the deny-by-default edge case."""
    return Principal(user_id="u-none-1", groups=frozenset())
