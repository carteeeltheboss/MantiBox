"""Access control (SPEC §4.2, §9) — AGENTS.md NEVER #7.

Security-critical: a human reviews every diff here.
"""

from __future__ import annotations

import inspect
import logging

import pytest
from qdrant_client import models as qm

from maintidoc.errors import AccessDeniedError
from maintidoc.observability import access_event
from maintidoc.security.access import (
    ACCESS_GROUPS_FIELD,
    Principal,
    build_access_filter,
    require_access,
)

SITE_A_DOC = frozenset({"site_a", "maintenance"})
RESTRICTED_DOC = frozenset({"supervisors"})
ORPHAN_DOC: frozenset[str] = frozenset()


class TestCanAccess:
    def test_shared_group_grants_access(self, technician: Principal, supervisor: Principal) -> None:
        assert technician.can_access(SITE_A_DOC) is True
        assert supervisor.can_access(SITE_A_DOC) is True

    def test_no_shared_group_denies(self, outsider: Principal) -> None:
        assert outsider.can_access(SITE_A_DOC) is False

    def test_restricted_doc_denied_to_plain_technician(self, technician: Principal) -> None:
        assert technician.can_access(RESTRICTED_DOC) is False

    def test_supervisor_sees_restricted_doc(self, supervisor: Principal) -> None:
        assert supervisor.can_access(RESTRICTED_DOC) is True

    def test_empty_allowed_groups_denies_everyone(self, supervisor: Principal) -> None:
        """Deny by default: an unlabelled document is not a public document."""
        assert supervisor.can_access(ORPHAN_DOC) is False

    def test_deactivated_principal_denied_even_with_matching_groups(
        self, deactivated: Principal
    ) -> None:
        assert deactivated.groups == {"maintenance", "site_a"}
        assert deactivated.can_access(SITE_A_DOC) is False

    def test_principal_with_no_groups_denied_everything(self, no_groups: Principal) -> None:
        assert no_groups.can_access(SITE_A_DOC) is False
        assert no_groups.can_access({"public"}) is False


class TestBuildAccessFilter:
    def test_filter_matches_any_of_the_principals_groups(self, technician: Principal) -> None:
        result = build_access_filter(technician)
        assert isinstance(result, qm.Filter)
        assert len(result.must) == 1
        condition = result.must[0]
        assert isinstance(condition, qm.FieldCondition)
        assert condition.key == ACCESS_GROUPS_FIELD
        assert isinstance(condition.match, qm.MatchAny)
        # Sorted so the built filter is stable and comparable in tests.
        assert condition.match.any == ["maintenance", "site_a"]

    def test_filter_is_applied_as_must_not_should(self, technician: Principal) -> None:
        """A `should` clause would be OR'd with other filters and could widen
        the result set. Access narrowing must be a `must`."""
        result = build_access_filter(technician)
        assert isinstance(result, qm.Filter)
        assert result.must
        assert not result.should

    def test_no_groups_raises_instead_of_building_a_permissive_filter(
        self, no_groups: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError, match="no access groups"):
            build_access_filter(no_groups)

    def test_inactive_principal_raises(self, deactivated: Principal) -> None:
        with pytest.raises(AccessDeniedError, match="inactive"):
            build_access_filter(deactivated)


class TestRequireAccess:
    def test_grants_and_logs_allow(
        self, technician: Principal, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = logging.getLogger("test.access.allow")
        with caplog.at_level(logging.INFO, logger=logger.name):
            require_access(
                technician,
                SITE_A_DOC,
                resource_type="document",
                resource_id="doc-42",
                logger=logger,
            )
        record = caplog.records[-1]
        assert record.outcome == "allow"  # type: ignore[attr-defined]
        assert record.subject_id == "u-tech-1"  # type: ignore[attr-defined]
        assert record.resource_id == "doc-42"  # type: ignore[attr-defined]

    def test_denies_and_raises(self, outsider: Principal) -> None:
        with pytest.raises(AccessDeniedError, match="access denied"):
            require_access(
                outsider,
                SITE_A_DOC,
                resource_type="document",
                resource_id="doc-42",
            )

    def test_denial_is_logged_as_an_access_event(
        self, outsider: Principal, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SPEC §9 requires access events; a silent denial cannot be audited."""
        logger = logging.getLogger("test.access.deny")
        with (
            caplog.at_level(logging.INFO, logger=logger.name),
            pytest.raises(AccessDeniedError),
        ):
            require_access(
                outsider,
                SITE_A_DOC,
                resource_type="image",
                resource_id="img-7",
                logger=logger,
            )
        record = caplog.records[-1]
        assert record.outcome == "deny"  # type: ignore[attr-defined]
        assert record.reason == "no_shared_group"  # type: ignore[attr-defined]
        assert record.resource_type == "image"  # type: ignore[attr-defined]

    def test_orphan_document_denied_on_open(self, supervisor: Principal) -> None:
        with pytest.raises(AccessDeniedError):
            require_access(
                supervisor,
                ORPHAN_DOC,
                resource_type="document",
                resource_id="doc-orphan",
            )


class TestNoContentInAccessLogs:
    """SPEC §9: logs carry metadata, never raw document/image/user content."""

    def test_access_event_exposes_no_content_parameter(self) -> None:
        params = set(inspect.signature(access_event).parameters)
        forbidden = {"content", "text", "body", "raw", "image", "question", "payload", "data"}
        assert not (params & forbidden), f"access_event must not accept {params & forbidden}"

    def test_logged_record_carries_only_metadata(
        self, technician: Principal, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = logging.getLogger("test.access.metadata")
        with caplog.at_level(logging.INFO, logger=logger.name):
            require_access(
                technician,
                SITE_A_DOC,
                resource_type="document",
                resource_id="doc-42",
                logger=logger,
            )
        record = caplog.records[-1]

        # The rendered line is a fixed phrase: identifiers travel in `extra`,
        # never interpolated into free text that could also carry content.
        assert record.getMessage() == "access open"

        # Group names are resource metadata and must not be echoed into the log.
        emitted_keys = {
            key
            for key, value in record.__dict__.items()
            if isinstance(value, str) and value in {"maintenance", "site_a"}
        }
        assert not emitted_keys
