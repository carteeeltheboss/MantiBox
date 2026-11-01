"""Domain exceptions.

Every exception that can cross the API boundary carries a stable `code` and a
`safe_detail` that is free of document content, user content and secrets
(SPEC §9: logs and responses carry technical metadata only).
"""

from __future__ import annotations


class MaintiDocError(Exception):
    """Base class for all domain errors."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, safe_detail: str = "", *, cause: BaseException | None = None) -> None:
        super().__init__(safe_detail or self.code)
        self.safe_detail = safe_detail or self.code
        self.cause = cause


class ConfigurationError(MaintiDocError):
    """Invalid or missing configuration; the app must not start."""

    code = "configuration_error"
    http_status = 500


class AuthenticationError(MaintiDocError):
    """Missing, expired or malformed credentials."""

    code = "unauthenticated"
    http_status = 401


class AccessDeniedError(MaintiDocError):
    """The caller may not see this resource. Applies to retrieval AND to
    opening a document or image (SPEC §4.2, §9)."""

    code = "access_denied"
    http_status = 403


class ValidationError(MaintiDocError):
    """A request or a model output failed contract validation."""

    code = "validation_error"
    http_status = 422


class BudgetExceededError(MaintiDocError):
    """A SPEC §9 budget was hit: MCP calls, model calls or wall-clock."""

    code = "budget_exceeded"
    http_status = 503


class ApprovalRequiredError(MaintiDocError):
    """A sensitive action was attempted without a valid, bound approval."""

    code = "approval_required"
    http_status = 403


class ApprovalInvalidError(MaintiDocError):
    """The approval token does not match the user or the exact content, or was
    replayed. Any edit to the content invalidates it (SPEC §6)."""

    code = "approval_invalid"
    http_status = 403


class ToolAuthorizationError(MaintiDocError):
    """The MCP server refused a call: authorization is checked server-side on
    every single call (AGENTS.md NEVER #2)."""

    code = "tool_unauthorized"
    http_status = 403


class ModelUnavailableError(MaintiDocError):
    """No inference provider could be started on this machine."""

    code = "model_unavailable"
    http_status = 503


class CitationVerificationError(MaintiDocError):
    """A citation was not actually retrieved, or does not support the claim
    (AGENTS.md NEVER #5)."""

    code = "citation_invalid"
    http_status = 500
