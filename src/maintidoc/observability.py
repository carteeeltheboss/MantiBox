"""Logging and observability.

SPEC §9 / AGENTS.md NEVER #1: logs carry **technical metadata and access events
only** — never raw document text, OCR output, image bytes or user content, and
never a secret.

Two mechanisms enforce that rather than relying on discipline:

* `SecretRedactionFilter` scrubs any configured secret value out of every record
  before it reaches a handler, so an accidental f-string cannot leak a password.
* `content_fingerprint()` gives code a way to correlate a document or message
  across log lines *without* carrying the content itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "***REDACTED***"

# Anything shorter is not worth redacting and would mangle ordinary text (a
# 3-char secret matches inside normal words).
_MIN_SECRET_LENGTH = 6

_configured = False

_RESERVED_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class SecretRedactionFilter(logging.Filter):
    """Replace known secret values wherever they appear in a log record."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        # Longest first: a secret that contains another secret is then replaced
        # in one pass instead of leaving a recognisable fragment behind.
        self._secrets: tuple[str, ...] = tuple(
            sorted(
                {s for s in secrets if s and len(s) >= _MIN_SECRET_LENGTH},
                key=len,
                reverse=True,
            )
        )

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        message = record.getMessage()
        redacted = message
        for secret in self._secrets:
            if secret in redacted:
                redacted = redacted.replace(secret, REDACTED)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: parseable, and impossible to smuggle content
    into via a multi-line message."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            if isinstance(value, str | int | float | bool) or value is None:
                payload[key] = value
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, default=str)


class PlainFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)-7s %(name)s :: %(message)s")


def content_fingerprint(data: str | bytes) -> str:
    """Stable short identifier for content that must never be logged directly.

    Correlate "the same document appeared twice" across log lines while keeping
    the content itself out of the log stream.
    """
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()[:16]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def access_event(
    logger: logging.Logger,
    *,
    action: str,
    subject_id: str,
    resource_type: str,
    resource_id: str,
    outcome: str,
    reason: str | None = None,
    **fields: Any,
) -> None:
    """Emit an access-control event (SPEC §9: "événements d'accès").

    Only identifiers and outcomes are accepted — there is deliberately no
    parameter for document text, image data or the user's question.
    """
    logger.info(
        "access %s",
        action,
        extra={
            "action": action,
            "subject_id": subject_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "outcome": outcome,
            "reason": reason,
            **fields,
        },
    )


def configure_logging(
    *,
    level: str = "INFO",
    json_output: bool = False,
    secrets: Iterable[str] = (),
) -> None:
    """Idempotent root-logger setup. Safe to call from every entrypoint."""
    global _configured
    root = logging.getLogger()
    root.setLevel(level.upper())
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else PlainFormatter())
    handler.addFilter(SecretRedactionFilter(secrets))
    root.handlers.clear()
    root.addHandler(handler)

    # Third-party libraries are noisy and some echo request payloads.
    for noisy in ("httpx", "httpcore", "urllib3", "multipart", "PIL", "fitz"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def collect_secret_values(root: Any) -> list[str]:
    """Walk a settings object and return every SecretStr value it holds.

    Feeds SecretRedactionFilter so that configured credentials are scrubbed from
    logs even if some code path formats them by mistake.
    """
    from pydantic import BaseModel, SecretStr

    found: list[str] = []
    seen: set[int] = set()
    stack: list[Any] = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, SecretStr):
            value = obj.get_secret_value()
            if value:
                found.append(value)
        elif isinstance(obj, BaseModel | Mapping):
            stack.extend(obj.__dict__.values() if isinstance(obj, BaseModel) else obj.values())
    return found
