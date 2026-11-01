"""MCP maintenance service — typed contracts (SPEC §6).

Contract first: these signatures exist before the server and client that use
them (AGENTS.md conventions).

The server exposes exactly three tools:

    get_equipment_history(equipment_id)
    get_open_tickets(equipment_id)
    create_ticket(equipment_id, title, description, approval_token)

Authorization is re-checked server-side on **every** call (AGENTS.md NEVER #2).
A caller-supplied `ToolCallContext` is not trusted because it says it is valid;
the server validates the API key and re-evaluates group access against the
equipment and the ticket rows it is about to return.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from maintidoc.security.access import Principal


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolName(StrEnum):
    GET_EQUIPMENT_HISTORY = "get_equipment_history"
    GET_OPEN_TICKETS = "get_open_tickets"
    CREATE_TICKET = "create_ticket"


class TicketState(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    CLOSED = "closed"


class ToolCallContext(_Strict):
    """Everything the server needs to authorize one call.

    `budget_remaining` lets the server cooperate with the SPEC §9 limit of 3 MCP
    calls per request instead of only enforcing it client-side.
    """

    principal: Principal
    api_key: SecretStr
    request_id: str = Field(min_length=1, max_length=128)
    budget_remaining: int = Field(default=3, ge=0, le=10)


class EquipmentHistoryEntry(_Strict):
    intervention_id: str = Field(min_length=1, max_length=128)
    equipment_id: str = Field(min_length=1, max_length=128)
    occurred_at: datetime
    summary: str = Field(max_length=2000)
    technician_group: str | None = Field(default=None, max_length=128)
    #: Groups allowed to read this row; the server filters on it before return.
    access_groups: frozenset[str] = frozenset()


class EquipmentHistory(_Strict):
    equipment_id: str
    entries: tuple[EquipmentHistoryEntry, ...] = ()
    #: Set when some rows were withheld by the access filter, so the caller can
    #: tell "nothing happened" apart from "you may not see what happened".
    truncated_by_access: bool = False


class Ticket(_Strict):
    ticket_id: str = Field(min_length=1, max_length=128)
    equipment_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(max_length=8000)
    state: TicketState = TicketState.OPEN
    opened_at: datetime
    access_groups: frozenset[str] = frozenset()


class TicketList(_Strict):
    equipment_id: str
    tickets: tuple[Ticket, ...] = ()
    truncated_by_access: bool = False


class CreateTicketRequest(_Strict):
    """`approval_token` is mandatory — there is no path that creates a ticket
    without one (AGENTS.md NEVER #3)."""

    equipment_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=8000)
    approval_token: str = Field(min_length=16)
    idempotency_key: str = Field(min_length=16, max_length=128)


class CreateTicketResult(_Strict):
    ticket_id: str = Field(min_length=1, max_length=128)
    equipment_id: str
    created: bool
    #: True when `idempotency_key` matched an existing ticket, i.e. this call was
    #: a replay and nothing new was created.
    deduplicated: bool = False


#: The published tool signatures, used to generate the MCP tool list and to keep
#: server, client and docs from drifting apart.
TOOL_SIGNATURES: dict[ToolName, tuple[str, ...]] = {
    ToolName.GET_EQUIPMENT_HISTORY: ("equipment_id",),
    ToolName.GET_OPEN_TICKETS: ("equipment_id",),
    ToolName.CREATE_TICKET: (
        "equipment_id",
        "title",
        "description",
        "approval_token",
    ),
}


__all__ = [
    "TOOL_SIGNATURES",
    "CreateTicketRequest",
    "CreateTicketResult",
    "EquipmentHistory",
    "EquipmentHistoryEntry",
    "Ticket",
    "TicketList",
    "TicketState",
    "ToolCallContext",
    "ToolName",
]
