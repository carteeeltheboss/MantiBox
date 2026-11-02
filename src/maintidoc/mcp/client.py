"""MCP maintenance client (SPEC §6, §9).

Client for invoking MCP maintenance tools with client-side budget tracking
and strict authentication/context injection.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import SecretStr

from maintidoc.errors import BudgetExceededError
from maintidoc.mcp.contracts import (
    CreateTicketRequest,
    CreateTicketResult,
    EquipmentHistory,
    TicketList,
    ToolCallContext,
)
from maintidoc.mcp.server import MaintenanceService
from maintidoc.security.access import Principal


class McpClientProtocol(Protocol):
    """Protocol for MCP maintenance clients."""

    def get_equipment_history(self, equipment_id: str) -> EquipmentHistory: ...

    def get_open_tickets(self, equipment_id: str) -> TicketList: ...

    def create_ticket(self, request: CreateTicketRequest) -> CreateTicketResult: ...


class McpClient:
    """Concrete MCP client communicating directly with MaintenanceService."""

    def __init__(
        self,
        service: MaintenanceService,
        *,
        principal: Principal,
        api_key: SecretStr,
        request_id: str,
        max_calls: int = 3,
    ) -> None:
        self.service = service
        self.principal = principal
        self.api_key = api_key
        self.request_id = request_id
        self.max_calls = max_calls
        self._calls_made = 0

    @property
    def calls_made(self) -> int:
        return self._calls_made

    @property
    def budget_remaining(self) -> int:
        return max(0, self.max_calls - self._calls_made)

    def _check_and_build_context(self) -> ToolCallContext:
        if self._calls_made >= self.max_calls:
            raise BudgetExceededError(
                f"MCP call budget exceeded: max {self.max_calls} calls allowed per request"
            )
        ctx = ToolCallContext(
            principal=self.principal,
            api_key=self.api_key,
            request_id=self.request_id,
            budget_remaining=self.budget_remaining,
        )
        return ctx

    def get_equipment_history(self, equipment_id: str) -> EquipmentHistory:
        """Call get_equipment_history tool."""
        context = self._check_and_build_context()
        try:
            return self.service.get_equipment_history(equipment_id, context)
        finally:
            self._calls_made += 1

    def get_open_tickets(self, equipment_id: str) -> TicketList:
        """Call get_open_tickets tool."""
        context = self._check_and_build_context()
        try:
            return self.service.get_open_tickets(equipment_id, context)
        finally:
            self._calls_made += 1

    def create_ticket(self, request: CreateTicketRequest) -> CreateTicketResult:
        """Call create_ticket tool."""
        context = self._check_and_build_context()
        try:
            return self.service.create_ticket(request, context)
        finally:
            self._calls_made += 1


__all__ = [
    "McpClient",
    "McpClientProtocol",
]
