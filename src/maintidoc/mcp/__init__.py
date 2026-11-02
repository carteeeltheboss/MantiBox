"""MCP integration: contracts, demo maintenance server and client."""

from __future__ import annotations

from maintidoc.mcp.client import McpClient, McpClientProtocol
from maintidoc.mcp.contracts import (
    TOOL_SIGNATURES,
    CreateTicketRequest,
    CreateTicketResult,
    EquipmentHistory,
    EquipmentHistoryEntry,
    Ticket,
    TicketList,
    TicketState,
    ToolCallContext,
    ToolName,
)
from maintidoc.mcp.server import MaintenanceDatabase, MaintenanceService

__all__ = [
    "McpClient",
    "McpClientProtocol",
    "TOOL_SIGNATURES",
    "CreateTicketRequest",
    "CreateTicketResult",
    "EquipmentHistory",
    "EquipmentHistoryEntry",
    "MaintenanceDatabase",
    "MaintenanceService",
    "Ticket",
    "TicketList",
    "TicketState",
    "ToolCallContext",
    "ToolName",
]
