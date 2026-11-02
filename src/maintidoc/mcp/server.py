"""MCP maintenance service implementation with server-side authorization (SPEC §6, §9).

AGENTS.md NEVER #2: Authorization must be checked on EVERY call.
AGENTS.md NEVER #3: Ticket creation requires explicit human approval token bound to user + content.
"""

from __future__ import annotations

import hmac
import threading
import uuid
from datetime import UTC, datetime

from pydantic import SecretStr

from maintidoc.errors import AccessDeniedError, ApprovalInvalidError
from maintidoc.mcp.contracts import (
    CreateTicketRequest,
    CreateTicketResult,
    EquipmentHistory,
    EquipmentHistoryEntry,
    Ticket,
    TicketList,
    TicketState,
    ToolCallContext,
)
from maintidoc.observability import access_event, get_logger
from maintidoc.security.approval import (
    ApprovalService,
    canonical_content_hash,
    derive_idempotency_key,
)

logger = get_logger(__name__)


class MaintenanceDatabase:
    """Fictional in-memory data store for equipment history and maintenance tickets."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # equipment_id -> access_groups
        self.equipments: dict[str, frozenset[str]] = {}
        # equipment_id -> list[EquipmentHistoryEntry]
        self.history: dict[str, list[EquipmentHistoryEntry]] = {}
        # equipment_id -> list[Ticket]
        self.tickets: dict[str, list[Ticket]] = {}
        # idempotency_key -> ticket_id
        self.idempotency_map: dict[str, str] = {}

    @classmethod
    def seed_default(cls) -> MaintenanceDatabase:
        """Create and seed the database with fictional equipment and history (SPEC §2)."""
        db = cls()

        # Equipment family: pompes (Site A)
        pompe_groups = frozenset({"maintenance", "site_a"})
        db.equipments["pompe_centrifuge_01"] = pompe_groups
        db.equipments["pompe_centrifuge_02"] = pompe_groups

        # Equipment family: armoires (Site B)
        armoire_groups = frozenset({"maintenance", "site_b"})
        db.equipments["armoire_electrique_01"] = armoire_groups
        db.equipments["armoire_electrique_02"] = armoire_groups

        # Interventions for pompe_centrifuge_01
        db.history["pompe_centrifuge_01"] = [
            EquipmentHistoryEntry(
                intervention_id="int-pompe-001",
                equipment_id="pompe_centrifuge_01",
                occurred_at=datetime(2026, 3, 10, 8, 30, tzinfo=UTC),
                summary="Remplacement garniture mécanique d'étanchéité suite à micro-fuite.",
                technician_group="pompes",
                access_groups=pompe_groups,
            ),
            EquipmentHistoryEntry(
                intervention_id="int-pompe-002",
                equipment_id="pompe_centrifuge_01",
                occurred_at=datetime(2026, 6, 15, 14, 0, tzinfo=UTC),
                summary="Contrôle des vibrations de palier et graissage périodique.",
                technician_group="pompes",
                access_groups=pompe_groups,
            ),
            EquipmentHistoryEntry(
                intervention_id="int-pompe-003",
                equipment_id="pompe_centrifuge_01",
                occurred_at=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
                summary="Consignation haute pression spéciale pour visite décennale.",
                technician_group="supervisors",
                access_groups=frozenset({"supervisors"}),
            ),
        ]

        # Interventions for armoire_electrique_01
        db.history["armoire_electrique_01"] = [
            EquipmentHistoryEntry(
                intervention_id="int-armoire-001",
                equipment_id="armoire_electrique_01",
                occurred_at=datetime(2026, 5, 20, 11, 15, tzinfo=UTC),
                summary="Remplacement du fusible F3 et contrôle isolement circuit de commande.",
                technician_group="armoires",
                access_groups=armoire_groups,
            ),
        ]

        # Pre-existing tickets
        db.tickets["pompe_centrifuge_01"] = [
            Ticket(
                ticket_id="tick-pompe-001",
                equipment_id="pompe_centrifuge_01",
                title="Vibration anormale palier pompe",
                description="Léger bruit suspect relevé lors de la ronde.",
                state=TicketState.OPEN,
                opened_at=datetime(2026, 10, 28, 9, 0, tzinfo=UTC),
                access_groups=pompe_groups,
            )
        ]
        db.tickets["armoire_electrique_01"] = [
            Ticket(
                ticket_id="tick-armoire-001",
                equipment_id="armoire_electrique_01",
                title="Remplacement voyant K2",
                description="Voyant K2 éteint lors du test lampes.",
                state=TicketState.CLOSED,
                opened_at=datetime(2026, 9, 12, 16, 0, tzinfo=UTC),
                access_groups=armoire_groups,
            )
        ]

        return db


class MaintenanceService:
    """Maintenance service enforcing authentication and authorization on every tool call."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        approval_service: ApprovalService,
        database: MaintenanceDatabase,
    ) -> None:
        self._api_key = api_key
        self.approval_service = approval_service
        self.database = database

    def _authorize_call(self, context: ToolCallContext, equipment_id: str) -> frozenset[str]:
        """Authorize the tool call on authentication, budget, and equipment access.

        Raises AccessDeniedError on any failure. Never reveals whether equipment exists.
        """
        # 1. Active check
        if not context.principal.is_active:
            access_event(
                logger,
                action="mcp_tool_call",
                subject_id=context.principal.user_id,
                resource_type="equipment",
                resource_id=equipment_id,
                outcome="deny",
                reason="inactive_principal",
            )
            raise AccessDeniedError("inactive principal")

        # 2. API Key authentication (constant time comparison)
        provided_key = context.api_key.get_secret_value()
        expected_key = self._api_key.get_secret_value()
        if not hmac.compare_digest(provided_key, expected_key):
            access_event(
                logger,
                action="mcp_tool_call",
                subject_id=context.principal.user_id,
                resource_type="equipment",
                resource_id=equipment_id,
                outcome="deny",
                reason="invalid_api_key",
            )
            raise AccessDeniedError("invalid API key")

        # 3. Budget check (SPEC §9)
        if context.budget_remaining <= 0:
            access_event(
                logger,
                action="mcp_tool_call",
                subject_id=context.principal.user_id,
                resource_type="equipment",
                resource_id=equipment_id,
                outcome="deny",
                reason="budget_exceeded",
            )
            raise AccessDeniedError("MCP call budget exceeded")

        # 4. Equipment existence & access check
        equipment_groups = self.database.equipments.get(equipment_id)
        if equipment_groups is None:
            # Deny unknown equipment identically to unauthorized equipment (no enumeration oracle)
            access_event(
                logger,
                action="mcp_tool_call",
                subject_id=context.principal.user_id,
                resource_type="equipment",
                resource_id=equipment_id,
                outcome="deny",
                reason="unknown_equipment",
            )
            raise AccessDeniedError(f"access denied to equipment '{equipment_id}'")

        if not equipment_groups.intersection(context.principal.groups):
            access_event(
                logger,
                action="mcp_tool_call",
                subject_id=context.principal.user_id,
                resource_type="equipment",
                resource_id=equipment_id,
                outcome="deny",
                reason="group_mismatch",
            )
            raise AccessDeniedError(f"access denied to equipment '{equipment_id}'")

        access_event(
            logger,
            action="mcp_tool_call",
            subject_id=context.principal.user_id,
            resource_type="equipment",
            resource_id=equipment_id,
            outcome="allow",
        )
        return equipment_groups

    def get_equipment_history(
        self,
        equipment_id: str,
        context: ToolCallContext,
    ) -> EquipmentHistory:
        """Retrieve equipment intervention history with row-level access filtering."""
        self._authorize_call(context, equipment_id)

        all_entries = self.database.history.get(equipment_id, [])
        filtered_entries: list[EquipmentHistoryEntry] = []

        for entry in all_entries:
            if entry.access_groups.intersection(context.principal.groups):
                filtered_entries.append(entry)

        truncated = len(filtered_entries) < len(all_entries)
        return EquipmentHistory(
            equipment_id=equipment_id,
            entries=tuple(filtered_entries),
            truncated_by_access=truncated,
        )

    def get_open_tickets(
        self,
        equipment_id: str,
        context: ToolCallContext,
    ) -> TicketList:
        """Retrieve open tickets for an equipment with row-level access filtering."""
        self._authorize_call(context, equipment_id)

        all_tickets = self.database.tickets.get(equipment_id, [])
        open_tickets: list[Ticket] = []

        for ticket in all_tickets:
            if ticket.state is TicketState.OPEN and ticket.access_groups.intersection(
                context.principal.groups
            ):
                open_tickets.append(ticket)

        total_open = sum(1 for t in all_tickets if t.state is TicketState.OPEN)
        truncated = len(open_tickets) < total_open

        return TicketList(
            equipment_id=equipment_id,
            tickets=tuple(open_tickets),
            truncated_by_access=truncated,
        )

    def create_ticket(
        self,
        request: CreateTicketRequest,
        context: ToolCallContext,
    ) -> CreateTicketResult:
        """Create a maintenance ticket after verifying human approval token and idempotency."""
        equip_groups = self._authorize_call(context, request.equipment_id)

        # 1. Check idempotency for duplicates
        with self.database._lock:
            if request.idempotency_key in self.database.idempotency_map:
                existing_id = self.database.idempotency_map[request.idempotency_key]
                logger.info(
                    "Idempotent ticket retry deduplicated", extra={"ticket_id": existing_id}
                )
                return CreateTicketResult(
                    ticket_id=existing_id,
                    equipment_id=request.equipment_id,
                    created=False,
                    deduplicated=True,
                )

        # 2. Compute canonical content hash
        content_hash = canonical_content_hash(
            equipment_id=request.equipment_id,
            title=request.title,
            description=request.description,
        )

        # 3. Verify that idempotency key matches (user_id, content_hash)
        expected_idem = derive_idempotency_key(
            user_id=context.principal.user_id,
            content_hash=content_hash,
        )
        if not hmac.compare_digest(request.idempotency_key, expected_idem):
            raise ApprovalInvalidError("idempotency key does not match user and content hash")

        # 4. Verify and spend approval token (raises ApprovalInvalidError on failure)
        self.approval_service.verify(
            request.approval_token,
            user_id=context.principal.user_id,
            content_hash=content_hash,
            idempotency_key=request.idempotency_key,
            spend=True,
        )

        # 5. Create ticket in database
        ticket_id = f"tick-{uuid.uuid4().hex[:12]}"
        new_ticket = Ticket(
            ticket_id=ticket_id,
            equipment_id=request.equipment_id,
            title=request.title,
            description=request.description,
            state=TicketState.OPEN,
            opened_at=datetime.now(UTC),
            access_groups=equip_groups,
        )

        with self.database._lock:
            # Double check under lock
            if request.idempotency_key in self.database.idempotency_map:
                existing_id = self.database.idempotency_map[request.idempotency_key]
                return CreateTicketResult(
                    ticket_id=existing_id,
                    equipment_id=request.equipment_id,
                    created=False,
                    deduplicated=True,
                )

            self.database.tickets.setdefault(request.equipment_id, []).append(new_ticket)
            self.database.idempotency_map[request.idempotency_key] = ticket_id

        logger.info(
            "Created maintenance ticket with verified approval",
            extra={"ticket_id": ticket_id, "equipment_id": request.equipment_id},
        )

        return CreateTicketResult(
            ticket_id=ticket_id,
            equipment_id=request.equipment_id,
            created=True,
            deduplicated=False,
        )


__all__ = [
    "MaintenanceDatabase",
    "MaintenanceService",
]
