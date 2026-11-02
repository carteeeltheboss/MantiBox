"""FastAPI backend with auth, access control and workflow orchestration (SPEC §3, §6, §9)."""

from __future__ import annotations

import hmac
import uuid
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from maintidoc.agent.workflow import WorkflowLimits, run_workflow
from maintidoc.errors import (
    AccessDeniedError,
    ApprovalInvalidError,
    BudgetExceededError,
    ValidationError,
)
from maintidoc.mcp.client import McpClient
from maintidoc.mcp.contracts import (
    CreateTicketRequest,
    CreateTicketResult,
    EquipmentHistory,
    TicketList,
)
from maintidoc.mcp.server import MaintenanceService
from maintidoc.observability import get_logger
from maintidoc.schemas import AnswerPayload
from maintidoc.security.access import Principal
from maintidoc.security.approval import (
    ApprovalService,
    canonical_content_hash,
    derive_idempotency_key,
)
from maintidoc.vlm.base import VlmProvider

logger = get_logger(__name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB


class IssueTokenRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    equipment_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=8000)


class ApproveTicketRequestBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    equipment_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=8000)
    approval_token: str = Field(min_length=16)
    idempotency_key: str = Field(min_length=16, max_length=128)


def create_app(
    *,
    retriever: Any,
    vlm_provider: VlmProvider,
    maintenance_service: MaintenanceService,
    approval_service: ApprovalService,
    api_key: SecretStr,
    limits: WorkflowLimits | None = None,
) -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title="MaintiDoc API",
        version="0.1.0",
        description="Assistant documentaire multimodal et agentique pour la maintenance.",
    )

    expected_api_key = api_key.get_secret_value()

    async def get_current_principal(
        x_user_id: str | None = Header(default=None),
        x_user_groups: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> Principal:
        if not x_api_key or not hmac.compare_digest(x_api_key, expected_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: invalid or missing API key",
            )
        if not x_user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: missing user id",
            )
        groups = (
            frozenset(g.strip() for g in x_user_groups.split(",") if g.strip())
            if x_user_groups
            else frozenset()
        )
        return Principal(user_id=x_user_id, groups=groups, is_active=True)

    @app.get("/health")
    def health_check() -> dict[str, str]:
        return {"status": "ok", "app": "maintidoc"}

    @app.post("/ask", response_model=AnswerPayload)
    async def ask_question(
        question: str = Form(...),
        equipment_id: str | None = Form(default=None),
        request_ticket: bool = Form(default=False),
        image: UploadFile | None = File(default=None),
        principal: Principal = Depends(get_current_principal),
    ) -> AnswerPayload:
        image_bytes: bytes | None = None
        if image is not None:
            image_bytes = await image.read()
            if len(image_bytes) > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Image size {len(image_bytes)} exceeds 10MB limit",
                )

        req_id = f"req-{uuid.uuid4().hex[:12]}"
        client = McpClient(
            service=maintenance_service,
            principal=principal,
            api_key=api_key,
            request_id=req_id,
        )

        try:
            return run_workflow(
                question=question,
                principal=principal,
                retriever=retriever,
                vlm_provider=vlm_provider,
                mcp_client=client,
                image_bytes=image_bytes,
                equipment_id=equipment_id,
                request_ticket=request_ticket,
                limits=limits,
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except BudgetExceededError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except AccessDeniedError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from exc

    @app.post("/approval-token")
    def issue_approval_token(
        body: IssueTokenRequest,
        principal: Principal = Depends(get_current_principal),
    ) -> dict[str, str]:
        content_hash = canonical_content_hash(
            equipment_id=body.equipment_id,
            title=body.title,
            description=body.description,
        )
        idem = derive_idempotency_key(user_id=principal.user_id, content_hash=content_hash)
        grant = approval_service.issue(
            user_id=principal.user_id,
            content_hash=content_hash,
            idempotency_key=idem,
        )
        return {
            "approval_token": grant.token,
            "idempotency_key": grant.idempotency_key,
            "content_hash": grant.content_hash,
        }

    @app.post("/tickets/approve", response_model=CreateTicketResult)
    def approve_and_create_ticket(
        body: ApproveTicketRequestBody,
        principal: Principal = Depends(get_current_principal),
    ) -> CreateTicketResult:
        req_id = f"req-{uuid.uuid4().hex[:12]}"
        client = McpClient(
            service=maintenance_service,
            principal=principal,
            api_key=api_key,
            request_id=req_id,
        )
        req = CreateTicketRequest(
            equipment_id=body.equipment_id,
            title=body.title,
            description=body.description,
            approval_token=body.approval_token,
            idempotency_key=body.idempotency_key,
        )
        try:
            return client.create_ticket(req)
        except (ApprovalInvalidError, AccessDeniedError) as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from exc

    @app.get("/equipment/{equipment_id}/history", response_model=EquipmentHistory)
    def get_history(
        equipment_id: str,
        principal: Principal = Depends(get_current_principal),
    ) -> EquipmentHistory:
        req_id = f"req-{uuid.uuid4().hex[:12]}"
        client = McpClient(
            service=maintenance_service,
            principal=principal,
            api_key=api_key,
            request_id=req_id,
        )
        try:
            return client.get_equipment_history(equipment_id)
        except AccessDeniedError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from exc

    @app.get("/equipment/{equipment_id}/tickets", response_model=TicketList)
    def get_open_tickets(
        equipment_id: str,
        principal: Principal = Depends(get_current_principal),
    ) -> TicketList:
        req_id = f"req-{uuid.uuid4().hex[:12]}"
        client = McpClient(
            service=maintenance_service,
            principal=principal,
            api_key=api_key,
            request_id=req_id,
        )
        try:
            return client.get_open_tickets(equipment_id)
        except AccessDeniedError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=str(exc),
            ) from exc

    return app


__all__ = ["create_app"]
