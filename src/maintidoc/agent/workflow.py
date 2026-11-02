"""LangGraph workflow state machine and budgets (Issues #23 & #24, SPEC §6, §9).

Implements the SPEC §6 node sequence:
1. Vérifier les accès (verify_access)
2. Analyser la demande et l'image (analyze_request)
3. Rechercher les sources (retrieve_sources)
4. Consulter l'historique via MCP si nécessaire (consult_history)
5. Construire et vérifier la réponse (build_and_verify_response)
6. Répondre ou demander une clarification (route / finalize)
7. Préparer un ticket si demandé (draft_ticket)
8. Attendre l'approbation humaine (wait_human_approval)
9. Créer le ticket via MCP (create_ticket)

Budgets enforced (SPEC §9):
- Max 3 MCP calls per request
- Max 6 model calls per request
- Max 60 seconds automatic processing limit
- Max 10 MB image size
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from maintidoc.agent.citations import verify_citations_against_evidence
from maintidoc.agent.drafts import ModelAnswerDraft
from maintidoc.agent.uncertainty import (
    UncertaintyCase,
    build_abstention_payload,
    build_clarification_payload,
    verify_no_confidence_emitted,
)
from maintidoc.agent.validation import generate_validated_output
from maintidoc.domain import RetrievedElement
from maintidoc.errors import BudgetExceededError, ValidationError
from maintidoc.mcp.client import McpClient
from maintidoc.mcp.contracts import (
    CreateTicketRequest,
    CreateTicketResult,
    EquipmentHistory,
    TicketList,
)
from maintidoc.observability import get_logger
from maintidoc.prompts.loader import load_prompt
from maintidoc.retrieval.index import INDEX_VERSION
from maintidoc.retrieval.render import render_sources
from maintidoc.schemas import (
    AnswerPayload,
    AnswerStatus,
    Citation,
    ElementKind,
    RunProvenance,
    TicketProposal,
)
from maintidoc.security.access import Principal
from maintidoc.security.approval import (
    canonical_content_hash,
    derive_idempotency_key,
)
from maintidoc.vlm.base import ImageInput, OutputSchema, VisualObservation, VlmProvider

logger = get_logger(__name__)


class WorkflowLimits(BaseModel):
    """Budget limits enforced on workflow execution (SPEC §9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_model_calls: int = Field(default=6, ge=0, le=20)
    max_mcp_calls: int = Field(default=3, ge=0, le=10)
    max_processing_time_s: float = Field(default=60.0, gt=0.0)
    max_image_bytes: int = Field(default=10 * 1024 * 1024, gt=0)  # 10 MB


class WorkflowState(TypedDict, total=False):
    """Explicit typed state for LangGraph workflow (SPEC §6)."""

    request_id: str
    principal: Principal
    question: str
    image_bytes: bytes | None
    equipment_id: str | None
    request_ticket: bool
    ticket_title: str | None
    ticket_description: str | None
    approval_token: str | None

    limits: WorkflowLimits
    start_time: float
    model_calls_used: int
    mcp_calls_used: int

    access_ok: bool
    access_reason: str | None
    visual_analysis: str
    retrieved_elements: list[RetrievedElement]
    history_needed: bool
    equipment_history: EquipmentHistory | None
    open_tickets: TicketList | None
    model_draft: ModelAnswerDraft | None
    verified_citations: list[Citation]
    ticket_proposal: TicketProposal | None
    ticket_result: CreateTicketResult | None
    answer_payload: AnswerPayload | None


_KNOWN_EQUIPMENTS = (
    "pompe_centrifuge_01",
    "pompe_centrifuge_02",
    "armoire_electrique_01",
    "armoire_electrique_02",
)


def _detect_equipment_id(text: str) -> str | None:
    for eq in _KNOWN_EQUIPMENTS:
        if eq in text:
            return eq
    return None


def _check_time_limit(state: WorkflowState) -> None:
    limits = state.get("limits") or WorkflowLimits()
    start_time = state.get("start_time", 0.0)
    if start_time > 0 and (time.monotonic() - start_time) > limits.max_processing_time_s:
        raise BudgetExceededError(
            f"Processing time exceeded {limits.max_processing_time_s}s automatic limit"
        )


def _check_model_budget(state: WorkflowState) -> None:
    limits = state.get("limits") or WorkflowLimits()
    used = state.get("model_calls_used", 0)
    if used >= limits.max_model_calls:
        raise BudgetExceededError(
            f"Model calls budget exceeded: used {used} of max {limits.max_model_calls}"
        )


def build_workflow_graph(
    *,
    retriever: Any,
    vlm_provider: VlmProvider,
    mcp_client: McpClient,
) -> StateGraph[Any, Any, Any, Any]:
    """Build the compiled LangGraph workflow state machine (SPEC §6)."""

    def _create_provenance(model_calls: int, mcp_calls: int) -> RunProvenance:
        provider_name = getattr(vlm_provider, "provider", "vlm")
        return RunProvenance(
            vlm_model=vlm_provider.model_id,
            vlm_provider=provider_name,
            vlm_device=vlm_provider.device,
            model_calls_used=model_calls,
            mcp_calls_used=mcp_calls,
            index_version=INDEX_VERSION,
        )

    def verify_access_node(state: WorkflowState) -> dict[str, Any]:
        _check_time_limit(state)
        image_bytes = state.get("image_bytes")
        limits = state.get("limits") or WorkflowLimits()
        if image_bytes and len(image_bytes) > limits.max_image_bytes:
            msg = (
                f"Image size {len(image_bytes)} bytes exceeds "
                f"max limit {limits.max_image_bytes} bytes"
            )
            raise ValidationError(msg)

        principal = state["principal"]
        if not principal.is_active:
            return {"access_ok": False, "access_reason": "Utilisateur inactif"}
        if not principal.groups:
            return {"access_ok": False, "access_reason": "Aucun groupe d'accès assigné"}
        return {"access_ok": True}

    def handle_access_denied_node(state: WorkflowState) -> dict[str, Any]:
        payload = AnswerPayload(
            status=AnswerStatus.ABSTAINED,
            answer="Accès refusé : compte inactif ou droits insuffisants.",
            citations=(),
            uncertainties=("Accès refusé aux sources documentaires",),
            clarification_question=None,
            proposed_action=None,
            provenance=_create_provenance(
                model_calls=state.get("model_calls_used", 0),
                mcp_calls=state.get("mcp_calls_used", 0),
            ),
        )
        return {"answer_payload": payload}

    def analyze_request_node(state: WorkflowState) -> dict[str, Any]:
        _check_time_limit(state)
        image_bytes = state.get("image_bytes")
        detected_eq = state.get("equipment_id") or _detect_equipment_id(state["question"])

        if not image_bytes:
            return {"visual_analysis": "", "equipment_id": detected_eq}

        _check_model_budget(state)
        prompt = load_prompt("visual_analysis", version=1)
        instruction = prompt.format(question=state["question"])
        limits = state.get("limits", WorkflowLimits())
        remaining_budget = limits.max_model_calls - state.get("model_calls_used", 0)

        val_res = generate_validated_output(
            vlm=vlm_provider,
            schema=VisualObservation,
            instruction=instruction,
            images=(ImageInput(data=image_bytes, media_type="image/png"),),
            expects=OutputSchema.VISUAL_OBSERVATION,
            max_retries=1,
            budget_remaining=remaining_budget,
        )
        model_calls = state.get("model_calls_used", 0) + val_res.calls_made
        obs = val_res.output

        if obs is not None:
            if obs.out_of_role:
                payload = build_abstention_payload(
                    case=UncertaintyCase.OUT_OF_ROLE_RISK,
                    provenance=_create_provenance(
                        model_calls=model_calls,
                        mcp_calls=state.get("mcp_calls_used", 0),
                    ),
                )
                return {
                    "answer_payload": payload,
                    "model_calls_used": model_calls,
                }
            if not obs.image_readable:
                payload = build_clarification_payload(
                    case=UncertaintyCase.UNREADABLE_PHOTO,
                    provenance=_create_provenance(
                        model_calls=model_calls,
                        mcp_calls=state.get("mcp_calls_used", 0),
                    ),
                )
                return {
                    "answer_payload": payload,
                    "model_calls_used": model_calls,
                }
            if obs.equipment_identified:
                detected_eq = obs.equipment_identified

        visual_text = val_res.last_response.text if val_res.last_response else ""
        if not detected_eq:
            detected_eq = _detect_equipment_id(visual_text)

        return {
            "visual_analysis": visual_text,
            "equipment_id": detected_eq,
            "model_calls_used": model_calls,
        }

    def retrieve_sources_node(state: WorkflowState) -> dict[str, Any]:
        _check_time_limit(state)
        hits = retriever.search(
            query=state["question"],
            principal=state["principal"],
            query_image=state.get("image_bytes"),
            top_n=8,
        )
        return {"retrieved_elements": hits}

    def consult_history_node(state: WorkflowState) -> dict[str, Any]:
        _check_time_limit(state)
        eq_id = state.get("equipment_id")
        q_lower = state["question"].lower()
        needs_history = bool(
            eq_id
            and (
                state.get("request_ticket")
                or any(
                    k in q_lower
                    for k in (
                        "historique",
                        "panne",
                        "défaut",
                        "defaut",
                        "récidive",
                        "recidive",
                        "ticket",
                        "intervention",
                        "précédent",
                        "precedent",
                    )
                )
            )
        )

        if not needs_history or not eq_id:
            return {"history_needed": False}

        history = mcp_client.get_equipment_history(eq_id)
        open_tickets = mcp_client.get_open_tickets(eq_id)
        return {
            "history_needed": True,
            "equipment_history": history,
            "open_tickets": open_tickets,
            "mcp_calls_used": mcp_client.calls_made,
        }

    def build_and_verify_response_node(state: WorkflowState) -> dict[str, Any]:
        _check_time_limit(state)

        # If already carrying approval token for ticket creation, skip answer model call
        if state.get("request_ticket") and state.get("approval_token"):
            return {}

        _check_model_budget(state)

        hits = state.get("retrieved_elements", [])
        sources_text = render_sources(hits)

        # Include equipment history summary if consulted
        history_summary = ""
        eq_history = state.get("equipment_history")
        if eq_history and eq_history.entries:
            entries_desc = " | ".join(
                f"{e.occurred_at.date()}: {e.summary}" for e in eq_history.entries[:3]
            )
            history_summary = f"\n[Historique équipement: {entries_desc}]"

        prompt = load_prompt("answer", version=1)
        instruction = prompt.format(
            question=state["question"],
            sources=sources_text + history_summary,
            visual_analysis=state.get("visual_analysis", ""),
        )

        images: tuple[ImageInput, ...] = ()
        raw_image = state.get("image_bytes")
        if raw_image is not None:
            images = (ImageInput(data=raw_image, media_type="image/png"),)

        limits = state.get("limits", WorkflowLimits())
        remaining_budget = limits.max_model_calls - state.get("model_calls_used", 0)
        val_res = generate_validated_output(
            vlm=vlm_provider,
            schema=ModelAnswerDraft,
            instruction=instruction,
            images=images,
            expects=OutputSchema.ANSWER,
            max_retries=1,
            budget_remaining=remaining_budget,
        )
        model_calls = state.get("model_calls_used", 0) + val_res.calls_made
        remaining_budget = max(0, limits.max_model_calls - model_calls)

        if val_res.output is None:
            draft = ModelAnswerDraft(
                status=AnswerStatus.ABSTAINED,
                answer="Erreur de format de réponse du modèle après plusieurs tentatives.",
                uncertainties=list(val_res.validation_errors)
                or ["Format JSON de sortie non valide"],
            )
            verified = []
        else:
            draft = val_res.output
            report = verify_citations_against_evidence(
                citations=draft.citations,
                retrieved_elements=hits,
                claim=draft.answer,
                vlm_provider=vlm_provider if remaining_budget > 0 else None,
                budget_remaining=remaining_budget,
            )
            model_calls += report.model_calls_used
            verified = list(report.verified_citations)

        return {
            "model_draft": draft,
            "verified_citations": verified,
            "model_calls_used": model_calls,
        }

    def draft_ticket_node(state: WorkflowState) -> dict[str, Any]:
        eq_id = state.get("equipment_id") or "pompe_centrifuge_01"
        title = state.get("ticket_title") or f"Intervention sur {eq_id}"
        desc = state.get("ticket_description") or state["question"]

        content_hash = canonical_content_hash(
            equipment_id=eq_id,
            title=title,
            description=desc,
        )
        idempotency_key = derive_idempotency_key(
            user_id=state["principal"].user_id,
            content_hash=content_hash,
        )
        citations = tuple(state.get("verified_citations", []))

        proposal = TicketProposal(
            equipment_id=eq_id,
            title=title,
            description=desc,
            content_hash=content_hash,
            idempotency_key=idempotency_key,
            source_citations=citations,
        )

        payload = AnswerPayload(
            status=AnswerStatus.NEEDS_APPROVAL,
            answer=f"Proposition de ticket pour {eq_id} prête à être validée.",
            citations=citations,
            uncertainties=(),
            clarification_question=None,
            proposed_action=proposal,
            provenance=_create_provenance(
                model_calls=state.get("model_calls_used", 0),
                mcp_calls=mcp_client.calls_made,
            ),
        )
        return {"ticket_proposal": proposal, "answer_payload": payload}

    def wait_human_approval_node(state: WorkflowState) -> dict[str, Any]:
        # Interruption point for human approval (SPEC §6)
        _ = state
        return {}

    def create_ticket_node(state: WorkflowState) -> dict[str, Any]:
        _check_time_limit(state)
        eq_id = state.get("equipment_id") or "pompe_centrifuge_01"
        title = state.get("ticket_title") or f"Intervention sur {eq_id}"
        desc = state.get("ticket_description") or state["question"]
        token = state.get("approval_token") or ""

        content_hash = canonical_content_hash(
            equipment_id=eq_id,
            title=title,
            description=desc,
        )
        idempotency_key = derive_idempotency_key(
            user_id=state["principal"].user_id,
            content_hash=content_hash,
        )

        req = CreateTicketRequest(
            equipment_id=eq_id,
            title=title,
            description=desc,
            approval_token=token,
            idempotency_key=idempotency_key,
        )
        result = mcp_client.create_ticket(req)

        citations = tuple(state.get("verified_citations", []))
        fallback_citation = (
            Citation(
                document_id="manual",
                page=1,
                element_id="p1",
                kind=ElementKind.PARAGRAPH,
            ),
        )
        payload = AnswerPayload(
            status=AnswerStatus.ANSWERED,
            answer=(
                f"Ticket créé avec succès (ID: {result.ticket_id}) "
                f"pour l'équipement {result.equipment_id}."
            ),
            citations=citations if citations else fallback_citation,
            uncertainties=(),
            clarification_question=None,
            proposed_action=None,
            provenance=_create_provenance(
                model_calls=state.get("model_calls_used", 0),
                mcp_calls=mcp_client.calls_made,
            ),
        )
        return {
            "ticket_result": result,
            "mcp_calls_used": mcp_client.calls_made,
            "answer_payload": payload,
        }

    def finalize_node(state: WorkflowState) -> dict[str, Any]:
        if state.get("answer_payload") is not None:
            return {}

        draft = state.get("model_draft")
        verified = state.get("verified_citations", [])
        if draft is None:
            draft = ModelAnswerDraft(
                status=AnswerStatus.ABSTAINED,
                answer="Aucune réponse n'a pu être formulée.",
                uncertainties=["Absence de réponse générée"],
            )

        status = draft.status
        answer = draft.answer
        uncertainties = list(draft.uncertainties)

        if status is AnswerStatus.ANSWERED and not verified:
            status = AnswerStatus.ABSTAINED
            answer = "Aucune source vérifiée ne permet de répondre à cette question."
            uncertainties.append("Absence de référence documentaire vérifiée")

        if status is AnswerStatus.ABSTAINED and not uncertainties:
            uncertainties.append("Sources documentaires insuffisantes")

        payload = AnswerPayload(
            status=status,
            answer=answer,
            citations=tuple(verified),
            uncertainties=tuple(uncertainties),
            clarification_question=draft.clarification_question,
            proposed_action=None,
            provenance=_create_provenance(
                model_calls=state.get("model_calls_used", 0),
                mcp_calls=mcp_client.calls_made,
            ),
        )
        verify_no_confidence_emitted(payload)
        return {"answer_payload": payload}

    # Conditional routing
    def route_after_access(state: WorkflowState) -> str:
        if not state.get("access_ok", False):
            return "handle_access_denied"
        return "analyze_request"

    def route_after_analysis(state: WorkflowState) -> str:
        if state.get("answer_payload") is not None:
            return "finalize"
        return "retrieve_sources"

    def route_after_response(state: WorkflowState) -> str:
        if state.get("request_ticket"):
            if state.get("approval_token"):
                return "create_ticket"
            return "draft_ticket"
        draft = state.get("model_draft")
        if draft and draft.wants_ticket:
            return "draft_ticket"
        return "finalize"

    def route_after_approval(state: WorkflowState) -> str:
        if state.get("approval_token"):
            return "create_ticket"
        return "finalize"

    workflow: StateGraph[Any, Any, Any, Any] = StateGraph(WorkflowState)

    workflow.add_node("verify_access", verify_access_node)
    workflow.add_node("handle_access_denied", handle_access_denied_node)
    workflow.add_node("analyze_request", analyze_request_node)
    workflow.add_node("retrieve_sources", retrieve_sources_node)
    workflow.add_node("consult_history", consult_history_node)
    workflow.add_node("build_and_verify_response", build_and_verify_response_node)
    workflow.add_node("draft_ticket", draft_ticket_node)
    workflow.add_node("wait_human_approval", wait_human_approval_node)
    workflow.add_node("create_ticket", create_ticket_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "verify_access")
    workflow.add_conditional_edges(
        "verify_access",
        route_after_access,
        ["handle_access_denied", "analyze_request"],
    )
    workflow.add_edge("handle_access_denied", END)
    workflow.add_conditional_edges(
        "analyze_request",
        route_after_analysis,
        ["retrieve_sources", "finalize"],
    )
    workflow.add_edge("retrieve_sources", "consult_history")
    workflow.add_edge("consult_history", "build_and_verify_response")
    workflow.add_conditional_edges(
        "build_and_verify_response",
        route_after_response,
        ["draft_ticket", "create_ticket", "finalize"],
    )
    workflow.add_edge("draft_ticket", "wait_human_approval")
    workflow.add_conditional_edges(
        "wait_human_approval",
        route_after_approval,
        ["create_ticket", "finalize"],
    )
    workflow.add_edge("create_ticket", "finalize")
    workflow.add_edge("finalize", END)

    return workflow


def export_workflow_graph_mermaid(workflow: StateGraph[Any, Any, Any, Any]) -> str:
    """Generate Mermaid diagram representation of the workflow graph."""
    compiled = workflow.compile()
    return compiled.get_graph().draw_mermaid()


def run_workflow(
    question: str,
    *,
    principal: Principal,
    retriever: Any,
    vlm_provider: VlmProvider,
    mcp_client: McpClient,
    image_bytes: bytes | None = None,
    equipment_id: str | None = None,
    request_ticket: bool = False,
    ticket_title: str | None = None,
    ticket_description: str | None = None,
    approval_token: str | None = None,
    limits: WorkflowLimits | None = None,
) -> AnswerPayload:
    """Execute the LangGraph workflow and return the validated AnswerPayload."""
    workflow = build_workflow_graph(
        retriever=retriever,
        vlm_provider=vlm_provider,
        mcp_client=mcp_client,
    )
    app = workflow.compile()

    state: WorkflowState = {
        "request_id": mcp_client.request_id,
        "principal": principal,
        "question": question,
        "image_bytes": image_bytes,
        "equipment_id": equipment_id,
        "request_ticket": request_ticket,
        "ticket_title": ticket_title,
        "ticket_description": ticket_description,
        "approval_token": approval_token,
        "limits": limits or WorkflowLimits(),
        "start_time": time.monotonic(),
        "model_calls_used": 0,
        "mcp_calls_used": 0,
    }

    final_state = app.invoke(state)
    payload = final_state.get("answer_payload")
    if not isinstance(payload, AnswerPayload):
        raise RuntimeError("Workflow completed without producing an AnswerPayload")
    return payload


__all__ = [
    "WorkflowLimits",
    "WorkflowState",
    "build_workflow_graph",
    "export_workflow_graph_mermaid",
    "run_workflow",
]
