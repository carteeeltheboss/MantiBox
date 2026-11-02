"""Export LangGraph workflow diagram to docs/architecture/workflow_graph.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import SecretStr

from maintidoc.agent.workflow import build_workflow_graph, export_workflow_graph_mermaid
from maintidoc.mcp.client import McpClient
from maintidoc.mcp.server import MaintenanceDatabase, MaintenanceService
from maintidoc.retrieval.reranker import NoReranker
from maintidoc.security.access import Principal
from maintidoc.security.approval import ApprovalService, InMemoryApprovalStore
from maintidoc.vlm.stub import StubVlmProvider


class _DummyRetriever:
    reranker = NoReranker()

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def main() -> None:
    service = MaintenanceService(
        api_key=SecretStr("key"),
        approval_service=ApprovalService(
            secret_key="x" * 32,
            store=InMemoryApprovalStore(),
        ),
        database=MaintenanceDatabase(),
    )
    client = McpClient(
        service=service,
        principal=Principal(user_id="u", groups=frozenset({"g"}), is_active=True),
        api_key=SecretStr("k"),
        request_id="r",
    )
    workflow = build_workflow_graph(
        retriever=_DummyRetriever(),
        vlm_provider=StubVlmProvider(),
        mcp_client=client,
    )
    mermaid = export_workflow_graph_mermaid(workflow)

    content = f"""# LangGraph Workflow State Machine (SPEC §6)

This document visualizes the compiled LangGraph workflow state machine executing the
SPEC §6 node sequence, access checks, history consultation, and human-approval interrupt.

```mermaid
{mermaid}
```

## Node Flow Description

1. **`verify_access`**: Evaluates principal status and enforces image limits.
2. **`handle_access_denied`**: Short-circuits unauthorized requests to `END`.
3. **`analyze_request`**: Visual analysis with `visual_analysis_v1` when image present.
4. **`retrieve_sources`**: Access-filtered hybrid search (`HybridRetriever`).
5. **`consult_history`**: Consults MCP history and ticket tools when needed.
6. **`build_and_verify_response`**: Inference, draft parsing, and citation check.
7. **`draft_ticket`**: Formats candidate ticket, computes hash and idempotency key.
8. **`wait_human_approval`**: Interrupt state awaiting bound approval token.
9. **`create_ticket`**: Calls MCP tool `create_ticket` with verified approval token.
10. **`finalize`**: Assembles final `AnswerPayload` ensuring all invariants hold.
"""
    out_dir = Path("docs/architecture")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "workflow_graph.md"
    out_file.write_text(content, encoding="utf-8")
    print(f"Exported graph diagram to {out_file}")


if __name__ == "__main__":
    main()
