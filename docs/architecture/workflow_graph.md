# LangGraph Workflow State Machine (SPEC §6)

This document visualizes the compiled LangGraph workflow state machine executing the
SPEC §6 node sequence, access checks, history consultation, and human-approval interrupt.

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	verify_access(verify_access)
	handle_access_denied(handle_access_denied)
	analyze_request(analyze_request)
	retrieve_sources(retrieve_sources)
	consult_history(consult_history)
	build_and_verify_response(build_and_verify_response)
	draft_ticket(draft_ticket)
	wait_human_approval(wait_human_approval)
	create_ticket(create_ticket)
	finalize(finalize)
	__end__([<p>__end__</p>]):::last
	__start__ --> verify_access;
	analyze_request -.-> finalize;
	analyze_request -.-> retrieve_sources;
	build_and_verify_response -.-> create_ticket;
	build_and_verify_response -.-> draft_ticket;
	build_and_verify_response -.-> finalize;
	consult_history --> build_and_verify_response;
	create_ticket --> finalize;
	draft_ticket --> wait_human_approval;
	retrieve_sources --> consult_history;
	verify_access -.-> analyze_request;
	verify_access -.-> handle_access_denied;
	wait_human_approval -.-> create_ticket;
	wait_human_approval -.-> finalize;
	finalize --> __end__;
	handle_access_denied --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

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
