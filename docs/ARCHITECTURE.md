# ARCHITECTURE

> Living document. Reflects what is BUILT, not what is planned (the plan is in SPEC.md §3). Update when components land.

## Components built so far

| Component | Status | Location | Notes |
|---|---|---|---|
| Output contract (SPEC §7) | built | `src/maintidoc/schemas.py` | Four statuses with cross-field invariants; `extra="forbid"` makes a confidence field impossible. |
| Domain models | built | `src/maintidoc/domain.py` | `DocumentRecord`, `Element`, `RetrievedElement`; SPEC §4.1 metadata is required, not optional. |
| Configuration | built | `src/maintidoc/config.py` | pydantic-settings, secrets as `SecretStr`; `MAINTIDOC_ENV=prod` refuses a placeholder signing key. |
| Safe logging | built | `src/maintidoc/observability.py` | Secret redaction filter, JSON formatter, `access_event()` with no content parameter. |
| Identity + access control | built | `src/maintidoc/security/access.py` | Deny by default; filter applied inside the vector query and again on open. |
| Approval tokens | built | `src/maintidoc/security/approval.py` | HS256 with pinned algorithm, bound to user + content sha256, single-use `jti`. |
| MCP tool contracts | built | `src/maintidoc/mcp/contracts.py` | Signatures only; server and client pending. |
| Device selection + model ladder | built | `src/maintidoc/runtime/devices.py` | CUDA→MPS→CPU, loud step-down, audit trail in provenance. |
| Prompt loading | built | `src/maintidoc/prompts/loader.py` | Versioned YAML, jinja2 format, refuses undelimited untrusted input. |
| VLM provider seam | built | `src/maintidoc/vlm/` | Protocol + deterministic stub. Real Qwen2.5-VL provider pending. |
| Source rendering | built | `src/maintidoc/retrieval/render.py` | Render/parse round-trip prevents fabricated citations. |
| Model draft layer | built | `src/maintidoc/agent/drafts.py` | Lenient model output vs strict API contract. |
| Object storage | built | `src/maintidoc/storage/objects.py` | Interface + filesystem backend; S3 backend pending a backend choice. |
| CLI | built | `src/maintidoc/cli.py` | `doctor`, `version`. |
| Corpus generator | built | `scripts/generate_corpus.py` | Seeded; OCR, injection and restricted-access cases included. |
| PDF Ingestion | built | `src/maintidoc/ingestion/` | PyMuPDF + local OCR fallback; extracts paragraphs, tables, figures with SPEC §4.1 metadata. |
| Embeddings / index / retrieval | built | `src/maintidoc/retrieval/` | BGE-M3 (1024-d), CLIP (512-d), Qdrant index with access filter, RRF fusion, 4 reranker arms. |
| MCP server + client | built | `src/maintidoc/mcp/` | `MaintenanceService` with authorization on every call, `McpClient` with budget tracking. |
| LangGraph workflow | built | `src/maintidoc/agent/workflow.py` | State machine executing SPEC §6 node sequence, budgets, human approval interrupt. |
| FastAPI / Streamlit | built | `src/maintidoc/api/`, `src/maintidoc/ui/` | Authenticated endpoints `/ask`, `/approval-token`, `/tickets/approve`; interactive UI. |
| Docker Compose | built | `docker-compose.yml`, `Dockerfile` | Qdrant, Postgres, SeaweedFS (ADR 0006), MLflow, FastAPI, Streamlit. |
| Structured retry & citations | built | `src/maintidoc/agent/` | Bounded retries (`validation.py`), evidence check (`citations.py`), abstention (`uncertainty.py`). |
| Tracking & MLOps | built | `src/maintidoc/tracking.py` | Standardized MLflow run metadata convention (SPEC §7, §11). |

## Data flow (as implemented)

```
request ──> schemas.AnswerPayload          (the only response shape)
            ▲
            │
     agent.drafts.ModelAnswerDraft          (lenient model output)
            ▲
     vlm.VlmProvider.generate(expects=…)    (stub today, Qwen later)
            ▲
     retrieval.render.render_sources()      (retrieved elements → <sources>)
            ▲
     [ retrieval, embeddings, index ]       ← NOT IMPLEMENTED
```

The trusted boundary is explicit: retrieved text is rendered into delimited tags
and declared as `untrusted_variables` in the prompt, and the loader refuses to
load a prompt that does not delimit them. Nothing above `render_sources()` treats
document text as an instruction.

## Interfaces / contracts

- Output schema: `src/maintidoc/schemas.py` (SPEC §7)
- Model draft schema: `src/maintidoc/agent/drafts.py`
- MCP tools: `src/maintidoc/mcp/contracts.py` (SPEC §6)
- Object storage: `src/maintidoc/storage/objects.py`
- VLM provider: `src/maintidoc/vlm/base.py`

## Trust boundaries

| Boundary | Enforced by | Test |
|---|---|---|
| user/group → documents | `security/access.py` filter + `require_access` | `tests/test_security_access.py` |
| model → sensitive action | `security/approval.py` content-bound token | `tests/test_security_approval.py` |
| document text → instructions | prompt `untrusted_variables` + loader validation | `tests/test_prompts.py` |
| model output → API response | `schemas.AnswerPayload` invariants | `tests/test_schemas.py` |
| retrieval → citation | render/parse round-trip | `tests/test_retrieval_render.py` |

## Known deviations from SPEC

| Deviation | ADR | Status |
|---|---|---|
| `nox` is the task runner; `make` delegates to it | 0002 | accepted |
| Model may step down 7B → 3B → CPU instead of the spec'd 7B only | 0003 | accepted, pending measurement |
| Heavy ML deps optional; deterministic stub for tests and CPU baseline | 0004 | accepted |
| jinja2 template format for prompts | 0005 | accepted |
| MinIO replaced by SeaweedFS (S3-compatible) | 0006 | accepted |

## MLflow Run Metadata Convention (SPEC §7, §11)

Every evaluation run, retrieval benchmark, and fine-tuning experiment logged to MLflow (`src/maintidoc/tracking.py`) must record the following parameters:

| Parameter Key | Type | Description | Example |
|---|---|---|---|
| `model_id` | `str` | Exact identifier of the VLM serving the run | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` or `maintidoc-stub-vlm` |
| `provider` | `str` | Inference provider engine | `qwen` or `stub` |
| `prompt_versions` | `str` | Sorted map of versioned prompts used | `answer:v1,visual_analysis:v1` |
| `reranker` | `str` | Reranker arm configured | `none`, `deterministic`, `bge`, or `adapter` |
| `index_version` | `str` | Qdrant index schema and embedding version | `qdrant_v1` |
| `git_commit` | `str` | Exact git commit SHA under evaluation | `eeec7e9` |
| `device` | `str` | Execution device selected | `cuda`, `mps`, or `cpu` |
| `env` | `str` | Application environment | `dev`, `test`, or `prod` |

Metrics logged:
- Retrieval: `recall@5`, `ndcg@5`, `mrr@10`, `query_count`
- Citations: `validity_rate`, `supported_citations`, `total_citations`
- Workflow: `latency_ms`, `model_calls_used`, `mcp_calls_used`
- Security / Refusal: `refusal_rate`, `injection_success_rate`
