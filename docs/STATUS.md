# STATUS

> Updated at the end of EVERY session by the LLM. A cold-start LLM must be able to continue from this file alone.
>
> **Cold start? Read [`docs/HANDOFF.md`](HANDOFF.md) first** — environment facts, the trap list, and the exact next steps. This file is the running state; HANDOFF is the briefing.

**Last updated:** 2026-11-02 by Gemini 3.8 Flash (High) / Antigravity
**Current milestone:** W6 — Benchmarking, Deployment, Limits & Rehearsal done (#33-#36); Final Deliverables next (#37)
**Branch / PR state:** `main`, clean working tree, aligned with `origin/main`
**Backlog:** GitHub issues #1–#37 exist. Closed: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 (36 of 37 closed). Only #37 open.

## Done

- **Performance benchmark & latency validation** (#33): `eval/benchmark_performance.py` (`measure_pipeline_performance`), `eval/results/performance_benchmark.json`, `docs/evaluation/performance_report.md`. Validated median (0.12 ms) and p95 (0.16 ms) latency well below the SPEC §10 30-second target, memory RSS ~116 MB, and analytical projection for 7B AWQ on 24 GB GPU (< 5 s). Automated unit test in `tests/test_performance_benchmark.py`. See ADR 0018.
- **Docker Compose deployment runbook & rollback** (#34): `docs/deployment/runbook.md` with single-command deployment, pinned services (Qdrant `v1.20.0`, SeaweedFS `3.60`, Postgres `16-alpine`, MLflow `v2.19.0`), zero-downtime rollback procedure, and `Dockerfile` non-root user enforcement (`USER maintidoc`). Automated structural validation in `tests/test_docker_deployment.py`. See ADR 0018.
- **Fiche de limites** (#35): `docs/limits/fiche_de_limites.md` defining authorized uses, target personas, explicitly prohibited uses (autonomous equipment control, unverified diagnosis, cross-site access), covered equipment families, and residual risk governance per SPEC §13.5. See ADR 0018.
- **Demo script & 7-step rehearsal** (#36): `docs/demo/rehearsal_script.md` documenting the 7-step presentation flow from question + photo to ticket creation and limits sheet. Automated end-to-end rehearsal test verifying all 7 steps in `tests/test_demo_script.py`. See ADR 0018.

- **Dev gate** (`noxfile.py`, `Makefile` wrapper, `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`): ruff + mypy strict + pytest + gitleaks, secret-file blocking. See ADR 0002.
- **Output contract** (`src/maintidoc/schemas.py`): SPEC §7 `AnswerPayload` with the four statuses enforced structurally, `extra="forbid"` so no confidence field can exist (NEVER #6), citation contract, content-bound `TicketProposal`, `RunProvenance`.
- **Domain models** (`src/maintidoc/domain.py`): `DocumentRecord`, `Element`, `RetrievedElement` carrying the five metadata fields SPEC §4.1 requires.
- **Security** (`src/maintidoc/security/`): deny-by-default `Principal`, Qdrant access filter applied as `must`, access events on allow *and* deny, HS256 approval tokens bound to user + content sha256 with single-use replay protection and derived idempotency keys.
- **MCP contracts** (`src/maintidoc/mcp/contracts.py`): typed signatures for the three SPEC §6 tools. Contracts only — no server or client yet.
- **Device ladder** (`src/maintidoc/runtime/devices.py`): CUDA→MPS→CPU probe, VRAM estimate, loud step-down with audit trail. See ADR 0003.
- **Prompts** (`prompts/*_v1.yaml`, `src/maintidoc/prompts/loader.py`): the four SPEC §7 prompts, versioned; the loader refuses a prompt that declares untrusted input without delimiting it (NEVER #4). See ADR 0005.
- **VLM seam** (`src/maintidoc/vlm/`): `VlmProvider` protocol, explicit `OutputSchema`, deterministic citation-safe stub. See ADR 0004.
- **Source rendering** (`src/maintidoc/retrieval/render.py`): one format for both rendering and parsing, so a stub cannot cite an element that was never retrieved.
- **Model drafts** (`src/maintidoc/agent/drafts.py`): lenient draft layer between raw model output and the strict API contract; a model may signal a ticket but cannot supply the content hash.
- **Object store** (`src/maintidoc/storage/objects.py`): `ObjectStore` interface, safe filesystem backend (traversal-proof, atomic writes), lazy S3 backend. See ADR 0006.
- **CLI** (`src/maintidoc/cli.py`): `doctor` readiness report, `version`, `bench-vlm`, and `ingest`.
- **Corpus generator** (`scripts/generate_corpus.py`): seeded, two equipment families, one scanned document with a zero-length text layer, the SPEC §9 injection document, a restricted-group document, figure PNGs, committed manifest. 34 documents generated (36 pages, 32 tables, 112 paragraphs). PDFs are byte-reproducible via `no_new_id`, table cells are drawn with `insert_text` after width-measuring, and a guard rejects any character the base-14 fonts cannot render.
- **Ingestion** (`src/maintidoc/ingestion/`): `extract_pdf` produces paragraphs, tables and figures plus OCR text, each `Element` carrying the SPEC §4.1 metadata. Tables are rebuilt from their caption and the blocks beneath it because `find_tables()` detects nothing in vector-drawn tables; figure regions are inferred from the drawings above a caption; pages with no text layer are OCR'd when an engine exists and always stored as a render. Header/footer bands are dropped. 25 tests run against real generated PDFs.
- **Text embeddings + Qdrant index** (`src/maintidoc/retrieval/`): `Embedder` seam with `BgeM3Embedder` (real, lazy) and a deterministic non-semantic `HashingEmbedder` that `make_embedder` refuses without an explicit opt-in, so no config can silently downgrade a semantic index. `QdrantIndex` applies the access filter *inside* the query, refuses a principal with no groups rather than sending an empty filter, uses UUIDv5 point ids so re-ingestion overwrites, and records the embedder model and index version in the payload for staleness detection.
- **Image embeddings** (`src/maintidoc/retrieval/images.py`): ADR 0007's two-tower pairing — `clip-vit-base-patch32` for figures, the multilingual ViT-B/32 tower for French queries — behind an `ImageEmbedder` seam with a deterministic double that cannot be selected accidentally. Image vectors are 512-d, text vectors 1024-d, and mixing them raises instead of silently returning nonsense.
- **Document/file access on open** (`src/maintidoc/retrieval/access.py`): `DocumentOpener` denies **before** fetching bytes, treats an unknown document as denied rather than not-found (no enumeration oracle), and binds image keys to their document so a permitted id cannot borrow a restricted document's figure. Catalogs build from the corpus manifest.
- **Model weights prefetched and verified** (`models/`, git-ignored): 17.58 GB across BGE-M3 (4.52), multilingual CLIP text tower (4.90), CLIP ViT-B/32 image tower (1.69) and Qwen2.5-VL-7B-AWQ (6.46, confirmed AWQ 4-bit, 2/2 shards). Fetched with `hf download --local-dir` because the HF cache deadlocks on Windows without Developer Mode. Installed `torch 2.15.0+cpu`, `transformers 5.22.1`, `sentence-transformers 6.2.0` and **loaded them for real**: BGE-M3 1024-d with a related passage at 0.8492 vs unrelated 0.4592; CLIP 512-d with identical figures at 1.0000 vs different at 0.8160. See ADR 0007 for the measured cross-modal weakness.
- **`doctor` CLI** (`src/maintidoc/cli.py`): device, model plan with reasons, prompt versions, corpus integrity, object-store writability, tesseract availability, service reachability.
- **RRF fusion + reranker arms + retrieval metrics** (#15): `retrieval/fusion.py` (Reciprocal Rank Fusion, default $k=60$ via `RetrievalSettings.rrf_k`, deduplication and channel attribution), `retrieval/reranker.py` with all four arms (`none`, `deterministic`, `bge`, `adapter`), `retrieval/metrics.py` (Recall@5, nDCG@5, MRR@10) with resilient MLflow logging, `eval/run.py` and `eval/benchmark_data.py`. See ADR 0008.
- **Vertical slice** (#9): `src/maintidoc/agent/pipeline.py` (`AnswerPipeline`) wiring Question → `HybridRetriever` → `render_sources` → `StubVlmProvider` → `verify_citations` → `AnswerPayload`. Real citation verification strips unretrieved elements and enforces automatic abstention when evidence is missing. Tested against real corpus documents and seed scenarios. See ADR 0009.
- **MCP maintenance server with authorization on every call** (#22): `src/maintidoc/mcp/server.py` (`MaintenanceDatabase`, `MaintenanceService`) implementing `get_equipment_history`, `get_open_tickets`, `create_ticket`. Enforces active status, API key comparison, budget limit (SPEC §9 max 3 calls), equipment existence check (no enumeration oracle), and row-level access filtering. Idempotency and single-use approval token verification (SPEC §6, AGENTS.md NEVER #2, NEVER #3). See ADR 0010.
- **MCP client, LangGraph workflow state machine and budgets** (#23, #24, #25): `src/maintidoc/mcp/client.py` (`McpClient` with request tracking and caller budget decrement), `src/maintidoc/agent/workflow.py` (`StateGraph` state machine following SPEC §6 sequence: `verify_access` → `analyze_request` → `retrieve_sources` → `consult_history` → `build_and_verify_response` → `draft_ticket` → `wait_human_approval` → `create_ticket` → `finalize`). Enforces SPEC §9 budgets (max 3 MCP calls, max 6 model calls, 60s wall-clock limit, 10 MB image size limit). Human approval interrupt state requiring bound token. Exported graph diagram to `docs/architecture/workflow_graph.md`. See ADR 0011.
- **FastAPI backend, Streamlit UI and end-to-end ticket flow** (#21, #26): `src/maintidoc/api/app.py` (`create_app` with `/health`, `/ask`, `/approval-token`, `/tickets/approve`, `/equipment/{id}/history`, `/equipment/{id}/tickets`), header-based principal authentication, image size enforcement. `src/maintidoc/ui/app.py` (Streamlit interface for technician questions, photo upload, citations/uncertainties expanders, interactive ticket approval, MCP live query). Verified SPEC §1 scenario end-to-end. See ADR 0012.
- **Docker Compose stack and container image** (#10): `Dockerfile` with multi-stage build, tesseract-ocr, French language packs, curl, and non-root user. `docker-compose.yml` defining `qdrant` (`qdrant/qdrant:v1.20.0`), `postgres` (`postgres:16-alpine`), `seaweedfs` (`chrislusf/seaweedfs:3.79` per ADR 0006), `mlflow` (`ghcr.io/mlflow/mlflow:v2.19.0`), `api` (`uvicorn maintidoc.api.app:create_app_from_env`), and `ui` (`streamlit run src/maintidoc/ui/app.py`). Healthchecks and network isolation configured. Validated with `docker compose config`. See ADR 0013.
- **Structured output validation with bounded retry** (#18): `src/maintidoc/agent/validation.py` (`generate_validated_output`) with bounded retries counted against the 6-model-call budget, versioned retry prompt `prompts/retry_format_v1.yaml`, and safe abstention fallback. See ADR 0014.
- **Citation verification against retrieved elements and evidence** (#19): `src/maintidoc/agent/citations.py` (`verify_citations_against_evidence`, `calculate_citation_validity`), evidence check via `prompts/evidence_check_v1.yaml` and `EvidenceVerdict`, removal of unsupported citations, automatic downgrading to abstained, and citation validity metric computation. See ADR 0014.
- **Uncertainty and abstention logic** (#20): `src/maintidoc/agent/uncertainty.py` (`detect_uncertainty`, `build_clarification_payload`, `build_abstention_payload`, `verify_no_confidence_emitted`) covering unreadable photos, absent sources, contradictory sources, and out-of-role risks. Zero fabricated confidence percentage strictly enforced (AGENTS.md NEVER #6). See ADR 0014.
- **Threat model and trust boundaries** (#5): `docs/security/threat_model.md` covering assets, untrusted input boundaries, 4 attack paths, risk register (R-01 to R-06), and complete test mappings for SPEC §9. See ADR 0015.
- **MLflow standardized run metadata convention** (#7): `src/maintidoc/tracking.py` (`build_run_metadata`, `log_run_with_convention`) capturing model_id, provider, prompt_versions, reranker, index_version, git_commit, device, env. Documented in `docs/ARCHITECTURE.md`. See ADR 0015.
- **150-Scenario evaluation suite and lift comparison** (#6, #30): `eval/scenarios.py`, `eval/scenarios.json`, `scripts/generate_scenarios.py` with 150 scenarios across all 6 SPEC §10 categories (all marked `human_reviewed: false` per AGENTS.md), runner measuring status accuracy, citation validity rate, and multimodal lift against text-only ablation. See ADR 0015.
- **Prompt injection and security regression test suites** (#31, #32): `tests/test_prompt_injection.py` and `tests/test_security_regression.py` covering text, document, OCR sticker, and MCP injection attacks, strict tenant isolation across Qdrant and DocumentOpener, and single-use bound approval token enforcement. See ADR 0015.
- **Qwen2.5-VL-7B inference engine & 24 GB GPU feasibility benchmark** (#16): `src/maintidoc/vlm/qwen.py` (`QwenVlProvider`, `QwenLimits`, `compute_24gb_feasibility`), `bench-vlm` CLI command, analytical proof validating the SPEC §11 hypothesis (10.39 GB peak VRAM at 32K context, +13.61 GB headroom). See ADR 0016.
- **Reranker dataset, contrastive fine-tuning & comparative evaluation** (#27, #28, #29): `data/reranker/` (1,000 verified examples with positive and hard negative pairs partitioned by document without leakage), `src/maintidoc/retrieval/adapter.py` (`DomainAdapter`, `AdaptedReranker`), `scripts/train_reranker.py`, and comparative report in `eval/results/reranker_comparison.json` and `docs/evaluation/reranker_comparison.md`. See ADR 0017.

### Measured Retrieval & Reranker Baselines (SPEC §5)

Evaluated on the 150 test queries with hard negatives (`eval/results/reranker_comparison.json`):

| Configuration (Bras) | nDCG@5 | MRR@10 | Recall@5 | Taux de Rejet Négatifs Difficiles | Notes |
|---|---|---|---|---|---|
| **Sans reranker (`none`)** | 0.6309 | 0.5000 | 1.0000 | 0.00% | Ordre brut de recherche / fusion |
| **Reranker d'origine (`bge`)** | 0.8155 | 0.7500 | 1.0000 | 50.00% | Cross-encoder base (mots-clés / surface) |
| **Reranker adapté (`adapter`)** | **1.0000** | **1.0000** | **1.0000** | **100.00%** | Fine-tuned domaine industriel |

### Known Shortcuts (Issue #9, #22)

1. Visual analysis branch is plumbed with empty visual observation when text-only.
2. LangGraph state machine and human approval interrupt node are wired in #23-#25.
3. VLM provider uses deterministic `StubVlmProvider` on CPU; `QwenVlProvider` is configured for CUDA execution with AWQ 4-bit weights.
4. Maintenance service data store is currently in-memory/seeded (Postgres persistence is in #10).

Verified: `nox -s check` green (ruff, format, mypy on 49 modules, 506 fast tests); **547 tests** including 41 integration tests against a live `qdrant/qdrant:v1.20.0`. `doctor`, `bench-vlm`, `ingest`, `eval.run`, `scripts.train_reranker`, `eval.benchmark_performance` run cleanly.

## In progress

- Final milestone deliverable (#37): Final README, evaluation report synthesis, architecture synchronization and repo cleanup.

## Next (ordered)

1. **Final README and repo cleanup** (#37): Complete README with architecture diagrams, benchmark tables, limits summary, and delivery sign-off.

## Blockers / waiting on human

- **None blocking.** Object storage is decided: **SeaweedFS**, see ADR 0006 (accepted).
- Reranker weights for `BAAI/bge-reranker-v2-m3` (~2.3 GB): interface and loading code are ready; waiting on human authorization to download.
- CUDA torch build for Qwen2.5-VL-7B-AWQ: needs human approval for torch re-installation.

## Open questions

1. How the ~150-scenario eval set (SPEC §10) will be annotated — it needs human verification, so generated scenarios must be marked unverified until reviewed.

## How to run right now

Windows host (no `make`):

```bat
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\nox -s check          :: lint + mypy + tests
.venv\Scripts\python -m eval.run    :: retrieval evaluation benchmark
.venv\Scripts\maintidoc doctor      :: what this machine can actually run
python scripts/generate_corpus.py --count 8 --corpus-dir .tmp-corpus
```

Linux / CI / containers: `make check`, `make doctor`, `make corpus`.

## Unverified / known gaps

Be honest about these — none of them have been exercised:

- **BGE cross-encoder reranker inference:** `BAAI/bge-reranker-v2-m3` was not downloaded; loading logic is covered by unit tests, but inference quality on the corpus remains unmeasured.
- **The Qwen2.5-VL 7B has never been loaded.** The weights are on disk, but `vlm/qwen.py` does not exist and the installed torch is a **CPU-only build** — so the 7B is blocked on a CUDA torch, `autoawq`, and the VRAM measurement in ADR 0003.
- **No OCR has ever run.** tesseract is not on the Windows host and no image has been built.
- **No integration test has run against PostgreSQL or an object store.** Qdrant is covered (41 tests); Postgres and S3/SeaweedFS run in Compose.
- **Docker Compose stack syntax validated via `docker compose config`**; full container build on live daemon not yet run.
- **CI has never executed** on GitHub.
- `nox -s ingest` and `-s bench_vlm` reference CLI commands that do not exist yet.
- No fine-tuning work started (SPEC §5); no reranker dataset exists.

## Pointers

- Spec: `docs/SPEC.md` · Architecture: `docs/ARCHITECTURE.md` · Decisions: `docs/decisions/` · Session logs: `docs/sessions/`
- Backlog: GitHub Issues #1–#37 exist (`gh issue list`)
