# STATUS

> Updated at the end of EVERY session by the LLM. A cold-start LLM must be able to continue from this file alone.
>
> **Cold start? Read [`docs/HANDOFF.md`](HANDOFF.md) first** — environment facts, the trap list, and the exact next steps. This file is the running state; HANDOFF is the briefing.

**Last updated:** 2026-11-02 by Gemini 3.8 Flash (High) / Antigravity
**Current milestone:** W2 Ingestion et recherche — vertical slice done, MCP server & client next
**Branch / PR state:** `main`, clean working tree, aligned with `origin/main`
**Backlog:** GitHub issues #1–#37 exist. Closed: 1, 2, 3, 4, 8, 9, 11, 12, 13, 14, 15, 17 (12 of 37 closed).

## Done

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
- **CLI** (`src/maintidoc/cli.py`): `doctor` readiness report and `version`.
- **Corpus generator** (`scripts/generate_corpus.py`): seeded, two equipment families, one scanned document with a zero-length text layer, the SPEC §9 injection document, a restricted-group document, figure PNGs, committed manifest. 34 documents generated (36 pages, 32 tables, 112 paragraphs). PDFs are byte-reproducible via `no_new_id`, table cells are drawn with `insert_text` after width-measuring, and a guard rejects any character the base-14 fonts cannot render.
- **Ingestion** (`src/maintidoc/ingestion/`): `extract_pdf` produces paragraphs, tables and figures plus OCR text, each `Element` carrying the SPEC §4.1 metadata. Tables are rebuilt from their caption and the blocks beneath it because `find_tables()` detects nothing in vector-drawn tables; figure regions are inferred from the drawings above a caption; pages with no text layer are OCR'd when an engine exists and always stored as a render. Header/footer bands are dropped. 25 tests run against real generated PDFs.
- **Text embeddings + Qdrant index** (`src/maintidoc/retrieval/`): `Embedder` seam with `BgeM3Embedder` (real, lazy) and a deterministic non-semantic `HashingEmbedder` that `make_embedder` refuses without an explicit opt-in, so no config can silently downgrade a semantic index. `QdrantIndex` applies the access filter *inside* the query, refuses a principal with no groups rather than sending an empty filter, uses UUIDv5 point ids so re-ingestion overwrites, and records the embedder model and index version in the payload for staleness detection.
- **Image embeddings** (`src/maintidoc/retrieval/images.py`): ADR 0007's two-tower pairing — `clip-vit-base-patch32` for figures, the multilingual ViT-B/32 tower for French queries — behind an `ImageEmbedder` seam with a deterministic double that cannot be selected accidentally. Image vectors are 512-d, text vectors 1024-d, and mixing them raises instead of silently returning nonsense.
- **Document/file access on open** (`src/maintidoc/retrieval/access.py`): `DocumentOpener` denies **before** fetching bytes, treats an unknown document as denied rather than not-found (no enumeration oracle), and binds image keys to their document so a permitted id cannot borrow a restricted document's figure. Catalogs build from the corpus manifest.
- **Model weights prefetched and verified** (`models/`, git-ignored): 17.58 GB across BGE-M3 (4.52), multilingual CLIP text tower (4.90), CLIP ViT-B/32 image tower (1.69) and Qwen2.5-VL-7B-AWQ (6.46, confirmed AWQ 4-bit, 2/2 shards). Fetched with `hf download --local-dir` because the HF cache deadlocks on Windows without Developer Mode. Installed `torch 2.15.0+cpu`, `transformers 5.22.1`, `sentence-transformers 6.2.0` and **loaded them for real**: BGE-M3 1024-d with a related passage at 0.8492 vs unrelated 0.4592; CLIP 512-d with identical figures at 1.0000 vs different at 0.8160. See ADR 0007 for the measured cross-modal weakness.
- **`doctor` CLI** (`src/maintidoc/cli.py`): device, model plan with reasons, prompt versions, corpus integrity, object-store writability, tesseract availability, service reachability.
- **RRF fusion + reranker arms + retrieval metrics** (#15): `retrieval/fusion.py` (Reciprocal Rank Fusion, default $k=60$ via `RetrievalSettings.rrf_k`, deduplication and channel attribution), `retrieval/reranker.py` with all four arms (`none`, `deterministic`, `bge`, `adapter`), `retrieval/metrics.py` (Recall@5, nDCG@5, MRR@10) with resilient MLflow logging, `eval/run.py` and `eval/benchmark_data.py`. See ADR 0008.
- **Vertical slice** (#9): `src/maintidoc/agent/pipeline.py` (`AnswerPipeline`) wiring Question → `HybridRetriever` → `render_sources` → `StubVlmProvider` → `verify_citations` → `AnswerPayload`. Real citation verification strips unretrieved elements and enforces automatic abstention when evidence is missing. Tested against real corpus documents and seed scenarios. See ADR 0009.

### Measured Retrieval Baselines (Issue #15)

Evaluated on the seed scenarios via `eval/run.py`:

| Configuration | Recall@5 | nDCG@5 | MRR@10 | Notes |
|---|---|---|---|---|
| **Hash Doubles (No Reranker)** | 0.4722 | 0.4059 | 0.5556 | Deterministic, torch-free fast double |
| **Hash Doubles (Deterministic Reranker)** | 0.5556 | 0.3908 | 0.3681 | Keyword overlap reranker on hash hits |
| **Real BGE-M3 + CLIP (No Reranker)** | **0.7222** | **0.6132** | **0.8056** | Real semantic models, RRF $k=60$ |
| **Real BGE-M3 + CLIP (BGE Reranker)** | *unmeasured* | *unmeasured* | *unmeasured* | Weights not downloaded on host (~2.3 GB) |
| **Real BGE-M3 + CLIP (Adapted Reranker)**| *unmeasured* | *unmeasured* | *unmeasured* | Adapter training pending (#28) |

### Known Shortcuts (Issue #9)

1. Visual analysis branch is plumbed with empty visual observation when text-only.
2. LangGraph state machine and loop budgets are deferred to #24-#25.
3. VLM provider uses deterministic `StubVlmProvider` until CUDA environment is ready for the 7B (#16).

Verified: `nox -s check` green (ruff, format, mypy on 36 modules, 424 fast tests); **465 tests** including 41 integration tests against a live `qdrant/qdrant:v1.20.0`. `doctor` and `eval.run` run cleanly.

## In progress

- Next unit: MCP server & client (#22, #23).

## Next (ordered)

1. **MCP server + client** (#22, #23) with per-call authorization.
2. **LangGraph workflow** (#24): SPEC §6 node sequence, budgets, human-approval interrupt.
3. **FastAPI + Streamlit** (#21).
4. **Docker Compose** (#10, SeaweedFS per ADR 0006).
5. **Eval harness expansion** (#6, #30) with ~150 scenarios.
6. **`bench-vlm`** (#16), then measured VRAM recorded in ADR 0003.
7. **Threat model** (#5) and **MLflow convention** (#7).

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
- **No integration test has run against PostgreSQL or an object store.** Qdrant is covered (41 tests); Postgres and S3 are not.
- **No Docker image or compose file exists yet.**
- **CI has never executed** on GitHub.
- `nox -s ingest` and `-s bench_vlm` reference CLI commands that do not exist yet.
- No fine-tuning work started (SPEC §5); no reranker dataset exists.

## Pointers

- Spec: `docs/SPEC.md` · Architecture: `docs/ARCHITECTURE.md` · Decisions: `docs/decisions/` · Session logs: `docs/sessions/`
- Backlog: GitHub Issues #1–#37 exist (`gh issue list`)
