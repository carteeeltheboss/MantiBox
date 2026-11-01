# STATUS

> Updated at the end of EVERY session by the LLM. A cold-start LLM must be able to continue from this file alone.

**Last updated:** 2026-11-01 by Qwen Code
**Current milestone:** W1 Cadrage — tooling and contracts done, pipeline in progress
**Branch / PR state:** `main`, pushed through `bffbe9c`, nothing in flight

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
- **Corpus generator** (`scripts/generate_corpus.py`): seeded, two equipment families, one scanned document with a zero-length text layer, the SPEC §9 injection document, a restricted-group document, figure PNGs, committed manifest.

Verified: `nox -s check` green (ruff, format, mypy on 25 modules, **259 tests**), `doctor` runs.

## In progress

- Nothing half-written. The next unit of work (ingestion) has not been started.

## Next (ordered)

1. **Ingestion**: `ingestion/pdf.py` (PyMuPDF extraction: paragraphs, tables, figures, page images) + `ingestion/ocr.py` (tesseract, optional) producing `Element`s with the SPEC §4.1 metadata.
2. **Embeddings** behind seams: BGE-M3 text, multilingual CLIP image, each with a deterministic test double.
3. **Qdrant indexing** with the access-groups payload and access-filtered search.
4. **RRF fusion** + reranker (base, adapted, and none — the "no reranker" arm is required by SPEC §5).
5. **MCP server + client** implementing the contracts, with per-call authorization.
6. **LangGraph workflow**: the SPEC §6 node sequence, budgets (3 MCP / 6 model / 60 s), citation verification, abstention logic.
7. **FastAPI + Streamlit**: JWT auth, access filtering on every endpoint, upload limits, approval screen.
8. **Docker Compose + Dockerfiles**: CPU baseline plus an optional `gpu` profile.
9. **Eval harness** with seed scenarios and Recall@5 / nDCG@5 / MRR@10.
10. **`bench-vlm`**, then update ADR 0003 with measured VRAM.

## Blockers / waiting on human

- **Object storage (SPEC §3)**: MinIO is unpullable from every public source (Docker Hub denied, quay.io and bitnami no manifest, `dl.min.io` HTTP 410). ADR 0006 is `proposed`. **Waiting for the user to supply a working MinIO image reference**, after which either SPEC §3 is satisfied or a replacement backend is chosen.
- **No `gh` CLI** on the dev host, so `scripts/create_backlog.sh` cannot run and the backlog is not in GitHub Issues.

## Open questions

1. Object storage backend (ADR 0006).
2. How the ~150-scenario eval set (SPEC §10) will be annotated — it needs human verification, so generated scenarios must be marked unverified until reviewed.

## How to run right now

Windows host (no `make`):

```bat
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\nox -s check          :: lint + mypy + tests
.venv\Scripts\maintidoc doctor      :: what this machine can actually run
python scripts/generate_corpus.py --count 8 --corpus-dir .tmp-corpus
```

Linux / CI / containers: `make check`, `make doctor`, `make corpus`.

## Unverified / known gaps

Be honest about these — none of them have been exercised:

- **The Qwen2.5-VL 7B has never been loaded.** `vlm/qwen.py` does not exist yet. Every VRAM figure in ADR 0003 is an *estimate*; the measurement SPEC §11 requires has not been taken.
- **No OCR has ever run.** tesseract is not on the Windows host and no image has been built.
- **No integration test has run** against real Qdrant, PostgreSQL or an object store.
- **No Docker image or compose file exists yet.**
- **CI has never executed** on GitHub.
- `nox -s ingest`, `-s eval` and `-s bench_vlm` reference modules that do not exist yet; those sessions currently fail by design.
- The corpus has only been generated at `--count 8` for a smoke test. The default of 34 has not been produced or verified.
- No fine-tuning work started (SPEC §5); no reranker dataset exists.

## Pointers

- Spec: `docs/SPEC.md` · Architecture: `docs/ARCHITECTURE.md` · Decisions: `docs/decisions/` · Session logs: `docs/sessions/`
- Backlog: GitHub Issues/Projects **not yet created** (no `gh` CLI)