# STATUS

> Updated at the end of EVERY session by the LLM. A cold-start LLM must be able to continue from this file alone.

**Last updated:** 2026-11-01 by Qwen Code
**Current milestone:** W2 Ingestion et recherche — ingestion done, embeddings next
**Branch / PR state:** `main`, pushed through `a8fb813`, nothing in flight
**Backlog:** GitHub issues #1–#37 exist. Closed: 1, 2, 3, 4, 8, 11, 12, 13, 14, 17.

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
- **`doctor` CLI** (`src/maintidoc/cli.py`): device, model plan with reasons, prompt versions, corpus integrity, object-store writability, tesseract availability, service reachability.

Verified: `nox -s check` green (ruff, format, mypy on 32 modules); **409 tests** including 41 integration tests against a live `qdrant/qdrant:v1.20.0`. `doctor` runs.

## In progress

- Nothing half-written. Next unit: RRF fusion and the reranker arms (issue #15).

## Next (ordered)

1. **RRF fusion + reranker arms** (#15): base, adapted and none — the "no reranker" arm is required by SPEC §5.
2. **Vertical slice** (#9): question → retrieve → cited answer through the stub VLM.
3. **MCP server + client** (#22, #23) with per-call authorization.
4. **LangGraph workflow** (#24): SPEC §6 node sequence, budgets, citation verification, abstention.
5. **FastAPI + Streamlit** (#21).
6. **Docker Compose** (#10, SeaweedFS per ADR 0006).
7. **Eval harness** (#6, #30) with seed scenarios.
8. **`bench-vlm`** (#16), then measured VRAM recorded in ADR 0003.
9. **Install the `ml` extra** and verify the real BGE-M3 and CLIP towers load and align, replacing the doubles in the integration tests.
10. **Threat model** (#5) and **MLflow convention** (#7).

## Blockers / waiting on human

- **None blocking.** Object storage is decided: **SeaweedFS**, see ADR 0006
  (accepted). MinIO is unpullable from every public source, so SPEC §3's vendor is
  replaced while its semantics are kept; the code talks the `ObjectStore`
  interface either way.
- **No `gh` CLI was on PATH** at session start; it is now installed and the
  backlog exists in GitHub Issues.

## Open questions

1. How the ~150-scenario eval set (SPEC §10) will be annotated — it needs human
   verification, so generated scenarios must be marked unverified until reviewed.

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

- **BGE-M3 and CLIP weights have never been loaded.** Weight downloads are deferred, so the index is proven with the deterministic hashing double. Ordering, access filtering and payload shape are verified; **semantic retrieval quality is not**, and any Recall@5 computed with the double would be meaningless.
- **The Qwen2.5-VL 7B has never been loaded.** `vlm/qwen.py` does not exist yet. Every VRAM figure in ADR 0003 is an *estimate*; the measurement SPEC §11 requires has not been taken.
- **No OCR has ever run.** tesseract is not on the Windows host and no image has been built.
- **No integration test has run against PostgreSQL or an object store.** Qdrant
  is covered (32 tests); Postgres and S3 are not.
- **No Docker image or compose file exists yet.**
- **CI has never executed** on GitHub.
- `nox -s ingest`, `-s eval` and `-s bench_vlm` reference modules that do not exist yet; those sessions currently fail by design.
- The corpus is generated at the full 34 documents, but nothing has been **ingested** into the index yet: no end-to-end run has loaded the corpus, embedded it and searched it.
- No fine-tuning work started (SPEC §5); no reranker dataset exists.

## Pointers

- Spec: `docs/SPEC.md` · Architecture: `docs/ARCHITECTURE.md` · Decisions: `docs/decisions/` · Session logs: `docs/sessions/`
- Backlog: GitHub Issues/Projects **not yet created** (no `gh` CLI)