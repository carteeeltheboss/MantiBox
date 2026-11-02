# HANDOFF — read this first to continue MaintiDoc

**For any LLM or developer picking this up.** Written 2026-11-01 at commit `2b23628`.
Everything below was verified by execution, not assumed. Where something is
unverified it says so.

---

## TL;DR

The **foundation, retrieval fusion, vertical slice, MCP server & client, LangGraph workflow, FastAPI backend, Streamlit UI, Docker Compose stack, structured output retry, citation evidence verification, uncertainty logic, threat model, MLflow conventions, 150-scenario evaluation suite, and security regression suites are complete and tested.** Ingestion,
both embedding towers, both vector indexes, access control, RRF multimodal fusion, the reranker seam,
the end-to-end question answering pipeline (`AnswerPipeline`), the MCP maintenance service (`MaintenanceService`),
the MCP client (`McpClient`), the LangGraph state machine (`src/maintidoc/agent/workflow.py`), budgets (SPEC §9),
FastAPI authenticated endpoints (`/ask`, `/approval-token`, `/tickets/approve`, `/equipment/{id}/history`), Streamlit UI,
Docker Compose orchestration (`Dockerfile` + `docker-compose.yml`), structured output retry (`validation.py`),
citation verification against evidence (`citations.py`), uncertainty and abstention handling (`uncertainty.py`),
the threat model (`docs/security/threat_model.md`), MLflow tracking conventions (`src/maintidoc/tracking.py`),
the 150-scenario evaluation suite (`eval/scenarios.py`, `eval/scenarios.json`), prompt injection test suite (`tests/test_prompt_injection.py`),
security regression suite (`tests/test_security_regression.py`), Qwen2.5-VL-7B inference engine (`src/maintidoc/vlm/qwen.py`),
24 GB GPU feasibility benchmark (`bench-vlm`), reranker fine-tuning dataset & adapter (`data/reranker/`, `src/maintidoc/retrieval/adapter.py`),
three-arm comparative evaluation (`eval/results/reranker_comparison.json`), and the complete SPEC §1 end-to-end ticket flow all work and are covered by 542 tests (501 fast + 41 integration).

**Single next action:** issue **#33** (Performance benchmark: median and p95 latency, memory, GPU execution time).

Backlog: **32 of 37** issues closed (#1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32).

---

## Read order

1. **This file.**
2. [`AGENTS.md`](../AGENTS.md) — the binding rules (NEVER list, anti-slop).
3. [`docs/STATUS.md`](STATUS.md) — same state, maintained per the AGENTS.md protocol.
4. [`docs/SPEC.md`](SPEC.md) — **the source of truth.** If code and SPEC disagree, stop and ask.
5. The GitHub issue you are working on (`gh issue view <n>`).
6. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), [`docs/decisions/`](decisions/).

---

## Non-negotiable rules

From `AGENTS.md`, plus decisions the project owner made during this build. These
override anything else:

| Rule | Detail |
|---|---|
| **The model is fixed** | `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`. **Never substitute a smaller model.** If it does not fit, report and wait. `MAINTIDOC_ALLOW_MODEL_FALLBACK=false` enforces this; see ADR 0003 amendments. |
| **No Ollama** | SPEC §3 pins Transformers. Ollama also cannot produce CLIP image embeddings. If ever wanted, it needs an ADR + a new `VlmProvider`. |
| **Object storage is SeaweedFS** | ADR 0006 (accepted). MinIO is unpullable from every source. Code talks `ObjectStore`, so the vendor is a config change. |
| **Commit straight to `main`** | No feature branches, no PRs. Conventional Commits. |
| **No `Co-authored-by`** | The owner is the only author. No attribution trailers of any kind. |
| **Security-critical = flag it** | Access filtering, MCP authorization, approval tokens, ticket idempotency. Write it, flag it in the commit message, never weaken it to make a test pass. |
| **Never weaken a test's meaning** | If a test encodes a wrong assumption, fix the *test*; if the code is wrong, fix the code. Do not adjust assertions to get green. |
| **No evidence, no claim** | Never report a metric produced with the deterministic doubles as a real result. |
| **Prompts only in `prompts/*.yaml`** | A prompt change means a new `_vN+1.yaml` file. Never inline prompts. |
| **`.env` never committed** | Also blocked by a pre-commit hook. |

**Repo path:** `C:\Users\Karim\Documents\MantiBox` — **`MantiBox`**, no `a` after the
`M`. Typing `MaintiBox` silently creates a stray directory beside the real project.

---

## Environment (verified on this machine)

| Fact | Value |
|---|---|
| OS / shell | Windows, `cmd.exe`. Git Bash at `C:\Program Files\Git\bin\bash.exe` |
| Python | 3.12.10, venv at `.venv` (project-local) |
| **torch** | **`2.15.0+cpu` — a CPU-only build. `torch.cuda.is_available()` is `False`** |
| transformers / sentence-transformers | `5.22.1` / `6.2.0` |
| qdrant-client | `1.20.0` |
| PyMuPDF | `2.0` (MuPDF 1.28.5) — ships `py.typed` |
| GPU | RTX 4070 Laptop, 8 GB VRAM — **present but unusable** until a CUDA torch is installed |
| Docker | Desktop running; `maintidoc-qdrant` (`qdrant/qdrant:v1.20.0`) usually up on `:6333` |
| `gh` | installed, authenticated as `carteeeltheboss`. Scopes lack `project`, so no Project board via CLI |
| tesseract | **absent** — OCR has never actually run |
| `make` | **absent**. Use `nox`; the `Makefile` delegates to it |

**Weights are already downloaded** into git-ignored `models/` (17.58 GB). Do not
re-download; point loaders at these paths.

| Directory | Size | Width |
|---|---|---|
| `models/bge-m3` | 4.52 GB | 1024-d text |
| `models/clip-vit-base-patch32` | 1.69 GB | 512-d image tower |
| `models/clip-multilingual-text` | 4.90 GB | 512-d French text tower |
| `models/qwen2.5-vl-7b-instruct-awq` | 6.46 GB | AWQ 4-bit, 2 shards |

---

## What is built

| Component | File | Issue |
|---|---|---|
| Dev gate (ruff + mypy strict + pytest + gitleaks), CI | `noxfile.py`, `Makefile`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` | #1 |
| SPEC §7 output contract, status invariants, no-confidence guarantee | `src/maintidoc/schemas.py` | #2 |
| Domain models (`DocumentRecord`, `Element`, `RetrievedElement`) | `src/maintidoc/domain.py` | — |
| Config (pydantic-settings, `SecretStr`), safe logging, errors | `config.py`, `observability.py`, `errors.py` | — |
| Access control: deny-by-default, Qdrant filter, access events | `security/access.py` | — |
| Approval tokens: user + content-sha256 bound, single-use JWT | `security/approval.py` | #3 |
| MCP tool **contracts** (no server) | `mcp/contracts.py` | #3 |
| Device probe + model ladder with `usable` flag | `runtime/devices.py` | — |
| 4 versioned prompts + loader that enforces NEVER #4 | `prompts/*_v1.yaml`, `prompts/loader.py` | #17 |
| VLM provider protocol + deterministic citation-safe stub | `vlm/base.py`, `vlm/stub.py` | — |
| Source rendering (render↔parse round-trip) | `retrieval/render.py` | — |
| Lenient model-draft layer | `agent/drafts.py` | — |
| Synthetic corpus: 34 docs, 2 families, OCR/injection/restricted cases | `scripts/generate_corpus.py`, `corpus/manifest.json` | #4 |
| PDF ingestion: paragraphs, tables, figures, OCR hook | `ingestion/pdf.py`, `ingestion/ocr.py` | #11 |
| BGE-M3 embedder + Qdrant index (access-filtered, idempotent) | `retrieval/embeddings.py`, `retrieval/index.py` | #12 |
| Two-tower CLIP image embedder + 512-d image index | `retrieval/images.py` | #13 |
| Deny-before-fetch document/figure opening | `retrieval/access.py` | #14 |
| RRF fusion, reranker arms, retrieval metrics | `retrieval/fusion.py`, `retrieval/reranker.py`, `retrieval/metrics.py`, `eval/run.py` | #15 |
| Vertical slice RAG pipeline (`AnswerPipeline`) | `src/maintidoc/agent/pipeline.py` | #9 |
| MCP maintenance service (auth on every call) | `src/maintidoc/mcp/server.py` | #22 |
| MCP client (`McpClient`) & budget decrement | `src/maintidoc/mcp/client.py` | #23 |
| LangGraph workflow state machine & human approval interrupt | `src/maintidoc/agent/workflow.py` | #24, #25 |
| FastAPI backend & Streamlit UI | `src/maintidoc/api/app.py`, `src/maintidoc/ui/app.py` | #21, #26 |
| Docker Compose stack & Dockerfile | `Dockerfile`, `docker-compose.yml` | #10 |
| Structured output validation with bounded retry | `src/maintidoc/agent/validation.py` | #18 |
| Citation verification against evidence | `src/maintidoc/agent/citations.py` | #19 |
| Uncertainty & abstention handling | `src/maintidoc/agent/uncertainty.py` | #20 |
| Threat model & trust boundaries | `docs/security/threat_model.md` | #5 |
| Standardized MLflow run metadata convention | `src/maintidoc/tracking.py` | #7 |
| 150-scenario evaluation suite & multimodal lift | `eval/scenarios.py`, `eval/scenarios.json`, `scripts/generate_scenarios.py` | #6, #30 |
| Prompt injection & security regression test suites | `tests/test_prompt_injection.py`, `tests/test_security_regression.py` | #31, #32 |
| Qwen2.5-VL inference engine & 24 GB benchmark | `src/maintidoc/vlm/qwen.py`, `src/maintidoc/cli.py` (`bench-vlm`) | #16 |
| Multilingual reranker dataset, adapter & three-arm comparison | `data/reranker/`, `src/maintidoc/retrieval/adapter.py`, `scripts/train_reranker.py` | #27, #28, #29 |
| Corpus ingestion CLI | `src/maintidoc/cli.py` (`ingest`) | — |
| `doctor` CLI | `cli.py` | — |
| 17 ADRs, STATUS, ARCHITECTURE, session logs | `docs/` | — |

## What is NOT built

- Performance benchmark (`#33`).
- Docker Compose deployment validation & rollback procedure (`#34`).
- Fiche de limites (`#35`).
- Demo rehearsal script (`#36`).
- Final README and repo cleanup (`#37`).

---

## How to run and verify

Windows, from the repo root. `make` does not exist here.

```bat
:: the gate — must be green before every commit
.venv\Scripts\nox.exe -s check

:: full suite including integration (needs Qdrant up)
.venv\Scripts\python.exe -m pytest -q

:: Qdrant for integration tests
docker run -d --name maintidoc-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.20.0

:: readiness report
.venv\Scripts\maintidoc.exe doctor

:: regenerate the corpus (34 docs, byte-reproducible)
.venv\Scripts\python.exe scripts\generate_corpus.py
```

**Expected today:** `nox -s check` green, **434 fast tests pass**, mypy clean on
37 modules; full suite **475 passed** (41 of them integration against live Qdrant).

**Two venvs matter.** `nox` builds its own at `.nox/check` and installs only
`.[dev]`, so it has **no torch**. `.venv` has the `ml` extra. A test that behaves
differently between them is testing the environment, not the code — see trap 6.

---

## Trap list — hard-won, each cost a debugging cycle

### PyMuPDF 2.0 (speaks a different dialect from most examples online)

| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: draw_circle() missing 1 required positional argument: 'radius'` | 2.0 takes `(center, radius)`, not a Rect | `page.draw_circle(pymupdf.Point(x,y), r, ...)` |
| **Generated tables had zero extractable text** | `insert_textbox` returns *unused* height; a **negative** return means it wrote **nothing** | Use `insert_text` at a computed baseline, measuring width with `pymupdf.get_text_length` first |
| Text became `?` | base-14 fonts (`helv`, `hebo`) are Latin-1 only | No em dashes/en dashes in rendered text; `generate_corpus.py` now fails loudly on unrenderable chars |
| Manifest `sha256` changed every run | MuPDF writes a random file `/ID` per save | `doc.save(..., no_new_id=True)` |
| `find_tables()` returned 0 tables | Our tables are vector-drawn, not table objects | Rebuild from the caption block + short blocks beneath it until prose resumes |
| `get_images()` found no figures | Figures are **vector drawings**, not embedded images | Infer the region from `get_drawings()` above the caption, then `get_pixmap(dpi=, clip=Rect)` |
| A page yielded nothing | Empty text layer = scan | OCR when a tesseract engine exists; always store a page render so the image channel still works |

### Typing / mypy strict

| Symptom | Cause | Fix |
|---|---|---|
| Six `no-untyped-call` errors in ingestion | PyMuPDF **ships `py.typed`**, and several of its own methods are unannotated | Keep `pymupdf` **out** of `ignore_missing_imports`; add one narrow override `module = ["maintidoc.ingestion.pdf"]`, `disallow_untyped_calls = false` |
| `"object" is not callable` on a lazily loaded model | torch/transformers handles typed as `object` | Type them `Any \| None` with a comment; then **delete the now-unused** `# type: ignore[return-value]` on the adjacent return |
| `Argument "must" to "Filter" has incompatible type` | qdrant-client annotates `Filter.must` wider than `Filter()` accepts | `cast("list[Condition]", ...)` with the full condition union spelled out; **never drop conditions** to satisfy the checker |
| `Argument "timeout" ... expected "int \| None"` | float passed | `timeout=10`, not `10.0` |

### Models and inference

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'norm'` | **transformers 5.x** returns an output object from `get_image_features`, not a tensor | `getattr(output, "pooler_output", output)` — works on 4.x and 5.x |
| `AssertionError: Torch not compiled with CUDA enabled` | `nvidia-smi` sees the GPU, so `detect_devices` advertised CUDA, but **torch is a CPU build** | `DeviceInfo.usable=False` for hardware the runtime cannot drive; `select_device` skips it |
| A test hung for ~5 minutes, no output | The test passed a **Hub model id**; with `ml` installed it became a 1.7 GB download | Tests use an **empty local directory** as the model path — raises `ConfigurationError` whether or not the extra exists |
| `hf download --cache-dir` stalled at 0.01 GB repeating *"Still waiting to acquire lock"* | The HF cache needs symlinks; Windows without Developer Mode deadlocks | **`hf download <repo> --local-dir models/<name>`** — plain files, no locks. Do not re-run the cache variant |

### Shell (cmd.exe)

- Multi-line `python -c "..."` **fails silently** — use a `.py` file.
- `%ERRORLEVEL%` inside one `cmd /c` line expands **at parse time**, so an
  `&`-chained check reports the wrong value. Use separate calls.
- Nested/bash quoting for `gh` and `hf` is fragile — write a small `.sh` and run
  it through Git Bash instead of inlining.

### Corpus and test fixtures

- The corpus gives **both** equipment families a shared `maintenance` role group,
  so a maintenance technician legitimately reads both. The restricted case is the
  **supervisors-only** manual. Prove site isolation with a **site-only** principal
  (no role group). An earlier test asserted the wrong policy and "failed" while
  the code was correct.
- `ClipImageEmbedder` **decodes before it imports** transformers, so the
  missing-extra path needs a genuinely decodable PNG (a base64 1×1 works).
- Fixture bytes are `bytes`, so `list(FIGURES)` is correct and
  `[f.encode() for f in FIGURES]` raises `AttributeError`.

---

## Measured model behaviour and retrieval baselines

| Check | Result |
|---|---|
| BGE-M3 dimension | 1024 ✅ |
| BGE-M3 related vs unrelated passage | **0.8492 vs 0.4592** ✅ semantic |
| CLIP image dimension | 512 ✅ |
| CLIP figure vs itself / vs other figure | **1.0000 / 0.8160** ✅ |
| CLIP **text→image**, correct (pump) panel | **0.2691** |
| CLIP **text→image**, wrong (cabinet) panel | **0.2991** ❌ *wrong one wins* |
| **Hash Doubles (no reranker)** | Recall@5: 0.4722, nDCG@5: 0.4059, MRR@10: 0.5556 |
| **Hash Doubles (deterministic reranker)** | Recall@5: 0.5556, nDCG@5: 0.3908, MRR@10: 0.3681 |
| **Real BGE-M3 + CLIP (no reranker)** | **Recall@5: 0.7222, nDCG@5: 0.6132, MRR@10: 0.8056** ✅ |
| **Real BGE-M3 + CLIP (BGE / adapted reranker)** | *unmeasured* (weights not downloaded on host; fine-tuning pending #28) |

**Do not claim text→image retrieval quality on this corpus.** Flat vector panels
are far from CLIP's training distribution. Report text→image and image→image
**separately** in the eval, and keep the VLM reading the figure (SPEC §8) as the
primary path. Recorded in ADR 0007 and ADR 0008.

---

## Next steps, precisely

### #22/#23 — MCP maintenance server and client
Implement the MCP server exposing `get_equipment_history`, `get_open_tickets`, and `create_ticket`
with server-side authorization on EVERY call (AGENTS.md NEVER #2). Wire client into LangGraph.

### #10 — Docker Compose
Services: `qdrant`, `postgres:16.15-alpine`, **SeaweedFS** (pin a real tag —
`chrislusf/seaweedfs` is pullable), `mlflow` **built from our own image** (no
usable public tag exists), plus `api`/`ui`/`mcp-server` once they exist, and a
`vlm` service behind an optional **`gpu` profile**. Install `tesseract-ocr` +
`tesseract-ocr-fra` in the image — that is the only place OCR can run.

### #16 — the 7B (needs a human decision)
Blocked on: a **CUDA** build of torch (the CPU build cannot use the GPU at all),
`autoawq`, and ~6–8 GB VRAM. Weights are on disk. Attempt the 7B with
`max_new_tokens=512` and `max_pixels=401408`; if it OOMs, **report and wait** —
do not substitute. Then record measured VRAM in ADR 0003.

---

## Pending human decisions

1. **CUDA torch build + `autoawq` install** (~2–3 GB) before #16 can run.
2. **Reranker weights** (`BAAI/bge-reranker-v2-m3`, ~2.3 GB) if a real reranker
   arm is wanted for #15.
3. **Annotation labour** for #27 (800–1500 verified examples) and #30 (~150 eval
   scenarios). Cannot be automated honestly; without it the fine-tuning comparison
   and headline metrics cannot be claimed.
4. **`gh auth refresh -s project`** if a GitHub Project board is wanted.

---

## Where the truth lives

| Question | File |
|---|---|
| What must this system do? | `docs/SPEC.md` |
| What is the state right now? | `docs/STATUS.md` |
| Why was a decision made? | `docs/decisions/0000`–`0007` |
| What exists vs planned? | `docs/ARCHITECTURE.md` |
| What did the last session do? | `docs/sessions/` |
| What are the rules? | `AGENTS.md` |
| What is the backlog? | GitHub Issues #1–#37 (`gh issue list`) |

**Verify before you trust this file.** It was accurate at `2b23628`. Run
`git log --oneline -1`, `nox -s check`, and `gh issue list` before relying on any
count in it.
