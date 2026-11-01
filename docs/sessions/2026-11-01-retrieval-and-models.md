# Session 2026-11-01 (continued) — backlog, ingestion, retrieval core, real models

- **LLM/tool:** Qwen Code (Windows host)
- **Issues closed:** #11, #12, #13, #14 (and #4 re-verified with fixes), plus 7 issues
  closed as already-built earlier in the day
- **Goal:** turn the verified foundation into a working retrieval core, and stop
  guessing about the models.

## What was done

- **Backlog created on GitHub** (#1–#37, six milestones, seven labels) via
  `scripts/create_backlog.sh`, after adding a `DRY_RUN` mode because 37 live issues
  deserved review first.
- **Corpus generator fixed** — three silent defects: `insert_textbox` writes
  *nothing* when text overflows (all 32 tables had zero cell text), base-14 fonts
  turned em dashes into `?`, and a random PDF `/ID` broke byte-reproducibility.
- **Ingestion** (#11) with the SPEC §4.1 metadata on every element. Three probes
  shaped it: `find_tables()` detects nothing in vector tables, figures are drawings
  rather than images, and a page with no text layer is a scan.
- **Text embeddings + Qdrant index** (#12) with the access filter inside the query,
  refusal instead of an empty filter for a group-less principal, and UUIDv5 point
  ids for idempotent re-ingestion.
- **Two-tower CLIP image embedder + 512-d image index** (#13), with the 512-vs-1024
  width hazard pinned by tests.
- **Document/file access on open** (#14): deny before fetch, unknown-is-denied, and
  image keys bound to their document.
- **Weights prefetched** (17.58 GB) with `hf download --local-dir`, after the cache
  path deadlocked on Windows, then **loaded and measured** once torch was installed.
- **Four bugs found only by executing**: unusable CUDA device being selected with a
  CPU-only torch, transformers 5's `pooler_output`, untyped load failures, and two
  tests that had become network downloads.

## Decisions (link ADRs)

- **ADR 0007** — two licence-clean CLIP towers; `jina-clip-v2` rejected as
  CC-BY-NC-4.0. Also records the measured text→image weakness.
- **ADR 0003** — two amendments: substitution is off by default (owner rule), and
  `usable` distinguishes hardware from a runtime that can drive it.
- **ADR 0006** — SeaweedFS accepted (owner decision) after MinIO proved unpullable.

## Commands run + results

```
nox -s check                -> ruff + format clean, mypy 32 modules, 372 fast tests
pytest -q (dev venv)        -> 413 passed (41 integration vs live qdrant:v1.20.0)
hf download --local-dir     -> 17.58 GB across four repos, all verified structurally
real model load             -> BGE-M3 1024-d, related 0.8492 vs unrelated 0.4592
                               CLIP 512-d, same figure 1.0000 vs other 0.8160
                               SURPRISE: text->image wrong panel won (0.2991 vs 0.2691)
```

## Not verified / risks

- **The 7B has never been loaded.** Blocked on a CUDA torch build (the installed
  one is CPU-only) and `autoawq`, not merely on VRAM.
- **OCR has never run** — no tesseract binary, no image built.
- No Dockerfile or Compose file exists.
- Pipeline integration tests use the deterministic doubles for speed, so retrieval
  quality end-to-end is unmeasured; only the model-level checks above are real.
- Text→image retrieval is measurably weak on this corpus and must be reported
  separately in the eval.

## Next session should

1. Read `docs/HANDOFF.md`, then `docs/STATUS.md`.
2. Do **#15** (RRF + three reranker arms), then **#9** (vertical slice), then
   **#10** (Compose).
3. Ask the owner about the CUDA torch build before **#16**.
