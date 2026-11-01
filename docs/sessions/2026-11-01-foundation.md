# Session 2026-11-01 — foundation: tooling, contracts, security, prompts, device ladder

- **LLM/tool:** Qwen Code (Windows host)
- **Issue(s):** pre-backlog. `scripts/create_backlog.sh` has not been run (no `gh` CLI), so this covers the first three planned issues: scaffold/tooling, Pydantic output schema + citation model, MCP signatures + approval-token contract.
- **Goal:** stand up a verified foundation for the whole project on a Windows laptop, without pretending anything works that does not.

## What was done

- Added the dev gate: `noxfile.py` as the single source of truth, `Makefile` as a
  thin wrapper (no GNU make on the Windows host), ruff + mypy strict + pytest +
  pre-commit with gitleaks, and a GitHub Actions workflow.
- Implemented the SPEC §7 output contract with cross-field status invariants and
  `extra="forbid"`, which makes NEVER #6 structural rather than review-dependent.
- Implemented domain models, config, safe logging with secret redaction, access
  control (deny by default) and content-bound approval tokens.
- Added typed MCP tool contracts (SPEC §6) — signatures only.
- Added the device probe and model fallback ladder with an audit trail.
- Wrote the four SPEC §7 prompts as versioned YAML and a loader that refuses a
  prompt which declares untrusted input without delimiting it.
- Added the VLM provider seam with a deterministic, citation-safe stub, the
  source render/parse round-trip, and the lenient model-draft layer.
- Added the object-store interface with a traversal-proof filesystem backend.
- Added the `doctor` CLI.
- Wrote the seeded synthetic corpus generator with OCR, injection and
  restricted-access cases.
- Committed in six Conventional Commits and pushed to `main`.

## Decisions (link ADRs)

- **ADR 0002** — nox is the task runner; make delegates to it. Forced by the
  absence of GNU make on Windows.
- **ADR 0003** — probe CUDA→MPS→CPU and step the model down loudly. Forced by
  8 GB VRAM versus SPEC §11's 24 GB assumption, and required so NEVER #8 is not
  broken silently.
- **ADR 0004** — torch stays optional behind a provider seam; a deterministic
  stub serves tests and the portable CPU deployment.
- **ADR 0005** — jinja2 template format, because prompts contain literal JSON
  braces that f-string formatting cannot express.
- **ADR 0006** — **proposed.** MinIO is unpullable; the object-store backend is
  deferred at the user's request pending a working image reference.

## Commands run + results

```
docker pull minio/minio:latest
  -> pull access denied, repository does not exist or may require 'docker login'
docker manifest inspect quay.io/minio/minio:latest   -> no such manifest
docker manifest inspect bitnami/minio:latest         -> no such manifest
curl https://dl.min.io/server/minio/release/linux-amd64/ -> HTTP 410 Gone
docker manifest inspect chrislusf/seaweedfs:latest   -> pullable
docker manifest inspect qdrant/qdrant:v1.20.0        -> pullable
docker manifest inspect postgres:16.15-alpine        -> pullable

.venv\Scripts\nox.exe -s check
  -> ruff check: All checks passed
  -> ruff format --check: 47 files already formatted
  -> mypy src: Success: no issues found in 25 source files
  -> pytest: 259 passed in 2.29s
  -> Session check was successful

.venv\Scripts\maintidoc.exe doctor --corpus-dir .tmp-corpus
  -> 13 checks: 0 fail, 8 warn
     (warns: model stepped down to CPU, tesseract absent, qdrant/postgres unreachable)

python scripts/generate_corpus.py --count 8 --corpus-dir .tmp-corpus
  -> 8 documents, 2 families, 1 scanned (0 text chars / 1 image), 1 injection doc

git push origin main -> 6ad0671..bffbe9c
```

## Not verified / risks

- **Qwen2.5-VL 7B has never been loaded.** No `vlm/qwen.py` exists. All VRAM
  numbers are estimates from nominal parameter counts, not measurements; SPEC
  §11's benchmark is outstanding.
- **No OCR has run.** tesseract is absent from the host; the scanned-corpus
  claim is that the text layer is empty, not that OCR reads it.
- No integration test against Qdrant/PostgreSQL; no image built; CI never ran.
- `nox -s ingest|eval|bench_vlm` reference modules that do not exist yet.
- `.tmp-corpus/` was used as scratch before it was known to be pre-existing. All
  current contents are from this session's run; it is now git-ignored.
- Free VRAM measured between 3.8 and 5.5 GB during the session, so the ladder's
  decision is sensitive to whatever else is running — expected, and precisely why
  the decision is recorded in provenance.

## Next session should

1. Read `docs/STATUS.md`, then ask the human for the MinIO image reference
   (ADR 0006) if it is still outstanding.
2. Implement `ingestion/pdf.py` and `ingestion/ocr.py` against the verified
   PyMuPDF 2.0 API, producing `Element`s with SPEC §4.1 metadata.
3. Add embeddings behind seams and index into Qdrant with the access filter.
4. Do not claim Docker works until `docker compose up` has actually been run.