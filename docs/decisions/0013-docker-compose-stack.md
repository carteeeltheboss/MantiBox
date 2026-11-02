# 0013. Docker Compose deployment stack

Date: 2026-11-02  
Status: accepted  
Issues: #10, #34

## Context

SPEC §3 and §11 specify deployment using Docker Compose:
- Vector store: Qdrant
- Metadata & relational database: PostgreSQL
- File/object store: SeaweedFS (per ADR 0006, replacing unreachable MinIO)
- Experiment tracking: MLflow
- Web backend: FastAPI application container with Tesseract OCR (fra) installed
- Frontend: Streamlit interactive user interface

Acceptance criteria (Issue #10):
- `make up` (and `nox -s compose_up`) starts all services.
- Images and versions are strictly pinned.
- Secrets are injected via environment variables, not committed to git.

## Decision

1. **Service Images & Versions:**
   - `qdrant/qdrant:v1.20.0`
   - `postgres:16-alpine`
   - `chrislusf/seaweedfs:3.79` (runs S3 server on 8333 per ADR 0006)
   - `ghcr.io/mlflow/mlflow:v2.19.0` (MLflow server on 5000)
   - `api`: Container built from `Dockerfile` with `python:3.12-slim`, system packages `tesseract-ocr`, `tesseract-ocr-fra`, `curl`, `libpq-dev`. Runs `uvicorn maintidoc.api.app:create_app_from_env`.
   - `ui`: Streamlit frontend on port 8501 talking to `http://api:8000`.

2. **Healthchecks & Start Ordering:**
   - Qdrant, Postgres, and SeaweedFS declare container healthchecks.
   - API depends on healthy storage and database services before accepting requests.
   - Secrets are configured with fallback development placeholders for local runs while reading `.env` when present.

## Consequences

- The entire MaintiDoc ecosystem can be booted with a single command (`docker compose up -d` / `make up`).
- Portable across Linux, macOS, and Windows.
- Prepares deployment for Issue #34 (demo rehearsal).
