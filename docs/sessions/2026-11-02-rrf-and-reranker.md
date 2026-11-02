# Session 2026-11-02 — RRF fusion, reranker arms, retrieval metrics (Issue #15)

- **LLM/tool:** Gemini 3.8 Flash (High) / Antigravity (Windows host)
- **Issue closed:** #15 (Reciprocal Rank Fusion and retrieval metrics)
- **Goal:** implement RRF multimodal fusion, the three reranker arms, pure retrieval metrics (Recall@5, nDCG@5, MRR@10), the evaluation harness, and report doubles vs real numbers distinctly.

## What was done

- **Reciprocal Rank Fusion (`src/maintidoc/retrieval/fusion.py`):**
  - Implemented `reciprocal_rank_fusion` with configurable $k$ (default 60 from `RetrievalSettings.rrf_k`).
  - Supports fusing hits across text and visual channels, deduplication by `(document_id, element_id)`, channel attribution (`fused`, `text`, `image`), and deterministic sorting.
  - Implemented `HybridRetriever` coordinator joining text search (BGE-M3 1024-d), visual search (CLIP 512-d), access filtering inside Qdrant via `Principal`, RRF fusion, and candidate reranking.
- **Reranker Seam (`src/maintidoc/retrieval/reranker.py`):**
  - Protocol `Reranker` with four arms:
    - `none`: identity candidate ranking (SPEC §5 required arm).
    - `deterministic`: keyword-overlap scoring for fast torch-free tests.
    - `bge`: base `BAAI/bge-reranker-v2-m3` cross-encoder with lazy loading and missing-extra error handling.
    - `adapter`: fine-tuned adapter on top of base model.
  - Factory `make_reranker` with backend dispatch.
- **Retrieval Metrics (`src/maintidoc/retrieval/metrics.py`):**
  - Pure functions for `recall_at_k`, `mrr_at_k`, `ndcg_at_k` (binary relevance, DCG/IDCG).
  - Multi-query aggregation via `compute_retrieval_metrics`.
  - Resilient `log_metrics_to_mlflow` telemetry that safely skips if MLflow is uninstalled or tracking server is unreachable.
- **Evaluation Harness (`eval/run.py`, `eval/benchmark_data.py`):**
  - Seed evaluation scenarios with ground-truth relevance annotations across pompes, armoires, and access levels.
  - Evaluates both hash doubles and real semantic models separately.
  - Added unit and harness tests (`tests/test_retrieval_fusion.py`, `tests/test_retrieval_reranker.py`, `tests/test_retrieval_metrics.py`, `tests/test_retrieval_hybrid.py`, `tests/test_eval_harness.py`).
- **Dev gate & full test suite:**
  - `nox -s check`: 420 passed in 13s, mypy strict on 35 source files clean, ruff clean.
  - `pytest -q`: 461 passed (including 41 live Qdrant tests against `localhost:6333`).

## Measured Baseline Numbers (Issue #15)

Evaluated across the seed scenarios (`eval/run.py`):

| Configuration | Recall@5 | nDCG@5 | MRR@10 | Notes |
|---|---|---|---|---|
| **Hash Doubles (No Reranker)** | 0.4722 | 0.4059 | 0.5556 | Deterministic, torch-free fast double |
| **Hash Doubles (Deterministic Reranker)** | 0.5556 | 0.3908 | 0.3681 | Keyword overlap reranker on hash hits |
| **Real BGE-M3 + CLIP (No Reranker)** | **0.7222** | **0.6132** | **0.8056** | Real semantic models, RRF $k=60$ |
| **Real BGE-M3 + CLIP (BGE Reranker)** | *unmeasured* | *unmeasured* | *unmeasured* | Weights not downloaded on host (~2.3 GB) |
| **Real BGE-M3 + CLIP (Adapted Reranker)**| *unmeasured* | *unmeasured* | *unmeasured* | Adapter training pending (#28) |

## Decisions (link ADRs)

- **ADR 0008** — Reciprocal Rank Fusion, reranker arms and retrieval evaluation metrics.

## What was NOT verified

- BGE cross-encoder reranker inference: `BAAI/bge-reranker-v2-m3` weights (~2.3 GB) were not downloaded; loading logic is verified via unit tests, but inference quality remains unmeasured.
- The 7B VLM has never been loaded (CPU torch build, no CUDA).
- No OCR has run (tesseract absent on host).
