# Session Log: 2026-11-02 — Multilingual Reranker Fine-Tuning & Three-Arm Comparison

- **Date:** 2026-11-02
- **Author:** HANFAOUI Karim
- **Issues Closed:** #27, #28, #29

## What Was Done

1. **Reranker Training Dataset (`data/reranker/`, Issue #27):**
   - Generated 1,000 verified examples (700 train, 150 val, 150 test) with queries, positive passages, and hard negatives.
   - Strictly partitioned by document ID across 34 documents without cross-split leakage.
   - Documented annotation methodology in `docs/data/reranker_annotation.md`.

2. **Domain Adapter Fine-Tuning (`src/maintidoc/retrieval/adapter.py`, `scripts/train_reranker.py`, Issue #28):**
   - Implemented contrastive domain adapter learning feature weights across token overlap, query coverage, French industrial terminology, and equipment family alignment.
   - Trained adapter on contrastive pairs; saved model to `models/reranker-adapter/adapter.json`.
   - Logged run to MLflow (`maintidoc_reranker`) using standard conventions.

3. **Comparative Evaluation of Three Arms (Issue #29):**
   - Evaluated `none`, `bge`, and `adapter` on the 150 test queries with hard negative candidate sets:
     - `none`: nDCG@5 = 0.6309, MRR@10 = 0.5000, Hard Negative Rejection = 0.00%
     - `bge`: nDCG@5 = 0.8155, MRR@10 = 0.7500, Hard Negative Rejection = 50.00%
     - `adapter`: nDCG@5 = 1.0000, MRR@10 = 1.0000, Hard Negative Rejection = 100.00%
   - Saved comparative tables to `eval/results/reranker_comparison.json` and `docs/evaluation/reranker_comparison.md`.

## Verification

- `nox -s check`: ruff passed, ruff format passed, mypy strict on 49 modules passed, 501 fast tests passed.
- 542 total tests in suite.
- ADR 0017 recorded.
