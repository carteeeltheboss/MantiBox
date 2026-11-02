# 0008 — Reciprocal Rank Fusion, reranker arms and retrieval evaluation metrics

- **Date:** 2026-11-02
- **Status:** accepted
- **Issue/PR:** #15 (RRF fusion + reranker arms + retrieval metrics)

## Context
SPEC §4.2 specifies multimodal search over the authorized corpus:
1. Access filtering applied at query time inside Qdrant;
2. Text search (BGE-M3, 1024-d) and visual search (CLIP two-tower, 512-d);
3. Fusion via Reciprocal Rank Fusion (RRF);
4. Reranking candidates with a cross-encoder reranker;
5. Delivery of top cited passages and figures to the VLM.

SPEC §5 requires an explicit comparison of three reranker arms:
- sans reranker ("none")
- avec reranker d'origine ("bge" -> BAAI/bge-reranker-v2-m3)
- avec reranker adapté ("adapter")
plus SPEC §10 requires computing Recall@5, nDCG@5, and MRR@10.

Furthermore, the fast gate (`nox -s check`) must remain torch-free and run in CI or a lightweight environment without downloading gigabytes of weights.

## Decisions

1. **RRF formula and parameterization (`retrieval/fusion.py`):**
   - For an element $d$ retrieved in channel $c \in \{\text{text}, \text{image}\}$ at 1-based rank $r_c(d)$:
     $$RRF(d) = \sum_{c \in C} \frac{1}{k + r_c(d)}$$
   - Constant $k$ defaults to `60` from `RetrievalSettings.rrf_k` (standard IR benchmark value).
   - Elements are deduplicated by `(document_id, element_id)`.
   - Channel attribution: `channel="fused"` if retrieved by both channels; otherwise `"text"` or `"image"`.
   - Deterministic tie-breaking by `(document_id, element_id)`.

2. **Reranker seam (`retrieval/reranker.py`):**
   - Implemented via the `Reranker` protocol exposing `model_id`, `arm`, and `rerank(query, elements, *, top_n=...)`.
   - Four concrete arms:
     - `none` (`NoReranker`): identity reranker preserving candidate order (SPEC §5 required arm).
     - `deterministic` (`DeterministicReranker`): fast keyword-overlap scoring for unit tests and torch-free gate.
     - `bge` (`BgeReranker`): base `BAAI/bge-reranker-v2-m3` cross-encoder.
     - `adapter` (`BgeReranker`): `BAAI/bge-reranker-v2-m3` + fine-tuned PEFT adapter.
   - Factory `make_reranker` dispatches across arms.

3. **Weight status of BAAI/bge-reranker-v2-m3:**
   - Per project instructions, downloading ~2.3 GB for the reranker requires explicit human confirmation. The arm is fully implemented behind the interface, covered by lazy-load tests, and marked unmeasured in the benchmark until weights are on disk.

4. **Metrics and evaluation harness (`retrieval/metrics.py`, `eval/run.py`):**
   - Pure verifiable functions for `recall_at_k`, `mrr_at_k`, `ndcg_at_k`.
   - Seed benchmark scenarios (`eval/benchmark_data.py`) with ground truth targets across pump and electrical cabinet families.
   - Strict separation of deterministic hashing double numbers and real semantic model numbers — never blended.
   - Telemetry: safe MLflow logging if tracking server is active.

## Measured Baseline Numbers (Issue #15)

Evaluated across the seed scenarios (`eval/run.py`):

| Configuration | Recall@5 | nDCG@5 | MRR@10 | Notes |
|---|---|---|---|---|
| **Hash Doubles (No Reranker)** | 0.4722 | 0.4059 | 0.5556 | Deterministic, torch-free fast double |
| **Hash Doubles (Deterministic Reranker)** | 0.5556 | 0.3908 | 0.3681 | Keyword overlap reranker on hash hits |
| **Real BGE-M3 + CLIP (No Reranker)** | **0.7222** | **0.6132** | **0.8056** | Real semantic models, RRF $k=60$ |
| **Real BGE-M3 + CLIP (BGE Reranker)** | *unmeasured* | *unmeasured* | *unmeasured* | Weights not downloaded on host |
| **Real BGE-M3 + CLIP (Adapted Reranker)**| *unmeasured* | *unmeasured* | *unmeasured* | Adapter training pending (#28) |

## Consequences
- Fast gate (`nox -s check`) remains green in seconds without requiring torch.
- Text (1024-d) and image (512-d) vectors are never mixed; each is queried in its respective collection and joined at rank level via RRF.
- Downstream vertical slice (#9) can cleanly consume `HybridRetriever`.
