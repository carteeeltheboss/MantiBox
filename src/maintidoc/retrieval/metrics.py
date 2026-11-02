"""Retrieval evaluation metrics: Recall@K, MRR@K, nDCG@K (SPEC §5, §10).

Pure, verifiable implementations of the three metrics SPEC §10 requires:
- Recall@5: fraction of relevant items retrieved in top 5.
- MRR@10: reciprocal rank of the first relevant item retrieved in top 10.
- nDCG@5: normalized discounted cumulative gain at rank 5 with binary relevance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from maintidoc.observability import get_logger

logger = get_logger(__name__)


def recall_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int = 5,
) -> float:
    """Fraction of relevant items retrieved in the top K."""
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def mrr_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int = 10,
) -> float:
    """Reciprocal rank of the first relevant item in the top K, or 0.0."""
    if not relevant or k <= 0:
        return 0.0
    for rank_0, item in enumerate(retrieved[:k]):
        if item in relevant:
            return 1.0 / (rank_0 + 1)
    return 0.0


def dcg_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int = 5,
) -> float:
    """Discounted Cumulative Gain at rank K with binary relevance."""
    if not relevant or k <= 0:
        return 0.0
    dcg = 0.0
    for rank_0, item in enumerate(retrieved[:k]):
        if item in relevant:
            # For rank i (1-based), discount is log2(i + 1)
            dcg += 1.0 / math.log2(rank_0 + 2)
    return dcg


def idcg_at_k(relevant_count: int, k: int = 5) -> float:
    """Ideal Discounted Cumulative Gain at rank K."""
    if relevant_count <= 0 or k <= 0:
        return 0.0
    ideal_hits = min(relevant_count, k)
    return sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int = 5,
) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    idcg = idcg_at_k(len(relevant), k=k)
    if idcg <= 0.0:
        return 0.0
    dcg = dcg_at_k(retrieved, relevant, k=k)
    return dcg / idcg


def compute_retrieval_metrics(
    queries_results: Sequence[tuple[Sequence[str], set[str]]],
    *,
    k_recall: int = 5,
    k_ndcg: int = 5,
    k_mrr: int = 10,
) -> dict[str, float]:
    """Compute mean Recall@K, nDCG@K, MRR@K across a sequence of query results.

    Each item in `queries_results` is a pair: `(retrieved_ids, relevant_ids)`.
    """
    if not queries_results:
        return {
            f"recall@{k_recall}": 0.0,
            f"ndcg@{k_ndcg}": 0.0,
            f"mrr@{k_mrr}": 0.0,
            "query_count": 0.0,
        }

    total_recall = 0.0
    total_ndcg = 0.0
    total_mrr = 0.0

    for retrieved, relevant in queries_results:
        total_recall += recall_at_k(retrieved, relevant, k=k_recall)
        total_ndcg += ndcg_at_k(retrieved, relevant, k=k_ndcg)
        total_mrr += mrr_at_k(retrieved, relevant, k=k_mrr)

    n = len(queries_results)
    return {
        f"recall@{k_recall}": total_recall / n,
        f"ndcg@{k_ndcg}": total_ndcg / n,
        f"mrr@{k_mrr}": total_mrr / n,
        "query_count": float(n),
    }


def log_metrics_to_mlflow(
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    *,
    run_name: str | None = None,
    experiment_name: str = "maintidoc_retrieval",
) -> bool:
    """Log metrics and parameters to MLflow if available and reachable.

    Returns True if logged successfully, False if MLflow was skipped or failed.
    Never raises an exception (safe telemetry).
    """
    try:
        import mlflow
    except ImportError:
        logger.debug("MLflow is not installed; skipping run logging")
        return False

    try:
        from maintidoc.config import get_settings

        settings = get_settings()
        if settings.inference.tracking_backend == "null":
            logger.debug("Tracking backend is 'null'; skipping MLflow")
            return False

        mlflow.set_tracking_uri(settings.inference.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            if params:
                # MLflow requires string/number/boolean param values
                clean_params = {k: str(v) for k, v in params.items()}
                mlflow.log_params(clean_params)
            mlflow.log_metrics(metrics)
        logger.info("Logged retrieval metrics to MLflow successfully")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to log metrics to MLflow (backend reachability or config error)",
            extra={"error": str(exc)},
        )
        return False


__all__ = [
    "compute_retrieval_metrics",
    "dcg_at_k",
    "idcg_at_k",
    "log_metrics_to_mlflow",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
]
