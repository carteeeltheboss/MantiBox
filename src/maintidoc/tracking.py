"""MLflow run metadata convention and tracking helpers (Issue #7, SPEC §7, §11).

Standardizes logging parameters across evaluation and training runs:
- Model identifier and provider
- Prompt versions map
- Reranker configuration
- Index version
- Git commit hash (for exact reproducibility)
- Execution device and environment
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from typing import Any

from maintidoc.config import get_settings
from maintidoc.observability import get_logger
from maintidoc.retrieval.index import INDEX_VERSION

logger = get_logger(__name__)


def get_git_commit() -> str | None:
    """Retrieve current Git commit SHA (from GIT_COMMIT env var or git command)."""
    env_commit = os.environ.get("GIT_COMMIT") or os.environ.get("GITHUB_SHA")
    if env_commit:
        return env_commit.strip()

    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            check=False,
            timeout=2.0,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:  # noqa: BLE001
        logger.debug("Could not resolve git commit via subprocess")

    return None


def build_run_metadata(
    *,
    model_id: str,
    provider: str,
    prompt_versions: Mapping[str, str] | None = None,
    reranker: str | None = None,
    index_version: str = INDEX_VERSION,
    device: str = "cpu",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the standardized dictionary of run metadata required by SPEC §7 & §11."""
    settings = get_settings()

    prompt_str = (
        ",".join(f"{k}:{v}" for k, v in sorted(prompt_versions.items()))
        if prompt_versions
        else "none"
    )

    metadata: dict[str, Any] = {
        "model_id": model_id,
        "provider": provider,
        "prompt_versions": prompt_str,
        "reranker": reranker or "none",
        "index_version": index_version,
        "git_commit": get_git_commit() or "uncommitted",
        "device": device,
        "env": settings.app.env,
    }

    if extra:
        metadata.update(extra)

    return metadata


def log_run_with_convention(
    *,
    experiment_name: str,
    run_name: str,
    metrics: Mapping[str, float],
    params: Mapping[str, Any],
    tags: Mapping[str, str] | None = None,
) -> bool:
    """Log an evaluation or training run to MLflow adhering to the MaintiDoc convention.

    Never raises; safely falls back if MLflow is not running or unreachable.
    """
    settings = get_settings()
    if settings.inference.tracking_backend == "null":
        logger.debug("Tracking backend is 'null'; skipping MLflow run logging")
        return False

    try:
        import mlflow
    except ImportError:
        logger.debug("MLflow not installed; skipping run logging")
        return False

    try:
        mlflow.set_tracking_uri(settings.inference.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name):
            clean_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(clean_params)
            mlflow.log_metrics(dict(metrics))
            if tags:
                mlflow.set_tags(dict(tags))
        logger.info(
            "MLflow run '%s' logged to experiment '%s'",
            run_name,
            experiment_name,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to log run to MLflow tracking server: %s",
            exc,
        )
        return False


__all__ = [
    "build_run_metadata",
    "get_git_commit",
    "log_run_with_convention",
]
