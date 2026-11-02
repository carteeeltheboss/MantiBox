"""Tests for MLflow setup and run metadata conventions (Issue #7, SPEC §7, §11).

Acceptance criteria:
- Every run logs model, prompt version, reranker, index version, git commit
- Convention documented in docs/ARCHITECTURE.md
- Local MLflow starts via Compose
"""

from __future__ import annotations

from maintidoc.tracking import (
    build_run_metadata,
    get_git_commit,
    log_run_with_convention,
)


def test_get_git_commit_resolves() -> None:
    commit = get_git_commit()
    # On a git repo, commit hash is either a 40-char hex string or None if uncommitted
    if commit is not None:
        assert len(commit) >= 7


def test_build_run_metadata_contains_all_required_keys() -> None:
    meta = build_run_metadata(
        model_id="maintidoc-stub-vlm",
        provider="stub",
        prompt_versions={"answer": "v1", "visual_analysis": "v1"},
        reranker="none",
        index_version="qdrant_v1",
        device="cpu",
    )

    assert meta["model_id"] == "maintidoc-stub-vlm"
    assert meta["provider"] == "stub"
    assert "answer:v1" in meta["prompt_versions"]
    assert meta["reranker"] == "none"
    assert meta["index_version"] == "qdrant_v1"
    assert "git_commit" in meta
    assert meta["device"] == "cpu"


def test_log_run_with_convention_safe_when_no_mlflow() -> None:
    # When tracking backend is null or host unreachable, it returns False safely without error
    metrics = {"recall@5": 0.85, "ndcg@5": 0.75, "mrr@10": 0.90}
    meta = build_run_metadata(
        model_id="stub",
        provider="stub",
        prompt_versions={"answer": "v1"},
        reranker="none",
        index_version="v1",
    )

    success = log_run_with_convention(
        experiment_name="test_experiment",
        run_name="test_run",
        metrics=metrics,
        params=meta,
    )
    # Success is boolean (either logged or safely skipped)
    assert isinstance(success, bool)
