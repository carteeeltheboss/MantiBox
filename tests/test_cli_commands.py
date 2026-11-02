"""Tests for CLI bench-vlm and ingest commands (SPEC §4, §11)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from maintidoc.cli import app

runner = CliRunner()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "maintidoc" in result.stdout


def test_cli_doctor() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code in (0, 1)
    assert "readiness report" in result.stdout


def test_cli_bench_vlm_text_output() -> None:
    result = runner.invoke(app, ["bench-vlm"])
    assert result.exit_code == 0
    assert "Qwen2.5-VL-7B Feasibility Benchmark (SPEC §11)" in result.stdout
    assert "AWQ 4-bit" in result.stdout
    assert "VALIDATED" in result.stdout
    assert "Target VRAM: 24.0 GB" in result.stdout


def test_cli_bench_vlm_json_output() -> None:
    result = runner.invoke(app, ["bench-vlm", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model_id"] == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    assert data["spec_hypothesis_validated"] is True
    assert data["target_vram_gb"] == 24.0
    assert "profiles" in data
    assert "4096" in data["profiles"]
    assert "32768" in data["profiles"]


def test_cli_ingest_help() -> None:
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--corpus-dir" in result.stdout
    assert "--batch-size" in result.stdout
