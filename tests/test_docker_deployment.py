"""Tests for Docker Compose demo deployment and rollback configuration (SPEC §11)."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_docker_compose_config_validity() -> None:
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml must exist"

    content = compose_path.read_text(encoding="utf-8")
    data = yaml.safe_load(content)

    assert "services" in data, "compose file must define services"
    services = data["services"]

    # Verify required SPEC §11 infrastructure services exist
    required_services = {"qdrant", "postgres", "seaweedfs", "mlflow", "api", "ui"}
    assert required_services.issubset(set(services.keys()))

    # Verify images are pinned (no :latest)
    for s_name in ("qdrant", "postgres", "seaweedfs", "mlflow"):
        s_cfg = services[s_name]
        assert "image" in s_cfg
        image = s_cfg["image"]
        assert ":" in image, f"Service {s_name} image must be pinned with a version tag"
        assert not image.endswith(":latest"), f"Service {s_name} must not use :latest tag"

    # Verify healthchecks are configured on data stores
    assert "healthcheck" in services["qdrant"]
    assert "healthcheck" in services["postgres"]
    assert "healthcheck" in services["seaweedfs"]


def test_dockerfile_multi_stage_structure() -> None:
    dockerfile_path = Path("Dockerfile")
    assert dockerfile_path.exists(), "Dockerfile must exist"

    content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in content
    assert "tesseract-ocr" in content
    assert "tesseract-ocr-fra" in content
    assert "USER" in content  # Non-root user


def test_rollback_runbook_exists() -> None:
    runbook_path = Path("docs/deployment/runbook.md")
    assert runbook_path.exists(), "Deployment and rollback runbook must exist"
    text = runbook_path.read_text(encoding="utf-8")
    assert "docker compose" in text
    assert "Rollback" in text
    assert "SeaweedFS" in text
