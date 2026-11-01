# Thin wrapper over noxfile.py — the real source of truth for every command.
#
# GNU make is absent on the Windows dev host, so `make` is for CI, containers and
# WSL; on Windows run the equivalent `nox -s <session>`. Both resolve to the same
# code. See ADR 0002.
#
# `make help` lists the targets.

NOX ?= nox
COMPOSE ?= docker compose
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install lint format types test check test-all lock lock-ml corpus ingest eval bench-vlm doctor up down logs ps clean

help:
	@echo "MaintiDoc targets (delegating to nox):"
	@echo "  install     create dev env + pre-commit hooks"
	@echo "  lint        ruff check + format --check"
	@echo "  format      apply ruff fixes and formatting"
	@echo "  types       mypy --strict on src/"
	@echo "  test        unit tests (no services, no GPU, no torch)"
	@echo "  check       lint + types + test  <- gate before every commit"
	@echo "  test-all    every test, incl. integration (run 'make up' first)"
	@echo "  lock        regenerate requirements.lock.txt"
	@echo "  lock-ml     regenerate requirements.ml.lock.txt inside linux/amd64"
	@echo "  corpus      generate the synthetic document corpus"
	@echo "  ingest      index the corpus into Qdrant/Postgres/MinIO"
	@echo "  eval        run the evaluation harness"
	@echo "  bench-vlm   benchmark the VLM on the detected device (SPEC §11)"
	@echo "  doctor      report device/VRAM/service readiness"
	@echo "  up          docker compose up -d --build   (add PROFILE=gpu)"
	@echo "  down        docker compose down"
	@echo "  logs        tail compose logs"
	@echo "  ps          compose ps"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	-pre-commit install
	@echo "Now: make check"

lint:
	$(NOX) -s lint

format:
	$(NOX) -s format

types:
	$(NOX) -s types

test:
	$(NOX) -s test

check:
	$(NOX) -s check

test-all:
	$(NOX) -s test_all

lock:
	$(NOX) -s lock

lock-ml:
	$(NOX) -s lock_ml

corpus:
	$(NOX) -s corpus

ingest:
	$(NOX) -s ingest

eval:
	$(NOX) -s eval

bench-vlm:
	$(NOX) -s bench_vlm

doctor:
	$(NOX) -s doctor

up:
	$(NOX) -s compose_up -- $(if $(PROFILE),--profile $(PROFILE),)

down:
	$(NOX) -s compose_down

logs:
	$(NOX) -s compose_logs

ps:
	$(COMPOSE) ps

clean:
	$(NOX) -s compose_down -- -v
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in map(pathlib.Path, ['.nox','.mypy_cache','.ruff_cache','.pytest_cache','dist','build'])]"
