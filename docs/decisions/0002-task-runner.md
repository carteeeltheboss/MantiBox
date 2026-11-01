# 0002 — nox as the task runner, with `make` as a wrapper

- **Date:** 2026-11-01
- **Status:** accepted
- **Issue/PR:** scaffold tooling (W1)

## Context
AGENTS.md names `make check` as the gate before every commit. GNU make is not
installed on the Windows development host, and installing it system-wide would
put a build tool on the user's machine outside the project. The environment also
lacks `uv`, `just`, `tesseract`, `pre-commit` and `gh`.

Two further constraints: the gate must run identically in CI (Linux) and on the
developer machine (Windows), and duplicating the command list in two files
guarantees they drift.

## Options considered
1. **Install GNU make via winget/choco** — keeps the Makefile literally, but
   modifies the system outside the project, and GNU Make on Windows is fragile
   around the shell features this Makefile would need.
2. **Run everything in a dev container** — most faithful to Linux deployment,
   but slower iteration and the developer would live inside a container.
3. **Python task runner as the single source of truth** — `noxfile.py` holds the
   real logic; the Makefile delegates. Works natively on Windows, and the same
   session runs in CI and containers.

## Decision
Option 3. `noxfile.py` is the source of truth for `lint`, `format`, `types`,
`test`, `check`, `test_all`, `lock`, `corpus`, `ingest`, `eval`, `bench_vlm`,
`doctor` and the compose helpers. `Makefile` is a thin wrapper so `make check`
still works wherever make exists.

## Consequences
- One command list, two entry points. Adding a step means editing `noxfile.py`
  only.
- Sessions use `reuse_existing_virtualenvs`, so repeated runs are fast after the
  first.
- The heavy ML extra (`torch`, `transformers`, `autoawq`) is deliberately **not**
  in the `dev` extra, so the gate stays installable on a laptop. See ADR 0004.
- Revisit if a contributor has make but no Python packaging workflow, or if CI
  moves to a container-only pipeline.