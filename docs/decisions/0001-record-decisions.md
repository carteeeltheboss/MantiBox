# 0001 — Record decisions as ADRs in the repo

- **Date:** (today)
- **Status:** accepted

## Context
Work will be done by different LLMs across sessions. Chat history is not shared between them.

## Decision
The repo is the only memory. Every non-trivial decision gets a short ADR in `docs/decisions/`. `docs/STATUS.md` is updated at the end of every session.

## Consequences
Switching LLM = starting a new session with the session-start prompt (docs/PROMPTS.md). Anything not in the repo is considered lost.
