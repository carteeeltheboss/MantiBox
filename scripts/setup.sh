#!/usr/bin/env bash
# One-time repo bootstrap. Run from the repo root.
set -euo pipefail

[ -d .git ] || git init -b main

# One rules file, many tool-specific names
for f in CLAUDE.md GEMINI.md .cursorrules; do
  [ -e "$f" ] || ln -s AGENTS.md "$f"
done

if command -v pre-commit >/dev/null; then
  pre-commit autoupdate   # replaces placeholder revs with real ones
  pre-commit install
else
  echo "Install pre-commit (pip install pre-commit), then re-run this script."
fi

cat <<MSG
Next:
  1. gh repo create maintidoc --private --source=. --push
  2. bash scripts/create_backlog.sh
  3. Create a GitHub Project (board), add all issues, group by Milestone
  4. Start a session with prompt #1 from docs/PROMPTS.md
MSG
