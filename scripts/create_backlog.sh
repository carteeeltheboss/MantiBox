#!/usr/bin/env bash
# Creates labels, weekly milestones and the full issue backlog from SPEC §12/§13.
# Requires: `gh auth login`, run inside the repo after it has a GitHub remote.
# Then in GitHub: create a Project (board), "Add items" -> all issues, group by Milestone.
set -euo pipefail

repo=$(gh repo view --json nameWithOwner -q .nameWithOwner)

for l in rag mcp security eval infra data docs; do
  gh label create "$l" --force >/dev/null
done

for m in "W1 Cadrage" "W2 Ingestion et recherche" "W3 VLM et citations" \
         "W4 Agent et MCP" "W5 Reranker et evaluation" "W6 Deploiement et soutenance"; do
  gh api "repos/$repo/milestones" -f title="$m" >/dev/null || true
done

# issue "title" "milestone" "labels" "spec ref" "acceptance criteria separated by \n"
issue() {
  local ac
  ac=$(printf '%b' "$5" | sed 's/^/- [ ] /')
  gh issue create --title "$1" --milestone "$2" --label "$3" --body "$(cat <<BODY
**Spec:** $4

**Acceptance criteria**
$ac

**Definition of done**
- [ ] Tests written first and passing (\`make check\`)
- [ ] No secrets, no raw content in logs, no inline prompts
- [ ] docs/STATUS.md updated; ADR added if a decision was made
- [ ] PR links this issue
BODY
)"
}

# ---------- W1 Cadrage ----------
M="W1 Cadrage"
issue "Scaffold repo, tooling and CI" "$M" "infra" "§11" \
"pyproject, lockfile with pinned versions\nruff, mypy strict, pytest run via make check\npre-commit with gitleaks installed\nCI runs make check and secret scan on PRs"
issue "Pydantic output schema and citation model" "$M" "rag" "§7" \
"Schema covers status, answer, citations, uncertainties, clarification_question, proposed_action\nstatus enum includes answered / needs_clarification / abstained / needs_approval\nCitation requires document_id, page, element_id\nTests validate the SPEC §7 example and reject malformed outputs"
issue "MCP tool signatures and approval token contract" "$M" "mcp,security" "§6" \
"Typed signatures for get_equipment_history, get_open_tickets, create_ticket\nApproval token bound to user + exact ticket content hash; any edit invalidates it\nIdempotency key defined\nContract tests written (implementation can be stubbed)"
issue "Assemble corpus (30-50 docs, 2 equipment families) and license log" "$M" "data" "§2" \
"Documents synthetic or authorized, license/source recorded per document\nTwo equipment families identified\nPersonal data removed or anonymized\nAccess groups assigned per document\nCorpus manifest committed (files themselves in MinIO / not in git if large)"
issue "Threat model and risk register" "$M" "security,docs" "§9" \
"Assets, trust boundaries and attack paths listed\nEach SPEC §9 row mapped to a concrete test planned\nRisks table with owner and mitigation"
issue "Eval harness skeleton with 10 seed scenarios" "$M" "eval" "§10" \
"Scenario file format defined (question, optional image, expected status, expected citations)\nRunner executes scenarios and prints metrics\n10 seed scenarios across categories\nmake eval works (even against a stub system)"
issue "MLflow setup and run metadata convention" "$M" "infra" "§7, §11" \
"Every run logs model, prompt version, reranker, index version, git commit\nConvention documented in docs/ARCHITECTURE.md\nLocal MLflow starts via Compose"
issue "Write ARCHITECTURE.md and data-flow diagram of the plan" "$M" "docs" "§3, §13.1" \
"Diagram and component table\nTrust boundaries marked\nDeliverable 1 (cadrage et architecture) complete"

# ---------- W2 Ingestion et recherche ----------
M="W2 Ingestion et recherche"
issue "Vertical slice: question -> retrieve -> cited answer (ugly but end-to-end)" "$M" "rag" "§4.2" \
"Text-only question returns an answer with at least one real citation\nRuns through the real Pydantic schema\nOne eval scenario passes end-to-end\nKnown shortcuts listed in STATUS.md"
issue "Docker Compose: Qdrant, PostgreSQL, MinIO" "$M" "infra" "§3, §11" \
"make up starts all services\nImages and versions pinned\nSecrets injected via env, not committed"
issue "PDF ingestion: PyMuPDF + local OCR with element metadata" "$M" "rag,data" "§4.1" \
"Paragraphs, tables, pages, figures extracted\nEach element stores document id+version, page, element id, equipment type, access groups\nScanned docs handled by OCR\nOriginal images kept in MinIO"
issue "Text embeddings (BGE-M3) and Qdrant indexing" "$M" "rag" "§3, §4.2" \
"Elements indexed with metadata payload\nIndex version recorded\nRecall@5 measured on seed scenarios"
issue "Visual embeddings and image index" "$M" "rag" "§3, §4.2" \
"Multilingual CLIP-compatible encoder selected via ADR (license checked)\nImages/figures indexed with same metadata\nText-to-image and image-to-image retrieval tested"
issue "Access control: filter before retrieval and on document/image open" "$M" "security,rag" "§4.2, §9" \
"User/group filter applied in the vector query itself\nOpening a document or image re-checks permissions\nTests prove no cross-group leak (text and images)"
issue "Reciprocal Rank Fusion and retrieval metrics" "$M" "rag,eval" "§4.2, §10" \
"RRF fuses text and visual results\nRecall@5, nDCG@5, MRR@10 computed and logged to MLflow\nBaseline numbers recorded in STATUS.md"

# ---------- W3 VLM et citations ----------
M="W3 VLM et citations"
issue "Qwen2.5-VL inference and 24 GB feasibility benchmark" "$M" "infra,rag" "§3, §11" \
"Model revision pinned\nQuantization and image/context limits chosen\nMemory and latency measured on actual hardware\nResult recorded in an ADR (hypothesis validated or not)"
issue "Versioned prompts (4 YAML files) and loader" "$M" "rag" "§7" \
"answer_v1, visual_analysis_v1, evidence_check_v1, ticket_draft_v1 exist\nLoader reads by name+version; version logged on each run\nPrompts treat sources as data, never instructions"
issue "Structured output validation with bounded retry" "$M" "rag" "§7" \
"All model outputs validated against the schema\nRetries bounded and counted against the 6-model-call budget\nInvalid output after retries returns a safe abstention"
issue "Citation verification: exists and supports the claim" "$M" "rag,eval" "§4.2, §10" \
"Every citation matches a retrieved page/element\nEvidence check verifies the cited element supports the claim\nUnsupported citations removed or answer downgraded\nCitation validity metric implemented"
issue "Uncertainty and abstention logic" "$M" "rag" "§8" \
"Clarification or abstention for unreadable photo, no source, contradictory sources, out-of-role risk\nNo model-generated confidence percentage anywhere\nEvidence and missing info shown instead\nEval scenarios for each case"
issue "FastAPI with auth and minimal Streamlit UI (question + photo)" "$M" "infra,security" "§3" \
"Authentication and authorization on every endpoint\nImage size limits enforced\nUI uploads photo, shows answer, citations and uncertainties"

# ---------- W4 Agent et MCP ----------
M="W4 Agent et MCP"
issue "MCP maintenance server with server-side authorization" "$M" "mcp,security" "§6, §9" \
"Fictional data for equipment history and tickets\nAuthorization checked on every call\nNetwork destinations restricted\nTests: unauthorized call rejected"
issue "MCP client and LangGraph workflow state machine" "$M" "mcp" "§6" \
"Nodes follow the SPEC §6 flow\nState explicit and typed\nHistory consulted only when needed\nGraph diagram exported to docs/"
issue "Budgets and loop limits enforced in code" "$M" "security" "§9" \
"Max 3 MCP calls and 6 model calls per request\n60 s automatic processing limit (excluding human wait)\nImage, context and token limits\nPer-user quotas and GPU cap\nTests prove each limit triggers"
issue "Human approval interrupt, bound token and idempotency" "$M" "mcp,security" "§6" \
"User sees exact ticket content before approval\nToken bound to user and content hash; edit forces re-approval\nIdempotency key prevents duplicates\nTests: no creation without approval, replay and tamper rejected"
issue "End-to-end ticket flow: draft -> approve -> create" "$M" "mcp" "§6, §14" \
"Scenario from SPEC §1 works end to end\nDraft, approval and creation visible in UI\nEval scenarios for MCP consultation pass"

# ---------- W5 Reranker et evaluation ----------
M="W5 Reranker et evaluation"
issue "Reranker dataset: 800-1500 verified examples with hard negatives" "$M" "data,eval" "§5" \
"Question, positive passage, hard negative per example\nSplits by document or equipment family (no leakage)\nAnnotation process and quality checks documented\nDataset versioned"
issue "Fine-tune the multilingual reranker" "$M" "rag" "§5" \
"Base model and license recorded via ADR\nTraining config pinned and reproducible\nRuns tracked in MLflow\nAdapted model versioned"
issue "Compare: no reranker vs original vs adapted" "$M" "eval" "§5, §10" \
"nDCG@5, MRR@10 and final answer quality reported for all three\nResults reported honestly even if no improvement\nTable stored in eval/results"
issue "Expand eval set to ~150 scenarios" "$M" "eval" "§10" \
"40 text / 30 image / 20 MCP / 20 missing-contradictory / 20 access+approval / 20 injection\nAnnotations reviewed\nText-only vs multimodal comparison runs"
issue "Prompt-injection tests (text, OCR images, MCP responses)" "$M" "security,eval" "§9" \
"Representative injection from SPEC §9 included\nInjections in text, OCR'd images and MCP outputs\nNo exfiltration, no permission escalation, no unauthorized call\nFailures analysed manually"
issue "Access-control and approval regression suite" "$M" "security,eval" "§10" \
"Zero cross-group leaks and zero ticket creations without approval in tests\nRuns in CI"

# ---------- W6 Deploiement et soutenance ----------
M="W6 Deploiement et soutenance"
issue "Performance benchmark: median and p95 latency, memory, GPU time" "$M" "eval,infra" "§10" \
"Measured on retained hardware\np95 vs the 30 s target reported honestly"
issue "Docker Compose demo deployment and rollback" "$M" "infra" "§11" \
"One-command deploy of full stack\nPinned images, models, index and prompt versions\nRollback to previous version documented and tested"
issue "Fiche de limites (limits sheet)" "$M" "docs" "§13.5" \
"Authorized uses, risks, covered equipment, conditions of use\nStates it is a documentation aid, no diagnosis guarantee"
issue "Demo script and rehearsal (7 steps)" "$M" "docs" "§14" \
"Nominal, ambiguity, injection, denied access, approval scenarios scripted\nFixtures ready and repeatable\nRehearsed end to end"
issue "Final README, evaluation report and repo cleanup" "$M" "docs" "§13" \
"Reproducible setup in README\nEvaluation report with before/after fine-tuning\nSTATUS.md and ARCHITECTURE.md match reality\nEvery SPEC deviation has an ADR"

echo "Done. Create a GitHub Project board and add all issues."
