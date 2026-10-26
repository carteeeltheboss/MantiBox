# PROMPTS — copy/paste playbook for working with any LLM

Replace `#N` with the issue number. These are the ONLY prompts you need to drive the project.

---

## 1. Session start (use for every session, every LLM)

```
You are working on MaintiDoc. You have no memory of previous sessions; the repo is the memory.

Read, in this order: AGENTS.md, docs/STATUS.md, issue #N (`gh issue view N`), and the relevant section of docs/SPEC.md (cited in the issue). Also read the ADRs in docs/decisions/ that touch this area.

Then reply with:
1. Your understanding of the task in 3 lines.
2. A plan: files to create/modify, tests to write first, risks.
3. Questions about anything ambiguous or any conflict between the issue, SPEC and current code.

Do NOT write code yet. Wait for my approval.
```

## 2. Tests/contract first

```
Plan approved. Step 1 only: write the failing tests (and the Pydantic schemas / interfaces they need). No implementation. Run `make test` and show me that they fail for the right reason.
```

## 3. Implement

```
Tests approved. Implement the smallest change that makes them pass and satisfies the issue's acceptance criteria. Do not touch unrelated code. Run `make check` and show the real output. Commit with Conventional Commits on branch feat/N-short-name.
```

## 4. Self-review before PR (anti-slop pass)

```
Review your own diff (`git diff main`) as a skeptical reviewer. Check:
- every acceptance criterion of issue #N, one by one, with evidence (test name or command)
- invented APIs or functions not present in the pinned library versions
- scope creep, dead code, unreferenced TODOs, inline prompts
- AGENTS.md "NEVER" rules
- logging of raw content, secrets, missing access checks
List what you did NOT verify. Then fix what you found.
```

## 5. Security-critical review (auth, access filter, MCP, approval, tickets)

```
Act as an attacker. For the code changed in this PR, try to: bypass the user/group filter, call an MCP tool without authorization, reuse or alter an approval token, double-create a ticket, make a document/tool result act as an instruction, and leak content into logs. For each, say where the code prevents it (file:line) and which test proves it, or say "not covered" and write the missing test.
```

## 6. Session end (mandatory)

```
Before we stop:
1. Update docs/STATUS.md (done / in progress / next / blockers / unverified).
2. Add an ADR in docs/decisions/ for every decision made this session.
3. Add docs/sessions/YYYY-MM-DD-topic.md if the session was non-trivial.
4. Update docs/ARCHITECTURE.md if a component landed.
5. Tell me the PR/branch state and the exact first step for the next session.
Show me the diff of the docs. Do not say "done" until this is finished.
```

## 7. Switching LLM (handoff check)

Open a new chat in the new tool and run prompt 1. Then add:

```
Before planning, prove you have the context: summarize in 10 lines what is built, what is in progress, the last 3 decisions (from ADRs) and the next step, citing the files you read. I will correct you if anything is wrong.
```

If the summary is wrong, the repo docs are incomplete: fix the docs, not the chat.

## 8. When the LLM goes off the rails

```
Stop. Revert to the plan we approved. List what you changed that was not in the plan, and undo it. Re-read AGENTS.md "NEVER" and "Anti-slop rules", then continue with the smallest change only.
```

## 9. Sanity audit (run once a week)

```
Audit the repo against docs/SPEC.md. For each section 1–14, mark: implemented / partial / missing / deviates (link ADR). Check that docs/STATUS.md and docs/ARCHITECTURE.md match reality. Check that every pinned version, model revision and prompt version is recorded. Output a table. Do not change code.
```
