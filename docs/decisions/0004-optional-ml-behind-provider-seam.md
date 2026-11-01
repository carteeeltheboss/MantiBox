# 0004 — Keep torch optional behind a provider seam

- **Date:** 2026-11-01
- **Status:** accepted
- **Issue/PR:** VLM and embedding providers (W1/W2)

## Context
The stack needs `torch`, `transformers` and several GB of weights for the VLM,
and `sentence-transformers` for BGE-M3. If those were core dependencies, then
`nox -s check`, CI and every unit test would need a 2–3 GB install, and the
whole test suite would need a GPU or would silently skip.

Separately, AGENTS.md forbids claiming tests pass when they were not run, so a
suite that mostly skips is worse than useless.

## Options considered
1. **torch as a core dependency** — one import style, no seam, but a heavy gate
   and tests that cannot run meaningfully on a laptop or in CI.
2. **Skip ML tests when torch is absent** — the suite stays green while covering
   none of the inference path. Rejected: it manufactures a false signal.
3. **Provider interface with a deterministic implementation** — production code
   depends on a protocol; tests and the portable CPU deployment use a stub.

## Decision
Option 3, applied consistently:

- `vlm/base.py` defines `VlmProvider` and the validated output contracts;
- `vlm/stub.py` is deterministic, performs no inference, prefixes every answer
  with `[STUB]`, reports `provider="stub"`, and may only cite sources actually
  present in the rendered context (so it cannot fabricate a citation);
- the heavy stack lives in the `ml` / `ml-gpu` extras and is installed in the
  container image, never in the `dev` extra;
- `describe_image` and `generate` take an explicit `expects: OutputSchema`. The
  caller declares the contract; the provider never infers it from prompt text.
  Inferring was tried first and is unsound: the `answer` prompt embeds a rendered
  `visual_analysis` payload, so its text legitimately contains another schema's
  field names and a content-sniffing heuristic picks the wrong contract.

## Consequences
- 259 unit tests run in ~2 s with no GPU, no torch and no weights.
- Every stub answer is visibly synthetic, in the text and in the provenance, so a
  stub run can never be mistaken for a real one.
- Two code paths to keep in sync — mitigated by tests that render each shipped
  prompt and assert the stub still produces schema-valid output.
- Revisit if a real provider needs a capability the protocol cannot express.