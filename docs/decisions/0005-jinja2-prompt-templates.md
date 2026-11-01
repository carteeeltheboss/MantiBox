# 0005 — jinja2 template format for versioned prompts

- **Date:** 2026-11-01
- **Status:** accepted
- **Issue/PR:** prompts (W3)

## Context
SPEC §7 requires prompts to live in `prompts/*.yaml`, versioned as
`name_vN.yaml`, and to instruct the model to emit strict JSON. A prompt
therefore contains literal JSON braces, e.g. `{"status": "answered", ...}`, as
well as the `{{ variable }}` placeholders it interpolates.

langchain-core's default `f-string` template format cannot express both: literal
JSON braces must be escaped as `{{`/`}}`, which is unreadable and error-prone in
a file that is mostly JSON examples, and any unescaped brace raises at render
time.

## Options considered
1. **f-string format with escaped braces** — no new dependency, but every JSON
   example in every prompt must be brace-escaped, and a missed brace fails at
   runtime rather than at load.
2. **Custom `{{var}}` regex substitution** — no dependency and trivially
   correct, but abandons SPEC §3's "LangChain fournit les composants de
   prompting" and reimplements templating.
3. **jinja2 template format, which langchain-core supports natively** — literal
   braces are inert, `{{ var }}` is the placeholder, and validation can use
   langchain-core's own `get_template_variables`.

## Decision
Option 3. `jinja2>=3.1` added to core dependencies. `prompts/loader.py` builds a
`PromptTemplate` with `template_format="jinja2"` and validates that the
variables found in the template exactly match the declared `input_variables`, so
a typo fails at load instead of silently rendering an empty prompt.

## Consequences
- Prompt files stay readable: JSON examples are written as JSON.
- jinja2 is a small, ubiquitous dependency and is pulled in transitively by the
  UI and tracking stacks anyway.
- The loader also enforces the injection guard: a variable declared in
  `untrusted_variables` must be wrapped in `<var>...</var>` in the template, or
  the prompt refuses to load (AGENTS.md NEVER #4).
- Revisit only if jinja2 proves incompatible with a pinned langchain-core major.