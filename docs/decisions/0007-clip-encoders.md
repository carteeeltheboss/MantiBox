# 0007 — CLIP-compatible encoders for the visual index

- **Date:** 2026-11-01
- **Status:** accepted
- **Issue/PR:** #13 (visual embeddings and image index)

## Context
SPEC §3 requires a "encodeur texte-image multilingue compatible CLIP" for image
search, and SPEC §4.2 searches text *and* images. Two things are needed from one
shared vector space: a way to embed a **figure** and a way to embed a **French
query** so that "voyant orange" can retrieve a control-panel diagram.

SPEC §11 also requires that licences be checked and pinned, and §2 that the
corpus be authorized material. A non-commercial encoder would poison a
deliverable that has to be handed in and possibly published.

Sizes and licences below were read from the Hugging Face model API, not assumed.

## Options considered

| Model | Size | Licence | Problem |
|---|---|---|---|
| `jinaai/jina-clip-v2` | 13.92 GB | **CC-BY-NC-4.0** | **Non-commercial.** Rejected on licence, despite being the best multilingual fit. |
| `sentence-transformers/clip-ViT-B-32-multilingual-v1` | 4.90 GB | apache-2.0 | **Text tower only.** It maps multilingual text into CLIP ViT-B/32's image space but has no image encoder, so it cannot embed figures. |
| `openai/clip-vit-base-patch32` | 1.69 GB | permissive (MIT per the model card; HF reports "unknown") | Image tower only; its text tower is English-centric. |
| `google/siglip2-base-patch16-256` | 1.43 GB | apache-2.0 | Not CLIP-compatible in the shared-space sense, and multilingual coverage unverified. |

## Decision
**Two towers, one shared space**, which is the standard pairing for exactly this
requirement:

- **image tower:** `openai/clip-vit-base-patch32` — embeds the rendered figure PNGs;
- **text tower:** `sentence-transformers/clip-ViT-B-32-multilingual-v1` — distils
  50+ languages into that same ViT-B/32 image space, so French queries match.

Both are permissively licensed, together ~6.6 GB. `jina-clip-v2` is rejected
purely on its non-commercial licence and must not be substituted later without
revisiting this ADR.

Consequence for configuration: one model id cannot express two towers, so
`MAINTIDOC_IMAGE_EMBEDDING_MODEL` is replaced by
`MAINTIDOC_IMAGE_TOWER_MODEL` and `MAINTIDOC_IMAGE_TEXT_MODEL`.

## Consequences
- Image embeddings are **512-dimensional** (ViT-B/32). The image collection must
  be created at 512, not at BGE-M3's 1024 — the two collections have different
  widths and must never be mixed.
- Text-to-image search uses the multilingual tower; image-to-image uses the image
  tower. Both write into the same collection, so the embedder used for a query
  must match the one used at index time. This is why the embedder id is stored in
  each payload.
- `huggingface_hub[cli]` is used operationally to prefetch weights. It is already
  a transitive dependency of `transformers`, so this is not a new stack entry.
- Weights are prefetched into the git-ignored `models/hf` cache, so a later
  session without reliable internet can still load them.
- Revisit if a permissively licensed multilingual CLIP at comparable quality
  appears, or if the project is ever allowed non-commercial assets.