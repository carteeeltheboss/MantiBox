# 0006 — Object storage is SeaweedFS (MinIO is no longer deployable)

- **Date:** 2026-11-01
- **Status:** accepted
- **Issue/PR:** Docker stack (W1/W2), #10

## Context
SPEC §3 specifies **MinIO** for file storage. As of 2026-11-01 MinIO cannot be
deployed from any public source, verified directly rather than assumed:

| Reference | Result |
|---|---|
| `docker pull minio/minio:latest` | `pull access denied, repository does not exist or may require 'docker login'` |
| `docker manifest inspect minio/minio:latest` | denied |
| `docker manifest inspect quay.io/minio/minio:latest` | no such manifest |
| `docker manifest inspect bitnami/minio:latest` | no such manifest |
| `https://dl.min.io/server/minio/release/linux-amd64/` | **HTTP 410 Gone** |

`dxflrs/garage` is also unavailable. `chrislusf/seaweedfs` and
`localstack/localstack` are pullable.

This blocks the object-storage half of the Docker stack only. It does not block
ingestion, retrieval or the API, because those depend on an interface.

## Options considered
1. **SeaweedFS behind an `ObjectStore` interface** — a real S3-compatible store,
   pullable and small. Keeps SPEC §3's object-store semantics and keeps the
   stack portable.
2. **Filesystem backend only** — zero containers, but the compose stack no
   longer demonstrates an object store, weakening the architecture deliverable.
3. **LocalStack S3 emulation** — supported S3 surface, but a test emulator
   rather than a store, and heavy.
4. **A MinIO image the user can reach** — keeps SPEC §3 literally, but the stack
   is then not deployable by anyone without that access.

## Decision
**SeaweedFS**, decided by the project owner on 2026-11-01, with the storage layer
kept behind the `ObjectStore` interface so the backend is a configuration change
rather than a refactor:

- `storage/objects.py` defines `ObjectStore` (put/get/exists/delete/stat) with a
  filesystem backend as the working default and an S3 backend that imports its
  client lazily.
- `docker compose` runs SeaweedFS and the application selects the S3 backend.
- Nothing above the interface references a vendor, so returning to MinIO later —
  or moving to real S3 — requires no code change.

This is a deviation from SPEC §3 ("MinIO pour les fichiers") and is recorded as
such; SPEC §3 is not edited.

## Consequences
- Ingestion and the API are unblocked and already built against the filesystem
  backend.
- SPEC §3's object-store *semantics* (S3 API, files outside PostgreSQL, originals
  preserved for the VLM) are preserved; only the vendor changes.
- SeaweedFS must be pinned to a specific image tag when the Compose file lands.
- Revisit if MinIO becomes deployable again and exact SPEC §3 fidelity is wanted.