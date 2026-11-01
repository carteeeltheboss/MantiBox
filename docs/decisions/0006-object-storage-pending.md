# 0006 — Object storage backend is undecided (MinIO unavailable)

- **Date:** 2026-11-01
- **Status:** proposed — **awaiting a human decision**
- **Issue/PR:** Docker stack (W1/W2)

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
**Deferred at the user's request** until they can supply a working MinIO image
reference. In the meantime:

- `storage/objects.py` defines `ObjectStore` (put/get/exists/delete/stat) with a
  filesystem backend as the default and an S3 backend that imports its client
  lazily;
- nothing above the interface references a vendor, so adopting MinIO later is a
  configuration change, not a refactor.

## Consequences
- Ingestion and the API can be built and tested now.
- The compose stack ships without an object-store container until this is
  resolved; that gap is recorded in `docs/STATUS.md`, not hidden.
- **Must be revisited before the deployment deliverable is considered done.**
  Whoever resolves it replaces the status with `accepted`, records the chosen
  backend, and updates SPEC §3 or accepts a permanent deviation here.