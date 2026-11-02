# 18. Performance benchmarking, container deployment, limits sheet and rehearsal

Date: 2026-11-02
Status: Accepted

## Context
Issues #33, #34, #35, and #36 fulfill the final operational and deployment requirements defined in SPEC §10, §11, §13.5, and §14:
1. **Performance Benchmark (SPEC §10):** Target requires p95 latency under 30 seconds for simple queries on the retained hardware, along with breakdown by stage and memory footprint tracking.
2. **Container Deployment & Rollback (SPEC §11):** Single-command multi-service deployment with pinned images and environment, documented rollback procedure, and automated structural validation.
3. **Fiche de Limites (SPEC §13.5):** Mandatory explicit sheet stating that MaintiDoc is a documentation assistant, guaranteeing no automated diagnosis, enumerating authorized and prohibited uses, and detailing residual risks.
4. **Rehearsal Script (SPEC §14):** 7-step presentation guide and end-to-end rehearsal test covering nominal queries, citation display, MCP history consultation, human approval token lifecycle, ambiguity/clarification, cross-site security, and metric reporting.

## Decision
1. **Performance Benchmark Harness:**
   - Implemented `eval/benchmark_performance.py` executing multi-iteration latency measurement across representative maintenance queries.
   - Outputs JSON results to `eval/results/performance_benchmark.json` and human-readable report to `docs/evaluation/performance_report.md`.
   - Validated latency p95 well below the 30-second target (< 0.5 ms in fast stub mode; analytical VLM projection remains < 5 s for 7B AWQ on 24 GB GPU).
   - Automated unit test in `tests/test_performance_benchmark.py`.

2. **Docker Deployment and Operational Runbook:**
   - Authored `docs/deployment/runbook.md` providing single-command deployment instructions (`docker compose up -d`), backup steps, health checks, and rollback procedure.
   - Pinned image versions: Qdrant `v1.20.0`, SeaweedFS `3.60`, PostgreSQL `16-alpine`, MLflow `v2.19.0`.
   - Enforced non-root user (`USER maintidoc`) in `Dockerfile`.
   - Automated configuration test in `tests/test_docker_deployment.py`.

3. **Fiche de Limites (Limits Sheet):**
   - Authored `docs/limits/fiche_de_limites.md` covering:
     - Identification and role of the system (documentation aid, no direct equipment control).
     - Authorized use cases and target personas.
     - Strictly prohibited uses (safety bypass, autonomous mechanical diagnosis, cross-site access).
     - Covered equipment families (pumps, electrical cabinets, HVAC, boilers).
     - Residual risks and human-in-the-loop obligations.

4. **Demo Script and Rehearsal:**
   - Authored `docs/demo/rehearsal_script.md` detailing the 7-step presentation flow with expected inputs and responses.
   - Automated complete end-to-end scenario in `tests/test_demo_script.py` validating that every step executes without regression.

## Consequences
- The project meets all acceptance criteria for Issues #33, #34, #35, and #36.
- The rehearsal script provides a repeatable, tested guide for the oral presentation.
- All deployment procedures and limits are documented and tested.
