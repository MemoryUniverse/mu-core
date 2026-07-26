# Memory Universe — Project Rules

Production memory infrastructure for AI agents. Two products on one engine: **Product A** (end-user
memory app) and **Product B** (developer platform / SDK). This directory holds the **three
repositories** the system is built from.

## Repositories (three, separate)

| Repo | License | Contains | Depends on |
|---|---|---|---|
| **mu-core** | **Apache-2.0 (open)** | `mu-contracts` (lean pydantic vocabulary) · `mu-engine` (the FULL memory engine) · `mu-sdk` + `mu-sdk-ts` (wire clients, no engine) | nothing |
| **mu-client** | **Apache-2.0 (open)** | the local daemon: host capture/inject/bridge, local stores, on-device engine host | mu-core only |
| **mu-server** | **Commercial / private** | ONLY the multi-user/hosted machinery: governance/ACL/transfer, governed rooms, gateway edge, sync-hub, multi-tenancy, metering | mu-core only |

`mu-client` and `mu-server` **never import each other** — they talk only over the versioned wire
contracts in `mu-core`. (ADR 0011, ADR 0022.)

## The boundary rule — LEAN CORE, but FULL-LOCAL MUST WORK WELL

- **The complete memory engine at GOOD quality is OPEN, in `mu-core`:** capture, extraction, all
  three tiers, promotion/demotion, conflict resolution, **good ranking/salience**, persona, the
  model router, the local graph. **FULL-LOCAL is a complete, good, on-device system — never a
  crippled baseline.** Do NOT gate engine quality behind the server.
- **`mu-server` holds ONLY what exists *because* other people / devices / tenants / bills are
  involved:** cross-user governance, sharing, the gateway edge, the sync-hub, multi-tenancy,
  metering. A solo local user needs none of it, so keeping it server-side costs local nothing.
- The moat is the **team plane + hosting + operational trust + the accumulated memory graph** —
  NOT withholding single-user quality (our own analysis: the moat is not primarily code).

## Decided stack (adopt over build; build only the moat)

Valkey (KV/STM/bus) · Qdrant (MTM vector, hybrid dense+sparse) · **graph LTM: Neo4j Community +
LadybugDB embedded behind `GraphStorePort` — PENDING multi-tenancy verification** (SSPL FalkorDB
dropped) · Postgres (control plane) · **LiteLLM + RouteLLM** as the model/provider router substrate
(+ our 5 thin layers) · sentence-transformers (embed + rerank) · Temporal (server durable exec) ·
SQLite-WAL outbox (client durable exec) · Centrifugo (push) · OTel + prometheus-client + structlog +
Grafana LGTM (observability). We BUILD only the moat: the engine, governance, federate-live recall,
the two-appender sync log, the content-free discipline.

## Binding rules for every agent working here

1. **Code adoption:** when porting from a reference repo, work from the ACTUAL cloned source,
   re-implement faithfully, cite `file:line`, **never guess** — see
   `CODE-ADOPTION-METHODOLOGY.md`.
2. **Study `other_repos/`** (`/home/user/D/abstract_project/mma/other_repos/`) before building;
   follow proven patterns + their empirical findings. Reference repos are **READ-ONLY**.
3. **Content-free discipline:** never put memory content in logs, traces, the event bus, or metering.
4. **Multi-tenant:** every store access is namespace-scoped (η = org/workspace/user/agent/session +
   `to_prefix()`).
5. **Full observability:** every billable dimension (tokens, ingest, storage, any op) is tracked —
   pricing is designed later on top of the meter.
6. Never touch `GCMem` / `gcmem-*` / `milvus-standalone`.
7. **No `git push`** without explicit user authorization.
8. **Design authority:** `CANONICAL-CONTRACTS.md` is rank-above-all. The full design set currently
   lives at `/home/user/hackathon/docs/superpowers/design/` (+ `/docs/decisions/` ADRs) and will be
   migrated into this project.
9. TDD. No feature code until the design + implementation plan are approved.
