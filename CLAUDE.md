# mu-core

**Open (Apache-2.0). Depends on nothing.** The shared foundation, imported by both `mu-client` and
`mu-server`. See `../CLAUDE.md` for the project-wide rules — they bind here.

## Ships three distributions

- **`mu-contracts`** — the lean vocabulary: pydantic DTOs, the frozen event catalog, wire schemas
  (REST/MCP), ports/protocols, error hierarchy, config base. Pydantic-only; **no engine, no stores,
  no strategies, no embedders.** This is the versioned public API `mu-client` and `mu-server` both
  pin.
- **`mu-engine`** — the FULL memory engine at good quality: stores/tiers/façade, ingest / recall /
  promotion / demotion / conflict services, the pipelines framework, engine workflows, and **good
  baseline strategy implementations**. This is what makes FULL-LOCAL a complete, good system.
- **`mu-sdk`** (+ **`mu-sdk-ts`**) — typed wire clients (REST + Centrifugo + MCP) over
  `mu-contracts` only. **Carries no engine, no store, no strategy, no embedder.** The daemon reuses
  it as its one server-facing client (dogfooded, drift-proof).

## Rules specific to this repo

- **Nothing server-only** (governance, ACL, rooms runtime for multi-human, gateway, sync-hub,
  multi-tenancy, metering) lives here — that is `mu-server`.
- **`mu-sdk` must not import `mu-engine`** (lint-enforced: `sdk-has-no-engine`).
- Engine quality stays HERE and open — do not move ranking/salience/promotion quality to the server.
- Model access goes through the LiteLLM-backed `ModelRouter` behind the provider ports (many-to-many
  model↔provider, background health, local-priority, streaming+single, long-text chunking,
  startup-warm local models via LiteLLM `CustomLLM`).
