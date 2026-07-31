# mu-engine-server

The thin, single-tenant HTTP server over the `SurfaceFacade` (`mu-engine`), backed by durable
stores — the "run it, then point any SDK at it" target (`docs/superpowers/specs/2026-07-31-sdk-engine-server-design.md`
§2.1). Depends on `{mu-contracts, mu-engine}` only; never `mu-local`, never `mu-server`, never
`mu-client` (§2.2/§8, CI: `mu-engine-server-boundary`).
