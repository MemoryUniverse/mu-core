"""``EngineServerSettings`` — the env boundary for ``mu-engine-server``'s composition root
(build-plan §4 C4, item 6c).

WHY a SEPARATE ``Settings`` tree from ``mu_contracts.config.Settings`` (``get_settings()``):
that central tree is imported transitively by ``mu_engine.storage.factories`` (each factory's
own ``store_io_timeout_s``/etc. fallback, e.g. ``_build_qdrant``'s ``qdrant_settings =
get_settings().storage.vector``) and stays available — this module does not replace it. But
``mu_engine_server`` is its own deployable process (design §7.1/§7.2: "docker run/compose it
once"), so its OWN endpoint resolution (which host/port each store lives at, the bearer-token
path) is a purpose-built, single-purpose tree here, not threaded through the shared multi-role
``StorageSettings`` shape ``mu_local.config.StorageSettings`` builds on (that class cannot be
imported — module docstring / design §8 boundary rule — and its ``BackendChoice``/pluggable-
multi-backend generality is more than this single-tenant server needs: it always binds exactly
Valkey + Qdrant + FalkorDB + sqlite-control-plane, never a config-selected alternative backend).

Dev defaults below mirror ``docker-compose.dev.yml``'s host-facing ports exactly (module
docstring there: "Host ports ... valkey 16379 -> 6379 | qdrant 16333 -> 6333 | falkordb
16380 -> 6379") plus the dev SLM (``mu-dev-slm``, Ollama, host port 11435 — the SAME endpoint
``mu_local.config.ModelProfileSettings``'s own docstring names, and every ``test_*_slm_int.py``
integration test in this tree already points at). A caller overrides any of them via env
(``MU_ENGINE_SERVER_VALKEY__HOST=...``, nested delimiter ``__``, same convention as the central
``Settings`` tree, ``mu_contracts/config/settings.py``'s own ``env_nested_delimiter="__"``) for a
non-dev deployment — never a hardcoded literal read directly by :mod:`mu_engine_server.composition`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mu_engine_server.auth import DEFAULT_TOKEN_PATH

__all__ = [
    "EngineServerSettings",
    "FalkorDBEndpoint",
    "LifecycleSweepSettings",
    "QdrantEndpoint",
    "SlmProfile",
    "ValkeyEndpoint",
]


class ValkeyEndpoint(BaseModel):
    """The ONE Valkey endpoint this server binds for BOTH the STM tier repository (via
    ``STORE_REGISTRY.build("kv", "valkey", ...)``) AND the durable ``RedisStageLedger``/
    ``RedisConflictRecordRepository`` (item 6c) — same host/port, two ``redis.asyncio.Redis``
    client instances (one wrapped by the tier adapter, one raw for the ledger/conflict repo),
    since neither of those adapters exposes its wrapped client for reuse."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = 16379  # docker-compose.dev.yml: cache/valkey 16379 -> 6379
    db: int = 0

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class QdrantEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    http_port: int = 16333  # docker-compose.dev.yml: qdrant 16333 -> 6333 (REST)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}"


class FalkorDBEndpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = 16380  # docker-compose.dev.yml: falkordb 16380 -> 6379


class SlmProfile(BaseModel):
    """Optional local SLM profile (Ollama OpenAI-compat shim) — re-declared here at the same
    field shape/defaults as ``mu_local.config.ModelProfileSettings`` (that class cannot be
    imported, module docstring), pointed at the SAME dev SLM every ``*_slm_int.py`` integration
    test in this tree already targets (``mu-dev-slm``, host port 11435, model ``qwen2.5:0.5b``).

    ``enabled=True`` by default (unlike ``mu_local.config.StorageSettings.llm: ... | None = None``
    heuristic-mode default): the context this server is built against has the dev SLM confirmed
    up and healthy, and building a ``ModelRouter`` for a LOCAL_HTTP deployment opens no in-process
    weights (mirrors ``LocalContainer``'s own docstring: "a cheap, synchronous construction" —
    no I/O at composition time, only at first ``generate()`` call). Set ``enabled=False`` (or
    point ``base_url``/``model`` elsewhere) for a deployment with no local SLM reachable — the
    composition root then stays in heuristic mode, byte-identical to ``LocalContainer``'s own
    ``llm=None`` default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    provider: str = "openai"  # litellm's provider prefix; OpenAI-compatible local servers use this
    base_url: str = "http://127.0.0.1:11435/v1"  # Ollama's OpenAI-compat shim
    model: str = "qwen2.5:0.5b"
    api_key: str = "sk-mu-engine-server-placeholder"  # NOT a secret; local shims rarely check it
    max_tokens: int = 512
    temperature: float = 0.0
    provider_key: str = "mu_engine_server_llm"  # registry key stamped on the ProviderRecord
    model_group: str = "mu-engine-server-llm"  # the ONE logical group every LLM task routes to


class LifecycleSweepSettings(BaseModel):
    """Config for `mu_engine_server.lifecycle_runner.EngineLifecycleSweepRunner` — the in-process
    automatic lifecycle-sweep runner that fixes cross-session recall (CONFIG-AND-DATA-FIX-PLAN.md
    T2, `docs/superpowers/design/memory-layer-design.md` §0.2/§6.1 path (ii)/(iii)).

    Root cause this closes: a fact `add()`ed lands in session-scoped STM and only becomes
    cross-session-recallable once promoted to MTM or distilled to LTM (the federating tiers,
    `mu_engine/storage/adapters/qdrant_mtm.py`'s `_user_prefix` scoping) — but nothing moved it
    there automatically (`mu_local/composition.py`'s own docstring: "mu-local ships NO automatic
    sweep... the caller drives the sweep"; the MLM `MaintenanceLoop` equivalent ran only in
    mu-client's daemon, S1-07). This runner is `mu-engine-server`'s own always-on equivalent.

    `enabled` is DISTINCT from `LifecycleSettings.enabled` (`mu_engine.lifecycle.settings`, which
    gates the lifecycle subsystem's decision logic generally and defaults True everywhere): this
    flag gates only whether an AUTOMATIC BACKGROUND RUNNER exists — a server-only concern. Default
    True here (this server is always-on, single-tenant, design §2.2) — an embedded/daemonless
    `mu-local` caller that ever opts into an equivalent runner MUST default it False (module
    docstring of `mu_local/composition.py`: "daemonless: the caller drives the sweep" stays the
    documented contract there; this class is never imported by `mu_local`, per the
    `mu-engine-server-boundary` import-linter contract, so there is no way for that default to leak
    across the boundary).

    `interval_s`/`idle_threshold_s` are THIS runner's own scheduling cadence — distinct from (and
    layered on top of) `LifecycleSettings.maintenance_interval_s`/`session_idle_s` (which the MLM
    itself would use for its OWN `periodic_tick`/registry, a path this runner deliberately does NOT
    call — see the module docstring for why: it mirrors `mu-client`'s `MaintenanceLoop`, which keeps
    its own active-user registry and calls `sweep_user` directly, exactly as
    `MemoryLifecycleManager`'s own docstring says a plane WITHOUT its own MaintenanceLoop-equivalent
    would).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    # Periodic safety-net sweep — the slow backstop that fires a sweep for every user this
    # process has seen recent activity for, even with no idle gap (spec §6.1 path (iii)).
    interval_s: int = Field(default=300, ge=1)
    # Session-boundary ("sleep-time") idle timer — sweeps a user once `idle_threshold_s` has
    # elapsed since their last observed `MemoryCaptured`/`MemoryPromoted` event (spec §6.1 path
    # (ii), the "sleep-time" trigger set: `Stop`/`SessionEnd`/`PreCompact`/an idle timer).
    idle_threshold_s: int = Field(default=120, ge=1)


class EngineServerSettings(BaseSettings):
    """The ONE settings root :mod:`mu_engine_server.composition` reads (build-plan §4 C4).

    Env convention mirrors the central ``Settings`` tree: prefix ``MU_ENGINE_SERVER_``, nested
    delimiter ``__`` (e.g. ``MU_ENGINE_SERVER_VALKEY__HOST=cache.internal``). ``.env``/``.env.test``
    are read if present (same discovery mechanism as ``mu_contracts.config.Settings``), so this
    root is dev-usable with ZERO env vars set — every field already defaults to the
    ``docker-compose.dev.yml`` host-facing ports.
    """

    model_config = SettingsConfigDict(
        env_prefix="MU_ENGINE_SERVER_",
        env_nested_delimiter="__",
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    valkey: ValkeyEndpoint = Field(default_factory=ValkeyEndpoint)
    qdrant: QdrantEndpoint = Field(default_factory=QdrantEndpoint)
    falkordb: FalkorDBEndpoint = Field(default_factory=FalkorDBEndpoint)
    llm: SlmProfile = Field(default_factory=SlmProfile)
    lifecycle_sweep: LifecycleSweepSettings = Field(default_factory=LifecycleSweepSettings)

    # Bearer-token file (C3, auth.py) — same default resolution auth.py's own
    # `get_token_path()`/`DEFAULT_TOKEN_PATH` use, so a caller that sets NEITHER this field NOR
    # `MU_ENGINE_SERVER_TOKEN_PATH` (auth.py's OWN env var, deliberately un-prefixed to match its
    # already-landed contract) still gets byte-identical resolution through both paths.
    token_path: Path = Field(default_factory=lambda: DEFAULT_TOKEN_PATH)

    # The single-tenant η-fixing values (design §2.2) — threaded into BOTH SurfaceFacade's
    # constructor and build_app's workspace/namespace params (app.py docstring: "the caller (C4)
    # MUST pass the matching values here").
    workspace: str = "local"
    namespace: str = "default"

    # Keyspace prefixes for the two durable Redis-shaped adapters (item 6c) — distinct defaults
    # from mu-local's in-process callers so a shared Valkey instance never collides keys with an
    # embedded LocalContainer test run pointed at the same server.
    ledger_key_prefix: str = "mu-engine-server:pipeline-ledger"
    conflict_key_prefix: str = "mu-engine-server:conflict"


def load_settings() -> EngineServerSettings:
    """Fresh read of the env boundary. NOT cached (unlike ``mu_contracts.config.get_settings``'s
    ``@lru_cache``) — ``EngineServerApp`` is expected to be constructed exactly once per process
    by its own caller (uvicorn's target factory), so a second, independently-cached global here
    would only add a second source of truth for no benefit; tests construct their own instances
    directly via ``EngineServerSettings(...)`` overrides instead of relying on a cache."""
    return EngineServerSettings()
