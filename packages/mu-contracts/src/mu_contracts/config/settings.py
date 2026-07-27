"""The central Settings tree — the single env boundary (CANONICAL-CONTRACTS.md §7.27,
platform-layer0 §1).

WHY this shape: DEV-STANDARDS rule 3 forbids ALL hardcoding — store hosts/ports/creds are
NEVER inline; they flow from here, sourced from the environment (or an env file such as
`.env.test` for the local mu-dev-* container stack). `get_settings()` is `@lru_cache` so the
boundary is read exactly once per process.

SCAFFOLD SCOPE: this pins the storage subtree that the `docker-compose.dev.yml` mu-dev-*
stores wire into. The remaining sibling subtrees pinned in CANONICAL §7.27 (ModelSettings,
ModelCatalogSettings, TenancySettings, HealthSettings/PinSettings, NotificationSettings,
trust_ledger, DaemonSettings, SyncSettings, SurfaceSettings/AuthSettings) are added in
their owning phases; `extra="ignore"` keeps unknown env vars from breaking the scaffold.

Env convention (nested, prefixed): `MU_STORAGE__POSTGRES__PORT=15432`, etc.
Defaults below are the IN-CONTAINER defaults; the host-facing ports for the dev stack come
from `.env.test` (never hardcoded in code).
"""

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ChromaSettings",
    "FaissSettings",
    "FalkorDBSettings",
    "InMemoryKvSettings",
    "MemcachedSettings",
    "MySQLSettings",
    "PgVectorSettings",
    "PostgresSettings",
    "QdrantSettings",
    "RedisSettings",
    "RuntimeMode",
    "Settings",
    "StorageSettings",
    "ValkeySettings",
    "get_settings",
]


class RuntimeMode(StrEnum):
    """Which plane this process is (CANONICAL §4.1). LOCAL co-hosts the daemon; SHARED is
    the team server. A SHARED process is constructed WITHOUT any local store adapter."""

    LOCAL = "local"
    SHARED = "shared"


class PostgresSettings(BaseModel):
    """Relational store (control-plane / schema via Alembic; CANONICAL §5.5a, §7)."""

    host: str = "localhost"
    port: int = 5432
    user: str = "mu"
    password: SecretStr = SecretStr("mu")
    database: str = "mu"
    # Per-attempt I/O budget for RelationalControlPlaneAdapter's retry_io wrapper (DEV-STANDARDS
    # async sharpener: "timeouts on every external call") — DI-threaded by
    # mu_engine.storage.factories._build_postgres, never a bare literal in the adapter.
    store_io_timeout_s: float = 15.0

    @property
    def dsn(self) -> str:
        """Async SQLAlchemy DSN (asyncpg driver — DEV-STANDARDS: async drivers only)."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class MySQLSettings(BaseModel):
    """Relational store — new backend (``storage-pluggable-spec.md §3.1``): "dev's stack is
    MySQL". Async driver = ``asyncmy`` (SQLAlchemy-recommended, DBAPI-2 + native async), a
    SEPARATE relational endpoint from :class:`PostgresSettings`. SQLAlchemy Core stays
    dialect-agnostic (DEV-STANDARDS rule 3/6) — no MySQL-only SQL lives in the adapter; the
    Postgres-only partial-index/GIN DDL degrades to plain composite indexes here (**D7**,
    ``RELATIONAL_INDEX_REDUCED``), never a correctness loss."""

    host: str = "localhost"
    port: int = 3306
    user: str = "mu"
    password: SecretStr = SecretStr("mu")
    database: str = "mu"
    # Per-attempt I/O budget (same seam as PostgresSettings.store_io_timeout_s) — DI-threaded by
    # mu_engine.storage.factories._build_mysql.
    store_io_timeout_s: float = 15.0

    @property
    def dsn(self) -> str:
        """Async SQLAlchemy DSN (``asyncmy`` driver)."""
        return (
            f"mysql+asyncmy://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisSettings(BaseModel):
    """KV / STM cache — Valkey-compatible, wire-identical (DEV-STANDARDS decided stack)."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    # Per-attempt I/O budget for RedisStmAdapter's retry_io wrapper — DI-threaded by
    # mu_engine.storage.factories._build_redis.
    store_io_timeout_s: float = 5.0

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class ValkeySettings(BaseModel):
    """KV / STM — the DECIDED (BSD-licensed) backend registered EXPLICITLY under its own
    ``(kv, "valkey")`` registry key (owner stage 2026-07-27), distinct from the generic
    ``(kv, "redis")`` key even though ``redis-py`` is wire-identical to both
    (``storage-pluggable-spec.md §3.2``: "valkey ... wire-identical, no gap"). Defaults point at
    the SAME ``mu-dev-cache`` container as :class:`RedisSettings` — it was recreated from
    ``redis:7-alpine`` to ``valkey/valkey:8-alpine`` (``docker-compose.dev.yml``) precisely so the
    ``valkey`` registry key names a REAL Valkey server, not merely a same-wire alias."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    # Per-attempt I/O budget — DI-threaded by mu_engine.storage.factories._build_valkey (the
    # ``ValkeyStmAdapter`` reuses ``RedisStmAdapter.__init__`` verbatim, DRY).
    store_io_timeout_s: float = 5.0

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class InMemoryKvSettings(BaseModel):
    """KV / STM — the embedded, zero-server floor (``storage-pluggable-spec.md §1`` "in-process
    (dict + TTL heap + emulated recency ZSET)"; §3.2 ``memory``). ``max_items_per_namespace`` is
    the adapter's bounded-growth knob (DEV-STANDARDS async sharpener: "bounded queues/backpressure
    — never unbounded") — NOT durable/cross-process (degrade **D2**, ``KV_NONDURABLE``)."""

    max_items_per_namespace: int = 10_000
    default_ttl_s: int = 3600


class MemcachedSettings(BaseModel):
    """KV / STM — Memcached backend (``storage-pluggable-spec.md §3.2``). Has no native sorted
    set, so the recency floor is emulated via a single CAS-guarded per-namespace key
    (degrade **D6**, ``RECENCY_ZSET_UNAVAILABLE``) capped at ``recency_cap`` entries."""

    host: str = "localhost"
    port: int = 11211
    recency_cap: int = 500
    default_ttl_s: int = 3600
    cas_max_attempts: int = 5  # bounded CAS-loop retry cap for the recency-list write (D6)
    # Per-attempt I/O budget for MemcachedStmAdapter's retry_io wrapper — DI-threaded by
    # mu_engine.storage.factories._build_memcached.
    store_io_timeout_s: float = 5.0


class QdrantSettings(BaseModel):
    """Vector store (MTM). REST + gRPC endpoints."""

    host: str = "localhost"
    http_port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    # Per-attempt I/O budget for QdrantMtmAdapter's retry_io wrapper — DI-threaded by
    # mu_engine.storage.factories._build_qdrant.
    store_io_timeout_s: float = 10.0

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}"


class FalkorDBSettings(BaseModel):
    """Graph store (LTM bi-temporal KG; graph is MANDATORY per CANONICAL storage). RESP wire.

    ``entity_shortlist_size``/``entity_similarity_threshold``/``store_io_timeout_s`` are the
    :class:`~mu_engine.storage.adapters.falkor_ltm.FalkorLtmAdapter` tunables (DEV-STANDARDS
    rule 3 — no hardcoded constant lives in adapter logic). They are DI-threaded into the
    adapter's constructor by the ``STORE_REGISTRY`` factory
    (``mu_engine.storage.factories._build_falkordb``), never read by the adapter itself.
    """

    host: str = "localhost"
    port: int = 6379
    graph_name: str = "mu"
    entity_shortlist_size: int = 5  # resolve_entity bounded candidate-set size
    entity_similarity_threshold: float = 0.84  # deterministic-match band (graph_falkor.py)
    store_io_timeout_s: float = 10.0  # per-attempt retry_io budget for every openCypher call

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}"


class PgVectorSettings(BaseModel):
    """pgvector — alternate MTM vector backend (Postgres + the ``vector`` extension), SHARED-safe
    (``storage-pluggable-spec.md §3.3``: SQL ``WHERE`` + HNSW filter BEFORE ANN truncation).

    A SEPARATE Postgres endpoint from :class:`PostgresSettings` (the control-plane store) — the
    dev stack runs a dedicated ``mu-dev-pgvector`` container (``pgvector/pgvector:pg16``) so the
    control-plane and MTM roles can be pointed at different Postgres instances (or, in a smaller
    deployment, the SAME one) without one role's settings leaking into the other's.
    """

    host: str = "localhost"
    port: int = 5432
    user: str = "mu"
    password: SecretStr = SecretStr("mu")
    database: str = "mu"
    hnsw: bool = True  # HNSW vs IVFFlat vector-index kind (mem0 configs/vector_stores/pgvector.py)
    # Per-attempt I/O budget for PgVectorMtmAdapter's retry_io wrapper — DI-threaded by
    # mu_engine.storage.factories._build_pgvector.
    store_io_timeout_s: float = 10.0
    # asyncpg pool bounds (mem0 configs/vector_stores/pgvector.py has no pool-size knob of its
    # own; this adapter's UPGRADE to a real async pool needs one — never hardcoded in the
    # adapter, DEV-STANDARDS rule 3).
    pool_min_size: int = 1
    pool_max_size: int = 5

    @property
    def dsn(self) -> str:
        """Plain ``postgresql://`` DSN for the RAW ``asyncpg`` driver (NOT the ``+asyncpg``
        SQLAlchemy-dialect suffix ``PostgresSettings.dsn`` uses) — this adapter talks asyncpg
        directly so the ``pgvector`` codec can bind ``vector`` columns."""
        return (
            f"postgresql://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class ChromaSettings(BaseModel):
    """Chroma — embedded (persistent, on-disk) MTM floor (``storage-pluggable-spec.md §1``:
    "Vector / MTM ... Chroma (embedded, ``path=``)"). No server/container; DEV-STANDARDS rule 3
    still applies — the path is config-sourced, never a hardcoded literal in the adapter."""

    path: str = "./.mu_data/chroma"
    # Per-attempt I/O budget for ChromaMtmAdapter's retry_io wrapper — DI-threaded by
    # mu_engine.storage.factories._build_chroma.
    store_io_timeout_s: float = 10.0
    # Floor on the authorized_ids widen (spec §3.3 note) — DI-threaded, never a bare literal in
    # the adapter's semantic() logic.
    min_widen: int = 50


class FaissSettings(BaseModel):
    """FAISS — in-proc, brute-force MTM floor. PRIVATE-plane ONLY (degrade rule D3,
    ``storage-pluggable-spec.md §2.3``): the registry refuses this backend on the SHARED plane."""

    path: str | None = "./.mu_data/faiss"


class StorageSettings(BaseModel):
    """The decided stores, each behind a pluggable port (DEV-STANDARDS rule 5). ``vector``
    (Qdrant) is the SHARED-plane reference; ``pgvector``/``chroma``/``faiss`` are the additional
    MTM backends the ``STORE_REGISTRY`` self-registers under the SAME ``vector`` role
    (``storage-pluggable-spec.md §2.3`` — mem0's ``VectorStoreFactory`` pattern, one config value
    picks which binds; ``faiss`` is PRIVATE-only, D3)."""

    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    mysql: MySQLSettings = Field(default_factory=MySQLSettings)
    cache: RedisSettings = Field(default_factory=RedisSettings)
    valkey: ValkeySettings = Field(default_factory=ValkeySettings)
    kv_memory: InMemoryKvSettings = Field(default_factory=InMemoryKvSettings)
    memcached: MemcachedSettings = Field(default_factory=MemcachedSettings)
    vector: QdrantSettings = Field(default_factory=QdrantSettings)
    graph: FalkorDBSettings = Field(default_factory=FalkorDBSettings)
    pgvector: PgVectorSettings = Field(default_factory=PgVectorSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    faiss: FaissSettings = Field(default_factory=FaissSettings)


class Settings(BaseSettings):
    """The ONE Settings root (CANONICAL §7.27). Sibling subtrees hang off this."""

    model_config = SettingsConfigDict(
        env_prefix="MU_",
        env_nested_delimiter="__",
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    runtime_mode: RuntimeMode = RuntimeMode.LOCAL
    storage: StorageSettings = Field(default_factory=StorageSettings)


@lru_cache
def get_settings() -> Settings:
    """The single, cached read of the env boundary. Never construct Settings() elsewhere."""
    return Settings()
