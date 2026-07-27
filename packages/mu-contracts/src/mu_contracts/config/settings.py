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
    "FalkorDBSettings",
    "PostgresSettings",
    "QdrantSettings",
    "RedisSettings",
    "RuntimeMode",
    "Settings",
    "StorageSettings",
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

    @property
    def dsn(self) -> str:
        """Async SQLAlchemy DSN (asyncpg driver — DEV-STANDARDS: async drivers only)."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password.get_secret_value()}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisSettings(BaseModel):
    """KV / STM cache — Valkey-compatible, wire-identical (DEV-STANDARDS decided stack)."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class QdrantSettings(BaseModel):
    """Vector store (MTM). REST + gRPC endpoints."""

    host: str = "localhost"
    http_port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}"


class FalkorDBSettings(BaseModel):
    """Graph store (LTM bi-temporal KG; graph is MANDATORY per CANONICAL storage). RESP wire."""

    host: str = "localhost"
    port: int = 6379
    graph_name: str = "mu"

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}"


class StorageSettings(BaseModel):
    """The four decided stores, each behind a pluggable port (DEV-STANDARDS rule 5)."""

    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    cache: RedisSettings = Field(default_factory=RedisSettings)
    vector: QdrantSettings = Field(default_factory=QdrantSettings)
    graph: FalkorDBSettings = Field(default_factory=FalkorDBSettings)


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
