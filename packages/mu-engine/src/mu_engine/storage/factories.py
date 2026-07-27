"""Default ``StoreRegistry`` population — one factory per (role, backend).

The ONE place adapters bind to real clients (``storage-pluggable §4.2/§4.3``). Each factory
reads its knobs from a ``BackendChoice.config``-shaped dict (``dsn``/``url``/``host``/
``port``/``dim``), NEVER a hardcoded literal — the values flow from the central Settings
tree (DEV-STANDARDS rule 3). Registered on import so ``STORE_REGISTRY.build(role, backend,
**cfg)`` resolves.
"""

from __future__ import annotations

from typing import Any

from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine

from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.adapters.relational_control import RelationalControlPlaneAdapter
from mu_engine.storage.registry import StoreRegistry

__all__ = ["STORE_REGISTRY"]

STORE_REGISTRY = StoreRegistry()


@STORE_REGISTRY.register("relational", "postgres")
def _build_postgres(**cfg: Any) -> RelationalControlPlaneAdapter:
    engine = create_async_engine(cfg["dsn"], pool_pre_ping=True)
    return RelationalControlPlaneAdapter(engine)


@STORE_REGISTRY.register("relational", "sqlite")
def _build_sqlite(**cfg: Any) -> RelationalControlPlaneAdapter:
    engine = create_async_engine(cfg.get("dsn", "sqlite+aiosqlite:///:memory:"))
    return RelationalControlPlaneAdapter(engine)


@STORE_REGISTRY.register("vector", "qdrant")
def _build_qdrant(*, dim: int, **cfg: Any) -> QdrantMtmAdapter:
    client = AsyncQdrantClient(url=cfg["url"], prefer_grpc=cfg.get("prefer_grpc", False))
    return QdrantMtmAdapter(client, dim=dim)


@STORE_REGISTRY.register("graph", "falkordb")
def _build_falkordb(**cfg: Any) -> FalkorLtmAdapter:
    db = FalkorDB(host=cfg["host"], port=cfg["port"])
    return FalkorLtmAdapter(db)


@STORE_REGISTRY.register("kv", "redis")
def _build_redis(**cfg: Any) -> RedisStmAdapter:
    client = Redis.from_url(cfg["url"], decode_responses=True)
    return RedisStmAdapter(client)
