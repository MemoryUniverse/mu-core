"""``PromotionService`` fixtures — REAL mu-dev-cache (STM/valkey) + mu-dev-qdrant (MTM) +
mu-dev-falkordb (LTM), ZERO mocks (DEV-STANDARDS non-negotiable). Ports come from the central
``Settings`` tree (``.env.test`` -> ``mu_contracts.config.Settings``), never a literal — mirrors
``tests/pipelines/conftest.py`` + ``tests/storage/conftest.py`` (same pattern, STM swapped from
Redis to the real Valkey container per the S1-01 dev profile).

If a container is not up, a fixture RAISES (BLOCKED, never faked).
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryTier,
    Polarity,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.redis_mapper import RedisMapper

# Tiny deterministic dim — no ML dep needed to prove the store/gate/promote wiring (matches
# tests/pipelines/conftest.py's own `_MTM_TEST_DIM`).
_MTM_TEST_DIM = 8


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


@pytest.fixture
def make_ns(uid: str) -> Callable[..., Namespace]:
    def _make(*, user: str = "u1", session: str = "s1") -> Namespace:
        return Namespace(
            org=f"org{uid}",
            workspace=f"ws{uid}",
            user=user,
            session=session,
            visibility=Visibility.PRIVATE,
        )

    return _make


@pytest.fixture
def make_item() -> Callable[..., MemoryItem]:
    """A ``MemoryItem`` factory exposing the salience-relevant knobs (importance/access_count/
    created_at) + the optional SPO triple, for both the STM->MTM and MTM->LTM gate tests."""

    def _make(
        ns: Namespace,
        content: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        polarity: Polarity = Polarity.POSITIVE,
        tier: MemoryTier = MemoryTier.STM,
        importance: float = 0.5,
        access_count: int = 0,
        created_at: datetime | None = None,
        memory_id: str | None = None,
    ) -> MemoryItem:
        kwargs: dict[str, object] = {}
        if memory_id is not None:
            kwargs["id"] = memory_id
        if created_at is not None:
            kwargs["created_at"] = created_at
            kwargs["updated_at"] = created_at
        return MemoryItem(
            content=content,
            kind=MemoryKind.PROPOSITION,
            namespace=ns,
            owner_id=ns.user,
            workspace_id=ns.workspace,
            session_id=ns.session,
            subject=subject,
            predicate=predicate,
            object=obj,
            polarity=polarity,
            tier=tier,
            source=MemorySource.USER,
            importance_score=importance,
            access_count=access_count,
            valid_at=created_at,
            **kwargs,
        )

    return _make


@pytest_asyncio.fixture
async def valkey_client(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.storage.valkey.url, decode_responses=True)
    await client.ping()  # fail-loud if mu-dev-cache is down
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def make_stm(valkey_client: Redis) -> Callable[..., ValkeyStmAdapter]:
    """A ``ValkeyStmAdapter`` factory with a caller-chosen TTL — the pre-TTL rescue test needs a
    SHORT one; every other test keeps the production default."""

    def _make(*, ttl_s: int = 3600) -> ValkeyStmAdapter:
        return ValkeyStmAdapter(valkey_client, mapper=RedisMapper(default_ttl_s=ttl_s))

    return _make


@pytest_asyncio.fixture
async def falkor_db(settings: Settings) -> AsyncIterator[FalkorDB]:
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    await db.select_graph("_probe").query("RETURN 1")  # fail-loud probe
    try:
        yield db
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()


@pytest_asyncio.fixture
async def ltm(falkor_db: FalkorDB) -> AsyncIterator[FalkorLtmAdapter]:
    try:
        yield FalkorLtmAdapter(falkor_db)
    finally:
        for g in await falkor_db.list_graphs():
            name = g.decode() if isinstance(g, bytes) else g
            if name.startswith("mu_g__ws"):
                with contextlib.suppress(Exception):  # best-effort teardown only
                    await falkor_db.select_graph(name).delete()


@pytest_asyncio.fixture
async def qdrant_client(settings: Settings, uid: str) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        await client.get_collections()  # fail-loud probe
        yield client
    finally:
        for coll in (await client.get_collections()).collections:
            if uid in coll.name:
                with contextlib.suppress(Exception):
                    await client.delete_collection(coll.name)
        await client.close()


@pytest_asyncio.fixture
async def mtm(qdrant_client: AsyncQdrantClient) -> QdrantMtmAdapter:
    return QdrantMtmAdapter(qdrant_client, dim=_MTM_TEST_DIM)
