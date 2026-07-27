"""Integration-test fixtures — REAL mu-dev-* containers, ZERO mocks (DEV-STANDARDS).

Every store client here connects to a real running container whose host port comes from the
central Settings tree (``.env.test`` -> ``get_settings()``), never a hardcoded literal.
Tests isolate themselves with a unique ``org``/``workspace``/``session`` per test so no
cross-test contamination, and tear down the collections/graphs/keys they create.

If a container is not up, the fixture RAISES (test is BLOCKED/errored) — it is never faked
(DEV-STANDARDS: "if a real dependency isn't up, the test is BLOCKED, never faked").
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable

import aiomcache
import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mu_contracts.config import Settings
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.relational.schema import Base

VECTOR_DIM = 8  # tiny deterministic vectors — no ML dep needed to test store/filter/recall


@pytest.fixture(scope="session")
def settings() -> Settings:
    # single env-boundary read; .env.test wires the mu-dev-* host ports.
    return Settings()


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


@pytest.fixture
def make_ns(uid: str) -> Callable[..., Namespace]:
    """A per-test-unique namespace factory (isolation by construction)."""

    def _make(
        *,
        visibility: Visibility = Visibility.PRIVATE,
        user: str = "u1",
        session: str = "s1",
        workspace: str | None = None,
    ) -> Namespace:
        ws = workspace or f"ws{uid}"
        if visibility is Visibility.SHARED:
            return Namespace.shared(org=f"org{uid}", workspace=ws, session=session)
        return Namespace(
            org=f"org{uid}",
            workspace=ws,
            user=user,
            session=session,
            visibility=Visibility.PRIVATE,
        )

    return _make


@pytest.fixture
def make_item() -> Callable[..., MemoryItem]:
    """A MemoryItem factory with a deterministic tiny embedding derived from the content."""

    def _make(
        ns: Namespace,
        content: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        polarity: Polarity = Polarity.POSITIVE,
        authorized_ids: list[str] | None = None,
        memory_id: str | None = None,
    ) -> MemoryItem:
        seed = sum(ord(c) for c in content)
        embedding = [((seed + i) % 17) / 17.0 for i in range(VECTOR_DIM)]
        meta = {"authorized_ids": authorized_ids} if authorized_ids is not None else {}
        kwargs = {}
        if memory_id is not None:
            kwargs["id"] = memory_id
        return MemoryItem(
            content=content,
            kind=MemoryKind.PROPOSITION,
            namespace=ns,
            owner_id=ns.user if ns.visibility is Visibility.PRIVATE else "owner1",
            workspace_id=ns.workspace,
            session_id=ns.session,
            subject=subject,
            predicate=predicate,
            object=obj,
            polarity=polarity,
            embedding=embedding,
            embedding_model="test-fixture",
            metadata=meta,
            **kwargs,
        )

    return _make


@pytest_asyncio.fixture
async def redis_client(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=True)
    await client.ping()  # fail-loud if the container is down
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def qdrant_client(settings: Settings) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=settings.storage.vector.url)
    await client.get_collections()  # fail-loud
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def falkor_db(settings: Settings) -> AsyncIterator[FalkorDB]:
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    # fail-loud probe
    await db.select_graph("_probe").query("RETURN 1")
    yield db


@pytest_asyncio.fixture
async def pg_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(settings.storage.postgres.dsn, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # idempotent (checkfirst)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def mysql_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(settings.storage.mysql.dsn, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # idempotent (checkfirst)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def valkey_client(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.storage.valkey.url, decode_responses=True)
    await client.ping()  # fail-loud if the container is down
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def memcached_client(settings: Settings) -> AsyncIterator[aiomcache.Client]:
    client = aiomcache.Client(settings.storage.memcached.host, settings.storage.memcached.port)
    await client.version()  # fail-loud if the container is down
    try:
        yield client
    finally:
        await client.close()
