"""DISTILL pipeline fixtures — REAL mu-dev-falkordb (LTM), ZERO mocks (DEV-STANDARDS).

The graph client connects to the live ``mu-dev-falkordb`` whose host port comes from the central
Settings tree (``.env.test`` -> ``Settings``), never a literal. Each test uses a unique workspace
so its per-``(workspace,user)`` graph is isolated, and the fixture tears every ``mu_g__ws*`` graph
it created down. If the container is down the fixture RAISES (BLOCKED, never faked).
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

from mu_contracts.config import Settings
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.domain.memory import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryTier,
    Polarity,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility


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
    """A structured MTM-tier ``MemoryItem`` factory (an S2 atomic-fact output shape)."""

    def _make(
        ns: Namespace,
        content: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        valid_at: datetime | None = None,
        polarity: Polarity = Polarity.POSITIVE,
        tier: MemoryTier = MemoryTier.MTM,
        memory_id: str | None = None,
    ) -> MemoryItem:
        kwargs = {"id": memory_id} if memory_id is not None else {}
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
            valid_at=valid_at,
            **kwargs,
        )

    return _make


@pytest_asyncio.fixture
async def falkor_db(settings: Settings) -> AsyncIterator[FalkorDB]:
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    await db.select_graph("_probe").query("RETURN 1")  # fail-loud probe
    try:
        yield db
    finally:
        # close the underlying redis-protocol connection (resource hygiene, DEV-STANDARDS):
        # an unclosed connection per test accumulates and exhausts FalkorDB clients in a full run.
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


# MTM dim for the cross-store supersession test — a tiny vector is enough to prove the point-state
# flip (no semantic search here); the real embedder dim is exercised in the mu-local E2E.
_MTM_TEST_DIM = 8


@pytest_asyncio.fixture
async def qdrant_client(settings: Settings, uid: str) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        await client.get_collections()  # fail-loud probe; BLOCKED (never faked) if qdrant is down
        yield client
    finally:
        for coll in (await client.get_collections()).collections:
            if uid in coll.name:
                with contextlib.suppress(Exception):  # best-effort teardown only
                    await client.delete_collection(coll.name)
        await client.close()


@pytest_asyncio.fixture
async def mtm(qdrant_client: AsyncQdrantClient) -> QdrantMtmAdapter:
    """REAL Qdrant MTM adapter (the cross-store supersede target), ZERO mocks."""
    return QdrantMtmAdapter(qdrant_client, dim=_MTM_TEST_DIM)
