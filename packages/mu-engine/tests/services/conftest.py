"""INGEST service fixtures — REAL mu-dev-* containers + the REAL local MiniLM embedder, ZERO mocks.

Every store client connects to a live container whose host port comes from the central Settings
tree (``.env.test`` -> ``Settings``), never a hardcoded literal. The embedder is the actual offline
sentence-transformers MiniLM (``local_files_only`` — loads from the HF cache). If a container is
down the fixture RAISES (BLOCKED, never faked — DEV-STANDARDS).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.pipelines.ledger import RedisStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
from mu_engine.providers.catalog import ModelKind, WarmLocalConfig
from mu_engine.providers.embedding import SentenceTransformerEmbedder
from mu_engine.services.ingest import IngestService
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def embedder() -> SentenceTransformerEmbedder:
    # REAL MiniLM, loaded ONCE for the session (heavy). Offline from the HF cache.
    return SentenceTransformerEmbedder(
        WarmLocalConfig(
            model_id="minilm_local",
            kind=ModelKind.EMBED,
            model_load_path=_MINILM,
            normalize_embeddings=True,
        )
    )


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


@pytest.fixture
def private_ns(uid: str) -> Namespace:
    return Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user="u1",
        session="s1",
        visibility=Visibility.PRIVATE,
    )


@pytest_asyncio.fixture
async def redis_client(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
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
async def ingest_service(
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
    private_ns: Namespace,
) -> AsyncIterator[IngestService]:
    """A fully-wired IngestService over the REAL stores + REAL embedder + a REAL Redis ledger."""
    stm = RedisStmAdapter(redis_client)
    mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
    ledger = RedisStageLedger(redis_client, key_prefix=f"mu:test-ledger:{private_ns.workspace}")
    service = IngestService(
        stm=stm,
        mtm=mtm,
        embedder=embedder,
        bus=InprocBus(),
        ledger=ledger,
        clock=SystemClock(),
    )
    try:
        yield service
    finally:
        # tear down everything this test's unique η created (isolation by construction).
        coll = collection_name(private_ns, embedder.dimension)
        if await qdrant_client.collection_exists(coll):
            await qdrant_client.delete_collection(coll)
        for pattern in (
            f"mu/{private_ns.to_prefix()}*",
            f"mu:test-ledger:{private_ns.workspace}*",
        ):
            keys = [k async for k in redis_client.scan_iter(match=pattern.encode())]
            if keys:
                await redis_client.delete(*keys)


@pytest.fixture
def query_vector(embedder: SentenceTransformerEmbedder) -> Callable[[str], list[float]]:
    async def _embed(text: str) -> list[float]:
        return (await embedder.embed([text]))[0]

    return _embed
