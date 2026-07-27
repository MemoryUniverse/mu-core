"""INGEST service — REAL mu-dev-cache (STM) + mu-dev-qdrant (MTM) + REAL MiniLM, ZERO mocks.

The load-bearing acceptance test (engine-core-spec §5): ``IngestService.remember(activity)`` writes
the item to STM (Redis) AND, on deterministic promotion, embeds the atomic-fact vector with the REAL
local embedder and upserts it to MTM (Qdrant). Idempotency (M12/B4) and the not-promoted path are
verified on the same live stores.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.providers.embedding import SentenceTransformerEmbedder
from mu_engine.services.ingest import IngestService
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.memory import MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name

pytestmark = pytest.mark.integration


def _fact(ns: Namespace, *, offset: str, importance: float) -> IngestActivity:
    return IngestActivity(
        namespace=ns,
        host="claude-code",
        session_offset=offset,
        kind="user_message",
        text="Ada works at Acme as a staff engineer",
        importance=importance,
        subject="Ada",
        predicate="works_at",
        object="Acme",
    )


async def test_remember_writes_stm_and_mtm(
    ingest_service: IngestService,
    private_ns: Namespace,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
    query_vector: Callable[[str], Awaitable[list[float]]],
) -> None:
    result = await ingest_service.remember(_fact(private_ns, offset="off-1", importance=0.9))

    # fast-return receipt
    assert result.promoted is True
    assert result.tiers_written == ("stm", "mtm")
    assert "MemoryCaptured" in result.events_emitted
    assert "MemoryPromoted" in result.events_emitted
    assert "IngestCompleted" in result.events_emitted

    # STM — the item is durable on the REAL Redis, still tier=STM, content_hash stable
    stm = RedisStmAdapter(redis_client)
    stm_item = await stm.get(private_ns, result.memory_id)
    assert stm_item is not None
    assert stm_item.tier is MemoryTier.STM
    assert stm_item.content_hash == result.content_hash

    # MTM — the promoted copy is on the REAL Qdrant, semantically retrievable, tier=MTM/active
    mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
    qv = await query_vector("Where does Ada work?")
    hits = await mtm.semantic(private_ns, qv, limit=5)
    by_id = {h.item.id: h.item for h in hits}
    assert result.memory_id in by_id
    assert by_id[result.memory_id].tier is MemoryTier.MTM
    assert by_id[result.memory_id].state is MemoryState.ACTIVE
    assert by_id[result.memory_id].embedding_model == embedder.model_name


async def test_remember_is_idempotent_on_replay(
    ingest_service: IngestService,
    private_ns: Namespace,
    qdrant_client: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
    query_vector: Callable[[str], Awaitable[list[float]]],
) -> None:
    activity = _fact(private_ns, offset="off-1", importance=0.9)
    first = await ingest_service.remember(activity)
    second = await ingest_service.remember(activity)  # exact replay: same activity_id + chash

    # same identity, and the replay still reports promoted (events re-published from the ledger, B4)
    assert second.memory_id == first.memory_id
    assert second.promoted is True
    assert "MemoryPromoted" in second.events_emitted

    # no double write — exactly one MTM point for this content
    mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
    qv = await query_vector("Ada works at Acme")
    hits = await mtm.semantic(private_ns, qv, limit=10)
    assert [h.item.id for h in hits] == [first.memory_id]


async def test_low_importance_stays_stm_only(
    ingest_service: IngestService,
    private_ns: Namespace,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
    query_vector: Callable[[str], Awaitable[list[float]]],
) -> None:
    result = await ingest_service.remember(_fact(private_ns, offset="off-9", importance=0.2))

    assert result.promoted is False
    assert result.tiers_written == ("stm",)
    assert "MemoryPromoted" not in result.events_emitted

    stm = RedisStmAdapter(redis_client)
    assert await stm.get(private_ns, result.memory_id) is not None

    # nothing promoted to MTM (collection absent or empty for this η)
    coll = collection_name(private_ns, embedder.dimension)
    if await qdrant_client.collection_exists(coll):
        mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
        qv = await query_vector("Ada works at Acme")
        hits = await mtm.semantic(private_ns, qv, limit=10)
        assert result.memory_id not in [h.item.id for h in hits]
