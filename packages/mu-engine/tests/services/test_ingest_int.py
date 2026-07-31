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


async def test_second_call_with_different_activity_and_same_content_is_honest_about_reinforcement(
    ingest_service: IngestService,
    private_ns: Namespace,
    qdrant_client: AsyncQdrantClient,
    embedder: SentenceTransformerEmbedder,
    query_vector: Callable[[str], Awaitable[list[float]]],
) -> None:
    """F2a receipt-honesty fix (Stage-F acceptance, ``tests/acceptance/test_f2a_crash_replay.py``).

    Two INDEPENDENT ``remember()`` calls (distinct ``session_offset`` — never colliding on
    ``WriteStmStage``'s own activity-id ledger, exactly what ``SurfaceFacade.add()`` mints fresh
    per call, ``facade.py::_fresh_offset``) carrying the IDENTICAL content: a legitimate second
    occurrence (reinforcement) mints its OWN ``memory_id`` (``activity_id_for``'s own docstring —
    a new source offset is kept, never treated as a pure replay), but the content-hash-scoped
    ``DeterministicPromoteStage`` ledger correctly dedupes the MTM write. Pre-fix, the SECOND
    call's receipt falsely claimed ``promoted=True, tiers_written=("stm", "mtm")`` for a memory_id
    that was NEVER actually written to MTM (only the FIRST memory_id was) — this test pins the
    honest receipt this fix produces instead, distinct from
    ``test_remember_is_idempotent_on_replay`` above (which replays the SAME activity/offset and
    correctly still reports ``promoted=True``, matching that op's own completed state)."""
    first = await ingest_service.remember(_fact(private_ns, offset="off-a", importance=0.9))
    second = await ingest_service.remember(_fact(private_ns, offset="off-b", importance=0.9))

    # two independent occurrences -> two distinct memory_ids (never collapsed to the first's id)
    assert second.memory_id != first.memory_id
    assert second.content_hash == first.content_hash

    # the HONEST receipt: this call did NOT perform a new MTM write (reinforcement, not promotion)
    assert second.promoted is False
    assert second.tiers_written == ("stm",)

    # the first call's receipt is unaffected — still a genuine, freshly-promoted write
    assert first.promoted is True
    assert first.tiers_written == ("stm", "mtm")

    # ground truth: the content-hash MTM dedup itself still holds — exactly ONE point, at the
    # FIRST call's memory_id (never duplicated, never re-pointed at the second's id)
    mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
    qv = await query_vector("Where does Ada work?")
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
