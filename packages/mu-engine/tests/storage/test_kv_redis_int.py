"""KV/STM adapter — REAL mu-dev-cache (Redis/Valkey), ZERO mocks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from redis.asyncio import Redis

from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace

pytestmark = pytest.mark.integration


async def test_put_get_roundtrip(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    item = make_item(ns, "the sky is blue")
    await adapter.put(item)
    got = await adapter.get(ns, item.id)
    assert got is not None
    assert got == item  # lossless blob round-trip on the REAL store
    await adapter.evict(ns, item.id)
    assert await adapter.get(ns, item.id) is None


async def test_recency_floor(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    items = [make_item(ns, f"fact {i}") for i in range(3)]
    for it in items:
        await adapter.put(it)
    recent = await adapter.recent(ns, limit=10)
    assert {s.item.id for s in recent} == {it.id for it in items}
    assert all(s.is_floor for s in recent)  # STM floor members (spec §1.1)
    for it in items:
        await adapter.evict(ns, it.id)


async def test_namespace_isolation(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = RedisStmAdapter(redis_client)
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")  # differs in one η component
    item_a = make_item(ns_a, "belongs to A")
    await adapter.put(item_a)
    # a read scoped to B never returns A's row (to_prefix() key partition).
    assert await adapter.get(ns_b, item_a.id) is None
    assert (await adapter.recent(ns_b, limit=10)) == []
    await adapter.evict(ns_a, item_a.id)


async def test_write_time_dedup_skips_the_second_row(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """D4 (conformance D-8): two `put()`s of byte-identical content (same content_hash, two
    DIFFERENT random ids — exactly what two independent `add()` calls produce) must land as
    ONE STM row, not two (the empirically observed `ada_coffee` written-twice defect)."""
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    first = make_item(ns, "Ada drinks black coffee")
    second = make_item(ns, "Ada drinks black coffee")
    assert first.id != second.id
    assert first.content_hash == second.content_hash

    await adapter.put(first)
    await adapter.put(second)

    recent = await adapter.recent(ns, limit=10)
    assert len(recent) == 1, "duplicate content forked a second STM row"
    assert recent[0].item.id == first.id, "the FIRST (winner) id must be kept, not overwritten"
    assert await adapter.get(ns, second.id) is None, "the duplicate's own id must never be written"

    await adapter.evict(ns, first.id)


async def test_write_time_dedup_bumps_recency_on_the_winner(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """A later duplicate still moves the WINNER to the front of the recency ordering (a
    resubmission of the same fact is treated as a fresh mention, not silently dropped)."""
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    base = datetime.now(UTC)
    winner = make_item(ns, "Ada drinks black coffee")
    winner = winner.model_copy(update={"created_at": base})
    older_other = make_item(ns, "an unrelated fact")
    older_other = older_other.model_copy(update={"created_at": base + timedelta(seconds=1)})
    dup = make_item(ns, "Ada drinks black coffee")
    dup = dup.model_copy(update={"created_at": base + timedelta(seconds=2)})

    await adapter.put(winner)
    await adapter.put(older_other)
    # before the dup: older_other (later created_at) is more recent than winner.
    before = await adapter.recent(ns, limit=10)
    assert next(s.item.id for s in before) == older_other.id

    await adapter.put(dup)  # bump: winner should now be most-recent again (dup's created_at).
    after = await adapter.recent(ns, limit=10)
    assert next(s.item.id for s in after) == winner.id
    assert len(after) == 2  # still no forked third row

    await adapter.evict(ns, winner.id)
    await adapter.evict(ns, older_other.id)


async def test_write_time_dedup_toggle_off_allows_duplicates(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """``stm_dedup_enabled=False`` (the ``MU_INGEST__STM_DEDUP=false`` DI path) reverts to the
    pre-fix behavior — proves the toggle genuinely drives the mechanism, not just a default."""
    adapter = RedisStmAdapter(redis_client, stm_dedup_enabled=False)
    ns = make_ns()
    first = make_item(ns, "Ada drinks black coffee")
    second = make_item(ns, "Ada drinks black coffee")

    await adapter.put(first)
    await adapter.put(second)

    recent = await adapter.recent(ns, limit=10)
    assert len(recent) == 2, "toggle off must allow the duplicate through (pre-fix parity)"

    await adapter.evict(ns, first.id)
    await adapter.evict(ns, second.id)
