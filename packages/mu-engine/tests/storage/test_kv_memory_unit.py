"""In-process KV/STM adapter — pure logic, no container needed (marked unit).

Covers: put/get/evict round-trip, recency-floor ordering, namespace isolation, TTL expiry
(lazy, on read), and bounded per-namespace growth (D2 — never unbounded).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from mu_engine.storage.adapters.memory_stm import InMemoryStmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace

pytestmark = pytest.mark.unit


async def test_put_get_evict_roundtrip(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    adapter = InMemoryStmAdapter()
    ns = make_ns()
    item = make_item(ns, "the sky is blue")
    await adapter.put(item)
    got = await adapter.get(ns, item.id)
    assert got == item
    await adapter.evict(ns, item.id)
    assert await adapter.get(ns, item.id) is None


async def test_recency_floor_most_recent_first(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    adapter = InMemoryStmAdapter()
    ns = make_ns()
    items = [make_item(ns, f"fact {i}") for i in range(3)]
    for it in items:
        await adapter.put(it)
    recent = await adapter.recent(ns, limit=10)
    assert {s.item.id for s in recent} == {it.id for it in items}
    assert all(s.is_floor for s in recent)


async def test_namespace_isolation(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    adapter = InMemoryStmAdapter()
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")
    item_a = make_item(ns_a, "belongs to A")
    await adapter.put(item_a)
    assert await adapter.get(ns_b, item_a.id) is None
    assert (await adapter.recent(ns_b, limit=10)) == []


async def test_ttl_expiry_never_returned(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    adapter = InMemoryStmAdapter(default_ttl_s=0)  # expires immediately
    ns = make_ns()
    item = make_item(ns, "gone soon")
    await adapter.put(item)
    await asyncio.sleep(0.01)
    assert await adapter.get(ns, item.id) is None
    assert (await adapter.recent(ns, limit=10)) == []


async def test_no_ttl_means_no_expiry(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    adapter = InMemoryStmAdapter(default_ttl_s=None)
    ns = make_ns()
    item = make_item(ns, "never expires")
    await adapter.put(item)
    assert await adapter.get(ns, item.id) == item


async def test_bounded_growth_evicts_least_recent(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    adapter = InMemoryStmAdapter(max_items_per_namespace=2)
    ns = make_ns()
    # distinct creation timestamps so recency ordering is deterministic (no wall-clock flake).
    base = datetime(2026, 1, 1, tzinfo=UTC)
    items = []
    for i in range(3):
        it = make_item(ns, f"fact {i}")
        it = it.model_copy(update={"created_at": base + timedelta(seconds=i)})
        items.append(it)
        await adapter.put(it)
    # only the 2 most-recent survive (bounded, never unbounded — D2/async-sharpener).
    remaining = {s.item.id for s in await adapter.recent(ns, limit=10)}
    assert remaining == {items[1].id, items[2].id}
    assert await adapter.get(ns, items[0].id) is None


async def test_reput_same_id_does_not_fork(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    adapter = InMemoryStmAdapter()
    ns = make_ns()
    item = make_item(ns, "v1", memory_id="mem_fixed")
    await adapter.put(item)
    item_v2 = make_item(ns, "v2", memory_id="mem_fixed")
    await adapter.put(item_v2)
    recent = await adapter.recent(ns, limit=10)
    assert len(recent) == 1  # id-stability: a re-put never forks a second recency entry
    assert recent[0].item.content == "v2"


async def test_write_time_dedup_skips_the_second_row(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    """D4 (conformance D-8), parity with the Redis/Valkey adapters: two `put()`s of
    byte-identical content (same content_hash, two DIFFERENT random ids) land as ONE STM
    row — never a fork."""
    adapter = InMemoryStmAdapter()
    ns = make_ns()
    first = make_item(ns, "Ada drinks black coffee")
    second = make_item(ns, "Ada drinks black coffee")
    assert first.id != second.id
    assert first.content_hash == second.content_hash

    await adapter.put(first)
    await adapter.put(second)

    recent = await adapter.recent(ns, limit=10)
    assert len(recent) == 1, "duplicate content forked a second STM row"
    assert recent[0].item.id == first.id
    assert await adapter.get(ns, second.id) is None


async def test_write_time_dedup_bumps_recency_on_the_winner(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    adapter = InMemoryStmAdapter()
    ns = make_ns()
    base = datetime(2026, 7, 31, tzinfo=UTC)
    winner = make_item(ns, "Ada drinks black coffee")
    winner = winner.model_copy(update={"created_at": base})
    older_other = make_item(ns, "an unrelated fact")
    older_other = older_other.model_copy(update={"created_at": base + timedelta(seconds=1)})
    dup = make_item(ns, "Ada drinks black coffee")
    dup = dup.model_copy(update={"created_at": base + timedelta(seconds=2)})

    await adapter.put(winner)
    await adapter.put(older_other)
    before = await adapter.recent(ns, limit=10)
    assert next(s.item.id for s in before) == older_other.id

    await adapter.put(dup)
    after = await adapter.recent(ns, limit=10)
    assert next(s.item.id for s in after) == winner.id
    assert len(after) == 2  # still no forked third row


async def test_write_time_dedup_self_heals_a_stale_mapping(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    """If the WINNER of a content_hash has since been evicted, a later duplicate write is
    treated as fresh (overwrites the stale index entry) rather than silently vanishing."""
    adapter = InMemoryStmAdapter()
    ns = make_ns()
    first = make_item(ns, "Ada drinks black coffee")
    await adapter.put(first)
    await adapter.evict(ns, first.id)  # winner explicitly evicted (not TTL expiry)
    assert await adapter.get(ns, first.id) is None

    second = make_item(ns, "Ada drinks black coffee")  # SAME content_hash as the evicted winner
    await adapter.put(second)
    # self-healed: the stale mapping is overwritten, the new write actually lands.
    assert await adapter.get(ns, second.id) == second
    recent = await adapter.recent(ns, limit=10)
    assert [s.item.id for s in recent] == [second.id]


async def test_write_time_dedup_toggle_off_allows_duplicates(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    adapter = InMemoryStmAdapter(stm_dedup_enabled=False)
    ns = make_ns()
    first = make_item(ns, "Ada drinks black coffee")
    second = make_item(ns, "Ada drinks black coffee")

    await adapter.put(first)
    await adapter.put(second)

    recent = await adapter.recent(ns, limit=10)
    assert len(recent) == 2, "toggle off must allow the duplicate through (pre-fix parity)"
