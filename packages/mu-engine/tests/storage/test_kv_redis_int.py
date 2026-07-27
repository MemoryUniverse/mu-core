"""KV/STM adapter — REAL mu-dev-cache (Redis/Valkey), ZERO mocks."""

from __future__ import annotations

from collections.abc import Callable

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
