"""KV/STM adapter — REAL ``mu-dev-memcached``, ZERO mocks.

Covers the same conformance body as Redis/Valkey PLUS the memcached-specific recency
emulation: the CAS-guarded list (degrade D6) still produces a correctly-ordered, self-healing
recency floor, and the cap (``recency_cap``) is honored (never unbounded).
"""

from __future__ import annotations

from collections.abc import Callable

import aiomcache
import pytest

from mu_contracts.config import Settings
from mu_engine.storage.adapters.memcached_stm import MemcachedStmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.factories import STORE_REGISTRY

pytestmark = pytest.mark.integration


def _build(settings: Settings, *, recency_cap: int | None = None) -> MemcachedStmAdapter:
    cfg: dict[str, object] = {
        "host": settings.storage.memcached.host,
        "port": settings.storage.memcached.port,
    }
    if recency_cap is not None:
        cfg["recency_cap"] = recency_cap
    adapter: MemcachedStmAdapter = STORE_REGISTRY.build("kv", "memcached", **cfg)
    return adapter


async def test_put_get_evict_roundtrip(
    settings: Settings,
    memcached_client: aiomcache.Client,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = _build(settings)
    ns = make_ns()
    item = make_item(ns, "the sky is blue")
    await adapter.put(item)
    got = await adapter.get(ns, item.id)
    assert got is not None
    assert got == item  # lossless blob round-trip on the REAL memcached server
    await adapter.evict(ns, item.id)
    assert await adapter.get(ns, item.id) is None


async def test_recency_floor_self_heals_expired_members(
    settings: Settings,
    memcached_client: aiomcache.Client,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = _build(settings)
    ns = make_ns()
    items = [make_item(ns, f"fact {i}") for i in range(3)]
    for it in items:
        await adapter.put(it)
    recent = await adapter.recent(ns, limit=10)
    assert {s.item.id for s in recent} == {it.id for it in items}
    assert all(s.is_floor for s in recent)
    for it in items:
        await adapter.evict(ns, it.id)
    # self-heal: evicted ids are pruned from the CAS recency list, not just the row.
    assert (await adapter.recent(ns, limit=10)) == []


async def test_namespace_isolation(
    settings: Settings,
    memcached_client: aiomcache.Client,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = _build(settings)
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")
    item_a = make_item(ns_a, "belongs to A")
    await adapter.put(item_a)
    assert await adapter.get(ns_b, item_a.id) is None
    assert (await adapter.recent(ns_b, limit=10)) == []
    await adapter.evict(ns_a, item_a.id)


async def test_recency_cap_is_bounded(
    settings: Settings,
    memcached_client: aiomcache.Client,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """D6: no native sorted set — the CAS-emulated recency list is capped, never unbounded."""
    adapter = _build(settings, recency_cap=2)
    ns = make_ns()
    items = [make_item(ns, f"fact {i}") for i in range(4)]
    for it in items:
        await adapter.put(it)
    recent = await adapter.recent(ns, limit=10)
    assert len(recent) == 2  # capped — the two oldest fell off the recency list
    for it in items:
        await adapter.evict(ns, it.id)
