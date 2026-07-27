"""KV/STM adapter — REAL ``mu-dev-cache`` (recreated as ``valkey/valkey:8-alpine``), ZERO mocks.

Mirrors ``test_kv_redis_int.py`` — same conformance body, different registry key/adapter class
(spec §6 invariant: a backend swap changes latency, never correctness). Built THROUGH the
``STORE_REGISTRY`` ``(kv, "valkey")`` seam, exactly as a real composition root selects it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.factories import STORE_REGISTRY

pytestmark = pytest.mark.integration


def _build(url: str) -> ValkeyStmAdapter:
    adapter: ValkeyStmAdapter = STORE_REGISTRY.build("kv", "valkey", url=url)
    return adapter


async def test_put_get_roundtrip(
    settings: Settings,
    valkey_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = _build(settings.storage.valkey.url)
    ns = make_ns()
    item = make_item(ns, "the sky is blue")
    await adapter.put(item)
    got = await adapter.get(ns, item.id)
    assert got is not None
    assert got == item  # lossless blob round-trip on the REAL Valkey server
    await adapter.evict(ns, item.id)
    assert await adapter.get(ns, item.id) is None


async def test_recency_floor(
    settings: Settings,
    valkey_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = _build(settings.storage.valkey.url)
    ns = make_ns()
    items = [make_item(ns, f"fact {i}") for i in range(3)]
    for it in items:
        await adapter.put(it)
    recent = await adapter.recent(ns, limit=10)
    assert {s.item.id for s in recent} == {it.id for it in items}
    assert all(s.is_floor for s in recent)
    for it in items:
        await adapter.evict(ns, it.id)


async def test_namespace_isolation(
    settings: Settings,
    valkey_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = _build(settings.storage.valkey.url)
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")
    item_a = make_item(ns_a, "belongs to A")
    await adapter.put(item_a)
    assert await adapter.get(ns_b, item_a.id) is None
    assert (await adapter.recent(ns_b, limit=10)) == []
    await adapter.evict(ns_a, item_a.id)


def test_valkey_is_a_distinct_registry_identity() -> None:
    """(kv, "valkey") is registered EXPLICITLY, not merely reachable through "redis" (owner
    stage 2026-07-27: "register the valkey backend explicitly")."""
    assert "valkey" in STORE_REGISTRY.known("kv")
    assert "redis" in STORE_REGISTRY.known("kv")
