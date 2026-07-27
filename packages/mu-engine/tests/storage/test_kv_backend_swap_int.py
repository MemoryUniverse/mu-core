"""KV backend-swap correctness proof (spec §8 test obligation 1: "run the SAME test body
against every registered backend for a role, parametrized"): the SAME put/recent/evict body run
against ``redis``, ``valkey``, ``memory``, and ``memcached`` — built THROUGH the
``STORE_REGISTRY`` seam — returns IDENTICAL result sets. Only the config value
(``STORE_REGISTRY.build("kv", <backend>, ...)``) changes, never the calling code.

``redis``/``valkey``/``memcached`` need their real ``mu-dev-*`` containers (marked
``integration``); ``memory`` is embedded but exercised for real here too — zero mocks on any
side.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import pytest

from mu_contracts.config import Settings
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import Scored
from mu_engine.storage.factories import STORE_REGISTRY

pytestmark = pytest.mark.integration


class _StmLike(Protocol):
    """The ``StmTierRepository`` shape — a local ``Protocol`` so the swap body below stays
    typed without importing every concrete adapter class."""

    async def put(self, item: MemoryItem) -> None: ...
    async def recent(self, ns: Namespace, *, limit: int) -> list[Scored[MemoryItem]]: ...
    async def evict(self, ns: Namespace, memory_id: str) -> None: ...


async def _run_body(
    adapter: _StmLike, ns: Namespace, make_item: Callable[..., MemoryItem]
) -> set[str]:
    """The IDENTICAL body — the only thing differing across backends is the adapter instance."""
    fact_a = make_item(ns, "Ada uses Postgres", memory_id="kvswap_a")
    fact_b = make_item(ns, "Grace uses Cobol", memory_id="kvswap_b")
    await adapter.put(fact_a)
    await adapter.put(fact_b)
    recent = await adapter.recent(ns, limit=10)
    ids = {s.item.id for s in recent}
    await adapter.evict(ns, fact_a.id)
    await adapter.evict(ns, fact_b.id)
    return ids


async def test_all_kv_backends_return_identical_result_sets(
    settings: Settings,
    redis_client: object,
    valkey_client: object,
    memcached_client: object,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    backends: dict[str, _StmLike] = {
        "redis": STORE_REGISTRY.build("kv", "redis", url=settings.storage.cache.url),
        "valkey": STORE_REGISTRY.build("kv", "valkey", url=settings.storage.valkey.url),
        "memory": STORE_REGISTRY.build("kv", "memory"),
        "memcached": STORE_REGISTRY.build(
            "kv",
            "memcached",
            host=settings.storage.memcached.host,
            port=settings.storage.memcached.port,
        ),
    }
    ns = make_ns()
    results = {name: await _run_body(adapter, ns, make_item) for name, adapter in backends.items()}
    expected = {"kvswap_a", "kvswap_b"}
    assert all(r == expected for r in results.values()), results
