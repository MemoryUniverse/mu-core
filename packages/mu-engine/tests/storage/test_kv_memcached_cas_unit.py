"""``MemcachedStmAdapter``'s CAS-loop logic in isolation — marked unit.

DEV-STANDARDS permits a test double ONLY here (pure logic, isolated from the real store); the
happy-path/self-heal/namespace-isolation behavior against a REAL memcached server lives in
``test_kv_memcached_int.py``. This file exercises the one path that is impractical to trigger
deterministically against a real server: sustained CAS contention exhausting the bounded retry
budget, which must raise the NAMED error rather than silently drop the write (DEV-STANDARDS
rule 8, spec §7).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from mu_engine.storage.adapters.memcached_stm import MemcachedStmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.errors import TierRepositoryUnavailableError

pytestmark = pytest.mark.unit


class _FakeMemcachedClient:
    """A minimal, in-memory double of the ``aiomcache.Client`` surface this adapter uses."""

    def __init__(self, *, cas_always_fails: bool = False) -> None:
        self._store: dict[bytes, bytes] = {}
        self._cas_token = 0
        self._cas_always_fails = cas_always_fails

    async def set(self, key: bytes, value: bytes, *, exptime: int = 0) -> bool:
        self._store[key] = value
        self._cas_token += 1
        return True

    async def get(self, key: bytes, default: bytes | None = None) -> bytes | None:
        return self._store.get(key, default)

    async def gets(
        self, key: bytes, default: bytes | None = None
    ) -> tuple[bytes | None, int | None]:
        if key not in self._store:
            return default, None
        return self._store[key], self._cas_token

    async def cas(self, key: bytes, value: bytes, cas_token: int, *, exptime: int = 0) -> bool:
        if self._cas_always_fails:
            return False
        if self._cas_token != cas_token:
            return False
        self._store[key] = value
        self._cas_token += 1
        return True

    async def add(self, key: bytes, value: bytes, *, exptime: int = 0) -> bool:
        if key in self._store:
            return False
        self._store[key] = value
        self._cas_token += 1
        return True

    async def delete(self, key: bytes) -> bool:
        return self._store.pop(key, None) is not None


async def test_cas_loop_succeeds_when_uncontended(
    make_ns: Callable[..., Namespace], make_item: Callable[..., MemoryItem]
) -> None:
    client = _FakeMemcachedClient()
    adapter = MemcachedStmAdapter(
        client,  # type: ignore[arg-type]  # duck-typed double, not aiomcache.Client
        recency_cap=10,
        cas_max_attempts=3,
        default_ttl_s=3600,
    )
    ns = make_ns()
    item = make_item(ns, "fact")
    await adapter.put(item)
    recent = await adapter.recent(ns, limit=10)
    assert {s.item.id for s in recent} == {item.id}


async def test_sustained_cas_contention_raises_named_error_never_silent(
    make_ns: Callable[..., Namespace],
) -> None:
    """A CAS token that never matches (sustained contention) exhausts the bounded retry budget
    and raises the named error — never a silent drop (DEV-STANDARDS rule 8, spec §7)."""
    client = _FakeMemcachedClient(cas_always_fails=True)
    adapter = MemcachedStmAdapter(
        client,  # type: ignore[arg-type]  # duck-typed double, not aiomcache.Client
        recency_cap=10,
        cas_max_attempts=2,
        default_ttl_s=3600,
    )
    ns = make_ns()
    # seed the key so `gets` returns a real cas_token (forcing the `cas` branch, not `add`).
    recency_key = adapter._mapper.recency_key(ns).encode("utf-8")
    await client.set(recency_key, b"[]")
    with pytest.raises(TierRepositoryUnavailableError):
        await adapter._update_recency(ns, "mem_x", 1.0)
