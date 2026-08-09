"""Memcached KV/STM adapter (``storage-pluggable-spec.md §3.2`` "``memcached`` | new | dev only
has Memcached | no sorted set -> recency floor via relational (**D6**)").

Memcached has no native sorted-set primitive, so this adapter emulates the recency floor with
a single **CAS-guarded** JSON list under one per-namespace key (``gets``/``cas``, the
compare-and-swap read-modify-write loop that is memcached's actual concurrency primitive — the
closest real analogue to the async-lock discipline DEV-STANDARDS rule 1 requires for shared
mutable state, here located in the EXTERNAL store rather than in-process). The list is capped
at ``recency_cap`` entries (:class:`mu_contracts.config.MemcachedSettings`), never unbounded.

This is a documented, real (non-mocked) design choice, distinct from the spec's *alternative*
D6 compensating path ("recency via relational ``ORDER BY created_at``") — that alternative
requires a cross-role dependency on the relational tier that ``StmTierRepository.recent()``'s
signature does not carry, and is out of scope for a KV-only adapter. A CAS-contended write
retries up to ``cas_max_attempts`` times before raising
:class:`~mu_engine.storage.errors.TierRepositoryUnavailableError` (fail-loud, never a silent
drop — DEV-STANDARDS rule 8).

Content is stored LOSSLESSLY as ``MemoryItem`` JSON — the SAME key catalog and blob shape as
the Redis/Valkey adapters (:class:`RedisMapper` is reused verbatim; the ``RowMapper`` seam is
backend-agnostic by construction, spec §5).
"""

from __future__ import annotations

import json

import aiomcache

from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import RecallChannel, Scored
from mu_engine.storage.errors import TierRepositoryUnavailableError
from mu_engine.storage.mappers.redis_mapper import RedisMapper
from mu_engine.storage.ports import RedisRecord

__all__ = ["MemcachedStmAdapter"]

# Constructor DEFAULT only (DEV-STANDARDS rule 3: no hardcoded constant lives in adapter LOGIC),
# consistent with the other KV adapters — DI-threaded from ``MemcachedSettings.store_io_timeout_s``
# by ``mu_engine.storage.factories._build_memcached``.
_DEFAULT_STORE_IO_TIMEOUT_S = 5.0


class MemcachedStmAdapter:
    """Implements ``StmTierRepository`` over a real ``aiomcache`` Memcached connection."""

    def __init__(
        self,
        client: aiomcache.Client,
        *,
        recency_cap: int,
        cas_max_attempts: int,
        default_ttl_s: int,
        mapper: RedisMapper | None = None,
        store_io_timeout_s: float = _DEFAULT_STORE_IO_TIMEOUT_S,
    ) -> None:
        self._mc = client
        self._recency_cap = recency_cap
        self._cas_max_attempts = cas_max_attempts
        # ONE TTL source for both the row and the recency-list key (never let a removal-only
        # write accidentally clear the list's expiry by defaulting to memcached's "0 = forever").
        self._default_ttl_s = default_ttl_s
        self._mapper = mapper or RedisMapper(default_ttl_s=default_ttl_s)
        # per-instance, DI-threaded retry wrapper (never a class-level decorator baking in a
        # module constant) so `store_io_timeout_s` is genuinely tunable from Settings.
        self._retry = retry_io(timeout_s=store_io_timeout_s)

    async def _read_recency(self, key: bytes) -> tuple[list[tuple[str, float]], int | None]:
        raw, cas_token = await self._mc.gets(key)
        if raw is None:
            return [], None
        entries = [(str(mid), float(ts)) for mid, ts in json.loads(raw)]
        return entries, cas_token

    async def _update_recency(self, ns: Namespace, memory_id: str, ts: float | None) -> None:
        return await self._retry(self._update_recency_impl)(ns, memory_id, ts)

    async def _update_recency_impl(self, ns: Namespace, memory_id: str, ts: float | None) -> None:
        """CAS read-modify-write of the per-namespace recency list; ``ts=None`` removes the id."""
        key = RedisMapper.recency_key(ns).encode("utf-8")
        for _attempt in range(self._cas_max_attempts):
            entries, cas_token = await self._read_recency(key)
            entries = [(mid, t) for mid, t in entries if mid != memory_id]
            if ts is not None:
                entries.append((memory_id, ts))
            entries.sort(key=lambda e: e[1], reverse=True)
            entries = entries[: self._recency_cap]
            new_raw = json.dumps(entries).encode("utf-8")
            ok = (
                await self._mc.add(key, new_raw, exptime=self._default_ttl_s)
                if cas_token is None
                else await self._mc.cas(key, new_raw, cas_token, exptime=self._default_ttl_s)
            )
            if ok:
                return
        raise TierRepositoryUnavailableError(
            f"memcached recency CAS failed after {self._cas_max_attempts} attempts "
            "(sustained write contention) — D6 recency floor unavailable"
        )

    async def put(self, item: MemoryItem) -> str:
        return await self._retry(self._put_impl)(item)

    async def _put_impl(self, item: MemoryItem) -> str:
        row = self._mapper.to_store(item)
        await self._mc.set(
            row.key.encode("utf-8"), row.blob.encode("utf-8"), exptime=row.ttl_s or 0
        )
        await self._update_recency(item.namespace, item.id, item.created_at.timestamp())
        # NO write-time content-hash dedup on this adapter (D6 — memcached has no HASH primitive
        # this adapter's D4 index could reuse; a tracked, separate gap, not this fix's scope), so
        # every write is resident under its OWN `item.id` — the ``StmTierRepository.put`` return-
        # idempotency contract (``ports.py``) degrades to a no-op here: always the given id.
        return item.id

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return await self._retry(self._get_impl)(ns, memory_id)

    async def _get_impl(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        key = RedisMapper.memory_key(ns, memory_id).encode("utf-8")
        blob = await self._mc.get(key)
        if blob is None:
            return None
        row = RedisRecord(key=key.decode("utf-8"), ttl_s=None, blob=blob.decode("utf-8"))
        return self._mapper.from_store(row)

    async def recent(self, ns: Namespace, *, limit: int) -> list[Scored[MemoryItem]]:
        return await self._retry(self._recent_impl)(ns, limit=limit)

    async def _recent_impl(self, ns: Namespace, *, limit: int) -> list[Scored[MemoryItem]]:
        key = RedisMapper.recency_key(ns).encode("utf-8")
        entries, _ = await self._read_recency(key)
        out: list[Scored[MemoryItem]] = []
        for rank, (memory_id, _ts) in enumerate(entries[: max(0, limit)]):
            item = await self.get(ns, memory_id)
            if item is None:
                # TTL-expired member still lingering in the recency list — self-heal, same as
                # the Redis adapter's ZSET pruning.
                await self._update_recency(ns, memory_id, None)
                continue
            out.append(
                Scored(
                    item=item, score=1.0, channel=RecallChannel.STM_FLOOR, rank=rank, is_floor=True
                )
            )
        return out

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        return await self._retry(self._evict_impl)(ns, memory_id)

    async def _evict_impl(self, ns: Namespace, memory_id: str) -> None:
        key = RedisMapper.memory_key(ns, memory_id).encode("utf-8")
        await self._mc.delete(key)
        await self._update_recency(ns, memory_id, None)
