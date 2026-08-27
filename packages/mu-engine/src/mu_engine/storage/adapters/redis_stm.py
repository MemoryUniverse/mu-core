"""Redis/Valkey KV adapter — the STM tier repository (``storage-pluggable §3.2``).

PORT of the ``SETEX`` payload + recency ``ZSET`` + scope-in-every-key defence from the
prototype ``/home/user/hackathon/memory_universe/shared/stores/stm_redis.py:96-97,226-230``,
re-homed to the ``mu/…:stm:…`` key catalog via :class:`RedisMapper`.

Fully async (``redis.asyncio``); no blocking in the loop (DEV-STANDARDS rule 1). Tenancy is
enforced by :meth:`RedisMapper.memory_key` prefixing every key with ``Namespace.to_prefix()``
(CANONICAL §1 rule 5). Field-projection is delegated to the mapper (spec Contract-change 5).

WRITE-TIME DEDUP (D4, CONFIG-AND-DATA-FIX-PLAN.md PART 2 D4; conformance D-8): ``ports.py``'s
``StmTierRepository`` docstring has promised "Recency floor + TTL + chash dedup" since the port
was written, but no adapter implemented the last third of that — ``put()`` keyed SOLELY on
``item.id`` (a fresh random id every call, ``storage/domain/memory.py::_memory_id``), so two
``add()`` calls carrying byte-identical content landed as TWO STM rows (the empirically observed
``ada_coffee`` written-twice defect, DATA-QUALITY-ASSESSMENT.md §3.1/#5). ``put()`` now maintains
a namespace-prefixed ``content_hash -> memory_id`` index (:meth:`RedisMapper.content_hash_key`,
a Redis HASH) alongside the existing payload + recency ZSET: a hit means "this exact content
already has a live STM row" — bump that row's recency + TTL and SKIP the duplicate payload write
(never fork a second entry); a miss (first-seen content, or the previous holder already expired/
was evicted — self-healed via an ``EXISTS`` check, no separate cleanup needed elsewhere) writes
through as before AND records the mapping. Gated by ``IngestSettings.stm_dedup`` (env
``MU_INGEST__STM_DEDUP``, default on) — DI-threaded by ``storage.factories._build_redis``/
``_build_valkey``, never hardcoded (DEV-STANDARDS rule 3).

RETURN-IDEMPOTENCY (``add()`` return contract, DATA-QUALITY-REASSESSMENT §3 "add() idempotency" /
the D4 report): ``put()`` now RETURNS the resident memory id — ``item.id`` on a fresh write, or the
WINNING (pre-existing) id on a dedup hit — so a caller (``WriteStmStage``) can surface the id the
store actually kept, instead of the fresh id it discarded (``ports.py``'s ``StmTierRepository.put``
docstring).
"""

from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime
from typing import cast

from redis.asyncio import Redis

from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import RecallChannel, Scored
from mu_engine.storage.mappers.redis_mapper import RedisMapper
from mu_engine.storage.ports import RedisRecord
from mu_engine.storage.tier_capabilities import (
    ENUMERATE_INSPECT_BUDGET,
    PAGE_SLACK,
    decode_rank_cursor,
    encode_rank_cursor,
    item_matches,
    with_pin_group,
)

__all__ = ["RedisStmAdapter"]

# Constructor DEFAULT only (DEV-STANDARDS rule 3: no hardcoded constant lives in adapter LOGIC).
# The live value is DI-threaded from the central Settings tree (``RedisSettings``/
# ``ValkeySettings`` ``.store_io_timeout_s``) by the ``STORE_REGISTRY`` factory
# (``mu_engine.storage.factories._build_redis``/``_build_valkey``); a bare
# ``RedisStmAdapter(client)`` (e.g. in a unit test) still gets a sane, named default.
_DEFAULT_STORE_IO_TIMEOUT_S = 5.0
# Same DEFAULT-only discipline for the D4 write-time dedup toggle: DI-threaded from
# ``IngestSettings.stm_dedup`` (env ``MU_INGEST__STM_DEDUP``) by the SAME factories; a bare
# ``RedisStmAdapter(client)`` still gets the safe (dedup ON) default.
_DEFAULT_STM_DEDUP_ENABLED = True


class RedisStmAdapter:
    """Implements ``StmTierRepository`` over a real Redis/Valkey connection.

    Every external call is wrapped by :func:`retry_io` (transient-only retry/backoff +
    per-attempt timeout) so no store call is unbounded (MAJOR-4). The timeout is a
    per-instance, DI-threaded ``retry_io`` wrapper (never a class-level decorator baking in a
    module constant) so ``store_io_timeout_s`` is genuinely tunable from Settings.
    """

    def __init__(
        self,
        client: Redis,
        *,
        mapper: RedisMapper | None = None,
        store_io_timeout_s: float = _DEFAULT_STORE_IO_TIMEOUT_S,
        stm_dedup_enabled: bool = _DEFAULT_STM_DEDUP_ENABLED,
    ) -> None:
        self._redis = client
        self._mapper = mapper or RedisMapper()
        self._retry = retry_io(timeout_s=store_io_timeout_s)
        self._stm_dedup_enabled = stm_dedup_enabled

    async def put(self, item: MemoryItem) -> str:
        return await self._retry(self._put_impl)(item)

    async def _put_impl(self, item: MemoryItem) -> str:
        row = self._mapper.to_store(item)
        recency = RedisMapper.recency_key(item.namespace)

        if self._stm_dedup_enabled:
            existing_id = await self._bump_if_duplicate(item, row.ttl_s, recency)
            if existing_id is not None:
                # duplicate content: recency/TTL bumped on the WINNER, no second row — RETURN
                # that winner's id (return-idempotency, module docstring), never `item.id` (the
                # fresh id the store never actually kept).
                return existing_id

        # ONE atomic transaction (MINOR-6): SET payload + ZADD recency + EXPIRE apply together
        # or not at all — no torn state where the payload exists without its recency member.
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(row.key, row.blob, ex=row.ttl_s)
        pipe.zadd(recency, {item.id: item.created_at.timestamp()})
        if row.ttl_s:
            pipe.expire(recency, row.ttl_s)
        if self._stm_dedup_enabled:
            chash_key = RedisMapper.content_hash_key(item.namespace)
            pipe.hset(chash_key, item.content_hash, item.id)
            if row.ttl_s:
                pipe.expire(chash_key, row.ttl_s)
        await pipe.execute()
        return item.id

    async def _bump_if_duplicate(
        self, item: MemoryItem, ttl_s: int | None, recency: str
    ) -> str | None:
        """D4 write-time dedup (conformance D-8): if ``item.content_hash`` already maps to a
        DIFFERENT, still-live memory_id in this namespace, bump that row's recency score + TTL
        and return that WINNING id (skip the caller's payload write) instead of forking a second
        entry. ``None`` means no bump happened (a miss, or a genuine re-put of the same id).

        Self-healing (no separate cleanup path needed): the ``EXISTS`` check below treats a stale
        mapping — the previous holder's TTL already expired, or it was explicitly ``evict()``-ed —
        as a miss, so ``_put_impl`` falls through to a normal write-through that overwrites the
        mapping with the new id (exactly the existing recency-ZSET self-heal pattern
        ``_recent_impl`` already uses for expired members).
        """
        chash_key = RedisMapper.content_hash_key(item.namespace)
        # redis-py's stub types `hget` as `Awaitable[str | None] | str | None` (a sync/async
        # overload union the type checker can't narrow from a plain client instance, unlike
        # `get`'s `Awaitable[Any] | Any` — same root cause as the `no-any-unimported` ignores
        # elsewhere in this package's adapters); this client is always the real async one.
        hget_call = cast(Awaitable[str | None], self._redis.hget(chash_key, item.content_hash))
        existing_raw = await hget_call
        if existing_raw is None:
            return None
        existing_id = _as_str(existing_raw)
        if existing_id == item.id:
            return None  # a genuine re-put of the SAME id — let the normal path re-write it.
        existing_key = RedisMapper.memory_key(item.namespace, existing_id)
        if not await self._redis.exists(existing_key):
            return None  # stale index entry (winner expired/evicted) — treat as a fresh write.
        pipe = self._redis.pipeline(transaction=True)
        pipe.zadd(recency, {existing_id: item.created_at.timestamp()})
        if ttl_s:
            pipe.expire(existing_key, ttl_s)
            pipe.expire(recency, ttl_s)
            pipe.expire(chash_key, ttl_s)
        await pipe.execute()
        return existing_id

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return await self._retry(self._get_impl)(ns, memory_id)

    async def _get_impl(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        key = RedisMapper.memory_key(ns, memory_id)
        blob = await self._redis.get(key)
        if blob is None:
            return None
        return self._mapper.from_store(RedisRecord(key=key, ttl_s=None, blob=_as_str(blob)))

    async def recent(self, ns: Namespace, *, limit: int) -> list[Scored[MemoryItem]]:
        return await self._retry(self._recent_impl)(ns, limit=limit)

    async def _recent_impl(self, ns: Namespace, *, limit: int) -> list[Scored[MemoryItem]]:
        recency = RedisMapper.recency_key(ns)
        ids = await self._redis.zrevrange(recency, 0, max(0, limit - 1))
        out: list[Scored[MemoryItem]] = []
        for rank, raw in enumerate(ids):
            memory_id = _as_str(raw)
            item = await self.get(ns, memory_id)
            if item is None:
                # a TTL-expired member still lingering in the ZSET — drop it, self-heal.
                await self._redis.zrem(recency, memory_id)
                continue
            out.append(
                Scored(
                    item=item, score=1.0, channel=RecallChannel.STM_FLOOR, rank=rank, is_floor=True
                )
            )
        return out

    async def enumerate_page(
        self,
        ns: Namespace,
        *,
        states: frozenset[MemoryState],
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        """STM's half of the bounded partition walk (``TierEnumerationPort``).

        Walks the recency ZSET this adapter already maintains — an ordered, namespace-scoped,
        rank-addressable index — rather than a ``KEYS``/``SCAN`` glob over the keyspace. That
        choice is the whole point: a glob is O(keyspace) across every tenant in the Redis
        instance and would have to filter foreign keys out client-side, which is exactly the
        filter-only isolation CANONICAL §1 rule 5 refuses. The ZSET key is itself derived from
        ``ns.to_prefix()``, so a walk physically cannot see another partition's members.

        The cursor is the ZSET RANK to resume from — a position in the inspected window, not a
        count of returned items. The two differ whenever the filters reject rows, and conflating
        them would silently skip the rejected rows' successors on the next page.
        """
        return await self._retry(self._enumerate_page_impl)(
            ns, states=states, pinned=pinned, cursor=cursor, limit=limit
        )

    async def _enumerate_page_impl(
        self,
        ns: Namespace,
        *,
        states: frozenset[MemoryState],
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        if limit <= 0:
            return [], None
        start = decode_rank_cursor(cursor)
        recency = RedisMapper.recency_key(ns)
        out: list[MemoryItem] = []
        inspected = 0
        rank = start
        while len(out) < limit and inspected < ENUMERATE_INSPECT_BUDGET:
            window = min(ENUMERATE_INSPECT_BUDGET - inspected, limit - len(out) + PAGE_SLACK)
            raw_ids = await self._redis.zrevrange(recency, rank, rank + window - 1)
            if not raw_ids:
                return out, None  # ZSET exhausted: the walk is genuinely finished.
            for raw in raw_ids:
                rank += 1
                inspected += 1
                memory_id = _as_str(raw)
                item = await self._get_impl(ns, memory_id)
                if item is None:
                    # TTL-expired member still lingering in the ZSET — drop it, self-heal, and
                    # do NOT count it against the page (same discipline as ``_recent_impl``).
                    await self._redis.zrem(recency, memory_id)
                    continue
                if item_matches(item, states=states, pinned=pinned):
                    out.append(item)
                    if len(out) >= limit:
                        break
        return out, encode_rank_cursor(rank)

    async def set_pinned(
        self,
        ns: Namespace,
        memory_id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None,
    ) -> int | None:
        """STM's half of the id-stable cross-store pin upsert (``TierPinPort``)."""
        return await self._retry(self._set_pinned_impl)(
            ns, memory_id, pinned, at=at, by=by, reason=reason
        )

    async def _set_pinned_impl(
        self,
        ns: Namespace,
        memory_id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None,
    ) -> int | None:
        # The key is `mu/{ns.to_prefix()}:stm:mem:{id}` — tenancy rides INSIDE the address, so a
        # memory_id belonging to another partition simply resolves to a key that does not exist
        # and the write is reported ABSENT (`None`), never applied to a foreign row. This is the
        # STM twin of `qdrant_mtm._scoped_point_selector`'s server-side predicate.
        key = RedisMapper.memory_key(ns, memory_id)
        blob = await self._redis.get(key)
        if blob is None:
            return None
        current = self._mapper.from_store(RedisRecord(key=key, ttl_s=None, blob=_as_str(blob)))
        updated = with_pin_group(current, pinned=pinned, at=at, by=by, reason=reason)
        # KEEPTTL: the STM row's remaining lifetime is its recency semantics. Re-SETting without
        # it would silently make every pinned row immortal in a tier whose whole contract is a
        # TTL'd window — and re-SETting with a FRESH TTL would let pin masquerade as a recall
        # and reinforce the item, which memory-health §6.5 rule 2 forbids outright.
        await self._redis.set(key, updated.model_dump_json(), keepttl=True)
        return updated.version

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        return await self._retry(self._evict_impl)(ns, memory_id)

    async def _evict_impl(self, ns: Namespace, memory_id: str) -> None:
        pipe = self._redis.pipeline(transaction=True)
        pipe.delete(RedisMapper.memory_key(ns, memory_id))
        pipe.zrem(RedisMapper.recency_key(ns), memory_id)
        await pipe.execute()


def _as_str(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value
