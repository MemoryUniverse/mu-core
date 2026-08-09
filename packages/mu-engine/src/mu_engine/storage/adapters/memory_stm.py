"""In-process KV/STM adapter — the embedded, zero-server floor
(``storage-pluggable-spec.md §1`` "in-process (dict + TTL heap + emulated recency ZSET)",
§3.2 ``memory``).

Single-process, non-durable by construction (degrade **D2**, ``KV_NONDURABLE`` —
``storage-pluggable-spec.md §7``): a crash or a second process loses the data; this is the
documented trade-off of the embedded floor, never a silently-broken promise. Namespace
isolation, id-stability, and TTL/recency semantics are otherwise IDENTICAL to the Redis/Valkey
adapters (spec §6 invariants hold across every backend).

Concurrency (DEV-STANDARDS rule 1, "no shared mutable state without an async lock"): one
``asyncio.Lock`` guards the whole in-process structure. All ops are pure in-memory (no I/O),
so the critical section is always O(log n) and never blocks the event loop.

Bounded growth (DEV-STANDARDS async sharpener, "bounded queues/backpressure — never
unbounded"): each namespace partition is capped at ``max_items_per_namespace``
(:class:`mu_contracts.config.InMemoryKvSettings`) — the LEAST-recent item is evicted first
when a ``put`` would exceed the cap, mirroring what a real recency-bounded STM floor does.

Expiry model: TTL is checked LAZILY on every read (``get``/``recent``) rather than via a
background sweep task — this keeps the adapter free of a second concurrently-running
coroutine (a deliberate simplicity trade-off for an embedded/test floor) while still
guaranteeing an expired item is NEVER returned (the correctness property that matters); an
expired entry is pruned opportunistically the next time it is touched or the namespace is
capacity-evicted.

WRITE-TIME DEDUP (D4, CONFIG-AND-DATA-FIX-PLAN.md PART 2 D4; conformance D-8), for PARITY with
``RedisStmAdapter``/``ValkeyStmAdapter`` (same package, ``redis_stm.py`` module docstring): each
``_Partition`` also carries a ``content_hash -> memory_id`` dict. On ``put()``, a content_hash
already held by a DIFFERENT, still-resident id bumps that id's recency/expiry instead of forking
a second entry — gated by ``stm_dedup_enabled`` (DI-threaded from ``IngestSettings.stm_dedup``,
env ``MU_INGEST__STM_DEDUP``, by ``storage.factories._build_memory_kv``).

RETURN-IDEMPOTENCY (``add()`` return contract, DATA-QUALITY-REASSESSMENT §3 "add() idempotency" /
the D4 report): ``put()`` now RETURNS the resident memory id — ``item.id`` on a fresh write, or the
WINNING (pre-existing) id on a dedup hit — so a caller (``WriteStmStage``) can surface the id the
store actually kept, instead of the fresh id it discarded (``ports.py``'s ``StmTierRepository.put``
docstring).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from sortedcontainers import SortedList

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import RecallChannel, Scored

__all__ = ["InMemoryStmAdapter"]


class _Partition:
    """One namespace's KV rows + recency index (spec §1 "dict + TTL heap + recency ZSET")."""

    __slots__ = ("chash", "items", "recency")

    def __init__(self) -> None:
        self.items: dict[str, tuple[MemoryItem, datetime | None]] = {}  # id -> (item, expires_at)
        # (-created_at_ts, id) so ascending iteration = most-recent-first (ZREVRANGE emulation).
        # sortedcontainers ships no stubs (same as asyncpg/falkordb elsewhere in this package,
        # pgvector_mtm.py/falkor_ltm.py) — left unannotated so it infers `Any` rather than
        # tripping `disallow_any_unimported` on an explicit `SortedList[...]` hint.
        self.recency = SortedList()
        # D4 write-time dedup index (conformance D-8): content_hash -> memory_id, parity with
        # RedisMapper.content_hash_key's Redis HASH (redis_stm.py module docstring).
        self.chash: dict[str, str] = {}


class InMemoryStmAdapter:
    """Implements ``StmTierRepository`` over a plain in-process dict (embedded floor)."""

    def __init__(
        self,
        *,
        max_items_per_namespace: int = 10_000,
        default_ttl_s: int | None = 3600,
        stm_dedup_enabled: bool = True,
    ) -> None:
        self._max_items = max_items_per_namespace
        self._default_ttl_s = default_ttl_s
        self._stm_dedup_enabled = stm_dedup_enabled
        self._lock = asyncio.Lock()
        self._partitions: dict[str, _Partition] = {}

    def _partition(self, ns: Namespace) -> _Partition:
        prefix = ns.to_prefix()
        part = self._partitions.get(prefix)
        if part is None:
            part = _Partition()
            self._partitions[prefix] = part
        return part

    @staticmethod
    def _is_expired(expires_at: datetime | None, *, now: datetime) -> bool:
        return expires_at is not None and expires_at <= now

    def _evict_locked(self, part: _Partition, memory_id: str) -> None:
        entry = part.items.pop(memory_id, None)
        if entry is not None:
            item, _ = entry
            key = (-item.created_at.timestamp(), memory_id)
            if key in part.recency:
                part.recency.remove(key)

    def _prune_expired_locked(self, part: _Partition, *, now: datetime) -> None:
        expired = [mid for mid, (_, exp) in part.items.items() if self._is_expired(exp, now=now)]
        for mid in expired:
            self._evict_locked(part, mid)

    async def put(self, item: MemoryItem) -> str:
        async with self._lock:
            part = self._partition(item.namespace)
            now = datetime.now(UTC)
            self._prune_expired_locked(part, now=now)

            if self._stm_dedup_enabled:
                existing_id = self._bump_if_duplicate_locked(part, item, now=now)
                if existing_id is not None:
                    # duplicate content: recency/TTL bumped on the WINNER, no second entry —
                    # RETURN that winner's id (return-idempotency, module docstring), never
                    # `item.id` (the fresh id the store never actually kept).
                    return existing_id

            # re-put of an existing id: drop its stale recency entry first (id-stability, no fork).
            self._evict_locked(part, item.id)
            # `None` means "no TTL" (never expires); `0` is a valid (immediate-expiry) TTL —
            # must check `is not None`, not truthiness, so `default_ttl_s=0` isn't misread as
            # "no TTL" (0 is falsy but semantically distinct from None here).
            expires_at = (
                now + timedelta(seconds=self._default_ttl_s)
                if self._default_ttl_s is not None
                else None
            )
            part.items[item.id] = (item, expires_at)
            part.recency.add((-item.created_at.timestamp(), item.id))
            if self._stm_dedup_enabled:
                part.chash[item.content_hash] = item.id
            # bounded growth: evict the least-recent entries once over cap (never unbounded).
            while len(part.items) > self._max_items:
                _, oldest_id = part.recency[-1]
                self._evict_locked(part, oldest_id)
            return item.id

    def _bump_if_duplicate_locked(
        self, part: _Partition, item: MemoryItem, *, now: datetime
    ) -> str | None:
        """D4 write-time dedup (conformance D-8), parity with ``RedisStmAdapter.
        _bump_if_duplicate``: if ``item.content_hash`` already maps to a DIFFERENT, still-resident
        id in this partition, bump ITS recency/expiry and return THAT id instead of forking a
        second entry (``None`` when no bump happened — a miss, or a genuine re-put of the same
        id). A stale mapping (the previous holder already expired/was evicted) is a miss here too
        — ``part.items.get`` returning ``None`` — so ``put()`` falls through to the normal
        write-through path, which overwrites ``part.chash`` with the new id (self-healing, no
        separate cleanup needed on eviction)."""
        existing_id = part.chash.get(item.content_hash)
        if existing_id is None or existing_id == item.id:
            return None
        existing = part.items.get(existing_id)
        if existing is None:
            return None  # stale index entry — treat as a fresh write.
        existing_item, _ = existing
        self._evict_locked(part, existing_id)  # drop the stale (item, recency) pair for this id.
        # Recency score bumps to the DUPLICATE submission's own `created_at` (mirrors
        # `redis_stm.py._bump_if_duplicate`'s `ZADD ... item.created_at.timestamp()` — respects a
        # test/caller-supplied deterministic timestamp exactly like the primary write path below
        # does); TTL, however, is wall-clock `now` (the SAME split the primary path already makes:
        # recency-order key vs. real-time expiry are independent axes in this adapter).
        bumped = existing_item.model_copy(update={"created_at": item.created_at})
        expires_at = (
            now + timedelta(seconds=self._default_ttl_s)
            if self._default_ttl_s is not None
            else None
        )
        part.items[existing_id] = (bumped, expires_at)
        part.recency.add((-item.created_at.timestamp(), existing_id))
        part.chash[item.content_hash] = existing_id
        return existing_id

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        async with self._lock:
            part = self._partition(ns)
            entry = part.items.get(memory_id)
            if entry is None:
                return None
            item, expires_at = entry
            if self._is_expired(expires_at, now=datetime.now(UTC)):
                self._evict_locked(part, memory_id)
                return None
            return item

    async def recent(self, ns: Namespace, *, limit: int) -> list[Scored[MemoryItem]]:
        async with self._lock:
            part = self._partition(ns)
            self._prune_expired_locked(part, now=datetime.now(UTC))
            out: list[Scored[MemoryItem]] = []
            for rank, (_neg_ts, memory_id) in enumerate(part.recency[: max(0, limit)]):
                item, _ = part.items[memory_id]
                out.append(
                    Scored(
                        item=item,
                        score=1.0,
                        channel=RecallChannel.STM_FLOOR,
                        rank=rank,
                        is_floor=True,
                    )
                )
            return out

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        async with self._lock:
            self._evict_locked(self._partition(ns), memory_id)
