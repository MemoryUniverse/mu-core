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

from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_engine.storage.authz import authorized_item, authorized_window
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import RecallChannel, Scored
from mu_engine.storage.tier_capabilities import (
    ENUMERATE_INSPECT_BUDGET,
    decode_rank_cursor,
    encode_rank_cursor,
    item_matches,
    with_pin_group,
)

__all__ = ["InMemoryStmAdapter"]


class _Partition:
    """One namespace's KV rows + recency index (spec §1 "dict + TTL heap + recency ZSET")."""

    __slots__ = ("chash", "demoted", "items", "recency")

    def __init__(self) -> None:
        # id -> (item, expires_at, demoted_at). `demoted_at` (AD-250 fix, ADR 0061) is `None` for
        # an ORDINARY row — indexed in `recency`, scored by `item.created_at` — and the DEMOTION
        # INSTANT for a demoted write-ahead copy — indexed in `demoted` instead. See
        # `put_demoted`'s own docstring for why the two indices are kept separate (parity with
        # `RedisStmAdapter`'s `stm:recency` vs `stm:demoted` ZSETs).
        self.items: dict[str, tuple[MemoryItem, datetime | None, float | None]] = {}
        # (-created_at_ts, id) so ascending iteration = most-recent-first (ZREVRANGE emulation).
        # sortedcontainers ships no stubs (same as asyncpg/falkordb elsewhere in this package,
        # pgvector_mtm.py/falkor_ltm.py) — left unannotated so it infers `Any` rather than
        # tripping `disallow_any_unimported` on an explicit `SortedList[...]` hint.
        self.recency = SortedList()
        # (-demoted_at_ts, id), the SAME ordering shape as `recency` — AD-250 fix, ADR 0061.
        self.demoted = SortedList()
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
        if entry is None:
            return
        item, _exp, demoted_at = entry
        if demoted_at is None:
            key = (-item.created_at.timestamp(), memory_id)
            if key in part.recency:
                part.recency.remove(key)
        else:
            # AD-250 fix (ADR 0061): a demoted row lives in `demoted`, not `recency`.
            dkey = (-demoted_at, memory_id)
            if dkey in part.demoted:
                part.demoted.remove(dkey)

    def _evict_oldest_over_cap_locked(self, part: _Partition) -> None:
        """Bounded growth (module docstring): evict the least-recent entry once ``part.items``
        exceeds ``self._max_items`` — checked against WHICHEVER index (``recency`` or
        ``demoted``, AD-250 fix ADR 0061) is non-empty, ``recency`` first (the common,
        fresh-capture-dominated case, unchanged from the pre-fix single-index behaviour)."""
        while len(part.items) > self._max_items:
            if part.recency:
                _, oldest_id = part.recency[-1]
            elif part.demoted:
                _, oldest_id = part.demoted[-1]
            else:  # pragma: no cover - items non-empty implies one of the two indices is too
                break
            self._evict_locked(part, oldest_id)

    def _prune_expired_locked(self, part: _Partition, *, now: datetime) -> None:
        expired = [
            mid for mid, (_, exp, _dat) in part.items.items() if self._is_expired(exp, now=now)
        ]
        for mid in expired:
            self._evict_locked(part, mid)

    async def put(self, item: MemoryItem, *, ttl_s: int | None = None) -> str:
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
            # F1 fix (ADR 0054): an explicit per-write `ttl_s` (e.g. DemotionService's write-ahead
            # copy) overrides `self._default_ttl_s` for THIS write only — `None` for both means
            # "no TTL" (never expires); `sentinel is not None` distinguishes that from a valid
            # `0` (immediate-expiry) TTL either one could legitimately carry.
            effective_ttl_s = self._default_ttl_s if ttl_s is None else ttl_s
            expires_at = (
                now + timedelta(seconds=effective_ttl_s) if effective_ttl_s is not None else None
            )
            part.items[item.id] = (item, expires_at, None)
            part.recency.add((-item.created_at.timestamp(), item.id))
            if self._stm_dedup_enabled:
                part.chash[item.content_hash] = item.id
            self._evict_oldest_over_cap_locked(part)  # bounded growth, never unbounded.
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
        existing_item, _exp, _dat = existing
        self._evict_locked(part, existing_id)  # drop the stale (item, recency) pair for this id.
        # Recency score bumps to the DUPLICATE submission's own `created_at` (mirrors
        # `redis_stm.py._bump_if_duplicate`'s `ZADD ... item.created_at.timestamp()` — respects a
        # test/caller-supplied deterministic timestamp exactly like the primary write path below
        # does); TTL, however, is wall-clock `now` (the SAME split the primary path already makes:
        # recency-order key vs. real-time expiry are independent axes in this adapter). Dedup is
        # an ordinary-capture concern only (AD-250 fix, ADR 0061 — `put_demoted`'s own docstring):
        # this path always bumps the ORDINARY `recency` index, never `demoted`.
        # D3 fix (AD-266b), parity with `RedisStmAdapter._bump_if_duplicate`: a repeat
        # content-hash hit is a re-assertion of the same fact — bump `mention_count` too, not
        # just recency, or `IngestSettings.mention_promote` can never fire against this backend.
        bumped = existing_item.model_copy(
            update={
                "created_at": item.created_at,
                "mention_count": existing_item.mention_count + 1,
                "last_seen": item.created_at,
            }
        )
        expires_at = (
            now + timedelta(seconds=self._default_ttl_s)
            if self._default_ttl_s is not None
            else None
        )
        part.items[existing_id] = (bumped, expires_at, None)
        part.recency.add((-item.created_at.timestamp(), existing_id))
        part.chash[item.content_hash] = existing_id
        return existing_id

    async def get(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> MemoryItem | None:
        """Keyed read, Model-A authorized on a SHARED η — the SAME predicate the Redis adapter
        applies (``storage/authz.py``, AD-129). The in-memory adapter holds the property too: a
        fake that authorizes more than the real store hides exactly this class of defect."""
        async with self._lock:
            part = self._partition(ns)
            entry = part.items.get(memory_id)
            if entry is None:
                found: MemoryItem | None = None
            else:
                item, expires_at, _demoted_at = entry
                if self._is_expired(expires_at, now=datetime.now(UTC)):
                    self._evict_locked(part, memory_id)
                    found = None
                else:
                    found = item
        return authorized_item(
            found, ns=ns, caller_identity_set=caller_identity_set, operation="stm.get"
        )

    async def recent(
        self,
        ns: Namespace,
        *,
        limit: int,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> list[Scored[MemoryItem]]:
        """Recency floor, Model-A filtered on a SHARED η — the SAME predicate the Redis adapter
        applies (``storage/authz.py``, AD-128). The in-memory adapter holds the property too: it
        is what unit tests and the local dev stack run against, so a fake that authorizes more
        than the real store would hide exactly this class of defect."""
        async with self._lock:
            part = self._partition(ns)
            self._prune_expired_locked(part, now=datetime.now(UTC))
            out: list[Scored[MemoryItem]] = []
            for rank, (_neg_ts, memory_id) in enumerate(part.recency[: max(0, limit)]):
                item, _exp, _dat = part.items[memory_id]
                out.append(
                    Scored(
                        item=item,
                        score=1.0,
                        channel=RecallChannel.STM_FLOOR,
                        rank=rank,
                        is_floor=True,
                    )
                )
            return authorized_window(
                out, ns=ns, caller_identity_set=caller_identity_set, operation="stm.recent"
            )

    async def put_demoted(self, item: MemoryItem, *, ttl_s: int, at: datetime) -> str:
        """The demotion write-ahead verb (AD-250 fix, ADR 0061 — ``ports.py``'s own docstring has
        the full rationale). Writes into the SAME ``part.items`` row store as :meth:`put` but
        indexes the id in ``part.demoted`` (scored by ``at``, the demotion instant) instead of
        ``part.recency`` (scored by ``item.created_at``) — the in-process twin of
        ``RedisStmAdapter``'s separate ``stm:demoted`` ZSET."""
        async with self._lock:
            part = self._partition(item.namespace)
            now = datetime.now(UTC)
            self._prune_expired_locked(part, now=now)
            self._evict_locked(part, item.id)  # drop any stale prior entry for this id first.
            expires_at = now + timedelta(seconds=ttl_s)
            demoted_ts = at.timestamp()
            part.items[item.id] = (item, expires_at, demoted_ts)
            part.demoted.add((-demoted_ts, item.id))
            self._evict_oldest_over_cap_locked(part)  # bounded growth, never unbounded.
            return item.id

    async def demoted(
        self,
        ns: Namespace,
        *,
        limit: int,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> list[Scored[MemoryItem]]:
        """The demoted-item channel (AD-250 fix, ADR 0061 — ``ports.py``'s own docstring has the
        full rationale)."""
        async with self._lock:
            part = self._partition(ns)
            self._prune_expired_locked(part, now=datetime.now(UTC))
            out: list[Scored[MemoryItem]] = []
            for rank, (_neg_ts, memory_id) in enumerate(part.demoted[: max(0, limit)]):
                item, _exp, _dat = part.items[memory_id]
                out.append(
                    Scored(
                        item=item,
                        score=1.0,
                        channel=RecallChannel.STM_DEMOTED,
                        rank=rank,
                        is_floor=False,  # never `_protected_floor_ids`-eligible.
                    )
                )
            return authorized_window(
                out, ns=ns, caller_identity_set=caller_identity_set, operation="stm.demoted"
            )

    async def enumerate_page(
        self,
        ns: Namespace,
        *,
        states: frozenset[MemoryState],
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        """STM's half of the bounded partition walk (``TierEnumerationPort``), in-process twin of
        ``RedisStmAdapter.enumerate_page``.

        Same cursor semantics as the Redis leg — a RANK into the recency order, not a count of
        returned items — so the two backends page identically and a test proving the bound on one
        is proving the same contract on the other. ``self._partition(ns)`` is keyed by
        ``ns.to_prefix()``, so a walk never sees another partition's entries.
        """
        if limit <= 0:
            return [], None
        async with self._lock:
            part = self._partition(ns)
            self._prune_expired_locked(part, now=datetime.now(UTC))
            start = decode_rank_cursor(cursor)
            out: list[MemoryItem] = []
            rank = start
            window = part.recency[start : start + ENUMERATE_INSPECT_BUDGET]
            for _neg_ts, memory_id in window:
                rank += 1
                entry = part.items.get(memory_id)
                if entry is None:
                    continue
                item, _exp, _dat = entry
                if item_matches(item, states=states, pinned=pinned):
                    out.append(item)
                    if len(out) >= limit:
                        break
            exhausted = rank >= len(part.recency)
            return out, None if exhausted else encode_rank_cursor(rank)

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
        """STM's half of the id-stable cross-store pin upsert (``TierPinPort``).

        Held under the SAME ``self._lock`` as every other mutation, so the read-modify-write is
        atomic for this partition — the in-process equivalent of the Redis leg's single ``SET``.
        The TTL/expiry pair is carried over UNCHANGED (pin is retention, never a recall: it must
        not extend the STM window, memory-health §6.5 rule 2).
        """
        async with self._lock:
            part = self._partition(ns)
            entry = part.items.get(memory_id)
            if entry is None:
                return None
            item, expires_at, demoted_at = entry
            if self._is_expired(expires_at, now=datetime.now(UTC)):
                self._evict_locked(part, memory_id)
                return None
            updated = with_pin_group(item, pinned=pinned, at=at, by=by, reason=reason)
            part.items[memory_id] = (updated, expires_at, demoted_at)
            return updated.version

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        async with self._lock:
            self._evict_locked(self._partition(ns), memory_id)

    async def reinforce(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        at: datetime,
        relevance_score: float | None = None,
    ) -> MemoryItem | None:
        """The read-stat write-back (AD-250 fix, ADR 0061 — ``ports.py``'s own docstring has the
        full rationale; AD-266 added ``relevance_score``/``last_seen``). Held under the SAME
        ``self._lock`` as every other mutation — the in-process equivalent of the Redis leg's
        single ``SET ... KEEPTTL``."""
        async with self._lock:
            part = self._partition(ns)
            entry = part.items.get(memory_id)
            if entry is None:
                return None
            item, expires_at, demoted_at = entry
            if self._is_expired(expires_at, now=datetime.now(UTC)):
                self._evict_locked(part, memory_id)
                return None
            update: dict[str, object] = {
                "access_count": item.access_count + 1,
                "updated_at": at,
                "last_seen": at,
            }
            if relevance_score is not None:
                update["relevance_score"] = relevance_score
            reinforced = item.model_copy(update=update)
            # expires_at/demoted_at UNCHANGED — no index re-scoring needed (`put_demoted`'s own
            # docstring: the demoted channel's window does not shrink from unrelated activity).
            part.items[memory_id] = (reinforced, expires_at, demoted_at)
            return reinforced
