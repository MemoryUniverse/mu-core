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
from collections.abc import Awaitable, Callable
from datetime import datetime

import aiomcache

from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_engine.platform.decorators import retry_io
from mu_engine.storage.authz import authorized_item, authorized_window
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
        key = RedisMapper.recency_key(ns).encode("utf-8")
        await self._update_index(key, memory_id, ts, ttl_s=self._default_ttl_s)

    async def _update_demoted(
        self, ns: Namespace, memory_id: str, ts: float | None, *, ttl_s: int
    ) -> None:
        """AD-250 fix (ADR 0061): the demoted-index twin of :meth:`_update_recency`, over its
        OWN CAS-guarded list key (``RedisMapper.demoted_key``) — see that key's own docstring
        for why a demoted item needs a list separate from the fresh-capture recency one."""
        key = RedisMapper.demoted_key(ns).encode("utf-8")
        await self._update_index(key, memory_id, ts, ttl_s=ttl_s)

    async def _update_index(
        self, key: bytes, memory_id: str, ts: float | None, *, ttl_s: int
    ) -> None:
        return await self._retry(self._update_index_impl)(key, memory_id, ts, ttl_s=ttl_s)

    async def _update_index_impl(
        self, key: bytes, memory_id: str, ts: float | None, *, ttl_s: int
    ) -> None:
        """CAS read-modify-write of the list at ``key``; ``ts=None`` removes the id. Shared by
        the recency list and the demoted list (AD-250 fix, ADR 0061) — the SAME CAS shape over
        WHICHEVER namespace-scoped list key the caller names (DEV-STANDARDS rule 6, DRY)."""
        for _attempt in range(self._cas_max_attempts):
            entries, cas_token = await self._read_recency(key)
            entries = [(mid, t) for mid, t in entries if mid != memory_id]
            if ts is not None:
                entries.append((memory_id, ts))
            entries.sort(key=lambda e: e[1], reverse=True)
            entries = entries[: self._recency_cap]
            new_raw = json.dumps(entries).encode("utf-8")
            ok = (
                await self._mc.add(key, new_raw, exptime=ttl_s)
                if cas_token is None
                else await self._mc.cas(key, new_raw, cas_token, exptime=ttl_s)
            )
            if ok:
                return
        raise TierRepositoryUnavailableError(
            f"memcached index CAS failed after {self._cas_max_attempts} attempts "
            "(sustained write contention) — D6 recency/demoted floor unavailable"
        )

    async def put(self, item: MemoryItem, *, ttl_s: int | None = None) -> str:
        return await self._retry(self._put_impl)(item, ttl_s=ttl_s)

    async def _put_impl(self, item: MemoryItem, *, ttl_s: int | None = None) -> str:
        row = self._mapper.to_store(item)
        effective_ttl_s = row.ttl_s if ttl_s is None else ttl_s  # F1 fix (ADR 0054): per-write
        # TTL override (e.g. DemotionService's write-ahead copy) beats the mapper's default.
        await self._mc.set(
            row.key.encode("utf-8"), row.blob.encode("utf-8"), exptime=effective_ttl_s or 0
        )
        await self._update_recency(item.namespace, item.id, item.created_at.timestamp())
        # NO write-time content-hash dedup on this adapter (D6 — memcached has no HASH primitive
        # this adapter's D4 index could reuse; a tracked, separate gap, not this fix's scope), so
        # every write is resident under its OWN `item.id` — the ``StmTierRepository.put`` return-
        # idempotency contract (``ports.py``) degrades to a no-op here: always the given id.
        return item.id

    async def get(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> MemoryItem | None:
        """Keyed read, Model-A authorized on a SHARED η — the SAME predicate the Redis adapter
        applies (``storage/authz.py``, AD-129); a security property may not differ by backend."""
        return authorized_item(
            await self._retry(self._get_impl)(ns, memory_id),
            ns=ns,
            caller_identity_set=caller_identity_set,
            operation="stm.get",
        )

    async def _get_impl(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        key = RedisMapper.memory_key(ns, memory_id).encode("utf-8")
        blob = await self._mc.get(key)
        if blob is None:
            return None
        row = RedisRecord(key=key.decode("utf-8"), ttl_s=None, blob=blob.decode("utf-8"))
        return self._mapper.from_store(row)

    async def recent(
        self,
        ns: Namespace,
        *,
        limit: int,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> list[Scored[MemoryItem]]:
        """Recency floor, Model-A filtered on a SHARED η — the SAME predicate the Redis adapter
        applies (``storage/authz.py``, AD-128); a security property may not differ by backend."""
        return await self._retry(self._recent_impl)(
            ns, limit=limit, caller_identity_set=caller_identity_set
        )

    async def _recent_impl(
        self,
        ns: Namespace,
        *,
        limit: int,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> list[Scored[MemoryItem]]:
        key = RedisMapper.recency_key(ns).encode("utf-8")
        out = await self._scan_index(
            key,
            ns,
            limit=limit,
            channel=RecallChannel.STM_FLOOR,
            is_floor=True,
            self_heal=self._update_recency,
        )
        return authorized_window(
            out, ns=ns, caller_identity_set=caller_identity_set, operation="stm.recent"
        )

    async def _scan_index(
        self,
        key: bytes,
        ns: Namespace,
        *,
        limit: int,
        channel: RecallChannel,
        is_floor: bool,
        self_heal: Callable[[Namespace, str, None], Awaitable[None]],
    ) -> list[Scored[MemoryItem]]:
        """The read-then-hydrate-then-self-heal shape :meth:`_recent_impl` and
        :meth:`_demoted_impl` (AD-250 fix, ADR 0061) both need, over WHICHEVER namespace-scoped
        list key the caller names — factored out so the two indices share one implementation
        (DEV-STANDARDS rule 6, DRY)."""
        entries, _ = await self._read_recency(key)
        out: list[Scored[MemoryItem]] = []
        for rank, (memory_id, _ts) in enumerate(entries[: max(0, limit)]):
            # ``_get_impl``, NOT the public ``get`` — the window is authorized ONCE, by the
            # caller's own ``authorized_window`` pass. Hydrating through the public verb would
            # apply Model-A twice and, since this loop reads ``None`` as "TTL-expired, self-heal",
            # a DENIAL would EVICT the row from the index. Same reasoning, same fix, as the Redis
            # adapter.
            item = await self._get_impl(ns, memory_id)
            if item is None:
                # TTL-expired member still lingering in the list — self-heal, same as the Redis
                # adapter's ZSET pruning.
                await self_heal(ns, memory_id, None)
                continue
            out.append(Scored(item=item, score=1.0, channel=channel, rank=rank, is_floor=is_floor))
        return out

    async def put_demoted(self, item: MemoryItem, *, ttl_s: int, at: datetime) -> str:
        """The demotion write-ahead verb (AD-250 fix, ADR 0061 — ``ports.py``'s own docstring has
        the full rationale)."""
        return await self._retry(self._put_demoted_impl)(item, ttl_s=ttl_s, at=at)

    async def _put_demoted_impl(self, item: MemoryItem, *, ttl_s: int, at: datetime) -> str:
        row = self._mapper.to_store(item)
        await self._mc.set(row.key.encode("utf-8"), row.blob.encode("utf-8"), exptime=ttl_s)
        await self._update_demoted(item.namespace, item.id, at.timestamp(), ttl_s=ttl_s)
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
        key = RedisMapper.demoted_key(ns).encode("utf-8")

        async def _self_heal(ns_: Namespace, memory_id: str, _ts: None) -> None:
            await self._update_demoted(ns_, memory_id, None, ttl_s=self._default_ttl_s)

        out = await self._scan_index(
            key,
            ns,
            limit=limit,
            channel=RecallChannel.STM_DEMOTED,
            is_floor=False,
            self_heal=_self_heal,
        )
        return authorized_window(
            out, ns=ns, caller_identity_set=caller_identity_set, operation="stm.demoted"
        )

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        return await self._retry(self._evict_impl)(ns, memory_id)

    async def _evict_impl(self, ns: Namespace, memory_id: str) -> None:
        key = RedisMapper.memory_key(ns, memory_id).encode("utf-8")
        await self._mc.delete(key)
        await self._update_recency(ns, memory_id, None)
        # AD-250 fix (ADR 0061): also strip the DEMOTED list — see `RedisStmAdapter._evict_impl`'s
        # identical comment for why (removing an id absent from a list is a harmless no-op here).
        await self._update_demoted(ns, memory_id, None, ttl_s=self._default_ttl_s)

    async def reinforce(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        at: datetime,
        relevance_score: float | None = None,
    ) -> MemoryItem | None:
        """The read-stat write-back (AD-250 fix, ADR 0061 — ``ports.py``'s own docstring has the
        full rationale; AD-266 added ``relevance_score``/``last_seen``). CAS read-modify-write
        over the SAME row primitive ``put``/``get`` use — parity with ``_update_index_impl``'s
        own CAS loop, for the SAME concurrency reason.

        Memcached exposes no ``KEEPTTL``/remaining-TTL-read primitive (D6, module docstring: "dev
        only has Memcached"), so unlike the Redis/Valkey leg this cannot preserve the row's exact
        REMAINING ttl — it re-stamps this adapter's own configured ``default_ttl_s`` instead
        (the same ``effective_ttl_s`` fallback ``_put_impl`` already uses when no override is
        given). Documented, not silently approximated: this is a real behavioural gap on a
        dev-only backend, not the production (Redis/Valkey) path this fix is proven against."""
        return await self._retry(self._reinforce_impl)(
            ns, memory_id, at=at, relevance_score=relevance_score
        )

    async def _reinforce_impl(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        at: datetime,
        relevance_score: float | None = None,
    ) -> MemoryItem | None:
        key = RedisMapper.memory_key(ns, memory_id).encode("utf-8")
        for _attempt in range(self._cas_max_attempts):
            raw, cas_token = await self._mc.gets(key)
            if raw is None or cas_token is None:
                # `cas_token is None` alongside a live `raw` is not a real memcached outcome
                # (`gets` always pairs a hit with a token), but the stub types it as possible —
                # treated the same as a miss: self-heal, never call `cas` with a token type it
                # cannot accept.
                return None  # expired/evicted between the caller's read and this call.
            current = self._mapper.from_store(
                RedisRecord(key=key.decode("utf-8"), ttl_s=None, blob=raw.decode("utf-8"))
            )
            update: dict[str, object] = {
                "access_count": current.access_count + 1,
                "updated_at": at,
                "last_seen": at,
            }
            if relevance_score is not None:
                update["relevance_score"] = relevance_score
            reinforced = current.model_copy(update=update)
            new_blob = reinforced.model_dump_json().encode("utf-8")
            ok = await self._mc.cas(key, new_blob, cas_token, exptime=self._default_ttl_s)
            if ok:
                return reinforced
        raise TierRepositoryUnavailableError(
            f"memcached reinforce CAS failed after {self._cas_max_attempts} attempts "
            "(sustained write contention) — D6 read-stat write-back unavailable"
        )
