"""``RedisConflictRecordRepository`` — the durable ``ConflictRecordRepository`` over Valkey/Redis.

Authority: engine-core-spec.md §7.6 (``ConflictRecordRepository`` surface,
``mu_contracts.ports.governance``), build-plan §4 Stage-C item C1. There is NO existing durable
adapter for this port — only :class:`~mu_engine.lifecycle.conflict.InMemoryConflictRecordRepository`
(the LOCAL-plane in-process default, ``conflict.py:244``). This module is the from-scratch SHARED/
durable counterpart the mu-engine-server composition root wires when a persistent conflict inbox is
required (spec §2.9), mirroring :class:`~mu_engine.pipelines.ledger.RedisStageLedger`'s keyspace +
``retry_io`` discipline exactly (PIPELINES-design.md §2.4) — same per-instance DI-threaded retry
wrapper, same "content is a JSON blob under one key" shape, same fail-loud-on-unknown-container
posture (no silent mock fallback).

**Keyspace.** Two key families under one ``key_prefix`` (default ``mu:conflict``):

- ``{prefix}:record:{ns.to_prefix()}:{conflict_id}`` — a ``STRING`` holding
  ``ConflictRecord.model_dump_json()``. One record, one key (mirrors the ledger's one-key-per-stage
  shape) — the whole record round-trips through pydantic, never a hand-rolled field mapping.
- ``{prefix}:pending:{ns.to_prefix()}`` — a ``SET`` of ``conflict_id`` members: the inbox index the
  ``pending()`` query (state=``MANUAL_PENDING``) reads, so that call is an ``SMEMBERS`` + N record
  ``GET``s instead of an unbounded per-namespace scan. The index is kept in sync on every
  ``add``/``upsert``: the write always ``SREM``s first, then ``SADD``s back only when the record's
  CURRENT state is ``MANUAL_PENDING`` — a record that transitions OUT of pending (resolved/
  dismissed/reopened-then-resolved) drops out of the index on its very next write, never left
  stale. Record write + index update are issued inside one ``MULTI``/``EXEC`` pipeline (redis-py's
  ``transaction=True`` pipeline) so the two keys never observe a torn intermediate state.

``add`` and ``upsert`` share ONE implementation (``_write``) — same behavior as
``InMemoryConflictRecordRepository`` (``conflict.py:256-260``, ``upsert`` delegates to ``add``): the
port draws no idempotency distinction between the two for this repository (contrast
``GrantRepository.add``'s explicit put-if-absent contract), so neither does this adapter.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis

from mu_contracts.domain.model.conflict import ConflictRecord, ConflictState
from mu_contracts.domain.model.memory import Namespace
from mu_engine.platform.decorators import retry_io

__all__ = ["ConflictRedisSettings", "RedisConflictRecordRepository"]


class ConflictRedisSettings(BaseModel):
    """``RedisConflictRecordRepository`` I/O knobs — same seam shape as
    ``mu_engine.pipelines.ledger.LedgerSettings`` (a tracked, DI-threaded default; DEV-STANDARDS
    rule 3 forbids a bare hardcoded timeout inside the decorated methods themselves)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Per-attempt I/O budget (DEV-STANDARDS async sharpener: "timeouts on every external call").
    redis_io_timeout_s: float = 5.0


class RedisConflictRecordRepository:
    """Durable ``ConflictRecordRepository`` over Valkey/Redis (engine-core §7.6 port).

    Every public method is wrapped by :func:`retry_io` (transient-only retry + per-attempt
    timeout, same predicate ``RedisStageLedger`` uses) so none of the four port methods is
    unbounded I/O. Structurally satisfies ``mu_contracts.ports.governance.ConflictRecordRepository``
    (``runtime_checkable`` Protocol) — no explicit inheritance needed.
    """

    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str = "mu:conflict",
        settings: ConflictRedisSettings | None = None,
    ) -> None:
        self._redis = client
        self._prefix = key_prefix
        # per-instance, DI-threaded retry wrapper (never a class-level decorator baking in a
        # module constant) so `redis_io_timeout_s` is genuinely tunable from Settings.
        self._settings = settings or ConflictRedisSettings()
        self._retry = retry_io(timeout_s=self._settings.redis_io_timeout_s)

    def _record_key(self, ns: Namespace, conflict_id: str) -> str:
        return f"{self._prefix}:record:{ns.to_prefix()}:{conflict_id}"

    def _pending_key(self, ns: Namespace) -> str:
        return f"{self._prefix}:pending:{ns.to_prefix()}"

    # --------------------------------------------------------------------------------- add/upsert
    async def add(self, record: ConflictRecord) -> None:
        return await self._retry(self._write_impl)(record)

    async def upsert(self, record: ConflictRecord) -> None:
        # Same behavior as `add` — mirrors InMemoryConflictRecordRepository.upsert (conflict.py
        # `async def upsert(self, record): await self.add(record)`); the port draws no
        # idempotency distinction between the two for THIS repository.
        return await self._retry(self._write_impl)(record)

    async def _write_impl(self, record: ConflictRecord) -> None:
        record_key = self._record_key(record.namespace, record.conflict_id)
        pending_key = self._pending_key(record.namespace)
        payload = record.model_dump_json()
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(record_key, payload)
            # Always SREM first — a record that transitioned OUT of pending must not linger in
            # the index; SADD back in only if its CURRENT state is still pending.
            pipe.srem(pending_key, record.conflict_id)
            if record.state is ConflictState.MANUAL_PENDING:
                pipe.sadd(pending_key, record.conflict_id)
            await pipe.execute()

    # ----------------------------------------------------------------------------------------- get
    async def get(self, ns: Namespace, conflict_id: str) -> ConflictRecord | None:
        return await self._retry(self._get_impl)(ns, conflict_id)

    async def _get_impl(self, ns: Namespace, conflict_id: str) -> ConflictRecord | None:
        raw = await self._redis.get(self._record_key(ns, conflict_id))
        if raw is None:
            return None
        blob = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return ConflictRecord.model_validate_json(blob)

    # ------------------------------------------------------------------------------------- pending
    async def pending(self, ns: Namespace) -> list[ConflictRecord]:
        return await self._retry(self._pending_impl)(ns)

    async def _pending_impl(self, ns: Namespace) -> list[ConflictRecord]:
        raw_ids = await self._redis.smembers(self._pending_key(ns))  # type: ignore[misc]
        ids = {i.decode("utf-8") if isinstance(i, bytes) else i for i in raw_ids}
        out: list[ConflictRecord] = []
        for conflict_id in ids:
            raw = await self._redis.get(self._record_key(ns, conflict_id))
            if raw is None:
                continue  # index/record race (e.g. concurrent write) — skip, never fabricate
            blob = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            rec = ConflictRecord.model_validate_json(blob)
            if rec.state is ConflictState.MANUAL_PENDING:  # defensive re-check, index is a hint
                out.append(rec)
        return out
