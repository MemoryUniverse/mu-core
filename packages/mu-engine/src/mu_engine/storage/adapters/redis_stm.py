"""Redis/Valkey KV adapter — the STM tier repository (``storage-pluggable §3.2``).

PORT of the ``SETEX`` payload + recency ``ZSET`` + scope-in-every-key defence from the
prototype ``/home/user/hackathon/memory_universe/shared/stores/stm_redis.py:96-97,226-230``,
re-homed to the ``mu/…:stm:…`` key catalog via :class:`RedisMapper`.

Fully async (``redis.asyncio``); no blocking in the loop (DEV-STANDARDS rule 1). Tenancy is
enforced by :meth:`RedisMapper.memory_key` prefixing every key with ``Namespace.to_prefix()``
(CANONICAL §1 rule 5). Field-projection is delegated to the mapper (spec Contract-change 5).
"""

from __future__ import annotations

from redis.asyncio import Redis

from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import RecallChannel, Scored
from mu_engine.storage.mappers.redis_mapper import RedisMapper

__all__ = ["RedisStmAdapter"]

# Constructor DEFAULT only (DEV-STANDARDS rule 3: no hardcoded constant lives in adapter LOGIC).
# The live value is DI-threaded from the central Settings tree (``RedisSettings``/
# ``ValkeySettings`` ``.store_io_timeout_s``) by the ``STORE_REGISTRY`` factory
# (``mu_engine.storage.factories._build_redis``/``_build_valkey``); a bare
# ``RedisStmAdapter(client)`` (e.g. in a unit test) still gets a sane, named default.
_DEFAULT_STORE_IO_TIMEOUT_S = 5.0


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
    ) -> None:
        self._redis = client
        self._mapper = mapper or RedisMapper()
        self._retry = retry_io(timeout_s=store_io_timeout_s)

    async def put(self, item: MemoryItem) -> None:
        return await self._retry(self._put_impl)(item)

    async def _put_impl(self, item: MemoryItem) -> None:
        row = self._mapper.to_store(item)
        recency = RedisMapper.recency_key(item.namespace)
        # ONE atomic transaction (MINOR-6): SET payload + ZADD recency + EXPIRE apply together
        # or not at all — no torn state where the payload exists without its recency member.
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(row.key, row.blob, ex=row.ttl_s)
        pipe.zadd(recency, {item.id: item.created_at.timestamp()})
        if row.ttl_s:
            pipe.expire(recency, row.ttl_s)
        await pipe.execute()

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return await self._retry(self._get_impl)(ns, memory_id)

    async def _get_impl(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        key = RedisMapper.memory_key(ns, memory_id)
        blob = await self._redis.get(key)
        if blob is None:
            return None
        from mu_engine.storage.ports import RedisRecord

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

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        return await self._retry(self._evict_impl)(ns, memory_id)

    async def _evict_impl(self, ns: Namespace, memory_id: str) -> None:
        pipe = self._redis.pipeline(transaction=True)
        pipe.delete(RedisMapper.memory_key(ns, memory_id))
        pipe.zrem(RedisMapper.recency_key(ns), memory_id)
        await pipe.execute()


def _as_str(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value
