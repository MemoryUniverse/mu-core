"""``RedisMapper`` — MemoryItem <-> RedisRecord (STM), LOSSLESS by JSON blob.

PORT of the key families + ``SETEX``/``chash`` seam from the prototype
``/home/user/hackathon/memory_universe/shared/stores/stm_redis.py:52,96-97,226-230``,
re-homed to the ``mu/…:stm:…`` key catalog (``storage-indexing §1.1``).

Round-trip fidelity (spec §5 contract 1): the blob is ``MemoryItem`` JSON so every field
survives; id-stability (contract 2): the key is derived deterministically from
``item.id``; tenancy (contract 3): keyed under ``Namespace.to_prefix()``.
"""

from __future__ import annotations

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.ports import RedisRecord

__all__ = ["RedisMapper"]

_KEY_PREFIX = "mu"


class RedisMapper:
    """Implements ``RowMapper[RedisRecord]`` (spec §5)."""

    def __init__(self, *, default_ttl_s: int = 3600) -> None:
        # TTL is a knob, not a hardcode at a call-site — defaulted here, overridable.
        self.default_ttl_s = default_ttl_s

    @staticmethod
    def memory_key(ns: Namespace, memory_id: str) -> str:
        """Deterministic id-stable key: ``mu/{to_prefix}:stm:mem:{id}``."""
        return f"{_KEY_PREFIX}/{ns.to_prefix()}:stm:mem:{memory_id}"

    @staticmethod
    def recency_key(ns: Namespace) -> str:
        return f"{_KEY_PREFIX}/{ns.to_prefix()}:stm:recency"

    def to_store(self, item: MemoryItem) -> RedisRecord:
        return RedisRecord(
            key=self.memory_key(item.namespace, item.id),
            ttl_s=self.default_ttl_s,
            blob=item.model_dump_json(),
        )

    def from_store(self, row: RedisRecord) -> MemoryItem:
        return MemoryItem.model_validate_json(row.blob)
