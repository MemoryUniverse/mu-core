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

# Genuine structural constant, NOT a tunable (DEV-STANDARDS rule 3 exemption): this is the fixed
# on-wire key-catalog root (storage-indexing §1.1), shared byte-for-byte with every other
# component that reads/writes ``mu/...`` keys. Making it Settings-driven would let one process
# silently diverge from the documented key format everyone else assumes — it is a protocol
# constant like an enum value, not an operator-facing knob.
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

    @staticmethod
    def demoted_key(ns: Namespace) -> str:
        """AD-250 fix (ADR 0061): a namespace-scoped index of currently-resident DEMOTED STM
        rows, SEPARATE from :meth:`recency_key`. A demoted MTM->STM write-ahead copy
        (:class:`~mu_engine.lifecycle.demotion.DemotionService`) is a fundamentally different
        thing from a fresh capture — it was demoted precisely for LOW recency/usage, so sharing
        the ordinary recency ZSET's bounded, newest-`recency_floor_limit`-only window means it is
        pushed out by every genuinely new turn in the session almost immediately (FAULT-HUNT-0924
        F1 / AD-250: "the STM recency ZSET is scored by item.created_at ... a demoted memory
        enters the recency floor at its original age, underneath every genuinely recent turn").
        A demoted item is conceptually its own small, slowly-growing population — a COLD-ish
        sub-tier still living in the KV store, not a member of "what was just said" — so it gets
        its OWN discoverability index instead of competing for the fresh-capture floor's slots."""
        return f"{_KEY_PREFIX}/{ns.to_prefix()}:stm:demoted"

    @staticmethod
    def content_hash_key(ns: Namespace) -> str:
        """D4 write-time dedup index (conformance D-8): a namespace-scoped Redis HASH mapping
        ``content_hash -> memory_id`` (one field per distinct content seen in this partition's STM
        window). Lets :class:`~mu_engine.storage.adapters.redis_stm.RedisStmAdapter` detect "this
        exact content already has a live STM row" in O(1) — ``ports.py``'s ``StmTierRepository``
        docstring promise ("Recency floor + TTL + chash dedup") this key finally implements."""
        return f"{_KEY_PREFIX}/{ns.to_prefix()}:stm:chash"

    @staticmethod
    def user_registry_key() -> str:
        """AD-268 fix (ADR 0075) — the durable, **cross-namespace** user-prefix registry.

        Deliberately the ONLY key in this class with no ``ns.to_prefix()`` component: every other
        key here is scoped INSIDE one tenant's partition (the whole point of ``mu/{to_prefix}:…``),
        but this registry's entire job is to be readable BEFORE any specific namespace is known —
        "which user prefixes have STM data at all" is a question a per-namespace key cannot
        answer. ``mu:registry:…`` (colon-joined, no ``/``) is the same "global, not a tenant
        partition" shape ``storage-models-design.md``'s (un-ported) ``mu:session_last_activity``
        already reserves, so this key can never collide with a real ``ns.to_prefix()`` value
        (``Namespace._no_separator_injection`` forbids ``:`` inside a component, and every
        per-tenant key here starts ``mu/…`` — a ``/`` immediately after ``mu``, not a ``:``).

        Holds a single ZSET, ``member=str(UserPrefix)``, ``score=`` last-write unix timestamp —
        deliberately NOT per-org/workspace-sharded: a single local daemon's whole STM store is
        small enough (PROTOTYPE-DEBT-0924.md D5) that one bounded ``ZREVRANGE`` answers "every
        user prefix this store has ever written for" in one round trip, which is exactly what
        :class:`~mu_client.daemon.maintenance.MaintenanceLoop` needs to rebuild its active-user
        registry on a restart instead of starting from empty (the defect this key exists to fix).
        """
        return f"{_KEY_PREFIX}:registry:user_prefix"

    def to_store(self, item: MemoryItem) -> RedisRecord:
        return RedisRecord(
            key=self.memory_key(item.namespace, item.id),
            ttl_s=self.default_ttl_s,
            blob=item.model_dump_json(),
        )

    def from_store(self, row: RedisRecord) -> MemoryItem:
        return MemoryItem.model_validate_json(row.blob)
