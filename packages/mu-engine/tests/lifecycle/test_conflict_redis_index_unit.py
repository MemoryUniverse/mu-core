"""``RedisConflictRecordRepository``'s two INDEXES, offline.

The durable adapter's ``pending`` index decides what the conflict inbox can see, and its
``apply`` index decides what a crashed resolve can recover. Both are pure keyspace bookkeeping
inside ``_write_impl``, so both are testable without a server — and the one that shipped was
wrong in a way the integration suite (which never reopens a record) could not catch: a write
``SREM``ed the id out of the pending set and re-``SADD``ed it only for ``MANUAL_PENDING``, so a
``reopen`` actively EVICTED the record from the only index the inbox reads.

The fake below implements exactly the six Redis calls this adapter makes. It is not a mock of the
repository — the repository under test is the real one; only the transport is in-process, the
same way the in-memory tier adapters stand in for stores elsewhere. The real-Valkey behaviour is
covered by ``test_conflict_redis_int.py``; this pins the INDEX LOGIC, which is ours, not Redis's.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from mu_contracts.domain.model.conflict import (
    ConflictRecord,
    ConflictResolutionKind,
    ConflictState,
    ResolutionOrigin,
)
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.lifecycle.conflict_redis import RedisConflictRecordRepository

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


class _FakePipeline:
    def __init__(self, store: _FakeRedis) -> None:
        self._store = store
        self._ops: list[tuple[str, tuple[Any, ...]]] = []

    async def __aenter__(self) -> _FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def set(self, key: str, value: str) -> None:
        self._ops.append(("set", (key, value)))

    def srem(self, key: str, member: str) -> None:
        self._ops.append(("srem", (key, member)))

    def sadd(self, key: str, member: str) -> None:
        self._ops.append(("sadd", (key, member)))

    async def execute(self) -> None:
        for op, args in self._ops:
            getattr(self._store, f"_{op}")(*args)
        self._ops.clear()


class _FakeRedis:
    """The six-call subset ``RedisConflictRecordRepository`` uses. No behaviour of its own."""

    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    def pipeline(self, transaction: bool = False) -> _FakePipeline:
        return _FakePipeline(self)

    def _set(self, key: str, value: str) -> None:
        self.strings[key] = value

    def _srem(self, key: str, member: str) -> None:
        self.sets.setdefault(key, set()).discard(member)

    def _sadd(self, key: str, member: str) -> None:
        self.sets.setdefault(key, set()).add(member)

    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _record(ns: Namespace, state: ConflictState, **kw: object) -> ConflictRecord:
    base: dict[str, object] = {
        "conflict_id": "c1",
        "namespace": ns,
        "member_ids": ("a", "b"),
        "predicate_key": "lives_in",
        "method": "nli",
        "detected_confidence": 0.6,
        "state": state,
        "detected_at": _T0,
    }
    base.update(kw)
    return ConflictRecord(**base)  # type: ignore[arg-type]


@pytest.fixture
def repo() -> tuple[RedisConflictRecordRepository, _FakeRedis]:
    fake = _FakeRedis()
    return RedisConflictRecordRepository(fake, key_prefix="mu:t"), fake  # type: ignore[arg-type]


async def test_a_reopened_record_stays_in_the_pending_index(
    repo: tuple[RedisConflictRecordRepository, _FakeRedis], ns: Namespace
) -> None:
    """The shipped write evicted it: SREM always, SADD back only for MANUAL_PENDING. So
    ``reopen`` removed the conflict from the inbox's only index — and ``_pending_impl``'s
    defensive re-check filtered on MANUAL_PENDING too, so even a stale index entry would have
    been dropped. Spec §5 line 197 defines pending as MANUAL_PENDING or REOPENED."""
    store, _ = repo
    await store.add(_record(ns, ConflictState.MANUAL_PENDING))
    assert [r.conflict_id for r in await store.pending(ns)] == ["c1"]

    await store.upsert(_record(ns, ConflictState.REOPENED))
    assert [r.conflict_id for r in await store.pending(ns)] == ["c1"]


async def test_a_settled_record_leaves_the_pending_index(
    repo: tuple[RedisConflictRecordRepository, _FakeRedis], ns: Namespace
) -> None:
    store, _ = repo
    await store.add(_record(ns, ConflictState.MANUAL_PENDING))
    await store.upsert(
        _record(
            ns,
            ConflictState.RESOLVED,
            resolution_kind=ConflictResolutionKind.SUPERSEDE,
            resolved_winner_id="b",
            resolution_origin=ResolutionOrigin.MANUAL,
            resolution_applied_at=_T0,
        )
    )
    assert await store.pending(ns) == []


async def test_an_unapplied_decision_is_recoverable_from_the_apply_index(
    repo: tuple[RedisConflictRecordRepository, _FakeRedis], ns: Namespace
) -> None:
    """The durable half of the resolve hand-off: the decision must be findable again after the
    process that accepted it is gone."""
    store, _ = repo
    await store.upsert(
        _record(
            ns,
            ConflictState.RESOLVED,
            resolution_kind=ConflictResolutionKind.SUPERSEDE,
            resolved_winner_id="b",
            resolution_origin=ResolutionOrigin.MANUAL,
        )
    )
    assert [r.conflict_id for r in await store.awaiting_apply(ns)] == ["c1"]

    await store.upsert(
        _record(
            ns,
            ConflictState.RESOLVED,
            resolution_kind=ConflictResolutionKind.SUPERSEDE,
            resolved_winner_id="b",
            resolution_origin=ResolutionOrigin.MANUAL,
            resolution_applied_at=_T0,
        )
    )
    assert await store.awaiting_apply(ns) == []


async def test_both_indexes_are_namespace_scoped(
    repo: tuple[RedisConflictRecordRepository, _FakeRedis], ns: Namespace
) -> None:
    """CLAUDE.md rule 4 — the key carries ``ns.to_prefix()``, so another tenant sees neither."""
    store, _ = repo
    other = ns.model_copy(update={"user": "u2"})
    await store.add(_record(ns, ConflictState.MANUAL_PENDING))
    assert await store.pending(other) == []
    assert await store.awaiting_apply(other) == []
