"""``RedisConflictRecordRepository`` — REAL mu-dev-cache (Valkey), ZERO mocks.

Authority: build-plan §4 Stage-C C1 (this repository's own build task). Mirrors
``tests/services/test_ingest_int.py``'s ``RedisStageLedger`` fixture pattern — a live client from
the central ``Settings`` tree, fail-loud probe, per-test-unique namespace for isolation.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from redis.asyncio import Redis

from mu_contracts.domain.model.conflict import ConflictRecord, ConflictState, ResolutionOrigin
from mu_engine.lifecycle.conflict_redis import RedisConflictRecordRepository

pytestmark = pytest.mark.integration


@pytest.fixture
def make_record(make_ns: Callable[..., object]) -> Callable[..., ConflictRecord]:
    def _make(
        *,
        conflict_id: str | None = None,
        state: ConflictState = ConflictState.DETECTED,
        ns: object | None = None,
    ) -> ConflictRecord:
        namespace = ns or make_ns()
        return ConflictRecord(
            conflict_id=conflict_id or uuid.uuid4().hex[:24],
            namespace=namespace,
            member_ids=("mem-a", "mem-b"),
            predicate_key="likes",
            method="nli",
            detected_confidence=0.87,
            state=state,
            detected_at=datetime.now(UTC),
        )

    return _make


async def test_add_then_get_roundtrip(
    valkey_client: Redis,
    make_record: Callable[..., ConflictRecord],
    uid: str,
) -> None:
    repo = RedisConflictRecordRepository(valkey_client, key_prefix=f"mu:test-conflict:{uid}")
    record = make_record()

    await repo.add(record)
    got = await repo.get(record.namespace, record.conflict_id)

    assert got is not None
    assert got == record  # lossless JSON round-trip on the REAL Valkey server


async def test_get_missing_returns_none(
    valkey_client: Redis,
    make_ns: Callable[..., object],
    uid: str,
) -> None:
    repo = RedisConflictRecordRepository(valkey_client, key_prefix=f"mu:test-conflict:{uid}")
    assert await repo.get(make_ns(), "does-not-exist") is None


async def test_pending_indexes_only_manual_pending_state(
    valkey_client: Redis,
    make_record: Callable[..., ConflictRecord],
    make_ns: Callable[..., object],
    uid: str,
) -> None:
    repo = RedisConflictRecordRepository(valkey_client, key_prefix=f"mu:test-conflict:{uid}")
    ns = make_ns()
    pending_rec = make_record(ns=ns, state=ConflictState.MANUAL_PENDING)
    resolved_rec = make_record(ns=ns, state=ConflictState.RESOLVED)

    await repo.add(pending_rec)
    await repo.add(resolved_rec)

    pending = await repo.pending(ns)
    assert {r.conflict_id for r in pending} == {pending_rec.conflict_id}
    assert all(r.state is ConflictState.MANUAL_PENDING for r in pending)


async def test_pending_index_drops_record_that_transitions_out(
    valkey_client: Redis,
    make_record: Callable[..., ConflictRecord],
    make_ns: Callable[..., object],
    uid: str,
) -> None:
    repo = RedisConflictRecordRepository(valkey_client, key_prefix=f"mu:test-conflict:{uid}")
    ns = make_ns()
    conflict_id = uuid.uuid4().hex[:24]
    pending_rec = make_record(ns=ns, conflict_id=conflict_id, state=ConflictState.MANUAL_PENDING)
    await repo.add(pending_rec)
    assert {r.conflict_id for r in await repo.pending(ns)} == {conflict_id}

    resolved_rec = pending_rec.model_copy(
        update={
            "state": ConflictState.RESOLVED,
            "resolved_winner_id": "mem-a",
            "resolution_origin": ResolutionOrigin.MANUAL,
            "resolved_at": datetime.now(UTC),
        }
    )
    await repo.upsert(resolved_rec)

    assert await repo.pending(ns) == []
    got = await repo.get(ns, conflict_id)
    assert got is not None
    assert got.state is ConflictState.RESOLVED


async def test_durability_across_fresh_repository_instance(
    valkey_client: Redis,
    make_record: Callable[..., ConflictRecord],
    uid: str,
) -> None:
    """A second repository instance, built fresh over the SAME live Valkey connection, reads back
    what the first instance wrote — proves the write actually persisted server-side rather than
    living in any process-local cache (the durability contract this adapter exists to satisfy)."""
    prefix = f"mu:test-conflict:{uid}"
    repo_a = RedisConflictRecordRepository(valkey_client, key_prefix=prefix)
    record = make_record(state=ConflictState.MANUAL_PENDING)

    await repo_a.add(record)
    del repo_a  # drop the writer instance entirely — nothing process-local can leak through

    repo_b = RedisConflictRecordRepository(valkey_client, key_prefix=prefix)
    got = await repo_b.get(record.namespace, record.conflict_id)
    assert got == record

    pending = await repo_b.pending(record.namespace)
    assert {r.conflict_id for r in pending} == {record.conflict_id}
