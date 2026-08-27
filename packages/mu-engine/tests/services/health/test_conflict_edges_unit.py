"""``PendingConflictEdgeReader`` — the shipped ``ConflictEdgeReader`` over the real conflict inbox.

Authority: memory-health-pinning-spec §3.3 / §4 line 229 · storage-schema §1.4.

**Why this file exists.** ``ConflictEdges`` is the ONLY input that can raise ``CONFLICTING`` /
``LOW_CONFIDENCE``, and ``ConflictRecord.pin_blocked`` — the §6.4 "a new fact contradicts a
memory you pinned" signal — reaches a user only through this projection. The port had no
implementer, so both were fed exclusively by test literals: the whole §6.4 surface was
unreachable from real data. These tests run the projection over records written by the REAL
``ConflictAdjudicator`` into the REAL ``InMemoryConflictRecordRepository``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.model.conflict import ConflictRecord, ConflictState
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.lifecycle.conflict import InMemoryConflictRecordRepository
from mu_engine.services.health.conflict_edges import PendingConflictEdgeReader

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _record(
    ns: Namespace,
    *,
    conflict_id: str,
    members: tuple[str, ...],
    pin_blocked: bool = False,
    confidence: float = 0.4,
    detected_at: datetime = _T0,
) -> ConflictRecord:
    return ConflictRecord(
        conflict_id=conflict_id,
        namespace=ns,
        member_ids=members,
        predicate_key="prefers",
        method="polarity_cardinality_heuristic",
        detected_confidence=confidence,
        state=ConflictState.MANUAL_PENDING,
        detected_at=detected_at,
        pin_blocked=pin_blocked,
    )


async def test_a_parked_conflict_becomes_an_adjacency_row_for_every_member(
    ns: Namespace,
) -> None:
    repo = InMemoryConflictRecordRepository()
    await repo.add(_record(ns, conflict_id="c1", members=("a", "b"), confidence=0.25))

    edges = await PendingConflictEdgeReader(repo).edges_for(ns, frozenset({"a", "b"}))

    assert edges.unresolved_for("a") is True
    assert edges.unresolved_for("b") is True
    assert edges.peers_for("a") == ("b",)
    assert edges.peers_for("b") == ("a",)
    assert edges.confidence_for("a") == 0.25


async def test_pin_blocked_reaches_the_assessor(ns: Namespace) -> None:
    """The end of the §6.4 wire: adjudicator -> ``ConflictRecord.pin_blocked`` -> this row ->
    ``MemoryHealthFlag.CONFLICTING``."""
    repo = InMemoryConflictRecordRepository()
    await repo.add(
        _record(ns, conflict_id="c1", members=("pinned", "challenger"), pin_blocked=True)
    )

    edges = await PendingConflictEdgeReader(repo).edges_for(ns, frozenset({"pinned"}))

    assert edges.pin_blocked_for("pinned") is True


async def test_an_ordinary_parked_conflict_is_not_pin_blocked(ns: Namespace) -> None:
    """Non-vacuity control: the flag is projected, not stamped on."""
    repo = InMemoryConflictRecordRepository()
    await repo.add(_record(ns, conflict_id="c1", members=("a", "b")))

    edges = await PendingConflictEdgeReader(repo).edges_for(ns, frozenset({"a"}))

    assert edges.pin_blocked_for("a") is False


async def test_the_read_is_bounded_by_the_page_ids(ns: Namespace) -> None:
    repo = InMemoryConflictRecordRepository()
    await repo.add(_record(ns, conflict_id="c1", members=("a", "b")))
    await repo.add(_record(ns, conflict_id="c2", members=("y", "z")))

    edges = await PendingConflictEdgeReader(repo).edges_for(ns, frozenset({"a"}))

    assert set(edges.rows_by_memory) == {"a"}


async def test_an_empty_page_does_no_io(ns: Namespace) -> None:
    class _Exploding:
        async def add(self, record: ConflictRecord) -> None: ...
        async def get(self, ns: Namespace, conflict_id: str) -> ConflictRecord | None: ...
        async def upsert(self, record: ConflictRecord) -> None: ...
        async def pending(self, ns: Namespace) -> list[ConflictRecord]:
            raise AssertionError("the reader must not query for an empty page")

    edges = await PendingConflictEdgeReader(_Exploding()).edges_for(ns, frozenset())  # type: ignore[arg-type]

    assert edges.rows_by_memory == {}


async def test_a_foreign_partitions_conflicts_never_appear(ns: Namespace) -> None:
    """The repository keys on ``to_prefix()``, so a record parked in another partition is
    invisible even when it names the same memory id."""
    foreign = Namespace(
        org="org1", workspace="ws1", user="INTRUDER", session="s1", visibility=Visibility.PRIVATE
    )
    repo = InMemoryConflictRecordRepository()
    await repo.add(_record(foreign, conflict_id="c1", members=("a", "b")))

    edges = await PendingConflictEdgeReader(repo).edges_for(ns, frozenset({"a"}))

    assert edges.rows_by_memory == {}


async def test_a_pin_blocked_record_wins_over_a_newer_ordinary_one(ns: Namespace) -> None:
    """``ConflictEdges`` holds ONE row per memory. Losing the pin-blocked record to a merely-newer
    conflict would drop the only signal that tells the owner their pin is holding something back.
    """
    repo = InMemoryConflictRecordRepository()
    await repo.add(
        _record(ns, conflict_id="c_pin", members=("a", "b"), pin_blocked=True, detected_at=_T0)
    )
    await repo.add(
        _record(ns, conflict_id="c_new", members=("a", "z"), detected_at=_T0 + timedelta(days=5))
    )

    edges = await PendingConflictEdgeReader(repo).edges_for(ns, frozenset({"a"}))

    assert edges.pin_blocked_for("a") is True
    assert edges.peers_for("a") == ("b",)


async def test_two_ordinary_conflicts_collapse_to_the_newest_deterministically(
    ns: Namespace,
) -> None:
    repo = InMemoryConflictRecordRepository()
    await repo.add(_record(ns, conflict_id="c_old", members=("a", "old"), detected_at=_T0))
    await repo.add(
        _record(ns, conflict_id="c_new", members=("a", "new"), detected_at=_T0 + timedelta(days=1))
    )

    edges = await PendingConflictEdgeReader(repo).edges_for(ns, frozenset({"a"}))

    assert edges.peers_for("a") == ("new",)


# ── end-to-end over the REAL adjudicator: a pinned target really does light up CONFLICTING ──
async def test_the_adjudicators_own_pin_block_surfaces_as_conflicting(ns: Namespace) -> None:
    """No literals: the record under test is the one ``ConflictAdjudicator`` writes when it
    refuses to make a pinned item the automatic loser."""
    from mu_contracts.domain.model.health import MemoryHealthFlag
    from mu_engine.lifecycle.conflict import ConflictAdjudicator
    from mu_engine.platform.clock import FrozenClock
    from mu_engine.services.health.assessor import HeuristicV1Assessor
    from mu_engine.services.health.settings import HealthSettings
    from mu_engine.storage.domain.memory import (
        MemoryItem as EngineItem,
    )
    from mu_engine.storage.domain.memory import (
        MemoryState,
        MemoryTier,
    )
    from mu_engine.storage.domain.namespace import Namespace as EngineNamespace
    from mu_engine.storage.domain.namespace import Visibility as EngineVisibility

    from .conftest import T0 as CONFTEST_T0

    engine_ns = EngineNamespace(
        org=ns.org,
        workspace=ns.workspace,
        user=ns.user,
        session=ns.session,
        visibility=EngineVisibility.PRIVATE,
    )

    def _engine_item(memory_id: str, *, pinned: bool, created_at: datetime) -> EngineItem:
        return EngineItem(
            id=memory_id,
            content="c",
            namespace=engine_ns,
            owner_id=ns.user,
            workspace_id=ns.workspace,
            session_id=ns.session,
            tier=MemoryTier.LTM,
            state=MemoryState.ACTIVE,
            subject="user",
            predicate="prefers",
            object="dark mode",
            pinned=pinned,
            created_at=created_at,
            valid_at=created_at,
        )

    repo = InMemoryConflictRecordRepository()
    adjudicator = ConflictAdjudicator(
        router=None, clock=FrozenClock(_T0 + timedelta(days=10)), conflict_records=repo
    )
    verdict = await adjudicator.adjudicate(
        ns=ns,
        winner=_engine_item("pinned_target", pinned=True, created_at=_T0),
        candidate=_engine_item("challenger", pinned=False, created_at=_T0 + timedelta(days=1)),
        heuristic_contradicts=True,
        budget=adjudicator.new_budget(),
    )
    assert verdict.pin_blocked is True

    edges = await PendingConflictEdgeReader(repo).edges_for(ns, frozenset({"pinned_target"}))
    item = _pinned_contract_item(ns, CONFTEST_T0)
    flags = HeuristicV1Assessor(HealthSettings()).assess(
        item, now=CONFTEST_T0, conflict_edges=edges
    )

    assert edges.pin_blocked_for("pinned_target") is True
    assert MemoryHealthFlag.CONFLICTING in flags
    assert MemoryHealthFlag.PINNED in flags


def _pinned_contract_item(ns: Namespace, at: datetime):  # type: ignore[no-untyped-def]
    from mu_contracts.domain.model.memory import MemoryItem, MemoryKind, Tier, Validity

    return MemoryItem(
        id="pinned_target",
        namespace=ns,
        kind=MemoryKind.PROPOSITION,
        content="the user prefers dark mode",
        tier=Tier.LTM,
        validity=Validity(valid_at=at, recorded_at=at),
        last_seen=at,
        pinned=True,
        provenance_id="prov_pinned_target",
    )
