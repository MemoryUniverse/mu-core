"""``ConflictDetected`` has a PRODUCER — both lanes, once per genuine detection.

``conflict-resolution-async-design.md`` §2's stage table binds one sentence to
``DetectConflictsStage``: *"Emits ``ConflictDetected`` and opens/updates a ``ConflictRecord``"*.
``_open_record``'s own docstring QUOTED that sentence and then implemented only its second half —
the class was declared in the frozen catalog (``mu_contracts.domain.events.ConflictDetected``) and
constructed in no ``src/`` in any repository, so the single event that says "a contradiction
exists" was silent vocabulary. An audit tap, a metering tap or a UI timeline written against it
could not distinguish "this deployment never conflicts" from "nobody ever emits it".

Covers: §2 table (both halves, BOTH lanes — §1.1 "the AUTOMATIC lane opens a record too"), §3
lines 104-106 (idempotent re-detection ⇒ no per-tick spam), CANONICAL §3.1 (content-free).
Offline: no router, no store, no network, no keys.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.events import (
    ConflictDetected,
    ConflictResolutionPending,
    DomainEvent,
)
from mu_contracts.domain.model.conflict import ConflictResolutionMode, ConflictState
from mu_contracts.domain.model.memory import Namespace as ContractNamespace
from mu_contracts.domain.model.memory import Visibility as ContractVisibility
from mu_engine.lifecycle.conflict import (
    AdjudicationKind,
    AdjudicationVerdict,
    ConflictAdjudicator,
    ConflictResolutionPolicy,
    InMemoryConflictRecordRepository,
)
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture
def ns() -> ContractNamespace:
    return ContractNamespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=ContractVisibility.PRIVATE
    )


@pytest.fixture
def engine_ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)

    def detected(self) -> list[ConflictDetected]:
        return [e for e in self.events if isinstance(e, ConflictDetected)]

    def pending(self) -> list[ConflictResolutionPending]:
        return [e for e in self.events if isinstance(e, ConflictResolutionPending)]


def _fact(ns: Namespace, *, memory_id: str, obj: str, created_at: datetime) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=f"the user lives in {obj}",
        namespace=ns,
        owner_id="u1",
        workspace_id="ws1",
        session_id="s1",
        tier=MemoryTier.LTM,
        state=MemoryState.ACTIVE,
        created_at=created_at,
        valid_at=created_at,
        subject="user",
        predicate="lives_in",
        object=obj,
    )


def _adjudicator(
    records: InMemoryConflictRecordRepository,
    bus: _RecordingBus,
    *,
    clock: FrozenClock,
    mode: ConflictResolutionMode,
) -> ConflictAdjudicator:
    return ConflictAdjudicator(
        router=None,  # LLM off ⇒ the deterministic heuristic floor decides; no network
        policy=ConflictResolutionPolicy(mode=mode),
        clock=clock,
        bus=bus,
        conflict_records=records,
    )


async def _sweep(
    adj: ConflictAdjudicator, ns: ContractNamespace, winner: MemoryItem, cand: MemoryItem
) -> AdjudicationVerdict:
    return await adj.adjudicate(
        ns=ns,
        winner=winner,
        candidate=cand,
        heuristic_contradicts=True,
        budget=adj.new_budget(),
    )


# ═══════════════ 1. THE MANUAL (PARKED) LANE EMITS DETECTED, NOT ONLY PENDING ════════════════
async def test_manual_lane_emits_conflict_detected_beside_the_pending_bookend(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    clock = FrozenClock(_T0)
    bus, records = _RecordingBus(), InMemoryConflictRecordRepository()
    adj = _adjudicator(records, bus, clock=clock, mode=ConflictResolutionMode.MANUAL)
    winner = _fact(engine_ns, memory_id="m-new", obj="berlin", created_at=_T0)
    cand = _fact(engine_ns, memory_id="m-old", obj="paris", created_at=_T0 - timedelta(days=1))

    verdict = await _sweep(adj, ns, winner, cand)

    assert verdict.apply is False
    detected = bus.detected()
    assert len(detected) == 1, "the parked lane must announce the detection itself"
    assert detected[0].namespace == ns
    assert detected[0].incoming_id == "m-new"
    assert detected[0].candidate_ids == ["m-old"]
    # `method` is the record's own provenance token, so event and audit row cannot disagree.
    assert detected[0].method == "polarity_cardinality_heuristic"
    stored = await records.get(engine_ns, detected_conflict_id(records))
    assert stored is not None and stored.method == detected[0].method
    assert len(bus.pending()) == 1  # the bookend still fires, and is a DIFFERENT event


def detected_conflict_id(records: InMemoryConflictRecordRepository) -> str:
    (only,) = list(records._records.values())
    return only.conflict_id


# ══════════ 2. THE AUTOMATIC LANE EMITS DETECTED AND *NOT* THE "WAITING FOR YOU" EVENT ═══════
async def test_automatic_lane_emits_detected_but_never_pending(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """§1.1: the automatic lane opens a record too — so it detected a conflict too. But nothing
    is waiting for a human, so the inbox bookend must stay silent (a phantom inbox row)."""
    clock = FrozenClock(_T0)
    bus, records = _RecordingBus(), InMemoryConflictRecordRepository()
    adj = _adjudicator(records, bus, clock=clock, mode=ConflictResolutionMode.AUTOMATIC)
    winner = _fact(engine_ns, memory_id="m-new", obj="berlin", created_at=_T0)
    cand = _fact(engine_ns, memory_id="m-old", obj="paris", created_at=_T0 - timedelta(days=1))

    verdict = await _sweep(adj, ns, winner, cand)

    assert verdict.apply is True
    assert verdict.kind is AdjudicationKind.SUPERSEDE
    assert len(bus.detected()) == 1, "AUTO-resolved contradictions are audit facts too"
    assert bus.pending() == [], "nothing is waiting for a human on the automatic lane"
    (record,) = list(records._records.values())  # private read: the inbox state IS the assertion
    assert record.state is ConflictState.DETECTED


# ═══════════════ 3. RE-DETECTION IS SILENT — ONE CONFLICT IS NOT ONE EVENT PER TICK ══════════
async def test_three_sweeps_of_one_unchanged_conflict_emit_one_detected(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """The daemon ``MaintenanceLoop`` sweeps on a timer and re-derives the same ``conflict_id``.
    An ungated emit would be one ``ConflictDetected`` per tick forever for one ignored conflict —
    the exact defect already fixed for the pending bookend (§3 lines 104-106)."""
    clock = FrozenClock(_T0)
    bus, records = _RecordingBus(), InMemoryConflictRecordRepository()
    adj = _adjudicator(records, bus, clock=clock, mode=ConflictResolutionMode.MANUAL)
    winner = _fact(engine_ns, memory_id="m-new", obj="berlin", created_at=_T0)
    cand = _fact(engine_ns, memory_id="m-old", obj="paris", created_at=_T0 - timedelta(days=1))

    for _ in range(3):
        await _sweep(adj, ns, winner, cand)
        clock.advance(timedelta(minutes=5))

    assert len(bus.detected()) == 1
    assert len(bus.pending()) == 1


# ═══════════════ 4. COEXIST DETECTS NOTHING — NO RECORD, NO EVENT ════════════════════════════
async def test_coexist_emits_no_detected_event(ns: ContractNamespace, engine_ns: Namespace) -> None:
    """``COEXIST`` asserts no winner/loser relationship at all: there is no conflict, so there
    must be neither a record nor a detection event."""
    clock = FrozenClock(_T0)
    bus, records = _RecordingBus(), InMemoryConflictRecordRepository()
    adj = _adjudicator(records, bus, clock=clock, mode=ConflictResolutionMode.AUTOMATIC)
    winner = _fact(engine_ns, memory_id="m-a", obj="berlin", created_at=_T0)
    cand = _fact(engine_ns, memory_id="m-b", obj="paris", created_at=_T0 - timedelta(days=1))

    verdict = await adj.adjudicate(
        ns=ns,
        winner=winner,
        candidate=cand,
        heuristic_contradicts=False,  # the heuristic saw no contradiction
        budget=adj.new_budget(),
    )

    assert verdict.kind is AdjudicationKind.COEXIST
    assert bus.detected() == []
    assert list(records._records.values()) == []


# ═══════════════ 5. THE EVENT IS CONTENT-FREE — IT GOES THROUGH THE SAME GUARD ═══════════════
async def test_detected_event_carries_no_memory_content(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """CANONICAL §3.1. The facts in play literally contain "berlin"/"paris"; the event may not."""
    clock = FrozenClock(_T0)
    bus, records = _RecordingBus(), InMemoryConflictRecordRepository()
    adj = _adjudicator(records, bus, clock=clock, mode=ConflictResolutionMode.MANUAL)
    winner = _fact(engine_ns, memory_id="m-new", obj="berlin", created_at=_T0)
    cand = _fact(engine_ns, memory_id="m-old", obj="paris", created_at=_T0 - timedelta(days=1))

    await _sweep(adj, ns, winner, cand)

    (event,) = bus.detected()
    blob = event.model_dump_json()
    assert "berlin" not in blob and "paris" not in blob
    assert "the user lives in" not in blob
