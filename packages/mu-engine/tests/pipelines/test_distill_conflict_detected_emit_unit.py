"""``ConflictDetected`` on the DEFAULT full-local composition — the lane with no adjudicator.

``ConflictDetected`` got a producer in ``ConflictAdjudicator._open_record``
(``lifecycle/conflict.py``, ``test_conflict_detected_emit_unit.py``). That closed the LLM-judged
lane only. The lane that actually ships by default did not move: ``mu-local/composition.py``
leaves ``conflict_adjudicator=None`` whenever no LLM router is wired, and
``DistillPipeline._resolve`` then decides every contradiction itself through
``_heuristic_only_verdict`` — the degrade floor that opens **no** ``ConflictRecord`` and therefore
reached **no** emit. On that composition the whole polarity/cardinality lane superseded facts,
tagged pairs ``CONFLICTS_WITH`` in the graph and blocked pinned losers while the one event in the
frozen catalog meaning *"a contradiction exists"* stayed silent — the exact defect the audit
names, one layer down from where it was fixed.

Authority: ``conflict-resolution-async-design.md`` §2 stage table (*"Emits ``ConflictDetected``
and opens/updates a ``ConflictRecord``"*, bound to detection and placed BEFORE the
AUTOMATIC/MANUAL branch) · §1.1 (the automatic lane detects too) · CANONICAL §3.1 (content-free).

Offline: an in-memory ``GraphStorePort`` double, a frozen clock, no router, no store, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mu_contracts.domain.events import ConflictDetected, DomainEvent, MemorySuperseded
from mu_engine.lifecycle.conflict import ConflictAdjudicator, InMemoryConflictRecordRepository
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)
#: In ``DistillSettings.functional_predicates`` -> two different objects genuinely CONTRADICT.
_FUNCTIONAL = "lives_in"
#: NOT in that set -> two different objects legitimately COEXIST, and nothing was detected.
_NON_FUNCTIONAL = "likes"


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _fact(
    ns: Namespace,
    *,
    memory_id: str,
    obj: str,
    created_at: datetime,
    predicate: str = _FUNCTIONAL,
    pinned: bool = False,
    tier: MemoryTier = MemoryTier.LTM,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=f"the user {predicate} {obj}",
        namespace=ns,
        owner_id="u1",
        workspace_id="ws1",
        session_id="s1",
        tier=tier,
        state=MemoryState.ACTIVE,
        created_at=created_at,
        valid_at=created_at,
        subject="user",
        predicate=predicate,
        object=obj,
        pinned=pinned,
    )


class _FakeLtm:
    """In-memory ``GraphStorePort`` double — the same shape the sibling distill unit tests use."""

    def __init__(self, resident: list[MemoryItem]) -> None:
        self.facts: dict[str, MemoryItem] = {m.id: m for m in resident}
        self.invalidated: list[tuple[str, str]] = []
        self.conflicts_marked: list[tuple[str, str]] = []

    async def find_conflicts(self, ns: Namespace, subject: str, predicate: str) -> list[MemoryItem]:
        return [
            m
            for m in self.facts.values()
            if m.subject == subject and m.predicate == predicate and m.state is MemoryState.ACTIVE
        ]

    async def upsert_fact(self, item: MemoryItem) -> None:
        self.facts[item.id] = item

    async def get_fact(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return self.facts.get(memory_id)

    async def facts_at(
        self, ns: Namespace, at: Any, *, subject: str | None = None
    ) -> list[MemoryItem]:
        return [
            m
            for m in self.facts.values()
            if m.state is MemoryState.ACTIVE and (subject is None or m.subject == subject)
        ]

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: Any, reason: str
    ) -> None:
        self.invalidated.append((loser_id, winner_id))
        loser = self.facts.get(loser_id)
        if loser is not None:
            loser.state = MemoryState.SUPERSEDED
            loser.invalid_at = at

    async def mark_conflict(self, ns: Namespace, a_id: str, b_id: str, *, at: Any) -> None:
        self.conflicts_marked.append(tuple(sorted((a_id, b_id))))  # type: ignore[arg-type]


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)

    def detected(self) -> list[ConflictDetected]:
        return [e for e in self.events if isinstance(e, ConflictDetected)]


def _pipeline(ltm: _FakeLtm, bus: _RecordingBus, *, adjudicator: Any = None) -> DistillPipeline:
    """The DEFAULT full-local shape: a real bus, and NO adjudicator unless a test wires one."""
    return DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=FrozenClock(_T0 + timedelta(days=30)),
        bus=bus,
        adjudicator=adjudicator,
    )


# ═════════════════ 1. THE NO-ADJUDICATOR LANE (THE DEFAULT) EMITS THE DETECTION ═════════════
async def test_a_heuristic_only_supersession_emits_conflict_detected(ns: Namespace) -> None:
    """The reproduction: a genuine contradiction, no adjudicator, and the catalog stayed silent.

    Before the fix this fixture superseded ``resident`` — ``MemorySuperseded`` on the bus, the
    loser flipped in the graph — and published ZERO ``ConflictDetected``. A consumer written
    against the catalog saw a supersession appear with no detection ever announced.
    """
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([resident])
    bus = _RecordingBus()

    report = await _pipeline(ltm, bus).distill(ns, [incoming])

    assert report.superseded == 1
    assert ltm.facts["resident"].state is MemoryState.SUPERSEDED
    detected = bus.detected()
    assert len(detected) == 1
    assert detected[0].incoming_id == "incoming"
    assert detected[0].candidate_ids == ["resident"]
    assert detected[0].method == "polarity_cardinality_heuristic"
    assert detected[0].namespace == ns
    # The detection is the CAUSE of the supersession, so it must reach the bus first — a
    # consumer that sees the effect before the cause cannot build a timeline from the catalog.
    kinds = [type(e).__name__ for e in bus.events]
    assert kinds.index("ConflictDetected") < kinds.index("MemorySuperseded")


# ═════════════════ 2. THE PIN-BLOCKED PAIR — THE ONE AN OWNER MUST BE TOLD ABOUT ════════════
async def test_a_pin_blocked_contradiction_still_emits_conflict_detected(ns: Namespace) -> None:
    """A pinned loser blocks the supersession (CANONICAL §7.17 item 4a(b)) and both facts stay
    ACTIVE. Nothing is superseded, so ``MemorySuperseded`` correctly never fires — which made
    this the one contradiction with NO bus trace of any kind before the fix, even though it is
    precisely the one a health view or an owner has to act on."""
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0, pinned=True)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([resident])
    bus = _RecordingBus()

    await _pipeline(ltm, bus).distill(ns, [incoming])

    assert ltm.facts["resident"].state is MemoryState.ACTIVE  # the pin held
    assert ltm.conflicts_marked == [("incoming", "resident")]  # tagged in the graph
    assert [e for e in bus.events if isinstance(e, MemorySuperseded)] == []
    detected = bus.detected()
    assert len(detected) == 1
    assert detected[0].incoming_id == "incoming"
    assert detected[0].candidate_ids == ["resident"]


# ═════════════════ 3. NEVER OVER-EMITS: A COEXISTENCE IS NOT A CONTRADICTION ════════════════
async def test_a_coexisting_non_functional_fact_emits_no_detection(ns: Namespace) -> None:
    """``likes`` is non-functional: two objects are two true facts, not a conflict. An event that
    fired here would be worse than silence — it would train every consumer to ignore it."""
    resident = _fact(ns, memory_id="resident", obj="tea", created_at=_T0, predicate=_NON_FUNCTIONAL)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="coffee",
        created_at=_T0 + timedelta(days=1),
        predicate=_NON_FUNCTIONAL,
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([resident])
    bus = _RecordingBus()

    await _pipeline(ltm, bus).distill(ns, [incoming])

    assert ltm.facts["resident"].state is MemoryState.ACTIVE
    assert bus.detected() == []


# ═════════════════ 4. ONE DETECTION PER SWEEP-WINNER, NOT ONE PER CANDIDATE ═════════════════
async def test_one_winner_contending_with_two_candidates_emits_one_event_carrying_both(
    ns: Namespace,
) -> None:
    """``ConflictDetected.candidate_ids`` is a LIST because the residue is many-to-one. Emitting
    per candidate would multiply one detection into N bus events describing the same instant."""
    r1 = _fact(ns, memory_id="r1", obj="Berlin", created_at=_T0)
    r2 = _fact(ns, memory_id="r2", obj="Madrid", created_at=_T0 + timedelta(hours=1))
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([r1, r2])
    bus = _RecordingBus()

    await _pipeline(ltm, bus).distill(ns, [incoming])

    detected = bus.detected()
    assert len(detected) == 1
    assert sorted(detected[0].candidate_ids) == ["r1", "r2"]


# ═════════════════ 5. THE TWO LANES DO NOT DOUBLE-EMIT ══════════════════════════════════════
async def test_an_adjudicated_conflict_is_announced_exactly_once(ns: Namespace) -> None:
    """With an adjudicator wired, ``ConflictAdjudicator._open_record`` owns the announcement.
    This pipeline must stay silent on that path or every LLM-judged conflict would be reported
    twice — the failure mode a naive "just always emit here" fix introduces."""
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([resident])
    bus = _RecordingBus()
    clock = FrozenClock(_T0 + timedelta(days=30))
    # router=None -> the ADJUDICATOR's own deterministic heuristic floor, which still opens the
    # real ``ConflictRecord`` and emits from ``lifecycle/conflict.py``. No network.
    adjudicator = ConflictAdjudicator(
        router=None, clock=clock, bus=bus, conflict_records=InMemoryConflictRecordRepository()
    )

    await _pipeline(ltm, bus, adjudicator=adjudicator).distill(ns, [incoming])

    assert len(bus.detected()) == 1
