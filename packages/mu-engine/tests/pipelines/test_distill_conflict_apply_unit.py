"""``DistillPipeline`` as ``ResolveConflictStage`` — the decisions actually LAND.

Authority: ``conflict-resolution-async-design.md`` §1.1 (the diagram's two apply arms), §2 (the
``DetectConflictsStage``/``ResolveConflictStage`` split table), §3.1 (the ``ConflictRecord`` FSM),
§5 line 218 (*"A resolve action **does not execute inline** ... and **enqueues**
``ResolveConflictStage``"*), §9 obligation 4 (*"Manual-resolve applies on the lease"*).

**What was inert before these tests existed.** The whole async-conflict lane was built and left
unwired at exactly two call sites, and each gap had the same shape — a durable decision that no
code path could ever act on:

* a human's ``POST /resolve`` moved a ``ConflictRecord`` to ``RESOLVED`` and enqueued an intent
  that nothing drained, so both contending items stayed ``state='active'`` forever while the
  record claimed the conflict was settled;
* an AUTOMATIC supersession wrote ``state=SUPERSEDED`` on the loser and never closed its own
  record, so ``ConflictState.AUTO_RESOLVED`` and ``ResolutionOrigin.AUTO`` — the two values spec
  line 117 makes the entire distinction between an automatic and a manual resolution — were
  states no production path could enter.

Every test here is offline: an in-memory ``GraphStorePort`` double, a frozen clock, the REAL
``ConflictResolutionService`` / ``ConflictAdjudicator`` / ``InMemoryConflictRecordRepository`` /
``RecordBackedResolutionQueue``. No store, no model, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from mu_contracts.domain.events import ConflictResolved, DomainEvent, MemorySuperseded
from mu_contracts.domain.model.conflict import (
    ConflictResolutionKind,
    ConflictState,
    ResolutionOrigin,
)
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.lifecycle.conflict import (
    ConflictAdjudicator,
    InMemoryConflictRecordRepository,
    compute_conflict_id,
)
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.conflict.ports import RecordBackedResolutionQueue
from mu_engine.services.conflict.resolution import (
    ConflictResolutionService,
    ManualDecision,
    ManualDecisionKind,
)
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)
#: In ``DistillSettings.functional_predicates``, so two different objects CONTRADICT and the
#: supersede path is genuinely entered (same fixture shape as ``test_distill_pin_guard_unit``).
_PREDICATE = "lives_in"


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


@pytest.fixture
def scope() -> ClientScope:
    return ClientScope(
        principal_id="u1",
        org_id="org1",
        workspace_id="ws1",
        session_id="s1",
        agent_principal_id="u1",
    )


def _fact(
    ns: Namespace,
    *,
    memory_id: str,
    obj: str,
    created_at: datetime,
    pinned: bool = False,
    tier: MemoryTier = MemoryTier.LTM,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content=f"the user lives in {obj}",
        namespace=ns,
        owner_id="u1",
        workspace_id="ws1",
        session_id="s1",
        tier=tier,
        state=MemoryState.ACTIVE,
        created_at=created_at,
        valid_at=created_at,
        subject="user",
        predicate=_PREDICATE,
        object=obj,
        pinned=pinned,
    )


class _FakeLtm:
    """An in-memory ``GraphStorePort`` that also records WHEN each write happened.

    The shared ``timeline`` is what makes the ordering obligation testable at all: a test can
    only assert "the record was closed AFTER the supersession landed" if both events land on one
    ordered log. Every entry is content-free (ids and op names only).
    """

    def __init__(self, resident: list[MemoryItem], timeline: list[str]) -> None:
        self.facts: dict[str, MemoryItem] = {m.id: m for m in resident}
        self.invalidated: list[tuple[str, str]] = []
        self.conflicts_marked: list[tuple[str, str]] = []
        self.timeline = timeline

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
        self.timeline.append(f"invalidate:{loser_id}")
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


class _TimelineApply:
    """A pass-through around the REAL ``ConflictResolutionService`` that timestamps the two
    apply-side callbacks onto the same log ``_FakeLtm`` writes to.

    Deliberately NOT a stand-in for the service: every call is delegated, so the FSM edge, the
    idempotency guard and the ``ConflictResolved`` emission are the production ones. It only
    observes ORDER, which is the one property no return value can express.
    """

    def __init__(self, inner: ConflictResolutionService, timeline: list[str]) -> None:
        self._inner = inner
        self._timeline = timeline

    async def mark_applied(
        self, ns: Namespace, conflict_id: str, *, superseded_valid_at: datetime | None = None
    ) -> Any:
        self._timeline.append(f"mark_applied:{conflict_id}")
        return await self._inner.mark_applied(
            ns, conflict_id, superseded_valid_at=superseded_valid_at
        )

    async def record_automatic_resolution(
        self,
        ns: Namespace,
        conflict_id: str,
        *,
        winner_id: str,
        superseded_valid_at: datetime | None = None,
    ) -> Any:
        self._timeline.append(f"record_auto:{conflict_id}")
        return await self._inner.record_automatic_resolution(
            ns, conflict_id, winner_id=winner_id, superseded_valid_at=superseded_valid_at
        )


# ═══════════════════════════════════════════════ the AUTOMATIC lane reaches AUTO_RESOLVED ══
async def test_an_automatic_supersession_closes_its_record_as_auto_resolved_with_origin_auto(
    ns: Namespace,
) -> None:
    """§3.1 ``DETECTED -> AUTO_RESOLVED``, and spec line 117's ``resolution_origin`` distinction.

    Before ``_close_automatic_record`` existed, ``ConflictAdjudicator`` opened the automatic
    lane's record in ``DETECTED`` and nothing ever moved it: this exact fixture superseded the
    resident fact and left a record permanently claiming the conflict was merely *detected*.
    """
    timeline: list[str] = []
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([resident], timeline)
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    clock = FrozenClock(_T0 + timedelta(days=30))
    # router=None -> the adjudicator's own deterministic heuristic floor decides (C=1.0), which
    # is the DEFAULT shipped behaviour on a composition with no LLM router. The record it opens
    # is the real one, from the real `_open_record`.
    adjudicator = ConflictAdjudicator(router=None, clock=clock, bus=bus, conflict_records=records)
    service = ConflictResolutionService(
        records=records, queue=RecordBackedResolutionQueue(records), clock=clock, bus=bus
    )

    pipeline = DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=clock,
        bus=bus,
        adjudicator=adjudicator,
        conflict_apply=_TimelineApply(service, timeline),  # type: ignore[arg-type]
    )
    report = await pipeline.distill(ns, [incoming])

    assert report.superseded == 1
    assert ltm.facts["resident"].state is MemoryState.SUPERSEDED

    conflict_id = compute_conflict_id(ns, ("incoming", "resident"), _PREDICATE)
    record = await records.get(ns, conflict_id)
    assert record is not None
    assert record.state is ConflictState.AUTO_RESOLVED
    assert record.resolution_origin is ResolutionOrigin.AUTO
    assert record.resolved_winner_id == "incoming"
    assert record.resolution_kind is ConflictResolutionKind.SUPERSEDE
    # The automatic lane's supersession is written under the lease that decided it, so there is
    # no gap to recover from and the record must never enter `awaiting_apply`.
    assert record.resolution_applied_at is not None
    assert await records.awaiting_apply(ns) == []

    # §8 line 279 — the automatic bookend on the content-free bus. Before this wiring the event
    # existed but could only ever be emitted with `origin=manual`.
    resolved_events = [e for e in bus.events if isinstance(e, ConflictResolved)]
    assert [e.resolution_origin for e in resolved_events] == [ResolutionOrigin.AUTO]
    assert resolved_events[0].winner_id == "incoming"
    assert resolved_events[0].loser_ids == ["resident"]


async def test_the_record_is_closed_after_the_supersession_write_never_before(
    ns: Namespace,
) -> None:
    """THE ORDERING OBLIGATION. ``record_automatic_resolution`` stamps ``AUTO_RESOLVED`` plus a
    ``superseded_valid_at``; if it ran before the store write, a crash in between would leave a
    record asserting a supersession that never happened — an audit trail that lies, which is
    strictly worse than no audit trail. The write comes first, always.
    """
    timeline: list[str] = []
    resident = _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )
    ltm = _FakeLtm([resident], timeline)
    records = InMemoryConflictRecordRepository()
    clock = FrozenClock(_T0 + timedelta(days=30))
    adjudicator = ConflictAdjudicator(router=None, clock=clock, conflict_records=records)
    service = ConflictResolutionService(records=records, queue=RecordBackedResolutionQueue(records))

    await DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=clock,
        adjudicator=adjudicator,
        conflict_apply=_TimelineApply(service, timeline),  # type: ignore[arg-type]
    ).distill(ns, [incoming])

    conflict_id = compute_conflict_id(ns, ("incoming", "resident"), _PREDICATE)
    assert "invalidate:resident" in timeline, "non-vacuity: the supersession really was written"
    assert f"record_auto:{conflict_id}" in timeline, "non-vacuity: the record really was closed"
    assert timeline.index("invalidate:resident") < timeline.index(f"record_auto:{conflict_id}")


# ══════════════════════════════════════════════════ the MANUAL lane's decision actually lands ══
async def _park_a_conflict(
    ns: Namespace,
    ltm: _FakeLtm,
    records: InMemoryConflictRecordRepository,
    clock: FrozenClock,
) -> str:
    """Drive the REAL adjudicator under a MANUAL policy so a genuine ``MANUAL_PENDING`` record
    exists — never a hand-built record, so the test cannot pass against a shape the production
    detect side would not have produced."""
    from mu_engine.lifecycle.conflict import ConflictResolutionMode, ConflictResolutionPolicy

    adjudicator = ConflictAdjudicator(
        router=None,
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL),
        clock=clock,
        conflict_records=records,
    )
    await DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=clock,
        adjudicator=adjudicator,
    ).distill(ns, [ltm.facts["incoming"]])
    parked = await records.pending(ns)
    assert len(parked) == 1, "the MANUAL policy must have parked exactly one conflict"
    assert parked[0].state is ConflictState.MANUAL_PENDING
    return parked[0].conflict_id


async def test_a_human_decision_is_drained_applied_and_marked_applied(
    ns: Namespace, scope: ClientScope
) -> None:
    """§9 obligation 4. The full round trip: park -> a human resolves -> the NEXT distill tick
    supersedes the loser on the store and stamps ``resolution_applied_at``.

    The window handed to that tick is deliberately EMPTY. A human decision must land whether or
    not new memories arrived — an apply that only ran when the window happened to contain a
    conflicting fact would leave the most common case (a quiet namespace) permanently unapplied.
    """
    timeline: list[str] = []
    clock = FrozenClock(_T0 + timedelta(days=30))
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    ltm = _FakeLtm(
        [
            _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0),
            _fact(ns, memory_id="incoming", obj="Lisbon", created_at=_T0 + timedelta(days=1)),
        ],
        timeline,
    )
    conflict_id = await _park_a_conflict(ns, ltm, records, clock)
    assert ltm.invalidated == [], "MANUAL parks: no supersession is stamped at detection time"

    queue = RecordBackedResolutionQueue(records)
    service = ConflictResolutionService(records=records, queue=queue, clock=clock, bus=bus)
    decided = await service.resolve(
        scope,
        ns,
        conflict_id,
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="incoming", resolved_by="u1"),
    )
    assert decided.state is ConflictState.RESOLVED
    assert decided.resolution_applied_at is None, "accepted, NOT applied — §5 line 218"
    assert ltm.facts["resident"].state is MemoryState.ACTIVE, "still not applied to any store"

    # The next background tick — with an EMPTY window.
    pipeline = DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=clock,
        bus=bus,
        resolution_queue=queue,
        conflict_apply=service,
    )
    await pipeline.distill(ns, [])

    assert ltm.facts["resident"].state is MemoryState.SUPERSEDED
    assert ltm.invalidated == [("resident", "incoming")]
    applied = await records.get(ns, conflict_id)
    assert applied is not None
    assert applied.resolution_applied_at is not None
    assert applied.resolution_origin is ResolutionOrigin.MANUAL
    assert [
        e for e in bus.events if isinstance(e, MemorySuperseded) and e.loser_id == "resident"
    ], "the §7.2 cross-store supersession event is published for a manual apply too"


async def test_an_applied_decision_leaves_the_drain_and_is_never_applied_twice(
    ns: Namespace, scope: ClientScope
) -> None:
    """``mark_applied`` is the stopping condition of a NON-DESTRUCTIVE drain (spec's at-least-once
    + idempotent-apply discipline). A second tick must find nothing and write nothing — the drain
    re-derives its intents from the records, so a stopping condition that did not hold would
    re-supersede on every sweep forever."""
    timeline: list[str] = []
    clock = FrozenClock(_T0 + timedelta(days=30))
    records = InMemoryConflictRecordRepository()
    ltm = _FakeLtm(
        [
            _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0),
            _fact(ns, memory_id="incoming", obj="Lisbon", created_at=_T0 + timedelta(days=1)),
        ],
        timeline,
    )
    conflict_id = await _park_a_conflict(ns, ltm, records, clock)
    queue = RecordBackedResolutionQueue(records)
    service = ConflictResolutionService(records=records, queue=queue, clock=clock)
    await service.resolve(
        scope,
        ns,
        conflict_id,
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="incoming", resolved_by="u1"),
    )
    pipeline = DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=clock,
        resolution_queue=queue,
        conflict_apply=service,
    )

    await pipeline.distill(ns, [])
    assert await queue.drain(ns) == (), "an applied record leaves awaiting_apply"
    writes_after_first_tick = list(ltm.invalidated)

    await pipeline.distill(ns, [])

    assert ltm.invalidated == writes_after_first_tick, "the second tick wrote nothing new"


async def test_a_decision_lost_to_a_crash_is_re_derived_and_applied_on_the_next_tick(
    ns: Namespace, scope: ClientScope
) -> None:
    """The property the NON-DESTRUCTIVE drain exists for, asserted rather than assumed.

    A brand-new ``RecordBackedResolutionQueue`` over the SAME records is exactly what a restarted
    process has: it stores nothing of its own, so everything it hands out was re-derived from the
    durable records. If the drain were a destructive pop, the decision would have died with the
    process while the record still said ``RESOLVED`` — the failure the record-backed queue was
    introduced to remove.
    """
    timeline: list[str] = []
    clock = FrozenClock(_T0 + timedelta(days=30))
    records = InMemoryConflictRecordRepository()
    ltm = _FakeLtm(
        [
            _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0),
            _fact(ns, memory_id="incoming", obj="Lisbon", created_at=_T0 + timedelta(days=1)),
        ],
        timeline,
    )
    conflict_id = await _park_a_conflict(ns, ltm, records, clock)
    accepting_service = ConflictResolutionService(
        records=records, queue=RecordBackedResolutionQueue(records), clock=clock
    )
    await accepting_service.resolve(
        scope,
        ns,
        conflict_id,
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="incoming", resolved_by="u1"),
    )

    # ── process restart: a FRESH queue and a FRESH service over the same durable records ──
    revived_queue = RecordBackedResolutionQueue(records)
    revived_service = ConflictResolutionService(records=records, queue=revived_queue, clock=clock)
    await DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=clock,
        resolution_queue=revived_queue,
        conflict_apply=revived_service,
    ).distill(ns, [])

    assert ltm.facts["resident"].state is MemoryState.SUPERSEDED
    reloaded = await records.get(ns, conflict_id)
    assert reloaded is not None and reloaded.resolution_applied_at is not None


async def test_a_human_may_resolve_against_a_pinned_loser_the_automatic_sweep_could_not(
    ns: Namespace, scope: ClientScope
) -> None:
    """CANONICAL §7.10 / §7.17 4a(b): *"a pinned item is never the AUTO-supersede loser"* is a
    statement about SWEEPS. The apply routes every loser through the CENTRAL
    ``LifecyclePolicy`` with ``trigger=EXPLICIT`` — the guard is consulted, never re-implemented,
    and never bypassed — so a human who opened the inbox and chose the winner is honoured.
    Refusing here would make a pinned memory impossible to ever resolve.
    """
    timeline: list[str] = []
    clock = FrozenClock(_T0 + timedelta(days=30))
    records = InMemoryConflictRecordRepository()
    ltm = _FakeLtm(
        [
            _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0, pinned=True),
            _fact(ns, memory_id="incoming", obj="Lisbon", created_at=_T0 + timedelta(days=1)),
        ],
        timeline,
    )
    conflict_id = await _park_a_conflict(ns, ltm, records, clock)
    queue = RecordBackedResolutionQueue(records)
    service = ConflictResolutionService(records=records, queue=queue, clock=clock)
    await service.resolve(
        scope,
        ns,
        conflict_id,
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="incoming", resolved_by="u1"),
    )

    await DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=clock,
        resolution_queue=queue,
        conflict_apply=service,
    ).distill(ns, [])

    assert ltm.facts["resident"].state is MemoryState.SUPERSEDED


async def test_a_merge_decision_is_left_queued_rather_than_stamped_applied(
    ns: Namespace, scope: ClientScope
) -> None:
    """REPORTED GAP, asserted so it cannot silently become a lie. ``MERGE`` needs
    ``ComposeService`` to mint the composed item (spec line 219) and this pipeline holds none, so
    the intent is NOT applied and NOT marked applied: the human's decision stays recoverable
    instead of being stamped against a merge that never happened."""
    timeline: list[str] = []
    clock = FrozenClock(_T0 + timedelta(days=30))
    records = InMemoryConflictRecordRepository()
    ltm = _FakeLtm(
        [
            _fact(ns, memory_id="resident", obj="Berlin", created_at=_T0),
            _fact(ns, memory_id="incoming", obj="Lisbon", created_at=_T0 + timedelta(days=1)),
        ],
        timeline,
    )
    conflict_id = await _park_a_conflict(ns, ltm, records, clock)
    queue = RecordBackedResolutionQueue(records)
    service = ConflictResolutionService(records=records, queue=queue, clock=clock)
    await service.resolve(
        scope,
        ns,
        conflict_id,
        ManualDecision(
            kind=ManualDecisionKind.MERGE,
            winner_id="incoming",
            merged_text_ref="draft:abc123",
            resolved_by="u1",
        ),
    )

    await DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=clock,
        resolution_queue=queue,
        conflict_apply=service,
    ).distill(ns, [])

    assert ltm.invalidated == [], "nothing was superseded for an unimplementable merge"
    still_pending = await records.get(ns, conflict_id)
    assert still_pending is not None
    assert still_pending.resolution_applied_at is None
    assert len(await queue.drain(ns)) == 1, "still offered to the next tick — never lost"


async def test_an_unwired_pipeline_is_unchanged(ns: Namespace) -> None:
    """NON-VACUITY / backward-compatibility control. With neither port wired — the pre-existing
    shape of every current caller — the pipeline supersedes exactly as before and never reaches
    for a queue or an apply port."""
    timeline: list[str] = []
    ltm = _FakeLtm([_fact(ns, memory_id="resident", obj="Berlin", created_at=_T0)], timeline)
    incoming = _fact(
        ns,
        memory_id="incoming",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        tier=MemoryTier.MTM,
    )

    report = await DistillPipeline(
        ltm=ltm,  # type: ignore[arg-type]
        clock=FrozenClock(_T0 + timedelta(days=30)),
    ).distill(ns, [incoming])

    assert report.superseded == 1
    assert ltm.facts["resident"].state is MemoryState.SUPERSEDED
