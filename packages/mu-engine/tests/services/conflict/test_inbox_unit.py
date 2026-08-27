"""§5 — the ``ConflictInboxView`` projector. Offline: in-process repo + hydrator, frozen clock.

Covers conflict-resolution-async-design.md §5 (lines 168-206, 222) and §8 line 273.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.errors import NamespaceIsolationError
from mu_contracts.domain.events import DegradedModeEntered, DegradeReason, DomainEvent
from mu_contracts.domain.model.conflict import (
    ConflictRecord,
    ConflictResolutionMode,
    ConflictState,
)
from mu_contracts.domain.model.memory import Namespace, Tier, Visibility
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.lifecycle.conflict import (
    ConflictResolutionPolicy,
    InMemoryConflictRecordRepository,
)
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.conflict.inbox import ConflictInboxProjector
from mu_engine.services.conflict.ports import ConflictMemberHydration
from mu_engine.services.conflict.settings import ConflictSettings

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)
_NOW = datetime(2026, 6, 10, tzinfo=UTC)


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


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


class _Hydrator:
    def __init__(self, bodies: dict[str, str], *, fail: bool = False) -> None:
        self._bodies = bodies
        self._fail = fail
        self.asked: frozenset[str] = frozenset()

    async def hydrate(
        self, ns: Namespace, memory_ids: frozenset[str]
    ) -> dict[str, ConflictMemberHydration]:
        if self._fail:
            raise RuntimeError("store is down")
        self.asked = memory_ids
        return {
            memory_id: ConflictMemberHydration(
                content=body,
                tier=Tier.LTM.value,
                valid_at=_T0.isoformat(),
                provenance_id=f"prov-{memory_id}",
                source_label="laptop",
            )
            for memory_id, body in self._bodies.items()
            if memory_id in memory_ids
        }


def _record(
    ns: Namespace,
    *,
    conflict_id: str = "c1",
    members: tuple[str, ...] = ("a", "b"),
    state: ConflictState = ConflictState.MANUAL_PENDING,
    detected_at: datetime = _T0,
    pin_blocked: bool = False,
    mode: ConflictResolutionMode = ConflictResolutionMode.MANUAL,
) -> ConflictRecord:
    return ConflictRecord(
        conflict_id=conflict_id,
        namespace=ns,
        member_ids=members,
        predicate_key="lives_in",
        method="llm_adjudicator",
        detected_confidence=0.6,
        proposed_winner_id=members[0],
        state=state,
        detected_at=detected_at,
        pin_blocked=pin_blocked,
        policy_snapshot=ConflictResolutionPolicy(mode=mode).snapshot(),
    )


def _projector(
    records: InMemoryConflictRecordRepository,
    *,
    hydrator: _Hydrator | None = None,
    bus: _RecordingBus | None = None,
    settings: ConflictSettings | None = None,
) -> ConflictInboxProjector:
    return ConflictInboxProjector(
        records=records,
        hydrator=hydrator,  # type: ignore[arg-type]
        settings=settings,
        clock=FrozenClock(_NOW),
        bus=bus,
    )


# ══════════════════════════════════════ the view itself ═════════════════════════════════════
async def test_a_parked_conflict_appears_with_hydrated_bodies(
    ns: Namespace, scope: ClientScope
) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    hydrator = _Hydrator({"a": "the user lives in Berlin", "b": "the user lives in Lisbon"})

    view = await _projector(records, hydrator=hydrator).view(scope, ns)

    assert view.pending_count == 1
    (item,) = view.pending
    assert item.conflict_id == "c1"
    assert {m.content for m in item.members} == {
        "the user lives in Berlin",
        "the user lives in Lisbon",
    }
    assert [m.is_proposed_winner for m in item.members] == [True, False]
    assert view.generated_at == _NOW
    assert hydrator.asked == frozenset({"a", "b"}), "bounded by the page's own member ids"


async def test_the_effective_policy_is_read_off_the_snapshot_not_re_resolved(
    ns: Namespace, scope: ClientScope
) -> None:
    """Spec line 158: the snapshot exists so an audit shows which policy GOVERNED the decision.
    Showing today's namespace policy next to yesterday's parked conflict is the exact confusion
    the snapshot was added to prevent."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns, mode=ConflictResolutionMode.AUTOMATIC))

    view = await _projector(records).view(scope, ns)

    assert view.pending[0].effective_policy is ConflictResolutionMode.AUTOMATIC


async def test_the_inbox_is_ordered_oldest_first_and_deterministically(
    ns: Namespace, scope: ClientScope
) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns, conflict_id="new", detected_at=_T0 + timedelta(days=2)))
    await records.add(_record(ns, conflict_id="old", detected_at=_T0))
    await records.add(_record(ns, conflict_id="mid", detected_at=_T0 + timedelta(days=1)))

    view = await _projector(records).view(scope, ns)

    assert [i.conflict_id for i in view.pending] == ["old", "mid", "new"]


async def test_a_hydration_outage_still_renders_the_inbox(
    ns: Namespace, scope: ClientScope
) -> None:
    """A user who can see "decisions are waiting" is better served than one who sees an error;
    an empty body is visibly missing rather than plausibly wrong."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))

    view = await _projector(records, hydrator=_Hydrator({}, fail=True)).view(scope, ns)

    assert view.pending_count == 1
    assert all(m.content == "" for m in view.pending[0].members)


async def test_with_no_hydrator_wired_the_view_still_renders(
    ns: Namespace, scope: ClientScope
) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    view = await _projector(records).view(scope, ns)
    assert view.pending_count == 1


async def test_the_pin_blocked_signal_reaches_the_inbox(ns: Namespace, scope: ClientScope) -> None:
    """memory-health §6.4's "a new fact contradicts a memory you pinned" — the reason the system
    parked this FOR the user rather than because they asked."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns, pin_blocked=True))
    view = await _projector(records).view(scope, ns)
    assert view.pending[0].pin_blocked is True


async def test_another_partitions_conflicts_are_invisible(
    ns: Namespace, scope: ClientScope
) -> None:
    records = InMemoryConflictRecordRepository()
    foreign = Namespace(
        org="org1", workspace="ws1", user="u2", session="s1", visibility=Visibility.PRIVATE
    )
    await records.add(_record(foreign))
    view = await _projector(records).view(scope, ns)
    assert view.pending_count == 0


async def test_a_caller_outside_the_partition_is_refused(ns: Namespace) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    intruder = ClientScope(
        principal_id="u2",
        org_id="org2",
        workspace_id="ws2",
        session_id="s9",
        agent_principal_id="u2",
    )
    with pytest.raises(NamespaceIsolationError):
        await _projector(records).view(intruder, ns)


# ═══════════════════════════════════════ the backlog nudge ══════════════════════════════════
async def test_a_backlog_over_the_threshold_raises_the_named_degrade(
    ns: Namespace, scope: ClientScope
) -> None:
    records = InMemoryConflictRecordRepository()
    for i in range(3):
        await records.add(_record(ns, conflict_id=f"c{i}", members=(f"a{i}", f"b{i}")))
    bus = _RecordingBus()

    view = await _projector(
        records, bus=bus, settings=ConflictSettings(manual_backlog_alert=2)
    ).view(scope, ns)

    assert view.backlog_alert is True
    (event,) = bus.events
    assert isinstance(event, DegradedModeEntered)
    assert event.reason is DegradeReason.CONFLICT_MANUAL_BACKLOG
    assert event.detail == "pending=3"


async def test_a_backlog_under_the_threshold_is_silent(ns: Namespace, scope: ClientScope) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    bus = _RecordingBus()
    view = await _projector(
        records, bus=bus, settings=ConflictSettings(manual_backlog_alert=2)
    ).view(scope, ns)
    assert view.backlog_alert is False
    assert bus.events == []


async def test_a_zero_threshold_disables_the_nudge_entirely(
    ns: Namespace, scope: ClientScope
) -> None:
    records = InMemoryConflictRecordRepository()
    for i in range(5):
        await records.add(_record(ns, conflict_id=f"c{i}", members=(f"a{i}", f"b{i}")))
    bus = _RecordingBus()
    view = await _projector(
        records, bus=bus, settings=ConflictSettings(manual_backlog_alert=0)
    ).view(scope, ns)
    assert view.backlog_alert is False
    assert bus.events == []


# ═════════════════════════════════════ cross-plane fusion ═══════════════════════════════════
async def test_fusing_two_planes_dedups_by_conflict_id(ns: Namespace, scope: ClientScope) -> None:
    """Spec line 222 — the SDK fuses the LOCAL and SHARED inboxes, deduped by ``conflict_id``.
    One implementation, so the three faces cannot each dedup differently."""
    local_records = InMemoryConflictRecordRepository()
    await local_records.add(_record(ns, conflict_id="shared-one"))
    await local_records.add(
        _record(ns, conflict_id="local-only", members=("x", "y"), detected_at=_T0)
    )
    shared_records = InMemoryConflictRecordRepository()
    await shared_records.add(
        _record(ns, conflict_id="shared-one", detected_at=_T0 + timedelta(days=5))
    )

    local = await _projector(local_records).view(scope, ns)
    shared = await _projector(shared_records).view(scope, ns)
    fused = ConflictInboxProjector.fuse([local, shared], generated_at=_NOW)

    assert [i.conflict_id for i in fused.pending] == ["local-only", "shared-one"]
    assert fused.pending_count == 2
    assert fused.namespace is None, "a fused view spans planes by definition"


def test_fusing_nothing_is_a_loud_error() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        ConflictInboxProjector.fuse([], generated_at=_NOW)
