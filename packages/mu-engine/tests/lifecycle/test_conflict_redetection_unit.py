"""Re-detection is IDEMPOTENT IN EFFECT, not merely in id — plus the surfaces it feeds.

A sweep re-derives the same ``conflict_id`` on every tick. The shipped code turned that into an
overwrite: ``detected_at`` and ``policy_snapshot`` were rebuilt each tick and the "waiting for
you" bookend was re-published unconditionally, so a daemon ``MaintenanceLoop`` running on a timer
emitted one ``ConflictResolutionPending`` per tick forever for one ignored conflict, the audit
snapshot spec line 158 exists for was silently re-derived from TODAY's policy, and the inbox's
"longest-waiting decision leads" ordering was meaningless.

Covers conflict-resolution-async-design.md §1.1 (the AUTOMATIC lane opens a record too), §3
lines 104-106 (idempotent id, no resurrection), §4.1 line 158 (the policy snapshot is an AUDIT
fact), §5 line 197 (pending = MANUAL_PENDING or REOPENED).
Offline: no router, no store, no network, no keys.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.events import (
    ConflictResolutionPending,
    DegradedModeEntered,
    DomainEvent,
)
from mu_contracts.domain.model.conflict import (
    ConflictResolutionMode,
    ConflictState,
)
from mu_contracts.domain.model.memory import Namespace as ContractNamespace
from mu_contracts.domain.model.memory import Visibility as ContractVisibility
from mu_engine.lifecycle.conflict import (
    ConflictAdjudicator,
    ConflictResolutionPolicy,
    InMemoryConflictRecordRepository,
    build_conflict_adjudicator,
)
from mu_engine.lifecycle.conflict_policy import ConflictLifecyclePolicy
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

    def pending(self) -> list[DomainEvent]:
        return [e for e in self.events if isinstance(e, ConflictResolutionPending)]


class _ExplodingResolver:
    """A §4.1 resolver whose store is down."""

    async def for_conflict(self, ns: object, member_ids: tuple[str, ...]) -> object:
        raise RuntimeError("policy store unreachable")


def _fact(
    ns: Namespace, *, memory_id: str, obj: str, created_at: datetime, predicate: str = "lives_in"
) -> MemoryItem:
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
        predicate=predicate,
        object=obj,
    )


def _adjudicator(
    records: InMemoryConflictRecordRepository,
    bus: _RecordingBus,
    *,
    clock: FrozenClock,
    policy: ConflictResolutionPolicy | None = None,
    policy_resolver: object | None = None,
) -> ConflictAdjudicator:
    return ConflictAdjudicator(
        router=None,
        policy=policy or ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL),
        policy_resolver=policy_resolver,  # type: ignore[arg-type]
        clock=clock,
        bus=bus,  # type: ignore[arg-type]
        conflict_records=records,
    )


async def _sweep(
    adj: ConflictAdjudicator, ns: ContractNamespace, winner: MemoryItem, cand: MemoryItem
) -> object:
    return await adj.adjudicate(
        ns=ns,  # type: ignore[arg-type]
        winner=winner,
        candidate=cand,
        heuristic_contradicts=True,
        budget=adj.new_budget(),
    )


# ═════════════ 1. A REPEATED SWEEP DOES NOT REWRITE THE AUDIT OR SPAM THE BUS ═══════════════
async def test_three_sweeps_of_one_conflict_produce_one_record_one_detected_at_one_event(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """The measured shipped behaviour was three distinct ``detected_at`` values and three
    ``ConflictResolutionPending`` events for one unchanged conflict."""
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    clock = FrozenClock(_T0)
    adj = _adjudicator(records, bus, clock=clock)
    winner = _fact(engine_ns, memory_id="a", obj="Paris", created_at=_T0)
    cand = _fact(engine_ns, memory_id="b", obj="Berlin", created_at=_T0 + timedelta(days=1))

    first = await _sweep(adj, ns, winner, cand)
    detected_at = first.conflict_record.detected_at  # type: ignore[union-attr]
    for _ in range(3):
        clock.advance(timedelta(hours=1))
        await _sweep(adj, ns, winner, cand)

    stored = await records.pending(ns)
    assert len(stored) == 1
    assert stored[0].detected_at == detected_at  # the age the inbox sorts on is STABLE
    assert len(bus.pending()) == 1  # one nudge, not one per tick, forever


async def test_the_policy_snapshot_is_not_re_derived_from_a_later_policy(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """Spec line 158: *"a later audit shows which policy governed the decision even if the
    namespace policy later changes"*. Rebuilding the snapshot each sweep made that false for
    every pending conflict — silently, since nothing errors."""
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    clock = FrozenClock(_T0)
    winner = _fact(engine_ns, memory_id="a", obj="Paris", created_at=_T0)
    cand = _fact(engine_ns, memory_id="b", obj="Berlin", created_at=_T0 + timedelta(days=1))

    strict = ConflictResolutionPolicy(
        mode=ConflictResolutionMode.MANUAL, auto_min_confidence=0.99, quarantine_below=0.9
    )
    await _sweep(_adjudicator(records, bus, clock=clock, policy=strict), ns, winner, cand)
    conflict_id = (await records.pending(ns))[0].conflict_id
    opened = await records.get(ns, conflict_id)
    assert opened is not None
    at_detection = opened.policy_snapshot

    loose = ConflictResolutionPolicy(
        mode=ConflictResolutionMode.MANUAL, auto_min_confidence=0.10, quarantine_below=0.05
    )
    await _sweep(_adjudicator(records, bus, clock=clock, policy=loose), ns, winner, cand)

    assert (await records.pending(ns))[0].policy_snapshot == at_detection


async def test_a_changed_member_hash_does_re_announce(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """Suppression must key on "nothing changed", not on "we have seen this id" — a genuinely
    new delta on one member is exactly the case the human should hear about again."""
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    clock = FrozenClock(_T0)
    adj = _adjudicator(records, bus, clock=clock)
    winner = _fact(engine_ns, memory_id="a", obj="Paris", created_at=_T0)
    cand = _fact(engine_ns, memory_id="b", obj="Berlin", created_at=_T0 + timedelta(days=1))
    await _sweep(adj, ns, winner, cand)

    changed = _fact(engine_ns, memory_id="b", obj="Madrid", created_at=_T0 + timedelta(days=1))
    assert changed.content_hash != cand.content_hash
    await _sweep(adj, ns, winner, changed)
    assert len(bus.pending()) == 2


# ═══════════ 2. A SETTLED CONFLICT IS NOT RESURRECTED — BY DESIGN, NOT BY EXCEPTION ═════════
def test_the_repark_guard_actually_refuses_settled_states() -> None:
    """``permits`` could never answer this: for every settled state the edge to MANUAL_PENDING is
    ILLEGAL, not trigger-blocked, and ``permits`` deliberately re-raises those. Its ``False``
    branch was unreachable dead code and the protection came from an outer ``except Exception``,
    which logged a false error and then still published the pending bookend."""
    fsm = ConflictLifecyclePolicy()

    class _Rec:
        def __init__(self, state: ConflictState) -> None:
            self.state = state

    for settled in (
        ConflictState.AUTO_RESOLVED,
        ConflictState.RESOLVED,
        ConflictState.DISMISSED,
    ):
        assert fsm.may_repark(_Rec(settled), ConflictState.MANUAL_PENDING) is False
    for live in (ConflictState.DETECTED, ConflictState.MANUAL_PENDING):
        assert fsm.may_repark(_Rec(live), ConflictState.MANUAL_PENDING) is True


async def test_re_detecting_an_answered_conflict_emits_nothing_and_logs_no_error(
    ns: ContractNamespace, engine_ns: Namespace, caplog: pytest.LogCaptureFixture
) -> None:
    """The measured shipped behaviour on a RESOLVED record: a ``conflict_record_park_failed``
    error, a fresh ``ConflictResolutionPending`` on the bus, and a verdict whose
    ``conflict_record.state`` came back ``manual_pending`` while the store said ``resolved``."""
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    clock = FrozenClock(_T0)
    adj = _adjudicator(records, bus, clock=clock)
    winner = _fact(engine_ns, memory_id="a", obj="Paris", created_at=_T0)
    cand = _fact(engine_ns, memory_id="b", obj="Berlin", created_at=_T0 + timedelta(days=1))
    first = await _sweep(adj, ns, winner, cand)
    conflict_id = first.conflict_record.conflict_id  # type: ignore[union-attr]

    settled = (await records.get(ns, conflict_id)).model_copy(  # type: ignore[union-attr]
        update={"state": ConflictState.RESOLVED, "resolved_winner_id": "b"}
    )
    await records.upsert(settled)
    bus.events.clear()

    again = await _sweep(adj, ns, winner, cand)

    assert again.conflict_record.state is ConflictState.RESOLVED  # type: ignore[union-attr]
    assert (await records.get(ns, conflict_id)).state is ConflictState.RESOLVED  # type: ignore[union-attr]
    assert bus.pending() == []
    assert "conflict_record_park_failed" not in caplog.text


# ══════════════ 3. THE AUTOMATIC LANE LEAVES AN AUDIT TRAIL (and no phantom inbox) ══════════
async def test_an_automatically_applied_conflict_still_opens_a_record(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """``if not apply:`` guarded the ONLY record write, so an auto-superseded memory had zero
    conflict aggregate: no id to deep-link, no ``policy_snapshot``, and ``DETECTED`` /
    ``AUTO_RESOLVED`` / ``ResolutionOrigin.AUTO`` unreachable in production."""
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    adj = _adjudicator(
        records,
        bus,
        clock=FrozenClock(_T0),
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.AUTOMATIC),
    )
    winner = _fact(engine_ns, memory_id="a", obj="Paris", created_at=_T0)
    cand = _fact(engine_ns, memory_id="b", obj="Berlin", created_at=_T0 + timedelta(days=1))

    verdict = await _sweep(adj, ns, winner, cand)

    assert verdict.apply is True  # type: ignore[union-attr]
    record = verdict.conflict_record  # type: ignore[union-attr]
    assert record is not None
    assert record.state is ConflictState.DETECTED  # nothing superseded YET — never a lie
    assert record.policy_snapshot  # the audit fact the auto lane previously had none of
    assert await records.get(ns, record.conflict_id) is not None


async def test_the_automatic_lane_does_not_put_a_phantom_in_the_inbox(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """A ``DETECTED`` record is an audit row, not a request for attention: publishing "waiting
    for you" for a conflict the machine resolves itself would be a permanent phantom on the
    surface §8 line 278 routes the event to."""
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    adj = _adjudicator(
        records,
        bus,
        clock=FrozenClock(_T0),
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.AUTOMATIC),
    )
    await _sweep(
        adj,
        ns,
        _fact(engine_ns, memory_id="a", obj="Paris", created_at=_T0),
        _fact(engine_ns, memory_id="b", obj="Berlin", created_at=_T0 + timedelta(days=1)),
    )
    assert bus.pending() == []
    assert await records.pending(ns) == []


# ═══════════════════ 4. REOPENED IS VISIBLE TO THE ONLY SURFACE THAT CAN SHOW IT ════════════
async def test_pending_includes_reopened(ns: ContractNamespace, engine_ns: Namespace) -> None:
    """Spec §5 line 197 defines pending as ``{MANUAL_PENDING, REOPENED}``. Filtering to
    MANUAL_PENDING made ``reopen()`` write a state that nothing could ever see again: the inbox
    reads ``pending()`` then intersects with ``ACTIONABLE_STATES``, an intersection that could
    never contain REOPENED, so a reopened conflict was decidable only by someone who already knew
    its id."""
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    adj = _adjudicator(records, bus, clock=FrozenClock(_T0))
    verdict = await _sweep(
        adj,
        ns,
        _fact(engine_ns, memory_id="a", obj="Paris", created_at=_T0),
        _fact(engine_ns, memory_id="b", obj="Berlin", created_at=_T0 + timedelta(days=1)),
    )
    conflict_id = verdict.conflict_record.conflict_id  # type: ignore[union-attr]
    reopened = (await records.get(ns, conflict_id)).model_copy(  # type: ignore[union-attr]
        update={"state": ConflictState.REOPENED}
    )
    await records.upsert(reopened)

    assert [r.conflict_id for r in await records.pending(ns)] == [conflict_id]


# ═════════════ 5. THE §4.1 RESOLVER IS REACHABLE, AND ITS FAILURE PARKS ════════════════════
def test_the_factory_forwards_a_policy_resolver() -> None:
    """Without this parameter §4.1 most-specific-wins was structurally unreachable: this factory
    is the only construction seam the composition roots use, so every conflict in every shipped
    deployment was governed by a hardcoded ``ConflictResolutionPolicy()``."""
    import inspect

    resolver = _ExplodingResolver()
    assert "policy_resolver" in inspect.signature(build_conflict_adjudicator).parameters
    adj = build_conflict_adjudicator(
        use_llm=True,
        router=object(),  # type: ignore[arg-type]
        policy_resolver=resolver,  # type: ignore[arg-type]
    )
    assert adj is not None
    assert adj._policy_resolver is resolver


async def test_a_policy_store_failure_parks_instead_of_killing_the_sweep(
    ns: ContractNamespace, engine_ns: Namespace
) -> None:
    """Two wrong answers were on the table. Falling back to the (AUTOMATIC) default would
    auto-supersede a fact in a namespace the owner had set MANUAL because a store hiccupped.
    Propagating would abort the whole namespace sweep tick and discard the extracted window —
    from inside a bare list comprehension with no per-candidate guard, while the conflict-record
    write twelve lines below is deliberately best-effort. Degrading the POLICY to MANUAL parks:
    it writes nothing, destroys nothing, and names the degrade on the bus."""
    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    adj = _adjudicator(
        records,
        bus,
        clock=FrozenClock(_T0),
        policy=ConflictResolutionPolicy(mode=ConflictResolutionMode.AUTOMATIC),
        policy_resolver=_ExplodingResolver(),
    )

    verdict = await _sweep(
        adj,
        ns,
        _fact(engine_ns, memory_id="a", obj="Paris", created_at=_T0),
        _fact(engine_ns, memory_id="b", obj="Berlin", created_at=_T0 + timedelta(days=1)),
    )

    assert verdict.apply is False  # parked, not auto-applied on a guessed policy
    assert len(await records.pending(ns)) == 1
    degrades = [e for e in bus.events if isinstance(e, DegradedModeEntered)]
    assert any(e.detail == "policy_resolver_failed" for e in degrades)
