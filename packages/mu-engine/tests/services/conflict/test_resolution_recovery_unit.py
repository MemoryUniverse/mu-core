"""The resolve hand-off: recoverable, atomic, content-free, and honest about KEEP_BOTH.

Every test here pins a defect that shipped GREEN through the lane's other 161 tests, so each one
names the failure it prevents rather than the happy path it walks.

Covers conflict-resolution-async-design.md §1 invariant 1, §3 line 107 (content-free), §5 lines
211-218 (the resolve actions and "does not execute inline"), §6 (keep-both coexistence).
Offline: in-process repo + queue, frozen clock, no store, no network, no keys.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from mu_contracts.domain.errors import (
    ConflictUnresolvedError,
    IllegalConflictTransitionError,
)
from mu_contracts.domain.events import ConflictResolved, DomainEvent
from mu_contracts.domain.model.conflict import (
    ConflictRecord,
    ConflictResolutionKind,
    ConflictState,
    ContentFreeModel,
    ResolutionOrigin,
)
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.lifecycle.conflict import (
    ConflictResolutionPolicy,
    InMemoryConflictRecordRepository,
    awaits_apply,
)
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.conflict.ports import (
    InMemoryConflictResolutionQueue,
    RecordBackedResolutionQueue,
    ResolutionIntent,
)
from mu_engine.services.conflict.resolution import (
    ConflictResolutionService,
    ManualDecision,
    ManualDecisionKind,
)

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)
_T1 = datetime(2026, 6, 2, tzinfo=UTC)


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


def _record(ns: Namespace, state: ConflictState = ConflictState.MANUAL_PENDING) -> ConflictRecord:
    return ConflictRecord(
        conflict_id="c1",
        namespace=ns,
        member_ids=("a", "b"),
        member_content_hashes=("h_a", "h_b"),
        predicate_key="lives_in",
        method="llm_adjudicator",
        detected_confidence=0.6,
        proposed_winner_id="b",
        state=state,
        detected_at=_T0,
        policy_snapshot=ConflictResolutionPolicy().snapshot(),
    )


def _service(
    records: InMemoryConflictRecordRepository,
    queue: object,
    bus: _RecordingBus | None = None,
) -> ConflictResolutionService:
    return ConflictResolutionService(
        records=records,
        queue=queue,  # type: ignore[arg-type]
        clock=FrozenClock(_T1),
        bus=bus,
    )


# ═══════════════ 1. THE DECISION SURVIVES THE PROCESS (the queue is recoverable) ═════════════
async def test_a_recorded_decision_is_re_derivable_after_the_queue_is_lost(
    ns: Namespace, scope: ClientScope
) -> None:
    """The shipped hand-off was: flip the record terminal, then hand the intent to a plain dict.

    Nothing drained that dict, and nothing could rebuild it: ``ACTIONABLE_STATES`` excludes
    ``RESOLVED``, so ``_load_actionable`` refused every retry and the FSM routes ``RESOLVED``
    only to ``REOPENED``. A crash after the upsert left a record that SAID resolved while both
    items stayed active, unreachable from every surface. Recovery must come off the DURABLE
    record, not off the volatile queue.
    """
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    volatile = InMemoryConflictResolutionQueue()
    await _service(records, volatile).resolve(
        scope,
        ns,
        "c1",
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="b", resolved_by="u1"),
    )

    # the process dies: the in-memory queue is gone entirely.
    del volatile

    recovered = await RecordBackedResolutionQueue(records).drain(ns)
    assert [i.conflict_id for i in recovered] == ["c1"]
    assert recovered[0].winner_id == "b"
    assert recovered[0].loser_ids == ("a",)


async def test_marking_applied_is_what_takes_a_decision_out_of_the_recovery_set(
    ns: Namespace, scope: ClientScope
) -> None:
    """At-least-once delivery needs a durable stopping condition. ``resolution_applied_at`` is
    it: until the stage stamps it, every drain hands the intent back."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    service = _service(records, InMemoryConflictResolutionQueue())
    await service.resolve(
        scope,
        ns,
        "c1",
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="b", resolved_by="u1"),
    )
    queue = RecordBackedResolutionQueue(records)

    assert len(await queue.drain(ns)) == 1
    assert len(await queue.drain(ns)) == 1  # non-destructive: a lost batch comes back

    applied = await service.mark_applied(ns, "c1")
    assert applied.resolution_applied_at == _T1
    assert await queue.drain(ns) == ()
    assert not awaits_apply(applied)


async def test_marking_applied_twice_does_not_re_stamp(ns: Namespace, scope: ClientScope) -> None:
    """A redelivered intent must be idempotent or the audit stamp drifts on every retry."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    service = _service(records, InMemoryConflictResolutionQueue())
    await service.resolve(
        scope,
        ns,
        "c1",
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="b", resolved_by="u1"),
    )
    clock = FrozenClock(_T1)
    service = ConflictResolutionService(
        records=records,
        queue=InMemoryConflictResolutionQueue(),
        clock=clock,
    )
    first = await service.mark_applied(ns, "c1")
    clock.advance(timedelta(hours=6))  # a MOVING clock, or "idempotent" proves nothing
    second = await service.mark_applied(ns, "c1")
    assert first.resolution_applied_at == _T1
    assert second == first


async def test_a_dismissal_never_enters_the_recovery_set(ns: Namespace, scope: ClientScope) -> None:
    """A dismissal asserts there was never a conflict — there is nothing for a stage to apply,
    so a record waiting forever for one would be a permanent phantom in the queue."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    await _service(records, InMemoryConflictResolutionQueue()).resolve(
        scope, ns, "c1", ManualDecision(kind=ManualDecisionKind.DISMISS, resolved_by="u1")
    )
    assert await RecordBackedResolutionQueue(records).drain(ns) == ()


async def test_recovery_is_namespace_scoped(ns: Namespace, scope: ClientScope) -> None:
    """CLAUDE.md rule 4 — one tenant's undelivered decisions are never another's to apply."""
    other = ns.model_copy(update={"user": "u2"})
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    await _service(records, InMemoryConflictResolutionQueue()).resolve(
        scope,
        ns,
        "c1",
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="b", resolved_by="u1"),
    )
    assert await RecordBackedResolutionQueue(records).drain(other) == ()


# ══════════════════ 2. KEEP_BOTH DOES NOT ASK THE STAGE TO KILL BOTH ════════════════════════
async def test_keep_both_enqueues_no_apply_intent_at_all(ns: Namespace, scope: ClientScope) -> None:
    """The live defect: ``loser_ids`` was "every member that is not the winner", and KEEP_BOTH
    names no winner — so the durable payload told ``ResolveConflictStage``, whose job is to
    supersede ``loser_ids``, to invalidate BOTH members, from the one decision that means "both
    remain active" (spec §5 line 211, §6). The only prior KEEP_BOTH test asserted the record's
    ``resolution_kind`` and an empty bus and never drained the queue."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    queue = InMemoryConflictResolutionQueue()
    resolved = await _service(records, queue).resolve(
        scope, ns, "c1", ManualDecision(kind=ManualDecisionKind.KEEP_BOTH, resolved_by="u1")
    )
    assert resolved.resolution_kind is ConflictResolutionKind.KEEP_BOTH
    assert len(queue) == 0
    assert await queue.drain(ns) == ()
    assert await RecordBackedResolutionQueue(records).drain(ns) == ()


def test_a_winnerless_intent_cannot_name_losers_at_all(ns: Namespace) -> None:
    """The structural back-stop, so no future caller can rebuild the shape by hand."""
    with pytest.raises(ValidationError, match="cannot name losers"):
        ResolutionIntent(
            conflict_id="c1",
            namespace=ns,
            kind=ConflictResolutionKind.KEEP_BOTH,
            winner_id=None,
            loser_ids=("a", "b"),
            resolved_by="u1",
        )


def test_the_winner_cannot_also_be_a_loser(ns: Namespace) -> None:
    with pytest.raises(ValidationError, match="cannot also be a loser"):
        ResolutionIntent(
            conflict_id="c1",
            namespace=ns,
            kind=ConflictResolutionKind.SUPERSEDE,
            winner_id="a",
            loser_ids=("a",),
            resolved_by="u1",
        )


# ═════════════════ 3. THE MERGED REF IS A REF — CHECKED, NOT PROMISED ═══════════════════════
def test_the_queued_intent_carries_the_content_free_guard() -> None:
    """``ResolutionIntent`` is the DTO that travels furthest in this lane — a durable queue, a
    log, and (§7) a sync delta — and it was the one travelling conflict DTO that did not inherit
    the guard, and the one the lane's content-free test did not parametrize."""
    assert issubclass(ResolutionIntent, ContentFreeModel)


@pytest.mark.parametrize(
    "leak",
    [
        "The user lives in Berlin, passport A1234567.",
        "Berlin not Paris",
        "x" * 300,
        "",
    ],
)
def test_prose_cannot_enter_the_lane_through_merged_text_ref(ns: Namespace, leak: str) -> None:
    """The content-free guard is a field-NAME check and ``merged_text_ref`` walks past it. A
    runtime probe put a whole sentence, passport number included, into the queued payload
    verbatim. Refs have no whitespace; prose does."""
    with pytest.raises(ValidationError):
        ManualDecision(
            kind=ManualDecisionKind.MERGE,
            winner_id="b",
            merged_text_ref=leak,
            resolved_by="u1",
        )
    with pytest.raises(ValidationError):
        ResolutionIntent(
            conflict_id="c1",
            namespace=ns,
            kind=ConflictResolutionKind.MERGE,
            winner_id="b",
            loser_ids=("a",),
            merged_text_ref=leak,
            resolved_by="u1",
        )


def test_a_real_reference_still_passes(ns: Namespace) -> None:
    """The bound must not be so tight that the feature is unusable — an id/key shape works."""
    decision = ManualDecision(
        kind=ManualDecisionKind.MERGE,
        winner_id="b",
        merged_text_ref="draft:01a02061-d40f-7ff2",
        resolved_by="u1",
    )
    assert decision.merged_text_ref == "draft:01a02061-d40f-7ff2"


# ═══════════ 4. VALIDATE BEFORE THE POINT OF NO RETURN (resolve is atomic) ══════════════════
async def test_a_rejected_merge_ref_leaves_the_conflict_still_decidable(
    ns: Namespace, scope: ClientScope
) -> None:
    """The shipped order was upsert-then-build-intent, with a LOOSER bound on the decision than
    on the intent. An over-long ref therefore raised only AFTER the record was durably
    ``RESOLVED``: queue empty, no event, no log line (it sits after the enqueue), and
    ``_load_actionable`` refusing every retry. The human's decision was unrecoverable.

    Both bounds are now the same annotated type, so this input is refused at the boundary — and
    the ordering fix means that even a rejection deeper in the intent leaves the record
    untouched and the conflict still answerable."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    queue = InMemoryConflictResolutionQueue()
    service = _service(records, queue)

    with pytest.raises(ValidationError):
        await service.resolve(
            scope,
            ns,
            "c1",
            ManualDecision(
                kind=ManualDecisionKind.MERGE,
                winner_id="b",
                merged_text_ref="x" * 640,
                resolved_by="u1",
            ),
        )

    still = await records.get(ns, "c1")
    assert still is not None
    assert still.state is ConflictState.MANUAL_PENDING  # NOT terminal — still decidable
    assert len(queue) == 0
    # and the human can now answer it properly
    fixed = await service.resolve(
        scope,
        ns,
        "c1",
        ManualDecision(
            kind=ManualDecisionKind.MERGE,
            winner_id="b",
            merged_text_ref="draft-1",
            resolved_by="u1",
        ),
    )
    assert fixed.state is ConflictState.RESOLVED


async def test_nothing_is_committed_until_the_intent_has_been_built(
    ns: Namespace, scope: ClientScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ORDER is the contract, independent of which validation happens to fail today.

    ``resolve``'s step 4 docstring calls the record upsert "the point of no return", and it is:
    ``ACTIONABLE_STATES`` excludes ``RESOLVED``, so once the record is terminal
    ``_load_actionable`` refuses every retry and only an explicit ``reopen`` can recover it. Any
    failure while building the apply-intent must therefore happen BEFORE that write. A live
    ``merged_text_ref`` bound mismatch was one way in; this pins the invariant itself, so the
    next asymmetry someone introduces cannot silently strand a human's decision again."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    queue = InMemoryConflictResolutionQueue()

    def _boom(record: ConflictRecord) -> object:
        raise ValueError("intent rejected")

    monkeypatch.setattr("mu_engine.services.conflict.resolution.intent_from_record", _boom)

    with pytest.raises(ValueError, match="intent rejected"):
        await _service(records, queue).resolve(
            scope,
            ns,
            "c1",
            ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="b", resolved_by="u1"),
        )

    survived = await records.get(ns, "c1")
    assert survived is not None
    assert survived.state is ConflictState.MANUAL_PENDING  # still answerable
    assert survived.resolution_kind is None
    assert len(queue) == 0


async def test_a_recovered_merge_intent_still_knows_what_to_merge(
    ns: Namespace, scope: ClientScope
) -> None:
    """The ref rides on the RECORD, not only on the transient payload — otherwise an intent
    rebuilt after a crash would be a merge with nothing to merge."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    await _service(records, InMemoryConflictResolutionQueue()).resolve(
        scope,
        ns,
        "c1",
        ManualDecision(
            kind=ManualDecisionKind.MERGE,
            winner_id="b",
            merged_text_ref="draft-1",
            resolved_by="u1",
        ),
    )
    recovered = await RecordBackedResolutionQueue(records).drain(ns)
    assert recovered[0].merged_text_ref == "draft-1"


# ═══════════════════ 5. THE AUTOMATIC LANE CAN BE CLOSED, AND SAYS SO ═══════════════════════
async def test_the_automatic_lane_reaches_auto_resolved_and_emits_origin_auto(
    ns: Namespace,
) -> None:
    """Nothing in the tree ever wrote ``AUTO_RESOLVED`` or ``ResolutionOrigin.AUTO``: the
    adjudicator opened a record only on the PARK branch, so ``ConflictResolved`` could only ever
    say ``origin=manual`` and spec line 117's whole AUTO_RESOLVED-vs-RESOLVED distinction
    described an unreachable state."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns, ConflictState.DETECTED))
    bus = _RecordingBus()
    service = _service(records, InMemoryConflictResolutionQueue(), bus)
    resolved = await service.record_automatic_resolution(ns, "c1", winner_id="b")

    assert resolved.state is ConflictState.AUTO_RESOLVED
    assert resolved.resolution_origin is ResolutionOrigin.AUTO
    assert resolved.resolution_applied_at == _T1
    [event] = bus.events
    assert isinstance(event, ConflictResolved)
    assert event.resolution_origin is ResolutionOrigin.AUTO
    assert event.winner_id == "b"
    # applied under the same lease that decided it — never in the recovery set
    assert await RecordBackedResolutionQueue(records).drain(ns) == ()


async def test_a_human_surface_cannot_forge_an_automatic_resolution(
    ns: Namespace, scope: ClientScope
) -> None:
    """``AUTOMATIC_ONLY_EDGES`` only means something if a caller actually takes the edge with the
    right trigger. ``resolve`` is EXPLICIT and cannot reach ``AUTO_RESOLVED``; the record's
    ``resolution_origin`` audit is what that protects."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns, ConflictState.MANUAL_PENDING))
    service = _service(records, InMemoryConflictResolutionQueue())
    with pytest.raises(IllegalConflictTransitionError):
        await service.record_automatic_resolution(ns, "c1", winner_id="b")
    assert (await records.get(ns, "c1")).state is ConflictState.MANUAL_PENDING  # type: ignore[union-attr]


async def test_an_automatic_resolution_cannot_name_a_non_member(ns: Namespace) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns, ConflictState.DETECTED))
    service = _service(records, InMemoryConflictResolutionQueue())
    with pytest.raises(ConflictUnresolvedError, match="not a member"):
        await service.record_automatic_resolution(ns, "c1", winner_id="zzz")
