"""§5 — the manual-resolution write actions. Offline: in-process repo + queue, frozen clock.

Covers conflict-resolution-async-design.md §5 (lines 207-220) and §8 (lines 279-281).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.errors import ConflictUnresolvedError, NamespaceIsolationError
from mu_contracts.domain.events import (
    ConflictDismissed,
    ConflictPolicyChanged,
    ConflictResolved,
    DomainEvent,
)
from mu_contracts.domain.model.conflict import (
    ConflictRecord,
    ConflictResolutionKind,
    ConflictResolutionMode,
    ConflictState,
    ResolutionOrigin,
)
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.lifecycle.conflict import (
    ConflictResolutionPolicy,
    InMemoryConflictRecordRepository,
)
from mu_engine.lifecycle.conflict_policy import TransitionTrigger
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.conflict.ports import (
    InMemoryConflictResolutionQueue,
    InMemoryMemoryConflictPolicyStore,
    InMemoryNamespaceConflictPolicyStore,
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
    queue: InMemoryConflictResolutionQueue,
    bus: _RecordingBus | None = None,
    **kw: object,
) -> ConflictResolutionService:
    return ConflictResolutionService(
        records=records, queue=queue, clock=FrozenClock(_T1), bus=bus, **kw
    )  # type: ignore[arg-type]


# ═══════════════════════════════════ 1. OFF THE WRITE PATH ══════════════════════════════════
def test_the_service_structurally_cannot_supersede_anything() -> None:
    """§1 invariant 1 / §2 line 54. This is asserted on the CONSTRUCTOR, not on behaviour: the
    guarantee is that no tier repository, no writer lease and no model router can be injected,
    so no future edit can make a caller wait on resolution without changing this signature."""
    import inspect

    params = set(inspect.signature(ConflictResolutionService.__init__).parameters)
    assert params == {
        "self",
        "records",
        "queue",
        "clock",
        "bus",
        "tenancy",
        "namespace_policies",
        "memory_policies",
    }


async def test_resolve_records_the_intent_and_enqueues_without_applying(
    ns: Namespace, scope: ClientScope
) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    queue = InMemoryConflictResolutionQueue()

    resolved = await _service(records, queue).resolve(
        scope,
        ns,
        "c1",
        ManualDecision(
            kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="principal-u1"
        ),
    )

    assert resolved.state is ConflictState.RESOLVED
    assert resolved.resolution_kind is ConflictResolutionKind.SUPERSEDE
    assert resolved.resolved_winner_id == "a"
    assert resolved.resolution_origin is ResolutionOrigin.MANUAL
    assert resolved.resolved_by == "principal-u1"
    assert resolved.resolved_at == _T1

    queued = await queue.drain(ns)
    assert len(queued) == 1
    assert queued[0].winner_id == "a"
    assert queued[0].loser_ids == ("b",)
    assert queued[0].kind is ConflictResolutionKind.SUPERSEDE


async def test_the_durable_intent_survives_on_the_record_not_only_in_the_queue(
    ns: Namespace, scope: ClientScope
) -> None:
    """Spec line 218: *"the record's ``resolution_*`` fields are the durable intent"*. If the
    decision lived only in the queue, a crash between enqueue and apply would lose a human's
    answer with nothing to replay from."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    await _service(records, InMemoryConflictResolutionQueue()).resolve(
        scope,
        ns,
        "c1",
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="u1"),
    )

    persisted = await records.get(ns, "c1")
    assert persisted is not None
    assert persisted.resolved_winner_id == "a"
    assert persisted.resolution_origin is ResolutionOrigin.MANUAL


async def test_a_dismissal_enqueues_nothing_at_all(ns: Namespace, scope: ClientScope) -> None:
    """ "Not a conflict" means no supersession is ever applied — an enqueued dismissal would give
    the resolve stage something to write."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    queue = InMemoryConflictResolutionQueue()

    dismissed = await _service(records, queue).resolve(
        scope, ns, "c1", ManualDecision(kind=ManualDecisionKind.DISMISS, resolved_by="u1")
    )

    assert dismissed.state is ConflictState.DISMISSED
    assert dismissed.resolution_kind is None
    assert len(queue) == 0


async def test_a_retried_decision_does_not_enqueue_two_supersessions(
    ns: Namespace, scope: ClientScope
) -> None:
    """A double-clicked button or a replayed IPC frame must not supersede twice."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    queue = InMemoryConflictResolutionQueue()
    decision = ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="u1")
    service = _service(records, queue)
    await service.resolve(scope, ns, "c1", decision)
    with pytest.raises(ConflictUnresolvedError):
        await service.resolve(scope, ns, "c1", decision)
    assert len(queue) == 1


# ═══════════════════════════════════════ 2. THE FSM GATE ════════════════════════════════════
@pytest.mark.parametrize(
    "state",
    [
        ConflictState.DETECTED,
        ConflictState.AUTO_RESOLVED,
        ConflictState.RESOLVED,
        ConflictState.DISMISSED,
    ],
)
async def test_a_conflict_not_awaiting_a_decision_is_refused(
    ns: Namespace, scope: ClientScope, state: ConflictState
) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns, state))
    with pytest.raises(ConflictUnresolvedError, match="not awaiting a decision"):
        await _service(records, InMemoryConflictResolutionQueue()).resolve(
            scope,
            ns,
            "c1",
            ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="u1"),
        )


async def test_a_reopened_conflict_is_decidable(ns: Namespace, scope: ClientScope) -> None:
    """Spec line 218 names REOPENED as resolvable and line 197 puts it in the inbox's pending
    set — refusing the edge would render the inbox's own contents undecidable."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns, ConflictState.REOPENED))
    resolved = await _service(records, InMemoryConflictResolutionQueue()).resolve(
        scope,
        ns,
        "c1",
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="u1"),
    )
    assert resolved.state is ConflictState.RESOLVED


async def test_a_winner_that_is_not_a_member_is_refused(ns: Namespace, scope: ClientScope) -> None:
    """Without this the stage would supersede BOTH real members in favour of an unrelated
    memory."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    with pytest.raises(ConflictUnresolvedError, match="not a member"):
        await _service(records, InMemoryConflictResolutionQueue()).resolve(
            scope,
            ns,
            "c1",
            ManualDecision(
                kind=ManualDecisionKind.SUPERSEDE, winner_id="somebody-else", resolved_by="u1"
            ),
        )


async def test_keep_both_may_not_name_a_winner(ns: Namespace, scope: ClientScope) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    with pytest.raises(ConflictUnresolvedError, match="does not take a winner_id"):
        await _service(records, InMemoryConflictResolutionQueue()).resolve(
            scope,
            ns,
            "c1",
            ManualDecision(kind=ManualDecisionKind.KEEP_BOTH, winner_id="a", resolved_by="u1"),
        )


async def test_merge_requires_a_reference_to_the_composed_draft(
    ns: Namespace, scope: ClientScope
) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    with pytest.raises(ConflictUnresolvedError, match="merged_text_ref"):
        await _service(records, InMemoryConflictResolutionQueue()).resolve(
            scope,
            ns,
            "c1",
            ManualDecision(kind=ManualDecisionKind.MERGE, winner_id="a", resolved_by="u1"),
        )


# ══════════════════════════════════════════ 3. AUTHZ ════════════════════════════════════════
async def test_a_caller_outside_the_partition_is_refused_non_enumeratingly(
    ns: Namespace,
) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    intruder = ClientScope(
        principal_id="u2",
        org_id="org2",
        workspace_id="ws2",
        session_id="s9",
        agent_principal_id="u2",
    )
    with pytest.raises(NamespaceIsolationError) as excinfo:
        await _service(records, InMemoryConflictResolutionQueue()).resolve(
            intruder,
            ns,
            "c1",
            ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="u2"),
        )
    assert "c1" not in str(excinfo.value), "a denial must not echo the requested id back"


async def test_an_unknown_conflict_is_refused_without_echoing_the_id(
    ns: Namespace, scope: ClientScope
) -> None:
    with pytest.raises(ConflictUnresolvedError) as excinfo:
        await _service(
            InMemoryConflictRecordRepository(), InMemoryConflictResolutionQueue()
        ).resolve(
            scope,
            ns,
            "no-such-conflict",
            ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="u1"),
        )
    assert str(excinfo.value) == "not found"


# ══════════════════════════════════════════ 4. REOPEN ═══════════════════════════════════════
async def test_an_automatic_reopen_of_a_dismissed_conflict_needs_a_changed_hash(
    ns: Namespace, scope: ClientScope
) -> None:
    """§3 line 106. Without the gate, a source that keeps re-asserting a dismissed fact would
    resurrect the conflict on every sweep and the dismiss button would mean nothing."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns, ConflictState.DISMISSED))
    service = _service(records, InMemoryConflictResolutionQueue())

    unchanged = await service.reopen(
        scope,
        ns,
        "c1",
        trigger=TransitionTrigger.AUTOMATIC,
        current_member_hashes=("h_a", "h_b"),
    )
    assert unchanged.state is ConflictState.DISMISSED

    changed = await service.reopen(
        scope,
        ns,
        "c1",
        trigger=TransitionTrigger.AUTOMATIC,
        current_member_hashes=("h_a", "h_b_NEW"),
    )
    assert changed.state is ConflictState.REOPENED


async def test_reopening_keeps_the_human_decision_on_the_record(
    ns: Namespace, scope: ClientScope
) -> None:
    """§7 line 262: a later automatic delta may ASK the human again; it may never flip their
    answer. Clearing ``resolution_origin`` on reopen would drop the stickiness term that stops
    another replica from re-deriving the automatic winner."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    service = _service(records, InMemoryConflictResolutionQueue())
    await service.resolve(
        scope,
        ns,
        "c1",
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="u1"),
    )

    reopened = await service.reopen(scope, ns, "c1")

    assert reopened.state is ConflictState.REOPENED
    assert reopened.resolution_origin is ResolutionOrigin.MANUAL
    assert reopened.resolved_winner_id == "a"


# ══════════════════════════════════════════ 5. EVENTS ═══════════════════════════════════════
async def test_a_supersede_decision_emits_conflict_resolved(
    ns: Namespace, scope: ClientScope
) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    bus = _RecordingBus()
    await _service(records, InMemoryConflictResolutionQueue(), bus).resolve(
        scope,
        ns,
        "c1",
        ManualDecision(kind=ManualDecisionKind.SUPERSEDE, winner_id="a", resolved_by="u1"),
    )
    (event,) = bus.events
    assert isinstance(event, ConflictResolved)
    assert event.winner_id == "a"
    assert event.loser_ids == ["b"]
    assert event.resolution_origin is ResolutionOrigin.MANUAL


async def test_a_dismissal_emits_conflict_dismissed(ns: Namespace, scope: ClientScope) -> None:
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    bus = _RecordingBus()
    await _service(records, InMemoryConflictResolutionQueue(), bus).resolve(
        scope, ns, "c1", ManualDecision(kind=ManualDecisionKind.DISMISS, resolved_by="u1")
    )
    (event,) = bus.events
    assert isinstance(event, ConflictDismissed)
    assert event.by == "u1"


async def test_a_winnerless_resolution_emits_no_false_winner(
    ns: Namespace, scope: ClientScope
) -> None:
    """REPORTED contract gap: the ratified ``ConflictResolved`` declares ``winner_id`` REQUIRED,
    so KEEP_BOTH cannot be expressed. Emitting an empty winner would put a lie on the bus, so
    nothing is emitted — the record still carries the outcome."""
    records = InMemoryConflictRecordRepository()
    await records.add(_record(ns))
    bus = _RecordingBus()
    resolved = await _service(records, InMemoryConflictResolutionQueue(), bus).resolve(
        scope, ns, "c1", ManualDecision(kind=ManualDecisionKind.KEEP_BOTH, resolved_by="u1")
    )
    assert resolved.resolution_kind is ConflictResolutionKind.KEEP_BOTH
    assert bus.events == []


async def test_setting_a_namespace_policy_emits_conflict_policy_changed(
    ns: Namespace, scope: ClientScope
) -> None:
    bus = _RecordingBus()
    store = InMemoryNamespaceConflictPolicyStore()
    service = _service(
        InMemoryConflictRecordRepository(),
        InMemoryConflictResolutionQueue(),
        bus,
        namespace_policies=store,
    )
    policy = ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL)

    await service.set_namespace_policy(scope, ns, policy)

    assert await store.policy_for(ns) == policy
    (event,) = bus.events
    assert isinstance(event, ConflictPolicyChanged)
    assert event.policy == "manual"


async def test_setting_a_policy_with_no_store_wired_fails_loud(
    ns: Namespace, scope: ClientScope
) -> None:
    """A silent no-op here would tell the owner their namespace is MANUAL while every conflict
    kept auto-resolving."""
    service = _service(InMemoryConflictRecordRepository(), InMemoryConflictResolutionQueue())
    with pytest.raises(ConflictUnresolvedError, match="no namespace conflict-policy store"):
        await service.set_namespace_policy(scope, ns, ConflictResolutionPolicy())
    with pytest.raises(ConflictUnresolvedError, match="no writable per-memory"):
        await service.set_memory_policy(scope, ns, "a", ConflictResolutionPolicy())


async def test_a_memory_override_can_be_set_and_cleared(ns: Namespace, scope: ClientScope) -> None:
    store = InMemoryMemoryConflictPolicyStore()
    service = _service(
        InMemoryConflictRecordRepository(),
        InMemoryConflictResolutionQueue(),
        memory_policies=store,
    )
    policy = ConflictResolutionPolicy(mode=ConflictResolutionMode.MANUAL)

    await service.set_memory_policy(scope, ns, "a", policy)
    assert await store.override_for(ns, "a") == policy

    await service.set_memory_policy(scope, ns, "a", None)
    assert await store.override_for(ns, "a") is None
