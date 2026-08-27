"""§6 — how a still-pending conflict affects RECALL. Offline: pure, no store, no clock.

Covers conflict-resolution-async-design.md §6 (lines 226-253).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.events import DegradedModeEntered, DegradeReason, DomainEvent
from mu_contracts.domain.model.conflict import (
    ConflictEdgeRow,
    ConflictEdges,
    ConflictState,
    PendingRecallMode,
)
from mu_engine.lifecycle.conflict import ConflictResolutionPolicy
from mu_engine.services.conflict.recall import PendingConflictRecallPolicy
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


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


def _edges(
    pairs: dict[str, tuple[str, ...]],
    *,
    conflict_id: str = "c1",
    state: ConflictState = ConflictState.MANUAL_PENDING,
) -> ConflictEdges:
    return ConflictEdges(
        rows_by_memory={
            memory_id: ConflictEdgeRow(
                memory_id=memory_id,
                peer_ids=frozenset(peers),
                conflict_id=conflict_id,
                state=state,
                detected_confidence=0.6,
            )
            for memory_id, peers in pairs.items()
        }
    )


def _policy(mode: PendingRecallMode) -> ConflictResolutionPolicy:
    return ConflictResolutionPolicy(manual_recall_mode=mode)


# ════════════════════════════════════ SURFACE_BOTH_MARKED (default) ═════════════════════════
def test_the_default_is_the_honest_one() -> None:
    assert ConflictResolutionPolicy().manual_recall_mode is PendingRecallMode.SURFACE_BOTH_MARKED


async def test_both_members_surface_each_marked(ns: Namespace) -> None:
    """The read-time analogue of "never delete": at recall, never PRETEND the conflict is
    decided (spec line 248)."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    outcome = await PendingConflictRecallPolicy().apply(
        [a, b],
        _edges({"a": ("b",), "b": ("a",)}),
        policy=_policy(PendingRecallMode.SURFACE_BOTH_MARKED),
    )

    assert {i.id for i in outcome.surfaced} == {"a", "b"}, "never a silent drop"
    assert outcome.suppressed_ids == ()
    for memory_id, peer in (("a", "b"), ("b", "a")):
        annotation = outcome.annotation_for(memory_id)
        assert annotation is not None
        assert annotation.conflict_pending is True
        assert annotation.conflict_id == "c1"
        assert annotation.conflict_peer_ids == (peer,)
        assert annotation.is_provisional_winner is False, "nothing has been preferred"


async def test_an_unconflicted_hit_is_untouched_and_unannotated(ns: Namespace) -> None:
    """AUTOMATIC mode needs none of this (spec line 251) — an item nobody is arguing about must
    not pay for the feature."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    quiet = _fact(ns, memory_id="quiet", obj="Oslo", created_at=_T0)

    outcome = await PendingConflictRecallPolicy().apply(
        [a, quiet], ConflictEdges(), policy=_policy(PendingRecallMode.SURFACE_BOTH_MARKED)
    )

    assert outcome.surfaced == (a, quiet)
    assert outcome.annotations == ()


async def test_a_resolved_conflict_leaves_the_annotations_dark(ns: Namespace) -> None:
    """Under an APPLIED resolution the loser is already superseded and the §7.5 floor excludes
    it; ``ConflictEdges.unresolved_for`` is False, so nothing is marked (spec line 251)."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0)

    outcome = await PendingConflictRecallPolicy().apply(
        [a, b],
        _edges({"a": ("b",), "b": ("a",)}, state=ConflictState.RESOLVED),
        policy=_policy(PendingRecallMode.SURFACE_BOTH_MARKED),
    )

    assert outcome.annotations == ()
    assert len(outcome.surfaced) == 2


# ═══════════════════════════════════════ PREFER_PROVISIONAL ═════════════════════════════════
async def test_only_the_provisional_winner_surfaces_and_is_marked_as_such(
    ns: Namespace,
) -> None:
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    outcome = await PendingConflictRecallPolicy().apply(
        [a, b],
        _edges({"a": ("b",), "b": ("a",)}),
        policy=_policy(PendingRecallMode.PREFER_PROVISIONAL),
    )

    assert [i.id for i in outcome.surfaced] == ["b"], "the §4.2 total order's winner"
    assert outcome.suppressed_ids == ("a",)
    winner_annotation = outcome.annotation_for("b")
    assert winner_annotation is not None
    assert winner_annotation.is_provisional_winner is True
    assert winner_annotation.conflict_pending is True, "still honest that it is contested"


async def test_the_provisional_pick_is_the_same_deterministic_order(ns: Namespace) -> None:
    """Two devices under PREFER_PROVISIONAL must show the SAME item — that is the whole reason
    it is safe to prefer one before the human has chosen."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))
    edges = _edges({"a": ("b",), "b": ("a",)})
    policy = _policy(PendingRecallMode.PREFER_PROVISIONAL)

    forward = await PendingConflictRecallPolicy().apply([a, b], edges, policy=policy)
    backward = await PendingConflictRecallPolicy().apply([b, a], edges, policy=policy)

    assert [i.id for i in forward.surfaced] == [i.id for i in backward.surfaced]


async def test_preferring_a_winner_suppresses_but_never_invalidates(ns: Namespace) -> None:
    """Spec line 249: a provisional preference is a READ-TIME RANKING, not a write. The loser
    must come back from this call in exactly the state it went in."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))

    await PendingConflictRecallPolicy().apply(
        [a, b],
        _edges({"a": ("b",), "b": ("a",)}),
        policy=_policy(PendingRecallMode.PREFER_PROVISIONAL),
    )

    assert a.state is MemoryState.ACTIVE
    assert a.invalid_at is None
    assert b.state is MemoryState.ACTIVE


async def test_a_lone_member_on_the_page_still_surfaces(ns: Namespace) -> None:
    """Its peer is off-page, so there is nothing to prefer it OVER; suppressing it would
    withhold more than the policy asked for."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)

    outcome = await PendingConflictRecallPolicy().apply(
        [a], _edges({"a": ("off-page-peer",)}), policy=_policy(PendingRecallMode.PREFER_PROVISIONAL)
    )

    assert [i.id for i in outcome.surfaced] == ["a"]
    assert outcome.suppressed_ids == ()


# ═════════════════════════════════════════ SUPPRESS_BOTH ════════════════════════════════════
async def test_suppress_both_withholds_everything_and_emits_the_named_degrade(
    ns: Namespace,
) -> None:
    """Spec line 250: a silent suppression is indistinguishable from "you have no memory about
    that", so it must be observable."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))
    bus = _RecordingBus()

    outcome = await PendingConflictRecallPolicy(bus=bus).apply(
        [a, b],
        _edges({"a": ("b",), "b": ("a",)}),
        policy=_policy(PendingRecallMode.SUPPRESS_BOTH),
    )

    assert outcome.surfaced == ()
    assert outcome.suppressed_ids == ("a", "b")
    (event,) = bus.events
    assert isinstance(event, DegradedModeEntered)
    assert event.reason is DegradeReason.CONFLICT_PENDING_SUPPRESSED
    assert event.component == "recall"
    assert event.detail == "withheld=2"


async def test_suppress_both_keeps_the_unconflicted_hits(ns: Namespace) -> None:
    """The mode withholds the CONTESTED pair, not the whole result page."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0)
    quiet = _fact(ns, memory_id="quiet", obj="Oslo", created_at=_T0)

    outcome = await PendingConflictRecallPolicy().apply(
        [a, quiet, b],
        _edges({"a": ("b",), "b": ("a",)}),
        policy=_policy(PendingRecallMode.SUPPRESS_BOTH),
    )

    assert [i.id for i in outcome.surfaced] == ["quiet"]


async def test_no_degrade_is_emitted_when_nothing_was_actually_suppressed(
    ns: Namespace,
) -> None:
    bus = _RecordingBus()
    quiet = _fact(ns, memory_id="quiet", obj="Oslo", created_at=_T0)
    await PendingConflictRecallPolicy(bus=bus).apply(
        [quiet], ConflictEdges(), policy=_policy(PendingRecallMode.SUPPRESS_BOTH)
    )
    assert bus.events == []


async def test_prefer_provisional_does_not_emit_the_suppression_degrade(
    ns: Namespace,
) -> None:
    """The two modes withhold for different reasons; ``CONFLICT_PENDING_SUPPRESSED`` names
    SUPPRESS_BOTH specifically (spec line 250) and must not fire for a ranking."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))
    bus = _RecordingBus()
    await PendingConflictRecallPolicy(bus=bus).apply(
        [a, b],
        _edges({"a": ("b",), "b": ("a",)}),
        policy=_policy(PendingRecallMode.PREFER_PROVISIONAL),
    )
    assert bus.events == []
