"""``ConflictLifecyclePolicy`` — the §3.1 conflict FSM. Offline, pure, no store, no clock.

Covers conflict-resolution-async-design.md §3.1 (lines 109-117) and the three recorded spec
deltas in the module docstring of ``mu_engine.lifecycle.conflict_policy``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.errors import IllegalConflictTransitionError
from mu_contracts.domain.model.conflict import ConflictRecord, ConflictState
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.lifecycle.conflict_policy import (
    ConflictLifecyclePolicy,
    ConflictTransitionBlockedError,
    TransitionTrigger,
)

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _record(
    ns: Namespace,
    state: ConflictState,
    *,
    hashes: tuple[str, ...] = (),
) -> ConflictRecord:
    return ConflictRecord(
        conflict_id="c1",
        namespace=ns,
        member_ids=("a", "b"),
        member_content_hashes=hashes,
        predicate_key="lives_in",
        method="polarity_cardinality_heuristic",
        detected_confidence=0.9,
        state=state,
        detected_at=_T0,
    )


@pytest.fixture
def policy() -> ConflictLifecyclePolicy:
    return ConflictLifecyclePolicy()


# ══════════════════════════════════════════ the spec §3.1 table, edge by edge ═══════════════
@pytest.mark.parametrize(
    ("current", "target", "trigger"),
    [
        (ConflictState.DETECTED, ConflictState.AUTO_RESOLVED, TransitionTrigger.AUTOMATIC),
        (ConflictState.DETECTED, ConflictState.MANUAL_PENDING, TransitionTrigger.AUTOMATIC),
        (ConflictState.MANUAL_PENDING, ConflictState.RESOLVED, TransitionTrigger.EXPLICIT),
        (ConflictState.MANUAL_PENDING, ConflictState.DISMISSED, TransitionTrigger.EXPLICIT),
        (ConflictState.AUTO_RESOLVED, ConflictState.REOPENED, TransitionTrigger.AUTOMATIC),
        (ConflictState.RESOLVED, ConflictState.REOPENED, TransitionTrigger.AUTOMATIC),
        (ConflictState.REOPENED, ConflictState.DETECTED, TransitionTrigger.AUTOMATIC),
    ],
)
def test_every_edge_the_design_table_draws_is_legal(
    ns: Namespace,
    policy: ConflictLifecyclePolicy,
    current: ConflictState,
    target: ConflictState,
    trigger: TransitionTrigger,
) -> None:
    policy.assert_transition(_record(ns, current), target, trigger=trigger)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        # a parked conflict may never be taken back by the machine (recorded delta 3)
        (ConflictState.MANUAL_PENDING, ConflictState.AUTO_RESOLVED),
        # nothing skips the decision
        (ConflictState.DETECTED, ConflictState.RESOLVED),
        (ConflictState.DETECTED, ConflictState.DISMISSED),
        # a settled conflict is not re-settled without a REOPEN
        (ConflictState.RESOLVED, ConflictState.DISMISSED),
        (ConflictState.AUTO_RESOLVED, ConflictState.RESOLVED),
        (ConflictState.DISMISSED, ConflictState.RESOLVED),
    ],
)
def test_an_edge_outside_the_table_raises_loud(
    ns: Namespace,
    policy: ConflictLifecyclePolicy,
    current: ConflictState,
    target: ConflictState,
) -> None:
    with pytest.raises(IllegalConflictTransitionError):
        policy.assert_transition(_record(ns, current), target)


def test_the_illegal_edge_message_names_the_edge_and_nothing_else(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    """Non-enumerating: a member id or a predicate in the message would put partition-shaped
    detail into whatever logged it (``LifecyclePolicy``'s own convention)."""
    with pytest.raises(IllegalConflictTransitionError) as excinfo:
        policy.assert_transition(_record(ns, ConflictState.DETECTED), ConflictState.RESOLVED)
    message = str(excinfo.value)
    assert "detected -> resolved" in message
    assert "a" not in message.split()  # no bare member id
    assert "lives_in" not in message
    assert "c1" not in message


# ═════════════════════════════════════════════════ trigger asymmetry ════════════════════════
def test_a_sweep_can_never_forge_a_human_decision(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    """MANUAL_PENDING -> RESOLVED is EXPLICIT-only. An AUTOMATIC caller taking it would mean a
    background sweep marking a conflict as decided by a human who never saw it."""
    record = _record(ns, ConflictState.MANUAL_PENDING)
    with pytest.raises(ConflictTransitionBlockedError):
        policy.assert_transition(
            record, ConflictState.RESOLVED, trigger=TransitionTrigger.AUTOMATIC
        )
    assert policy.permits(record, ConflictState.RESOLVED) is False


def test_a_human_surface_can_never_write_auto_resolved(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    """DETECTED -> AUTO_RESOLVED is AUTOMATIC-only: letting an explicit surface take it would
    destroy the ``resolution_origin`` audit spec line 117 makes the whole distinction."""
    with pytest.raises(ConflictTransitionBlockedError):
        policy.assert_transition(
            _record(ns, ConflictState.DETECTED),
            ConflictState.AUTO_RESOLVED,
            trigger=TransitionTrigger.EXPLICIT,
        )


def test_permits_swallows_the_soft_refusal_but_never_an_illegal_edge(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    """``LifecyclePolicy.permits``' asymmetry, mirrored: an expected trigger refusal is a bool,
    a caller bug is still an exception, so this helper can never hide one."""
    assert policy.permits(_record(ns, ConflictState.MANUAL_PENDING), ConflictState.RESOLVED) is (
        False
    )
    with pytest.raises(IllegalConflictTransitionError):
        policy.permits(_record(ns, ConflictState.DETECTED), ConflictState.RESOLVED)


# ═════════════════════════════════════════ idempotent re-detection ══════════════════════════
def test_re_asserting_the_same_state_is_a_legal_no_op(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    """``conflict_id`` is a hash of the members, so re-detection re-asserts MANUAL_PENDING on
    every sweep. Raising there would make an idempotent detector crash on its second run."""
    for state in ConflictState:
        policy.assert_transition(_record(ns, state), state)


# ═══════════════════════════════════ the dismiss-no-reopen rule (§3 line 106) ═══════════════
def test_a_dismissed_conflict_is_not_reopened_on_unchanged_hashes(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    record = _record(ns, ConflictState.DISMISSED, hashes=("h_a", "h_b"))
    assert policy.may_reopen_dismissed(record, ("h_a", "h_b")) is False


def test_a_genuinely_new_delta_does_reopen_a_dismissed_conflict(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    record = _record(ns, ConflictState.DISMISSED, hashes=("h_a", "h_b"))
    assert policy.may_reopen_dismissed(record, ("h_a", "h_b_CHANGED")) is True


def test_a_dismissed_record_with_no_recorded_hashes_fails_open(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    """Unknown prior state cannot PROVE the pair is unchanged, and the safe direction is to ask
    the human again rather than silently swallow a possibly-new contradiction."""
    record = _record(ns, ConflictState.DISMISSED)
    assert policy.may_reopen_dismissed(record, ("h_a", "h_b")) is True


def test_the_no_reopen_rule_only_governs_dismissed_records(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    record = _record(ns, ConflictState.RESOLVED, hashes=("h_a", "h_b"))
    assert policy.may_reopen_dismissed(record, ("h_a", "h_b")) is True


def test_is_actionable_is_exactly_the_inbox_pending_set(
    ns: Namespace, policy: ConflictLifecyclePolicy
) -> None:
    actionable = {s for s in ConflictState if policy.is_actionable(_record(ns, s))}
    assert actionable == {ConflictState.MANUAL_PENDING, ConflictState.REOPENED}
