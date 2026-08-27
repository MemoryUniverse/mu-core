"""``LifecyclePolicy`` — the central FSM guard + the pin clause (memory-health §6.1).

Pure unit test of isolated logic: the policy is stateless, has no I/O and no clock, so every
assertion here runs with zero infra. This is the file that pins the LOAD-BEARING behaviour of the
whole subsystem — if the guard permits an exit for a pinned item, every enforcement site below it
is decoration.

Deliberately asserts the SPLIT that CANONICAL forces on spec §6.1's flat ``_EXIT_STATES``:

* retention exits (ARCHIVED / EXPIRED / DELETED) are blocked on EVERY trigger — CANONICAL §7.10
  *"never garbage-collected regardless"*;
* adjudicated exits (SUPERSEDED / QUARANTINED) are blocked only on the AUTOMATIC trigger —
  CANONICAL §7.10 *"a pinned item can still be superseded, but it is never the auto-loser"*.

A test asserting the spec's flat rule would have locked in a CANONICAL violation, so the split is
asserted in both directions (blocked automatic AND permitted explicit).
"""

from __future__ import annotations

import pytest

from mu_contracts.domain.errors import IllegalTransitionError, PinnedTransitionBlocked
from mu_contracts.domain.model.memory import State
from mu_engine.lifecycle.policy import (
    LifecyclePolicy,
    TransitionTrigger,
    to_contract_state,
)
from mu_engine.storage.domain.memory import MemoryState

pytestmark = pytest.mark.unit


class _Item:
    """The narrow ``PinnableItem`` face — a stand-in for either ``MemoryItem`` definition."""

    def __init__(self, *, state: State = State.ACTIVE, pinned: bool = False) -> None:
        self.id = "mem_1"
        self.state = state
        self.pinned = pinned


@pytest.fixture
def policy() -> LifecyclePolicy:
    return LifecyclePolicy()


# ── the legal-edge table (memory-layer §1.1) ─────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("frm", "to"),
    [
        (State.ACTIVE, State.ARCHIVED),
        (State.ACTIVE, State.SUPERSEDED),
        (State.ACTIVE, State.QUARANTINED),
        (State.ACTIVE, State.EXPIRED),
        (State.ARCHIVED, State.DELETED),
        (State.SUPERSEDED, State.ACTIVE),
        (State.SUPERSEDED, State.DELETED),
        (State.EXPIRED, State.DELETED),
    ],
)
def test_legal_edges_pass_for_an_unpinned_item(
    policy: LifecyclePolicy, frm: State, to: State
) -> None:
    policy.assert_transition(_Item(state=frm), to)


@pytest.mark.parametrize(
    ("frm", "to"),
    [
        (State.ACTIVE, State.DELETED),  # GC never skips the ARCHIVED/dead intermediate
        (State.DELETED, State.ACTIVE),  # DELETED is terminal
        (State.QUARANTINED, State.DELETED),
    ],
)
def test_illegal_edges_raise_and_never_name_content(
    policy: LifecyclePolicy, frm: State, to: State
) -> None:
    with pytest.raises(IllegalTransitionError) as exc:
        policy.assert_transition(_Item(state=frm), to)
    assert "illegal edge" in str(exc.value)


def test_same_state_is_a_legal_no_op(policy: LifecyclePolicy) -> None:
    """An idempotent re-application must not be an error — at-least-once writes replay."""
    policy.assert_transition(_Item(state=State.ARCHIVED, pinned=True), State.ARCHIVED)


# ── the pin clause: retention exits are blocked UNCONDITIONALLY ──────────────────────────────
@pytest.mark.parametrize(
    ("frm", "to"),
    [
        (State.ACTIVE, State.ARCHIVED),
        (State.ACTIVE, State.EXPIRED),
        (State.ARCHIVED, State.DELETED),
        (State.SUPERSEDED, State.DELETED),
        (State.EXPIRED, State.DELETED),
    ],
)
@pytest.mark.parametrize("trigger", list(TransitionTrigger))
def test_pinned_item_blocks_every_retention_exit_on_every_trigger(
    policy: LifecyclePolicy, frm: State, to: State, trigger: TransitionTrigger
) -> None:
    with pytest.raises(PinnedTransitionBlocked):
        policy.assert_transition(_Item(state=frm, pinned=True), to, trigger=trigger)


# ── the pin clause: adjudicated exits are blocked only AUTOMATICALLY ─────────────────────────
@pytest.mark.parametrize("to", [State.SUPERSEDED, State.QUARANTINED])
def test_pinned_item_is_never_the_automatic_loser(policy: LifecyclePolicy, to: State) -> None:
    with pytest.raises(PinnedTransitionBlocked):
        policy.assert_transition(_Item(pinned=True), to, trigger=TransitionTrigger.AUTOMATIC)


@pytest.mark.parametrize("to", [State.SUPERSEDED, State.QUARANTINED])
def test_pinned_item_can_still_be_superseded_explicitly(policy: LifecyclePolicy, to: State) -> None:
    """CANONICAL §7.10: pin is retention, not immunity from the total order."""
    policy.assert_transition(_Item(pinned=True), to, trigger=TransitionTrigger.EXPLICIT)


def test_force_unpinned_opens_even_a_retention_exit(policy: LifecyclePolicy) -> None:
    policy.assert_transition(_Item(pinned=True), State.ARCHIVED, force_unpinned=True)


def test_force_unpinned_does_not_open_an_illegal_edge(policy: LifecyclePolicy) -> None:
    """The bypass is for the PIN clause only — it must never launder an illegal edge."""
    with pytest.raises(IllegalTransitionError):
        policy.assert_transition(_Item(pinned=True), State.DELETED, force_unpinned=True)


# ── the "skip, keep" helper (spec §9 line 381) ───────────────────────────────────────────────
def test_permits_is_false_for_a_pin_block_and_true_otherwise(policy: LifecyclePolicy) -> None:
    assert policy.permits(_Item(pinned=True), State.ARCHIVED) is False
    assert policy.permits(_Item(pinned=False), State.ARCHIVED) is True


def test_permits_still_propagates_an_illegal_edge(policy: LifecyclePolicy) -> None:
    """It may swallow a pin refusal (expected outcome) — never a caller bug."""
    with pytest.raises(IllegalTransitionError):
        policy.permits(_Item(state=State.DELETED), State.ACTIVE)


# ── the tier-axis clause (spec §6.2, the SHIPPED MTM->STM demotion) ──────────────────────────
def test_pinned_item_blocks_automatic_tier_demotion(policy: LifecyclePolicy) -> None:
    assert policy.permits_tier_demotion(_Item(pinned=True)) is False
    assert policy.permits_tier_demotion(_Item(pinned=False)) is True
    assert (
        policy.permits_tier_demotion(_Item(pinned=True), trigger=TransitionTrigger.EXPLICIT) is True
    )


# ── the two un-reconciled state enums normalize onto one axis ────────────────────────────────
@pytest.mark.parametrize("member", list(MemoryState))
def test_every_engine_state_normalizes_explicitly(member: MemoryState) -> None:
    assert to_contract_state(member) is State(member.value)


def test_policy_accepts_the_engine_enum_directly(policy: LifecyclePolicy) -> None:
    """The GC call site hands the ENGINE enum; the guard must reach the same verdict."""
    item = _Item(state=State.SUPERSEDED, pinned=True)
    with pytest.raises(PinnedTransitionBlocked):
        policy.assert_transition(item, MemoryState.DELETED)


# ── the retention-exit guard: pin-only, edge-table-independent (§6.3) ─────────────────────────
def test_a_retention_exit_is_refused_for_a_pinned_item_whatever_state_it_carries(
    policy: LifecyclePolicy,
) -> None:
    """CANONICAL §7.10's *"never garbage-collected regardless"* cannot depend on the FROM state,
    and at the GC site it must not: the swept row's carried ``state`` is stale (see
    ``assert_retention_exit``'s docstring)."""
    for carried in (State.ACTIVE, State.SUPERSEDED, State.EXPIRED, State.ARCHIVED):
        with pytest.raises(PinnedTransitionBlocked):
            policy.assert_retention_exit(_Item(state=carried, pinned=True), State.DELETED)


def test_a_retention_exit_does_not_raise_an_illegal_edge_on_a_stale_carried_state(
    policy: LifecyclePolicy,
) -> None:
    """REGRESSION. ``ACTIVE -> DELETED`` is not a legal edge, and the GC sweep really is handed
    dead rows still carrying ``state=ACTIVE`` (``facts_by_state`` rebuilds from ``memory_json``,
    which ``expire``/``invalidate`` never rewrite). Asking the EDGE-checking guard there raised
    ``IllegalTransitionError`` out of the sweep and aborted the whole GC pass.
    """
    assert policy.permits_retention_exit(_Item(state=State.ACTIVE), State.DELETED) is True
    # ...and the edge-checking guard is exactly what it must not be:
    with pytest.raises(IllegalTransitionError):
        policy.assert_transition(_Item(state=State.ACTIVE), State.DELETED)


def test_force_unpinned_opens_the_retention_exit(policy: LifecyclePolicy) -> None:
    policy.assert_retention_exit(_Item(pinned=True), State.EXPIRED, force_unpinned=True)


def test_permits_retention_exit_is_false_only_for_a_pin(policy: LifecyclePolicy) -> None:
    assert policy.permits_retention_exit(_Item(pinned=True), State.EXPIRED) is False
    assert policy.permits_retention_exit(_Item(pinned=False), State.EXPIRED) is True


@pytest.mark.parametrize("to", [State.SUPERSEDED, State.QUARANTINED, State.ACTIVE])
def test_a_non_retention_target_is_a_loud_caller_bug(policy: LifecyclePolicy, to: State) -> None:
    """This guard answers ONLY the retention-class question; handing it an adjudicated exit would
    silently skip the trigger rule that governs those, so it fails loud instead."""
    with pytest.raises(ValueError, match="not a retention-class exit"):
        policy.assert_retention_exit(_Item(pinned=True), to)
