"""§4.2 — the AUTOMATIC winner-pickers. All deterministic, all over the ONE §7.17 total order.

Authority: ``conflict-resolution-async-design.md`` §4.2 (lines 162-164): *"``RECENCY`` is exactly
the §7.17 deterministic total order ... ``CONFIDENCE`` and ``PROVENANCE`` reorder the primary key
but fall back to the same total order for ties, so every replica computes the identical winner
with no coordination."* · CANONICAL §7.17 item 4a · memory-health §6.4 / CANONICAL §7.17 4a(b)
(the pin clause).

**Three properties this module exists to guarantee, in order of how badly each fails silently:**

1. **A PINNED memory is never the auto-supersede loser.** Not re-implemented here — every
   candidate loser is put to ``LifecyclePolicy.permits(loser, SUPERSEDED,
   trigger=AUTOMATIC)``, the ONE owner of that rule (``lifecycle/policy.py``), so an explicit
   user-driven supersede of a pinned item stays legal exactly as CANONICAL §7.10 requires while
   no sweep can take one. A pin-blocked resolution comes back ``applicable=False`` and the
   caller PARKS it — the pinned item stays ACTIVE and the conflict is surfaced, never dropped.
2. **Determinism.** Every picker is a pure function of the candidates plus the (replicated)
   policy: no clock, no I/O, no randomness, no iteration-order dependence, no LLM. Feeding the
   candidates in a different order returns the same winner, which is what makes AUTOMATIC
   resolution safe to run independently on each device (§4.2 line 164).
3. **One ordering.** Every strategy's tie-break — and RECENCY in its entirety — is
   ``services.conflict.order.total_order_key_items``, which shares its seven-term chain with
   ``total_order_key`` (the ``PrivateDelta`` face). There is deliberately no second comparison
   anywhere in this file.

**CONFIDENCE is reinterpreted, and the reinterpretation is REPORTED.** Spec line 131 defines it
as *"highest adjudicator C wins; tie → recency"*. There is no per-item confidence in this system
and never has been: ``C`` is a property of the CONFLICT (``ConflictRecord.detected_confidence`` /
``AdjudicationVerdict.confidence``) — one number for the pair — so "highest C wins" has no
per-candidate quantity to compare and cannot be implemented as written. The faithful reading of
the intent is implemented instead: the adjudicator's own ``proposed_winner_id`` stands when the
pair's ``C`` clears ``policy.auto_min_confidence``, and otherwise the pick falls back to the same
total order. That is still "confidence decides, tie → recency", still deterministic given
``(items, verdict, policy)``, and still LLM-free at pick time — the model call happened upstream
on the detect side; nothing here calls one.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cmp_to_key
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.conflict import (
    AutoResolveStrategy,
    ConflictRecord,
    ConflictResolutionKind,
)
from mu_engine.lifecycle.conflict import ConflictResolutionPolicy
from mu_engine.lifecycle.policy import LifecyclePolicy, TransitionTrigger
from mu_engine.services.conflict.order import total_order_key_items
from mu_engine.storage.domain.memory import MemoryItem, MemorySource, MemoryState

__all__ = [
    "SOURCE_TRUST_RANK",
    "AutoResolution",
    "AutoWinnerPicker",
    "ConfidencePicker",
    "ProvenancePicker",
    "RecencyPicker",
    "picker_for",
    "recommended_resolution_kind",
    "resolve_automatically",
]


#: PROVENANCE's primary key: how far a source is trusted to assert a fact about the user, higher
#: wins. The ordering is the S_trust/S_prov intent of spec line 132 made concrete and content-free:
#: what the USER said outranks what a TOOL observed, which outranks what an AGENT concluded, which
#: outranks an EXTERNAL import, which outranks something the system INFERRED. It is a fixed table
#: rather than a tunable because a per-deployment trust order would make the winner
#: deployment-dependent and break the §4.2 "every replica computes the identical winner"
#: guarantee the moment two devices disagreed on the table.
SOURCE_TRUST_RANK: dict[MemorySource, int] = {
    MemorySource.USER: 4,
    MemorySource.TOOL: 3,
    MemorySource.AGENT: 2,
    MemorySource.EXTERNAL: 1,
    MemorySource.INFERRED: 0,
}


class AutoResolution(BaseModel):
    """One strategy's decision over a conflict's candidate set.

    ``applicable=False`` is NOT an error and NOT a "no conflict": it means the automatic path
    declines to write and the caller must PARK the conflict for a human (§2 line 58 —
    *"uncertain-automatic degrades to manual, it does not fabricate a winner"*). The winner is
    still reported, because a parked conflict with an honest advisory pick is strictly more
    useful to the human than one with none.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: AutoResolveStrategy
    winner_id: str = Field(min_length=1)
    loser_ids: tuple[str, ...] = ()
    applicable: bool = True
    #: Content-free named reason (an enum-ish token, never memory text) — the audit trail for
    #: WHY the automatic path declined.
    reason: str = Field(default="auto_resolved", min_length=1)
    #: True iff a candidate loser is ``pinned`` and ``LifecyclePolicy`` refused the automatic
    #: exit. Projected onto ``ConflictRecord.pin_blocked`` by the caller.
    pin_blocked: bool = False


@runtime_checkable
class AutoWinnerPicker(Protocol):
    """The Strategy seam (DEV-STANDARDS rule 6). One method, pure, total."""

    def pick(
        self,
        candidates: Sequence[MemoryItem],
        *,
        policy: ConflictResolutionPolicy,
        proposed_winner_id: str | None,
        confidence: float,
    ) -> MemoryItem: ...


def _total_order_winner(candidates: Sequence[MemoryItem]) -> MemoryItem:
    """``max`` under the §7.17 item-shaped total order. The single comparison in this module."""
    return max(candidates, key=cmp_to_key(total_order_key_items))


class RecencyPicker:
    """``RECENCY`` — the §7.17 total order itself, with no primary key prepended (spec line 130).

    Note what this deliberately does NOT do: it does not compare raw ``valid_at`` first. Term 4
    of the order (the replica's monotone assertion counter, ``created_at`` for a live item) is
    consulted BEFORE term 5 (``valid_at``), so an inferred or back-dated world-time cannot
    outrank what was actually asserted later — which is both §7.17's own term order and the
    shipped BUG1 rule (``distill.asserted_later``). See ``order._item_terms`` for the full
    reasoning and the reported spec conflict with line 130's wording.
    """

    def pick(
        self,
        candidates: Sequence[MemoryItem],
        *,
        policy: ConflictResolutionPolicy,
        proposed_winner_id: str | None,
        confidence: float,
    ) -> MemoryItem:
        return _total_order_winner(candidates)


class ConfidencePicker:
    """``CONFIDENCE`` — honor the adjudicator's pick when the pair's C clears the floor.

    See the module docstring for why this is a reported reinterpretation of spec line 131 rather
    than a literal implementation of it. Falls back to the same total order when C is below the
    floor, or when the adjudicator proposed nothing, or when what it proposed is not one of the
    candidates (a stale or malformed pick must never silently select nobody).
    """

    def pick(
        self,
        candidates: Sequence[MemoryItem],
        *,
        policy: ConflictResolutionPolicy,
        proposed_winner_id: str | None,
        confidence: float,
    ) -> MemoryItem:
        if proposed_winner_id is not None and confidence >= policy.auto_min_confidence:
            for candidate in candidates:
                if candidate.id == proposed_winner_id:
                    return candidate
        return _total_order_winner(candidates)


class ProvenancePicker:
    """``PROVENANCE`` — most-trusted source wins; ties fall to the same total order (line 132).

    :data:`SOURCE_TRUST_RANK` is the primary key. When every candidate shares a source — the
    common case, since most facts in a namespace come from the same capture path — every
    candidate ties on the primary key and this degenerates exactly to RECENCY, which is the
    intended behaviour, not a gap.
    """

    def pick(
        self,
        candidates: Sequence[MemoryItem],
        *,
        policy: ConflictResolutionPolicy,
        proposed_winner_id: str | None,
        confidence: float,
    ) -> MemoryItem:
        best_rank = max(SOURCE_TRUST_RANK[c.source] for c in candidates)
        most_trusted = [c for c in candidates if SOURCE_TRUST_RANK[c.source] == best_rank]
        return _total_order_winner(most_trusted)


#: Strategy registry (DEV-STANDARDS rule 6: dispatch table, never an if/elif ladder). EXHAUSTIVE
#: over ``AutoResolveStrategy`` — :func:`picker_for` indexes it directly so a member added to the
#: enum without a picker raises ``KeyError`` at the point of use, loudly, instead of silently
#: falling back to recency and quietly changing which memory survives.
_PICKERS: dict[AutoResolveStrategy, AutoWinnerPicker] = {
    AutoResolveStrategy.RECENCY: RecencyPicker(),
    AutoResolveStrategy.CONFIDENCE: ConfidencePicker(),
    AutoResolveStrategy.PROVENANCE: ProvenancePicker(),
}


def picker_for(strategy: AutoResolveStrategy) -> AutoWinnerPicker:
    """The picker for ``strategy``. Raises ``KeyError`` for an unregistered member — see
    :data:`_PICKERS`."""
    return _PICKERS[strategy]


def resolve_automatically(
    candidates: Sequence[MemoryItem],
    *,
    policy: ConflictResolutionPolicy,
    proposed_winner_id: str | None = None,
    confidence: float = 1.0,
    lifecycle: LifecyclePolicy | None = None,
) -> AutoResolution:
    """Pick a winner deterministically and check every loser against the central pin guard.

    Raises ``ValueError`` for fewer than two candidates: a "conflict" with one member has no
    second side, and returning a trivially-successful resolution for it would let a caller
    supersede nothing while believing it had resolved something.

    The pin check is the reason this is a function and not just ``picker.pick(...)``: picking a
    winner and being ALLOWED to write the losers are two different questions, and conflating them
    is how a pinned memory gets superseded. ``trigger=AUTOMATIC`` is passed explicitly — no
    automatic sweep may ever pass ``EXPLICIT`` (``lifecycle/policy.py``'s ``TransitionTrigger``
    contract), and it is what makes an explicit user-driven supersede of a pinned item still
    legal on the manual path.
    """
    if len(candidates) < 2:
        raise ValueError("an auto-resolution needs at least two candidates")
    guard = lifecycle or LifecyclePolicy()
    picker = picker_for(policy.strategy)
    winner = picker.pick(
        candidates,
        policy=policy,
        proposed_winner_id=proposed_winner_id,
        confidence=confidence,
    )
    losers = tuple(sorted(c.id for c in candidates if c.id != winner.id))
    blocked = [
        c
        for c in candidates
        if c.id != winner.id
        and not guard.permits(c, MemoryState.SUPERSEDED, trigger=TransitionTrigger.AUTOMATIC)
    ]
    if blocked:
        return AutoResolution(
            strategy=policy.strategy,
            winner_id=winner.id,
            loser_ids=losers,
            applicable=False,
            reason="pin_blocked_pinned_item_is_never_the_auto_loser",
            pin_blocked=True,
        )
    return AutoResolution(
        strategy=policy.strategy, winner_id=winner.id, loser_ids=losers, applicable=True
    )


def recommended_resolution_kind(
    record: ConflictRecord, policy: ConflictResolutionPolicy
) -> ConflictResolutionKind:
    """Which resolution the §4 uncertainty bands RECOMMEND for a parked record.

    The third band of ``ConflictResolutionPolicy``'s truth table (``C < quarantine_below``) is
    quarantine-grade (spec line 139, paper H4); everything else parked is a supersede decision
    awaiting a human. This is DERIVED from ``detected_confidence`` + the policy rather than
    stored as a fourth field, so the record cannot carry a recommendation that contradicts the
    policy it was snapshotted under.

    A RECOMMENDATION only: nothing in this lane writes ``state=QUARANTINED``. The resolve stage
    applies it, off the write path, and the human may override it with any other kind.
    """
    if record.detected_confidence < policy.quarantine_below:
        return ConflictResolutionKind.QUARANTINE
    return ConflictResolutionKind.SUPERSEDE
