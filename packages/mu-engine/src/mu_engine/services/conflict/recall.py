"""``PendingConflictRecallPolicy`` — §6, how a still-pending conflict affects RECALL.

Authority: ``conflict-resolution-async-design.md`` §6 (lines 226-253).

The one place MANUAL mode changes READ behaviour, because in MANUAL mode no supersession has
happened: both contending items are ``state='active'`` and both pass the §7.5 hot-read floor. The
owner's question — *"surface both? prefer active?"* — is answered per-policy by
``PendingRecallMode``:

* ``SURFACE_BOTH_MARKED`` (DEFAULT) — return both, annotate each. Honest; never a silent drop,
  consistent with the system's named-marker doctrine.
* ``PREFER_PROVISIONAL`` — return only the strategy's provisional winner, still annotated. A
  READ-TIME RANKING, not a write: the loser is not invalidated, so a later human decision the
  other way needs no ``REINSTATE`` (spec line 249).
* ``SUPPRESS_BOTH`` — return neither, and emit ``DegradedModeEntered(reason=
  CONFLICT_PENDING_SUPPRESSED)`` so the suppression is observable, never silent (spec line 250).

**Read-only by construction.** This class holds no repository, no writer and no lease — the same
structural purity ``MemoryHealthService`` has. §6's whole point is that a pending conflict changes
what recall SHOWS, never what the store CONTAINS; a version of this that could invalidate the
provisional loser would have quietly turned a read into the supersession the user was asked to
decide.

**The provisional winner is the SAME deterministic §4.2 order**, via
``strategies.picker_for(policy.strategy)`` — not a second ranking. Two devices under
``PREFER_PROVISIONAL`` therefore show the same item, which is the whole reason it is safe to
prefer one before the human has chosen.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.events import DegradedModeEntered, DegradeReason
from mu_contracts.domain.model.conflict import ConflictEdges, PendingRecallMode
from mu_contracts.domain.model.conflict_recall import RecallConflictAnnotation
from mu_engine.lifecycle.conflict import ConflictResolutionPolicy
from mu_engine.lifecycle.conflict_events import ConflictEventSink, publish_content_free
from mu_engine.services.conflict.strategies import picker_for
from mu_engine.storage.domain.memory import MemoryItem

__all__ = ["PendingConflictRecallPolicy", "RecallConflictOutcome"]

_DEGRADE_COMPONENT = "recall"
_DEGRADE_MODE = "conflict_suppressed"


class RecallConflictOutcome(BaseModel):
    """What recall should return once §6 has been applied.

    ``arbitrary_types_allowed`` because ``surfaced`` carries the live ``MemoryItem``s the caller
    already had — this is an internal engine result handed straight back to the recall path, not
    a wire DTO. It is deliberately NOT a ``ContentFreeModel``: it contains items, i.e. content, by
    design, and it is never published (the ANNOTATIONS are the content-free half, and they are
    what a log or a metric would carry).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    surfaced: tuple[MemoryItem, ...] = ()
    annotations: tuple[RecallConflictAnnotation, ...] = ()
    #: Ids withheld under ``SUPPRESS_BOTH`` / ``PREFER_PROVISIONAL``. Named, never silent — a
    #: caller can report "2 results withheld pending your review" instead of showing a short list.
    suppressed_ids: tuple[str, ...] = ()

    def annotation_for(self, memory_id: str) -> RecallConflictAnnotation | None:
        for annotation in self.annotations:
            if annotation.memory_id == memory_id:
                return annotation
        return None


class PendingConflictRecallPolicy:
    """Apply the §6 pending-conflict read semantics to one page of recall hits."""

    def __init__(self, *, bus: ConflictEventSink | None = None) -> None:
        self._bus = bus

    async def apply(
        self,
        items: Sequence[MemoryItem],
        edges: ConflictEdges,
        *,
        policy: ConflictResolutionPolicy,
    ) -> RecallConflictOutcome:
        """Annotate (and, per mode, filter) ``items`` given the namespace's conflict adjacency.

        ``edges`` is the bounded, content-free ``ConflictEdges`` projection the health lane
        already builds from parked records — reused rather than re-queried, so recall does no
        extra per-item round-trip and the two surfaces can never disagree about what is
        conflicted.

        Items in NO unresolved conflict pass through untouched and un-annotated: AUTOMATIC mode
        needs none of this (spec line 251), and an item nobody is arguing about must not pay for
        the feature.
        """
        outcome = self._decide(items, edges, policy=policy)
        if outcome.suppressed_ids and policy.manual_recall_mode is (
            PendingRecallMode.SUPPRESS_BOTH
        ):
            await self._emit_suppressed(len(outcome.suppressed_ids))
        return outcome

    def _decide(
        self,
        items: Sequence[MemoryItem],
        edges: ConflictEdges,
        *,
        policy: ConflictResolutionPolicy,
    ) -> RecallConflictOutcome:
        """The pure core: no I/O, no clock, deterministic. Split out so the decision can be
        tested — and mutation-proven — without a bus."""
        conflicted = [i for i in items if edges.unresolved_for(i.id)]
        if not conflicted:
            return RecallConflictOutcome(surfaced=tuple(items))

        mode = policy.manual_recall_mode
        conflicted_ids = {i.id for i in conflicted}
        provisional_winners = self._provisional_winners(conflicted, edges, policy=policy)

        surfaced: list[MemoryItem] = []
        suppressed: list[str] = []
        annotations: list[RecallConflictAnnotation] = []
        for item in items:
            if item.id not in conflicted_ids:
                surfaced.append(item)
                continue
            is_winner = item.id in provisional_winners
            keep = mode is PendingRecallMode.SURFACE_BOTH_MARKED or (
                mode is PendingRecallMode.PREFER_PROVISIONAL and is_winner
            )
            annotations.append(
                RecallConflictAnnotation(
                    memory_id=item.id,
                    conflict_pending=True,
                    conflict_id=_conflict_id_for(edges, item.id),
                    conflict_peer_ids=edges.peers_for(item.id),
                    # Only meaningful under PREFER_PROVISIONAL (spec line 245); under the honest
                    # default nothing has been preferred, and stamping a winner there would be
                    # the silent pick §6 exists to prevent.
                    is_provisional_winner=(
                        is_winner and mode is PendingRecallMode.PREFER_PROVISIONAL
                    ),
                )
            )
            if keep:
                surfaced.append(item)
            else:
                suppressed.append(item.id)
        return RecallConflictOutcome(
            surfaced=tuple(surfaced),
            annotations=tuple(annotations),
            suppressed_ids=tuple(sorted(suppressed)),
        )

    @staticmethod
    def _provisional_winners(
        conflicted: Sequence[MemoryItem],
        edges: ConflictEdges,
        *,
        policy: ConflictResolutionPolicy,
    ) -> frozenset[str]:
        """The strategy's advisory pick per conflict group, by the SAME §4.2 deterministic order.

        Grouped by ``conflict_id`` so two independent conflicts on one page each get their own
        winner. A group whose peers are not all on this page still picks among what IS here: a
        provisional preference is a ranking over the page, and withholding an answer because an
        off-page peer exists would suppress more than the policy asked for. Only computed under
        ``PREFER_PROVISIONAL`` — under the other two modes there is no pick to make, and running
        a picker anyway would be work whose result is discarded.
        """
        if policy.manual_recall_mode is not PendingRecallMode.PREFER_PROVISIONAL:
            return frozenset()
        groups: dict[str, list[MemoryItem]] = {}
        for item in conflicted:
            conflict_id = _conflict_id_for(edges, item.id)
            if conflict_id is not None:
                groups.setdefault(conflict_id, []).append(item)
        picker = picker_for(policy.strategy)
        winners: set[str] = set()
        for members in groups.values():
            if len(members) == 1:
                # The only member of this conflict that is on the page — nothing to prefer it
                # OVER, so it surfaces (marked) rather than being suppressed as a "loser".
                winners.add(members[0].id)
                continue
            winners.add(
                picker.pick(members, policy=policy, proposed_winner_id=None, confidence=0.0).id
            )
        return frozenset(winners)

    async def _emit_suppressed(self, count: int) -> None:
        """Spec line 250 — ``SUPPRESS_BOTH`` is offered for high-stakes namespaces where a wrong
        fact is worse than a missing one, and it MUST be observable: a silent suppression is
        indistinguishable from "you have no memory about that"."""
        await publish_content_free(
            self._bus,
            DegradedModeEntered(
                component=_DEGRADE_COMPONENT,
                mode=_DEGRADE_MODE,
                reason=DegradeReason.CONFLICT_PENDING_SUPPRESSED,
                detail=f"withheld={count}",
            ),
        )


def _conflict_id_for(edges: ConflictEdges, memory_id: str) -> str | None:
    row = edges.rows_by_memory.get(memory_id)
    return None if row is None else row.conflict_id
