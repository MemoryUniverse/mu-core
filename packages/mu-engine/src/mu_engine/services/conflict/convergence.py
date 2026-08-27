"""§7 — a manual decision is a sticky, first-class sync delta that no replica may flip.

Authority: ``conflict-resolution-async-design.md`` §7 (lines 257-264) · CANONICAL §7.5 X5 /
§7.17 item 4a(b) (``resolution_origin == "manual"`` is term 2 of the total order).

**The failure this module prevents.** §7.5's per-replica re-derivation already makes AUTOMATIC
supersession converge with no coordination. A MANUAL decision needs one thing more: it must BEAT
automatic re-derivation everywhere, *"otherwise a phone that auto-picks the recency winner would
silently override the human's laptop decision"* (line 259). That is a data-loss bug with no error
message anywhere in it.

**Three pieces, and what each is NOT:**

1. :func:`manual_supersede_delta` / :func:`manual_reinstate_delta` — the WRITERS §7 line 261
   requires and the repo did not have. ``PrivateDelta`` already carried every field
   (``resolution_origin``, ``resolved_by``, ``winner_id``/``loser_id``, ``lamport``, ``pinned``);
   what was missing was anything that ever SET ``resolution_origin=MANUAL``, which made term 2 of
   the total order permanently dead and stickiness unreachable. These are pure builders — no
   clock, no I/O, every varying value passed in — so the delta a device appends is reproducible.
2. :func:`converge_pair` — the RE-DERIVATION. It does not implement a second ordering: it is
   ``max`` under ``services.conflict.order.total_order_key``, whose term 2 IS the stickiness.
   Nothing here special-cases "manual"; the order already does, which is the point.
3. The REOPEN rule (line 262) — *"a later purely-automatic contradicting delta does **not** flip
   it (it instead ``REOPENED``s the conflict for human review, never a silent auto-override of a
   human)"*. :func:`converge_pair` reports ``reopen=True`` exactly when an automatic delta WOULD
   have won had the manual term not been there. That counterfactual is computed by asking the
   SAME comparator with term 2 neutralised on both sides — again, not a second ordering.

**REINSTATE covers the flip** (line 263): a device that had locally auto-superseded before the
human's delta arrived gets back the ids that the converged decision says are winners, so it can
emit ``REINSTATE`` and re-apply. That is what makes the ``state='active'`` sets byte-identical on
every device and the hub.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from functools import cmp_to_key

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.conflict import ResolutionOrigin
from mu_contracts.domain.model.device_sync import PrivateDelta, SyncOp
from mu_engine.services.conflict.order import total_order_key

__all__ = [
    "ConvergenceOutcome",
    "converge_pair",
    "manual_reinstate_delta",
    "manual_supersede_delta",
]


class ConvergenceOutcome(BaseModel):
    """The re-derived, coordination-free decision for ONE conflicting pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    winner_id: str = Field(min_length=1)
    loser_ids: tuple[str, ...] = Field(min_length=1)
    #: ``MANUAL`` iff the winning assertion was a human's. This is what a replica stamps onto its
    #: own ``ConflictRecord`` so every device's inbox agrees on how the conflict was settled
    #: (§7 line 264).
    origin: ResolutionOrigin
    #: Ids this replica has locally superseded that the converged decision says are WINNERS.
    #: Each needs a ``REINSTATE`` before the winner is re-applied (§7 line 263).
    reinstate_ids: tuple[str, ...] = ()
    #: An automatic delta contradicts a STICKY manual decision. The conflict must be ``REOPEN``ed
    #: for human review and a new ``ConflictResolutionPending`` raised — never a silent flip.
    reopen: bool = False


def manual_supersede_delta(
    *,
    seq: int,
    origin_device_id: str,
    winner_id: str,
    loser_id: str,
    content_hash: str,
    tier: str,
    valid_at: datetime,
    occurred_at: datetime,
    provenance_id: str,
    lamport: int,
    resolved_by: str,
    pinned: bool = False,
    valid_at_inferred: bool = False,
) -> PrivateDelta:
    """Build the ``SyncOp.SUPERSEDE`` delta a MANUAL resolution emits (§7 line 261).

    ``memory_id`` is the LOSER. That is the item whose state actually changes
    (``active -> superseded``), so a replica applying the delta acts on the memory the delta
    names; using the winner would make every apply an indirection through ``loser_id`` and would
    break dedupe, which keys on ``(memory_id, content_hash)``.

    ``resolution_origin=MANUAL`` + ``resolved_by`` are the whole point: they are what make term 2
    of the §7.17 order fire on every replica that receives this delta.
    """
    return PrivateDelta(
        seq=seq,
        origin_device_id=origin_device_id,
        op=SyncOp.SUPERSEDE,
        memory_id=loser_id,
        content_hash=content_hash,
        tier=tier,
        valid_at=valid_at,
        valid_at_inferred=valid_at_inferred,
        lamport=lamport,
        occurred_at=occurred_at,
        provenance_id=provenance_id,
        winner_id=winner_id,
        loser_id=loser_id,
        pinned=pinned,
        resolution_origin=ResolutionOrigin.MANUAL,
        resolved_by=resolved_by,
    )


def manual_reinstate_delta(
    *,
    seq: int,
    origin_device_id: str,
    winner_id: str,
    loser_id: str,
    content_hash: str,
    tier: str,
    valid_at: datetime,
    occurred_at: datetime,
    provenance_id: str,
    lamport: int,
    resolved_by: str,
    pinned: bool = False,
    valid_at_inferred: bool = False,
) -> PrivateDelta:
    """The ``SyncOp.REINSTATE`` counterpart (§7 line 263 / CANONICAL §7.5).

    Emitted when a replica had locally auto-superseded the item the human's decision makes the
    WINNER. ``memory_id`` is again the item whose state changes — here the one coming BACK to
    ``active`` — and the ``(winner_id, loser_id)`` pair is carried in full so any replica can
    re-derive the decision rather than trusting a one-way stamp (CANONICAL §7.5 X5's
    lost-update rule, which ``PrivateDelta``'s own validator enforces).
    """
    return PrivateDelta(
        seq=seq,
        origin_device_id=origin_device_id,
        op=SyncOp.REINSTATE,
        memory_id=winner_id,
        content_hash=content_hash,
        tier=tier,
        valid_at=valid_at,
        valid_at_inferred=valid_at_inferred,
        lamport=lamport,
        occurred_at=occurred_at,
        provenance_id=provenance_id,
        winner_id=winner_id,
        loser_id=loser_id,
        pinned=pinned,
        resolution_origin=ResolutionOrigin.MANUAL,
        resolved_by=resolved_by,
    )


def converge_pair(
    deltas: Sequence[PrivateDelta],
    *,
    locally_superseded: frozenset[str] = frozenset(),
) -> ConvergenceOutcome:
    """Re-derive the winner for ONE conflicting pair over the merged delta set.

    Every delta must be a ``SUPERSEDE``/``REINSTATE`` about the SAME unordered pair — the
    precondition is CHECKED, not assumed, because silently converging a mixed bag would produce a
    confident winner for a conflict nobody asserted. Raises ``ValueError`` otherwise.

    Deterministic and coordination-free: ``max`` under ``total_order_key``, so every replica
    computes the identical outcome from the same delta set with no clock and no network.

    The ``reopen`` counterfactual asks the same comparator with term 2 (manual) neutralised on
    both sides. Neutralising is done by ``model_copy`` rather than by a bespoke comparison, so
    there is exactly one place in the system that knows what "wins" means.
    """
    if not deltas:
        raise ValueError("cannot converge an empty delta set")
    pairs = {frozenset((d.winner_id, d.loser_id)) for d in deltas}
    if len(pairs) != 1 or None in next(iter(pairs)):
        raise ValueError("converge_pair requires deltas about exactly one (winner, loser) pair")

    winner_delta = max(deltas, key=cmp_to_key(total_order_key))
    winner_id = _require(winner_delta.winner_id)
    loser_id = _require(winner_delta.loser_id)
    origin = winner_delta.resolution_origin or ResolutionOrigin.AUTO

    reopen = False
    if origin is ResolutionOrigin.MANUAL:
        neutral_winner = winner_delta.model_copy(update={"resolution_origin": None})
        reopen = any(
            d is not winner_delta
            and d.resolution_origin is not ResolutionOrigin.MANUAL
            and d.winner_id != winner_id
            and total_order_key(d, neutral_winner) > 0
            for d in deltas
        )

    # A REINSTATE is needed for anything this replica already superseded that the converged
    # decision says survives. The loser is expected to be superseded — that is the decision.
    reinstate = tuple(sorted(locally_superseded - {loser_id}))
    return ConvergenceOutcome(
        winner_id=winner_id,
        loser_ids=(loser_id,),
        origin=origin,
        reinstate_ids=reinstate,
        reopen=reopen,
    )


def _require(value: str | None) -> str:
    """Narrow the ``str | None`` that ``PrivateDelta``'s own validator has already guaranteed to
    be present for ``SUPERSEDE``/``REINSTATE``. A loud raise, not an ``assert`` (``python -O``
    strips those)."""
    if value is None:
        raise ValueError("a SUPERSEDE/REINSTATE delta must carry both winner_id and loser_id")
    return value
