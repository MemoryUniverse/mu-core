"""``RecallConflictAnnotation`` — how an UNRESOLVED conflict is marked at read time.

Authority: ``conflict-resolution-async-design.md`` §6 (lines 226-253) and proposed contract
change 7 (line 316).

**The doctrine, in one line:** *"at recall, never **pretend** the conflict is decided"* (spec
line 248) — the read-time analogue of "never delete".

This only ever applies while NO supersession has happened, i.e. MANUAL policy or an AUTOMATIC one
degraded to manual: both contending items are still ``state='active'`` and both pass the §7.5
hot-read floor. Under an APPLIED resolution the loser is already ``superseded`` and the floor
excludes it, so every annotation here is dark (``conflict_pending=False``) — spec line 251.

**Where these fields are SUPPOSED to live, and why they are here instead.** Spec lines 240-246
put them directly on ``RecallItemView`` (``contracts/recall.py``). That file is another lane's,
so the annotation is modelled as its own content-free DTO that a recall surface can carry
alongside each hit. The engine-side decision — which items surface, which are suppressed, which
is the provisional winner — is complete either way; only the field placement differs. REPORTED as
a cross-lane change: ``RecallItemView`` needs ``conflict_pending``, ``conflict_id``,
``conflict_peer_ids`` and ``is_provisional_winner``, and until it does a recall response cannot
transport these to a client.
"""

from __future__ import annotations

from pydantic import Field

from mu_contracts.domain.model.conflict import ContentFreeModel

__all__ = ["RecallConflictAnnotation"]


class RecallConflictAnnotation(ContentFreeModel):
    """The conflict marker attached to ONE recall hit (spec §6 lines 240-246).

    Content-free by inheritance: ids and booleans. The conflicting TEXT is already in the recall
    hit the annotation rides on — this DTO adds only the LINK, so the annotation itself can be
    logged or metered without carrying anything new.
    """

    memory_id: str = Field(min_length=1)
    #: This hit is a live member of an unresolved conflict.
    conflict_pending: bool = False
    #: The record, so a caller/UI can deep-link into the §5 inbox.
    conflict_id: str | None = None
    #: The other contending member ids, sorted for determinism, so a consumer can reconcile.
    conflict_peer_ids: tuple[str, ...] = ()
    #: Under ``PREFER_PROVISIONAL``, the strategy's advisory pick. A READ-TIME RANKING, never a
    #: write: the loser is not invalidated, so if the human later picks the other item no
    #: ``REINSTATE`` is needed because nothing was superseded (spec line 249).
    is_provisional_winner: bool = False
