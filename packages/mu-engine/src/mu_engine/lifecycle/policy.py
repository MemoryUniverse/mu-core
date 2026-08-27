"""``LifecyclePolicy`` — the SINGLE owner of legal memory-FSM edges, and the central pin guard.

Authority: ``memory-layer-design.md`` §1.1 (lines 227-258 — the legal-edge table, *"a single
``LifecyclePolicy`` validates every transition and is the only place that knows the legal
edges; illegal transitions raise ``IllegalTransitionError`` (fail loud, never silently
no-op)"*) · ``memory-health-pinning-spec.md`` §6.1 (lines 277-287) · ``CANONICAL-CONTRACTS.md``
§7.10 / §7.17 item 4a(b) / §7.26.

**Why this file exists at all.** ``memory-health-pinning-spec`` §6.1 presents the pin clause as
three lines added onto *"existing edge-legality checks"* (spec line 283). There were none:
``LifecyclePolicy`` was specified in three design docs and never built, and every state flip in
the engine was a direct ``model_copy(update={"state": ...})``. Enforcing pin only inside
``PinService``/``RetentionService`` is exactly the defect §6 warns about — any other caller could
still evict a pinned memory. So the guard is built here, and the exit sites ASK it.

**The pin clause is NOT "a pinned item cannot leave" (that would contradict CANONICAL).**
Spec §6.1 line 281 sets ``_EXIT_STATES = {ARCHIVED, SUPERSEDED, QUARANTINED, DELETED}`` and blocks
all four unconditionally. CANONICAL §7.10 says the opposite for two of them: *"Pin = retention,
not access (§7.26) — a pinned item can still be superseded, but it is never the auto-loser
(§7.17) and never GC'd."* CANONICAL wins (spec §0 line 11), and §6.4 of the spec itself describes
only the AUTOMATIC path. The built rule therefore splits the exits:

* :data:`RETENTION_EXITS` — ``ARCHIVED`` / ``EXPIRED`` / ``DELETED``. Blocked for a pinned item
  **unconditionally**, on any trigger. This is CANONICAL §7.10's *"never garbage-collected
  regardless"* and spec §6.2's *"pin overrides the forgetting curve entirely"*. Only an explicit
  ``force_unpinned`` (i.e. the caller unpinned first) opens them.
* :data:`ADJUDICATED_EXITS` — ``SUPERSEDED`` / ``QUARANTINED``. Blocked only when the trigger is
  :attr:`TransitionTrigger.AUTOMATIC`. That is precisely *"a pinned item is never the AUTO-
  supersede/quarantine loser"* (CANONICAL §7.17 item 4a(b), spec §6.4 line 308); an EXPLICIT,
  user-driven supersede stays legal, as CANONICAL §7.10 requires.

``EXPIRED`` is in ``RETENTION_EXITS`` although spec line 281 omits it. Both ``State`` enums carry
six members including ``EXPIRED`` (ADR 0035/0049), ``RetentionService`` really does flip
``ACTIVE -> EXPIRED`` and then GCs it, and that flip is the same retention-class exit as
``ARCHIVED``. Omitting it would leave a pinned EPHEMERAL fact unguarded on the one path that
actually deletes things. Reported as a spec gap rather than silently assumed.

**Two state enums, one axis.** ``mu_contracts.domain.model.memory.State`` (published) and
``mu_engine.storage.domain.memory.MemoryState`` (shipped) are independent, un-reconciled
definitions of the same axis (``storage/domain/memory.py``'s own RE-HOME NOTE; ADR 0049). This
module normalizes through :data:`ENGINE_TO_CONTRACT_STATE` — an EXPLICIT, EXHAUSTIVE dict, never
``State(engine_state.value)`` string parity, for the reason ``RetentionService._STATE_MAP``'s
docstring already gives: a future member landing on one enum and not the other must fail LOUD
(``KeyError``) rather than silently coerce.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Protocol, runtime_checkable

from mu_contracts.domain.errors import IllegalTransitionError, PinnedTransitionBlocked
from mu_contracts.domain.model.memory import State
from mu_engine.storage.domain.memory import MemoryState

__all__ = [
    "ADJUDICATED_EXITS",
    "ENGINE_TO_CONTRACT_STATE",
    "EXIT_STATES",
    "RETENTION_EXITS",
    "LifecyclePolicy",
    "PinnableItem",
    "TransitionTrigger",
]


#: The engine -> published state map. EXHAUSTIVE by construction; a missing member raises
#: ``KeyError`` at the point of use rather than guessing (ADR 0049's lesson).
ENGINE_TO_CONTRACT_STATE: dict[MemoryState, State] = {
    MemoryState.ACTIVE: State.ACTIVE,
    MemoryState.ARCHIVED: State.ARCHIVED,
    MemoryState.SUPERSEDED: State.SUPERSEDED,
    MemoryState.QUARANTINED: State.QUARANTINED,
    MemoryState.DELETED: State.DELETED,
    MemoryState.EXPIRED: State.EXPIRED,
}

#: Retention-class exits. A pinned item may NEVER take one automatically OR explicitly —
#: CANONICAL §7.10 *"never garbage-collected regardless"*.
RETENTION_EXITS: frozenset[State] = frozenset({State.ARCHIVED, State.EXPIRED, State.DELETED})

#: Adjudicated exits. A pinned item may never be driven into one by an AUTOMATIC decision
#: (CANONICAL §7.17 item 4a(b)); an EXPLICIT, user-driven one stays legal (CANONICAL §7.10).
ADJUDICATED_EXITS: frozenset[State] = frozenset({State.SUPERSEDED, State.QUARANTINED})

#: Every exit edge (spec §6.1 line 281's ``_EXIT_STATES``, plus ``EXPIRED`` — see module docs).
EXIT_STATES: frozenset[State] = RETENTION_EXITS | ADJUDICATED_EXITS


class TransitionTrigger(StrEnum):
    """WHO drove the transition. The pin clause reads this and nothing else about intent.

    ``AUTOMATIC`` is every sweep, strategy, and adjudicator — no human in the loop. ``EXPLICIT``
    is an owner-driven request that arrived through an authorized surface. A sweep must never
    pass ``EXPLICIT`` (spec §6.1 line 287: *"No automatic sweep ever passes it"* — said there of
    ``force_unpinned``, and it holds identically for this flag, which is the softer of the two).
    """

    AUTOMATIC = "automatic"
    EXPLICIT = "explicit"


@runtime_checkable
class PinnableItem(Protocol):
    """The narrow read-only face the policy needs, so ONE guard serves BOTH ``MemoryItem``
    definitions (published and shipped) without either importing the other.

    Declared as read-only properties deliberately: a Protocol's plain attribute members are
    INVARIANT, so ``state: State | MemoryState`` as an attribute would reject a class whose
    ``state`` is exactly ``MemoryState``. Property members are covariant, and the policy only
    ever reads.
    """

    @property
    def id(self) -> str: ...

    @property
    def state(self) -> State | MemoryState: ...

    @property
    def pinned(self) -> bool: ...


def to_contract_state(state: State | MemoryState) -> State:
    """Normalize either lifecycle-state enum onto the published :class:`State` axis."""
    if isinstance(state, State):
        return state
    return ENGINE_TO_CONTRACT_STATE[state]


class LifecyclePolicy:
    """The only place that knows the legal FSM edges (memory-layer §1.1), and the central pin
    guard (memory-health §6.1). Pure, stateless, zero I/O — one instance is reusable everywhere.
    """

    #: The state-axis legal-edge table (memory-layer §1.1 lines 248-258). The TIER axis
    #: (STM->MTM->LTM adjacency) is owned by ``MemoryItem.can_promote_to`` and is deliberately
    #: not duplicated here.
    #:
    #: Two rows are NOT in the design doc's table and are recorded as deltas rather than
    #: silently assumed: ``SUPERSEDED -> DELETED`` and ``EXPIRED -> DELETED``. The doc lists only
    #: ``*/ARCHIVED -> */DELETED``, but the shipped GC (``RetentionService._sweep`` pass 2)
    #: garbage-collects the SUPERSEDED/EXPIRED "dead" set, so refusing those edges would make
    #: the guard reject the one path that actually deletes.
    LEGAL_EDGES: ClassVar[dict[State, frozenset[State]]] = {
        State.ACTIVE: frozenset(
            {State.ARCHIVED, State.SUPERSEDED, State.QUARANTINED, State.EXPIRED}
        ),
        State.ARCHIVED: frozenset({State.ACTIVE, State.DELETED}),
        State.SUPERSEDED: frozenset({State.ACTIVE, State.DELETED}),
        State.QUARANTINED: frozenset({State.ACTIVE}),
        State.EXPIRED: frozenset({State.DELETED}),
        State.DELETED: frozenset(),
    }

    def assert_transition(
        self,
        item: PinnableItem,
        to_state: State | MemoryState,
        *,
        trigger: TransitionTrigger = TransitionTrigger.AUTOMATIC,
        force_unpinned: bool = False,
    ) -> None:
        """Fail loud unless ``item`` may move to ``to_state``.

        Raises ``IllegalTransitionError`` for an edge that is not in :attr:`LEGAL_EDGES`, and
        ``PinnedTransitionBlocked`` (a subclass of it) for a legal edge that the item's pin
        forbids. A same-state call is a legal no-op: re-applying an idempotent write must not be
        an error.

        ``force_unpinned`` bypasses the PIN clause only — never the edge-legality table. It is
        set by ``PinService.unpin``'s follow-on and by an explicit user-driven delete that unpins
        first; **no automatic sweep ever passes it** (spec §6.1 line 287).
        """
        target = to_contract_state(to_state)
        current = to_contract_state(item.state)
        if target is current:
            return
        if target not in self.LEGAL_EDGES[current]:
            # Non-enumerating: names the edge, never the memory's content.
            raise IllegalTransitionError(f"illegal edge {current.value} -> {target.value}")
        if force_unpinned or not item.pinned:
            return
        if target in RETENTION_EXITS:
            raise PinnedTransitionBlocked(f"pinned item blocks retention exit -> {target.value}")
        if target in ADJUDICATED_EXITS and trigger is TransitionTrigger.AUTOMATIC:
            raise PinnedTransitionBlocked(f"pinned item is never the auto-loser -> {target.value}")

    def assert_retention_exit(
        self,
        item: PinnableItem,
        to_state: State | MemoryState,
        *,
        force_unpinned: bool = False,
    ) -> None:
        """The PIN clause for a RETENTION-class exit, asked WITHOUT the edge-legality table.

        CANONICAL §7.10 makes retention-class ineligibility for a pinned item unconditional
        ("never garbage-collected regardless"), so this answer never depends on which state the
        item is coming FROM — which is exactly why it must not consult :attr:`LEGAL_EDGES`. Two
        shipped call sites prove the point:

        * **GC (``RetentionService`` pass 2).** ``LtmRetentionStorePort.facts_by_state`` rebuilds
          each row from the node's ``memory_json`` blob, while ``expire``/``invalidate`` flip only
          the node's ``state`` PROPERTY and never rewrite that blob. A dead row therefore still
          reads ``state=ACTIVE``, and ``ACTIVE -> DELETED`` is not a legal edge — an edge-checking
          guard would raise ``IllegalTransitionError`` and abort the whole sweep instead of
          garbage-collecting. (The staleness is a storage-lane defect, reported separately; this
          method is not a way of hiding it.)
        * **``SurfaceFacade.delete``.** It soft-deletes to ``EXPIRED`` from whatever state the
          copy is in, ARCHIVED included, and ``ARCHIVED -> EXPIRED`` is not in the table either.

        Spec §6.3 line 305 writes this call as ``assert_transition(item, DELETED)``; the split is
        recorded as a spec delta rather than silently made. Edge legality still has exactly one
        owner — :meth:`assert_transition` — and the sites that DO drive the state machine keep
        using it.
        """
        target = to_contract_state(to_state)
        if target not in RETENTION_EXITS:
            # A caller bug, loud: this method answers ONLY the retention-class question.
            raise ValueError(f"{target.value} is not a retention-class exit")
        if force_unpinned or not item.pinned:
            return
        raise PinnedTransitionBlocked(f"pinned item blocks retention exit -> {target.value}")

    def permits_retention_exit(
        self,
        item: PinnableItem,
        to_state: State | MemoryState,
        *,
        force_unpinned: bool = False,
    ) -> bool:
        """ "Skip, keep" form of :meth:`assert_retention_exit` (spec §9 line 381)."""
        try:
            self.assert_retention_exit(item, to_state, force_unpinned=force_unpinned)
        except PinnedTransitionBlocked:
            return False
        return True

    def assert_tier_demotion(
        self,
        item: PinnableItem,
        *,
        trigger: TransitionTrigger = TransitionTrigger.AUTOMATIC,
        force_unpinned: bool = False,
    ) -> None:
        """The TIER-axis half of the pin override: a pinned item is never demoted (spec §6.2,
        line 294 — *"pin overrides the forgetting curve entirely"*).

        This needs its own clause because the SHIPPED demotion is not the state flip the
        design's §1.1 table describes. ``DemotionService`` performs an ``MTM -> STM`` **tier**
        move and leaves ``state=ACTIVE`` (ADR 0034; ``demotion.py``'s own docstring: *"a tier-down
        move is NOT an archival"*), so :meth:`assert_transition`, which guards the STATE axis,
        would never see it and a pinned item would be silently demoted. Recorded as a spec/design
        delta rather than papered over.

        Blocked on the AUTOMATIC trigger regardless of retention/salience; an EXPLICIT owner-
        driven tier move is not a forgetting-curve decision and stays legal, as does any call
        that unpinned first (``force_unpinned``).
        """
        if force_unpinned or not item.pinned:
            return
        if trigger is TransitionTrigger.AUTOMATIC:
            raise PinnedTransitionBlocked("pinned item blocks automatic tier demotion")

    def permits_tier_demotion(
        self,
        item: PinnableItem,
        *,
        trigger: TransitionTrigger = TransitionTrigger.AUTOMATIC,
        force_unpinned: bool = False,
    ) -> bool:
        """ "Skip, keep" form of :meth:`assert_tier_demotion` for an automatic sweep (spec §9)."""
        try:
            self.assert_tier_demotion(item, trigger=trigger, force_unpinned=force_unpinned)
        except PinnedTransitionBlocked:
            return False
        return True

    def permits(
        self,
        item: PinnableItem,
        to_state: State | MemoryState,
        *,
        trigger: TransitionTrigger = TransitionTrigger.AUTOMATIC,
        force_unpinned: bool = False,
    ) -> bool:
        """``True`` iff :meth:`assert_transition` would not raise ``PinnedTransitionBlocked``.

        The "skip, keep" form spec §9 line 381 mandates for an AUTOMATIC sweep: a pin is a normal,
        expected outcome there and must never escalate to an error. An ILLEGAL EDGE is a
        different thing — a caller bug — and still propagates, so this helper can never be used
        to silently swallow one.
        """
        try:
            self.assert_transition(item, to_state, trigger=trigger, force_unpinned=force_unpinned)
        except PinnedTransitionBlocked:
            return False
        return True
