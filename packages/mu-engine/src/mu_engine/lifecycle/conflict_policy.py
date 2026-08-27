"""``ConflictLifecyclePolicy`` — the SINGLE owner of legal ``ConflictRecord.state`` edges.

Authority: ``conflict-resolution-async-design.md`` §3.1 (lines 109-117 — the transition table and
*"Illegal transitions raise ``IllegalConflictTransitionError`` (fail loud, never silent
no-op)"*) · ``CANONICAL-CONTRACTS.md`` §7.20. Deliberately MIRRORS
:mod:`mu_engine.lifecycle.policy` (``LifecyclePolicy``) in shape and conventions: a plain,
pure, stateless class with the legal-edge table as a ``ClassVar``, an ``assert_``/``permits_``
pair where the bool form swallows ONLY the soft refusal, a same-state legal no-op, and
non-enumerating error text.

**Why this file exists.** ``ConflictState``'s own docstring in ``mu_contracts`` promised that
*"``ConflictLifecyclePolicy`` enforces legal transitions"* and
``IllegalConflictTransitionError`` had been defined for it — but nothing enforced anything, and
in practice exactly ONE of the six states was reachable: ``ConflictAdjudicator._park`` hardcodes
``MANUAL_PENDING`` and no code path ever called ``upsert``. Five states were dead vocabulary.

**Two axes, two owners — the boundary that matters.** ``LifecyclePolicy`` owns the ``MemoryItem``
STATE axis *and* the pin guard. This class owns ``ConflictRecord.state`` and NOTHING else. It
does not re-implement the pin check: spec line 117 records that ``AUTO_RESOLVED`` and
``RESOLVED`` *"both correspond to an applied supersession on the memory items"*, and that
supersession is still gated at the write site by ``LifecyclePolicy.permits(loser, SUPERSEDED)``
(``lifecycle/conflict.py`` and ``pipelines/distill.py``). A pinned memory is refused there, which
is the load-bearing refusal; conflict-state legality is a separate question and asking it twice
would create a second, divergent pin rule.

**Trigger-awareness (mirrors ``TransitionTrigger``, reused, not re-declared).** Two edges in the
table are decided by a machine and two by a human, and confusing them is a real defect class: a
background sweep that could write ``RESOLVED`` would silently forge a human decision, and a
manual surface that could write ``AUTO_RESOLVED`` would lose the ``resolution_origin`` audit.
So :data:`AUTOMATIC_ONLY_EDGES` and :data:`EXPLICIT_ONLY_EDGES` gate them, using the SAME
``TransitionTrigger`` enum the memory FSM uses (imported read-only; that module is another
lane's file and is not modified).

**Spec deltas recorded, not silently assumed** (the house standard ``policy.py`` sets):

1. **``DISMISSED -> REOPENED`` is legal here; the §3.1 table draws no exit from ``DISMISSED``.**
   The table (spec lines 112-116) and the prose (spec line 106) contradict each other: line 106
   says a dismissed record *"is **not** re-opened ... **unless** a genuinely new delta changes a
   member's content_hash → ``REOPENED``"*, which is an exit the table omits. Without the edge
   that sentence is unimplementable and a pathological source could silence a genuinely new
   contradiction forever by getting one dismissal. The edge is added, and the "unless" is
   enforced separately and explicitly by :meth:`may_reopen_dismissed`, which requires an actual
   member ``content_hash`` change — so the human's "not a conflict" still holds for the exact
   pair they dismissed. REPORTED as a spec gap needing an owner ruling.
2. **``REOPENED -> RESOLVED`` / ``REOPENED -> DISMISSED`` are legal here; the table routes
   ``REOPENED`` only back to ``DETECTED``.** Spec §5 line 218 makes a resolve action valid when
   *"the record is ``MANUAL_PENDING``/``REOPENED``"*, and §5 line 197 puts ``REOPENED`` in the
   inbox's actionable ``pending`` set. A human looking at a reopened conflict in their inbox must
   be able to decide it; refusing the edge would render the inbox's own contents undecidable.
   REPORTED.
3. **``MANUAL_PENDING -> AUTO_RESOLVED`` is NOT legal**, though nothing in the spec forbids it.
   Once a conflict has been parked for a human, an automatic sweep taking it back is precisely
   the *"silent auto-override of a human"* §7 line 262 forbids. An operator who wants the
   machine to decide it goes through ``reopen`` (EXPLICIT), which re-enters ``DETECTED``.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

from mu_contracts.domain.errors import IllegalConflictTransitionError
from mu_contracts.domain.model.conflict import ConflictRecord, ConflictState
from mu_engine.lifecycle.policy import TransitionTrigger

__all__ = [
    "ACTIONABLE_STATES",
    "AUTOMATIC_ONLY_EDGES",
    "EXPLICIT_ONLY_EDGES",
    "TERMINAL_STATES",
    "ConflictLifecyclePolicy",
    "ConflictTransitionBlockedError",
    "TransitionTrigger",
]


class ConflictTransitionBlockedError(IllegalConflictTransitionError):
    """A LEGAL conflict edge refused because the TRIGGER was wrong for it.

    A SUBCLASS of the illegal-edge error, exactly as ``PinnedTransitionBlocked`` subclasses
    ``IllegalTransitionError`` — that subclassing is what lets :meth:`ConflictLifecyclePolicy
    .permits` catch the soft refusal while an actual illegal edge (a caller bug) still
    propagates. A sweep hitting this has found a human-owned edge and must skip, not crash; a
    sweep hitting a bare ``IllegalConflictTransitionError`` has a bug.
    """


#: Edges only a machine may take. A human decision is ``RESOLVED``/``DISMISSED`` with
#: ``resolution_origin=MANUAL``; letting an explicit surface write ``AUTO_RESOLVED`` would
#: destroy the ``resolution_origin`` audit spec line 117 says is the whole distinction.
AUTOMATIC_ONLY_EDGES: frozenset[tuple[ConflictState, ConflictState]] = frozenset(
    {(ConflictState.DETECTED, ConflictState.AUTO_RESOLVED)}
)

#: Edges only a human may take (spec §5 write actions). A background sweep that could write
#: these would forge a decision nobody made.
EXPLICIT_ONLY_EDGES: frozenset[tuple[ConflictState, ConflictState]] = frozenset(
    {
        (ConflictState.MANUAL_PENDING, ConflictState.RESOLVED),
        (ConflictState.MANUAL_PENDING, ConflictState.DISMISSED),
        (ConflictState.REOPENED, ConflictState.RESOLVED),
        (ConflictState.REOPENED, ConflictState.DISMISSED),
    }
)

#: The inbox's actionable set (spec §5 line 197). A resolve/dismiss action validates against
#: exactly this set before it records any intent.
ACTIONABLE_STATES: frozenset[ConflictState] = frozenset(
    {ConflictState.MANUAL_PENDING, ConflictState.REOPENED}
)

#: States that carry a settled outcome. NOT dead ends — every one can still ``REOPEN`` on a
#: genuinely new contradicting delta; "terminal" here means "no further decision is pending".
TERMINAL_STATES: frozenset[ConflictState] = frozenset(
    {ConflictState.AUTO_RESOLVED, ConflictState.RESOLVED, ConflictState.DISMISSED}
)


@runtime_checkable
class ConflictLike(Protocol):
    """The narrow read-only face the policy needs. A ``Protocol`` (not ``ConflictRecord``
    directly) for the same reason ``PinnableItem`` is one: the storage lane maintains its own
    duplicate ``ConflictState``/record shapes, and one guard should serve both without either
    package importing the other. Declared as a read-only property — the policy only ever reads.
    """

    @property
    def state(self) -> ConflictState: ...


class ConflictLifecyclePolicy:
    """The only place that knows the legal ``ConflictRecord.state`` edges (spec §3.1). Pure,
    stateless, zero I/O — one instance is reusable everywhere."""

    #: The §3.1 transition table (spec lines 112-116), plus the three deltas the module
    #: docstring records. Every value is a ``frozenset`` so the table cannot be mutated by a
    #: caller that got hold of it.
    LEGAL_EDGES: ClassVar[dict[ConflictState, frozenset[ConflictState]]] = {
        ConflictState.DETECTED: frozenset(
            {ConflictState.AUTO_RESOLVED, ConflictState.MANUAL_PENDING}
        ),
        ConflictState.AUTO_RESOLVED: frozenset({ConflictState.REOPENED}),
        ConflictState.MANUAL_PENDING: frozenset({ConflictState.RESOLVED, ConflictState.DISMISSED}),
        ConflictState.RESOLVED: frozenset({ConflictState.REOPENED}),
        # delta 1 — see the module docstring; gated by `may_reopen_dismissed`.
        ConflictState.DISMISSED: frozenset({ConflictState.REOPENED}),
        # delta 2 — the inbox's own contents must be decidable.
        ConflictState.REOPENED: frozenset(
            {ConflictState.DETECTED, ConflictState.RESOLVED, ConflictState.DISMISSED}
        ),
    }

    def assert_transition(
        self,
        record: ConflictLike,
        to_state: ConflictState,
        *,
        trigger: TransitionTrigger = TransitionTrigger.AUTOMATIC,
    ) -> None:
        """Fail loud unless ``record`` may move to ``to_state`` under ``trigger``.

        Raises ``IllegalConflictTransitionError`` for an edge outside :attr:`LEGAL_EDGES`, and
        :class:`ConflictTransitionBlockedError` (a subclass) for a legal edge the trigger forbids.

        A same-state call is a legal no-op. This matters MORE here than on the memory FSM:
        ``conflict_id`` is idempotent by construction (``sha256(ns | sorted(member_ids) |
        predicate_key)``), so re-detecting the same conflict re-asserts the state it already has
        on every sweep. Raising there would make an idempotent detector crash on its second run.
        """
        current = record.state
        if to_state is current:
            return
        if to_state not in self.LEGAL_EDGES[current]:
            # Non-enumerating: names the EDGE only — never a member id, never a predicate.
            raise IllegalConflictTransitionError(
                f"illegal conflict edge {current.value} -> {to_state.value}"
            )
        edge = (current, to_state)
        if edge in AUTOMATIC_ONLY_EDGES and trigger is not TransitionTrigger.AUTOMATIC:
            raise ConflictTransitionBlockedError(
                f"conflict edge {current.value} -> {to_state.value} is automatic-only"
            )
        if edge in EXPLICIT_ONLY_EDGES and trigger is not TransitionTrigger.EXPLICIT:
            raise ConflictTransitionBlockedError(
                f"conflict edge {current.value} -> {to_state.value} requires a human decision"
            )

    def permits(
        self,
        record: ConflictLike,
        to_state: ConflictState,
        *,
        trigger: TransitionTrigger = TransitionTrigger.AUTOMATIC,
    ) -> bool:
        """``True`` iff :meth:`assert_transition` would not raise
        :class:`ConflictTransitionBlockedError`.

        The "skip, keep" form for an automatic sweep: meeting a human-owned edge is a normal,
        expected outcome there and must never escalate to an error. An ILLEGAL EDGE is a
        different thing — a caller bug — and still propagates, so this helper can never be used
        to silently swallow one. (Exactly ``LifecyclePolicy.permits``' asymmetry.)
        """
        try:
            self.assert_transition(record, to_state, trigger=trigger)
        except ConflictTransitionBlockedError:
            return False
        return True

    def may_repark(self, record: ConflictLike, to_state: ConflictState) -> bool:
        """``True`` iff an IDEMPOTENT re-detection may re-assert ``to_state`` on this record.

        The detect side is a sweep: it re-derives the same ``conflict_id`` on every tick and must
        answer "is this record still mine to write, or has a human settled it?" — a *question*,
        never an exception. :meth:`permits` cannot answer it, and that was a live defect: for
        every settled state (``AUTO_RESOLVED``/``RESOLVED``/``DISMISSED``) the edge to
        ``MANUAL_PENDING`` is not merely trigger-blocked but ILLEGAL, so ``permits`` raised the
        bare ``IllegalConflictTransitionError`` it deliberately does not catch. Its ``False``
        branch was unreachable dead code, and the caller's protection came only from an outer
        ``except Exception`` — which logged a false error and then still published the
        "waiting for you" bookend for a conflict the human had already answered.

        So this swallows BOTH refusals: an illegal edge and a trigger-blocked one are the same
        answer to a sweep ("not yours"). It is deliberately a separate method from
        :meth:`permits`, whose asymmetry (illegal edges propagate) is load-bearing for callers
        that are asserting a transition rather than asking whether re-detection may proceed.
        """
        try:
            self.assert_transition(record, to_state, trigger=TransitionTrigger.AUTOMATIC)
        except IllegalConflictTransitionError:
            return False
        return True

    def is_actionable(self, record: ConflictLike) -> bool:
        """``True`` iff a human resolve/dismiss action is valid against this record right now
        (spec §5 line 218). The resolve service asks THIS, never a hand-rolled state check."""
        return record.state in ACTIONABLE_STATES

    @staticmethod
    def may_reopen_dismissed(
        record: ConflictRecord, current_member_hashes: tuple[str, ...]
    ) -> bool:
        """The "unless" half of spec line 106, enforced rather than assumed.

        A ``DISMISSED`` record is re-opened ONLY when a member's ``content_hash`` genuinely
        changed since the dismissal — the human said "not a conflict" about a specific pair of
        facts, and re-asking about the identical pair would make the dismiss button useless
        (and, with a source that keeps re-asserting, a ``REOPENED`` storm — spec line 327).

        Fail-OPEN when the record predates ``member_content_hashes`` (empty tuple): an unknown
        prior state means we cannot prove the pair is unchanged, and the safe direction is to
        ask the human again rather than to silently swallow a possibly-new contradiction.
        ``current_member_hashes`` is compared positionally against ``record.member_ids``, which
        is the alignment ``ConflictRecord`` itself validates.
        """
        if record.state is not ConflictState.DISMISSED:
            return True
        if not record.member_content_hashes:
            return True
        return tuple(current_member_hashes) != record.member_content_hashes
