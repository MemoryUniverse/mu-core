"""``total_order_key`` — the §7.17 item 4a deterministic total order over ``PrivateDelta``
candidates for the SAME ``memory_id`` (spec decision D-29).

Authority: ``CANONICAL-CONTRACTS.md`` §7.17 item 4 (the seven-term order, :777) and item 4a
(:778-786, ``[resolves O-37]``) — the COMPARISON SEMANTICS three prior proposals (ADR 0046,
SUPERSEDED, kept as the reasoning of record) pinned there, plus clause **(d)** (added
2026-08-24, closing a same-day ambiguity this module's own review surfaced between clause (a)
and PINNED 2 — see below). This module implements those semantics exactly, clause by clause; it
does not re-derive or reinterpret them.

**ONE pure function, in the OPEN repo, never a comparator class or method** (D-29; CANONICAL
:786 "never a comparator inside a commercial plane: conflict resolution is engine quality and
stays in mu-core"). No clock, no I/O, no randomness, no store/registry access — the same two
candidates in, the same winner out, on every process and every replica, with no coordination
(the whole point of :777's "every replica computes the identical winner").

**The seven terms and their direction (item 4a(a)):**
 1. ``pinned``                          — PRECEDENCE, boolean, True beats False
 2. ``resolution_origin == "manual"``   — PRECEDENCE (below #1), boolean, True beats False
 3. ``valid_at_asserted_only``          — boolean (``not valid_at_inferred``), True beats False
 4. ``lamport_vc_order``                — item 4a(c), see ``_cmp_lamport_vc`` below
 5. ``valid_at``                        — later (greater) wins
 6. ``content_hash``                    — no semantic direction; ascending, for stability only
 7. ``device_id`` (``origin_device_id``)— no semantic direction; ascending; makes the chain
                                           TOTAL (unique per device, CANONICAL §7.11) so every
                                           comparison terminates in exactly one winner

**Why term 3 (`valid_at_asserted_only`) is an ORDINARY boolean term, and `valid_at` IS
consulted when BOTH candidates are inferred (item 4a(d), added 2026-08-24 to close an
ambiguity this module's own review surfaced).** Term 3 compares `True` beats `False` like any
other boolean term (item 4a(a)) — an asserted candidate beats an inferred one outright. The
question item 4a(d) settles is what happens when NEITHER side asserted: does `valid_at` (term
5) still get consulted once term 3 and `lamport_vc_order` both tie (the CROSS-DEVICE,
both-inferred case — same-device candidates are already totally ordered by term 4 and never
reach term 5 unpinned by device)? PINNED 2's closing sentence read alone — *"`valid_at` is
authoritative only when both sides asserted it"* — suggests no. Item 4a(d) resolves this in
favour of YES, consulting it, on three grounds: (1) PINNED 2's OPERATIVE sentence is "the
Lamport/vector-clock ordering WINS OVER `valid_at`" — term order already guarantees Lamport is
consulted first, so when it ties it has been consulted and declined, leaving nothing for
`valid_at` to "win over"; the closing sentence is a summary of that rule, not a fourth rule.
(2) PINNED 3 clamps an inferred `valid_at` "even where the clock is consulted **at all**" —
which presupposes an inferred `valid_at` IS sometimes consulted; under the opposite reading it
never would be, and the clamp would protect nothing. **The clamp is the mitigation** that makes
consulting it safe, and is why this does not reintroduce X8's "a phone with a clock 3 hours
fast always wins" — an inferred `valid_at` reaching this term has already been bounded to
`min(inferred, hub_receive_time + max_skew_s)` on append (PINNED 3), so a skewed clock can win
by at most `max_skew_s`, never arbitrarily. (3) Clause (a) names "the boolean terms" in the
PLURAL — the seven terms contain exactly three booleans (`pinned`, `manual`, term 3) — it is
classifying term 3 with the other two, not exempting it. The alternative (skip `valid_at`
entirely on a both-inferred tie, fall to `content_hash`) trades a bounded, unreliable-but
-meaningful signal for a value with no semantic meaning at all — between two beliefs whose
times are both inferred-and-clamped, the later one is the better guess; a hash is not a guess.

**Why a strict left-to-right short-circuit already IS the item 4a(b) PRECEDENCE, not a
lexicographic tuple read the wrong way.** ADR 0046 flagged the risk as: read "lexicographically"
and a pinned item could still lose on a later term. That risk is real for a WEIGHTED or SCORED
composite of the seven terms (e.g. a sum with per-term weights, where a large enough downstream
term could outweigh an earlier one) — it is NOT real for genuine left-to-right, short-circuiting
comparison, which is what a lexicographic tuple comparison and item 4a(b)'s PRECEDENCE both
mean here: once term 1 (``pinned``) differs, no other term is ever consulted, full stop. This
function never scores or sums; it returns the first non-zero term, exactly reproducing "pinned
is never the auto-supersede/quarantine loser" (CANONICAL:621/:777) and "a manual winner is
sticky... dominated only by pinned" (item 4a(b)).

**Why term 4 is comparable ONLY within one device (item 4a(c)).** A genuine vector-clock
dominance check needs each side's full view of every OTHER device's highest-accepted Lamport
value — the hub's durable ``VC = {device_id -> last_lamport}`` (CANONICAL §7.17-5;
device-registry-sync-design.md §6.2-5). That is REGISTRY state, not a field on either candidate,
and this function is a pure function of the two candidates alone. What IS on a ``PrivateDelta``
is only the AUTHORING device's own scalar ``lamport`` — "each device increments a monotonic
lamport on every private write it authors... within and across a device's OWN edits"
(device-registry-sync-design.md:551, emphasis on "own"). So: two candidates from the SAME
device have a real, total order (their shared counter is strictly monotonic — compare it
directly); two candidates from DIFFERENT devices are genuinely INCOMPARABLE from these two
candidates' fields alone — nothing here can tell "A happened-before B" from "A and B are
concurrent" — so they are a TIE that falls through to ``valid_at``, exactly item 4a(c)'s
prescribed handling of a concurrent pair. This is not an approximation of the "real" check; it
is what a real vector-clock dominance check, restricted to exactly the fields available on two
bare deltas, correctly returns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, NamedTuple, Protocol, TypeVar

from mu_contracts.domain.model.conflict import ResolutionOrigin
from mu_contracts.domain.model.device_sync import PrivateDelta
from mu_engine.storage.domain.memory import MemoryItem

__all__ = ["total_order_key", "total_order_key_items"]

#: The ``clock_group`` every ``MemoryItem`` projection carries. Two LIVE items being adjudicated
#: on one replica are, by construction, both "here": there is no second device in the comparison,
#: so term 4 is always COMPARABLE for items (contrast two deltas from different devices, which
#: are genuinely concurrent and tie). Named rather than inlined so the reason survives.
_LOCAL_REPLICA = "local"


class _OrderTerms(NamedTuple):
    """The seven §7.17 item-4a terms, projected off whatever candidate shape is being compared.

    **This is the ONE ordering in the system, and it is written once.** ``total_order_key``
    (deltas) and :func:`total_order_key_items` (live items) are two ADAPTERS onto the same term
    chain in :func:`_compare` — not two orderings. conflict-async §4.2 line 164 requires the auto
    strategies to reuse the §7.17 order, and the only faithful way to do that was a shared chain
    with two projections, because ``total_order_key``'s own signature takes two ``PrivateDelta``s
    for the SAME ``memory_id`` while a conflict is by definition two DIFFERENT memory ids that
    carry no ``lamport``, no ``origin_device_id`` and no ``resolution_origin`` at all.

    A ``NamedTuple``, not a pydantic model, for the same reason ``AdjudicationBudget`` is a plain
    class: this is an internal comparison vector built twice per comparison inside a pure hot
    function, never a domain DTO crossing a boundary — pydantic validation here would be pure
    cost with nothing to validate (every field is already typed at its projection site).
    """

    pinned: bool  # term 1 — PRECEDENCE, dominant over everything below
    manual: bool  # term 2 — sticky manual, below #1 only
    asserted: bool  # term 3 — valid_at_asserted_only
    clock_group: str  # term 4's comparability gate (device for deltas, replica for items)
    clock: Any  # term 4's monotone counter WITHIN that group; see the two projections
    valid_at: datetime  # term 5 — later wins
    content_hash: str  # term 6 — no semantic direction; ascending, stability only
    identity: str  # term 7 — no semantic direction; ascending; makes the chain TOTAL


def _compare(a: _OrderTerms, b: _OrderTerms) -> int:
    """The seven-term short-circuiting chain. The single source of ordering truth."""
    for term in (
        _cmp_bool(a.pinned, b.pinned),  # 4a(b) term 1 — dominant over everything below
        _cmp_bool(a.manual, b.manual),  # 4a(b) term 2 — sticky below #1 only
        _cmp_bool(a.asserted, b.asserted),  # 4a(d) — ordinary boolean; ties at False/False too
        _cmp_clock(a, b),  # 4a(c) — same group totally ordered; cross-group a TIE
        _cmp(a.valid_at, b.valid_at),  # later wins
        _cmp(a.content_hash, b.content_hash),  # no semantic direction; ascending tiebreak
        _cmp(a.identity, b.identity),  # ascending; TOTAL (unique per device / per item)
    ):
        if term != 0:
            return term
    return 0


def _cmp_clock(a: _OrderTerms, b: _OrderTerms) -> int:
    """item 4a(c) — see the module docstring for why same-group-only is the correct, faithful
    reading of a pure, registry-free vector-clock comparison."""
    if a.clock_group != b.clock_group:
        return 0  # concurrent — no cross-group causal info on the candidates themselves
    return _cmp(a.clock, b.clock)


def _delta_terms(delta: PrivateDelta) -> _OrderTerms:
    return _OrderTerms(
        pinned=delta.pinned,
        manual=delta.resolution_origin is ResolutionOrigin.MANUAL,
        # valid_at_asserted_only: True iff THIS candidate's valid_at was asserted, not inferred
        # from the DATE_EXTRACTION_FALLBACK wall-clock path (CANONICAL §7.17 PINNED 1).
        asserted=not delta.valid_at_inferred,
        clock_group=delta.origin_device_id,
        clock=delta.lamport,
        valid_at=delta.valid_at,
        content_hash=delta.content_hash,
        identity=delta.origin_device_id,
    )


def _item_terms(item: MemoryItem) -> _OrderTerms:
    """Project a LIVE ``MemoryItem`` onto the same seven terms (conflict-async §4.2).

    Four terms have no field on a live item, and each is answered rather than invented:

    * **term 2 (``manual``) is ``False``, always.** A manual decision is a property of a
      *resolution*, not of a memory: it lives on the ``ConflictRecord`` and on the ``PrivateDelta``
      the resolution emits. Stickiness is therefore enforced where it actually applies — in
      ``services.conflict.convergence`` over the delta set (§7) — and never here, because a live
      item that carried a manual flag would let one replica's already-applied decision be
      re-derived as if it were a fresh candidate property.
    * **term 3 (``asserted``) reads ``metadata["valid_at_inferred"]``.** That is the shipped
      convention, not a guess: ``DistillPipeline`` writes the flag there (``distill.py:535``,
      ``:551``) and reads it back at ``distill.py:1002`` to build the sync delta's own
      ``valid_at_inferred``. Absent key means asserted, matching that reader's ``False`` default.
      (``MemoryItem`` having no first-class ``valid_at_inferred`` column is a storage-lane gap,
      reported; this projection uses the same channel the sync-delta builder already uses, so the
      two can never disagree.)
    * **term 4 (``clock``) is ``created_at``, the ASSERTION instant.** On one replica every
      candidate is same-"device", so term 4 is comparable, and ``created_at`` is that replica's
      monotone author-order counter — the exact role ``lamport`` plays for a delta. This is also
      what makes the item order AGREE with the shipped BUG1 rule (``distill.asserted_later``,
      ``distill.py:705-720``) instead of contradicting it: assertion recency decides, and
      ``valid_at`` (term 5) is consulted only to break an assertion-time tie. Design line 130
      reads "newest asserted ``valid_at`` wins", which is the opposite priority; that conflict is
      REPORTED as a spec gap rather than resolved by silently picking one. Term order here follows
      §7.17, whose term 4 (the logical clock) genuinely does precede its term 5 (``valid_at``) —
      so this is the §7.17 order applied faithfully, and it happens to reproduce BUG1's fix.
    * **term 7 (``identity``) is ``item.id``.** Unique per item (CANONICAL §7.1 tier-stable id),
      which is what makes the chain TOTAL — the role ``origin_device_id`` plays for deltas.
    """
    return _OrderTerms(
        pinned=item.pinned,
        manual=False,
        asserted=not bool(item.metadata.get("valid_at_inferred", False)),
        clock_group=_LOCAL_REPLICA,
        clock=item.created_at,
        valid_at=item.valid_at if item.valid_at is not None else item.created_at,
        content_hash=item.content_hash,
        identity=item.id,
    )


def total_order_key_items(a: MemoryItem, b: MemoryItem) -> int:
    """Compare two LIVE ``MemoryItem`` conflict candidates under the SAME §7.17 item-4a order.

    Returns a positive int if ``a`` dominates ``b``, negative if ``b`` dominates, and ``0`` only
    if the two are the same item (term 7 is unique per id, so any two distinct items are
    strictly ordered — the property every AUTOMATIC strategy relies on to never need a
    coin-flip). Pure: no clock, no I/O, no randomness, no store — the same two items in, the
    same winner out, on every process and every replica.

    This is the entry point ``AutoResolveStrategy.RECENCY`` IS, and the tie-break every other
    strategy falls back to (conflict-async §4.2).
    """
    return _compare(_item_terms(a), _item_terms(b))


def total_order_key(a: PrivateDelta, b: PrivateDelta) -> int:
    """Compare two ``PrivateDelta`` candidates for the SAME ``memory_id`` under the §7.17 item
    4a total order.

    Returns a positive int if ``a`` is the winner (dominates ``b``), a negative int if ``b`` is
    the winner, and ``0`` only when every term ties — the two candidates are identical in every
    field this order reads (the idempotent-duplicate case; §7.11's unique ``device_id`` still
    makes the chain total for any two GENUINELY distinct deltas, since two deltas from the same
    device never share a ``lamport`` value and two from different devices never share a
    ``device_id``).

    Antisymmetric and deterministic by construction: ``total_order_key(a, b) ==
    -total_order_key(b, a)`` for any two candidates, always, on any process, on any replica
    (CANONICAL:777) — every branch below reads only ``a``/``b``'s own fields, never a clock,
    never I/O, never a store. Usable as a ``cmp``-style comparator, e.g. via
    ``functools.cmp_to_key(total_order_key)``.
    """
    return _compare(_delta_terms(a), _delta_terms(b))


def _cmp_bool(a_val: bool, b_val: bool) -> int:
    # True beats False (item 4a(a)) — equivalent to int comparison since True > False.
    return int(a_val) - int(b_val)


class _Orderable(Protocol):
    """Structural bound for the term VALUE types ``_cmp`` compares — ``datetime``, ``str``, and
    ``int`` all satisfy this via their own ``__eq__``/``__gt__``."""

    def __eq__(self, other: object) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...


_T = TypeVar("_T", bound=_Orderable)


def _cmp(a_val: _T, b_val: _T) -> int:
    # generic "greater wins" comparison (item 4a(a)) shared by valid_at/content_hash/device_id/
    # lamport — all four are plain values with no partial-order subtlety of their own (that
    # subtlety lives entirely in `_cmp_lamport_vc`'s device_id gate above).
    if a_val == b_val:
        return 0
    return 1 if a_val > b_val else -1
