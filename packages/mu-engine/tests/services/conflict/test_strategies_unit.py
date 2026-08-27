"""§4.2 — the deterministic AUTOMATIC winner-pickers. Offline: no store, no clock, no model.

Covers conflict-resolution-async-design.md §4.2 (lines 162-164) and the pin clause
(CANONICAL §7.17 item 4a(b) / memory-health §6.4), which this lane must never re-implement.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.model.conflict import (
    AutoResolveStrategy,
    ConflictRecord,
    ConflictResolutionKind,
    ConflictState,
)
from mu_contracts.domain.model.memory import Namespace as ContractNamespace
from mu_contracts.domain.model.memory import Visibility as ContractVisibility
from mu_engine.lifecycle.conflict import ConflictResolutionPolicy
from mu_engine.services.conflict.order import total_order_key_items
from mu_engine.services.conflict.strategies import (
    recommended_resolution_kind,
    resolve_automatically,
)
from mu_engine.storage.domain.memory import MemoryItem, MemorySource, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _fact(
    ns: Namespace,
    *,
    memory_id: str,
    obj: str,
    created_at: datetime,
    pinned: bool = False,
    source: MemorySource = MemorySource.USER,
    valid_at: datetime | None = None,
    valid_at_inferred: bool = False,
) -> MemoryItem:
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
        valid_at=valid_at if valid_at is not None else created_at,
        subject="user",
        predicate="lives_in",
        object=obj,
        pinned=pinned,
        source=source,
        metadata={"valid_at_inferred": True} if valid_at_inferred else {},
    )


def _policy(strategy: AutoResolveStrategy, **kw: object) -> ConflictResolutionPolicy:
    return ConflictResolutionPolicy(strategy=strategy, **kw)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════ 1. THE PIN CLAUSE ══════════════════════════
def test_a_pinned_memory_is_never_the_auto_supersede_loser(ns: Namespace) -> None:
    """The load-bearing rule. ``pinned`` is term 1 of the total order, so the pinned item wins
    outright; and even if the pick had gone the other way, the central ``LifecyclePolicy`` guard
    refuses the automatic exit and the whole resolution comes back inapplicable."""
    pinned = _fact(ns, memory_id="pinned", obj="Berlin", created_at=_T0, pinned=True)
    newer = _fact(ns, memory_id="newer", obj="Lisbon", created_at=_T0 + timedelta(days=30))

    outcome = resolve_automatically([pinned, newer], policy=_policy(AutoResolveStrategy.RECENCY))

    assert outcome.winner_id == "pinned", "term 1 (pinned) dominates assertion recency"
    assert outcome.loser_ids == ("newer",)
    assert outcome.applicable is True
    assert outcome.pin_blocked is False


def test_a_conflict_whose_only_available_loser_is_pinned_is_parked_not_applied(
    ns: Namespace,
) -> None:
    """Both members pinned: whichever wins, the loser is pinned, so ``LifecyclePolicy`` refuses
    the AUTOMATIC exit and the resolution declines rather than superseding a pinned memory."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0, pinned=True)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1), pinned=True)

    outcome = resolve_automatically([a, b], policy=_policy(AutoResolveStrategy.RECENCY))

    assert outcome.applicable is False, "a pinned loser must park the conflict, never supersede"
    assert outcome.pin_blocked is True
    assert outcome.reason == "pin_blocked_pinned_item_is_never_the_auto_loser"


def test_the_pin_clause_holds_under_every_strategy(ns: Namespace) -> None:
    """A rule enforced by only one of three code paths is a rule with two holes."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0, pinned=True)
    b = _fact(
        ns,
        memory_id="b",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=1),
        pinned=True,
        source=MemorySource.INFERRED,
    )
    for strategy in AutoResolveStrategy:
        outcome = resolve_automatically(
            [a, b], policy=_policy(strategy), proposed_winner_id="b", confidence=1.0
        )
        assert outcome.applicable is False, strategy
        assert outcome.pin_blocked is True, strategy


# ═══════════════════════════════════════════════ 2. DETERMINISM ═════════════════════════════
def test_the_winner_does_not_depend_on_candidate_order(ns: Namespace) -> None:
    """Every replica computes the identical winner with no coordination (spec line 164). Input
    order is the cheapest way for that to break, and the least visible."""
    items = [
        _fact(ns, memory_id="a", obj="Berlin", created_at=_T0),
        _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1)),
        _fact(ns, memory_id="c", obj="Oslo", created_at=_T0 + timedelta(days=2)),
    ]
    policy = _policy(AutoResolveStrategy.RECENCY)
    winners = {
        resolve_automatically(list(perm), policy=policy).winner_id
        for perm in itertools.permutations(items)
    }
    assert winners == {"c"}


def test_repeated_calls_return_the_identical_result(ns: Namespace) -> None:
    items = [
        _fact(ns, memory_id="a", obj="Berlin", created_at=_T0),
        _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1)),
    ]
    policy = _policy(AutoResolveStrategy.RECENCY)
    first = resolve_automatically(items, policy=policy)
    for _ in range(20):
        assert resolve_automatically(items, policy=policy) == first


def test_an_exact_assertion_tie_is_still_totally_ordered(ns: Namespace) -> None:
    """Terms 6-7 (``content_hash`` then ``id``) make the chain TOTAL, so even a perfect tie on
    every SEMANTIC term never needs a coin flip — which is what lets two replicas agree without
    talking. The test asserts the PROPERTY (one stable, order-independent winner), never a
    particular id: which of the two wins is a hash ordering with no semantic direction, and
    pinning it here would be asserting an implementation detail rather than the guarantee."""
    a = _fact(ns, memory_id="aaa", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="zzz", obj="Lisbon", created_at=_T0)
    policy = _policy(AutoResolveStrategy.RECENCY)
    forward = resolve_automatically([a, b], policy=policy).winner_id
    backward = resolve_automatically([b, a], policy=policy).winner_id
    assert forward == backward
    assert total_order_key_items(a, b) != 0, "distinct items must never compare equal"


# ══════════════════════════════════ 3. ONE ORDERING, NOT TWO ════════════════════════════════
def test_recency_is_exactly_the_shared_total_order(ns: Namespace) -> None:
    """RECENCY must BE ``total_order_key_items``, not agree with it by coincidence: if the
    picker ever grew its own comparison, this is what catches the divergence."""
    items = [
        _fact(ns, memory_id="a", obj="Berlin", created_at=_T0, source=MemorySource.INFERRED),
        _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1)),
        _fact(ns, memory_id="c", obj="Oslo", created_at=_T0, valid_at_inferred=True),
    ]
    policy = _policy(AutoResolveStrategy.RECENCY)
    for pair in itertools.combinations(items, 2):
        expected = pair[0].id if total_order_key_items(pair[0], pair[1]) > 0 else pair[1].id
        assert resolve_automatically(list(pair), policy=policy).winner_id == expected


def test_an_asserted_valid_at_beats_an_inferred_one(ns: Namespace) -> None:
    """Term 3 of the §7.17 order (CANONICAL PINNED 1): a wall-clock-guessed date must not
    outrank a date the source actually asserted."""
    asserted = _fact(ns, memory_id="asserted", obj="Berlin", created_at=_T0)
    inferred = _fact(ns, memory_id="inferred", obj="Lisbon", created_at=_T0, valid_at_inferred=True)
    outcome = resolve_automatically(
        [inferred, asserted], policy=_policy(AutoResolveStrategy.RECENCY)
    )
    assert outcome.winner_id == "asserted"


def test_a_back_dated_valid_at_cannot_outrank_a_later_assertion(ns: Namespace) -> None:
    """Term 4 (assertion recency) precedes term 5 (``valid_at``) — the shipped BUG1 rule and
    §7.17's own term order agreeing. A source claiming a fact was valid in 2099 must not win
    over what the user actually said most recently."""
    older_assertion = _fact(
        ns,
        memory_id="old",
        obj="Berlin",
        created_at=_T0,
        valid_at=datetime(2099, 1, 1, tzinfo=UTC),
    )
    newer_assertion = _fact(ns, memory_id="new", obj="Lisbon", created_at=_T0 + timedelta(days=1))
    outcome = resolve_automatically(
        [older_assertion, newer_assertion], policy=_policy(AutoResolveStrategy.RECENCY)
    )
    assert outcome.winner_id == "new"


# ═══════════════════════════════════ 4. THE OTHER TWO STRATEGIES ════════════════════════════
def test_provenance_prefers_the_more_trusted_source(ns: Namespace) -> None:
    inferred = _fact(
        ns,
        memory_id="inferred",
        obj="Lisbon",
        created_at=_T0 + timedelta(days=30),
        source=MemorySource.INFERRED,
    )
    stated = _fact(ns, memory_id="stated", obj="Berlin", created_at=_T0, source=MemorySource.USER)
    outcome = resolve_automatically(
        [inferred, stated], policy=_policy(AutoResolveStrategy.PROVENANCE)
    )
    assert outcome.winner_id == "stated", "what the USER said outranks what the system inferred"


def test_provenance_falls_back_to_the_same_total_order_on_a_trust_tie(ns: Namespace) -> None:
    """Line 132: "tie → the same total order". With one source it must degenerate exactly to
    RECENCY, not to something else."""
    a = _fact(ns, memory_id="a", obj="Berlin", created_at=_T0)
    b = _fact(ns, memory_id="b", obj="Lisbon", created_at=_T0 + timedelta(days=1))
    assert (
        resolve_automatically([a, b], policy=_policy(AutoResolveStrategy.PROVENANCE)).winner_id
        == resolve_automatically([a, b], policy=_policy(AutoResolveStrategy.RECENCY)).winner_id
    )


def test_confidence_honours_the_adjudicator_pick_only_above_the_floor(ns: Namespace) -> None:
    older = _fact(ns, memory_id="older", obj="Berlin", created_at=_T0)
    newer = _fact(ns, memory_id="newer", obj="Lisbon", created_at=_T0 + timedelta(days=1))
    policy = _policy(AutoResolveStrategy.CONFIDENCE, auto_min_confidence=0.8)

    confident = resolve_automatically(
        [older, newer], policy=policy, proposed_winner_id="older", confidence=0.95
    )
    assert confident.winner_id == "older", "a confident adjudicator pick stands"

    unsure = resolve_automatically(
        [older, newer], policy=policy, proposed_winner_id="older", confidence=0.2
    )
    assert unsure.winner_id == "newer", "below the floor it falls back to the total order"


def test_confidence_ignores_a_pick_that_is_not_even_a_member(ns: Namespace) -> None:
    """A stale or malformed pick must never select nobody, or a caller would supersede both
    real members in favour of an unrelated memory."""
    older = _fact(ns, memory_id="older", obj="Berlin", created_at=_T0)
    newer = _fact(ns, memory_id="newer", obj="Lisbon", created_at=_T0 + timedelta(days=1))
    outcome = resolve_automatically(
        [older, newer],
        policy=_policy(AutoResolveStrategy.CONFIDENCE),
        proposed_winner_id="a-memory-from-another-conflict",
        confidence=1.0,
    )
    assert outcome.winner_id == "newer"


# ═════════════════════════════════════════════ 5. GUARDS ════════════════════════════════════
def test_a_single_candidate_is_not_a_conflict(ns: Namespace) -> None:
    with pytest.raises(ValueError, match="at least two candidates"):
        resolve_automatically(
            [_fact(ns, memory_id="a", obj="Berlin", created_at=_T0)],
            policy=_policy(AutoResolveStrategy.RECENCY),
        )


def test_no_strategy_consults_a_model_or_a_clock(ns: Namespace) -> None:
    """§4.2: "auto strategies are ALL DETERMINISTIC". ``resolve_automatically`` takes no router,
    no clock and no repository at all — the absence is the guarantee, so it is asserted on the
    signature rather than on behaviour."""
    import inspect

    params = set(inspect.signature(resolve_automatically).parameters)
    assert params == {"candidates", "policy", "proposed_winner_id", "confidence", "lifecycle"}


# ═════════════════════════════════════ 6. THE §4 UNCERTAINTY BANDS ══════════════════════════
def _record(ns: ContractNamespace, confidence: float) -> ConflictRecord:
    return ConflictRecord(
        conflict_id="c1",
        namespace=ns,
        member_ids=("a", "b"),
        predicate_key="lives_in",
        method="llm_adjudicator",
        detected_confidence=confidence,
        state=ConflictState.MANUAL_PENDING,
        detected_at=_T0,
    )


def test_the_quarantine_band_is_derived_from_the_record_and_its_policy() -> None:
    ns = ContractNamespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=ContractVisibility.PRIVATE
    )
    policy = ConflictResolutionPolicy(auto_min_confidence=0.8, quarantine_below=0.5)
    assert recommended_resolution_kind(_record(ns, 0.2), policy) is (
        ConflictResolutionKind.QUARANTINE
    )
    assert recommended_resolution_kind(_record(ns, 0.6), policy) is (
        ConflictResolutionKind.SUPERSEDE
    )


def test_the_two_uncertainty_thresholds_may_not_contradict_each_other() -> None:
    """The §4 truth table's bands are ordered and disjoint by construction — a policy that
    inverted them would leave the middle band empty with no defined outcome."""
    with pytest.raises(ValueError, match="quarantine_below"):
        ConflictResolutionPolicy(auto_min_confidence=0.4, quarantine_below=0.9)
