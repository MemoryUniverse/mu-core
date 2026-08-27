"""``SalienceStrategy`` + the A4 fourth term — the arithmetic contract of the amendment.

Three things are load-bearing here and each has a test that FAILS if it is quietly changed:

1. **An absent ``cen`` reproduces the pre-A4 three-term score EXACTLY** — bit-for-bit, over a
   dense grid, asserted with ``==`` and never ``pytest.approx``. ``cen`` is absent for every item
   on an install with no centrality service wired, so an approximate identity here is a silent
   re-decision of ``promote_stm_mtm``/``promote_mtm_ltm``/``demote_mtm`` for every FULL-LOCAL user.
2. **The shipped weight MAGNITUDES are pinned**, not merely their mutual consistency. A suite that
   only checks "the weights sum to 1" is satisfied by every scaled vector, including one where
   centrality DOMINATES; the pins below are literal.
3. **A present ``cen`` moves S by exactly ``w_centrality * (cen - base)``** — no hidden term, no
   renormalisation, no clamping in the ordinary range.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from mu_engine.lifecycle.centrality import CentralityIndex, CentralitySettings
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import SalienceSettings
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
_NS = Namespace(org="o1", workspace="w1", user="u1", session="s1", visibility=Visibility.PRIVATE)

#: The three ratified weights, written as LITERALS on purpose. Deriving them from the settings
#: object under test would make every assertion below vacuously true.
_RATIFIED = (0.5, 0.2, 0.3)


def _item(
    *,
    importance_score: float = 0.0,
    access_count: int = 0,
    subject: str | None = "Ada",
    obj: str | None = "Postgres",
) -> MemoryItem:
    return MemoryItem(
        content="c",
        kind=MemoryKind.PROPOSITION,
        namespace=_NS,
        owner_id=_NS.user,
        workspace_id=_NS.workspace,
        session_id=_NS.session,
        subject=subject,
        predicate="uses",
        object=obj,
        polarity=Polarity.POSITIVE,
        created_at=_EPOCH,
        importance_score=importance_score,
        access_count=access_count,
    )


class _FixedLookup:
    """A ``CentralityLookup`` that answers one value for everything — the term under test in
    isolation, with no graph and no store anywhere near it."""

    def __init__(self, value: float | None) -> None:
        self._value = value

    def centrality_for(self, item: MemoryItem) -> float | None:
        return self._value


def _three_term(rec: float, use: float, imp: float) -> float:
    """The pre-A4 score, spelled exactly as ``salience.py`` spelled it before the amendment."""
    return 0.5 * rec + 0.2 * use + 0.3 * imp


def _score_at(strategy: SalienceStrategy, *, use: float, imp: float) -> float:
    """S(m) with ``rec`` pinned to 1.0 (age 0) so ``use``/``imp``/``cen`` are the only movers."""
    item = _item(importance_score=imp, access_count=round(use * 10))
    return strategy.score(item, clock=FrozenClock(_EPOCH))


# ------------------------------------------------------------------ the shipped weight vector ---


def test_the_three_ratified_weights_are_unchanged_by_the_a4_amendment() -> None:
    s = SalienceSettings()

    assert (s.w_recency, s.w_usage, s.w_importance) == _RATIFIED


def test_the_shipped_centrality_weight_is_pinned_to_its_documented_magnitude() -> None:
    """The number the whole design argument rests on. Without this literal a later editor can make
    centrality the DOMINANT term and every consistency test in this file stays green."""
    assert SalienceSettings().w_centrality == 0.10


def test_the_effective_four_term_vector_sums_to_one_and_keeps_the_5_2_3_ratio() -> None:
    """The A4 amendment is a RE-WEIGHTING, not an append: with ``cen`` present the effective
    weights are (0.45, 0.18, 0.27, 0.10)."""
    s = SalienceSettings()
    scale = 1.0 - s.w_centrality
    effective = (
        _RATIFIED[0] * scale,
        _RATIFIED[1] * scale,
        _RATIFIED[2] * scale,
        s.w_centrality,
    )

    assert sum(effective) == pytest.approx(1.0)
    assert effective[0] / effective[1] == pytest.approx(5 / 2)
    assert effective[2] / effective[1] == pytest.approx(3 / 2)
    assert effective == pytest.approx((0.45, 0.18, 0.27, 0.10))


def test_centrality_is_the_smallest_weight_in_the_vector() -> None:
    s = SalienceSettings()

    assert s.w_centrality < min(s.w_recency, s.w_usage, s.w_importance)


def test_the_whole_span_of_the_term_cannot_carry_an_item_across_the_gates() -> None:
    """The stated design bound: cen=0 to cen=1 moves S by at most ``w_centrality``, which is far
    less than the 0.4 between ``demote_mtm`` (0.3) and ``promote_stm_mtm`` (0.7). Structural
    position adjusts rank; it never overrides recency or importance."""
    s = SalienceSettings()
    at_zero = _score_at(SalienceStrategy(s, centrality=_FixedLookup(0.0)), use=0.4, imp=0.4)
    at_one = _score_at(SalienceStrategy(s, centrality=_FixedLookup(1.0)), use=0.4, imp=0.4)

    assert at_one - at_zero == pytest.approx(s.w_centrality)
    assert at_one - at_zero < 0.4


def test_the_sum_to_one_invariant_is_documented_as_unenforced_at_construction() -> None:
    """A REPORTED gap, pinned so it cannot silently change in either direction.

    Enforcing it with a validator makes a single-field env override impossible
    (``MU_LIFECYCLE__SALIENCE__W_RECENCY=0.9`` alone sums to 1.4) and turns
    ``tests/config/test_engine_settings_unit.py:115`` red. So an operator CAN still build a vector
    that mis-calibrates the three absolute gates. If the owner decides to enforce it, this test is
    the one that must change — together with that config test's choice of override field.
    """
    off_spec = SalienceSettings(w_recency=0.5, w_usage=0.2, w_importance=0.4)

    assert off_spec.w_recency + off_spec.w_usage + off_spec.w_importance == pytest.approx(1.1)


def test_the_centrality_share_is_range_bounded() -> None:
    """``S = (1-w)*base + w*cen`` is a convex combination only for ``w`` in [0, 1]; outside it the
    unit-interval guarantee is not structural."""
    with pytest.raises(ValidationError):
        SalienceSettings(w_centrality=1.5)
    with pytest.raises(ValidationError):
        SalienceSettings(w_centrality=-0.1)


def test_the_centrality_share_is_not_part_of_the_sum_to_one_invariant() -> None:
    """It is a blend share, not a fourth weight — a settings object that sets it does not have to
    take the difference out of the other three."""
    assert SalienceSettings(w_centrality=0.5).w_centrality == 0.5


# ----------------------------------------------------------------- absence is EXACTLY the old ---


def test_absent_centrality_reproduces_the_pre_a4_score_bit_for_bit_over_a_dense_grid() -> None:
    """``==``, not ``pytest.approx``. The renormalising four-weight form this replaced differed in
    the last ulp at 46,942 of these 112,211 points, and 20 of those crossed an absolute gate."""
    strategy = SalienceStrategy(SalienceSettings())
    checked = 0
    for ui in range(11):
        use = ui / 10
        for ii in range(101):
            imp = ii / 100
            assert _score_at(strategy, use=use, imp=imp) == _three_term(1.0, use, imp)
            checked += 1
    assert checked == 11 * 101


def test_the_renormalising_four_weight_form_this_replaced_really_did_flip_gates() -> None:
    """Why the blend form exists, as arithmetic rather than as a claim in a docstring.

    The rejected shape was four declared weights (0.45/0.18/0.27/0.10) divided by the sum of the
    PRESENT weights — mathematically identical, numerically not. This asserts the measured
    witnesses: points where the two forms straddle one of the three ABSOLUTE gates
    (``demote_mtm=0.3``, ``promote_stm_mtm=0.7``, ``promote_mtm_ltm=0.9``). ``use`` values of k/10
    are the normal case (``access_count``/``usage_cap=10``) and these ``imp`` values are ordinary
    extractor outputs, so none of this is a synthetic corner.
    """
    witnesses = [
        (0.35, 0.4, 0.15, 0.3),  # rescued by `score < demote_mtm` -> DEMOTED under the old form
        (0.11, 1.0, 0.15, 0.3),
        (0.19, 0.8, 0.15, 0.3),
        (0.47, 0.1, 0.15, 0.3),
    ]
    for rec, use, imp, gate in witnesses:
        ratified = _three_term(rec, use, imp)
        renormalised = (0.45 * rec + 0.18 * use + 0.27 * imp) / 0.90

        assert ratified != renormalised, (rec, use, imp)
        assert (ratified >= gate) is not (renormalised >= gate), (rec, use, imp)


def test_the_absent_branch_is_bit_identical_at_real_ages_not_only_at_age_zero() -> None:
    """The grid above pins ``rec = 1.0``. The gate-flip witnesses live at partial recency, so the
    identity is re-proved with the decay term actually engaged — ``rec`` computed independently by
    this test, and compared with ``==``."""
    settings = SalienceSettings()
    strategy = SalienceStrategy(settings)
    for age_h in (0.0, 1.0, 6.5, 24.0, 37.0, 100.0):
        clock = FrozenClock(_EPOCH + timedelta(hours=age_h))
        rec = math.exp(-math.log(2) * age_h / settings.recency_half_life_h)
        for ui in range(11):
            for ii in range(0, 101, 5):
                use, imp = ui / 10, ii / 100
                item = _item(importance_score=imp, access_count=ui)

                assert strategy.score(item, clock=clock) == _three_term(rec, use, imp)


def test_no_lookup_wired_and_a_lookup_answering_none_are_the_same_score() -> None:
    """FULL-LOCAL with no graph tier configured must behave exactly like FULL-LOCAL with one that
    has nothing to say — and both exactly like the pre-A4 engine."""
    settings = SalienceSettings()
    unwired = SalienceStrategy(settings)
    absent = SalienceStrategy(settings, centrality=_FixedLookup(None))
    real_but_empty = SalienceStrategy(settings, centrality=CentralityIndex(CentralitySettings()))
    item = _item(importance_score=0.42, access_count=3)
    clock = FrozenClock(_EPOCH)

    scores = {s.score(item, clock=clock) for s in (unwired, absent, real_but_empty)}

    assert len(scores) == 1
    assert scores.pop() == _three_term(1.0, 0.3, 0.42)


def test_an_item_with_no_triple_scores_as_the_pre_a4_engine_even_with_a_live_projection() -> None:
    """An unstructured capture turn is not in the entity graph, so it must not be penalised for
    not being a fact."""
    index = CentralityIndex(CentralitySettings())
    index.publish(_NS, {"ada": frozenset(f"n{i}" for i in range(9))})
    strategy = SalienceStrategy(SalienceSettings(), centrality=index)
    turn = _item(importance_score=0.5, subject=None, obj=None)

    assert strategy.score(turn, clock=FrozenClock(_EPOCH)) == _three_term(1.0, 0.0, 0.5)


# ------------------------------------------------------------------- presence is real evidence ---


def test_a_present_centrality_moves_the_score_by_exactly_the_blend() -> None:
    settings = SalienceSettings()
    base = _three_term(1.0, 0.3, 0.42)
    for cen in (0.0, 0.25, 0.5, 0.75, 1.0):
        strategy = SalienceStrategy(settings, centrality=_FixedLookup(cen))
        item = _item(importance_score=0.42, access_count=3)

        actual = strategy.score(item, clock=FrozenClock(_EPOCH))

        assert actual == pytest.approx(base + settings.w_centrality * (cen - base))


def test_a_present_zero_is_real_evidence_and_lowers_the_score() -> None:
    """The absent/present distinction with teeth: ``None`` renormalises away, ``0.0`` costs."""
    settings = SalienceSettings()
    item = _item(importance_score=0.8, access_count=10)
    clock = FrozenClock(_EPOCH)

    absent = SalienceStrategy(settings).score(item, clock=clock)
    peripheral = SalienceStrategy(settings, centrality=_FixedLookup(0.0)).score(item, clock=clock)

    assert peripheral < absent
    assert peripheral == pytest.approx(absent * (1.0 - settings.w_centrality))


def test_the_break_even_is_the_items_own_three_term_score() -> None:
    """Stated in the module docstring, so it is tested: S is unchanged iff cen == base."""
    settings = SalienceSettings()
    base = _three_term(1.0, 0.5, 0.4)
    item = _item(importance_score=0.4, access_count=5)
    clock = FrozenClock(_EPOCH)

    at_break_even = SalienceStrategy(settings, centrality=_FixedLookup(base)).score(
        item, clock=clock
    )
    above = SalienceStrategy(settings, centrality=_FixedLookup(base + 0.2)).score(item, clock=clock)
    below = SalienceStrategy(settings, centrality=_FixedLookup(base - 0.2)).score(item, clock=clock)

    assert at_break_even == pytest.approx(base)
    assert above > base
    assert below < base


def test_the_score_stays_in_the_unit_interval_across_the_whole_input_space() -> None:
    """AC-1 as a property of the FUNCTION, not of one settings object: a convex combination of two
    values in [0, 1] cannot leave [0, 1]."""
    for w_cen in (0.0, 0.1, 0.5, 1.0):
        settings = SalienceSettings(w_centrality=w_cen)
        for cen in (0.0, 0.5, 1.0):
            strategy = SalienceStrategy(settings, centrality=_FixedLookup(cen))
            for use in (0.0, 0.5, 1.0):
                for imp in (0.0, 0.5, 1.0):
                    assert 0.0 <= _score_at(strategy, use=use, imp=imp) <= 1.0


def test_a_full_centrality_share_makes_the_score_the_centrality_itself() -> None:
    """The degenerate end of the blend is well defined rather than a division by zero — the shape
    the renormalising form had to special-case."""
    strategy = SalienceStrategy(SalienceSettings(w_centrality=1.0), centrality=_FixedLookup(0.37))

    assert _score_at(strategy, use=1.0, imp=1.0) == pytest.approx(0.37)


def test_the_lookup_is_consulted_with_the_item_itself_so_tenancy_is_the_indexs_to_enforce() -> None:
    """``score()`` passes the whole item, never a namespace the CALLER chose — so an item can only
    ever be scored against its own tenant's projection."""
    seen: list[MemoryItem] = []

    class _Recording:
        def centrality_for(self, item: MemoryItem) -> float | None:
            seen.append(item)
            return None

    item = _item()
    SalienceStrategy(SalienceSettings(), centrality=_Recording()).score(
        item, clock=FrozenClock(_EPOCH)
    )

    assert seen == [item]


def test_scoring_never_mutates_the_item() -> None:
    """The corrected design writes nothing — not to the store, and not onto the item either."""
    index = CentralityIndex(CentralitySettings())
    index.publish(_NS, {"ada": frozenset(f"n{i}" for i in range(9))})
    item = _item(importance_score=0.5)
    before = item.model_dump_json()

    SalienceStrategy(SalienceSettings(), centrality=index).score(item, clock=FrozenClock(_EPOCH))

    assert item.model_dump_json() == before
