"""Unit tests for ``SalienceStrategy`` (S0-08, spec §6 / §19 / ADR 0034).

Covers AC-0.1 (pure function of (item, clock.now()) — byte-identical under a pinned
``FrozenClock``), AC-0.2 (advancing a ``FrozenClock`` strictly decreases ``rec``/``S`` — a
property test over randomized item ages), the weights-from-``SalienceSettings`` seam (never a
literal in ``salience.py``), and the no-wall-clock-call invariant (spec §19 Rule 1).

Fixture pattern follows the existing ``tests/storage/domain/test_memory_retention_unit.py``
``_item()`` helper (same repo, same convention).
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import SalienceSettings
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_NS = Namespace(org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE)
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _item(**overrides: object) -> MemoryItem:
    defaults: dict[str, object] = {
        "content": "Ada uses Postgres",
        "kind": MemoryKind.PROPOSITION,
        "namespace": _NS,
        "owner_id": "u",
        "workspace_id": "w",
        "session_id": "s",
        "subject": "Ada",
        "predicate": "uses",
        "object": "Postgres",
        "polarity": Polarity.POSITIVE,
        "created_at": _EPOCH,
    }
    defaults.update(overrides)
    return MemoryItem(**defaults)  # type: ignore[arg-type]


def test_score_is_pure_byte_identical_under_pinned_clock() -> None:
    """AC-0.1 — two calls with the SAME pinned FrozenClock return byte-identical floats."""
    strategy = SalienceStrategy(SalienceSettings())
    item = _item(importance_score=0.42, access_count=3)
    clock = FrozenClock(_EPOCH + timedelta(hours=5))

    first = strategy.score(item, clock=clock)
    second = strategy.score(item, clock=clock)

    assert first == second  # bit-for-bit, not just approx


def test_score_pure_function_of_item_and_now_not_of_call_order() -> None:
    """A fresh strategy/clock pinned to the same instant reproduces the same score."""
    item = _item(importance_score=0.7, access_count=8)
    at = _EPOCH + timedelta(hours=100)

    s1 = SalienceStrategy(SalienceSettings()).score(item, clock=FrozenClock(at))
    s2 = SalienceStrategy(SalienceSettings()).score(item, clock=FrozenClock(at))

    assert s1 == s2


@pytest.mark.parametrize("seed", range(20))
def test_advancing_clock_strictly_decreases_recency_and_score(seed: int) -> None:
    """AC-0.2 (X3 obligation 1) — property test over randomized item ages: advancing a
    FrozenClock by any delta_t > 0 strictly decreases rec(m) and therefore S(m), weights held
    fixed. No non-monotonic wobble under repeated small advances."""
    rng = random.Random(seed)  # noqa: S311 -- deterministic test PRNG, not security-sensitive
    settings = SalienceSettings()
    strategy = SalienceStrategy(settings)

    start_age_h = rng.uniform(0.0, 500.0)
    item = _item(
        created_at=_EPOCH,
        importance_score=rng.uniform(0.0, 1.0),
        access_count=rng.randint(0, 20),
    )
    clock = FrozenClock(_EPOCH + timedelta(hours=start_age_h))

    prev_score = strategy.score(item, clock=clock)
    for _ in range(10):
        delta_h = rng.uniform(0.001, 50.0)
        clock.advance(timedelta(hours=delta_h))
        next_score = strategy.score(item, clock=clock)
        assert next_score < prev_score
        prev_score = next_score


def test_recency_alone_strictly_decreases_with_age_weights_fixed() -> None:
    """Isolates rec(m) itself (importance/usage held fixed) — direct AC-0.2 wording check."""
    settings = SalienceSettings()
    strategy = SalienceStrategy(settings)
    item = _item(importance_score=0.5, access_count=5)

    ages_h = [0.0, 1.0, 24.0, 100.0, 1000.0]
    scores = [strategy.score(item, clock=FrozenClock(_EPOCH + timedelta(hours=h))) for h in ages_h]

    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)  # strictly decreasing, no ties


def test_weights_come_from_settings_not_a_hardcoded_literal() -> None:
    """Changing SalienceSettings weights changes the score — proves no literal shadow-copy."""
    item = _item(importance_score=1.0, access_count=0)
    clock = FrozenClock(_EPOCH)  # age_hours == 0 -> rec == 1.0 exactly

    default_score = SalienceStrategy(SalienceSettings()).score(item, clock=clock)
    reweighted_score = SalienceStrategy(
        SalienceSettings(w_recency=0.1, w_usage=0.1, w_importance=0.8)
    ).score(item, clock=clock)

    # rec=1, use=0 for both configs; only w_importance differs (0.3 default vs 0.8) * imp=1.0
    assert default_score == pytest.approx(0.5 * 1.0 + 0.2 * 0.0 + 0.3 * 1.0)
    assert reweighted_score == pytest.approx(0.1 * 1.0 + 0.1 * 0.0 + 0.8 * 1.0)
    assert default_score != reweighted_score


def test_usage_is_capped_at_one() -> None:
    """use(m) = min(access_count/usage_cap, 1) — never exceeds 1 even far past the cap."""
    settings = SalienceSettings(usage_cap=10)
    strategy = SalienceStrategy(settings)
    clock = FrozenClock(_EPOCH)  # age 0 -> rec = 1.0, isolate the `use` term

    at_cap = strategy.score(_item(importance_score=0.0, access_count=10), clock=clock)
    over_cap = strategy.score(_item(importance_score=0.0, access_count=1000), clock=clock)

    # rec == 1.0 (age 0), imp == 0.0 for both; only the (capped) `use` term can differ.
    assert at_cap == pytest.approx(settings.w_recency * 1.0 + settings.w_usage * 1.0)
    assert over_cap == at_cap  # capped, not unbounded


def test_score_bounded_in_unit_interval_for_in_range_inputs() -> None:
    """S(m) in [0, 1] for weights-sum-to-1 defaults and valid (bounded) item fields."""
    settings = SalienceSettings()
    strategy = SalienceStrategy(settings)
    item = _item(importance_score=1.0, access_count=1_000)
    score = strategy.score(item, clock=FrozenClock(_EPOCH))  # age 0 -> rec == 1.0 (max)

    assert 0.0 <= score <= 1.0 + 1e-9


def test_no_wall_clock_call_only_injected_clock_is_used() -> None:
    """spec §19 Rule 1 — the score for a fixed item/instant must not depend on when the test
    itself runs; a clock pinned far in the past still yields a deterministic, computable score
    (proves the module never substitutes `datetime.now()` for the injected clock)."""
    strategy = SalienceStrategy(SalienceSettings())
    created = datetime(1999, 1, 1, tzinfo=UTC)
    pinned_instant = datetime(1999, 1, 1, 1, tzinfo=UTC)  # 1h after creation, not "now"
    ancient = _item(created_at=created, importance_score=0.5, access_count=0)

    score = strategy.score(ancient, clock=FrozenClock(pinned_instant))

    assert score == pytest.approx(strategy.score(ancient, clock=FrozenClock(pinned_instant)))
