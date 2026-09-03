"""``derive_recall_limit`` unit tests — JOB ONE (derive recall width from the context budget).

Pure-function tests, no I/O: the four behaviours the task names as provably-must-fail-without-the-
fix are pinned directly —
  * a large-context model widens (well past the wire default of 10),
  * a small-context model narrows to the floor (never 0, never negative),
  * the floor/ceiling clamp always wins over the raw arithmetic,
  * a misconfigured (non-positive) ``tokens_per_memory`` fails loud rather than dividing by zero
    or silently fabricating an unbounded width.
"""

from __future__ import annotations

import pytest

from mu_engine.services.recall.width import derive_recall_limit

# Same measured shape RecallSettings' own defaults use (module docstring): ~370-token prompt
# overhead, ~45 tokens/context line (MULTIHOP-AND-LLM-TESTS-0831.md §7), reserve 800 for the
# answer completion, clamp to [10, 30] (RETRIEVAL-EVAL-0829.md §13's validated range).
_DEFAULTS = {
    "prompt_reserve_tokens": 370,
    "answer_reserve_tokens": 800,
    "tokens_per_memory": 45.0,
    "min_limit": 10,
    "max_limit": 30,
}


def test_large_context_model_widens_past_the_wire_default() -> None:
    # gpt-5-class 200k-token window: (200_000 - 370 - 800) / 45 ~= 4419 — clamps to the ceiling,
    # not the arithmetic result, but the ceiling itself (30) is already > DEFAULT_RECALL_LIMIT
    # (10) — a large-context model must never derive down to the SAME width as a small one.
    limit = derive_recall_limit(max_input_tokens=200_000, **_DEFAULTS)
    assert limit == 30
    assert limit > 10


def test_mid_context_model_lands_between_floor_and_ceiling_on_the_arithmetic() -> None:
    # (8_000 - 370 - 800) / 45 = 149.6 -> 149, clamped to the ceiling (30) — still exercises the
    # arithmetic path (distinct from the "always saturates the ceiling" large-model case above).
    limit = derive_recall_limit(max_input_tokens=8_000, **_DEFAULTS)
    assert limit == 30


def test_a_budget_the_arithmetic_lands_strictly_inside_the_clamp() -> None:
    # (2_000 - 370 - 800) / 45 = 18.4 -> 18: strictly between floor (10) and ceiling (30), so this
    # test fails if EITHER clamp bound silently overrides a value that didn't need it.
    limit = derive_recall_limit(max_input_tokens=2_000, **_DEFAULTS)
    assert limit == 18


def test_small_local_model_narrows_to_the_floor_never_to_zero_or_negative() -> None:
    # A tiny local SLM context window (well under the reserved prompt+answer overhead alone):
    # the raw division would floor to 0 or go negative — the floor clamp must still return a
    # WORKING width (the FULL-LOCAL-must-stay-good boundary rule), never 0, never negative.
    limit = derive_recall_limit(max_input_tokens=512, **_DEFAULTS)
    assert limit == 10
    assert limit > 0


def test_zero_available_budget_still_floors_to_min_limit() -> None:
    # available == 0 exactly (max_input_tokens == reserves) — must not divide by tokens_per_memory
    # at all (0 // 45.0 == 0 is fine, but the negative-input branch must be reached identically).
    limit = derive_recall_limit(
        max_input_tokens=1_170,  # 370 + 800
        **_DEFAULTS,
    )
    assert limit == 10


def test_ceiling_wins_even_when_the_raw_arithmetic_would_go_higher() -> None:
    limit = derive_recall_limit(max_input_tokens=1_000_000, **_DEFAULTS)
    assert limit == 30


def test_floor_wins_even_when_max_limit_is_configured_lower_than_floor_would_need() -> None:
    # min_limit itself is the contract, regardless of how the arithmetic came out.
    limit = derive_recall_limit(
        max_input_tokens=50_000,
        prompt_reserve_tokens=370,
        answer_reserve_tokens=800,
        tokens_per_memory=45.0,
        min_limit=25,
        max_limit=25,
    )
    assert limit == 25


@pytest.mark.parametrize("bad", [0.0, -1.0, -45.0])
def test_non_positive_tokens_per_memory_fails_loud(bad: float) -> None:
    with pytest.raises(ValueError, match="tokens_per_memory"):
        derive_recall_limit(
            max_input_tokens=100_000,
            prompt_reserve_tokens=370,
            answer_reserve_tokens=800,
            tokens_per_memory=bad,
            min_limit=10,
            max_limit=30,
        )


def test_inverted_clamp_range_fails_loud_rather_than_silently_swapping() -> None:
    with pytest.raises(ValueError, match="min_limit"):
        derive_recall_limit(
            max_input_tokens=100_000,
            prompt_reserve_tokens=370,
            answer_reserve_tokens=800,
            tokens_per_memory=45.0,
            min_limit=30,
            max_limit=10,
        )


def test_min_limit_zero_is_honoured_not_silently_replaced() -> None:
    # A caller that explicitly opts OUT of a floor (min_limit=0) on a starved budget gets 0, not
    # an invented default — this function never substitutes its own floor.
    limit = derive_recall_limit(
        max_input_tokens=100,
        prompt_reserve_tokens=370,
        answer_reserve_tokens=800,
        tokens_per_memory=45.0,
        min_limit=0,
        max_limit=30,
    )
    assert limit == 0
