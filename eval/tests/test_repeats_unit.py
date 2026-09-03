"""Unit tests for ``repeats.py`` — item 5 (variance made visible).

The motivating defect (module docstring): the SAME configuration measured twice gave 14.4% and
17.3% (spread 2.9pt) on a category where a +2.7pt delta had already been reported as a finding.
These tests pin the arithmetic that turns "N runs" into "mean, spread, and whether a delta is
inside it" — each independently mutation-checkable (swap ``max-min`` for ``stdev``, swap ``<=``
for ``<`` in ``is_within_spread``, etc.).
"""

from __future__ import annotations

import pytest
from mu_eval.repeats import run_n_times, summarize_repeats

pytestmark = pytest.mark.unit


def test_summarize_repeats_computes_mean_min_max_and_spread() -> None:
    summary = summarize_repeats(
        metric_name="multi_hop.accuracy", values=[0.144, 0.173], run_ids=["r0", "r1"]
    )
    assert summary.num_runs == 2
    assert summary.mean == pytest.approx((0.144 + 0.173) / 2)
    assert summary.minimum == pytest.approx(0.144)
    assert summary.maximum == pytest.approx(0.173)
    assert summary.spread == pytest.approx(0.173 - 0.144)
    assert summary.stdev > 0.0


def test_summarize_repeats_spread_exceeds_the_falsely_reported_delta() -> None:
    """The exact motivating numbers: a delta of +0.027 was reported as a finding between two runs
    that, measured honestly, spread by 0.029 — i.e. the delta was SMALLER than the noise floor."""
    summary = summarize_repeats(
        metric_name="multi_hop.accuracy", values=[0.144, 0.173], run_ids=["r0", "r1"]
    )
    reported_delta = 0.027
    assert summary.spread > reported_delta
    assert summary.is_within_spread(reported_delta) is True


def test_is_within_spread_is_false_for_a_delta_that_genuinely_exceeds_the_spread() -> None:
    summary = summarize_repeats(metric_name="m", values=[0.50, 0.51], run_ids=["r0", "r1"])
    assert summary.is_within_spread(0.5) is False


def test_summarize_repeats_single_run_has_zero_stdev_and_zero_spread() -> None:
    summary = summarize_repeats(metric_name="m", values=[0.42], run_ids=["r0"])
    assert summary.num_runs == 1
    assert summary.stdev == 0.0
    assert summary.spread == 0.0
    assert summary.mean == pytest.approx(0.42)


def test_summarize_repeats_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        summarize_repeats(metric_name="m", values=[], run_ids=[])


def test_summarize_repeats_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        summarize_repeats(metric_name="m", values=[0.1, 0.2], run_ids=["r0"])


async def test_run_n_times_calls_run_once_with_a_distinct_run_id_per_repeat() -> None:
    seen: list[str] = []

    async def _once(run_id: str) -> str:
        seen.append(run_id)
        return run_id

    results = await run_n_times(num_runs=3, run_id_prefix="base", run_once=_once)
    assert seen == ["base-r0", "base-r1", "base-r2"]
    assert results == seen
    assert len(set(seen)) == 3, "repeats must not silently reuse the same run id/partition"


async def test_run_n_times_rejects_a_non_positive_count() -> None:
    async def _once(run_id: str) -> str:  # pragma: no cover - never reached
        return run_id

    with pytest.raises(ValueError, match="num_runs must be >= 1"):
        await run_n_times(num_runs=0, run_id_prefix="base", run_once=_once)
