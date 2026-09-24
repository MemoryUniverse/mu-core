"""Unit tests: ``answer_quality.compute_pairwise_noise`` — the NOISE FLOOR measurement
(TRUSTWORTHY-MEASUREMENT-0924).

The brief this exists for: "measure the noise floor properly — run the SAME configuration twice
and report the run-to-run churn in both directions... the harness should say so itself." Before
this fix, `--num-runs >= 2` printed only `overall.accuracy`'s aggregate spread — two runs could
tie on accuracy while churning a dozen rows' parseability in opposite directions, invisible to a
spread-of-one-scalar. `compute_pairwise_noise` runs the SAME paired `compare.compare_runs` used
for a real A/B between each pair of CONSECUTIVE repeats of one unchanged config.
"""

from __future__ import annotations

import pytest
from mu_eval.answer_quality import AnswerQualityReport, QueryResult, _stats, compute_pairwise_noise

pytestmark = pytest.mark.unit


def _row(query_id: str, verdict: bool | None) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        category=1,
        question="q",
        gold_answer="g",
        generated_answer="a",
        context_items=3,
        verdict=verdict,
        gold_in_context=True,
    )


def _report(run_id: str, rows: list[QueryResult]) -> AnswerQualityReport:
    return AnswerQualityReport(
        run_id=run_id,
        dataset="locomo10",
        samples=1,
        answer_model="gpt-5",
        judge_model="gpt-5",
        recall_limit=10,
        importance=0.9,
        queries_scored=len(rows),
        skipped_adversarial=0,
        skipped_no_gold_in_corpus=0,
        overall=_stats(rows),
        by_category={},
        rows=rows,
    )


def test_empty_with_fewer_than_two_reports() -> None:
    assert compute_pairwise_noise([]) == []
    assert compute_pairwise_noise([_report("r0", [_row("q1", True)])]) == []


def test_three_repeats_produce_two_consecutive_pairs_not_three_choose_two() -> None:
    """Consecutive pairing (r0-vs-r1, r1-vs-r2), not every pair — the point is run-to-run drift
    across the sequence of repeats actually performed, not a combinatorial blow-up."""
    reports = [
        _report("r0", [_row("q1", True), _row("q2", False)]),
        _report("r1", [_row("q1", True), _row("q2", True)]),
        _report("r2", [_row("q1", False), _row("q2", True)]),
    ]
    result = compute_pairwise_noise(reports)
    assert len(result) == 2
    assert result[0]["run_a"] == "r0"
    assert result[0]["run_b"] == "r1"
    assert result[1]["run_a"] == "r1"
    assert result[1]["run_b"] == "r2"


def test_reports_the_real_churn_between_two_identical_config_runs() -> None:
    """The exact shape the brief asks for: two runs of ONE unchanged config still churn rows —
    this is the noise floor, and it must show up here even though "nothing changed"."""
    r0 = _report("r0", [_row("q1", True), _row("q2", None), _row("q3", False)])
    r1 = _report("r1", [_row("q1", True), _row("q2", True), _row("q3", None)])
    result = compute_pairwise_noise([r0, r1])
    assert len(result) == 1
    entry = result[0]
    assert entry["recovered_from_unparseable"] == ["q2"]
    assert entry["became_unparseable"] == ["q3"]
    assert entry["fixed"] == []
    assert entry["regressed"] == []


def test_skips_a_pair_missing_rows_rather_than_raising() -> None:
    """A run written with `keep_rows=False` (aggregate-only) cannot be paired — skipped, not a
    crash that would take down the whole repeated-run report over one missing role's row data."""
    with_rows = _report("r0", [_row("q1", True)])
    no_rows = AnswerQualityReport(
        run_id="r1",
        dataset="locomo10",
        samples=1,
        answer_model="gpt-5",
        judge_model="gpt-5",
        recall_limit=10,
        importance=0.9,
        queries_scored=1,
        skipped_adversarial=0,
        skipped_no_gold_in_corpus=0,
        overall=_stats([_row("q1", True)]),
        by_category={},
        rows=[],  # keep_rows=False shape
    )
    assert compute_pairwise_noise([with_rows, no_rows]) == []
