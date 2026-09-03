"""Unit tests for ``compare.py``'s COST delta — CLAUDE.md eval lane: "the delta in accuracy
alongside the delta in cost, because a gain that triples the bill is a different decision from a
free one."

Builds on the same synthetic-report pattern as ``test_compare_unit.py``, adding a ``usage`` block
(the shape ``usage.build_run_usage(...).model_dump(mode="json")`` produces) to each report.
"""

from __future__ import annotations

import pytest
from mu_eval.answer_quality import AnswerQualityReport, QueryResult, _stats
from mu_eval.compare import compare_runs

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


def _report(run_id: str, rows: list[QueryResult], *, cost_usd: float | None) -> AnswerQualityReport:
    usage = None if cost_usd is None else {"total_estimated_cost_usd": cost_usd}
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
        usage=usage,
    )


def test_compare_reports_the_cost_of_both_runs_and_their_delta() -> None:
    a = _report("a", [_row("q1", True)], cost_usd=3.00)
    b = _report("b", [_row("q1", True)], cost_usd=5.50)
    result = compare_runs(a, b)
    assert result.a_cost_usd == pytest.approx(3.00)
    assert result.b_cost_usd == pytest.approx(5.50)
    assert result.cost_delta_usd == pytest.approx(2.50)


def test_compare_cost_delta_is_negative_when_b_is_cheaper() -> None:
    a = _report("a", [_row("q1", True)], cost_usd=10.00)
    b = _report("b", [_row("q1", True)], cost_usd=4.00)
    result = compare_runs(a, b)
    assert result.cost_delta_usd == pytest.approx(-6.00)


def test_compare_cost_is_none_not_zero_when_a_report_has_no_usage_block() -> None:
    # Mutation this pins: defaulting a missing usage block's cost to 0.0 would make an
    # artifact from before this fix (or an unpriced model) look like a genuinely FREE run.
    a = _report("a", [_row("q1", True)], cost_usd=None)
    b = _report("b", [_row("q1", True)], cost_usd=5.00)
    result = compare_runs(a, b)
    assert result.a_cost_usd is None
    assert result.b_cost_usd == pytest.approx(5.00)
    assert result.cost_delta_usd is None  # cannot compute a delta with one side unknown


def test_compare_cost_note_is_none_when_either_side_is_unpriced() -> None:
    a = _report("a", [_row("q1", True)], cost_usd=None)
    b = _report("b", [_row("q1", True)], cost_usd=None)
    result = compare_runs(a, b)
    assert result.cost_note is None


def test_compare_cost_note_names_both_costs_and_the_delta_when_both_are_known() -> None:
    a = _report("a", [_row("q1", True)], cost_usd=3.00)
    b = _report("b", [_row("q1", True)], cost_usd=5.50)
    result = compare_runs(a, b)
    assert result.cost_note is not None
    assert "3.00" in result.cost_note
    assert "5.50" in result.cost_note
    assert "2.50" in result.cost_note


def test_compare_cost_fields_default_to_none_when_neither_report_has_usage() -> None:
    """Backwards compatibility: two pre-fix artifacts (no `usage` key at all) must still compare
    cleanly on accuracy, just with every cost field reported as unknown."""
    a = _report("a", [_row("q1", True)], cost_usd=None)
    b = _report("b", [_row("q1", False)], cost_usd=None)
    result = compare_runs(a, b)
    assert result.a_cost_usd is None
    assert result.b_cost_usd is None
    assert result.cost_delta_usd is None
    assert result.cost_note is None
    # Accuracy comparison itself is unaffected by the absence of cost data.
    assert result.delta == pytest.approx(b.overall.accuracy - a.overall.accuracy)
