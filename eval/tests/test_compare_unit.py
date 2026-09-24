"""Unit tests for ``compare.py`` — item 4 (comparison as a first-class, PAIRED operation).

The motivating defect (module docstring): a +2.7pt multi-hop "gain" was computed by subtracting
two aggregate accuracy numbers. It never reproduced — the same configuration measured twice gave
14.4% and 17.3%. A paired, per-row comparison would have shown WHICH rows moved and in which
direction, rather than one signed number. These tests build small synthetic
``AnswerQualityReport`` pairs and pin the row-flip bookkeeping and the McNemar arithmetic —
each independently mutation-checkable.
"""

from __future__ import annotations

import math

import pytest
from mu_eval.answer_quality import AnswerQualityReport, QueryResult, _stats
from mu_eval.compare import compare_runs, mcnemar_p_value

pytestmark = pytest.mark.unit


def _row(query_id: str, verdict: bool | None, *, gold_in_context: bool = True) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        category=1,
        question=f"question {query_id}",
        gold_answer="gold",
        generated_answer="generated",
        context_items=3,
        verdict=verdict,
        gold_in_context=gold_in_context,
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


def test_mcnemar_p_value_is_one_when_discordant_pairs_are_perfectly_balanced() -> None:
    # n10 == n01 -> statistic collapses to (|0|-1)^2/total, still symmetric; the KEY property
    # pinned here is monotonicity (next test), and the degenerate all-zero case below.
    assert mcnemar_p_value(0, 0) == 1.0


def test_mcnemar_p_value_decreases_as_the_discordance_becomes_more_one_sided() -> None:
    balanced = mcnemar_p_value(10, 10)
    lopsided = mcnemar_p_value(19, 1)
    very_lopsided = mcnemar_p_value(20, 0)
    assert balanced > lopsided > very_lopsided
    assert 0.0 <= very_lopsided <= balanced <= 1.0


def test_mcnemar_p_value_is_the_documented_closed_form() -> None:
    n10, n01 = 15, 3
    stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    expected = math.erfc(math.sqrt(stat / 2))
    assert mcnemar_p_value(n10, n01) == pytest.approx(expected)


def test_compare_runs_counts_fixed_and_regressed_rows_by_id() -> None:
    a = _report(
        "run-a",
        [
            _row("q1", True),  # correct in A
            _row("q2", False),  # wrong in A
            _row("q3", True),
            _row("q4", False),
        ],
    )
    b = _report(
        "run-b",
        [
            _row("q1", False),  # A correct -> B wrong: REGRESSED
            _row("q2", True),  # A wrong -> B correct: FIXED
            _row("q3", True),  # unchanged correct
            _row("q4", False),  # unchanged wrong
        ],
    )
    result = compare_runs(a, b)
    assert result.regressed == ["q1"]
    assert result.fixed == ["q2"]
    assert result.unchanged_correct == 1
    assert result.unchanged_wrong == 1
    assert result.common_rows == 4
    assert result.only_in_a == 0
    assert result.only_in_b == 0
    assert result.mcnemar_n_a_correct_b_wrong == 1
    assert result.mcnemar_n_a_wrong_b_correct == 1


def test_compare_runs_never_silently_drops_rows_present_in_only_one_run() -> None:
    a = _report("run-a", [_row("q1", True), _row("only-a", True)])
    b = _report("run-b", [_row("q1", True), _row("only-b", False)])
    result = compare_runs(a, b)
    assert result.common_rows == 1
    assert result.only_in_a == 1
    assert result.only_in_b == 1


def test_compare_runs_tracks_unparseable_transitions_separately_from_verdict_flips() -> None:
    a = _report("run-a", [_row("q1", True), _row("q2", None), _row("q3", None)])
    b = _report("run-b", [_row("q1", None), _row("q2", True), _row("q3", None)])
    result = compare_runs(a, b)
    # q1: parsed(A) -> unparseable(B); q2: unparseable(A) -> parsed(B); q3: both unparseable
    assert result.became_unparseable == ["q1"]
    assert result.recovered_from_unparseable == ["q2"]
    assert result.both_unparseable == 1
    # None of these three count as a fixed/regressed verdict flip.
    assert result.fixed == []
    assert result.regressed == []


def test_compare_runs_delta_matches_the_reports_own_overall_accuracy() -> None:
    a = _report("run-a", [_row("q1", True), _row("q2", False)])
    b = _report("run-b", [_row("q1", True), _row("q2", True)])
    result = compare_runs(a, b)
    assert result.a_accuracy == pytest.approx(a.overall.accuracy)
    assert result.b_accuracy == pytest.approx(b.overall.accuracy)
    assert result.delta == pytest.approx(b.overall.accuracy - a.overall.accuracy)


def test_compare_runs_raises_a_named_error_when_a_report_has_no_rows() -> None:
    a = _report("run-a", [])
    b = _report("run-b", [_row("q1", True)])
    with pytest.raises(ValueError, match="run-a"):
        compare_runs(a, b)


def test_significant_at_p05_flag_matches_the_computed_p_value() -> None:
    rows_a = [_row(f"q{i}", True) for i in range(20)]
    rows_b = [_row(f"q{i}", False) for i in range(20)]  # every single row flips to wrong
    result = compare_runs(_report("run-a", rows_a), _report("run-b", rows_b))
    assert result.mcnemar_p_value < 0.05
    assert result.significant_at_p05 is True


# ------------------------------------------------------------ TRUSTWORTHY-MEASUREMENT-0924: the
# judge-noise caveat. Real evidence (AD-232's paired 150-row run): +0.03pt accuracy from 4 fixed /
# 1 regressed, while 13 rows became unparseable and 7 recovered — parseability churn an order of
# magnitude larger than the verdict churn the delta was computed from. `compare_runs`'s own
# verdict string must say so, unprompted, whenever that shape recurs.


def test_verdict_warns_when_parseability_churn_dwarfs_the_verdict_churn() -> None:
    # Reproduces AD-232's own shape at small scale: 1 fixed / 0 regressed (verdict churn = 1),
    # but 3 became unparseable / 2 recovered (parseability churn = 5) — churn far exceeds signal.
    a_rows = (
        [_row("fixed", False)]
        + [_row(f"bu{i}", True) for i in range(3)]  # -> None in B: became_unparseable
        + [_row(f"rec{i}", None) for i in range(2)]  # -> parsed in B: recovered_from_unparseable
    )
    b_rows = (
        [_row("fixed", True)]
        + [_row(f"bu{i}", None) for i in range(3)]
        + [_row(f"rec{i}", True) for i in range(2)]
    )
    result = compare_runs(_report("run-a", a_rows), _report("run-b", b_rows))
    assert len(result.fixed) == 1
    assert len(result.regressed) == 0
    assert len(result.became_unparseable) == 3
    assert len(result.recovered_from_unparseable) == 2
    assert "CAVEAT" in result.verdict
    assert "judge-parseability churn" in result.verdict
    assert "do not report this delta as a finding" in result.verdict.lower()


def test_verdict_carries_no_caveat_when_parseability_is_stable() -> None:
    """The common, healthy case — no unparseable churn at all — must not carry the caveat text,
    so the caveat's presence is actually informative rather than boilerplate on every run."""
    a_rows = [_row("q1", False), _row("q2", True)]
    b_rows = [_row("q1", True), _row("q2", True)]
    result = compare_runs(_report("run-a", a_rows), _report("run-b", b_rows))
    assert len(result.fixed) == 1
    assert result.became_unparseable == []
    assert result.recovered_from_unparseable == []
    assert "CAVEAT" not in result.verdict


def test_verdict_caveat_fires_even_with_zero_verdict_churn() -> None:
    """The exact AD-232 edge: fixed=regressed=0 (the pre-existing 'no row flipped' branch) but
    unparseable churn is non-zero — must still warn, not silently fall into the "no row flipped"
    sentence with nothing said about the churn that DID happen."""
    a_rows = [_row("bu", True)]
    b_rows = [_row("bu", None)]
    result = compare_runs(_report("run-a", a_rows), _report("run-b", b_rows))
    assert result.fixed == []
    assert result.regressed == []
    assert result.became_unparseable == ["bu"]
    assert "CAVEAT" in result.verdict


def test_a_noise_limited_comparison_is_flagged_machine_readably_not_only_in_prose() -> None:
    """VERIFY 2026-09-24. The predecessor pass added a CAVEAT to `CompareResult.verdict` when
    judge-parseability churn is at least as large as the verdict churn a delta is computed from.
    That caveat is prose: `__main__._cmd_compare`'s own docstring tells a caller scripting a gate
    to read `significant_at_p05`/`delta` out of the artifact, and both of those stay clean on a
    comparison the harness has just declared unreportable. `noise_limited` is the machine-readable
    half — this pins BOTH halves on the same result, so a future refactor cannot fix one and leave
    the other lying.

    The shape below is the one TRUSTWORTHY-MEASUREMENT-0924 measured live (parseability churn >=
    verdict churn) with the verdict churn made significant on purpose, so the test also proves the
    two flags are INDEPENDENT: McNemar says "significant", the noise gate says "do not report".
    """
    rows_a = (
        [_row(f"f{i}", False) for i in range(6)]  # 6 rows A-wrong -> B-correct  (fixed)
        + [_row(f"u{i}", True) for i in range(4)]  # 4 rows A-parseable -> B-unparseable
        + [_row(f"r{i}", None) for i in range(4)]  # 4 rows A-unparseable -> B-parseable
    )
    rows_b = (
        [_row(f"f{i}", True) for i in range(6)]
        + [_row(f"u{i}", None) for i in range(4)]
        + [_row(f"r{i}", True) for i in range(4)]
    )
    result = compare_runs(_report("A", rows_a), _report("B", rows_b))

    assert len(result.fixed) == 6 and len(result.regressed) == 0
    assert len(result.became_unparseable) == 4
    assert len(result.recovered_from_unparseable) == 4
    assert result.significant_at_p05 is True, (
        "fixture broken: this test needs McNemar to say 'significant' so that it can prove the "
        "noise gate is a SEPARATE refusal, not a restatement of the p-value"
    )
    assert result.noise_limited is True, (
        "8 rows of parseability churn against 6 of verdict churn is the exact shape "
        "TRUSTWORTHY-MEASUREMENT-0924 measured, and it must set the machine-readable flag — not "
        "only the human-readable verdict string"
    )
    assert "CAVEAT" in result.verdict, "the prose caveat and the flag must agree on one comparison"


def test_a_clean_comparison_is_not_flagged_noise_limited() -> None:
    """The negative control the flag needs: churn well below the effect leaves it False, so
    `noise_limited` cannot be an always-True refusal that silences every real finding."""
    rows_a = [_row(f"f{i}", False) for i in range(20)] + [_row("u0", True)]
    rows_b = [_row(f"f{i}", True) for i in range(20)] + [_row("u0", None)]
    result = compare_runs(_report("A", rows_a), _report("B", rows_b))

    assert len(result.fixed) == 20
    assert len(result.became_unparseable) == 1
    assert result.noise_limited is False
    assert "CAVEAT" not in result.verdict
