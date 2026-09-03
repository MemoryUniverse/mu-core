"""Paired per-row comparison of two ``AnswerQualityReport`` artifacts — item 4.

HOW-THEY-MEASURE-0901.md's own opening claim: a multi-hop gain of +2.7pt was reported by
subtracting two AGGREGATES. It did not reproduce; it INVERTED (the same configuration measured
twice gave 14.4% and 17.3% — a 2.9pt spread, larger than the delta being reported). Subtracting two
means throws away exactly the information that would have caught this: WHICH rows moved, and in
which direction. A paired test (McNemar's, for two binary-outcome measurements of the same items)
is roughly twice as sensitive as a two-sample comparison of means for the same effect size, because
it removes each row's own base-rate variance from the comparison instead of averaging it away.

This module never compares two aggregate numbers directly. It always joins on ``query_id`` first,
counts the four possible transitions for each row that has a real verdict in both runs
(correct->correct, correct->wrong, wrong->correct, wrong->wrong), and names the query ids that
flipped in each direction — so "did this change anything" has a per-row answer, not just a
before/after mean.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from mu_eval.answer_quality import AnswerQualityReport, QueryResult

__all__ = ["CompareResult", "compare_runs", "mcnemar_p_value"]


def mcnemar_p_value(n_a_correct_b_wrong: int, n_a_wrong_b_correct: int) -> float:
    """Two-sided McNemar's test p-value (continuity-corrected), for the two DISCORDANT cells of a
    paired 2x2 table: rows where A and B disagreed. Concordant rows (both correct or both wrong)
    carry no information about a DIRECTIONAL change and are correctly excluded from the statistic
    itself (McNemar's own definition) — they are still counted and reported elsewhere in
    ``CompareResult``, never silently dropped from the artifact.

    Closed-form, no scipy dependency: a chi-square distribution with 1 degree of freedom has
    survival function ``erfc(sqrt(x/2))`` exactly (``P(chi2_1 <= x) = erf(sqrt(x/2))``), so the
    p-value is computed from the standard library's ``math.erfc`` alone.
    """
    n10, n01 = n_a_correct_b_wrong, n_a_wrong_b_correct
    total = n10 + n01
    if total == 0:
        return 1.0  # no discordant pairs at all -> no evidence of any directional change
    statistic = (abs(n10 - n01) - 1) ** 2 / total
    return math.erfc(math.sqrt(statistic / 2))


class CompareResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_a: str  # run_id of A
    run_b: str  # run_id of B

    # Denominator honesty (item 3, carried into item 4): every row is accounted for in exactly
    # ONE of these buckets — common_rows + only_in_a + only_in_b must equal the union's size.
    common_rows: int  # query_id present in both runs
    only_in_a: int  # present in A only (never silently dropped from the comparison)
    only_in_b: int  # present in B only

    a_accuracy: float  # A's parseable-subset accuracy, over ITS OWN full row set (not the join)
    b_accuracy: float
    a_accuracy_all_scoreable: float
    b_accuracy_all_scoreable: float
    delta: float  # b_accuracy - a_accuracy, parseable-subset

    # Per-row transitions, over the COMMON, BOTH-PARSEABLE subset only (verdict is bool in both
    # runs — McNemar needs a binary outcome on both sides of the pair).
    fixed: list[str] = Field(default_factory=list)  # wrong(A) -> correct(B)
    regressed: list[str] = Field(default_factory=list)  # correct(A) -> wrong(B)
    unchanged_correct: int = 0
    unchanged_wrong: int = 0

    # Rows that became/stopped being unparseable — counted and named, never folded into
    # fixed/regressed (an unparseable judgement is a judge-infra event, not a verdict flip).
    became_unparseable: list[str] = Field(default_factory=list)  # parsed(A) -> None(B)
    recovered_from_unparseable: list[str] = Field(default_factory=list)  # None(A) -> parsed(B)
    both_unparseable: int = 0

    mcnemar_n_a_correct_b_wrong: int  # = len(regressed)
    mcnemar_n_a_wrong_b_correct: int  # = len(fixed)
    mcnemar_p_value: float
    significant_at_p05: bool

    verdict: str  # a human-readable one-line summary of the above


def _rows_by_id(report: AnswerQualityReport) -> dict[str, QueryResult]:
    if not report.rows:
        raise ValueError(
            f"run {report.run_id!r} carries no per-row data (--no-rows was likely used when it "
            "was produced) — compare needs `rows`, an aggregate-only artifact cannot be paired"
        )
    return {r.query_id: r for r in report.rows}


def compare_runs(a: AnswerQualityReport, b: AnswerQualityReport) -> CompareResult:
    rows_a = _rows_by_id(a)
    rows_b = _rows_by_id(b)
    ids_a, ids_b = set(rows_a), set(rows_b)
    common = sorted(ids_a & ids_b)

    fixed: list[str] = []
    regressed: list[str] = []
    unchanged_correct = 0
    unchanged_wrong = 0
    became_unparseable: list[str] = []
    recovered_from_unparseable: list[str] = []
    both_unparseable = 0

    for qid in common:
        va, vb = rows_a[qid].verdict, rows_b[qid].verdict
        if va is None and vb is None:
            both_unparseable += 1
        elif va is not None and vb is None:
            became_unparseable.append(qid)
        elif va is None and vb is not None:
            recovered_from_unparseable.append(qid)
        elif va is False and vb is True:
            fixed.append(qid)
        elif va is True and vb is False:
            regressed.append(qid)
        elif va is True and vb is True:
            unchanged_correct += 1
        else:  # va is False and vb is False
            unchanged_wrong += 1

    n_a_correct_b_wrong = len(regressed)
    n_a_wrong_b_correct = len(fixed)
    p_value = mcnemar_p_value(n_a_correct_b_wrong, n_a_wrong_b_correct)

    delta = b.overall.accuracy - a.overall.accuracy
    significant = p_value < 0.05

    verdict = _summarize(
        delta=delta,
        p_value=p_value,
        significant=significant,
        fixed=len(fixed),
        regressed=len(regressed),
    )

    return CompareResult(
        run_a=a.run_id,
        run_b=b.run_id,
        common_rows=len(common),
        only_in_a=len(ids_a - ids_b),
        only_in_b=len(ids_b - ids_a),
        a_accuracy=a.overall.accuracy,
        b_accuracy=b.overall.accuracy,
        a_accuracy_all_scoreable=a.overall.accuracy_all_scoreable,
        b_accuracy_all_scoreable=b.overall.accuracy_all_scoreable,
        delta=delta,
        fixed=fixed,
        regressed=regressed,
        unchanged_correct=unchanged_correct,
        unchanged_wrong=unchanged_wrong,
        became_unparseable=became_unparseable,
        recovered_from_unparseable=recovered_from_unparseable,
        both_unparseable=both_unparseable,
        mcnemar_n_a_correct_b_wrong=n_a_correct_b_wrong,
        mcnemar_n_a_wrong_b_correct=n_a_wrong_b_correct,
        mcnemar_p_value=p_value,
        significant_at_p05=significant,
        verdict=verdict,
    )


def _summarize(
    *, delta: float, p_value: float, significant: bool, fixed: int, regressed: int
) -> str:
    direction = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
    if fixed == 0 and regressed == 0:
        return (
            "No row flipped verdict in either direction — the aggregate delta, if any, "
            "comes only from unparseable-count changes."
        )
    significance = (
        f"significant at p<0.05 (McNemar p={p_value:.4f})"
        if significant
        else (
            f"NOT significant at p<0.05 (McNemar p={p_value:.4f}) — "
            "do not report this delta as a finding"
        )
    )
    return (
        f"B {direction} vs A by {delta * 100:+.2f}pt (parseable-subset accuracy): "
        f"{fixed} row(s) fixed, {regressed} row(s) regressed — {significance}."
    )


def spread_note(delta: float, spread: float) -> str | None:
    """If a caller has a repeat spread (``repeats.RepeatSummary.spread``) for one of the two arms,
    fold it into the same "say so in the tool's own output" discipline item 5 asks for: a delta
    this comparison reports as significant can STILL be smaller than known run-to-run noise, and a
    reader should never have to notice that by hand. Returns ``None`` when the delta already
    exceeds the spread (nothing extra to say)."""
    if abs(delta) <= spread:
        return (
            f"NOTE: |delta|={abs(delta):.4f} does not exceed the measured run-to-run spread "
            f"({spread:.4f}) — this delta cannot be distinguished from repeat-to-repeat noise "
            "using the available repeats, regardless of what McNemar says about row flips."
        )
    return None
