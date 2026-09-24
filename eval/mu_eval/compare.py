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

    #: VERIFY 2026-09-24. `significant_at_p05` above is McNemar's answer to ONE question — "is the
    #: fixed/regressed split larger than chance" — computed over the rows that were parseable in
    #: BOTH runs. It says nothing about whether that parseable subset is itself stable, and on this
    #: harness it demonstrably is not: TRUSTWORTHY-MEASUREMENT-0924 measured 9 rows of parseability
    #: churn against 7 of verdict churn between two runs of the SAME configuration. Until that
    #: number comes down, a delta drawn from a smaller number of rows than the judge infrastructure
    #: moves on its own is not a finding, and a MACHINE reader had no way to tell: the caveat this
    #: pass's predecessor added lives only inside the human-readable `verdict` string, so any
    #: consumer that branches on `significant_at_p05` (or reads `delta` straight out of the
    #: artifact) still sees a clean "significant" on a noise-limited comparison. This flag is that
    #: missing machine-readable refusal — True when parseability churn is at least as large as the
    #: verdict churn `delta` was computed from. Callers must treat `significant_at_p05 and not
    #: noise_limited` as the reportable condition, never `significant_at_p05` alone.
    noise_limited: bool = False

    verdict: str  # a human-readable one-line summary of the above

    # COST DELTA (CLAUDE.md eval lane: "the delta in accuracy alongside the delta in cost, because
    # a gain that triples the bill is a different decision from a free one"). `None`, not 0.0, when
    # either report carries no `usage` block (an artifact from before this fix, or a run whose
    # model had no matching rate card entry) — a missing cost must never read as a FREE one.
    a_cost_usd: float | None = None
    b_cost_usd: float | None = None
    cost_delta_usd: float | None = None  # b_cost_usd - a_cost_usd
    cost_note: str | None = None  # None when either side's cost is unknown


def _total_cost(report: AnswerQualityReport) -> float | None:
    """The run's own ``total_estimated_cost_usd`` (``usage.build_run_usage``'s output, merged onto
    the artifact the same way ``provenance`` is), or ``None`` when the artifact carries no usage
    block, or a usage block whose total was itself ``None`` (an unpriced model — see
    ``usage.build_run_usage``'s own docstring for why that is never silently ``$0.00``)."""
    if not report.usage:
        return None
    value = report.usage.get("total_estimated_cost_usd")
    return float(value) if isinstance(value, int | float) else None


def _cost_note(a_cost: float | None, b_cost: float | None) -> str | None:
    if a_cost is None or b_cost is None:
        return None
    delta = b_cost - a_cost
    return (
        f"COST: A=${a_cost:.2f}  B=${b_cost:.2f}  (delta ${delta:+.2f}) — weigh this against the "
        "accuracy delta above; a gain that costs more is a different decision from a free one."
    )


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
    noise_limited = _is_noise_limited(
        fixed=len(fixed),
        regressed=len(regressed),
        became_unparseable=len(became_unparseable),
        recovered_from_unparseable=len(recovered_from_unparseable),
    )

    verdict = _summarize(
        delta=delta,
        p_value=p_value,
        significant=significant,
        fixed=len(fixed),
        regressed=len(regressed),
        became_unparseable=len(became_unparseable),
        recovered_from_unparseable=len(recovered_from_unparseable),
    )

    a_cost = _total_cost(a)
    b_cost = _total_cost(b)
    cost_delta = (b_cost - a_cost) if (a_cost is not None and b_cost is not None) else None

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
        noise_limited=noise_limited,
        verdict=verdict,
        a_cost_usd=a_cost,
        b_cost_usd=b_cost,
        cost_delta_usd=cost_delta,
        cost_note=_cost_note(a_cost, b_cost),
    )


def _is_noise_limited(
    *,
    fixed: int,
    regressed: int,
    became_unparseable: int,
    recovered_from_unparseable: int,
) -> bool:
    """Is judge-infrastructure churn at least as large as the effect being measured?

    ONE definition, read by both :class:`CompareResult.noise_limited` (the machine-readable
    refusal) and :func:`_summarize`'s human-readable CAVEAT, so the two can never disagree about
    the same comparison — they were two separate expressions of the same rule before this, which
    is how a caveat ends up printed next to a `significant_at_p05: true`."""
    parseability_churn = became_unparseable + recovered_from_unparseable
    verdict_churn = fixed + regressed
    return parseability_churn > 0 and parseability_churn >= max(verdict_churn, 1)


def _summarize(
    *,
    delta: float,
    p_value: float,
    significant: bool,
    fixed: int,
    regressed: int,
    became_unparseable: int,
    recovered_from_unparseable: int,
) -> str:
    direction = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
    parseability_churn = became_unparseable + recovered_from_unparseable
    verdict_churn = fixed + regressed
    # JUDGE-NOISE CAVEAT (TRUSTWORTHY-MEASUREMENT-0924). This is the exact gap a real paired run
    # exposed: two arms differing ONLY in retrieval moved accuracy by +0.03pt (4 fixed, 1
    # regressed) while 13 rows became unparseable and 7 recovered — parseability churn an order of
    # magnitude larger than the verdict churn the delta was computed from. McNemar's own p-value
    # already answers "is the fixed/regressed split real"; it says NOTHING about whether the
    # PARSEABLE SUBSET those rows were drawn from is itself stable, and a reader comparing two
    # printed accuracy numbers has no way to notice that unless the tool says so. Always computed
    # (never opt-in), because a reader should not have to remember to ask for it.
    noise_caveat = (
        (
            f" CAVEAT: judge-parseability churn ({became_unparseable} became unparseable, "
            f"{recovered_from_unparseable} recovered = {parseability_churn} rows) is >= the "
            f"verdict churn this delta is computed from ({fixed} fixed + {regressed} regressed = "
            f"{verdict_churn} rows) — judge-infra noise is at least as large as the measured "
            "effect. Do not report this delta as a finding until parseability churn is reduced "
            "(see the budget-exhaustion retry in answer_quality._complete_with_retry) or shown, "
            "by running the SAME configuration twice, to be smaller than this."
        )
        if _is_noise_limited(
            fixed=fixed,
            regressed=regressed,
            became_unparseable=became_unparseable,
            recovered_from_unparseable=recovered_from_unparseable,
        )
        else ""
    )
    if fixed == 0 and regressed == 0:
        return (
            "No row flipped verdict in either direction — the aggregate delta, if any, "
            "comes only from unparseable-count changes." + noise_caveat
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
        f"{fixed} row(s) fixed, {regressed} row(s) regressed — {significance}." + noise_caveat
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
