"""Unit tests for the gold-answer-word-coverage metric — `runner.gold_answer_word_coverage`,
`QueryResult.gold_word_coverage`, and `CategoryStats.mean_gold_word_coverage`.

This metric exists ADDITIVELY beside `gold_ids_present`/`gold_in_context` (verbatim turn-id join),
because that join scores a write-time transformation of the gold turn's body — distillation,
coreference resolution, an LLM extractor's paraphrase — as a retrieval MISS even when the
retrieved context plainly carries the answer's own words (AD-332,
`docs/decisions/0105-ad332-llm-extraction-arm-and-the-metric-blind-spot.md`). Every assertion here
is mutation-checkable: swap the intersection for a union, drop the stopword filter, or change the
empty-gold-words default, and the matching test goes RED.
"""

from __future__ import annotations

import pytest
from mu_eval.answer_quality import CategoryStats, QueryResult, _stats
from mu_eval.runner import gold_answer_word_coverage

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------------------------------------
# gold_answer_word_coverage — pure arithmetic
# ------------------------------------------------------------------------------------------------


def test_full_coverage_when_every_gold_word_appears_in_context() -> None:
    # Gold content words (>=3 chars, stopwords stripped): {"moved", "sweden"}. Both appear in the
    # context, even though the context ALSO carries unrelated words and different phrasing.
    cov = gold_answer_word_coverage(
        "she moved to Sweden",
        "- Alex: I miss cooking dishes from my home country, Sweden, since I moved there.",
    )
    assert cov == pytest.approx(1.0)


def test_zero_coverage_when_no_gold_word_appears() -> None:
    cov = gold_answer_word_coverage(
        "the black and white design", "- Alex: I made this bowl in my pottery class."
    )
    assert cov == 0.0


def test_partial_coverage_is_the_exact_fraction() -> None:
    # Content words (>=3 chars, stopwords stripped) in "7 May 2023": {"may", "2023"} — 2 words.
    # Only one appears in the context.
    cov = gold_answer_word_coverage("7 May 2023", "- Alex: it happened in 2023 sometime.")
    assert cov == pytest.approx(0.5)


def test_is_case_insensitive() -> None:
    cov = gold_answer_word_coverage("SWEDEN", "- alex: i moved from sweden last year.")
    assert cov == 1.0


def test_stopwords_never_count_toward_coverage() -> None:
    # Every word in the gold answer besides "sweden" is a stopword; the context has none of the
    # gold's real content, only stopword overlap — must NOT count as covered.
    cov = gold_answer_word_coverage("it was the one from Sweden", "- Alex: it was there.")
    assert cov == 0.0


def test_gold_answer_with_no_scorable_words_returns_one_not_zero() -> None:
    """ "Yes." has no token >=3 chars after stopword removal (case-folds to a stopword-only
    string) — nothing to penalise, so this returns the vacuous-truth 1.0 (matching
    `ranker._tail_density`'s own convention for a pool it cannot evaluate), never an arbitrary 0.0
    that would silently drag down an aggregate mean."""
    assert gold_answer_word_coverage("Yes.", "- Alex: anything at all.") == 1.0
    assert gold_answer_word_coverage("", "- Alex: anything at all.") == 1.0


def test_empty_context_gives_zero_coverage_for_a_real_gold_answer() -> None:
    assert gold_answer_word_coverage("Stockholm, Sweden", "") == 0.0


# ------------------------------------------------------------------------------------------------
# QueryResult / CategoryStats wiring — the per-row field and its aggregate mean
# ------------------------------------------------------------------------------------------------


def _row(query_id: str, *, gold_word_coverage: float) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        category=4,
        question="q",
        gold_answer="a",
        generated_answer="g",
        context_items=5,
        verdict=True,
        gold_in_context=True,
        gold_word_coverage=gold_word_coverage,
    )


def test_default_gold_word_coverage_is_one_for_a_pre_fix_artifact() -> None:
    """An artifact JSON written before this field existed has no `gold_word_coverage` key at
    all — pydantic's default must load it as 1.0 ("nothing to penalise"), never 0.0, so an old
    artifact's rows do not silently look like retrieval failures under the new metric."""
    row = QueryResult(
        query_id="q1",
        category=1,
        question="q",
        gold_answer="a",
        generated_answer="g",
        context_items=3,
        verdict=True,
        gold_in_context=True,
    )
    assert row.gold_word_coverage == 1.0


def test_mean_gold_word_coverage_is_the_plain_mean_over_rows() -> None:
    rows = [
        _row("q1", gold_word_coverage=1.0),
        _row("q2", gold_word_coverage=0.0),
        _row("q3", gold_word_coverage=0.5),
    ]
    stats = _stats(rows)
    assert stats.mean_gold_word_coverage == pytest.approx(0.5)


def test_mean_gold_word_coverage_is_zero_not_nan_for_an_empty_row_set() -> None:
    stats = _stats([])
    assert stats.mean_gold_word_coverage == 0.0


def test_category_stats_mean_gold_word_coverage_defaults_to_zero_for_a_pre_fix_artifact() -> None:
    """Unlike the per-row field (defaults 1.0, "nothing to penalise"), the AGGREGATE default is
    0.0 — an artifact predating this fix has zero REAL observations behind the mean, and 0.0 is
    the honest "no data" value a reader should not mistake for a real, fully-covered arm. Readers
    must check `n` before trusting this field, exactly as they already must for every other
    default-0 counter on this model (`answer_budget_retried` etc.)."""
    stats = CategoryStats(n=0, correct=0, wrong=0, unparseable=0)
    assert stats.mean_gold_word_coverage == 0.0


def test_stats_wires_gold_word_coverage_through_not_just_gold_in_context() -> None:
    """Regression guard for the exact defect this field would have if `_stats` computed
    `gold_retrieved` (the verbatim-join metric) but forgot to also thread `gold_word_coverage`
    through — the two metrics must never silently diverge into "one wired, one not"."""
    rows = [_row("q1", gold_word_coverage=0.25)]
    stats = _stats(rows)
    assert stats.mean_gold_word_coverage == pytest.approx(0.25)
    assert stats.gold_retrieved == 1  # gold_in_context=True in the fixture, sanity check
