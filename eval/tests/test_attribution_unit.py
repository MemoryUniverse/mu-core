"""Unit tests for item 1 (per-row retrieval attribution) and item 3 (honest denominators).

``gold_ids_present`` (``runner.py``) and ``CategoryStats``/``_stats`` (``answer_quality.py``) are
what turn "we are retrieval-capped" from an inference drawn by dividing two aggregates into a
per-row measurement: a WRONG verdict splits into a RETRIEVAL failure (gold never in context) and a
GENERATION failure (gold was right there and the model got it wrong anyway). Every assertion here
is mutation-checkable — swap ``wrong_retrieved``/``wrong_not_retrieved`` for each other, or swap
``accuracy``'s denominator for ``accuracy_all_scoreable``'s, and the matching test goes RED.
"""

from __future__ import annotations

import pytest
from mu_eval.answer_quality import CategoryStats, QueryResult, _stats
from mu_eval.corpus import TurnIndex
from mu_eval.locomo import Turn
from mu_eval.runner import gold_ids_present

pytestmark = pytest.mark.unit


class _StubItem:
    """Minimal stand-in for ``RecallItemView`` — ``gold_ids_present`` only reads ``.content``."""

    def __init__(self, content: str) -> None:
        self.content = content


def _index_with(*turns: tuple[str, str]) -> TurnIndex:
    """``turns`` = (dia_id, text) pairs, ingested under a fixed speaker so ``ingest_text`` is
    deterministic and the caller can pass the SAME string back as a recalled item's ``.content``.
    """
    index = TurnIndex()
    for dia_id, text in turns:
        index.add(
            Turn(dia_id=dia_id, session_index=1, session_date="1 Jan 2026", speaker="A", text=text)
        )
    return index


# --------------------------------------------------------------------- gold_ids_present


def test_gold_ids_present_true_when_a_retrieved_item_resolves_to_a_gold_turn() -> None:
    index = _index_with(("D1:1", "I adopted a greyhound named Pepper."))
    items = [_StubItem("A: I adopted a greyhound named Pepper.")]
    assert gold_ids_present(items, index, {"D1:1"}) is True


def test_gold_ids_present_false_when_no_retrieved_item_resolves_to_gold() -> None:
    index = _index_with(
        ("D1:1", "I adopted a greyhound named Pepper."),
        ("D1:2", "I signed up for a half marathon."),
    )
    items = [_StubItem("A: I signed up for a half marathon.")]  # resolves to D1:2, not gold D1:1
    assert gold_ids_present(items, index, {"D1:1"}) is False


def test_gold_ids_present_false_on_an_empty_retrieved_list() -> None:
    index = _index_with(("D1:1", "some fact"))
    assert gold_ids_present([], index, {"D1:1"}) is False


def test_gold_ids_present_true_if_any_one_of_several_items_matches() -> None:
    index = _index_with(("D1:1", "fact one"), ("D1:2", "fact two"), ("D1:3", "fact three"))
    items = [_StubItem("A: fact two"), _StubItem("A: fact three"), _StubItem("A: unrelated noise")]
    assert gold_ids_present(items, index, {"D1:2"}) is True


def test_gold_ids_present_ignores_a_body_that_is_not_in_the_corpus_index_at_all() -> None:
    """An LTM-distilled paraphrase (not a direct turn body) must not be mistaken for a gold hit."""
    index = _index_with(("D1:1", "fact one"))
    items = [_StubItem("A: a paraphrase the harness never wrote verbatim")]
    assert gold_ids_present(items, index, {"D1:1"}) is False


# ---------------------------------------------------------------------------------- CategoryStats


def _row(query_id: str, verdict: bool | None, *, gold_in_context: bool) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        category=4,
        question="q",
        gold_answer="a",
        generated_answer="g",
        context_items=5,
        verdict=verdict,
        gold_in_context=gold_in_context,
    )


def test_stats_splits_wrong_verdicts_by_retrieval_attribution() -> None:
    rows = [
        _row("q1", False, gold_in_context=True),  # generation failure
        _row("q2", False, gold_in_context=False),  # retrieval failure
        _row("q3", False, gold_in_context=False),  # retrieval failure
        _row("q4", True, gold_in_context=True),
    ]
    stats = _stats(rows)
    assert stats.wrong == 3
    assert stats.wrong_retrieved == 1
    assert stats.wrong_not_retrieved == 2
    assert stats.gold_retrieved == 2  # q1 and q4


def test_stats_gold_retrieved_counts_correct_rows_too_not_only_wrong_ones() -> None:
    rows = [_row("q1", True, gold_in_context=True), _row("q2", True, gold_in_context=False)]
    stats = _stats(rows)
    assert stats.gold_retrieved == 1
    assert stats.wrong_retrieved == 0
    assert stats.wrong_not_retrieved == 0


def test_accuracy_excludes_unparseable_from_the_denominator() -> None:
    stats = CategoryStats(n=10, correct=6, wrong=2, unparseable=2)
    assert stats.accuracy == pytest.approx(6 / 8)


def test_accuracy_all_scoreable_includes_unparseable_in_the_denominator_as_a_loss() -> None:
    stats = CategoryStats(n=10, correct=6, wrong=2, unparseable=2)
    assert stats.accuracy_all_scoreable == pytest.approx(6 / 10)


def test_the_two_denominators_diverge_exactly_when_there_are_unparseable_rows() -> None:
    """The regression this pins: F3 was two arms silently computed over DIFFERENT row sets. This
    proves the two accuracies are genuinely different numbers whenever unparseable > 0 — a caller
    who only ever printed one would be hiding real information, not a redundant duplicate."""
    stats = CategoryStats(n=20, correct=10, wrong=5, unparseable=5)
    assert stats.accuracy != stats.accuracy_all_scoreable
    assert stats.accuracy > stats.accuracy_all_scoreable  # excluding losses can only look better


def test_both_accuracies_agree_when_there_is_nothing_unparseable() -> None:
    stats = CategoryStats(n=10, correct=7, wrong=3, unparseable=0)
    assert stats.accuracy == pytest.approx(stats.accuracy_all_scoreable)


def test_accuracy_is_zero_not_a_crash_when_nothing_was_scoreable() -> None:
    stats = CategoryStats(n=3, correct=0, wrong=0, unparseable=3)
    assert stats.accuracy == 0.0
    assert stats.accuracy_all_scoreable == 0.0


def test_scoreable_equals_n_when_every_row_reached_the_judge() -> None:
    stats = CategoryStats(n=10, correct=6, wrong=3, unparseable=1)
    assert stats.scoreable == stats.n == 10
