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
from mu_eval.runner import (
    NEIGHBOR_EXPANSION_MARKER_ATTR,
    gold_context_attribution,
    gold_ids_present,
)

pytestmark = pytest.mark.unit


class _StubItem:
    """Minimal stand-in for ``RecallItemView`` — ``gold_ids_present`` only reads ``.content``."""

    def __init__(self, content: str) -> None:
        self.content = content


class _StubNeighborItem:
    """Stand-in for a ``RecallItemView`` the neighbour-expansion arm (S1b) marked — carries the
    proposed ``is_neighbor`` attribute (AD-228, ``ARCHITECTURE-DELTAS.md``) on top of ``.content``,
    same shape ``RecallItemView.is_floor`` already uses for "how did this item get here"."""

    def __init__(self, content: str, *, is_neighbor: bool = True) -> None:
        self.content = content
        setattr(self, NEIGHBOR_EXPANSION_MARKER_ATTR, is_neighbor)


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


# ------------------------------------------------------------------- gold_context_attribution
#
# AD-228 (ARCHITECTURE-DELTAS.md): "the neighbour-expansion arm (S1b) must be measurable, or the
# change will measure as zero and be thrown away." These pin the two halves of that guarantee:
# (1) an expanded neighbour with real content is ALREADY counted as a gold hit today, with no
# marker needed at all — content-join, not channel-aware; (2) the marker, when a surface DOES set
# it, correctly attributes whether the hit needed expansion or would have happened anyway.


def test_present_counts_an_unmarked_item_exactly_like_gold_ids_present_does() -> None:
    """A plain `_StubItem` (no `is_neighbor` attribute at all — every recall surface in this repo
    today) must resolve identically through `gold_context_attribution` and `gold_ids_present`;
    they must never quietly disagree (`gold_ids_present`'s own docstring)."""
    index = _index_with(("D1:1", "I adopted a greyhound named Pepper."))
    items = [_StubItem("A: I adopted a greyhound named Pepper.")]
    attribution = gold_context_attribution(items, index, {"D1:1"})
    assert attribution.present is True
    assert attribution.present == gold_ids_present(items, index, {"D1:1"})
    assert attribution.via_neighbor_only is False  # no marker present -> never attributed to it
    assert attribution.neighbor_items == 0


def test_a_neighbor_marked_item_with_real_content_is_counted_as_a_gold_hit() -> None:
    """The core S1b claim: an item the expansion arm added, carrying the NEIGHBOUR's own real
    turn text, is resolved by the SAME content join a primary-channel item uses — no special
    casing needed for `present` to be True."""
    index = _index_with(("D1:1", "fact one"), ("D1:2", "fact two, the neighbour"))
    items = [
        _StubItem("A: fact one"),  # the primary hit, NOT gold
        _StubNeighborItem("A: fact two, the neighbour"),  # expanded neighbour, IS gold
    ]
    attribution = gold_context_attribution(items, index, {"D1:2"})
    assert attribution.present is True
    assert attribution.neighbor_items == 1


def test_via_neighbor_only_true_when_every_resolving_item_is_marked() -> None:
    index = _index_with(("D1:1", "the neighbour fact"))
    items = [_StubNeighborItem("A: the neighbour fact")]
    attribution = gold_context_attribution(items, index, {"D1:1"})
    assert attribution.present is True
    assert attribution.via_neighbor_only is True
    assert attribution.neighbor_items == 1


def test_via_neighbor_only_false_when_a_primary_item_also_resolves_to_the_same_gold() -> None:
    """The exact "measures as zero" risk AD-228 names, made concrete: if a primary channel ALSO
    found the gold turn, the expansion arm contributed nothing NEW for this query — `present` is
    (correctly) unaffected by expansion, and `via_neighbor_only` must say so explicitly rather
    than let a reader assume the marked item was the one that mattered."""
    index = _index_with(("D1:1", "the shared gold fact"))
    items = [
        _StubItem("A: the shared gold fact"),  # primary channel already found it
        _StubNeighborItem("A: the shared gold fact"),  # expansion re-surfaced the SAME turn
    ]
    attribution = gold_context_attribution(items, index, {"D1:1"})
    assert attribution.present is True
    assert attribution.via_neighbor_only is False  # NOT solely due to expansion
    assert attribution.neighbor_items == 1


def test_neighbor_items_counted_even_when_none_of_them_are_gold() -> None:
    """Lets a reader distinguish "the arm produced nothing" (0) from "the arm produced items but
    none happened to be gold for this query" (nonzero here, `present` still False)."""
    index = _index_with(("D1:1", "gold fact"), ("D1:2", "an irrelevant neighbour"))
    items = [_StubNeighborItem("A: an irrelevant neighbour")]
    attribution = gold_context_attribution(items, index, {"D1:1"})
    assert attribution.present is False
    assert attribution.via_neighbor_only is False
    assert attribution.neighbor_items == 1


def test_gold_context_attribution_on_empty_items_matches_gold_ids_present() -> None:
    index = _index_with(("D1:1", "some fact"))
    attribution = gold_context_attribution([], index, {"D1:1"})
    assert attribution.present is False
    assert attribution.via_neighbor_only is False
    assert attribution.neighbor_items == 0


# ---------------------------------------------------------------------------------- CategoryStats


def _row(
    query_id: str,
    verdict: bool | None,
    *,
    gold_in_context: bool,
    judge_raw: str | None = None,
    answer_budget_retried: bool = False,
    judge_budget_retried: bool = False,
) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        category=4,
        question="q",
        gold_answer="a",
        generated_answer="g",
        context_items=5,
        verdict=verdict,
        gold_in_context=gold_in_context,
        judge_raw=judge_raw,
        answer_budget_retried=answer_budget_retried,
        judge_budget_retried=judge_budget_retried,
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


# ------------------------------------------------------- TRUSTWORTHY-MEASUREMENT-0924: honestly
# splitting WHY a row is unparseable, from the actual raw judge output, not the bare counter.


def test_stats_splits_unparseable_by_the_budget_exhaustion_wire_signature() -> None:
    rows = [
        _row("q1", None, gold_in_context=True, judge_raw=""),  # budget exhausted
        _row("q2", None, gold_in_context=True, judge_raw="   "),  # whitespace-only: also blank
        _row("q3", None, gold_in_context=True, judge_raw="not a label at all"),  # a DIFFERENT
        # failure (real content, just no CORRECT/WRONG) — must NOT be counted as budget-shaped.
        _row("q4", True, gold_in_context=True, judge_raw="CORRECT"),
    ]
    stats = _stats(rows)
    assert stats.unparseable == 3
    assert stats.unparseable_budget_exhausted == 2
    assert stats.unparseable_other == 1


def test_stats_does_not_guess_budget_exhaustion_for_a_row_predating_judge_raw() -> None:
    """An older artifact's row carries `judge_raw=None` (the field did not exist yet) — it must
    fall into `unparseable_other`, never a guessed `unparseable_budget_exhausted`, so an old
    artifact never silently claims a characterisation it has no evidence for."""
    rows = [_row("q1", None, gold_in_context=True, judge_raw=None)]
    stats = _stats(rows)
    assert stats.unparseable == 1
    assert stats.unparseable_budget_exhausted == 0
    assert stats.unparseable_other == 1


def test_stats_counts_budget_retries_even_on_rows_that_ended_up_parsed() -> None:
    """The retry can SUCCEED (the row ends up correctly parsed) — `answer_budget_retried`/
    `judge_budget_retried` must still count it, so a report can show how often the safety net
    fired, not only its effect on the unparseable count."""
    rows = [
        _row("q1", True, gold_in_context=True, judge_budget_retried=True),
        _row("q2", True, gold_in_context=True, answer_budget_retried=True),
        _row("q3", True, gold_in_context=True),
    ]
    stats = _stats(rows)
    assert stats.judge_budget_retried == 1
    assert stats.answer_budget_retried == 1


# ------------------------------------------------------- the marker contract, pinned for real
#
# VERIFY PASS 2026-09-23. AD-228 proposes `is_neighbor` as the attribute the recall surface will
# carry, and `gold_context_attribution` reads it with `getattr(..., False)` so an un-adopting
# surface reads as "nothing came from expansion". Two things that were NOT pinned, and both
# matter, because the failure mode of this design is a counter that silently reads 0 forever:
#
#  1. Every existing test builds its stub with `setattr(self, NEIGHBOR_EXPANSION_MARKER_ATTR, ...)`
#     — so renaming the constant renames BOTH sides and the suite stays green. VERIFIED: mutating
#     the constant to "is_neighbour_XX" left all 19 tests passing.
#  2. `mu_contracts.contracts.recall.RecallItemView` — the item type `LocalMemory.recall` actually
#     returns, and therefore the one `run_baseline` reads — does not carry the field, and
#     `mu_local.local_memory._to_recall_result` does not forward the engine-internal
#     `RecallItemView.is_neighbor` onto it. So `neighbor_items_seen`/
#     `gold_in_context_via_neighbor_only` are structurally 0 on every real run, whatever S1b does.


def test_neighbor_marker_name_is_pinned_to_the_literal_the_engine_sets() -> None:
    """A rename of the constant alone must not pass silently — every stub in this file derives its
    attribute name FROM the constant, so only a literal pin catches a drift away from the name
    `mu_engine.services.recall.dto.RecallItemView` actually sets."""
    assert NEIGHBOR_EXPANSION_MARKER_ATTR == "is_neighbor"


def test_the_recall_surface_carries_the_neighbour_marker() -> None:
    """AD-233's TRIPWIRE, fired and now inverted (VERIFY 2026-09-24).

    This test used to assert the OPPOSITE — that the canonical surface did NOT yet carry
    `is_neighbor` — as a deliberate tripwire whose failure would mean "the contract landed, delete
    this test and re-read every conclusion that rested on a `neighbor_items_seen=0`". AD-236 landed
    that contract and the tripwire went red exactly as designed. It was then reported as *"a
    pre-existing tripwire tripped by a different concurrent lane's uncommitted engine change —
    unrelated to this pass"*, which is the one reading it cannot have: a tripwire firing is not
    noise from someone else's lane, it is the signal, and the instruction attached to it ("re-read
    every conclusion that rested on a zero here") is exactly what this verify pass then did — see
    AD-238, where the zero turned out to be a second truncation nobody had looked for.

    Kept rather than deleted, inverted: the field is part of the wire-versioned contract now, and a
    regression that removed it would otherwise silently restore the structural zero AD-233 exists
    to have caught once already."""
    from mu_contracts.contracts.recall import RecallItemView as SurfaceRecallItemView

    assert NEIGHBOR_EXPANSION_MARKER_ATTR in SurfaceRecallItemView.model_fields, (
        "the canonical recall surface lost the neighbour marker — every downstream "
        "`getattr(item, 'is_neighbor', False)` attribution is a structural zero again (AD-233)"
    )
