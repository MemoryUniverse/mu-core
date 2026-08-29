"""Unit tests for the ranking metrics — each one provably able to FAIL.

Every assertion below was checked by mutation: the metric was broken on purpose, the test went
RED, and the mutation was reverted (mutations and their results are recorded in
``docs/tracking/RETRIEVAL-EVAL-0829.md``). A metric harness whose own tests cannot fail is worse
than no harness, because it launders a wrong number as a measured one.
"""

from __future__ import annotations

import math

import pytest
from mu_eval.locomo import Turn, normalize_text
from mu_eval.metrics import (
    aggregate,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    score_query,
)

pytestmark = pytest.mark.unit


def test_recall_at_k_counts_only_the_top_k() -> None:
    retrieved = ["a", "b", "c", "d"]
    gold = ["c", "d"]
    assert recall_at_k(retrieved, gold, 2) == 0.0
    assert recall_at_k(retrieved, gold, 3) == 0.5
    assert recall_at_k(retrieved, gold, 4) == 1.0


def test_recall_refuses_an_empty_gold_set() -> None:
    # An unanswerable (LoCoMo category 5) row must be FILTERED, never averaged in as a 0.0 —
    # that would depress every reported number for a reason unrelated to ranking.
    with pytest.raises(ValueError, match="empty gold set"):
        recall_at_k(["a"], [], 5)


def test_precision_at_k_divides_by_k_not_by_the_number_retrieved() -> None:
    # Only one of five slots is relevant; a system that returned two items must not be rewarded
    # with 0.5 for occupying fewer slots.
    assert precision_at_k(["x", "g"], ["g"], 5) == pytest.approx(0.2)


def test_mrr_is_the_reciprocal_of_the_first_hit_rank() -> None:
    assert mrr_at_k(["g", "x", "x"], ["g"], 3) == 1.0
    assert mrr_at_k(["x", "g", "x"], ["g"], 3) == pytest.approx(0.5)
    assert mrr_at_k(["x", "x", "g"], ["g"], 3) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "x", "g"], ["g"], 2) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking_and_discounts_a_late_hit() -> None:
    assert ndcg_at_k(["g1", "g2", "x"], ["g1", "g2"], 3) == pytest.approx(1.0)
    # gold at ranks 2 and 3: DCG = 1/log2(3) + 1/log2(4); IDCG = 1/log2(2) + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(["x", "g1", "g2"], ["g1", "g2"], 3) == pytest.approx(expected)
    assert ndcg_at_k(["x", "x", "x"], ["g1"], 3) == 0.0


def test_ndcg_does_not_pay_twice_for_the_same_gold_document() -> None:
    """Returning one gold turn three times is not finding three relevant things.

    This is not hypothetical here: MU's own recall path deduplicates by ``content_hash`` at two
    layers precisely because the same fact surfaces from several tiers under different ids
    (``fusion.dedup_by_content_hash``). If that dedup regresses, nDCG must NOT reward it.
    """
    assert ndcg_at_k(["g", "g", "g"], ["g", "h"], 3) == pytest.approx(
        1.0 / (1 / math.log2(2) + 1 / math.log2(3))
    )


def test_aggregate_macro_averages_over_queries() -> None:
    rows = [
        score_query(query_id="q1", category=1, retrieved=["g"], gold=["g"], ks=(1,)),
        score_query(query_id="q2", category=1, retrieved=["x"], gold=["g"], ks=(1,)),
    ]
    assert aggregate(rows, (1,))["recall"][1] == pytest.approx(0.5)
    assert aggregate([], (1,))["recall"][1] == 0.0


def test_normalize_text_collapses_whitespace_and_case_but_not_content() -> None:
    assert normalize_text("  Hello   World \n") == normalize_text("hello world")
    assert normalize_text("hello world") != normalize_text("hello worlds")


def test_turn_ingest_text_carries_the_speaker_and_not_the_date() -> None:
    turn = Turn(
        dia_id="D1:3",
        session_index=1,
        session_date="7 May 2023",
        speaker="Caroline",
        text="I went to a LGBTQ support group yesterday.",
    )
    assert turn.ingest_text.startswith("Caroline: ")
    # The dataset's own date must not leak into the body: temporal-reasoning queries would then
    # be answerable by lexical date matching rather than by the memory system's time model.
    assert "2023" not in turn.ingest_text
