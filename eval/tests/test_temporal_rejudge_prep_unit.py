"""AD-308 follow-up: `temporal_rejudge_prep.split_temporal_rows` — pure filter+remap, no store,
no LLM. Pins the two properties the real re-judge run depends on: only category==2 (temporal
reasoning) rows survive, and `--which product` swaps in `context_product` without touching any
other field (so `answer_h2h.py`'s unmodified `row["context"]` reader works on either output).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mem0_h2h"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mem0_h2h.temporal_rejudge_prep import (
    ORIGINAL_HEADLINE_QUERY_IDS,
    TEMPORAL_CATEGORY,
    split_temporal_rows,
)

pytestmark = pytest.mark.unit

_ROWS = [
    {
        "query_id": "q1",
        "category": 4,  # single hop — must be dropped
        "context": "- harness dated line",
        "context_product": "- product dated line",
        "product_dated_items": 1,
    },
    {
        "query_id": "q2",
        "category": TEMPORAL_CATEGORY,
        "context": "- 7 May 2023: Ada adopted a dog",
        "context_product": "- 2023-05-07: Ada adopted a dog",
        "product_dated_items": 1,
    },
    {
        "query_id": "q3",
        "category": TEMPORAL_CATEGORY,
        "context": "- 12 June 2023: Bo started a new job",
        "context_product": "- Bo started a new job",  # no product date recovered
        "product_dated_items": 0,
    },
]


def test_only_temporal_category_rows_survive() -> None:
    rows, _ = split_temporal_rows(_ROWS, which="harness")
    assert {r["query_id"] for r in rows} == {"q2", "q3"}


def test_harness_mode_leaves_context_untouched() -> None:
    rows, dated = split_temporal_rows(_ROWS, which="harness")
    by_id = {r["query_id"]: r for r in rows}
    assert by_id["q2"]["context"] == "- 7 May 2023: Ada adopted a dog"
    assert by_id["q3"]["context"] == "- 12 June 2023: Bo started a new job"
    assert dated == 0  # meaningless in harness mode, never computed


def test_product_mode_swaps_context_for_context_product() -> None:
    rows, dated = split_temporal_rows(_ROWS, which="product")
    by_id = {r["query_id"]: r for r in rows}
    assert by_id["q2"]["context"] == "- 2023-05-07: Ada adopted a dog"
    assert by_id["q3"]["context"] == "- Bo started a new job"
    assert dated == 1  # only q2 had >=1 product-dated item


def test_product_mode_does_not_mutate_the_input() -> None:
    original = [dict(r) for r in _ROWS]
    split_temporal_rows(_ROWS, which="product")
    assert _ROWS == original


def test_headline_only_restricts_to_the_published_query_ids() -> None:
    rows_with_headline_id = [
        {**_ROWS[1], "query_id": next(iter(ORIGINAL_HEADLINE_QUERY_IDS))},
        _ROWS[2],  # a temporal row NOT in the published headline set
    ]
    rows, _ = split_temporal_rows(rows_with_headline_id, which="harness", headline_only=True)
    assert {r["query_id"] for r in rows} == {next(iter(ORIGINAL_HEADLINE_QUERY_IDS))}


def test_headline_query_id_set_has_exactly_22_ids() -> None:
    assert len(ORIGINAL_HEADLINE_QUERY_IDS) == 22
