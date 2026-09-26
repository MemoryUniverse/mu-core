"""Unit tier for AD-309's efficiency harness (`mem0_h2h.efficiency`) — pure, no stores, no LLM.

Only the functions that do not need a real `LocalMemory`/recall/answer call are testable here
(`_percentile`, `_dist`, `_context_of`) — the same "pure piece first" split
`test_mem0_h2h_unit.py` already uses for `speaker_batches`. The two things worth pinning:

  * the percentile is a real linear-interpolation percentile, not a bucketed approximation that
    would silently mis-rank a p95 against mem0's published one, and
  * `_context_of` renders BYTE-IDENTICALLY to `answer_quality.py:538-540` / `ours_arm.py`'s own
    export, since a second, drifted rendering would tokenize a different string than the one the
    shipped answer-quality path actually sends.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from mem0_h2h.efficiency import _context_of, _dist, _percentile  # noqa: E402


def test_percentile_matches_known_linear_interpolation() -> None:
    # numpy's default ("linear") percentile of 1..10 at p95 is 9.55 — the textbook case this
    # implementation is supposed to reproduce without a numpy dependency in this harness.
    # `pytest.approx`, not `==`: the interpolation is a float multiply-add and 9.549999999999999
    # is the correct IEEE-754 result, not a bug to chase.
    values = [float(x) for x in range(1, 11)]
    assert _percentile(values, 0.95) == pytest.approx(9.55)
    assert _percentile(values, 0.50) == 5.5
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 10.0


def test_percentile_empty_is_zero_not_a_crash() -> None:
    assert _percentile([], 0.95) == 0.0


def test_dist_reports_n_and_every_stat() -> None:
    d = _dist([10.0, 20.0, 30.0])
    assert d["n"] == 3
    assert d["mean"] == 20.0
    assert d["min"] == 10.0
    assert d["max"] == 30.0
    # p50 of an odd-length list is the middle element exactly.
    assert d["p50"] == 20.0


def test_dist_empty_reports_zero_n_not_a_crash() -> None:
    d = _dist([])
    assert d["n"] == 0
    assert d["mean"] == 0.0


class _FakeItem:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeIndex:
    """`resolve(content) -> [dia_id, ...]` — only the shape `_context_of` reads."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    def resolve(self, content: str) -> list[str]:
        return self._mapping.get(content, [])


def test_context_of_prefixes_the_date_when_the_item_resolves_to_a_known_turn() -> None:
    index: Any = _FakeIndex({"hello there": ["D1:1"]})
    turn_date = {"D1:1": "1:00 pm on 3 May, 2023"}
    out = _context_of([_FakeItem("hello there")], index, turn_date)
    assert out == "- 1:00 pm on 3 May, 2023: hello there"


def test_context_of_falls_back_to_undated_when_the_item_does_not_resolve() -> None:
    # An LTM-distilled paraphrase, or any item this harness's own corpus never wrote — no
    # normalized-body match, so no date can be honestly attached (see module docstring).
    index: Any = _FakeIndex({})
    out = _context_of([_FakeItem("a distilled paraphrase")], index, {})
    assert out == "- a distilled paraphrase"


def test_context_of_empty_pool_is_the_explicit_placeholder() -> None:
    index: Any = _FakeIndex({})
    assert _context_of([], index, {}) == "(no memories retrieved)"
