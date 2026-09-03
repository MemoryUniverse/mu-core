"""Unit tests: per-row usage on ``QueryResult``, the ``usage`` round-trip on
``AnswerQualityReport``, and ``eligible_query_count`` — the pre-run row count
``__main__._print_projected_cost`` needs BEFORE a run starts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mu_eval.answer_quality import (
    AnswerQualityReport,
    QueryResult,
    _stats,
    eligible_query_count,
)
from mu_eval.locomo import Conversation, LabelledQuery, Turn
from mu_eval.usage import CallUsage

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------- QueryResult


def _row(
    query_id: str,
    *,
    answer_usage: CallUsage | None = None,
    judge_usage: CallUsage | None = None,
) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        category=1,
        question="q",
        gold_answer="g",
        generated_answer="a",
        context_items=3,
        verdict=True,
        gold_in_context=True,
        answer_usage=answer_usage,
        judge_usage=judge_usage,
    )


def test_query_result_usage_fields_default_to_none() -> None:
    """Backwards compatibility: a row built the pre-fix way (no usage kwargs at all) must still
    construct — same discipline as `gold_in_context`'s own optional-field precedent."""
    row = _row("q1")
    assert row.answer_usage is None
    assert row.judge_usage is None


def test_query_result_carries_distinct_answer_and_judge_usage() -> None:
    answer_usage = CallUsage(prompt_tokens=800, completion_tokens=10, total_tokens=810)
    judge_usage = CallUsage(prompt_tokens=127, completion_tokens=300, total_tokens=427)
    row = _row("q1", answer_usage=answer_usage, judge_usage=judge_usage)
    assert row.answer_usage == answer_usage
    assert row.judge_usage == judge_usage
    assert row.answer_usage != row.judge_usage


# ----------------------------------------------------------------------- AnswerQualityReport.usage


def _report(rows: list[QueryResult], *, usage: dict[str, object] | None) -> AnswerQualityReport:
    return AnswerQualityReport(
        run_id="r1",
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


def test_usage_defaults_to_none_when_not_given() -> None:
    report = _report([_row("q1")], usage=None)
    assert report.usage is None


def test_usage_round_trips_through_the_exact_write_shape_the_cli_uses(tmp_path: Path) -> None:
    """`__main__._cmd_answer_quality` writes `report.model_dump(mode="json") | {"provenance":
    ..., "usage": ...}` and `_cmd_compare` re-opens that file through
    `AnswerQualityReport.model_validate_json`. `extra="forbid"` on the model means an undeclared
    merged key is a hard ValidationError — the same seam `test_compare_roundtrip_unit.py` closed
    for `provenance` (an artifact that made `compare` unusable, discovered live, not hypothesised).
    This is that same regression, pinned for `usage`."""
    usage_block = {
        "by_role": {
            "answer": {
                "requested_model": "gpt-5",
                "served_models": ["gpt-5-2025-08-07"],
                "totals": {
                    "calls": 2,
                    "prompt_tokens": 1600,
                    "completion_tokens": 20,
                    "total_tokens": 1620,
                    "reasoning_tokens": 0,
                    "calls_with_reasoning_reported": 0,
                },
                "rate": {
                    "prompt_usd_per_1m": 1.25,
                    "completion_usd_per_1m": 10.0,
                    "source": "test",
                },
                "estimated_cost_usd": 0.002,
            }
        },
        "total_estimated_cost_usd": 0.002,
        "rate_card": {"gpt-5": {"prompt_usd_per_1m": 1.25, "completion_usd_per_1m": 10.0}},
    }
    # Mirrors `_cmd_answer_quality` exactly: the report `run_answer_quality` returns carries
    # `usage=None` (its own default); the CLI's `_write` call is what merges the REAL usage dict
    # on top at write time — so the report constructed here deliberately has `usage=None`, and
    # the merge below is what the test is actually pinning.
    report = _report([_row("q1")], usage=None)
    path = tmp_path / "run.json"
    payload = report.model_dump(mode="json") | {"usage": usage_block}
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = AnswerQualityReport.model_validate_json(path.read_text(encoding="utf-8"))

    assert loaded.usage == usage_block
    assert loaded.usage["by_role"]["answer"]["totals"]["prompt_tokens"] == 1600


def test_artifact_without_usage_still_reads_back(tmp_path: Path) -> None:
    """An artifact from before this fix (no `usage` key) must still load."""
    report = _report([_row("q1")], usage=None)
    path = tmp_path / "old.json"
    path.write_text(json.dumps(report.model_dump(mode="json")), encoding="utf-8")

    loaded = AnswerQualityReport.model_validate_json(path.read_text(encoding="utf-8"))

    assert loaded.usage is None


# ---------------------------------------------------------------------------- eligible_query_count


def _turn(dia_id: str, text: str = "hello") -> Turn:
    return Turn(dia_id=dia_id, session_index=1, session_date="1 Jan 2026", speaker="A", text=text)


def _query(
    query_id: str, *, category: int = 4, evidence: tuple[str, ...] = ("D1:1",)
) -> LabelledQuery:
    return LabelledQuery(
        query_id=query_id, question="q", answer="a", evidence=evidence, category=category
    )


def _conversation(sample_id: str, *, queries: tuple[LabelledQuery, ...]) -> Conversation:
    return Conversation(
        sample_id=sample_id,
        speaker_a="A",
        speaker_b="B",
        turns=(_turn("D1:1"),),
        queries=queries,
    )


def test_eligible_query_count_matches_the_same_admission_rule_the_run_uses() -> None:
    convo = _conversation(
        "c1",
        queries=(
            _query("q1"),  # eligible: category 4, real evidence
            _query("q2", category=5),  # adversarial -> excluded
            _query("q3", evidence=("D1:99",)),  # evidence names an unknown turn -> excluded
        ),
    )
    assert eligible_query_count([convo]) == 1


def test_eligible_query_count_sums_across_multiple_conversations() -> None:
    a = _conversation("a", queries=(_query("q1"), _query("q2")))
    b = _conversation("b", queries=(_query("q3"),))
    assert eligible_query_count([a, b]) == 3


def test_eligible_query_count_respects_max_queries_per_sample() -> None:
    convo = _conversation("c1", queries=(_query("q1"), _query("q2"), _query("q3")))
    assert eligible_query_count([convo], max_queries_per_sample=2) == 2


def test_eligible_query_count_is_zero_for_no_conversations() -> None:
    assert eligible_query_count([]) == 0
