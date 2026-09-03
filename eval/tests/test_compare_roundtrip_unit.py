"""The gap that let `mu_eval compare` ship unable to open its own artifacts.

`test_compare_unit.py` builds `AnswerQualityReport` objects in memory and passes them straight to
`compare_runs`. `__main__._cmd_answer_quality` does NOT write those objects — it writes
`report.model_dump(mode="json") | {"provenance": build_provenance(...)}`, and
`__main__._cmd_compare` reads that file back through `AnswerQualityReport.model_validate_json`.
Because the model sets `extra="forbid"`, the merged `provenance` key made every artifact the
harness produces unreadable by `compare` — measured live against a real run's `--out`, not
hypothesised. No in-memory test could see it: the defect lives exactly in the write->read seam
neither side covered.

These tests therefore go through the FILE, the same way the CLI does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mu_eval.answer_quality import AnswerQualityReport, CategoryStats, QueryResult, _stats
from mu_eval.compare import compare_runs

PROVENANCE: dict[str, Any] = {
    "dataset_path": "/data/locomo10.json",
    "dataset_sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    "code_revision": "644287d51f1d9ad4382c28fe8718999b332c8771",
    "code_dirty": True,
    "recall_settings": {"ambient": True, "recall": {"weight_ltm": 0.1}, "ingest": {}},
    "models": [{"label": "answer", "requested": "gpt-5", "served": ["gpt-5-2025-08-07"]}],
}


def _report(run_id: str, verdicts: list[bool | None]) -> AnswerQualityReport:
    rows = [
        QueryResult(
            query_id=f"q{i}",
            category=1,
            question="q",
            gold_answer="g",
            generated_answer="a",
            context_items=10,
            verdict=v,
            gold_in_context=bool(i % 2),
        )
        for i, v in enumerate(verdicts)
    ]
    stats: CategoryStats = _stats(rows)
    return AnswerQualityReport(
        run_id=run_id,
        dataset="locomo10.json",
        samples=1,
        answer_model="gpt-5",
        judge_model="gpt-5",
        recall_limit=10,
        importance=0.9,
        queries_scored=len(rows),
        skipped_adversarial=0,
        skipped_no_gold_in_corpus=0,
        overall=stats,
        by_category={"1:multi hop": stats},
        rows=rows,
    )


def _write_like_the_cli(path: Path, report: AnswerQualityReport) -> None:
    """Byte-for-byte the shape `__main__._cmd_answer_quality`/`_write` produce."""
    payload = report.model_dump(mode="json") | {"provenance": PROVENANCE}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_artifact_with_provenance_reads_back(tmp_path: Path) -> None:
    """THE REGRESSION: an artifact carrying a provenance block must re-open through the same
    model `_cmd_compare` uses. Before `provenance` was declared on the model this raised
    ValidationError('Extra inputs are not permitted')."""
    path = tmp_path / "run.json"
    _write_like_the_cli(path, _report("a", [True, False, True]))

    loaded = AnswerQualityReport.model_validate_json(path.read_text(encoding="utf-8"))

    assert loaded.run_id == "a"
    assert loaded.provenance is not None
    # The provenance must survive the round-trip INTACT — a reader reproducing a configuration
    # from the artifact alone (item 2) depends on every field, not just the block's presence.
    assert loaded.provenance == PROVENANCE
    assert loaded.provenance["models"][0]["served"] == ["gpt-5-2025-08-07"]


def test_compare_works_end_to_end_on_written_artifacts(tmp_path: Path) -> None:
    """The full CLI path: write two artifacts, read both back, pair them."""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write_like_the_cli(a, _report("a", [True, False, False, True]))
    _write_like_the_cli(b, _report("b", [True, True, False, False]))

    ra = AnswerQualityReport.model_validate_json(a.read_text(encoding="utf-8"))
    rb = AnswerQualityReport.model_validate_json(b.read_text(encoding="utf-8"))
    result = compare_runs(ra, rb)

    assert result.common_rows == 4
    assert result.fixed == ["q1"]  # wrong -> correct
    assert result.regressed == ["q3"]  # correct -> wrong
    assert result.unchanged_correct == 1
    assert result.unchanged_wrong == 1


def test_artifact_without_provenance_still_reads_back(tmp_path: Path) -> None:
    """Backwards compatibility: `provenance` is optional, so an artifact written before item 2
    (or with `--out` from a command that captures none) must still load rather than becoming
    unreadable by the new reader."""
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(_report("old", [True, False]).model_dump(mode="json")), encoding="utf-8"
    )

    loaded = AnswerQualityReport.model_validate_json(path.read_text(encoding="utf-8"))

    assert loaded.provenance is None


def test_genuinely_unknown_key_is_still_rejected(tmp_path: Path) -> None:
    """Declaring `provenance` must NOT have loosened the model into `extra="allow"` — a typo'd
    or unknown top-level key should still be a loud failure, not silently ignored."""
    path = tmp_path / "typo.json"
    payload = _report("x", [True]).model_dump(mode="json") | {"provenence": PROVENANCE}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="provenence"):
        AnswerQualityReport.model_validate_json(path.read_text(encoding="utf-8"))
