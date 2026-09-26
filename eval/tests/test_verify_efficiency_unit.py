"""Unit tier for AD-313's verify harness (`mem0_h2h.verify_efficiency`) — pure, no stores, no LLM.

Three things in that script can be wrong in a way no eyeball catches, and each of them would
corrupt a number that lands on the owner's scorecard:

  * the **instrument must restore** every method it patched, including when the body raises —
    a leaked timing wrapper keeps charging its own overhead to every later recall in the process,
    which is exactly how a "clean" pass stops being clean;
  * the **cold-start check must actually detect** a first-query outlier rather than reporting a
    reassuring shape regardless of the data (the brief's "confirm a cold-start outlier is not in
    the p50" is worthless if the confirmation cannot fail);
  * the **stage summary must never add a `concurrent` stage's time into a total**, and must sum a
    stage called several times WITHIN one recall before distributing across recalls — the
    embedder is called three times per recall, so a summary that averaged per-CALL instead of
    per-RECALL would report a third of the real serial cost.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from mem0_h2h.verify_efficiency import (  # noqa: E402
    _TARGETS,
    StageLog,
    _cold_start_check,
    _summarize_stages,
    instrumented,
)


def test_stage_log_records_only_while_recording() -> None:
    """The window matters: a `recall()` is bracketed by harness work (context rendering, token
    encoding) that must not land in the stage totals."""
    log = StageLog()
    log.record("embed", 5.0, 1)  # recording is False by default
    assert log.calls == []
    log.recording = True
    log.record("embed", 7.0, 3)
    log.recording = False
    log.record("embed", 9.0, 1)
    assert log.drain() == [("embed", 7.0, 3)]
    assert log.drain() == [], "drain must clear, or the next recall inherits this one's calls"


def test_summarize_sums_repeated_calls_within_one_recall_not_across() -> None:
    """The embedder is called THREE times per recall (query embed + two `_score_stm` embeds). The
    per-recall total is their SUM; a per-call mean would under-report the serial cost 3x."""
    per_query = [
        [("embed", 10.0, 1), ("embed", 20.0, 10), ("embed", 30.0, 10)],
        [("embed", 12.0, 1), ("embed", 18.0, 10), ("embed", 30.0, 10)],
    ]
    out = _summarize_stages(per_query)
    assert out["embed"]["ms_per_recall"]["p50"] == pytest.approx(60.0)
    assert out["embed"]["calls_per_recall_mean"] == pytest.approx(3.0)
    assert out["embed"]["embed_batch_sizes_seen"] == [1, 10]


def test_summarize_labels_every_stage_serial_or_concurrent() -> None:
    """The label is the guardrail against the one arithmetic error this breakdown invites: adding
    three concurrently-executed channel times together and calling the sum a latency budget."""
    out = _summarize_stages([[("embed", 1.0, 1)]])
    assert out["embed"]["critical_path"] == "serial"
    concurrent = {s for s in out if out[s]["critical_path"] == "concurrent"}
    assert concurrent == {"stm_recent", "stm_demoted", "mtm_semantic", "ltm_graph_recall"}
    assert {t[0] for t in _TARGETS} == set(out), "every target must appear in the summary"


def test_summarize_reports_zero_for_a_stage_that_never_fired() -> None:
    """A channel that is switched off must read 0 ms, not be absent — a missing key in the report
    reads as 'not measured', which is a different claim from 'measured, cost nothing'."""
    out = _summarize_stages([[("embed", 4.0, 1)]])
    assert out["ltm_graph_recall"]["ms_per_recall"]["p50"] == 0.0
    assert out["ltm_graph_recall"]["calls_per_recall_mean"] == 0.0


def test_cold_start_check_detects_a_first_query_outlier() -> None:
    series = [1800.0] + [300.0] * 20
    out = _cold_start_check(series)
    assert out["first_query_ms"] == 1800.0
    assert out["p50_excluding_first_ms"] == 300.0
    assert out["first_query_excess_over_p50_pct"] == pytest.approx(500.0)
    # ...and the whole point: with 21 rows, one 1800ms outlier does NOT move the p50.
    assert out["p50_shift_from_first_query_ms"] == 0.0


def test_cold_start_check_reports_a_p50_that_the_first_query_did_move() -> None:
    """The check has to be capable of failing. On a short series a single outlier DOES shift the
    p50, and the report must say so rather than print a comfortable 0.0."""
    out = _cold_start_check([1000.0, 100.0, 200.0])
    assert out["p50_including_first_ms"] == 200.0
    assert out["p50_excluding_first_ms"] == 150.0
    assert out["p50_shift_from_first_query_ms"] == pytest.approx(50.0)


def test_cold_start_check_refuses_to_judge_a_series_too_short() -> None:
    assert _cold_start_check([1.0, 2.0])["verdict"] == "series too short to judge"


def test_instrumented_restores_every_patched_method_even_when_the_body_raises() -> None:
    """The failure this pins is silent and durable: a leaked wrapper keeps timing (and keeps
    charging its overhead) for the rest of the process, so the 'clean' comparison pass would be
    measuring the instrument."""
    import importlib

    before = {}
    for _stage, module_path, dotted, _kind in _TARGETS:
        cls_name, meth = dotted.split(".")
        cls = getattr(importlib.import_module(module_path), cls_name)
        before[dotted] = cls.__dict__[meth]

    log = StageLog()
    with pytest.raises(RuntimeError, match="deliberate"):
        with instrumented(log):
            for _stage, module_path, dotted, _kind in _TARGETS:
                cls_name, meth = dotted.split(".")
                cls = getattr(importlib.import_module(module_path), cls_name)
                assert cls.__dict__[meth] is not before[dotted], f"{dotted} was not patched"
            raise RuntimeError("deliberate")

    for _stage, module_path, dotted, _kind in _TARGETS:
        cls_name, meth = dotted.split(".")
        cls = getattr(importlib.import_module(module_path), cls_name)
        assert cls.__dict__[meth] is before[dotted], f"{dotted} was left patched"


def test_instrumented_wrapper_times_the_real_call_and_records_the_batch_size() -> None:
    """Timing through the real patch path, not a hand-rolled copy of it — on the embedder, because
    its batch size is the number the breakdown's finding rests on."""
    import importlib

    module = importlib.import_module("mu_engine.providers.embedding")
    cls: Any = module.SentenceTransformerEmbedder
    calls: list[list[str]] = []

    async def fake_embed(self: Any, texts: Any) -> list[list[float]]:
        calls.append(list(texts))
        await asyncio.sleep(0.02)
        return [[0.0] for _ in texts]

    original = cls.__dict__.get("embed")
    cls.embed = fake_embed
    try:
        log = StageLog()
        log.recording = True
        with instrumented(log):
            out = asyncio.run(cls.embed(object(), ["a", "b", "c"]))
        assert len(out) == 3
        assert calls == [["a", "b", "c"]]
        (stage, ms, batch) = log.drain()[0]
        assert (stage, batch) == ("embed", 3)
        assert ms >= 20.0, "the wrapper must time the REAL awaited call, not just its scheduling"
    finally:
        if original is None:
            del cls.embed
        else:
            cls.embed = original
