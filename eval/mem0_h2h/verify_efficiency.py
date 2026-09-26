"""AD-313 — the VERIFY pass on AD-309's efficiency numbers, plus the breakdown it left open.

AD-309 (`efficiency.py`, ADR 0084) measured MU's context tokens and search latency against mem0's
published figures (arXiv 2504.19413: 1764 tokens, 148 ms p50 search) and closed with three named
open items. Two of them are what this script exists for:

  1. **"the ~2x search-latency gap vs mem0's published number has a stated candidate cause
     (3-store round trip) but no per-channel breakdown yet."** AD-309's own words. A candidate
     cause with no measurement behind it is a guess, and the scorecard this pass has to write
     would be repeating it. :func:`instrumented` measures it instead.
  2. **Run-to-run spread.** AD-309 reported "286-311 ms p50 (2 runs)" — a range across two whole
     process invocations, which conflates process-start effects (model load, client construction,
     connection pools) with the query-to-query variance that actually matters when someone asks
     "how fast is recall". This script runs REPEATS INSIDE ONE PROCESS against ONE ingest, so the
     spread it reports is the spread of the thing under test and nothing else.

WHY A SECOND SCRIPT AND NOT A FLAG ON THE FIRST. `efficiency.py` is a measurement whose output is
already published (ADR 0084, `EFFICIENCY-RESULT-0926.md`). Adding instrumentation to it would
change the timings of the very run its published numbers came from -- a patched `embed` and four
patched adapter methods add per-call overhead to the critical path. This script therefore
measures BOTH ways in one process: an UNINSTRUMENTED pass (directly comparable to AD-309's
numbers, the verification) and then an INSTRUMENTED pass (the breakdown, knowingly inflated by the
wrappers). Reporting one number from a patched process as if it were the clean one is the exact
error this separation prevents.

THE INSTRUMENT, AND ITS LIMIT, STATED. :func:`instrumented` wraps five bound-at-class-level
methods on the REAL adapters the engine wires, restoring every one in a `finally`:

  * ``SentenceTransformerEmbedder.embed``  -- the MiniLM forward pass (`providers/embedding.py:73`)
  * ``RedisStmAdapter.recent`` / ``.demoted`` -- the two STM channels (`redis_stm.py:229,321`;
    `ValkeyStmAdapter` subclasses this, so patching the base catches the shipped adapter)
  * ``QdrantMtmAdapter.semantic``          -- the MTM vector channel (`qdrant_mtm.py:702`)
  * ``FalkorLtmAdapter.graph_recall``      -- the LTM graph channel (`falkor_ltm.py:757`)

**The three store channels run CONCURRENTLY** under the ranker's `asyncio.TaskGroup`
(`services/recall/ranker.py:289-338`), so their measured times OVERLAP each other: their sum is
NOT their contribution to wall clock -- the slowest of them is, approximately. The embedder calls
are different and that difference is the finding: the query embed happens at the `RecallService`
façade BEFORE the channels (`services/recall/service.py:130`) and the two `_score_stm` embeds
happen AFTER them, sequentially (`ranker.py:349,357` -> `_score_stm`, `ranker.py:758`). Embedder
time is therefore SERIAL in the critical path and adds to it in full. Every number below is
labelled `serial` or `concurrent` accordingly; no total is computed by summing across that line.

COST: **$0.** No LLM call, no judge, no answer model -- `recall()`, `tiktoken.encode()` and
`time.perf_counter()` only. Same free-by-construction property AD-309's own free arm has, for the
same reason (CLAUDE.md: "~$5 of the owner's $20 remains").
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import json
import os
import statistics
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import tiktoken  # noqa: E402
from mu_eval.answer_quality import ANSWER_SYSTEM_PROMPT  # noqa: E402
from mu_eval.corpus import ingest_conversation, local_memory_for  # noqa: E402
from mu_eval.judge import answer_prompt  # noqa: E402
from mu_eval.locomo import load_locomo  # noqa: E402
from mu_eval.runner import _await_index, gold_ids_present  # noqa: E402

from mem0_h2h import emit  # noqa: E402
from mem0_h2h.efficiency import _context_of, _dist, _eligible_rows, _percentile  # noqa: E402

# ------------------------------------------------------------------------------------ instrument

#: Stage label -> (import path, attribute, "serial" | "concurrent"). The third element is the one
#: that keeps this honest: a `concurrent` stage's measured time overlaps its siblings', so it may
#: never be added to another stage's. See the module docstring.
_TARGETS: tuple[tuple[str, str, str, str], ...] = (
    ("embed", "mu_engine.providers.embedding", "SentenceTransformerEmbedder.embed", "serial"),
    ("stm_recent", "mu_engine.storage.adapters.redis_stm", "RedisStmAdapter.recent", "concurrent"),
    (
        "stm_demoted",
        "mu_engine.storage.adapters.redis_stm",
        "RedisStmAdapter.demoted",
        "concurrent",
    ),
    (
        "mtm_semantic",
        "mu_engine.storage.adapters.qdrant_mtm",
        "QdrantMtmAdapter.semantic",
        "concurrent",
    ),
    (
        "ltm_graph_recall",
        "mu_engine.storage.adapters.falkor_ltm",
        "FalkorLtmAdapter.graph_recall",
        "concurrent",
    ),
)


class StageLog:
    """Per-recall accumulator of (stage, ms, batch_size) observations."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, int]] = []
        self.recording = False

    def record(self, stage: str, ms: float, batch: int) -> None:
        if self.recording:
            self.calls.append((stage, ms, batch))

    def drain(self) -> list[tuple[str, float, int]]:
        out = self.calls
        self.calls = []
        return out


@contextlib.contextmanager
def instrumented(log: StageLog) -> Iterator[None]:
    """Patch the five real adapter methods to time themselves into ``log``; restore unconditionally.

    Restoring in a `finally` matters more than it looks: these are CLASS attributes on adapters the
    rest of this process (teardown included) goes on using, and a leaked wrapper would silently
    keep timing -- and keep paying wrapper overhead -- for the remainder of the run, including the
    uninstrumented pass if the two were ever reordered.
    """
    import importlib

    restore: list[tuple[Any, str, Any]] = []
    try:
        for stage, module_path, dotted, _kind in _TARGETS:
            module = importlib.import_module(module_path)
            cls_name, meth_name = dotted.split(".")
            cls = getattr(module, cls_name)
            original = cls.__dict__.get(meth_name)
            if original is None:  # pragma: no cover - guards a rename, never hit in a green tree
                raise AttributeError(
                    f"{dotted} is not defined on {cls_name} -- instrument is stale"
                )

            def _wrap(orig: Any = original, stage_name: str = stage) -> Any:
                @functools.wraps(orig)
                async def timed(self: Any, *args: Any, **kwargs: Any) -> Any:
                    batch = len(args[0]) if stage_name == "embed" and args else 0
                    t0 = time.perf_counter()
                    try:
                        return await orig(self, *args, **kwargs)
                    finally:
                        log.record(stage_name, (time.perf_counter() - t0) * 1000.0, batch)

                return timed

            setattr(cls, meth_name, _wrap())
            restore.append((cls, meth_name, original))
        yield
    finally:
        for cls, meth_name, original in restore:
            setattr(cls, meth_name, original)


def _summarize_stages(per_query: list[list[tuple[str, float, int]]]) -> dict[str, Any]:
    """Aggregate per-recall stage observations into a per-stage distribution.

    Each stage's row is the TOTAL time that stage spent within one recall (a stage called more
    than once per recall -- `embed` is, three times -- is summed within the recall first, then
    distributed across recalls). `calls_per_recall` and `batch_sizes` are reported because for the
    embedder they ARE the finding: three calls, not one.
    """
    kinds = {stage: kind for stage, _m, _d, kind in _TARGETS}
    out: dict[str, Any] = {}
    for stage in kinds:
        per_recall_ms: list[float] = []
        per_recall_calls: list[float] = []
        batches: list[int] = []
        for calls in per_query:
            rows = [(ms, b) for (s, ms, b) in calls if s == stage]
            per_recall_ms.append(sum(ms for ms, _b in rows))
            per_recall_calls.append(float(len(rows)))
            batches.extend(b for _ms, b in rows if b)
        out[stage] = {
            "critical_path": kinds[stage],
            "ms_per_recall": _dist(per_recall_ms),
            "calls_per_recall_mean": round(statistics.mean(per_recall_calls), 2)
            if per_recall_calls
            else 0.0,
            "embed_batch_sizes_seen": sorted(set(batches))[:12],
        }
    return out


# ------------------------------------------------------------------------------------- free sweep


async def _one_pass(
    memory: Any,
    rows: list[tuple[Any, set[str]]],
    index: Any,
    turn_date: dict[str, str],
    *,
    limit: int,
    enc: Any,
    log: StageLog | None,
    user: str,
    session: str,
) -> dict[str, Any]:
    """One full sweep of every eligible query at one width. Returns the raw per-query series.

    The per-query LISTS are returned, not just their percentiles, because the two things the brief
    asks to rule out -- a cold-start outlier sitting in the p50, and a drift between repeats --
    are both invisible in an aggregate and obvious in the series.
    """
    lat_ms: list[float] = []
    ctx_tokens: list[float] = []
    prompt_tokens: list[float] = []
    stages: list[list[tuple[str, float, int]]] = []
    gic_hits = 0
    for query, gold in rows:
        if log is not None:
            log.drain()
            log.recording = True
        t0 = time.perf_counter()
        result = await memory.recall(query.question, user=user, session=session, limit=limit)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        if log is not None:
            log.recording = False
            stages.append(log.drain())
        gic_hits += int(gold_ids_present(result.items, index, gold))
        context = _context_of(result.items, index, turn_date)
        ctx_tokens.append(float(len(enc.encode(context))))
        full = answer_prompt(context=context, question=query.question)
        prompt_tokens.append(float(len(enc.encode(full)) + len(enc.encode(ANSWER_SYSTEM_PROMPT))))
    return {
        "limit": limit,
        "queries": len(rows),
        "gold_in_context_rate": round(gic_hits / len(rows), 4) if rows else 0.0,
        "gold_in_context_hits": gic_hits,
        "search_latency_ms": _dist(lat_ms),
        "context_tokens": _dist(ctx_tokens),
        "full_answer_prompt_tokens": _dist(prompt_tokens),
        "latency_series_ms": [round(x, 2) for x in lat_ms],
        "stages": _summarize_stages(stages) if log is not None else None,
    }


def _cold_start_check(series: list[float]) -> dict[str, Any]:
    """Is a cold-start outlier sitting in the p50? Answer it with the series, not an assurance.

    Reports the first query's latency next to the p50 of everything EXCEPT the first query. If the
    two p50s differ by less than the first query's own excess, the cold start is not in the p50 --
    which is what a warmed sweep should show, and what the brief asks to confirm rather than
    assume.
    """
    if len(series) < 3:
        return {"n": len(series), "verdict": "series too short to judge"}
    first = series[0]
    p50_all = _percentile(series, 0.50)
    p50_tail = _percentile(series[1:], 0.50)
    return {
        "first_query_ms": round(first, 2),
        "p50_including_first_ms": round(p50_all, 2),
        "p50_excluding_first_ms": round(p50_tail, 2),
        "p50_shift_from_first_query_ms": round(p50_all - p50_tail, 2),
        "first_query_excess_over_p50_pct": round((first / p50_tail - 1.0) * 100.0, 1)
        if p50_tail
        else 0.0,
    }


async def run(
    conversation: Any,
    *,
    limits: list[int],
    importance: float,
    model: str,
    repeats: int,
    warmup: int,
    breakdown: bool,
) -> dict[str, Any]:
    enc = tiktoken.encoding_for_model(model)
    user, session = "evaluser", "evalsession"
    run_id = f"v313{uuid.uuid4().hex[:6]}"
    rows = _eligible_rows(conversation)
    turn_date = {t.dia_id: t.session_date for t in conversation.turns}
    log = StageLog()

    doc: dict[str, Any] = {
        "sample_id": conversation.sample_id,
        "queries_eligible": len(rows),
        "repeats": repeats,
        "warmup_recalls": warmup,
        "clean": {},
        "instrumented": {},
    }

    async with local_memory_for(conversation, run_id=run_id) as opaque:
        memory: Any = opaque
        t0 = time.perf_counter()
        index, report = await ingest_conversation(
            memory, conversation, user=user, session=session, importance=importance
        )
        doc["ingest_wall_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        doc["ingest_llm_calls"] = 0
        doc["corpus_turns"] = report.turns_written
        await _await_index(memory, conversation.turns[0].text[:120], user=user, session=session)

        from mu_engine.config import get_engine_settings

        cfg = get_engine_settings().recall
        doc["effective_config"] = {
            "rerank_enabled": cfg.rerank_enabled,
            "channel_pool_size": cfg.channel_pool_size,
            "recency_floor_limit": cfg.recency_floor_limit,
            "demoted_floor_limit": cfg.demoted_floor_limit,
            "stm_scoring": cfg.stm_scoring,
            "sparse_enabled": cfg.sparse_enabled,
        }
        doc["tokenizer"] = {"model": model, "encoding": enc.name}

        # WARMUP -- thrown away, never reported. Every store client's connection pool, the MiniLM
        # weights and the tokenizer cache are all cold on the first recall of a process; a p50 that
        # includes that is a p50 of a one-off event. Discarding it is only legitimate if the
        # discarded thing is SHOWN, which `cold_start` below does.
        for i in range(warmup):
            await memory.recall(
                rows[i % len(rows)][0].question, user=user, session=session, limit=limits[0]
            )

        emit(f"== CLEAN (uninstrumented) -- directly comparable to AD-309, repeats={repeats} ==")
        for limit in limits:
            reps = []
            for r in range(repeats):
                res = await _one_pass(
                    memory,
                    rows,
                    index,
                    turn_date,
                    limit=limit,
                    enc=enc,
                    log=None,
                    user=user,
                    session=session,
                )
                res["cold_start"] = _cold_start_check(res["latency_series_ms"])
                reps.append(res)
                emit(
                    f"  [clean] limit={limit:>3} rep={r + 1}/{repeats} "
                    f"search p50={res['search_latency_ms']['p50']:.0f}ms "
                    f"p95={res['search_latency_ms']['p95']:.0f}ms "
                    f"prompt_tok mean={res['full_answer_prompt_tokens']['mean']:.0f} "
                    f"gic={res['gold_in_context_rate']:.4f}"
                )
            doc["clean"][str(limit)] = {
                "reps": reps,
                "spread": {
                    "search_p50_ms": [r["search_latency_ms"]["p50"] for r in reps],
                    "search_p95_ms": [r["search_latency_ms"]["p95"] for r in reps],
                    "prompt_tokens_mean": [r["full_answer_prompt_tokens"]["mean"] for r in reps],
                    "gold_in_context_rate": [r["gold_in_context_rate"] for r in reps],
                },
            }

        if breakdown:
            emit("== INSTRUMENTED -- per-stage breakdown (wrapper overhead INCLUDED, see docs) ==")
            with instrumented(log):
                for limit in limits:
                    res = await _one_pass(
                        memory,
                        rows,
                        index,
                        turn_date,
                        limit=limit,
                        enc=enc,
                        log=log,
                        user=user,
                        session=session,
                    )
                    doc["instrumented"][str(limit)] = res
                    st = res["stages"]
                    emit(
                        f"  [instr] limit={limit:>3} "
                        f"recall p50={res['search_latency_ms']['p50']:.0f}ms | "
                        + " ".join(
                            f"{s}={st[s]['ms_per_recall']['p50']:.0f}ms"
                            f"({st[s]['calls_per_recall_mean']:.0f}x,{st[s]['critical_path'][:4]})"
                            for s in st
                        )
                    )
    return doc


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default=os.environ.get("H2H_SAMPLE", "conv-26"))
    parser.add_argument(
        "--dataset",
        default=os.environ.get("H2H_DATASET", "/home/user/mu_eval_data/locomo10.json"),
    )
    parser.add_argument("--widths", default=os.environ.get("H2H_LIMITS", "10,20"))
    parser.add_argument("--importance", type=float, default=0.9)
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("H2H_REPEATS", "3")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("H2H_WARMUP", "5")))
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--no-breakdown", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    conversations = load_locomo(args.dataset, samples=None)
    matching = [c for c in conversations if c.sample_id == args.sample]
    if not matching:
        raise SystemExit(f"{args.sample} not in {args.dataset}")

    doc = await run(
        matching[0],
        limits=[int(x) for x in args.widths.split(",")],
        importance=args.importance,
        model=args.model,
        repeats=args.repeats,
        warmup=args.warmup,
        breakdown=not args.no_breakdown,
    )
    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    emit(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
