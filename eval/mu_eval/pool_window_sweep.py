"""AD-306 — the POOL sweep, matched-width, `gold_in_context` + latency + tokens, free (no LLM).

Lane brief (workflow harness, 2026-09-26): "The pool is the untouched half of the truncation
loss. Measured (TRACE-0923.md §3.1): 100% of gold is written, 98.4% is reachable at width 200,
then the pool of 20 loses 71 rows and the window of 10 loses 49. Width [of the WINDOW/limit] was
already widened and bought +16.8 pt (RETRIEVAL-EVAL-0829.md §13.2). Nobody has attacked the pool.
At the shipped limit=10, max(20, ceil(10*2.0)) is still 20, so the multiplier that exists does not
help at the default. Sweep it properly and find where recall saturates against what it costs —
more candidates means more fusion work and more embedding comparisons, so measure latency
alongside recall."

METHOD. One ingest (real Valkey+Qdrant+FalkorDB, `--importance 0.9` so the corpus reaches MTM,
matching every prior width pass in this document set). Then N query-only `LocalMemory` instances
built against the SAME workspace/namespace (no re-ingest — `engine_settings` is the AD-294
injection seam, `local_memory.py:168`) each with a different `RecallSettings.channel_pool_size`,
so the CANDIDATE POOL varies while the returned WINDOW (`limit`) is held fixed. This isolates the
pool from the window the way AD-303's width curve and TRACE-0923's ladder never did in one pass.

METRICS, per (pool_size, limit) cell: `gold_in_context` (exact, our own join — TRACE-0923's own
caveat that this is a CEILING applies only to mem0's provenance-batch credit, not to us),
recall e2e latency (ms, the SAME wall-clock a caller would see — `memory.recall()` end to end, not
a store-only probe), and rendered-context tokens (`mu_eval.token_budget.count_tokens`, the same
tokenizer/fallback this repo already uses for its Ministral budget work — reused rather than a
second ad hoc estimate, cited so a reader can tell it is an approximation for a DIFFERENT model
family, not a gpt-5 tokenizer).

`limit` is swept too (10 == shipped default, 20 == mem0's own per-arm width from AD-303/HEAD-TO-
HEAD-RESULT.md) so the pool's effect at the window this project actually ships is not conflated
with the wider window a different lane already measured.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(
    0, os.environ.get("POOLSWEEP_EVAL_DIR", str(Path(__file__).resolve().parent.parent))
)

from mu_eval.corpus import ingest_conversation, local_memory_for
from mu_eval.locomo import load_locomo
from mu_eval.runner import _await_index, gold_ids_present
from mu_eval.token_budget import count_tokens


def emit(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


async def one_cell(
    memory: Any,
    rows: list[tuple[Any, set[str]]],
    index: Any,
    turn_date: dict[str, str],
    *,
    user: str,
    session: str,
    limit: int,
) -> dict[str, Any]:
    hits = 0
    lat: list[float] = []
    widths: list[int] = []
    toks: list[int] = []
    for query, gold in rows:
        t0 = time.perf_counter()
        result = await memory.recall(query.question, user=user, session=session, limit=limit)
        lat.append((time.perf_counter() - t0) * 1000.0)
        widths.append(len(result.items))
        present = gold_ids_present(result.items, index, gold)
        hits += int(present)
        lines = []
        for item in result.items:
            dia = index.resolve(item.content)
            stamp = turn_date.get(dia[0], "") if dia else ""
            lines.append(f"- {stamp}: {item.content}" if stamp else f"- {item.content}")
        toks.append(count_tokens("\n".join(lines)))

    def pctl(values: list[float], p: float) -> float:
        s = sorted(values)
        if not s:
            return 0.0
        idx = min(len(s) - 1, int(round(p * (len(s) - 1))))
        return s[idx]

    return {
        "queries_scored": len(rows),
        "gold_in_context": hits,
        "gold_in_context_rate": round(hits / len(rows), 4) if rows else 0.0,
        "width_mean": round(statistics.mean(widths), 3) if widths else 0.0,
        "tokens_mean": round(statistics.mean(toks), 1) if toks else 0.0,
        "tokens_p95": round(pctl(toks, 0.95), 1),
        "p50_ms": round(statistics.median(lat), 1) if lat else 0.0,
        "p95_ms": round(pctl(lat, 0.95), 1),
        "mean_ms": round(statistics.mean(lat), 1) if lat else 0.0,
    }


async def main() -> int:
    from mu_engine.config.engine_settings import EngineSettings
    from mu_engine.services.recall.dto import RecallSettings

    dataset = os.environ.get("POOLSWEEP_DATASET", "/home/user/mu_eval_data/locomo10.json")
    sample = os.environ.get("POOLSWEEP_SAMPLE", "conv-26")
    importance = float(os.environ.get("POOLSWEEP_IMPORTANCE", "0.9"))
    pool_sizes_10 = [
        int(x)
        for x in os.environ.get("POOLSWEEP_POOLS_L10", "20,30,40,60,80,120,160,200").split(",")
    ]
    pool_sizes_20 = [
        int(x) for x in os.environ.get("POOLSWEEP_POOLS_L20", "20,40,80,160").split(",")
    ]
    repeats = int(os.environ.get("POOLSWEEP_REPEATS", "2"))
    out = Path(os.environ["POOLSWEEP_OUT"])

    conversations = load_locomo(dataset, samples=None)
    matching = [c for c in conversations if c.sample_id == sample]
    if not matching:
        raise SystemExit(f"{sample} not in {dataset}")
    conversation = matching[0]

    user, session = "evaluser", "evalsession"
    run_id = f"pw{uuid.uuid4().hex[:6]}"
    known = {t.dia_id for t in conversation.turns}
    rows = []
    for query in conversation.queries:
        if query.is_adversarial or not query.evidence:
            continue
        gold = {e for e in query.evidence if e in known}
        if gold:
            rows.append((query, gold))
    turn_date = {t.dia_id: t.session_date for t in conversation.turns}

    cells: list[dict[str, Any]] = []

    async with local_memory_for(conversation, run_id=run_id) as base_memory:
        base: Any = base_memory
        index, report = await ingest_conversation(
            base, conversation, user=user, session=session, importance=importance
        )
        visible = await _await_index(
            base, conversation.turns[0].text[:120], user=user, session=session
        )
        emit(
            f"ingested sample={sample} turns_written={report.turns_written} "
            f"index_visible={visible} queries={len(rows)}"
        )

        ws = base._workspace
        org = base._org

        plan = [(limit, p) for limit in (10,) for p in pool_sizes_10] + [
            (limit, p) for limit in (20,) for p in pool_sizes_20
        ]

        for limit, pool_size in plan:
            reps = []
            for _rep in range(repeats):
                from mu_local import LocalMemory

                eng = EngineSettings(recall=RecallSettings(channel_pool_size=pool_size))
                probe = LocalMemory(workspace=ws, namespace=org, engine_settings=eng)
                try:
                    cell = await one_cell(
                        probe, rows, index, turn_date, user=user, session=session, limit=limit
                    )
                finally:
                    await probe.aclose()
                reps.append(cell)
            agg = {
                "limit": limit,
                "channel_pool_size": pool_size,
                "reps": reps,
                "gold_in_context_rate_mean": round(
                    statistics.mean(r["gold_in_context_rate"] for r in reps), 4
                ),
                "gold_in_context_rate_spread": round(
                    max(r["gold_in_context_rate"] for r in reps)
                    - min(r["gold_in_context_rate"] for r in reps),
                    4,
                ),
                "p50_ms_mean": round(statistics.mean(r["p50_ms"] for r in reps), 1),
                "p95_ms_mean": round(statistics.mean(r["p95_ms"] for r in reps), 1),
                "tokens_mean": round(statistics.mean(r["tokens_mean"] for r in reps), 1),
            }
            cells.append(agg)
            emit(
                f"  limit={limit:>2} pool={pool_size:>3} "
                f"gic={agg['gold_in_context_rate_mean']:.4f} "
                f"(spread={agg['gold_in_context_rate_spread']:.4f}) "
                f"p50={agg['p50_ms_mean']:.0f}ms p95={agg['p95_ms_mean']:.0f}ms "
                f"tok={agg['tokens_mean']:.0f}"
            )

    out.write_text(
        json.dumps(
            {
                "sample_id": conversation.sample_id,
                "queries_scored": len(rows),
                "importance": importance,
                "repeats": repeats,
                "corpus_turns": report.turns_written,
                "cells": cells,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
