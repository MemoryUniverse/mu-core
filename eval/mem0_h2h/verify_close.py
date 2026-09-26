"""AD-322 close-phase verification — re-run the three FREE claims instead of reading reports.

Verifies, in ONE process against real stores on ``mu-dev-vm``:

1. **Determinism of the fused result set** (AD-317/AD-318). The same query is recalled TWICE, back
   to back, against the SAME populated index in the SAME process, and the two returned item lists
   are compared both as ORDERED sequences and as SETS. AD-318 claims 0/150 rows differ with the
   ``(-score, eid)`` tie-break and 30/150 differed without it. Run this script on a checkout with
   the fix and on one without it: the difference between the two runs is the claim.

2. **Search latency** (AD-313 §4) — p50/p95 over the same 150 queries, with the first ``--warmup``
   recalls discarded and the cold-start effect on p50 reported rather than asserted. Both
   ``stm_scoring`` arms must be run back to back in one session for their comparison to mean
   anything (AD-313: spread is ~9 % quiet, up to 60 % under another lane's load), so this script
   records ``loadavg`` at start and end of every run.

3. **``gold_in_context``** — unchanged-ness is half of AD-318's claim (a tie-break must not move
   which GOLD items are in the window, only which non-gold candidate wins a coin flip).

No answering model, no judge, no LLM call of any kind: **$0.00**.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, os.environ.get("H2H_EVAL_DIR", str(Path(__file__).resolve().parent.parent)))

from mu_eval.corpus import ingest_conversation, local_memory_for
from mu_eval.locomo import load_locomo
from mu_eval.runner import _await_index, gold_ids_present

from mem0_h2h import emit


def _loadavg() -> list[float]:
    return list(os.getloadavg())


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


async def run(
    conversation: Any, *, width: int, warmup: int, label: str, reps: int = 2
) -> dict[str, Any]:
    from mu_engine.config import get_engine_settings

    cfg = get_engine_settings().recall
    user, session = "evaluser", "evalsession"
    run_id = f"v322{uuid.uuid4().hex[:6]}"
    known = {t.dia_id for t in conversation.turns}
    rows = [
        (q, {e for e in q.evidence if e in known})
        for q in conversation.queries
        if not q.is_adversarial and q.evidence
    ]
    rows = [(q, gold) for q, gold in rows if gold]

    load_start = _loadavg()
    out: dict[str, Any] = {
        "label": label,
        "width": width,
        "sample_id": conversation.sample_id,
        "queries": len(rows),
        "effective_config": {
            "stm_scoring": cfg.stm_scoring,
            "rerank_enabled": cfg.rerank_enabled,
            "channel_pool_size": cfg.channel_pool_size,
            "floor_protect_limit": cfg.floor_protect_limit,
            "recency_floor_limit": cfg.recency_floor_limit,
        },
        "loadavg_start": load_start,
    }

    async with local_memory_for(conversation, run_id=run_id) as opaque:
        memory: Any = opaque
        t_ingest = time.perf_counter()
        index, report = await ingest_conversation(
            memory, conversation, user=user, session=session, importance=0.9
        )
        out["ingest_s"] = round(time.perf_counter() - t_ingest, 2)
        out["corpus_turns"] = report.turns_written
        out["index_visible"] = await _await_index(
            memory, conversation.turns[0].text[:120], user=user, session=session
        )

        for _ in range(warmup):
            await memory.recall(rows[0][0].question, user=user, session=session, limit=width)

        rep_rows: list[dict[str, Any]] = []
        for rep in range(1, reps + 1):
            lat: list[float] = []
            hits = 0
            shapes: list[dict[str, Any]] = []
            for query, gold in rows:
                t0 = time.perf_counter()
                result = await memory.recall(
                    query.question, user=user, session=session, limit=width
                )
                lat.append((time.perf_counter() - t0) * 1000.0)
                present = gold_ids_present(result.items, index, gold)
                hits += int(present)
                shapes.append(
                    {
                        "query_id": query.query_id,
                        # Content, not id: the comparison must survive being read across runs.
                        "ordered": [it.content for it in result.items],
                        "gold_in_context": bool(present),
                    }
                )
            rep_rows.append(
                {
                    "rep": rep,
                    "p50_ms": round(statistics.median(lat), 1),
                    "p95_ms": round(_pct(lat, 0.95), 1),
                    "mean_ms": round(statistics.mean(lat), 1),
                    "first_query_ms": round(lat[0], 1),
                    "p50_without_first_ms": round(statistics.median(lat[1:]), 1),
                    "gold_in_context": hits,
                    "gold_in_context_rate": round(hits / len(rows), 4),
                    "shapes": shapes,
                }
            )

    pairs = []
    for i in range(len(rep_rows) - 1):
        a, b = rep_rows[i]["shapes"], rep_rows[i + 1]["shapes"]
        ordered_diff = [
            x["query_id"] for x, y in zip(a, b, strict=True) if x["ordered"] != y["ordered"]
        ]
        set_diff = [
            x["query_id"]
            for x, y in zip(a, b, strict=True)
            if set(x["ordered"]) != set(y["ordered"])
        ]
        gic_diff = [
            x["query_id"]
            for x, y in zip(a, b, strict=True)
            if x["gold_in_context"] != y["gold_in_context"]
        ]
        pairs.append(
            {
                "pair": f"rep{i + 1}_vs_rep{i + 2}",
                "rows": len(a),
                "ordered_differs": len(ordered_diff),
                "set_differs": len(set_diff),
                "gold_in_context_differs": len(gic_diff),
                "set_differs_ids": set_diff[:20],
            }
        )
    out["determinism"] = pairs
    out["latency"] = [{k: v for k, v in r.items() if k != "shapes"} for r in rep_rows]
    out["loadavg_end"] = _loadavg()
    out["reps_shapes_kept"] = False
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default=os.environ.get("H2H_SAMPLE", "conv-26"))
    parser.add_argument(
        "--dataset",
        default=os.environ.get("H2H_DATASET", "/home/user/mu_eval_data/locomo10.json"),
    )
    parser.add_argument("--width", type=int, default=int(os.environ.get("H2H_LIMITS", "20")))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("H2H_WARMUP", "5")))
    parser.add_argument("--reps", type=int, default=int(os.environ.get("H2H_REPEATS", "2")))
    parser.add_argument("--label", default=os.environ.get("H2H_LABEL", "verify"))
    parser.add_argument("--out", default=os.environ.get("H2H_OUT", "verify_close.json"))
    args = parser.parse_args()

    conversations = load_locomo(args.dataset, samples=None)
    matching = [c for c in conversations if c.sample_id == args.sample]
    if not matching:
        raise SystemExit(f"{args.sample} not in {args.dataset}")

    result = await run(
        matching[0], width=args.width, warmup=args.warmup, label=args.label, reps=args.reps
    )
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    emit(json.dumps({k: v for k, v in result.items() if k != "latency"}, indent=1))
    emit(json.dumps(result["latency"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
