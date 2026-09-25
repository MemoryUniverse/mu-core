"""OUR side of the mem0 head-to-head: conv-26 `gold_in_context`, swept over recall width.

WHY A WIDTH SWEEP AND NOT ONE NUMBER
------------------------------------
`gold_in_context` is monotone in the size of the returned window, so a single-width number is
only comparable to another arm measured at the SAME width. The two systems' own defaults are not
the same width and never were: `services/recall/width.py`'s own module docstring records it —
"mem0 sends the answering model **60** memories, MemOS **40** ... This engine sends **10**".
mem0's LoCoMo harness (`evaluation/src/memzero/search.py:19,91-96`) searches `top_k=10` per
speaker and concatenates BOTH speakers, so their benchmark context is 20 retrieved items.

Comparing our 10 against their 20 would measure the width gap and call it a quality gap. This
script therefore sweeps `limit` so the comparison can be read at any matched width, and so the
shipped default's position on our own curve is visible rather than assumed.

ONE INGEST, MANY WIDTHS: `limit` is a per-call argument to `LocalMemory.recall`, so every width in
the sweep is measured against the SAME populated index in the SAME process — not one ingest per
width, which would confound width with run-to-run ingest variation. Rerank configuration is read
at service construction, so THAT dimension does get its own ingest, one per arm.
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

sys.path.insert(0, os.environ.get("H2H_EVAL_DIR", str(Path(__file__).resolve().parent.parent)))

from mu_eval.corpus import ingest_conversation, local_memory_for
from mu_eval.locomo import load_locomo
from mu_eval.runner import _await_index, gold_ids_present

from mem0_h2h import emit


async def sweep_one_arm(
    conversation: Any, *, limits: list[int], importance: float, label: str
) -> dict[str, Any]:
    from mu_engine.config import get_engine_settings

    cfg = get_engine_settings().recall
    user, session = "evaluser", "evalsession"
    run_id = f"h2h{uuid.uuid4().hex[:6]}"
    known = {t.dia_id for t in conversation.turns}
    rows = []
    for query in conversation.queries:
        if query.is_adversarial or not query.evidence:
            continue
        gold = {e for e in query.evidence if e in known}
        if gold:
            rows.append((query, gold))

    by_limit: dict[int, dict[str, Any]] = {
        limit: {"hits": 0, "scored": 0, "widths": [], "lat": []} for limit in limits
    }
    # The rendered context for ONE nominated width, dumped so the answer-quality head-to-head can
    # answer + judge both systems' contexts in one process with one prompt and one judge. Rendered
    # EXACTLY as `mu_eval/answer_quality.py:538-540` renders it for a real run — `- {date}: {body}`
    # — so the arm that goes into the comparison is the arm this harness already measures, not a
    # second rendering that could drift from it.
    export_limit = int(os.environ.get("H2H_EXPORT_LIMIT", "0"))
    contexts: list[dict[str, Any]] = []

    turn_date = {t.dia_id: t.session_date for t in conversation.turns}

    async with local_memory_for(conversation, run_id=run_id) as opaque:
        memory: Any = opaque
        index, report = await ingest_conversation(
            memory, conversation, user=user, session=session, importance=importance
        )
        visible = await _await_index(
            memory, conversation.turns[0].text[:120], user=user, session=session
        )
        for limit in limits:
            bucket = by_limit[limit]
            for query, gold in rows:
                t0 = time.perf_counter()
                result = await memory.recall(
                    query.question, user=user, session=session, limit=limit
                )
                bucket["lat"].append((time.perf_counter() - t0) * 1000.0)
                bucket["widths"].append(len(result.items))
                present = gold_ids_present(result.items, index, gold)
                bucket["hits"] += int(present)
                bucket["scored"] += 1
                if limit == export_limit:
                    lines = []
                    for item in result.items:
                        dia = index.resolve(item.content)
                        stamp = turn_date.get(dia[0], "") if dia else ""
                        lines.append(f"- {stamp}: {item.content}" if stamp else f"- {item.content}")
                    contexts.append(
                        {
                            "query_id": query.query_id,
                            "question": query.question,
                            "gold_answer": query.answer,
                            "category": query.category,
                            "gold": sorted(gold),
                            "context": "\n".join(lines) or "(no memories retrieved)",
                            "items": len(result.items),
                            "gold_in_context": bool(present),
                        }
                    )
            emit(
                f"  [{label}] limit={limit:>3} width_mean="
                f"{statistics.mean(bucket['widths']):.2f} "
                f"gic={bucket['hits']}/{bucket['scored']}="
                f"{bucket['hits'] / bucket['scored']:.4f} "
                f"p50={statistics.median(bucket['lat']):.0f}ms",
            )

    return {
        "contexts": contexts,
        "label": label,
        "effective_config": {
            "rerank_enabled": cfg.rerank_enabled,
            "rerank_min_score": cfg.rerank_min_score,
            "rerank_top_fraction": cfg.rerank_top_fraction,
            "rerank_pool_size": cfg.rerank_pool_size,
            "channel_pool_size": cfg.channel_pool_size,
            "channel_pool_multiplier": getattr(cfg, "channel_pool_multiplier", None),
            "floor_protect_limit": cfg.floor_protect_limit,
        },
        "sample_id": conversation.sample_id,
        "corpus_turns": report.turns_written,
        "index_visible": visible,
        "by_limit": [
            {
                "limit": limit,
                "queries_scored": bucket["scored"],
                "gold_in_context": bucket["hits"],
                "gold_in_context_rate": bucket["hits"] / bucket["scored"],
                "width_mean": round(statistics.mean(bucket["widths"]), 3),
                "width_max": max(bucket["widths"]),
                "p50_ms": round(statistics.median(bucket["lat"]), 1),
            }
            for limit, bucket in by_limit.items()
        ],
    }


async def main() -> int:
    dataset = os.environ.get("H2H_DATASET", "/home/user/mu_eval_data/locomo10.json")
    sample = os.environ.get("H2H_SAMPLE", "conv-26")
    limits = [int(x) for x in os.environ.get("H2H_LIMITS", "5,10,20,30").split(",")]
    importance = float(os.environ.get("H2H_IMPORTANCE", "0.9"))
    label = os.environ.get("H2H_LABEL", "arm")
    out = Path(os.environ["H2H_OUT"])

    conversations = load_locomo(dataset, samples=None)
    matching = [c for c in conversations if c.sample_id == sample]
    if not matching:
        raise SystemExit(f"{sample} not in {dataset}")

    result = await sweep_one_arm(matching[0], limits=limits, importance=importance, label=label)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    emit(json.dumps(result["by_limit"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
