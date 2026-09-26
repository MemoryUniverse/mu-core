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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.environ.get("H2H_EVAL_DIR", str(Path(__file__).resolve().parent.parent)))

from dateutil import parser as _dateutil_parser
from mu_eval.corpus import ingest_conversation, local_memory_for
from mu_eval.locomo import Turn, load_locomo
from mu_eval.runner import _await_index, gold_ids_present

from mem0_h2h import emit


def _turn_occurred_at(turn: Turn) -> datetime | None:
    """AD-312: LoCoMo's own `session_date` ("1:56 pm on 8 May, 2023") parsed into a real,
    UTC-aware `datetime` — the true world-time the turn was said, exactly what mem0's own harness
    stamps onto every memory it ingests (`add.py:83`). Reached only through the PUBLIC
    `LocalMemory.add(occurred_at=...)` parameter (`corpus.py::ingest_conversation`'s own
    `occurred_at_of` seam) — no private hook, no harness-side content rewrite; the wire caller
    could pass the identical value. Returns `None` (not a guess) on any unparsed/empty date —
    `corpus.py` already treats `None` as "no assertion" exactly like a real caller omitting it.
    """
    if not turn.session_date.strip():
        return None
    try:
        parsed = _dateutil_parser.parse(turn.session_date)
    except (ValueError, OverflowError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


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
    #
    # AD-308 follow-up (team-lead review): `stamp = turn_date.get(...)` below is a HARNESS-SIDE
    # cheat — it re-matches each hit's body text back against the LoCoMo corpus's OWN known
    # session dates, a capability that exists nowhere in the shipped product (confirmed by reading
    # `mu-client`'s real context renderer, which has no date logic at all). The 77.3%/81.8%
    # temporal-reasoning numbers in HEAD-TO-HEAD-RESULT.md were produced with this rejoin. Now
    # that AD-308 threads `MemoryItem.valid_at` through to `RecallItemView.valid_at` for real, this
    # export builds BOTH variants from the SAME `recall()` call (no extra store round trip, no
    # re-ingest) so the two are exactly paired: `context` (unchanged, the harness rejoin — kept for
    # backward compatibility with any other consumer of this export) and `context_product` (dates
    # ONLY from `item.valid_at`, i.e. what the shipped product actually knows; undated when it does
    # not). `product_dated_items`/`items` on each row let a reader see the coverage gap directly
    # rather than infer it.
    export_limit = int(os.environ.get("H2H_EXPORT_LIMIT", "0"))
    contexts: list[dict[str, Any]] = []

    turn_date = {t.dia_id: t.session_date for t in conversation.turns}

    # AD-308 follow-up: does the product's OWN date-recovery (this pass's extraction fix) do
    # anything when the pipeline that actually calls it (MTM->LTM distill) is exercised, rather
    # than left unconditionally empty the way every prior baseline/answer-quality run left it
    # (`mu_eval/corpus.py`'s own docstring: "every prior baseline ... had an UNCONDITIONALLY
    # EMPTY graph")? Off by default (byte-identical to every prior run of this script);
    # `H2H_CONSOLIDATE=1` opts in.
    consolidate = os.environ.get("H2H_CONSOLIDATE", "0") == "1"

    # AD-312: LoCoMo's own real per-turn dates, threaded through the PUBLIC `LocalMemory.add(
    # occurred_at=...)` path (not a harness-side rewrite of what the model reads — see
    # `_turn_occurred_at`'s own docstring). Off by default (byte-identical to every prior run of
    # this script, including AD-308/310's own product_only measurement); `H2H_OCCURRED_AT=1`
    # opts in.
    use_occurred_at = os.environ.get("H2H_OCCURRED_AT", "0") == "1"

    async with local_memory_for(conversation, run_id=run_id) as opaque:
        memory: Any = opaque
        index, report = await ingest_conversation(
            memory,
            conversation,
            user=user,
            session=session,
            importance=importance,
            consolidate=consolidate,
            occurred_at_of=_turn_occurred_at if use_occurred_at else None,
        )
        if consolidate:
            emit(
                f"  [{label}] consolidate: facts_extracted={report.facts_extracted} "
                f"ltm_added={report.ltm_added} ltm_superseded={report.ltm_superseded} "
                f"ltm_noop={report.ltm_noop} consolidate_seconds={report.consolidate_seconds:.1f}"
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
                    product_lines = []
                    product_dated = 0
                    for item in result.items:
                        dia = index.resolve(item.content)
                        stamp = turn_date.get(dia[0], "") if dia else ""
                        lines.append(f"- {stamp}: {item.content}" if stamp else f"- {item.content}")
                        # AD-308: the PRODUCT'S OWN signal — `RecallItemView.valid_at`, threaded
                        # end to end by this pass's engine fix — nothing corpus-side. A `None`
                        # (never extracted/inferred for this item) renders undated, honestly.
                        product_valid_at = getattr(item, "valid_at", None)
                        if product_valid_at is not None:
                            product_dated += 1
                            product_lines.append(f"- {product_valid_at.date()}: {item.content}")
                        else:
                            product_lines.append(f"- {item.content}")
                    contexts.append(
                        {
                            "query_id": query.query_id,
                            "question": query.question,
                            "gold_answer": query.answer,
                            "category": query.category,
                            "gold": sorted(gold),
                            "context": "\n".join(lines) or "(no memories retrieved)",
                            "context_product": "\n".join(product_lines)
                            or "(no memories retrieved)",
                            "items": len(result.items),
                            "product_dated_items": product_dated,
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
