"""AD-307 — selective/ambiguity-gated rerank: pay the model cost only when the fused ranking
looks genuinely close, pass through untouched otherwise. Monkeypatches `AdaptiveRerankGate.apply`
at the seam (no source file touched) to add an ambiguity pre-check ahead of the existing gate,
so the shipped gate's own algebra (adaptive_rerank_gate, reorder-only cutoff) is untouched and
this stays a measurement, not a shipped behaviour change.

AMBIGUITY METRIC: relative margin between the fused_score of the pool's rank-1 and rank-`limit`
candidate — `(top - kth) / top` — computed over the pool BEFORE any rerank call, using the field
the fusion stage already populates (`RecallItemView.fused_score`, dto.py:111). Low margin means
the fused ordering barely separates the candidates competing for the result window's last slot:
exactly the case a reorder can flip. High margin means fusion already picked a clear winner and a
reorder has nothing to change. This is the SAME two-number shape (top vs a reference point) the
shipped `adaptive_rerank_gate` already uses for its own cutoff (`rerank_gate.py:93-127`), applied
to the PRE-rerank signal instead of the post-rerank one.

Reports, per threshold: trigger rate (fraction of 150 queries that call the model), and
gold_in_context under the MIXED policy (triggered queries get the real bge-v2-m3 rerank, reorder-
only, pool=20; non-triggered queries pass through in fused order) vs the two pure controls
(always-on, always-off). Latency is reported per-query so mean/p50/p95 can be computed honestly
(the brief: "be honest that p95 does not improve").
"""

# Diagnostic/one-off measurement script (AD-307, not shipped-path code): CLI print output, plain
# urllib against a localhost-only diagnostic endpoint, and a quick assert are all intentional here.
# ruff: noqa: T201, S310, S101, ANN001, ANN201, ANN202, F841, E501
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

sys.path.insert(0, os.environ["AD307_EVAL_DIR"])

from mu_eval.corpus import ingest_conversation, local_memory_for
from mu_eval.locomo import load_locomo
from mu_eval.runner import _await_index, gold_ids_present


def _pctl(v: list[float], p: float) -> float:
    if not v:
        return 0.0
    v = sorted(v)
    k = (len(v) - 1) * p
    f, c = int(k), min(int(k) + 1, len(v) - 1)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


async def run_policy(
    *, dataset: str, limit: int, importance: float, thresholds: list[float]
) -> dict[str, Any]:
    os.environ["MU_RECALL__RERANK_ENABLED"] = "true"
    os.environ["MU_RECALL__RERANK_MIN_SCORE"] = "0.0"
    os.environ["MU_RECALL__RERANK_TOP_FRACTION"] = "0.0"
    os.environ["MU_RECALL__RERANK_POOL_SIZE"] = "20"

    from mu_engine.config import get_engine_settings
    from mu_engine.services.recall import rerank_gate as rg_mod

    get_engine_settings.cache_clear()

    orig_apply = rg_mod.AdaptiveRerankGate.apply
    # per-query record: (margin, triggered_per_threshold, off_hit, on_hit, off_ms, on_ms)
    records: list[dict[str, Any]] = []

    async def spying_apply(self, pool, query):  # type: ignore[no-untyped-def]
        head = list(pool[: self._pool_size])
        margin = None
        if head:
            scores = [item.fused_score for item in head]
            top = scores[0]
            kth = scores[min(limit, len(scores)) - 1]
            margin = (top - kth) / top if top else 0.0
        t0 = time.perf_counter()
        on_result = await orig_apply(self, pool, query)
        on_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "margin": margin,
                "off_result": list(pool),
                "on_result": on_result,
                "on_ms": on_ms,
            }
        )
        return on_result

    rg_mod.AdaptiveRerankGate.apply = spying_apply  # type: ignore[method-assign]

    convs = load_locomo(dataset, samples=1)
    conv = convs[0]
    user, session = "evaluser", "evalsession"
    run_id = f"sel{uuid.uuid4().hex[:6]}"
    known = {t.dia_id for t in conv.turns}
    per_query: list[dict[str, Any]] = []
    async with local_memory_for(conv, run_id=run_id) as opaque:
        memory: Any = opaque
        index, report = await ingest_conversation(
            memory, conv, user=user, session=session, importance=importance
        )
        await _await_index(memory, conv.turns[0].text[:120], user=user, session=session)
        for q in conv.queries:
            if q.is_adversarial or not q.evidence:
                continue
            gold = {e for e in q.evidence if e in known}
            if not gold:
                continue
            records.clear()
            t0 = time.perf_counter()
            result = await memory.recall(q.question, user=user, session=session, limit=limit)
            base_ms = (time.perf_counter() - t0) * 1000.0
            if not records:
                continue  # gate never fired (e.g. empty pool) -- skip, matches AD-301's convention
            rec = records[0]
            off_in_context = gold_ids_present(rec["off_result"][:limit], index, gold)
            on_in_context = gold_ids_present(rec["on_result"][:limit], index, gold)
            # base_ms already includes the real (always-on) rerank call this pass made; the
            # OFF-equivalent latency is base_ms minus the measured rerank call time, floored at 0
            # -- what this query would have cost had the gate never called the model at all.
            off_ms = max(base_ms - rec["on_ms"], 0.0)
            per_query.append(
                {
                    "margin": rec["margin"],
                    "off_in_context": off_in_context,
                    "on_in_context": on_in_context,
                    "off_ms": off_ms,
                    "on_ms": base_ms,  # the real measured always-on latency for this query
                }
            )

    rg_mod.AdaptiveRerankGate.apply = orig_apply  # type: ignore[method-assign]

    n = len(per_query)
    always_off = sum(1 for r in per_query if r["off_in_context"]) / n
    always_on = sum(1 for r in per_query if r["on_in_context"]) / n
    always_off_lat = sorted(r["off_ms"] for r in per_query)
    always_on_lat = sorted(r["on_ms"] for r in per_query)

    out: dict[str, Any] = {
        "n_queries": n,
        "always_off_gic": round(always_off * 100, 2),
        "always_on_gic": round(always_on * 100, 2),
        "always_off_latency_ms": {
            "mean": round(statistics.mean(always_off_lat), 1),
            "p50": round(_pctl(always_off_lat, 0.5), 1),
            "p95": round(_pctl(always_off_lat, 0.95), 1),
        },
        "always_on_latency_ms": {
            "mean": round(statistics.mean(always_on_lat), 1),
            "p50": round(_pctl(always_on_lat, 0.5), 1),
            "p95": round(_pctl(always_on_lat, 0.95), 1),
        },
        "margin_distribution": {
            "mean": round(statistics.mean(r["margin"] for r in per_query), 4),
            "p10": round(_pctl([r["margin"] for r in per_query], 0.10), 4),
            "p25": round(_pctl([r["margin"] for r in per_query], 0.25), 4),
            "p50": round(_pctl([r["margin"] for r in per_query], 0.50), 4),
            "p75": round(_pctl([r["margin"] for r in per_query], 0.75), 4),
            "p90": round(_pctl([r["margin"] for r in per_query], 0.90), 4),
        },
        "thresholds": {},
    }
    for th in thresholds:
        triggered = [r for r in per_query if r["margin"] <= th]
        not_triggered = [r for r in per_query if r["margin"] > th]
        gic = (
            sum(
                (r["on_in_context"] if r["margin"] <= th else r["off_in_context"])
                for r in per_query
            )
            / n
        )
        mixed_lat = sorted((r["on_ms"] if r["margin"] <= th else r["off_ms"]) for r in per_query)
        out["thresholds"][str(th)] = {
            "trigger_rate": round(len(triggered) / n, 3),
            "n_triggered": len(triggered),
            "gold_in_context": round(gic * 100, 2),
            "delta_pp_vs_always_off": round((gic - always_off) * 100, 2),
            "delta_pp_vs_always_on": round((gic - always_on) * 100, 2),
            "mixed_latency_ms": {
                "mean": round(statistics.mean(mixed_lat), 1),
                "p50": round(_pctl(mixed_lat, 0.5), 1),
                "p95": round(_pctl(mixed_lat, 0.95), 1),
            },
        }
    return out


async def main() -> int:
    dataset = os.environ["AD307_DATASET"]
    limit = int(os.environ.get("AD307_LIMIT", "10"))
    imp = float(os.environ.get("AD307_IMPORTANCE", "0.9"))
    out_path = Path(os.environ["AD307_OUT"])
    thresholds = [
        float(x) for x in os.environ.get("AD307_THRESHOLDS", "0.1,0.2,0.3,0.5").split(",")
    ]

    result = await run_policy(dataset=dataset, limit=limit, importance=imp, thresholds=thresholds)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
