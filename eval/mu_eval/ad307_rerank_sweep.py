"""AD-307 — rerank latency/accuracy sweep: model swap x pool size, matched-width (reorder-only),
gold_in_context, zero LLM calls, $0. Same conv-26 sample, same 150-query methodology as AD-301's
verify_rerank_width.py (docs/tracking/eval-runs/2026-09-25-ad301/), reused rather than
reinvented (CLAUDE.md rule 1's spirit applied to our own prior work).

Arms, each reorder-only (rerank_min_score=0.0, rerank_top_fraction=0.0 — AD-301: cutoff cost
-8.00pp, reorder-only gained +6.67pp; never reintroduce a cutoff):
  A_off                  rerank disabled (baseline, reproduces AD-301 arm A)
  B_bge_k20 (shipped)    BAAI/bge-reranker-v2-m3, pool=20 (reproduces AD-301 arm C)
  C_minilm_k20           cross-encoder/ms-marco-MiniLM-L-6-v2, pool=20
  D_minilm_k10           cross-encoder/ms-marco-MiniLM-L-6-v2, pool=10
  E_bge_k10              BAAI/bge-reranker-v2-m3, pool=10          (isolates pool-size alone)

MEASUREMENT FLOOR: AD-301/AD-303 establish +/-1.3pp on 10 rows of 150 as the gold_in_context
noise floor on this sample — nothing inside that is claimed as a real delta.
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


async def one_conversation(conversation: Any, *, limit: int, importance: float) -> dict[str, Any]:
    user, session = "evaluser", "evalsession"
    run_id = f"ad307{uuid.uuid4().hex[:6]}"
    scored = in_context = 0
    lat: list[float] = []
    widths: list[int] = []
    reranked_counts: list[int] = []
    async with local_memory_for(conversation, run_id=run_id) as opaque:
        memory: Any = opaque
        index, report = await ingest_conversation(
            memory, conversation, user=user, session=session, importance=importance
        )
        visible = await _await_index(
            memory, conversation.turns[0].text[:120], user=user, session=session
        )
        known = {t.dia_id for t in conversation.turns}
        for q in list(conversation.queries):
            if q.is_adversarial or not q.evidence:
                continue
            gold = {e for e in q.evidence if e in known}
            if not gold:
                continue
            t0 = time.perf_counter()
            result = await memory.recall(q.question, user=user, session=session, limit=limit)
            lat.append((time.perf_counter() - t0) * 1000.0)
            widths.append(len(result.items))
            reranked_counts.append(
                sum(1 for it in result.items if getattr(it, "rerank_score", None) is not None)
            )
            in_context += int(gold_ids_present(result.items, index, gold))
            scored += 1
    return {
        "sample_id": conversation.sample_id,
        "corpus_turns": report.turns_written,
        "index_visible": visible,
        "queries_scored": scored,
        "gold_in_context": in_context,
        "gold_in_context_rate": in_context / scored if scored else 0.0,
        "width_mean": round(statistics.mean(widths), 3) if widths else 0.0,
        "queries_with_any_rerank_score": sum(1 for c in reranked_counts if c > 0),
        "reranked_items_mean": round(statistics.mean(reranked_counts), 3)
        if reranked_counts
        else 0.0,
        "latencies_ms": lat,
    }


def _pctl(v: list[float], p: float) -> float:
    if not v:
        return 0.0
    k = (len(v) - 1) * p
    f, c = int(k), min(int(k) + 1, len(v) - 1)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


async def one_arm(
    *, label: str, env: dict[str, str], dataset: str, limit: int, importance: float
) -> dict[str, Any]:
    from mu_engine.config import get_engine_settings

    # Full reset of every knob this sweep touches so arm N+1 never inherits arm N's env (the
    # AD-301 script only ever set rerank_enabled/min_score/top_fraction; this one also swaps the
    # model + api_base + pool_size, so a stale value surviving between arms would silently mix
    # them).
    for k in (
        "MU_RECALL__RERANK_ENABLED",
        "MU_RECALL__RERANK_MIN_SCORE",
        "MU_RECALL__RERANK_TOP_FRACTION",
        "MU_RECALL__RERANK_POOL_SIZE",
        "MU_MODEL_CATALOG__SHIPPED__LOCAL_EMBED_RERANK_API_BASE",
        "MU_MODEL_CATALOG__SHIPPED__LOCAL_RERANK_MODEL",
    ):
        os.environ.pop(k, None)
    for k, val in env.items():
        os.environ[k] = val
    get_engine_settings.cache_clear()
    cfg = get_engine_settings().recall
    cat = get_engine_settings().model_catalog.shipped
    convs = load_locomo(dataset, samples=int(os.environ.get("AD307_SAMPLES", "1")))
    per = []
    for conv in convs:
        r = await one_conversation(conv, limit=limit, importance=importance)
        print(
            f"  [{label}] {r['sample_id']}: gic {r['gold_in_context']}/{r['queries_scored']}"
            f" = {r['gold_in_context_rate']:.4f}"
            f" | queries_with_rerank_score={r['queries_with_any_rerank_score']}"
            f" | p50={statistics.median(r['latencies_ms']):.0f}ms"
            f" p95={_pctl(sorted(r['latencies_ms']), 0.95):.0f}ms",
            flush=True,
        )
        per.append(r)
    tot_s = sum(r["queries_scored"] for r in per)
    tot_h = sum(r["gold_in_context"] for r in per)
    all_lat = sorted(x for r in per for x in r["latencies_ms"])
    return {
        "label": label,
        "effective_config": {
            "rerank_enabled": cfg.rerank_enabled,
            "rerank_min_score": cfg.rerank_min_score,
            "rerank_top_fraction": cfg.rerank_top_fraction,
            "rerank_pool_size": cfg.rerank_pool_size,
            "rerank_model": cat.local_rerank_model,
            "rerank_api_base": cat.local_embed_rerank_api_base,
            "limit": limit,
        },
        "per_conversation": per,
        "queries_scored": tot_s,
        "gold_in_context": tot_h,
        "gold_in_context_rate": tot_h / tot_s if tot_s else 0.0,
        "latency_ms": {
            "n": len(all_lat),
            "p50": round(_pctl(all_lat, 0.50), 1),
            "p95": round(_pctl(all_lat, 0.95), 1),
            "max": round(all_lat[-1], 1) if all_lat else 0.0,
        },
    }


async def main() -> int:
    dataset = os.environ["AD307_DATASET"]
    limit = int(os.environ.get("AD307_LIMIT", "10"))
    imp = float(os.environ.get("AD307_IMPORTANCE", "0.9"))
    out = Path(os.environ["AD307_OUT"])
    minilm_base = os.environ.get("AD307_MINILM_BASE", "http://127.0.0.1:8081/v1")
    minilm_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    reorder_only = {"MU_RECALL__RERANK_MIN_SCORE": "0.0", "MU_RECALL__RERANK_TOP_FRACTION": "0.0"}

    arm_specs = [
        ("A_off", {"MU_RECALL__RERANK_ENABLED": "false"}),
        (
            "B_bge_k20_shipped",
            {
                "MU_RECALL__RERANK_ENABLED": "true",
                **reorder_only,
                "MU_RECALL__RERANK_POOL_SIZE": "20",
            },
        ),
        (
            "C_minilm_k20",
            {
                "MU_RECALL__RERANK_ENABLED": "true",
                **reorder_only,
                "MU_RECALL__RERANK_POOL_SIZE": "20",
                "MU_MODEL_CATALOG__SHIPPED__LOCAL_EMBED_RERANK_API_BASE": minilm_base,
                "MU_MODEL_CATALOG__SHIPPED__LOCAL_RERANK_MODEL": minilm_model,
            },
        ),
        (
            "D_minilm_k10",
            {
                "MU_RECALL__RERANK_ENABLED": "true",
                **reorder_only,
                "MU_RECALL__RERANK_POOL_SIZE": "10",
                "MU_MODEL_CATALOG__SHIPPED__LOCAL_EMBED_RERANK_API_BASE": minilm_base,
                "MU_MODEL_CATALOG__SHIPPED__LOCAL_RERANK_MODEL": minilm_model,
            },
        ),
        (
            "E_bge_k10",
            {
                "MU_RECALL__RERANK_ENABLED": "true",
                **reorder_only,
                "MU_RECALL__RERANK_POOL_SIZE": "10",
            },
        ),
        # F/G: the middle multilingual option (BAAI/bge-reranker-base, ~278M, same XLM-R family
        # as the shipped v2-m3 -> multilingual capability retained, ~2.75x cheaper per raw-HTTP
        # profile) and its int8-quantized variant (torch.quantization.quantize_dynamic() via
        # Infinity's --dtype int8 -- measured SLOWER than fp32 for v2-m3 on this AMD EPYC 7B12
        # box, no AVX-512/VNNI, so this arm exists to check whether the smaller base model
        # changes that verdict, not to assume it does).
        (
            "F_bgebase_k20",
            {
                "MU_RECALL__RERANK_ENABLED": "true",
                **reorder_only,
                "MU_RECALL__RERANK_POOL_SIZE": "20",
                "MU_MODEL_CATALOG__SHIPPED__LOCAL_EMBED_RERANK_API_BASE": os.environ.get(
                    "AD307_BGEBASE_BASE", "http://127.0.0.1:8083/v1"
                ),
                "MU_MODEL_CATALOG__SHIPPED__LOCAL_RERANK_MODEL": "BAAI/bge-reranker-base",
            },
        ),
        # G: MiniLM served by Infinity's `optimum` (ONNX Runtime) engine instead of `torch` --
        # raw HTTP measured 104ms vs 139ms p50 at n=20 (~25% win, same model/weights, engine
        # swap only). ONNX engine could not be measured for the bge-* RoBERTa-family models:
        # Infinity's own warmup is hardcoded to `n_tokens=512` (select_model.py:101) and
        # RoBERTa's position-embedding table only accepts [-514,513], so the warmup itself
        # crashes with an out-of-bounds Gather before the server ever starts -- an Infinity
        # engine defect for this model family, not a mu-core issue, and not reachable from any
        # CLI flag this build exposes.
        (
            "G_minilm_onnx_k20",
            {
                "MU_RECALL__RERANK_ENABLED": "true",
                **reorder_only,
                "MU_RECALL__RERANK_POOL_SIZE": "20",
                "MU_MODEL_CATALOG__SHIPPED__LOCAL_EMBED_RERANK_API_BASE": os.environ.get(
                    "AD307_MINILM_ONNX_BASE", "http://127.0.0.1:8085/v1"
                ),
                "MU_MODEL_CATALOG__SHIPPED__LOCAL_RERANK_MODEL": minilm_model,
            },
        ),
    ]
    only = os.environ.get("AD307_ARMS")
    if only:
        wanted = set(only.split(","))
        arm_specs = [a for a in arm_specs if a[0] in wanted]

    arms = []
    for label, env in arm_specs:
        arms.append(
            await one_arm(label=label, env=env, dataset=dataset, limit=limit, importance=imp)
        )

    base = next((a["gold_in_context_rate"] for a in arms if a["label"] == "A_off"), None)
    doc = {
        "dataset": dataset,
        "samples": os.environ.get("AD307_SAMPLES", "1"),
        "arms": arms,
        "deltas_pp_vs_A_off": (
            {a["label"]: round((a["gold_in_context_rate"] - base) * 100, 2) for a in arms}
            if base is not None
            else {}
        ),
    }
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    for a in arms:
        print(
            a["label"],
            "gic",
            round(a["gold_in_context_rate"] * 100, 2),
            "lat_p50",
            a["latency_ms"]["p50"],
            "lat_p95",
            a["latency_ms"]["p95"],
            "cfg",
            a["effective_config"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
