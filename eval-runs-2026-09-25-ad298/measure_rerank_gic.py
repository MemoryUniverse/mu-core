"""AD-298 (ADR 0078): honest gold_in_context measurement with the rerank gate actually live for
the first time (`ModelRouter.rerank` -> Infinity/`BAAI/bge-reranker-v2-m3` on mu-dev-vm:8080).

Same method as `docs/tracking/eval-runs/2026-09-03/gic_measure.py` (ZERO LLM calls — the metric
the accuracy attribution actually turns on, `gold_in_context`, is a retrieval fact joined via
`runner.gold_ids_present`), extended to loop conv-26/conv-30/conv-41 (the first three LoCoMo10
samples in file order — `load_locomo(path, samples=3)` returns exactly these, verified against
the dataset on this box before writing this file) in ONE process so the two arms (rerank off/on)
share the same corpus loop, and to record PER-CALL WALL LATENCY on every `recall()` call so the
cost side of the AD-298 question (p50/p95 against the 150ms `recall_e2e_rerank` budget,
`language-analysis-server.md`) is measured in the SAME pass as the gain, never a separate run that
could silently disagree on config.

WIDTH IS MATCHED BY CONSTRUCTION: `limit` is identical in both arms (the shipped default, 10).
The reranker only reorders/prunes the upstream fused pool (`rerank_pool_size=20` default,
independent of `limit`, `rerank_gate.py`'s own "Pool width vs limit" section) before the SAME
`_merge_floor` truncation to `limit` runs — so this is exactly the "retrieve wide, rerank, cut to
the window, compare the window" comparison the lane brief asks for, with no separate wide-pool
flag needed: it is how `ThreeChannelRecallRanker.rank()` already works.

Toggled ONLY by `MU_RECALL__RERANK_ENABLED` (env, read fresh per arm via `get_engine_settings()`
inside `one_rep`, mirroring `gic_measure.py`'s own "read what actually ran" discipline) — nothing
else in the recall config changes between arms.
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

# Runs from OUTSIDE the mu-core checkout on purpose (AD-298 measurement note): this script lives
# in a stable path so a concurrent lane's rsync of mu-core (root CLAUDE.md rule 13's own warning
# — verified live during this run: an untracked file dropped directly into ~/mu_project/mu-core
# was deleted mid-session by something else's sync) cannot delete it out from under a long rerank
# run. `AD298_EVAL_DIR` points at the real `eval/` package inside whichever mu-core checkout is
# current; defaults to the co-located `./eval` for the ordinary same-directory case.
_eval_dir = os.environ.get("AD298_EVAL_DIR")
sys.path.insert(0, _eval_dir if _eval_dir else str(Path(__file__).resolve().parent / "eval"))

from mu_eval.corpus import ingest_conversation, local_memory_for  # noqa: E402
from mu_eval.locomo import LabelledQuery, load_locomo  # noqa: E402
from mu_eval.runner import _await_index, gold_ids_present  # noqa: E402


async def one_conversation(
    conversation: Any, *, limit: int, importance: float, max_queries: int | None = None
) -> dict[str, Any]:
    user, session = "evaluser", "evalsession"
    run_id = f"ad298{uuid.uuid4().hex[:6]}"
    scored = 0
    in_context = 0
    latencies_ms: list[float] = []
    async with local_memory_for(conversation, run_id=run_id) as opaque:
        memory: Any = opaque
        index, report = await ingest_conversation(
            memory, conversation, user=user, session=session, importance=importance
        )
        visible = await _await_index(
            memory, conversation.turns[0].text[:120], user=user, session=session
        )
        known = {t.dia_id for t in conversation.turns}
        queries: list[LabelledQuery] = list(conversation.queries)
        for q in queries:
            if q.is_adversarial or not q.evidence:
                continue
            gold = {e for e in q.evidence if e in known}
            if not gold:
                continue
            if max_queries is not None and scored >= max_queries:
                # Deterministic FILE-ORDER cap, not a cherry-pick — applied identically to both
                # arms via the SAME env var, so `off` and `on` score the SAME query subset
                # (paired comparison preserved). Exists only to bound wall-clock time against the
                # rerank arm's measured ~4-8s/call CPU cost (see report); recorded honestly in the
                # output (`queries_scored`) rather than silently passed off as the full sample.
                break
            t0 = time.perf_counter()
            result = await memory.recall(q.question, user=user, session=session, limit=limit)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            hit = gold_ids_present(result.items, index, gold)
            scored += 1
            in_context += int(hit)
    return {
        "sample_id": conversation.sample_id,
        "corpus_turns": report.turns_written,
        "index_visible": visible,
        "queries_scored": scored,
        "gold_in_context": in_context,
        "gold_in_context_rate": in_context / scored if scored else 0.0,
        "latencies_ms": latencies_ms,
    }


def _pctl(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


async def one_arm(*, label: str, dataset: str, limit: int, importance: float) -> dict[str, Any]:
    from mu_engine.config import get_engine_settings

    # AD298_SAMPLES lets the caller run the cost-discipline-mandated escalation (root CLAUDE.md /
    # this lane's brief): conv-26 alone (150 scoreable) as the quick arm, conv-26/30/41 (383) only
    # once that looks real. `load_locomo` returns LoCoMo10's conversations in FILE order, so
    # samples=1 is deterministically conv-26 alone — verified against the dataset, not assumed.
    n_samples = int(os.environ.get("AD298_SAMPLES", "3"))
    conversations = load_locomo(dataset, samples=n_samples)
    assert [c.sample_id for c in conversations] == ["conv-26", "conv-30", "conv-41"][:n_samples], (
        f"expected the first {n_samples} of conv-26/30/41, got {[c.sample_id for c in conversations]}"
    )

    recall_cfg = get_engine_settings().recall  # read AFTER env is set for this arm (see __main__)
    max_per_conv_env = os.environ.get("AD298_MAX_PER_CONV")
    max_per_conv = int(max_per_conv_env) if max_per_conv_env else None
    per_conv = []
    for conv in conversations:
        r = await one_conversation(
            conv, limit=limit, importance=importance, max_queries=max_per_conv
        )
        print(
            f"  [{label}] {r['sample_id']}: gold_in_context {r['gold_in_context']}/"
            f"{r['queries_scored']} = {r['gold_in_context_rate']:.4f}  "
            f"latency p50={statistics.median(r['latencies_ms']):.0f}ms "
            f"max={max(r['latencies_ms']):.0f}ms" if r["latencies_ms"] else "n/a",
            flush=True,
        )
        per_conv.append(r)

    total_scored = sum(r["queries_scored"] for r in per_conv)
    total_hit = sum(r["gold_in_context"] for r in per_conv)
    all_lat = sorted(x for r in per_conv for x in r["latencies_ms"])

    return {
        "label": label,
        "limit": limit,
        "rerank_enabled": recall_cfg.rerank_enabled,
        "rerank_pool_size": getattr(recall_cfg, "rerank_pool_size", None),
        "rerank_min_score": getattr(recall_cfg, "rerank_min_score", None),
        "rerank_top_fraction": getattr(recall_cfg, "rerank_top_fraction", None),
        "sparse_enabled": recall_cfg.sparse_enabled,
        "per_conversation": per_conv,
        "queries_scored": total_scored,
        "gold_in_context": total_hit,
        "gold_in_context_rate": total_hit / total_scored if total_scored else 0.0,
        "latency_ms": {
            "n": len(all_lat),
            "p50": round(_pctl(all_lat, 0.50), 1),
            "p95": round(_pctl(all_lat, 0.95), 1),
            "max": round(all_lat[-1], 1) if all_lat else 0.0,
            "min": round(all_lat[0], 1) if all_lat else 0.0,
        },
    }


async def main() -> int:
    from mu_engine.config import get_engine_settings

    dataset = os.environ["AD298_DATASET"]  # e.g. ~/mu_eval_data/locomo10.json
    limit = int(os.environ.get("AD298_LIMIT", "10"))
    importance = float(os.environ.get("AD298_IMPORTANCE", "0.9"))
    out_path = Path(os.environ["AD298_OUT"])

    # Arm 1: rerank OFF (shipped default) — establishes the baseline this run's own env produces
    # (not borrowed from a different day's gic_dense.json), so the delta is apples-to-apples.
    #
    # BUG FOUND BY RUNNING (root CLAUDE.md: "verify by running, never by reading" — a smoke-test
    # run of this exact file, before this fix, printed `"on": {"rerank_enabled": false, ...}`,
    # i.e. BOTH arms were dark): `get_engine_settings` is `@lru_cache`-d
    # (`mu_engine/config/engine_settings.py:120`), so the ENV mutation below is invisible to the
    # SECOND `one_arm` call unless the cache is cleared first — the exact contract
    # `test_plane_model_wiring_int.py`'s own `keyless` fixture already documents
    # ("Tests that need a fresh read after mutating os.environ must call
    # get_engine_settings.cache_clear() first"). This file's original version read the env AFTER
    # setting it (this module's own docstring: "read fresh per arm") but never cleared the cache
    # that makes "fresh" true, so `one_arm`'s `recall_cfg = get_engine_settings().recall` silently
    # returned the FIRST arm's cached settings both times. `.cache_clear()` after every mutation.
    os.environ["MU_RECALL__RERANK_ENABLED"] = "false"
    get_engine_settings.cache_clear()
    off = await one_arm(label="rerank_off", dataset=dataset, limit=limit, importance=importance)

    # Arm 2: rerank ON — the arm that has NEVER run before this pass (AD-298's whole point).
    os.environ["MU_RECALL__RERANK_ENABLED"] = "true"
    get_engine_settings.cache_clear()
    on = await one_arm(label="rerank_on", dataset=dataset, limit=limit, importance=importance)
    assert off["rerank_enabled"] is False and on["rerank_enabled"] is True, (
        "the arms did not actually toggle — see the cache_clear() note above",
        off["rerank_enabled"],
        on["rerank_enabled"],
    )

    out = {
        "dataset": dataset,
        "conversations": ["conv-26", "conv-30", "conv-41"],
        "off": off,
        "on": on,
        "delta_gold_in_context_rate_pp": round(
            (on["gold_in_context_rate"] - off["gold_in_context_rate"]) * 100, 2
        ),
    }
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("off", "on")}, indent=1))
    print("OFF:", json.dumps({k: v for k, v in off.items() if k != "per_conversation"}, indent=1))
    print("ON: ", json.dumps({k: v for k, v in on.items() if k != "per_conversation"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
