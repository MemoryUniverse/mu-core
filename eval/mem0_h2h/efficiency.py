"""AD-306 — token + latency measurement for MU, on the same axes mem0 publishes.

mem0's own README names four things: retrieval, temporal reasoning, "**Token Consumption**:
Number of tokens required to generate final answer", and "**Latency**: Time required during
search and to generate response" (`other_repos/mem0/evaluation/README.md:171-172`). This script
is those two axes, plus ingestion cost, for MU, on `conv-26` — the sample the rest of this
project's mem0 comparison already uses (AD-303/AD-304/AD-305).

MEM0'S OWN DEFINITIONS, READ FROM THEIR SOURCE (not guessed — CODE-ADOPTION-METHODOLOGY rule 1):
  * **search time** — wall clock around `self.mem0_client.search(...)`, timed
    `start_time = time.time()` / `end_time = time.time()`
    (`other_repos/mem0/evaluation/src/memzero/search.py:38,64`). Their store is per-speaker, so
    their harness pays this TWICE per question and sums it
    (`speaker_1_memory_time + speaker_2_memory_time`, `search.py:91-96`). MU has one partition per
    conversation, so MU's "search" is the single `recall()` call — the honest per-system
    equivalent, not a re-shaped copy of their two-call shape.
  * **response_time** — wall clock around the answer-generation chat completion,
    `t1 = time.time()` / `response_time = t2 - t1` (`search.py:112-116`).
  * **total** — search + response, both timed SEQUENTIALLY, one question at a time: their default
    driver is a plain nested loop with no thread pool
    (`process_data_file`, `search.py:171-193`; the `ThreadPoolExecutor` variant right below it
    defaults `max_workers=1`, `search.py:198` — still effectively sequential). This script
    matches that: every timed call in `measure_total_latency` runs at concurrency=1.
  * **"Token Consumption"** — read literally, the tokens in the prompt that produces the final
    answer. Their own token-counting code elsewhere in this repo uses `tiktoken`
    (`other_repos/mem0/evaluation/src/rag.py:119`,
    `encoding = tiktoken.encoding_for_model(...)`) — the same real-tokenizer approach this script
    uses, generalized to each system's OWN answering model
    (`tiktoken.encoding_for_model("gpt-5")` resolves to `o200k_base`, verified live in this
    session — gpt-5 is already in tiktoken 0.13.0's model map, no proxy/estimate needed).

WHAT THIS SCRIPT DOES NOT DO: re-run mem0's own pipeline. Their ingest cost was already measured
for real (`docs/tracking/eval-runs/2026-09-25-h2h-mem0/mem0_ingest_usage.json`: 96 calls,
842,508 prompt / 321,845 completion tokens, $4.2716) and their published Table 1 numbers
(arXiv 2504.19413) are read as the target, not reproduced — different judge, different corpus
size, different model (gpt-4o-mini vs this project's gpt-5). Every place this script's output is
compared to theirs, the comparison is marked NOT DIRECTLY COMPARABLE.

TWO WIDTHS, PER CLAUDE.md's brief: `10` (`DEFAULT_RECALL_LIMIT`,
`mu_contracts/contracts/defaults.py:45` — what MU actually ships) and `20` (mem0's own harness
width: `top_k=10` per speaker, both speakers concatenated, `search.py:19,91-96` — the width the
AD-303 head-to-head judged comparison used, so the two are directly readable against it).

RERANK IS AN ENV-TIME KNOB (`MU_RECALL__RERANK_ENABLED`, read once when `LocalMemory`/
`get_engine_settings()` first constructs in this process — see `services/recall/dto.py:698`), so
"rerank off" and "rerank on" are two SEPARATE process invocations of this script, not one script
flipping the env mid-run (same reason `ours_gic_sweep_rerank_off.txt` / `_reorder.txt` are two
files from two runs). `vm_run.sh` is called twice; see the ADR for the exact commands.

COST DISCIPLINE (CLAUDE.md: "~$5 of the owner's $20 remains"). Search latency and every token
count above are FREE — no LLM call, `memory.recall()` and `tiktoken.encode()` only. The ONLY paid
arm is `measure_total_latency`, gated behind `--total-latency-n` (0 = skip) and a hard
`Budget` cap (reused from `answer_h2h.py`, which aborts the run the instant measured spend
crosses the cap — not an estimate). No judge call is made here; accuracy is already measured
(AD-303/AD-304/AD-305) and this script only needs wall-clock time, which the answer call alone
supplies.
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

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import tiktoken  # noqa: E402
from mu_eval.answer_quality import ANSWER_SYSTEM_PROMPT, _complete_with_retry  # noqa: E402
from mu_eval.corpus import ingest_conversation, local_memory_for  # noqa: E402
from mu_eval.judge import answer_prompt  # noqa: E402
from mu_eval.locomo import load_locomo  # noqa: E402
from mu_eval.openai_chat import OpenAICompatChat  # noqa: E402
from mu_eval.runner import _await_index, gold_ids_present  # noqa: E402

from mem0_h2h import emit  # noqa: E402
from mem0_h2h.answer_h2h import Budget, BudgetExceededError, MeteredChat, _usage_of  # noqa: E402

# --------------------------------------------------------------------------------- distributions


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (same definition numpy's default `interpolation="linear"`
    uses) — reported alongside mean/min/max rather than a bare mean, per the brief ("report the
    distribution rather than only a mean")."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "p50": round(_percentile(values, 0.50), 2),
        "p95": round(_percentile(values, 0.95), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def _context_of(result_items: Any, index: Any, turn_date: dict[str, str]) -> str:
    """Byte-identical join to `answer_quality.py:538-540` / `ours_arm.py`'s own export ("- {date}:
    {body}", session date re-attached by normalized-body join, since `RecallItemView` carries no
    per-item timestamp) — the SAME context a real answer-quality run would build, not a second
    rendering that could drift from it."""
    lines = []
    for item in result_items:
        dia = index.resolve(item.content)
        stamp = turn_date.get(dia[0], "") if dia else ""
        lines.append(f"- {stamp}: {item.content}" if stamp else f"- {item.content}")
    return "\n".join(lines) or "(no memories retrieved)"


def _eligible_rows(conversation: Any) -> list[tuple[Any, set[str]]]:
    known = {t.dia_id for t in conversation.turns}
    out = []
    for q in conversation.queries:
        if q.is_adversarial or not q.evidence:
            continue
        gold = {e for e in q.evidence if e in known}
        if gold:
            out.append((q, gold))
    return out


# --------------------------------------------------------------------------- FREE: search + tokens


async def sweep_search_and_tokens(
    conversation: Any, *, limits: list[int], importance: float, label: str, model: str
) -> dict[str, Any]:
    """FREE — no LLM call. Search latency (p50/p95) + context-token / full-prompt-token
    distributions, per width, under ONE ingest (rerank is fixed at process start; width is a
    per-call `recall()` argument, so every width in `limits` is measured against the SAME
    populated index — `ours_arm.py`'s own reasoning, ported)."""
    from mu_engine.config import get_engine_settings

    enc = tiktoken.encoding_for_model(model)
    cfg = get_engine_settings().recall
    user, session = "evaluser", "evalsession"
    run_id = f"eff{uuid.uuid4().hex[:6]}"
    rows = _eligible_rows(conversation)
    turn_date = {t.dia_id: t.session_date for t in conversation.turns}

    by_limit: dict[str, Any] = {}
    async with local_memory_for(conversation, run_id=run_id) as opaque:
        memory: Any = opaque
        t_ingest0 = time.perf_counter()
        index, report = await ingest_conversation(
            memory, conversation, user=user, session=session, importance=importance
        )
        ingest_ms = (time.perf_counter() - t_ingest0) * 1000.0
        await _await_index(memory, conversation.turns[0].text[:120], user=user, session=session)

        for limit in limits:
            lat_ms: list[float] = []
            ctx_tokens: list[float] = []
            prompt_tokens: list[float] = []
            gic_hits = 0
            for query, gold in rows:
                t0 = time.perf_counter()
                result = await memory.recall(
                    query.question, user=user, session=session, limit=limit
                )
                lat_ms.append((time.perf_counter() - t0) * 1000.0)
                gic_hits += int(gold_ids_present(result.items, index, gold))
                context = _context_of(result.items, index, turn_date)
                ctx_tokens.append(float(len(enc.encode(context))))
                full_prompt = answer_prompt(context=context, question=query.question)
                prompt_tokens.append(
                    float(len(enc.encode(full_prompt)) + len(enc.encode(ANSWER_SYSTEM_PROMPT)))
                )
            by_limit[str(limit)] = {
                "limit": limit,
                "queries": len(rows),
                "gold_in_context_rate": round(gic_hits / len(rows), 4) if rows else 0.0,
                "search_latency_ms": _dist(lat_ms),
                "context_tokens": _dist(ctx_tokens),
                "full_answer_prompt_tokens": _dist(prompt_tokens),
            }
            d = by_limit[str(limit)]
            emit(
                f"  [{label}] limit={limit:>3} "
                f"search p50={d['search_latency_ms']['p50']:.0f}ms "
                f"p95={d['search_latency_ms']['p95']:.0f}ms  "
                f"ctx_tok mean={d['context_tokens']['mean']:.0f} "
                f"p95={d['context_tokens']['p95']:.0f}  "
                f"prompt_tok mean={d['full_answer_prompt_tokens']['mean']:.0f}"
            )

    return {
        "label": label,
        "sample_id": conversation.sample_id,
        "corpus_turns": report.turns_written,
        "ingest_wall_ms": round(ingest_ms, 1),
        "ingest_llm_calls": 0,
        "ingest_llm_prompt_tokens": 0,
        "ingest_llm_completion_tokens": 0,
        "_ingest_note": (
            "consolidate=False (this run's default, matching run_answer_quality's own default) "
            "-- the write path makes zero LLM calls: no add()-time extraction call exists in this "
            "engine at all, unlike mem0's per-batch extraction call on every add(). If consolidate "
            "is later enabled for this measurement, LTM distillation DOES call the local model "
            "router and this field would need to change from a constant to a measurement."
        ),
        "effective_config": {
            "rerank_enabled": cfg.rerank_enabled,
            "rerank_min_score": cfg.rerank_min_score,
            "rerank_top_fraction": cfg.rerank_top_fraction,
            "rerank_pool_size": cfg.rerank_pool_size,
            "channel_pool_size": cfg.channel_pool_size,
        },
        "tokenizer": {"model": model, "encoding": enc.name},
        "by_limit": by_limit,
    }


# --------------------------------------------------------------------------- PAID: total latency


async def measure_total_latency(
    conversation: Any,
    *,
    limit: int,
    importance: float,
    chat: Any,
    budget: Budget,
    n_queries: int,
    answer_max_tokens: int,
) -> dict[str, Any]:
    """PAID — real answer-generation calls, no judge. Sequential (concurrency=1), matching mem0's
    own `process_data_file` (`search.py:171-193`). `total_ms = search_ms + answer_ms` per row,
    mirroring their `response_time` addition (`search.py:114`, consumed at the aggregate level in
    their published table)."""
    run_id = f"efftot{uuid.uuid4().hex[:6]}"
    user, session = "evaluser", "evalsession"
    rows = [q for q, _gold in _eligible_rows(conversation)]
    if n_queries:
        rows = rows[:n_queries]
    turn_date = {t.dia_id: t.session_date for t in conversation.turns}

    search_ms: list[float] = []
    answer_ms: list[float] = []
    total_ms: list[float] = []
    real_prompt_tokens: list[float] = []
    per_row: list[dict[str, Any]] = []
    aborted_after = None

    async with local_memory_for(conversation, run_id=run_id) as opaque:
        memory: Any = opaque
        index, _report = await ingest_conversation(
            memory, conversation, user=user, session=session, importance=importance
        )
        await _await_index(memory, conversation.turns[0].text[:120], user=user, session=session)

        for i, query in enumerate(rows):
            t0 = time.perf_counter()
            result = await memory.recall(query.question, user=user, session=session, limit=limit)
            t1 = time.perf_counter()
            context = _context_of(result.items, index, turn_date)
            try:
                answer = await _complete_with_retry(
                    chat,
                    system=ANSWER_SYSTEM_PROMPT,
                    user=answer_prompt(context=context, question=query.question),
                    max_tokens=answer_max_tokens,
                )
            except BudgetExceededError as exc:
                aborted_after = i
                emit(f"  !! budget cap hit after {i} rows: {exc}")
                break
            t2 = time.perf_counter()
            s_ms = (t1 - t0) * 1000.0
            a_ms = (t2 - t1) * 1000.0
            search_ms.append(s_ms)
            answer_ms.append(a_ms)
            total_ms.append(s_ms + a_ms)
            usage = _usage_of(answer)
            if usage and usage.get("prompt_tokens"):
                real_prompt_tokens.append(float(usage["prompt_tokens"]))
            per_row.append(
                {
                    "query_id": query.query_id,
                    "search_ms": round(s_ms, 1),
                    "answer_ms": round(a_ms, 1),
                    "total_ms": round(s_ms + a_ms, 1),
                }
            )
            if (i + 1) % 10 == 0:
                emit(f"  [total-latency] {i + 1}/{len(rows)} cost=${budget.cost_usd:.3f}")

    return {
        "limit": limit,
        "n_requested": len(rows),
        "n_completed": len(total_ms),
        "aborted_after": aborted_after,
        "search_latency_ms": _dist(search_ms),
        "answer_latency_ms": _dist(answer_ms),
        "total_latency_ms": _dist(total_ms),
        "real_answer_prompt_tokens": _dist(real_prompt_tokens),
        "budget": budget.snapshot(),
        "per_row": per_row,
    }


# --------------------------------------------------------------------------------------------- CLI


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default=os.environ.get("H2H_SAMPLE", "conv-26"))
    parser.add_argument(
        "--dataset",
        default=os.environ.get("H2H_DATASET", "/home/user/mu_eval_data/locomo10.json"),
    )
    parser.add_argument("--widths", default="10,20")
    parser.add_argument("--importance", type=float, default=0.9)
    parser.add_argument("--label", default=os.environ.get("H2H_LABEL", "arm"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--base-url", default="https://libo-ai.services.ai.azure.com/openai/v1")
    parser.add_argument("--answer-max-tokens", type=int, default=900)
    parser.add_argument(
        "--total-latency-n",
        type=int,
        default=0,
        help="0 = skip the PAID total-latency arm entirely (default: skip)",
    )
    parser.add_argument("--total-latency-limit", type=int, default=10)
    parser.add_argument("--budget-usd", type=float, default=1.5)
    args = parser.parse_args()

    conversations = load_locomo(args.dataset, samples=None)
    matching = [c for c in conversations if c.sample_id == args.sample]
    if not matching:
        raise SystemExit(f"{args.sample} not in {args.dataset}")
    conv = matching[0]
    widths = [int(x) for x in args.widths.split(",")]

    doc: dict[str, Any] = {"sample_id": args.sample, "label": args.label, "widths": widths}

    emit(f"== FREE: search latency + token distributions, label={args.label} ==")
    doc["search_and_tokens"] = await sweep_search_and_tokens(
        conv, limits=widths, importance=args.importance, label=args.label, model=args.model
    )

    if args.total_latency_n:
        api_key = os.environ.get("MU_EVAL_API_KEY")
        if not api_key:
            raise SystemExit("MU_EVAL_API_KEY is not set -- refusing to run the PAID arm")
        emit(
            f"== PAID: total latency (search+answer), n={args.total_latency_n}, "
            f"limit={args.total_latency_limit}, budget cap=${args.budget_usd:.2f} =="
        )
        chat = OpenAICompatChat(
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            completion_tokens_field="max_completion_tokens",
            omit_temperature=True,
            timeout=300.0,
        )
        budget = Budget(args.budget_usd)
        metered = MeteredChat(chat, budget)
        try:
            doc["total_latency"] = await measure_total_latency(
                conv,
                limit=args.total_latency_limit,
                importance=args.importance,
                chat=metered,
                budget=budget,
                n_queries=args.total_latency_n,
                answer_max_tokens=args.answer_max_tokens,
            )
            doc["total_latency"]["served_models"] = sorted(chat.served_models)
        finally:
            await chat.aclose()

    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    emit(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
