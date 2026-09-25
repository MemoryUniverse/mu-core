"""The decisive head-to-head: ANSWER QUALITY, both systems, one prompt, one judge, one model.

WHY THIS AND NOT `gold_in_context`
----------------------------------
`gold_in_context` is not symmetric between these two systems, and pretending it is would be the
single easiest way to publish a wrong answer. Our arm stores the dialogue turn VERBATIM, so "gold
in context" means the gold turn's own text is in the window — exact. mem0 stores LLM-EXTRACTED
facts, so the only provenance available is "a memory extracted from a batch that CONTAINED the
gold turn was retrieved". Whether the gold fact survived their extraction is not knowable from
that. Their number is therefore a CEILING, ours is exact, and the gap between the two is whatever
their extractor dropped — unmeasured.

Answer quality has no such asymmetry. Each system hands its own retrieved context to the SAME
answering model, under the SAME prompt (mem0's own `ANSWER_PROMPT`, ported verbatim in
`mu_eval/judge.py`), graded by the SAME judge (MemOS's official `locomo_grader` accuracy prompt,
also a verbatim port). Only the retrieval differs. That is the comparison the owner asked for and
it is the metric mem0 itself publishes against.

CONTROLS THIS SCRIPT ENFORCES RATHER THAN ASSUMES
-------------------------------------------------
* Same rows: every arm's context file is keyed by `query_id`; the run intersects them and refuses
  to score a row missing from either. A denominator that silently differs between arms is the
  exact defect the brief warns about.
* Same model, read from the SERVER: `OpenAICompatChat.served_models` accumulates `body["model"]`
  off every response. The run asserts both arms saw the same served model set, so "the A/B ran
  the same model in both arms without noticing" cannot happen here undetected.
* Same judge: one judge client, one prompt builder, one parse.
* A hard budget cap, checked after every call, that aborts the run rather than overspending.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from mu_eval.answer_quality import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
    _complete_with_retry,
)
from mu_eval.judge import (  # noqa: E402
    COMPACT_JUDGE_SYSTEM_PROMPT,
    answer_prompt,
    compact_judge_prompt,
    parse_judgement,
)
from mu_eval.locomo import CATEGORY_NAMES  # noqa: E402
from mu_eval.openai_chat import OpenAICompatChat  # noqa: E402

from mem0_h2h import emit  # noqa: E402

PROMPT_USD_PER_1M = 1.25
COMPLETION_USD_PER_1M = 10.0


class BudgetExceededError(RuntimeError):
    pass


class Budget:
    """Measured, not estimated: priced off each response's own `usage` block as it arrives."""

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.calls = 0
        self._lock = asyncio.Lock()

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * PROMPT_USD_PER_1M
            + self.completion_tokens / 1_000_000 * COMPLETION_USD_PER_1M
        )

    async def observe(self, usage: dict[str, Any] | None) -> None:
        async with self._lock:
            self.calls += 1
            if usage:
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)
                details = usage.get("completion_tokens_details") or {}
                self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
            cost = self.cost_usd
        if cost > self.cap_usd:
            raise BudgetExceededError(
                f"measured spend ${cost:.3f} crossed the ${self.cap_usd:.2f} cap "
                f"after {self.calls} calls"
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }


class MeteredChat:
    """Meters EVERY attempt, including the ones `_complete_with_retry` discards.

    `_complete_with_retry` returns only the LAST attempt's `CompletionResult`, so pricing off the
    returned object would silently omit the cost of every budget-exhausted attempt that preceded
    it — on a reasoning model those are the expensive ones (they burned the whole completion cap
    on hidden reasoning and returned nothing). Wrapping the client instead of the result means the
    bill is counted where it is actually incurred.
    """

    def __init__(self, inner: Any, budget: Budget) -> None:
        self._inner = inner
        self._budget = budget

    async def complete_with_usage(self, **kwargs: Any) -> Any:
        result = await self._inner.complete_with_usage(**kwargs)
        await self._budget.observe(_usage_of(result))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _usage_of(result: Any) -> dict[str, Any] | None:
    usage = getattr(result, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "completion_tokens_details": {
            "reasoning_tokens": getattr(usage, "reasoning_tokens", 0) or 0
        },
    }


async def score_arm(
    *,
    label: str,
    rows: list[dict[str, Any]],
    chat: Any,
    budget: Budget,
    concurrency: int,
    answer_max_tokens: int,
    judge_max_tokens: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    graded: list[dict[str, Any]] = []
    done = 0

    async def one(row: dict[str, Any]) -> None:
        nonlocal done
        async with semaphore:
            # `_complete_with_retry`, not a bare call: gpt-5 spends its completion budget on
            # hidden reasoning before a single visible token, and a row that exhausts it comes
            # back `finish_reason="length"` with EMPTY content. Scored naively that is a WRONG
            # answer — measured on the 5-row calibration of this very script, 2 of 5 mem0 rows
            # came back blank that way. Whichever arm has the longer context hits it more often,
            # so leaving it in would have measured prompt length and called it retrieval quality.
            answer = await _complete_with_retry(
                chat,
                system=ANSWER_SYSTEM_PROMPT,
                user=answer_prompt(context=row["context"], question=row["question"]),
                max_tokens=answer_max_tokens,
            )
            verdict = await _complete_with_retry(
                chat,
                system=COMPACT_JUDGE_SYSTEM_PROMPT,
                user=compact_judge_prompt(
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    response=answer.content or "",
                ),
                max_tokens=judge_max_tokens,
            )
        parsed = parse_judgement(verdict.content or "")
        graded.append(
            {
                "query_id": row["query_id"],
                "category": row["category"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "generated": answer.content,
                "verdict_raw": verdict.content,
                "answer_finish_reason": answer.finish_reason,
                "answer_budget_retried": getattr(answer, "budget_retried", False),
                "judge_finish_reason": verdict.finish_reason,
                "correct": parsed,
                "gold_in_context": row.get("gold_in_context"),
                "items": row.get("items"),
            }
        )
        done += 1
        if done % 25 == 0:
            emit(f"  [{label}] {done}/{len(rows)} cost=${budget.cost_usd:.2f}")

    await asyncio.gather(*(one(row) for row in rows))

    # `parse_judgement` returns None when the judge produced nothing parseable (a reasoning model
    # that spent its whole completion budget thinking returns empty content). Those rows are
    # counted and reported rather than folded into either bucket — a judge failure is not a wrong
    # answer, and quietly scoring it as one would flatter whichever arm had fewer of them.
    unparsed = sum(1 for row in graded if row["correct"] is None)
    scored = [row for row in graded if row["correct"] is not None]
    by_category: dict[str, dict[str, Any]] = {}
    for category, name in CATEGORY_NAMES.items():
        subset = [row for row in scored if row["category"] == category]
        if subset:
            by_category[name] = {
                "n": len(subset),
                "correct": sum(1 for row in subset if row["correct"]),
                "accuracy": sum(1 for row in subset if row["correct"]) / len(subset),
            }
    return {
        "label": label,
        "rows_in": len(rows),
        "rows_scored": len(scored),
        "answer_budget_exhausted_after_retries": sum(
            1
            for row in graded
            if row["answer_finish_reason"] == "length" and not (row["generated"] or "").strip()
        ),
        "judge_unparsed": unparsed,
        "correct": sum(1 for row in scored if row["correct"]),
        "accuracy": (sum(1 for row in scored if row["correct"]) / len(scored)) if scored else 0.0,
        "by_category": by_category,
        "gold_in_context_rate": (
            sum(1 for row in graded if row.get("gold_in_context")) / len(graded)
        )
        if graded
        else 0.0,
        "per_row": graded,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="label=/path/to/context.json — repeat once per arm",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-usd", type=float, default=2.5)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--base-url", default="https://libo-ai.services.ai.azure.com/openai/v1")
    parser.add_argument("--answer-max-tokens", type=int, default=900)
    parser.add_argument("--judge-max-tokens", type=int, default=500)
    parser.add_argument("--limit-rows", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("MU_EVAL_API_KEY")
    if not api_key:
        raise SystemExit("MU_EVAL_API_KEY is not set — refusing to run with an unusable key")

    arms: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in args.arm:
        label, _, path = spec.partition("=")
        doc = json.loads(Path(path).read_text())
        # `mem0_arm.export-context` writes "rows"; `ours_arm` carries its export under "contexts"
        # alongside the sweep it was produced by. Same row shape either way.
        arms[label] = {row["query_id"]: row for row in (doc.get("rows") or doc["contexts"])}

    # SAME ROWS, SAME DENOMINATOR, enforced rather than trusted.
    shared = set.intersection(*(set(rows) for rows in arms.values()))
    for label, rows in arms.items():
        missing = set(rows) - shared
        if missing:
            emit(f"  ! {label} has {len(missing)} rows no other arm has — excluded")
    order = sorted(shared)
    if args.limit_rows:
        order = order[: args.limit_rows]
    emit(f"scoring {len(order)} shared rows x {len(arms)} arms")

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
    results = []
    try:
        for label, rows in arms.items():
            results.append(
                await score_arm(
                    label=label,
                    rows=[rows[qid] for qid in order],
                    chat=metered,
                    budget=budget,
                    concurrency=args.concurrency,
                    answer_max_tokens=args.answer_max_tokens,
                    judge_max_tokens=args.judge_max_tokens,
                )
            )
    finally:
        served = sorted(chat.served_models)
        await chat.aclose()
        doc = {
            "rows": len(order),
            "requested_model": args.model,
            "served_models_read_from_response": served,
            "answer_prompt": "mem0 ANSWER_PROMPT (verbatim port, mu_eval/judge.py)",
            "judge_prompt": "MemOS locomo_grader compact accuracy prompt (verbatim port)",
            "usage": budget.snapshot(),
            "arms": results,
        }
        Path(args.out).write_text(json.dumps(doc, indent=1))
        emit(json.dumps({k: v for k, v in doc.items() if k != "arms"}, indent=1))
        for arm in results:
            emit(
                f"{arm['label']:>28}  acc={arm['accuracy']:.4f} "
                f"({arm['correct']}/{arm['rows_scored']}) "
                f"unparsed={arm['judge_unparsed']} gic={arm['gold_in_context_rate']:.4f}"
            )
        if len(served) > 1:
            emit(f"  !! MORE THAN ONE SERVED MODEL: {served} — the arms are not comparable")
        emit("categories:", Counter())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
