# ruff: noqa: E501, W291, RUF001, RUF002, RUF003
# ^ DELIBERATE, and scoped to this file: `MEM0_ACCURACY_PROMPT` below is a BYTE-IDENTICAL copy of
#   mem0's own grader, trailing spaces and typographic apostrophes included. Stripping a trailing
#   space or normalising a quote character EDITS THE METRIC — the prompt IS the scorer here.
"""Re-grade the STORED head-to-head answers under mem0's OWN grader, to price the grader itself.

WHY
---
`answer_h2h.py` graded every arm with `mu_eval.judge.COMPACT_JUDGE_SYSTEM_PROMPT` /
`compact_judge_prompt` — a rewrite built for Azure Foundry's Ministral-3B 400-token-total request
ceiling (see `mu_eval/judge.py`'s own argument for the cut). The head-to-head did not run on
Ministral-3B; it ran on `gpt-5`, which has no such ceiling. So the reported 52 / 62 / 65 were
produced by a grader whose only reason to exist was a constraint that did not apply to that run.

mem0 publishes its own grader: `ACCURACY_PROMPT` at
`other_repos/mem0/evaluation/metrics/llm_judge.py:12-36`, sent as a SINGLE user message with
`response_format={"type":"json_object"}` (`:41-53`). This script re-grades the SAME stored answers
with that prompt, byte-for-byte, so the only thing that changes between the two numbers is the
GRADER. No retrieval, no answering — the answers already exist and are already paid for.

Two prompt variants are run over the same rows:
  * ``verbatim`` — mem0's `ACCURACY_PROMPT`, user-only message, JSON response format. Their grader.
  * ``compact``  — a REPLICATION of what the original run used. Not redundant: gpt-5 is a
    reasoning model with no seed, so a verbatim-vs-compact gap read against the ARCHIVED compact
    verdicts would confound the prompt change with judge run-to-run noise. Re-running compact now
    measures that noise directly, and the comparison that matters is verbatim-vs-compact WITHIN
    this run.

Budget is metered off each response's own `usage` block and aborts the run on the cap, same rule
as `answer_h2h.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from mu_eval.judge import (  # noqa: E402
    COMPACT_JUDGE_SYSTEM_PROMPT,
    compact_judge_prompt,
    parse_judgement,
)
from mu_eval.locomo import CATEGORY_NAMES  # noqa: E402

from mem0_h2h import emit  # noqa: E402

PROMPT_USD_PER_1M = 1.25
COMPLETION_USD_PER_1M = 10.0

# --- VERBATIM PORT: mem0 evaluation/metrics/llm_judge.py:12-36 (ACCURACY_PROMPT) --------------
# Copied CHARACTER-FOR-CHARACTER from the cloned checkout, trailing spaces and typographic
# apostrophes included. Reflowing a line or normalising a quote EDITS THE METRIC.
MEM0_ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


class BudgetExceededError(RuntimeError):
    pass


class Budget:
    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.calls = 0
        self.served_models: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * PROMPT_USD_PER_1M
            + self.completion_tokens / 1_000_000 * COMPLETION_USD_PER_1M
        )

    async def observe(self, body: dict[str, Any]) -> None:
        usage = body.get("usage") or {}
        async with self._lock:
            self.calls += 1
            model = body.get("model")
            if model:
                self.served_models.add(str(model))
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            details = usage.get("completion_tokens_details") or {}
            self.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
            cost = self.cost_usd
        if cost > self.cap_usd:
            raise BudgetExceededError(
                f"measured spend ${cost:.3f} crossed the ${self.cap_usd:.2f} cap after "
                f"{self.calls} calls"
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "served_models": sorted(self.served_models),
        }


async def _post(
    client: httpx.AsyncClient, payload: dict[str, Any], budget: Budget
) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(3):
        try:
            response = await client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            last = exc
            await asyncio.sleep(5.0 * (attempt + 1))
            continue
        if response.status_code == 429:
            await asyncio.sleep(float(response.headers.get("retry-after") or 10))
            continue
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        await budget.observe(body)
        return body
    raise RuntimeError(f"judge call failed after 3 attempts: {last!r}")


def _content(body: dict[str, Any]) -> tuple[str, str | None]:
    choice = body["choices"][0]
    return str(choice["message"].get("content") or ""), choice.get("finish_reason")


async def judge_verbatim(
    client: httpx.AsyncClient, row: dict[str, Any], budget: Budget, model: str, max_tokens: int
) -> tuple[bool | None, str, str | None]:
    """mem0's own call shape: ONE user message, JSON response format, no system turn."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": MEM0_ACCURACY_PROMPT.format(
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    generated_answer=row["generated"] or "",
                ),
            }
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_tokens,
    }
    body = await _post(client, payload, budget)
    content, finish = _content(body)
    return parse_judgement(content), content, finish


async def judge_compact(
    client: httpx.AsyncClient, row: dict[str, Any], budget: Budget, model: str, max_tokens: int
) -> tuple[bool | None, str, str | None]:
    """Replication of the ORIGINAL run's grader call, same shape `answer_h2h.py` used."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": COMPACT_JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": compact_judge_prompt(
                    question=row["question"],
                    gold_answer=row["gold_answer"],
                    response=row["generated"] or "",
                ),
            },
        ],
        "max_completion_tokens": max_tokens,
    }
    body = await _post(client, payload, budget)
    content, finish = _content(body)
    return parse_judgement(content), content, finish


def summarise(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    scored = [r for r in rows if r[key] is not None]
    by_category: dict[str, dict[str, Any]] = {}
    for category, name in CATEGORY_NAMES.items():
        subset = [r for r in scored if r["category"] == category]
        if subset:
            correct = sum(1 for r in subset if r[key])
            by_category[name] = {
                "n": len(subset),
                "correct": correct,
                "accuracy": correct / len(subset),
            }
    correct = sum(1 for r in scored if r[key])
    return {
        "rows_scored": len(scored),
        "unparsed": len(rows) - len(scored),
        "correct": correct,
        "accuracy": correct / len(scored) if scored else 0.0,
        "by_category": by_category,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="answer_h2h_n100_rows.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-usd", type=float, default=3.0)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--base-url", default="https://libo-ai.services.ai.azure.com/openai/v1")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("MU_EVAL_API_KEY")
    if not api_key:
        raise SystemExit("MU_EVAL_API_KEY is not set")

    source: dict[str, list[dict[str, Any]]] = json.loads(Path(args.rows).read_text())
    budget = Budget(args.budget_usd)
    semaphore = asyncio.Semaphore(args.concurrency)
    out: dict[str, Any] = {
        "source": str(Path(args.rows).resolve()),
        "requested_model": args.model,
        "arms": {},
    }

    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers={"api-key": api_key, "Authorization": f"Bearer {api_key}"},
        timeout=300.0,
    ) as client:
        try:
            for arm, rows in source.items():
                rows = rows[: args.limit] if args.limit else rows
                graded: list[dict[str, Any]] = []

                async def one(row: dict[str, Any], graded: list[dict[str, Any]] = graded) -> None:
                    async with semaphore:
                        v_ok, v_raw, v_fin = await judge_verbatim(
                            client, row, budget, args.model, args.max_tokens
                        )
                        c_ok, c_raw, c_fin = await judge_compact(
                            client, row, budget, args.model, args.max_tokens
                        )
                    graded.append(
                        {
                            "query_id": row["query_id"],
                            "category": row["category"],
                            "question": row["question"],
                            "gold_answer": row["gold_answer"],
                            "generated": row["generated"],
                            "archived_compact": row["correct"],
                            "verbatim": v_ok,
                            "verbatim_raw": v_raw,
                            "verbatim_finish": v_fin,
                            "compact_replay": c_ok,
                            "compact_replay_raw": c_raw,
                            "compact_replay_finish": c_fin,
                        }
                    )

                await asyncio.gather(*(one(row) for row in rows))
                out["arms"][arm] = {
                    "verbatim_mem0_accuracy_prompt": summarise(graded, "verbatim"),
                    "compact_replay": summarise(graded, "compact_replay"),
                    "archived_compact": summarise(graded, "archived_compact"),
                    "per_row": graded,
                }
                emit(
                    f"{arm:30s} verbatim={out['arms'][arm]['verbatim_mem0_accuracy_prompt']['accuracy']:.4f} "
                    f"compact_replay={out['arms'][arm]['compact_replay']['accuracy']:.4f} "
                    f"archived={out['arms'][arm]['archived_compact']['accuracy']:.4f} "
                    f"cost=${budget.cost_usd:.2f}"
                )
        finally:
            out["usage"] = budget.snapshot()
            Path(args.out).write_text(json.dumps(out, indent=1))
            emit(json.dumps(out["usage"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
