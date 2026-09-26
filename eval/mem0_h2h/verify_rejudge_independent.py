"""INDEPENDENT verification of AD-304's re-judging (AD-305).

Deliberately shares no GRADING code with `rejudge_prompts.py`: it parses mem0's `ACCURACY_PROMPT`
out of the cloned reference checkout at runtime, and rebuilds
`mu_eval.judge.COMPACT_JUDGE_SYSTEM_PROMPT` from that module's source literals rather than
importing either. If the harness under audit had the wrong prompt text baked in, this script would
disagree with it rather than inherit the error. (`emit` is imported for output only.)

It measures three things the first pass could not measure about itself:

  1. **Agreement.** Re-grade rows already graded verbatim and compare verdict-for-verdict — this is
     the verbatim prompt's own run-to-run noise, which the first pass measured only for `compact`.
  2. **Reproduction.** Does the verbatim-over-compact gap appear again, in every arm, on fresh
     calls, with no reverse flips?
  3. **Which model actually served.** Read off each response body's `model`, never the config.

Usage (from `eval/`)::

    python -m mem0_h2h.verify_rejudge_independent --cap-usd 1.20 --per-arm 40 --out verify.json
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import httpx

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from mem0_h2h import emit  # noqa: E402

REF_JUDGE = Path(
    "/home/user/D/abstract_project/mma/other_repos/mem0/evaluation/metrics/llm_judge.py"
)
OUR_JUDGE = _EVAL_DIR / "mu_eval" / "judge.py"

PROMPT_USD_PER_1M = 1.25
COMPLETION_USD_PER_1M = 10.0


def _mem0_accuracy_prompt() -> str:
    """mem0's own `ACCURACY_PROMPT`, read from the cloned checkout at run time."""
    match = re.search(r'ACCURACY_PROMPT = """(.*?)"""', REF_JUDGE.read_text(), re.S)
    if match is None:
        raise RuntimeError(f"ACCURACY_PROMPT not found in {REF_JUDGE}")
    return match.group(1)


def _compact_system_prompt() -> str:
    """Rebuild `COMPACT_JUDGE_SYSTEM_PROMPT` from `judge.py`'s source literals, not by import."""
    match = re.search(r"COMPACT_JUDGE_SYSTEM_PROMPT = \((.*?)\n\)", OUR_JUDGE.read_text(), re.S)
    if match is None:
        raise RuntimeError(f"COMPACT_JUDGE_SYSTEM_PROMPT not found in {OUR_JUDGE}")
    return "".join(ast.literal_eval(line.strip()) for line in match.group(1).strip().splitlines())


def parse_label(text: str) -> bool | None:
    """The official parse, plus the bare-label tolerance `mu_eval.judge` documents."""
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass
    else:
        if isinstance(payload, dict) and "label" in payload:
            return str(payload["label"]).strip().lower() == "correct"
    labels = {m.upper() for m in re.findall(r"\b(CORRECT|WRONG)\b", stripped, re.I)}
    if not labels or labels == {"CORRECT", "WRONG"}:
        return None
    return labels == {"CORRECT"}


class Meter:
    """Budget + served-model accounting, metered off each response's own `usage` block."""

    def __init__(self, cap_usd: float) -> None:
        self.cap_usd = cap_usd
        self.prompt_tokens = 0
        self.completion_tokens = 0
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
            if body.get("model"):
                self.served_models.add(str(body["model"]))
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            cost = self.cost_usd
        if cost > self.cap_usd:
            raise RuntimeError(f"measured spend ${cost:.3f} crossed the ${self.cap_usd:.2f} cap")

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "served_models": sorted(self.served_models),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }


def _stratified(rows: list[dict[str, Any]], size: int, rng: random.Random) -> list[dict[str, Any]]:
    """Sample across categories, so the check is not all single-hop."""
    by_category: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    per = max(1, size // max(1, len(by_category)))
    picked: list[dict[str, Any]] = []
    for category in sorted(by_category):
        pool = sorted(by_category[category], key=lambda r: r["query_id"])
        picked += rng.sample(pool, min(per, len(pool)))
    chosen = {row["query_id"] for row in picked}
    ordered = sorted(rows, key=lambda r: r["query_id"])
    rest = [row for row in ordered if row["query_id"] not in chosen]
    picked += rng.sample(rest, max(0, min(size - len(picked), len(rest))))
    return picked


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stored", required=True, help="rejudge_n100_two_graders.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--cap-usd", type=float, default=1.20)
    parser.add_argument("--per-arm", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--base-url", default="https://libo-ai.services.ai.azure.com/openai/v1")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()

    api_key = os.environ.get("MU_EVAL_API_KEY")
    if not api_key:
        raise SystemExit("MU_EVAL_API_KEY is not set")

    accuracy_prompt = _mem0_accuracy_prompt()
    compact_system = _compact_system_prompt()
    stored = json.loads(Path(args.stored).read_text())
    rng = random.Random(args.seed)  # noqa: S311 — a reproducible sample, not a security decision
    meter = Meter(args.cap_usd)
    out: dict[str, Any] = {"stored": str(Path(args.stored).resolve()), "per_arm": {}}

    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers={"api-key": api_key, "Authorization": f"Bearer {api_key}"},
        timeout=300.0,
    ) as client:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def call(payload: dict[str, Any]) -> str:
            async with semaphore:
                body: dict[str, Any] | None = None
                for _attempt in range(3):
                    response = await client.post("/chat/completions", json=payload)
                    if response.status_code == 429:
                        await asyncio.sleep(10.0)
                        continue
                    response.raise_for_status()
                    body = response.json()
                    break
                if body is None:
                    raise RuntimeError("judge call rate-limited after 3 attempts")
            await meter.observe(body)
            return str(body["choices"][0]["message"].get("content") or "")

        try:
            for arm, payload in stored["arms"].items():
                sample = _stratified(payload["per_row"], args.per_arm, rng)

                async def one(row: dict[str, Any]) -> dict[str, Any]:
                    verbatim = await call(
                        {
                            "model": args.model,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": accuracy_prompt.format(
                                        question=row["question"],
                                        gold_answer=row["gold_answer"],
                                        generated_answer=row["generated"] or "",
                                    ),
                                }
                            ],
                            "response_format": {"type": "json_object"},
                            "max_completion_tokens": args.max_tokens,
                        }
                    )
                    compact = await call(
                        {
                            "model": args.model,
                            "messages": [
                                {"role": "system", "content": compact_system},
                                {
                                    "role": "user",
                                    "content": (
                                        f"Question: {row['question']}\n"
                                        f"Gold: {row['gold_answer']}\n"
                                        f"Generated: {row['generated'] or ''}"
                                    ),
                                },
                            ],
                            "max_completion_tokens": args.max_tokens,
                        }
                    )
                    return {
                        "query_id": row["query_id"],
                        "category": row["category"],
                        "my_verbatim": parse_label(verbatim),
                        "my_compact": parse_label(compact),
                        "their_verbatim": row["verbatim"],
                        "their_compact_replay": row["compact_replay"],
                        "archived": row["archived_compact"],
                    }

                graded = list(await asyncio.gather(*(one(row) for row in sample)))
                out["per_arm"][arm] = graded
                agree = sum(1 for r in graded if r["my_verbatim"] == r["their_verbatim"])
                forward = sum(1 for r in graded if r["my_verbatim"] and not r["my_compact"])
                reverse = sum(1 for r in graded if r["my_compact"] and not r["my_verbatim"])
                emit(
                    f"{arm:30s} n={len(graded)} agreement_with_stored={agree}/{len(graded)} "
                    f"compact->verbatim flips +{forward}/-{reverse} cost=${meter.cost_usd:.3f}"
                )
        finally:
            out["usage"] = meter.snapshot()
            Path(args.out).write_text(json.dumps(out, indent=1))
            emit(json.dumps(out["usage"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
