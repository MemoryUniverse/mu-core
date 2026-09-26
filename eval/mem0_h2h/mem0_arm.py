"""mem0 OSS (`mem0ai`) as a LoCoMo retrieval arm, scored by the SAME metric as our own engine.

WHY THIS EXISTS
---------------
`HEAD-TO-HEAD-0901.md` could not produce a mem0 number: their *hosted* pipeline needs a
`MEM0_API_KEY` this project does not have. `HEAD-TO-HEAD-0925-MEM0-OSS-UNBLOCKED.md` proved the
self-hostable `mem0.Memory` class runs against our own Azure `gpt-5` deployment. This module is
the actual measurement that unblocking was for.

WHAT IS MEASURED, AND WHY IT IS COMPARABLE
------------------------------------------
`gold_in_context`: for each LoCoMo QA row, did ANY retrieved item trace back to a gold evidence
turn? Our own arm answers that with `mu_eval.runner.gold_ids_present` (content-normalised join
back to the source turn). mem0 does not store turns — it stores LLM-extracted free-text facts —
so a content join is impossible by construction. Instead this module records the PROVENANCE
directly: every `add()` call is tagged with the `dia_id`s of the turns in that batch, and the
mapping memory_id -> {dia_id} is accumulated OUTSIDE mem0's payload.

That last point is not a style choice. `mem0/memory/main.py:2058-2060` (`_update_memory`) does
`new_metadata.update(metadata)` — an UPDATE overwrites the stored `dia_ids` with only the newest
batch's, silently destroying the earlier attribution. Reading provenance back off the payload
would therefore UNDER-count mem0's gold_in_context. The external accumulator keeps the union, so
a memory that absorbed facts from four batches is credited with all four batches' turns. This
scores mem0 *generously*: it is the ceiling of what their retrieval could be credited with.

FAIRNESS: WIDTH
---------------
mem0's own LoCoMo harness (`evaluation/src/memzero/search.py:19,91-96`) searches with `top_k=10`
*per speaker* and concatenates BOTH speakers' hits into the answer prompt — 20 items of context.
Our own conv-26 numbers were taken at `limit=10` — 10 items. A 20-vs-10 comparison measures width,
not retrieval. This module therefore sweeps k and reports every point, so any comparison can be
read at matched width AND at their own harness's setting. Search is a local embed + a local Chroma
query — ZERO LLM calls (verified in HEAD-TO-HEAD-0925 §1) — so the whole sweep is free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# `mu_eval` is the shared loader: same rows, same evidence, same adversarial exclusion as our arm.
_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from mu_eval.locomo import Conversation, load_locomo  # noqa: E402

from mem0_h2h import emit  # noqa: E402

# gpt-5 list rate, the same rate card our own runs price against
# (`eval/mu_eval/rate_card.json`).
PROMPT_USD_PER_1M = 1.25
COMPLETION_USD_PER_1M = 10.0


# ------------------------------------------------------------------ usage metering + budget guard


class BudgetExceededError(RuntimeError):
    """Raised the moment measured spend crosses the cap, so a run aborts instead of overrunning."""


@dataclass
class Meter:
    """Real measured usage, read from the SERVER's own `usage` block — never estimated.

    `served_model` is read back off the response too, because CLAUDE.md's head-to-head discipline
    requires the model be confirmed from the server rather than from the config that requested it.
    """

    cap_usd: float
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    served_models: set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * PROMPT_USD_PER_1M
            + self.completion_tokens / 1_000_000 * COMPLETION_USD_PER_1M
        )

    def observe(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        with self._lock:
            self.calls += 1
            model = getattr(response, "model", None)
            if model:
                self.served_models.add(str(model))
            if usage is not None:
                self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                details = getattr(usage, "completion_tokens_details", None)
                if details is not None:
                    self.reasoning_tokens += int(getattr(details, "reasoning_tokens", 0) or 0)
            cost = self.cost_usd
        if cost > self.cap_usd:
            raise BudgetExceededError(
                f"measured spend ${cost:.3f} crossed the ${self.cap_usd:.2f} cap after "
                f"{self.calls} calls — aborting before the next one"
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "served_models": sorted(self.served_models),
            "rate_card": {
                "prompt_usd_per_1m": PROMPT_USD_PER_1M,
                "completion_usd_per_1m": COMPLETION_USD_PER_1M,
            },
        }


def install_meter(memory: Any, meter: Meter) -> None:
    """Wrap the LLM client's `chat.completions.create` so EVERY call is metered and capped.

    Wrapping the client (not mem0's adapter) means a call mem0 makes from any code path — the
    extraction phase, a retry, a future phase this reader has not read — is counted. Nothing
    reaches the network without passing through here.
    """
    client = memory.llm.client
    original = client.chat.completions.create

    def metered(*args: Any, **kwargs: Any) -> Any:
        response = original(*args, **kwargs)
        meter.observe(response)
        return response

    client.chat.completions.create = metered  # type: ignore[method-assign]


class _DryRunLLM:
    """Counts what WOULD have been sent, calls nothing. Used to price a run before paying for it.

    Returns a plausible two-fact extraction so the pipeline keeps running and the
    "## Existing Memories" section of later prompts grows the way it really would; the projection
    is therefore an input-token count over a realistically-shaped conversation, not over an empty
    store.
    """

    def __init__(self) -> None:
        import tiktoken

        self.enc = tiktoken.get_encoding("o200k_base")
        self.calls = 0
        self.prompt_tokens = 0

    def generate_response(self, messages: list[dict[str, Any]], **_: Any) -> str:
        self.calls += 1
        text = "\n".join(str(m.get("content", "")) for m in messages)
        self.prompt_tokens += len(self.enc.encode(text))
        new_messages = text.split("## New Messages", 1)[-1].split("## Observation Date", 1)[0]
        snippet = " ".join(new_messages.split())[:180]
        return json.dumps({"memory": [{"text": f"Fact: {snippet}", "attributed_to": None}]})


# ------------------------------------------------------------------------------- mem0 construction


def build_memory(store_path: Path, collection: str, *, deployment: str = "gpt-5") -> Any:
    """Construct `mem0.Memory` on OUR gpt-5 deployment, a LOCAL embedder and a LOCAL Chroma store.

    Embedder and store are local on purpose: they cost nothing, they keep the run reproducible
    without a hosted account, and — critically — they make the retrieval half of the measurement
    free, so the k-sweep below can be run as many times as the analysis needs.
    """
    env: dict[str, str] = {}
    with open("/home/user/D/mu_project/mu-core/.env") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()

    os.environ["LLM_AZURE_OPENAI_API_KEY"] = env["MU_EVAL_API_KEY"]
    os.environ["LLM_AZURE_DEPLOYMENT"] = deployment
    os.environ["LLM_AZURE_ENDPOINT"] = env["AZURE_API_BASE"]
    os.environ["LLM_AZURE_API_VERSION"] = "2025-01-01-preview"

    from mem0 import Memory

    return Memory.from_config(
        {
            "llm": {
                "provider": "azure_openai",
                # mem0's OWN defaults for everything but the deployment name: no temperature
                # and no max_tokens override, so `BaseLlmConfig`'s shipped values apply
                # (`mem0/configs/llms/base.py:21`, max_tokens=2000). Tuning their generation
                # parameters would make this "mem0 as we configured it", not "mem0 as it ships".
                "config": {"model": deployment},
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "multi-qa-MiniLM-L6-cos-v1"},
            },
            "vector_store": {
                "provider": "chroma",
                "config": {"collection_name": collection, "path": str(store_path)},
            },
            "version": "v1.1",
        }
    )


# ------------------------------------------------------------------------------------ the ingestion


def speaker_batches(
    conversation: Conversation, *, for_speaker: str, batch_size: int
) -> list[dict[str, Any]]:
    """Rebuild mem0's own LoCoMo batching, per `evaluation/src/memzero/add.py:79-118`.

    Their `process_conversation` walks each session in order, renders every turn as a message
    whose role is `user` when the turn's speaker IS the user being ingested and `assistant`
    otherwise (their `messages` / `messages_reverse` pair), prefixes every message with the
    speaker's name, then slices the session's message list into `batch_size` chunks and sends one
    `add()` per chunk with `metadata={"timestamp": <session date>}`.

    The one thing added here is `dia_ids` on that metadata: the gold-label join our metric needs.
    It changes nothing about what mem0 reads — the model never sees the metadata.
    """
    by_session: dict[int, list[Any]] = {}
    for turn in conversation.turns:
        by_session.setdefault(turn.session_index, []).append(turn)

    batches: list[dict[str, Any]] = []
    for session_index in sorted(by_session):
        turns = by_session[session_index]
        messages = [
            {
                "role": "user" if turn.speaker == for_speaker else "assistant",
                "content": f"{turn.speaker}: {turn.text}",
                "_dia_id": turn.dia_id,
            }
            for turn in turns
        ]
        timestamp = turns[0].session_date
        for start in range(0, len(messages), batch_size):
            chunk = messages[start : start + batch_size]
            batches.append(
                {
                    "session_index": session_index,
                    "timestamp": timestamp,
                    "dia_ids": [m["_dia_id"] for m in chunk],
                    "messages": [{"role": m["role"], "content": m["content"]} for m in chunk],
                }
            )
    return batches


def ingest(
    memory: Any,
    conversation: Conversation,
    *,
    batch_size: int,
    meter: Meter,
    out_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest conv-N into mem0 for BOTH speakers, accumulating memory_id -> {dia_id} externally.

    Checkpointed after every batch: a budget abort, a rate-limit death or a killed workflow leaves
    a resumable, already-paid-for store rather than a wasted spend.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = out_dir / "provenance.json"
    state: dict[str, Any] = {"done": [], "memory_dia_ids": {}, "events": []}
    if provenance_path.exists():
        state = json.loads(provenance_path.read_text())

    done = set(state["done"])
    memory_dia_ids: dict[str, list[str]] = state["memory_dia_ids"]
    started = time.time()

    for speaker in (conversation.speaker_a, conversation.speaker_b):
        user_id = f"{speaker}_{conversation.sample_id}"
        batches = speaker_batches(conversation, for_speaker=speaker, batch_size=batch_size)
        for index, batch in enumerate(batches):
            key = f"{user_id}#{index}"
            if key in done:
                continue
            result = memory.add(
                batch["messages"],
                user_id=user_id,
                metadata={
                    "timestamp": batch["timestamp"],
                    "dia_ids": ",".join(batch["dia_ids"]),
                },
            )
            rows = result.get("results", []) if isinstance(result, dict) else (result or [])
            for row in rows:
                mem_id = row.get("id")
                if not mem_id:
                    continue
                merged = set(memory_dia_ids.get(mem_id, [])) | set(batch["dia_ids"])
                memory_dia_ids[mem_id] = sorted(merged)
                state["events"].append({"id": mem_id, "event": row.get("event"), "batch": key})
            done.add(key)
            state["done"] = sorted(done)
            state["memory_dia_ids"] = memory_dia_ids
            provenance_path.write_text(json.dumps(state))
            if (len(done) % 10) == 0 or dry_run:
                elapsed = time.time() - started
                emit(
                    f"  [{len(done)}] {key} mems={len(memory_dia_ids)} "
                    f"cost=${meter.cost_usd:.3f} calls={meter.calls} {elapsed:.0f}s",
                )
    return state


# --------------------------------------------------------------------------------------- the score


def score(
    memory: Any,
    conversation: Conversation,
    *,
    memory_dia_ids: dict[str, list[str]],
    per_speaker_k: int,
) -> dict[str, Any]:
    """`gold_in_context` for the mem0 arm, over exactly the rows our own arm scores.

    Row selection is copied from the width-controlled verify script our own conv-26 numbers came
    from (`docs/tracking/eval-runs/2026-09-25-ad301/verify_rerank_width.py:39-46`): skip
    adversarial rows, skip rows with no evidence, and intersect the evidence with the turn ids
    that actually exist in this conversation. Same rows, same denominator, both arms.
    """
    known = {turn.dia_id for turn in conversation.turns}
    user_ids = [
        f"{conversation.speaker_a}_{conversation.sample_id}",
        f"{conversation.speaker_b}_{conversation.sample_id}",
    ]
    scored = hits = 0
    item_counts: list[int] = []
    turn_counts: list[int] = []
    per_row: list[dict[str, Any]] = []

    for query in conversation.queries:
        if query.is_adversarial or not query.evidence:
            continue
        gold = {e for e in query.evidence if e in known}
        if not gold:
            continue
        retrieved_ids: list[str] = []
        for user_id in user_ids:
            # `top_k`, NOT `limit`. mem0 2.2.0's `Memory.search` is keyword-only
            # (`mem0/memory/main.py:1379-1391`) and takes `top_k`; `limit=` lands in `**kwargs`
            # and is SILENTLY IGNORED, so every k in the sweep runs at the default top_k=20.
            # Measured the hard way: the first sweep returned items=40.0 at every k from 1 to 20.
            found = memory.search(query.question, filters={"user_id": user_id}, top_k=per_speaker_k)
            rows = found.get("results", []) if isinstance(found, dict) else (found or [])
            retrieved_ids.extend(row["id"] for row in rows if row.get("id"))
        covered: set[str] = set()
        for mem_id in retrieved_ids:
            covered.update(memory_dia_ids.get(mem_id, []))
        present = bool(covered & gold)
        hits += int(present)
        scored += 1
        item_counts.append(len(retrieved_ids))
        turn_counts.append(len(covered))
        per_row.append(
            {
                "query_id": query.query_id,
                "category": query.category,
                "gold": sorted(gold),
                "present": present,
                "items": len(retrieved_ids),
                "turns_covered": len(covered),
            }
        )

    return {
        "per_speaker_k": per_speaker_k,
        "queries_scored": scored,
        "gold_in_context": hits,
        "gold_in_context_rate": hits / scored if scored else 0.0,
        "items_mean": sum(item_counts) / len(item_counts) if item_counts else 0.0,
        "turns_covered_mean": sum(turn_counts) / len(turn_counts) if turn_counts else 0.0,
        "per_row": per_row,
    }


def export_context(
    memory: Any,
    conversation: Any,
    *,
    memory_dia_ids: dict[str, list[str]],
    per_speaker_k: int,
) -> dict[str, Any]:
    """Dump the exact context mem0 would hand an answering model, one row per LoCoMo query.

    Rendering follows mem0's own LoCoMo harness (`evaluation/src/memzero/search.py:98-99`):
    one line per retrieved memory, `"{timestamp}: {memory}"`, both speakers' hits concatenated.
    The bullet prefix is the ONLY addition, and it is there because our own arm's context lines
    carry it (`mu_eval/answer_quality.py:539`) — the two arms must differ in WHAT was retrieved
    and in nothing else, or the answer-quality comparison measures prompt formatting.
    """
    known = {turn.dia_id for turn in conversation.turns}
    user_ids = [
        f"{conversation.speaker_a}_{conversation.sample_id}",
        f"{conversation.speaker_b}_{conversation.sample_id}",
    ]
    rows = []
    for query in conversation.queries:
        if query.is_adversarial or not query.evidence:
            continue
        gold = {e for e in query.evidence if e in known}
        if not gold:
            continue
        lines: list[str] = []
        covered: set[str] = set()
        for user_id in user_ids:
            found = memory.search(query.question, filters={"user_id": user_id}, top_k=per_speaker_k)
            hits = found.get("results", []) if isinstance(found, dict) else (found or [])
            for hit in hits:
                stamp = (hit.get("metadata") or {}).get("timestamp", "")
                lines.append(f"- {stamp}: {hit.get('memory', '')}".strip())
                if hit.get("id"):
                    covered.update(memory_dia_ids.get(hit["id"], []))
        rows.append(
            {
                "query_id": query.query_id,
                "question": query.question,
                "gold_answer": query.answer,
                "category": query.category,
                "gold": sorted(gold),
                "context": "\n".join(lines) or "(no memories retrieved)",
                "items": len(lines),
                "turns_covered": len(covered),
                "gold_in_context": bool(covered & gold),
            }
        )
    return {
        "arm": f"mem0_oss_top_k{per_speaker_k}_per_speaker",
        "sample_id": conversation.sample_id,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["project", "ingest", "score", "export-context"])
    parser.add_argument(
        "--dataset", default="/home/user/D/abstract_project/mma/data/locomo/locomo10.json"
    )
    parser.add_argument("--sample", default="conv-26")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--budget-usd", type=float, default=3.5)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ks", default="1,2,3,5,10,15,20")
    args = parser.parse_args()

    conversations = load_locomo(args.dataset, samples=None)
    matching = [c for c in conversations if c.sample_id == args.sample]
    if not matching:
        raise SystemExit(f"sample {args.sample} not in {args.dataset}")
    conversation = matching[0]
    out_dir = Path(args.out_dir)
    store_path = out_dir / "chroma"
    collection = f"mem0_{args.sample.replace('-', '_')}_b{args.batch_size}"

    if args.command == "project":
        memory = build_memory(store_path, collection + "_dry")
        dry = _DryRunLLM()
        memory.llm = dry  # type: ignore[assignment]
        ingest(
            memory,
            conversation,
            batch_size=args.batch_size,
            meter=Meter(cap_usd=0.0),
            out_dir=out_dir / "dryrun",
            dry_run=False,
        )
        projected = dry.prompt_tokens / 1_000_000 * PROMPT_USD_PER_1M
        emit(
            json.dumps(
                {
                    "batch_size": args.batch_size,
                    "add_calls": dry.calls,
                    "prompt_tokens": dry.prompt_tokens,
                    "prompt_cost_usd": round(projected, 3),
                    "note": "input tokens only; completion/reasoning priced from the live run",
                },
                indent=1,
            )
        )
        return 0

    memory = build_memory(store_path, collection)

    if args.command == "ingest":
        meter = Meter(cap_usd=args.budget_usd)
        install_meter(memory, meter)
        try:
            ingest(memory, conversation, batch_size=args.batch_size, meter=meter, out_dir=out_dir)
        finally:
            (out_dir / "usage.json").write_text(json.dumps(meter.snapshot(), indent=1))
            emit(json.dumps(meter.snapshot(), indent=1))
        return 0

    state = json.loads((out_dir / "provenance.json").read_text())

    if args.command == "export-context":
        k = int(args.ks.split(",")[0])
        doc = export_context(
            memory, conversation, memory_dia_ids=state["memory_dia_ids"], per_speaker_k=k
        )
        path = out_dir / f"context_k{k}.json"
        path.write_text(json.dumps(doc, indent=1))
        emit(f"{len(doc['rows'])} rows -> {path}")
        return 0

    results = [
        score(
            memory,
            conversation,
            memory_dia_ids=state["memory_dia_ids"],
            per_speaker_k=int(k),
        )
        for k in args.ks.split(",")
    ]
    doc = {
        "sample_id": conversation.sample_id,
        "batch_size": args.batch_size,
        "memories_total": len(state["memory_dia_ids"]),
        "add_calls": len(state["done"]),
        "sweep": results,
    }
    (out_dir / "score.json").write_text(json.dumps(doc, indent=1))
    for row in results:
        emit(
            f"k/speaker={row['per_speaker_k']:>3} items={row['items_mean']:.1f} "
            f"turns={row['turns_covered_mean']:.1f} "
            f"gic={row['gold_in_context']}/{row['queries_scored']}="
            f"{row['gold_in_context_rate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
