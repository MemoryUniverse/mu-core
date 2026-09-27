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
import contextlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.environ.get("H2H_EVAL_DIR", str(Path(__file__).resolve().parent.parent)))

from dateutil import parser as _dateutil_parser
from mu_eval.corpus import ingest_conversation, local_memory_for
from mu_eval.locomo import Turn, load_locomo
from mu_eval.runner import _await_index, gold_ids_present
from pydantic_settings import BaseSettings, SettingsConfigDict

from mem0_h2h import emit

# ---------------------------------------------------------------------------------------------
# AD-332: the LLM (SLM-backed) extraction arm. This harness has NEVER configured a model profile
# (AD-325/AD-328/SCORECARD.md §6) — every published number used `HeuristicSpoExtractor` only.
# Central-config home for the toggle (DEV-STANDARDS rule 3): no bare `os.environ.get` at the call
# site, one settings object, mirrors `packages/mu-local/tests/test_local_llm_slm_int.py`'s own
# `SlmTestSettings` exactly (same env shape, `H2H_SLM__` prefix instead of `MU_TEST_SLM__` so the
# eval harness's env is its own namespace, not a silent alias of the integration test's).
# ---------------------------------------------------------------------------------------------


class H2hSlmSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="H2H_SLM__",
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = "http://127.0.0.1:11435/v1"  # Ollama's OpenAI-compat shim (mu-dev-slm)
    probe_url: str = "http://127.0.0.1:11435"
    probe_timeout_s: float = 2.0
    model: str = "qwen2.5:0.5b"
    max_tokens: int = 256
    temperature: float = 0.0


def _slm_reachable(cfg: H2hSlmSettings) -> bool:
    """A real env probe (HTTP GET, short timeout) — never a fabricated "it's up"."""
    try:
        with urllib.request.urlopen(cfg.probe_url, timeout=cfg.probe_timeout_s) as resp:  # noqa: S310
            return bool(200 <= int(resp.status) < 300)
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


class LlmExtractProbe:
    """Counts REAL `LlmFactExtractor.extract` calls + their token usage — ASSERTED, never inferred.

    AD-328's own register row: "a rerank arm silently degraded to a broken fallback and the
    numbers came back BETTER" — the standing lesson this probe exists to not repeat for
    extraction. Wraps the bound method for the lifetime of the `async with` block only (restored
    in `finally`, so a probe left running never leaks into an unrelated arm of the same process).
    """

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @contextlib.asynccontextmanager
    async def watch(self) -> AsyncIterator[LlmExtractProbe]:
        from mu_engine.services.extract import LlmFactExtractor

        original = LlmFactExtractor.extract
        probe = self

        async def _counted(self_: LlmFactExtractor, text: str, *, now: datetime) -> Any:
            completion_holder: dict[str, Any] = {}
            orig_provider_complete = self_._provider.complete

            async def _counted_complete(*args: Any, **kwargs: Any) -> Any:
                completion = await orig_provider_complete(*args, **kwargs)
                completion_holder["completion"] = completion
                return completion

            self_._provider.complete = _counted_complete  # type: ignore[method-assign]
            try:
                result = await original(self_, text, now=now)
            finally:
                self_._provider.complete = orig_provider_complete  # type: ignore[method-assign]
            probe.calls += 1
            usage = completion_holder.get("completion")
            if usage is not None:
                probe.prompt_tokens += usage.usage.prompt_tokens
                probe.completion_tokens += usage.usage.completion_tokens
            return result

        LlmFactExtractor.extract = _counted  # type: ignore[assignment]
        try:
            yield probe
        finally:
            LlmFactExtractor.extract = original  # type: ignore[method-assign]


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

    # AD-332: point DISTILL's extractor at the real, $0 local SLM instead of the MVP-default
    # `HeuristicSpoExtractor` — this harness has never configured a model profile at all
    # (AD-325/AD-328/SCORECARD.md §6). Off by default (byte-identical to every prior run of this
    # script); `H2H_LLM_EXTRACT=1` opts in. Only meaningful with `consolidate=True` — the
    # extractor lives in DISTILL (`pipelines/distill.py`), never on the STM->MTM ingest path — so
    # a misconfigured combination fails LOUD here rather than silently running heuristic anyway
    # (the AD-328 lesson: assert, never infer).
    use_llm_extract = os.environ.get("H2H_LLM_EXTRACT", "0") == "1"
    if use_llm_extract and not consolidate:
        raise SystemExit(
            "H2H_LLM_EXTRACT=1 requires H2H_CONSOLIDATE=1 — LlmFactExtractor only runs in DISTILL "
            "(MTM->LTM); the heuristic-only ingest path never calls it."
        )
    storage = None
    slm_cfg: H2hSlmSettings | None = None
    if use_llm_extract:
        from mu_local.config import ModelProfileSettings, StorageSettings

        slm_cfg = H2hSlmSettings()
        if not _slm_reachable(slm_cfg):
            raise SystemExit(
                f"H2H_LLM_EXTRACT=1 but the local SLM at {slm_cfg.probe_url} is unreachable — "
                "guard, not fake (bring it up: infra/mu-vm/vm_reup.sh / the SLM compose stack)."
            )
        storage = StorageSettings(
            llm=ModelProfileSettings(
                base_url=slm_cfg.base_url,
                model=slm_cfg.model,
                max_tokens=slm_cfg.max_tokens,
                temperature=slm_cfg.temperature,
            )
        )

    probe = LlmExtractProbe()
    async with (
        probe.watch(),
        local_memory_for(conversation, run_id=run_id, storage=storage) as opaque,
    ):
        memory: Any = opaque
        ingest_started = time.perf_counter()
        index, report = await ingest_conversation(
            memory,
            conversation,
            user=user,
            session=session,
            importance=importance,
            consolidate=consolidate,
            occurred_at_of=_turn_occurred_at if use_occurred_at else None,
        )
        ingest_wall_s = time.perf_counter() - ingest_started
        if use_llm_extract and probe.calls == 0:
            # Guard, not fake (AD-328's standing lesson): a configured profile that never actually
            # invoked the LLM extractor must fail the run, not report a heuristic number under an
            # LLM-arm label.
            raise SystemExit(
                "H2H_LLM_EXTRACT=1 but LlmFactExtractor.extract was called ZERO times — the "
                "profile did not take. Refusing to report a heuristic result as an LLM arm."
            )
        if use_llm_extract:
            emit(
                f"  [{label}] llm_extract: calls={probe.calls} "
                f"prompt_tokens={probe.prompt_tokens} completion_tokens={probe.completion_tokens} "
                f"ingest_wall_s={ingest_wall_s:.1f}"
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
                        # AD-308/AD-316: the PRODUCT'S OWN signal, threaded end to end by this
                        # lane's own engine fixes — nothing corpus-side. AD-315's 7/8-row diagnostic
                        # found that prefixing a hit with `valid_at` (a RESOLVED date — the in-text
                        # relative clause already shifted against its anchor) while `item.content`
                        # still carries that SAME relative phrase unmodified causes the answering
                        # model to double-count: it re-applies the relative offset on top of the
                        # already-resolved prefix (the "off by exactly 7 days" / "double-applied
                        # two days ago" failure signatures). `RecallItemView.occurred_at` (AD-316)
                        # is the RAW, UNRESOLVED capture instant — the same shape the harness-
                        # assisted arm's `turn_date` rejoin rendered, which is why that arm scored
                        # 18/22 on these exact rows. Preferring it here is a real product signal,
                        # not a corpus lookup: `occurred_at` is exactly the value this same
                        # harness already passes into `LocalMemory.add(occurred_at=...)`
                        # (`_turn_occurred_at` above) and gets back out through the public recall
                        # API — never read from
                        # `turn_date`/the corpus directly. Falls back to `valid_at` when
                        # `occurred_at` is unset (an item with no AD-312 caller-asserted capture
                        # time — e.g. LTM-distilled content with only a resolved/inferred date),
                        # preserving AD-308's prior behaviour for that case.
                        product_date = getattr(item, "occurred_at", None) or getattr(
                            item, "valid_at", None
                        )
                        if product_date is not None:
                            product_dated += 1
                            product_lines.append(f"- {product_date.date()}: {item.content}")
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
        "llm_extract": {
            "enabled": use_llm_extract,
            "calls": probe.calls,
            "prompt_tokens": probe.prompt_tokens,
            "completion_tokens": probe.completion_tokens,
            "ingest_wall_s": round(ingest_wall_s, 3),
            "model": slm_cfg.model if slm_cfg is not None else None,
        },
        "effective_config": {
            "rerank_enabled": cfg.rerank_enabled,
            "rerank_min_score": cfg.rerank_min_score,
            "rerank_top_fraction": cfg.rerank_top_fraction,
            "rerank_pool_size": cfg.rerank_pool_size,
            "channel_pool_size": cfg.channel_pool_size,
            "channel_pool_multiplier": getattr(cfg, "channel_pool_multiplier", None),
            "floor_protect_limit": cfg.floor_protect_limit,
            # AD-330: S1b's three read-side knobs, so an artifact says which neighbour-expansion
            # shape produced it instead of leaving a reader to trust the launch command. Read via
            # `getattr` with the shipped defaults so an OLDER engine (no `neighbor_expand_anchor_
            # top_n`) still exports, rather than crashing an arm over a provenance field.
            "neighbor_expand_radius": cfg.neighbor_expand_radius,
            "neighbor_expand_placement": cfg.neighbor_expand_placement,
            "neighbor_expand_anchor_top_n": getattr(cfg, "neighbor_expand_anchor_top_n", 0),
            "neighbor_free_ride": cfg.neighbor_free_ride,
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
