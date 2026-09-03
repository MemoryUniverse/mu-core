"""CLI: ``python -m mu_eval <command>``. Runs ON THE VM (CLAUDE.md rule 13) — real stores only.

    python -m mu_eval baseline --dataset ~/mu_eval_data/locomo10.json --samples 2 --out run.json
    python -m mu_eval fuse     --dataset ~/mu_eval_data/locomo10.json --out fuse.json
    python -m mu_eval judge-control --dataset ... --base-url http://127.0.0.1:11435/v1

Every command prints a human-readable table to stdout AND writes the full machine-readable record
to ``--out``. Nothing is tuned here: this is a measurement instrument, and a knob added to make a
number look better belongs in a separate, argued change.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from mu_eval.locomo import load_locomo

# The SAFE way to hand this CLI a real Azure/OpenAI-compatible API key (SECURITY FIX, verified
# live: `eval/vm_eval.sh` used to take `--api-key <value>` as a bare CLI argument and `printf %q`
# it straight into the remote `ssh ... "... python -m mu_eval ..."` command string — which put the
# key in THIS process's own argv (`ps`/`/proc/<pid>/cmdline`, world-readable on a shared box) for
# the whole run's duration, in the VM's process table the SAME way, and in vm_eval.sh's own
# `echo "running: python -m mu_eval$ARGS"` stdout log verbatim. An env var sits in the process's
# `environ` block instead — owner-readable only, never in argv, never echoed by that log line
# (`vm_eval.sh` now transfers it out-of-band; see that script's own comment). DEV-STANDARDS
# security: "secrets only from the secret seam — never in logs, errors, commits, or config
# literals" — this is that seam for this harness.
_API_KEY_ENV_VAR = "MU_EVAL_API_KEY"


def _print(line: str = "") -> None:
    sys.stdout.write(line + "\n")


def _resolve_api_key(
    args: argparse.Namespace, *, subcommand: str, default: str | None = None
) -> str:
    """Resolve an API key for ``subcommand``. Precedence: ``MU_EVAL_API_KEY`` (the safe path, see
    the module-level comment above) beats an explicit ``--api-key`` flag (kept working for
    back-compat / one-off local use, but flagged every time it is used — the flag itself is the
    leak this whole fix closes, so choosing it is never silent) beats ``default``.

    ``default`` is the ONE narrowing per caller: ``answer-quality``/``judge-probe`` (both
    ``required=True``'d a key before this fix) call this with ``default=None`` and get a LOUD,
    actionable failure when neither source resolves — never a silently-fabricated key.
    ``judge-control`` (its pre-fix ``--api-key`` default was the literal string ``"unused"``,
    since its own default target is the keyless local SLM sidecar) passes
    ``default="unused"`` so THAT subcommand keeps working with no key configured, while still
    routing through the SAME argv-safe resolution as the other two — a user who DOES point
    ``judge-control`` at a real, key-requiring endpoint gets the identical safe path."""
    from_env = os.environ.get(_API_KEY_ENV_VAR)
    if from_env:
        return from_env
    explicit: str | None = getattr(args, "api_key", None)
    if explicit:
        _print(
            f"  ! '{subcommand}' got its API key from --api-key, not {_API_KEY_ENV_VAR}. "
            f"--api-key is visible in this process's own argv and in vm_eval.sh's run log for "
            f"the whole run; prefer `export {_API_KEY_ENV_VAR}=...` instead (vm_eval.sh forwards "
            f"it out of band — see that script's own header comment)."
        )
        return explicit
    if default is not None:
        return default
    raise SystemExit(
        f"'{subcommand}' needs a real API key and none was found. Set {_API_KEY_ENV_VAR} "
        f"(preferred — never appears in argv or a log; vm_eval.sh reads it from your local "
        f"environment and transfers it to the VM out of band) or pass --api-key explicitly "
        f"(discouraged; see --api-key's own --help text)."
    )


def _table(title: str, agg: dict[str, dict[int, float]], ks: tuple[int, ...]) -> None:
    _print(f"\n{title}")
    _print("  metric      " + "".join(f"@{k:<8}" for k in ks))
    for name in ("recall", "precision", "mrr", "ndcg"):
        cells = "".join(f"{agg[name][k]:<9.4f}" for k in ks)
        _print(f"  {name:<12}{cells}")


def _print_baseline_report(report: Any) -> None:
    arm = report.arms[0]
    _print(f"run_id={report.run_id}  samples={report.samples}  turns={report.corpus_turns}")
    _print(
        f"importance={report.importance}  recall_limit={report.recall_limit}  "
        f"scored={arm.queries_scored}  skipped_adversarial={arm.queries_skipped_adversarial}  "
        f"skipped_no_gold={arm.queries_skipped_no_gold_in_corpus}"
    )
    _table("OVERALL (macro-avg over queries)", arm.overall, arm.ks)
    for name, agg in sorted(arm.by_category.items()):
        _table(f"category {name}", agg, arm.ks)
    if arm.provenance is not None:
        p = arm.provenance
        _print("\nSCORE PROVENANCE (what fused_score actually contained)")
        _print(f"  items={p.items}  zero_scores={p.zero_scores}  distinct={p.distinct_scores}")
        _print(f"  min={p.min_score:.6f}  median={p.median_score:.6f}  max={p.max_score:.6f}")
        _print(f"  floor_items={p.floor_items}  degraded_results={p.degraded_results}")
        _print(f"  by tier/channel: {p.by_channel}")
    for note in report.notes:
        _print(f"NOTE: {note}")


def _headline_recall(report: Any) -> float:
    """The ONE scalar a repeated ``baseline`` run tracks for spread (item 5): recall at the
    widest requested cutoff, overall (macro-avg). Picked because it is the number this harness's
    own docs cite as the headline ranking metric — see ``runner.py``'s module docstring."""
    arm = report.arms[0]
    return float(arm.overall["recall"][arm.ks[-1]])


async def _cmd_baseline(args: argparse.Namespace) -> int:
    from mu_eval.provenance import build_provenance
    from mu_eval.repeats import run_n_times, summarize_repeats
    from mu_eval.runner import run_baseline

    conversations = load_locomo(args.dataset, samples=args.samples)
    run_id_base = args.run_id or uuid.uuid4().hex[:8]

    async def _once(run_id: str) -> Any:
        return await run_baseline(
            conversations=conversations,
            run_id=run_id,
            ks=tuple(args.k),
            recall_limit=args.limit,
            importance=args.importance,
            max_queries_per_sample=args.max_queries,
            tier=args.tier,
            consolidate=args.consolidate,
        )

    provenance = build_provenance(dataset_path=args.dataset)

    if args.num_runs <= 1:
        # UNCHANGED single-run path (byte-for-byte the pre-existing behaviour, minus the new
        # `provenance` key merged into `--out` — item 2, additive only): no repeats wrapper, no
        # different report shape.
        report = await _once(run_id_base)
        _print_baseline_report(report)
        _write(args.out, report.model_dump(mode="json") | {"provenance": provenance})
        return 0

    reports = await run_n_times(num_runs=args.num_runs, run_id_prefix=run_id_base, run_once=_once)
    for report in reports:
        _print_baseline_report(report)
        _print("")
    summary = summarize_repeats(
        metric_name=f"recall@{tuple(args.k)[-1] if args.k else '?'}",
        values=[_headline_recall(r) for r in reports],
        run_ids=[r.run_id for r in reports],
    )
    _print(
        f"REPEATS ({summary.num_runs} runs) — {summary.metric_name}: "
        f"mean={summary.mean:.4f}  stdev={summary.stdev:.4f}  "
        f"min={summary.minimum:.4f}  max={summary.maximum:.4f}  spread={summary.spread:.4f}"
    )
    _print(f"values: {[round(v, 4) for v in summary.values]}")
    _write(
        args.out,
        {
            "runs": [r.model_dump(mode="json") for r in reports],
            "repeat_summary": summary.model_dump(mode="json"),
            "provenance": provenance,
        },
    )
    return 0


async def _cmd_fuse(args: argparse.Namespace) -> int:
    from mu_eval.arms import run_two_arm_fuse

    conversations = load_locomo(args.dataset, samples=args.samples)
    run_id = args.run_id or uuid.uuid4().hex[:8]
    results = []
    for conversation in conversations:
        cmp_ = await run_two_arm_fuse(
            conversation,
            run_id=run_id,
            ks=tuple(args.k),
            recall_limit=args.limit,
            importance=args.importance,
            max_queries=args.max_queries,
        )
        results.append(cmp_.model_dump(mode="json"))
        _print(
            f"\nsample={cmp_.sample_id}  private_turns={cmp_.corpus_private}  "
            f"shared_turns={cmp_.corpus_shared}  queries={cmp_.queries_scored}  "
            f"straddling={cmp_.straddling_queries}"
        )
        _table("private-only", cmp_.private_only, cmp_.ks)
        _table("shared-only", cmp_.shared_only, cmp_.ks)
        _table("FUSED (private ⊕ shared)", cmp_.fused, cmp_.ks)
        _print(f"  fused beats best single arm (recall@k): {cmp_.fused_beats_best_single_arm}")
    _write(args.out, {"run_id": run_id, "comparisons": results})
    return 0


async def _cmd_probe(args: argparse.Namespace) -> int:
    from mu_eval.probe import probe_scores

    record = await probe_scores()
    _print(f"query: {record['query']}")
    _print("\nSURFACE (mu_contracts RecallItemView — what `mu recall` prints):")
    for item in record["surface_items"]:
        _print(
            f"  {item['fused_score']:.6f}  {item['tier']}/{item['channel']}"
            f"{' [floor]' if item['is_floor'] else ''}  {item['content']}"
        )
    _print("\nCROSS-SESSION SURFACE (same corpus, different session — STM window empty):")
    for item in record.get("cross_session_items", []):
        _print(
            f"  {item['fused_score']:.6f}  {item['tier']}/{item['channel']}"
            f"{' [floor]' if item['is_floor'] else ''}  {item['content']}"
        )
    _print("\nENGINE (mu_engine RecallItemView — one layer down):")
    for item in record["engine_items"]:
        _print(f"  {item['fused_score']:.6f}  {item['channel']}  {item['content']}")
    for channel, rows in record["raw_channels"].items():
        _print(f"\nRAW CHANNEL {channel} (score straight from the store adapter):")
        for row in rows[:10]:
            _print(f"  {row['score']:.6f}  {row['content']}")
    _print("\nRRF FUSION SCORE recomputed over those raw channels:")
    for row in record["rrf_scores"]:
        _print(f"  {row['rrf']:.6f}  {row['content']}")
    _write(args.out, record)
    return 0


async def _cmd_probe_promote(args: argparse.Namespace) -> int:
    from mu_eval.probe_promote import probe_promotion_paths

    record = await probe_promotion_paths()
    _print(f"promoted at ingest time? {record['ingest_promoted_at_write']}")
    _print(f"promoted at ingest time (low importance)? {record['lifecycle_promoted_at_write']}")
    _print(f"promote verb: {record['promote_verb']}")
    _print("\nSTORED QDRANT VECTORS:")
    for point in record["stored_points"]:
        _print(
            f"  dim={point['vector_dim']}  nonzero={point['vector_nonzero']}  {point['content']}"
        )
    _print("\nRECALL (cross-session, MTM arm only):")
    for item in record["recall_items"]:
        _print(f"  {item['fused_score']:.6f}  {item['tier']}/{item['channel']}  {item['content']}")
    _write(args.out, record)
    return 0


async def _cmd_judge_control(args: argparse.Namespace) -> int:
    from mu_eval.judge import (
        COMPACT_JUDGE_SYSTEM_PROMPT,
        JUDGE_SYSTEM_PROMPT,
        MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS,
        compact_judge_prompt,
        judge_control_set,
        judge_prompt,
    )
    from mu_eval.openai_chat import OpenAICompatChat, PacedOpenAICompatChat

    conversations = load_locomo(args.dataset, samples=1)
    rows = [q for q in conversations[0].queries if not q.is_adversarial and q.answer][: args.n]
    api_key = _resolve_api_key(args, subcommand="judge-control", default="unused")
    raw_chat = OpenAICompatChat(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        completion_tokens_field=args.completion_tokens_field,
        omit_temperature=args.omit_temperature,
    )
    chat = PacedOpenAICompatChat(raw_chat, min_interval_s=args.min_interval_s)
    system_prompt = COMPACT_JUDGE_SYSTEM_PROMPT if args.compact else JUDGE_SYSTEM_PROMPT
    build_prompt = compact_judge_prompt if args.compact else judge_prompt
    # An explicit --max-completion-tokens always wins, even in --compact mode: the Ministral-3B
    # default (6) assumes a bare-label completion with no hidden cost, which holds for Ministral
    # but not for a reasoning model (gpt-5 spends completion tokens on hidden reasoning tokens
    # before the visible label — 6 is not enough budget and returns empty content). Passing
    # --max-completion-tokens overrides the Ministral default for exactly this case; omitting it
    # keeps the original compact-mode behaviour unchanged.
    if args.max_completion_tokens is not None:
        max_tokens = args.max_completion_tokens
    elif args.compact:
        max_tokens = MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS
    else:
        max_tokens = None
    try:
        result = await judge_control_set(
            chat,
            questions=[r.question for r in rows],
            gold_answers=[r.answer for r in rows],
            system_prompt=system_prompt,
            build_prompt=build_prompt,
            max_tokens=max_tokens,
        )
    finally:
        await raw_chat.aclose()
    _print(f"judge model: {args.model} @ {args.base_url}  compact={args.compact}")
    _print(
        f"  positives  {result.positives_correct}/{result.positives} graded CORRECT "
        f"(gold answer handed back as the generated answer)"
    )
    _print(
        f"  negatives  {result.negatives_wrong}/{result.negatives} graded WRONG "
        f"(gold answer of a DIFFERENT question)"
    )
    _print(f"  unparseable judgements: {result.unparseable}")
    _print(f"  USABLE AS A HEADLINE JUDGE: {result.usable}")
    _write(args.out, result.model_dump(mode="json") | {"model": args.model})
    return 0 if result.usable else 2


async def _cmd_judge_probe(args: argparse.Namespace) -> int:
    """ONE real judge call, end to end — proves the compact prompt fits the 400-token ceiling
    against the LIVE usage block and rate-limit headers, rather than asserting it from a local
    tokenizer estimate. Makes no more than one HTTP request."""
    from mu_eval.judge import (
        COMPACT_JUDGE_SYSTEM_PROMPT,
        compact_judge_prompt,
        estimated_prompt_tokens,
    )
    from mu_eval.openai_chat import OpenAICompatChat

    user = compact_judge_prompt(
        question=args.question, gold_answer=args.gold_answer, response=args.response
    )
    estimate = estimated_prompt_tokens(COMPACT_JUDGE_SYSTEM_PROMPT, user)
    _print(f"local pre-flight estimate (prompt only): {estimate} tokens")
    _print(f"\nsystem:\n{COMPACT_JUDGE_SYSTEM_PROMPT}\n\nuser:\n{user}\n")

    api_key = _resolve_api_key(args, subcommand="judge-probe")
    chat = OpenAICompatChat(base_url=args.base_url, model=args.model, api_key=api_key)
    try:
        content = await chat.complete(
            system=COMPACT_JUDGE_SYSTEM_PROMPT, user=user, max_tokens=args.max_completion_tokens
        )
    finally:
        await chat.aclose()

    _print(f"model verdict: {content!r}")
    _print(f"\nusage block: {chat.last_usage}")
    headers = chat.last_headers or {}
    for name in (
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ):
        if name in headers:
            _print(f"  {name}: {headers[name]}")
    _write(
        args.out,
        {
            "model": args.model,
            "usage": chat.last_usage,
            "headers": {k: v for k, v in headers.items() if k.startswith("x-ratelimit")},
            "verdict": content,
        },
    )
    return 0


def _print_answer_quality_report(report: Any, *, elapsed: float) -> None:
    _print(
        f"run_id={report.run_id}  samples={report.samples}  "
        f"answer_model={report.answer_model}  judge_model={report.judge_model}  "
        f"elapsed={elapsed:.0f}s"
    )
    _print(
        f"scored={report.queries_scored}  skipped_adversarial={report.skipped_adversarial}  "
        f"skipped_no_gold={report.skipped_no_gold_in_corpus}"
    )
    o = report.overall
    _print(
        f"\nOVERALL  correct={o.correct}  wrong={o.wrong}  unparseable={o.unparseable}  "
        f"accuracy(parseable)={o.accuracy:.4f} ({o.accuracy * 100:.1f}%)  "
        f"accuracy(all-scoreable)={o.accuracy_all_scoreable:.4f} "
        f"({o.accuracy_all_scoreable * 100:.1f}%)"
    )
    _print(
        f"  gold_in_context={o.gold_retrieved}/{o.n}  "
        f"wrong-and-retrieved(generation failure)={o.wrong_retrieved}  "
        f"wrong-and-not-retrieved(retrieval failure)={o.wrong_not_retrieved}"
    )
    for name, stats in sorted(report.by_category.items()):
        _print(
            f"  {name:<24} n={stats.n:<5} correct={stats.correct:<5} wrong={stats.wrong:<5} "
            f"unparseable={stats.unparseable:<3} accuracy={stats.accuracy:.4f} "
            f"gold_retrieved={stats.gold_retrieved:<4} "
            f"wrong_retrieved={stats.wrong_retrieved:<4} "
            f"wrong_not_retrieved={stats.wrong_not_retrieved:<4}"
        )


async def _cmd_answer_quality(args: argparse.Namespace) -> int:
    import time

    from mu_eval.answer_quality import run_answer_quality
    from mu_eval.openai_chat import OpenAICompatChat
    from mu_eval.provenance import build_provenance
    from mu_eval.repeats import run_n_times, summarize_repeats

    conversations = load_locomo(args.dataset, samples=args.samples)
    run_id_base = args.run_id or uuid.uuid4().hex[:8]
    recall_limit = args.limit or max(args.k)
    api_key = _resolve_api_key(args, subcommand="answer-quality")
    answer_chat = OpenAICompatChat(
        base_url=args.base_url,
        model=args.answer_model,
        api_key=api_key,
        completion_tokens_field=args.completion_tokens_field,
        omit_temperature=args.omit_temperature,
    )
    judge_chat = OpenAICompatChat(
        base_url=args.base_url,
        model=args.judge_model,
        api_key=api_key,
        completion_tokens_field=args.completion_tokens_field,
        omit_temperature=args.omit_temperature,
    )
    started = time.monotonic()

    def _progress(done: int, total: int) -> None:
        if done == 1 or done == total or done % 25 == 0:
            elapsed = time.monotonic() - started
            _print(f"  ... {done}/{total} graded ({elapsed:.0f}s elapsed)")

    async def _once(run_id: str) -> Any:
        return await run_answer_quality(
            conversations=conversations,
            run_id=run_id,
            answer_chat=answer_chat,
            judge_chat=judge_chat,
            answer_model=args.answer_model,
            judge_model=args.judge_model,
            recall_limit=recall_limit,
            importance=args.importance,
            max_queries_per_sample=args.max_queries,
            answer_max_tokens=args.answer_max_tokens,
            judge_max_tokens=args.judge_max_tokens,
            concurrency=args.concurrency,
            keep_rows=not args.no_rows,
            progress=_progress,
            consolidate=args.consolidate,
        )

    try:
        # PROVENANCE captured AFTER the run(s): `build_provenance`'s model_provenance reads
        # `chat.served_models`, which is only populated once at least one real response has come
        # back — capturing it earlier would report an empty `served` list even on a successful run.
        if args.num_runs <= 1:
            # UNCHANGED single-run path: no repeats wrapper, no different --out shape — the ONLY
            # addition versus the pre-existing behaviour is the merged `provenance` key (additive).
            report = await _once(run_id_base)
            elapsed = time.monotonic() - started
            _print_answer_quality_report(report, elapsed=elapsed)
            provenance = build_provenance(
                dataset_path=args.dataset,
                chats={"answer": answer_chat, "judge": judge_chat},
            )
            _write(args.out, report.model_dump(mode="json") | {"provenance": provenance})
            return 0

        reports = await run_n_times(
            num_runs=args.num_runs, run_id_prefix=run_id_base, run_once=_once
        )
        for report in reports:
            elapsed = time.monotonic() - started
            _print_answer_quality_report(report, elapsed=elapsed)
            _print("")
        summary = summarize_repeats(
            metric_name="overall.accuracy",
            values=[r.overall.accuracy for r in reports],
            run_ids=[r.run_id for r in reports],
        )
        _print(
            f"REPEATS ({summary.num_runs} runs) — {summary.metric_name}: "
            f"mean={summary.mean:.4f}  stdev={summary.stdev:.4f}  "
            f"min={summary.minimum:.4f}  max={summary.maximum:.4f}  spread={summary.spread:.4f}"
        )
        _print(f"values: {[round(v, 4) for v in summary.values]}")
        provenance = build_provenance(
            dataset_path=args.dataset,
            chats={"answer": answer_chat, "judge": judge_chat},
        )
        _write(
            args.out,
            {
                "runs": [r.model_dump(mode="json") for r in reports],
                "repeat_summary": summary.model_dump(mode="json"),
                "provenance": provenance,
            },
        )
        return 0
    finally:
        await answer_chat.aclose()
        await judge_chat.aclose()


async def _cmd_compare(args: argparse.Namespace) -> int:
    """Item 4: 'did this actually change anything', paired per-row — never two aggregates
    subtracted. Reads two ``answer-quality --out`` JSON artifacts (each must have been written
    WITHOUT ``--no-rows``, since the pairing needs per-row verdicts) and reports flips in both
    directions, McNemar's p-value, and names the query ids that regressed. Exit code is 0
    regardless of the verdict (this is a report, not a gate) — a caller scripting a gate on top
    reads ``significant_at_p05``/``delta`` from ``--out``.
    """
    from mu_eval.answer_quality import AnswerQualityReport
    from mu_eval.compare import compare_runs

    report_a = AnswerQualityReport.model_validate_json(Path(args.a).read_text(encoding="utf-8"))
    report_b = AnswerQualityReport.model_validate_json(Path(args.b).read_text(encoding="utf-8"))
    result = compare_runs(report_a, report_b)

    _print(f"A={result.run_a}  B={result.run_b}")
    _print(
        f"common_rows={result.common_rows}  only_in_a={result.only_in_a}  "
        f"only_in_b={result.only_in_b}"
    )
    _print(
        f"A accuracy: parseable={result.a_accuracy:.4f}  "
        f"all-scoreable={result.a_accuracy_all_scoreable:.4f}"
    )
    _print(
        f"B accuracy: parseable={result.b_accuracy:.4f}  "
        f"all-scoreable={result.b_accuracy_all_scoreable:.4f}"
    )
    _print(f"delta (B-A, parseable): {result.delta:+.4f}")
    _print(
        f"fixed (wrong->correct): {len(result.fixed)}   "
        f"regressed (correct->wrong): {len(result.regressed)}"
    )
    _print(
        f"unchanged_correct={result.unchanged_correct}  "
        f"unchanged_wrong={result.unchanged_wrong}"
    )
    _print(
        f"became_unparseable={len(result.became_unparseable)}  "
        f"recovered_from_unparseable={len(result.recovered_from_unparseable)}  "
        f"both_unparseable={result.both_unparseable}"
    )
    _print(
        f"McNemar: n(A correct,B wrong)={result.mcnemar_n_a_correct_b_wrong}  "
        f"n(A wrong,B correct)={result.mcnemar_n_a_wrong_b_correct}  "
        f"p={result.mcnemar_p_value:.4f}  significant_at_p05={result.significant_at_p05}"
    )
    if result.regressed:
        _print(f"REGRESSED query ids: {result.regressed}")
    if result.fixed:
        _print(f"FIXED query ids: {result.fixed}")
    _print(f"\n{result.verdict}")
    _write(args.out, result.model_dump(mode="json"))
    return 0


def _write(out: str | None, payload: dict[str, Any]) -> None:
    if not out:
        return
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _print(f"\nwrote {path}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mu_eval", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--dataset", required=True, help="path to locomo10.json")
        sp.add_argument("--samples", type=int, default=None, help="cap conversations (default all)")
        sp.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
        sp.add_argument("--limit", type=int, default=None, help="recall limit (default max(k))")
        sp.add_argument("--importance", type=float, default=0.9)
        sp.add_argument("--max-queries", type=int, default=None)
        sp.add_argument("--run-id", default=None)
        sp.add_argument("--out", default=None)

    bl = sub.add_parser("baseline", help="ranked-recall baseline over real stores")
    common(bl)
    bl.add_argument(
        "--tier",
        choices=["stm", "mtm", "ltm"],
        default=None,
        help="DIAGNOSTIC: narrow recall to one channel instead of the shipped 3-channel fuse.",
    )
    bl.add_argument(
        "--consolidate",
        action="store_true",
        help="run LocalMemory.consolidate() (MTM->LTM DISTILL) once per conversation after "
        "ingest, before any query runs, so the LTM graph tier is actually populated. Without "
        "this flag (the harness's prior, and still default, behaviour) the graph tier is "
        "UNCONDITIONALLY EMPTY for the whole run — no eval command called consolidate() before.",
    )
    bl.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="run the WHOLE pipeline (fresh ingest, fresh --consolidate if set) this many times "
        "and report mean/stdev/spread of the headline recall@max(k) alongside every individual "
        "run (item 5: variance made visible). Default 1 preserves the exact pre-existing "
        "single-run output and --out shape.",
    )
    common(sub.add_parser("fuse", help="private ⊕ shared two-arm fuse comparison"))

    pr = sub.add_parser("probe-scores", help="the 0.0000 investigation: raw vs fused vs surface")
    pr.add_argument("--out", default=None)

    pp = sub.add_parser(
        "probe-promotion", help="which promotion path writes a ZERO vector into the vector tier"
    )
    pp.add_argument("--out", default=None)

    jc = sub.add_parser("judge-control", help="validate the LLM judge before believing it")
    jc.add_argument("--dataset", required=True)
    jc.add_argument("--base-url", default="http://127.0.0.1:11435/v1")
    jc.add_argument("--model", default="qwen2.5:0.5b")
    jc.add_argument(
        "--api-key",
        default=None,
        help=f"discouraged for a real key — visible in argv/logs; prefer {_API_KEY_ENV_VAR}. "
        "Unset resolves to 'unused' (the default local-SLM target needs no key).",
    )
    jc.add_argument("-n", type=int, default=20)
    jc.add_argument("--out", default=None)
    jc.add_argument(
        "--compact",
        action="store_true",
        help="use the token-budget-fit prompt (compact_judge_prompt) + capped completion, "
        "for providers with a hard per-request token ceiling (e.g. Azure Ministral-3B, 400 total)",
    )
    jc.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="cap completion tokens when NOT --compact (compact mode always uses "
        "MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS)",
    )
    jc.add_argument(
        "--min-interval-s",
        type=float,
        default=0.0,
        help="minimum seconds between request STARTS, enforced by PacedOpenAICompatChat "
        "(pass 60 for Azure Ministral-3B's 1-request/60s quota; 0 = no artificial pacing, "
        "still obeys 429 Retry-After reactively)",
    )
    jc.add_argument(
        "--completion-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_tokens",
        help="JSON field name for the completion-token cap. gpt-5 (and other reasoning-family "
        "models) reject 'max_tokens' with a 400 and require 'max_completion_tokens' instead; "
        "LiteLLM translates this for azure/gpt-5 automatically, but this hand-rolled client "
        "talking to the raw {base}/openai/v1/chat/completions route must send the field the "
        "deployment actually accepts.",
    )
    jc.add_argument(
        "--omit-temperature",
        action="store_true",
        help="do not send a 'temperature' field at all. gpt-5 (and other reasoning-family "
        "models) 400 on any non-default temperature ('Unsupported value: temperature does not "
        "support 0.0 with this model. Only the default (1) value is supported.', verified live "
        "2026-08-31) — this omits the field so the deployment's own default applies.",
    )

    aq = sub.add_parser(
        "answer-quality",
        help="retrieval -> answer generation -> OFFICIAL LoCoMo judge, end to end",
    )
    common(aq)
    aq.add_argument("--base-url", required=True)
    aq.add_argument(
        "--api-key",
        default=None,
        help=f"discouraged — visible in argv/logs; prefer the {_API_KEY_ENV_VAR} env var",
    )
    aq.add_argument("--answer-model", required=True)
    aq.add_argument("--judge-model", required=True)
    aq.add_argument("--answer-max-tokens", type=int, default=400)
    aq.add_argument("--judge-max-tokens", type=int, default=300)
    aq.add_argument("--concurrency", type=int, default=12)
    aq.add_argument(
        "--completion-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_completion_tokens",
    )
    aq.add_argument("--omit-temperature", action="store_true", default=True)
    aq.add_argument("--no-rows", action="store_true", help="omit per-query rows from --out")
    aq.add_argument(
        "--consolidate",
        action="store_true",
        help="same as baseline's --consolidate: populate the LTM graph tier before answering, "
        "instead of leaving it unconditionally empty.",
    )
    aq.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="run the WHOLE pipeline (fresh ingest, fresh LLM-generated answers/judgements, "
        "fresh --consolidate if set) this many times and report mean/stdev/spread of overall "
        "accuracy alongside every individual run (item 5). This is the option that would have "
        "caught the 14.4%%/17.3%% multi-hop inversion before it was reported as a finding. "
        "Default 1 preserves the exact pre-existing single-run output and --out shape.",
    )

    cmp_p = sub.add_parser(
        "compare",
        help="paired per-row comparison of two answer-quality artifacts (item 4) — 'did this "
        "actually change anything', not two aggregates subtracted",
    )
    cmp_p.add_argument("--a", required=True, help="path to run A's --out JSON (answer-quality)")
    cmp_p.add_argument("--b", required=True, help="path to run B's --out JSON (answer-quality)")
    cmp_p.add_argument("--out", default=None)

    jp = sub.add_parser(
        "judge-probe",
        help="ONE real judge call — proves the compact prompt fits against live usage/headers",
    )
    jp.add_argument("--base-url", default="https://libo-ai.services.ai.azure.com/openai/v1")
    jp.add_argument("--model", default="Ministral-3B")
    jp.add_argument(
        "--api-key",
        default=None,
        help=f"discouraged — visible in argv/logs; prefer the {_API_KEY_ENV_VAR} env var",
    )
    jp.add_argument("--question", required=True)
    jp.add_argument("--gold-answer", required=True)
    jp.add_argument("--response", required=True)
    jp.add_argument("--max-completion-tokens", type=int, default=6)
    jp.add_argument("--out", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "baseline": _cmd_baseline,
        "fuse": _cmd_fuse,
        "probe-scores": _cmd_probe,
        "probe-promotion": _cmd_probe_promote,
        "judge-control": _cmd_judge_control,
        "judge-probe": _cmd_judge_probe,
        "answer-quality": _cmd_answer_quality,
        "compare": _cmd_compare,
    }
    return asyncio.run(handlers[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
