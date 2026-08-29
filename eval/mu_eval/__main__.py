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
import sys
import uuid
from pathlib import Path
from typing import Any

from mu_eval.locomo import load_locomo


def _print(line: str = "") -> None:
    sys.stdout.write(line + "\n")


def _table(title: str, agg: dict[str, dict[int, float]], ks: tuple[int, ...]) -> None:
    _print(f"\n{title}")
    _print("  metric      " + "".join(f"@{k:<8}" for k in ks))
    for name in ("recall", "precision", "mrr", "ndcg"):
        cells = "".join(f"{agg[name][k]:<9.4f}" for k in ks)
        _print(f"  {name:<12}{cells}")


async def _cmd_baseline(args: argparse.Namespace) -> int:
    from mu_eval.runner import run_baseline

    conversations = load_locomo(args.dataset, samples=args.samples)
    run_id = args.run_id or uuid.uuid4().hex[:8]
    report = await run_baseline(
        conversations=conversations,
        run_id=run_id,
        ks=tuple(args.k),
        recall_limit=args.limit,
        importance=args.importance,
        max_queries_per_sample=args.max_queries,
        tier=args.tier,
    )
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
    _write(args.out, report.model_dump(mode="json"))
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
    from mu_eval.judge import judge_control_set
    from mu_eval.openai_chat import OpenAICompatChat

    conversations = load_locomo(args.dataset, samples=1)
    rows = [q for q in conversations[0].queries if not q.is_adversarial and q.answer][: args.n]
    chat = OpenAICompatChat(base_url=args.base_url, model=args.model, api_key=args.api_key)
    try:
        result = await judge_control_set(
            chat,
            questions=[r.question for r in rows],
            gold_answers=[r.answer for r in rows],
        )
    finally:
        await chat.aclose()
    _print(f"judge model: {args.model} @ {args.base_url}")
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
    jc.add_argument("--api-key", default="unused")
    jc.add_argument("-n", type=int, default=20)
    jc.add_argument("--out", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    handlers = {
        "baseline": _cmd_baseline,
        "fuse": _cmd_fuse,
        "probe-scores": _cmd_probe,
        "probe-promotion": _cmd_probe_promote,
        "judge-control": _cmd_judge_control,
    }
    return asyncio.run(handlers[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
