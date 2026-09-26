"""AD-308 follow-up: split one `ours_arm.py` export into two `answer_h2h.py`-compatible arm
files, restricted to the temporal-reasoning category (2) — so the re-judge this pass runs is
cheap (22 rows, not ~150) and pairs by construction (both files share the exact same
`recall()` calls; only which date source rendered the context line differs).

`--which harness` copies `context` unchanged (the pre-existing corpus-rejoin, reproducing what
produced the published 77.3%/81.8% numbers). `--which product` renames `context_product` ->
`context` (the shipped product's OWN `valid_at` signal, nothing corpus-side) so
`answer_h2h.py`'s existing `row["context"]` reader needs no change — this script is the only new
file, `answer_h2h.py` itself (a concurrent lane's file) is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.environ.get("H2H_EVAL_DIR", str(Path(__file__).resolve().parent.parent)))

from mem0_h2h import emit

TEMPORAL_CATEGORY = 2  # mu_eval.locomo.CATEGORY_NAMES[2] == "temporal reasoning"

# The EXACT 22 query_ids AD-303/304/305's published 77.3%/81.8% temporal-reasoning headline was
# computed over — the category==2 subset of the 100 rows shared between the mem0 and MU arms in
# `docs/tracking/eval-runs/2026-09-25-h2h-mem0/answer_h2h_n100_rows.json`'s `MU_shipped_k20` key
# (extracted by `[r["query_id"] for r in rows if r["category"] == 2]`, sorted). conv-26 alone has
# 37 category==2 rows among all 150 MU-scoreable questions (the mem0-shared 100-row sample is a
# subset); restricting to these exact 22 makes this pass's number DIRECTLY comparable to the
# published one, row for row — not just same-sized.
ORIGINAL_HEADLINE_QUERY_IDS: frozenset[str] = frozenset(
    {
        "conv-26::q0",
        "conv-26::q1",
        "conv-26::q10",
        "conv-26::q12",
        "conv-26::q16",
        "conv-26::q17",
        "conv-26::q20",
        "conv-26::q21",
        "conv-26::q25",
        "conv-26::q26",
        "conv-26::q28",
        "conv-26::q29",
        "conv-26::q31",
        "conv-26::q33",
        "conv-26::q35",
        "conv-26::q36",
        "conv-26::q41",
        "conv-26::q44",
        "conv-26::q45",
        "conv-26::q49",
        "conv-26::q5",
        "conv-26::q53",
    }
)


def split_temporal_rows(
    contexts: list[dict[str, Any]], *, which: str, headline_only: bool = False
) -> tuple[list[dict[str, Any]], int]:
    """Pure filter+remap. ``headline_only=True`` restricts to exactly
    ``ORIGINAL_HEADLINE_QUERY_IDS`` (the published n=22); ``False`` takes every category==2 row
    conv-26 has (n=37, more statistical power, not directly comparable to the published figure
    row-for-row). ``which="product"`` swaps ``context`` for ``context_product`` (dropping nothing
    else) so the output is a drop-in ``answer_h2h.py`` arm file either way. Returns
    ``(rows, dated_count)`` — ``dated_count`` is always 0 for ``which="harness"`` (the field is
    meaningless there, not computed).
    """
    rows = [c for c in contexts if c["category"] == TEMPORAL_CATEGORY]
    if headline_only:
        rows = [c for c in rows if c["query_id"] in ORIGINAL_HEADLINE_QUERY_IDS]
    out_rows = []
    dated = 0
    for row in rows:
        r = dict(row)
        if which == "product":
            r["context"] = row["context_product"]
            dated += 1 if row.get("product_dated_items", 0) > 0 else 0
        out_rows.append(r)
    return out_rows, dated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--which", choices=["harness", "product"], required=True)
    parser.add_argument(
        "--headline-only",
        action="store_true",
        help="restrict to the exact 22 query_ids the published 77.3%%/81.8%% temporal headline "
        "used, instead of all 37 category==2 rows conv-26 has",
    )
    args = parser.parse_args()

    doc = json.loads(Path(args.inp).read_text())
    out_rows, dated = split_temporal_rows(
        doc["contexts"], which=args.which, headline_only=args.headline_only
    )
    if not out_rows:
        raise SystemExit(f"no category={TEMPORAL_CATEGORY} rows in {args.inp}")

    Path(args.out).write_text(json.dumps({"rows": out_rows}, indent=1), encoding="utf-8")
    coverage = (
        f" ({dated}/{len(out_rows)} have >=1 product-dated item)" if args.which == "product" else ""
    )
    emit(f"{args.which}: wrote {len(out_rows)} temporal rows to {args.out}{coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
