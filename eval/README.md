# `mu-core/eval` — the retrieval-quality harness

CLAUDE.md's boundary rule says mu-core ships **"good ranking/salience"** and that FULL-LOCAL is
**"a complete, good, on-device system — never a crippled baseline"**. `DEV-STANDARDS.md` turns that
into an obligation:

> **Evaluate retrieval/extraction quality on REAL data with a repeatable harness** — not vibes; use
> the dataset's **official scorer** where one exists (never invent a metric to look good).

This directory is that harness. It measures; it does not tune.

## Why it lives here and not under `packages/`

`eval/` is deliberately **not a distribution**:

* no shipped wheel gains a dependency on a benchmark dataset or a judge model;
* the four `.importlinter` layer contracts that govern `mu-contracts` / `mu-engine` / `mu-local` /
  `mu-engine-server` are untouched — nothing here is importable *from* them;
* it may import `mu_local` (the top of the stack) freely, which no shipped package is allowed to do,
  and it must, because the thing under test is the assembled engine rather than a component.

`eval/conftest.py` puts this directory on `sys.path`, which is all `eval/tests/` needs.

## Data

**LoCoMo** (`snap-stanford/locomo`, MIT) — the long-horizon conversational-memory dataset already
provisioned on this machine at
`/home/user/D/abstract_project/mma/data/locomo/locomo10.json` (recorded in
`mma/data/MANIFEST.md`). 10 conversations, 5,882 dialogue turns, 1,986 QA rows.

It is **not committed here** (someone else's data), and the loader **raises** when it is absent —
this harness never falls back to a synthetic corpus.

Every QA row carries `evidence`: a list of dialogue-turn ids (`"D1:3"`). Those are the dataset's own
relevance labels, which is what makes turn-level `recall@k` / `MRR@k` / `nDCG@k` meaningful here.

## Methodology, and what is and is not "official"

| Piece | Provenance |
|---|---|
| Relevance labels | The dataset's own `qa[i].evidence` turn ids. Nothing invented. |
| `recall@k` / `precision@k` / `MRR@k` / `nDCG@k` | Textbook IR definitions (`metrics.py`). **Not** "the LoCoMo official retrieval score" — LoCoMo has none: neither `other_repos/MemOS/evaluation/scripts/locomo/` nor `other_repos/mem0/evaluation/` implements one (verified by grep). |
| Answer prompt | **Verbatim port** of `ANSWER_PROMPT_MEM0`, `MemOS/evaluation/scripts/locomo/prompts.py:1-38`. |
| Answer grader | **Verbatim port** of `locomo_grader`'s system + accuracy prompts, `MemOS/evaluation/scripts/locomo/locomo_eval.py:50-81`, with the official `json.loads(...)["label"]` parse. |

The LLM judge is the headline **only when it passes its own control set** (`judge-control`): the
gold answer handed back to itself must grade CORRECT, and a different row's gold answer must grade
WRONG. A judge that fails that is reported as unusable rather than quietly believed. Lexical
overlap is never the score.

## Running it — on the VM, always

`CLAUDE.md` rule 13. `eval/vm_eval.sh` syncs the repo and the dataset to `mu-dev-vm`, takes the
**shared** `/tmp/.mu_vm_test.lock` (so an eval run never overlaps a full pytest suite), runs
`uv sync`, and executes the CLI there.

```bash
# the 0.0000 investigation: raw channel scores vs the RRF score vs what the surface prints
./eval/vm_eval.sh probe-scores --out /tmp/probe.json

# which promotion path writes a ZERO vector into Qdrant
./eval/vm_eval.sh probe-promotion --out /tmp/pp.json

# the baseline (all 10 conversations, all answerable queries)
./eval/vm_eval.sh baseline --dataset /home/user/mu_eval_data/locomo10.json --out /tmp/base.json

# DIAGNOSTIC arms — labelled as such, never the headline
./eval/vm_eval.sh baseline --dataset ... --samples 2 --tier mtm --out /tmp/mtm.json
MU_EVAL_ENV="MU_RECALL__FLOOR_PROTECT_LIMIT=0" ./eval/vm_eval.sh baseline --dataset ... --samples 2

# does fusing private ⊕ shared beat either arm alone?
./eval/vm_eval.sh fuse --dataset ... --samples 1 --out /tmp/fuse.json

# validate the judge before believing any judge number
./eval/vm_eval.sh judge-control --dataset ... --model qwen2.5:0.5b
```

Unit tests (pure, no stores) go through the sanctioned path too:

```bash
infra/mu-vm/vm_test.sh mu-core eval/tests -q
```

## Knobs that change the numbers, and why they are stated

* `--importance` (default `0.9`). `LocalMemory.add` promotes STM→MTM only when
  `importance_score >= IngestSettings.importance_promote` (0.6). At the **default 0.5 nothing ever
  reaches the vector tier**, and recall collapses to the ~10-item STM recency window. The baseline
  therefore ingests at 0.9 so that what is measured is *ranking over a populated index*, not an
  ingest gate. Run `--importance 0.5` to measure the other thing.
* `--tier` narrows to one channel. A single-channel number is a diagnostic; the shipped behaviour
  is the 3-channel fuse.
* `MU_RECALL__FLOOR_PROTECT_LIMIT` / `MU_RECALL__RRF_K` / `MU_RECALL__WEIGHT_*` reach the engine
  through the normal config tree. Changing one and reporting the better number as "the baseline"
  would be tuning; every non-default arm is labelled with the override that produced it.

## Housekeeping

Every run tears down the Qdrant collections, FalkorDB graphs and Redis keys it created
(`corpus._teardown`). Note that the partition names are **computed** with the engine's own
`collection_name` / `graph_name_for`, because the org/workspace is hashed into them — a
`tag in name` substring sweep (what the integration suite's teardown does) matches nothing and
leaves the collections behind. That was measured, not assumed.

## Results

See `docs/tracking/RETRIEVAL-EVAL-0829.md` for the first baseline and what it says about the
"good ranking" claim.
