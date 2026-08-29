"""``mu_eval`` — the retrieval-quality evaluation harness for the open engine.

CLAUDE.md's boundary rule states that mu-core ships "good ranking/salience" and that FULL-LOCAL is
"a complete, good, on-device system — never a crippled baseline". DEV-STANDARDS makes that
checkable rather than rhetorical: "Evaluate retrieval/extraction quality on REAL data with a
repeatable harness — not vibes; use the dataset's official scorer where one exists (never invent a
metric to look good)."

This package is that harness. It is deliberately NOT a shipped distribution: it lives at
``mu-core/eval/`` rather than under ``packages/``, so that (a) no wheel ever gains a dependency on
a benchmark dataset or a judge model, (b) the ``.importlinter`` layer contracts that govern the
four shipped packages stay untouched, and (c) it can import ``mu_local`` (the top-of-stack embed)
freely, which no shipped package below it is allowed to do.

Modules:
  ``locomo``   real labelled dataset loader (turn-level gold evidence ids)
  ``metrics``  recall@k / precision@k / MRR@k / nDCG@k over binary relevance
  ``judge``    the OFFICIAL LoCoMo answer prompt + LLM judge, ported verbatim, plus a control set
  ``corpus``   ingest through ``LocalMemory.add`` into the REAL store stack
  ``runner``   the baseline run + score provenance
  ``arms``     the private ⊕ shared two-arm fuse experiment
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
