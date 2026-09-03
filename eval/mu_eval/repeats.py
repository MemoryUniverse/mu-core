"""Repeats as a first-class instrument: run N times, report the SPREAD alongside the mean.

Built for exactly one measured failure (HOW-THEY-MEASURE-0901.md item 5 / the brief's own
opening claim): the SAME configuration, measured twice, gave 14.4% and 17.3% on the multi-hop
category — a 2.9pt spread, LARGER than the +2.7pt delta that was reported as a finding. Nobody
had run it twice before quoting it once. This module makes "run it twice (or N times)" a normal
option of the CLI instead of a thing a careful person has to remember to do by hand, and makes the
spread part of the printed output rather than something a reader has to reconstruct from raw runs.

Deliberately dumb: no bootstrap, no confidence interval — just the empirical min/max/mean/stdev
of N real numbers, because a "the delta is inside the observed spread" check needs nothing fancier
to be honest, and a fancier stated guarantee (a CI assuming some distribution) would be a claim
this harness has not earned about a metric with N as small as 2 or 3.

Comparable systems' repeat features, and why this one is shaped differently (module docstring of
``compare.py`` has the fuller citation trail): A-mem/HippoRAG cache the corpus/LLM-call layer and
"repeat" only the cheap parts underneath; MemOS's own ``--num_runs`` (default 3,
``locomo_eval.py:339-350``) repeats ONLY the judge, loading previously-generated responses from
disk. Neither would have caught the defect this module exists for, because the defect was in the
INGEST/CONSOLIDATE layer both of those skip on a repeat. This harness's repeats (wired in
``__main__.py`` / ``answer_quality.run_answer_quality_repeated``) re-run the WHOLE pipeline —
fresh ingest, fresh LLM-driven LTM derivation when ``--consolidate`` is set — every time, which is
strictly more expensive and is the only shape that actually measures what MemOS's own shape
cannot.
"""

from __future__ import annotations

import statistics
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

__all__ = ["RepeatSummary", "run_n_times", "summarize_repeats"]

R = TypeVar("R")


class RepeatSummary(BaseModel):
    """Mean + spread over N runs of ONE scalar metric. Never report the mean alone — a single
    number invites exactly the "reported once, believed forever" failure this module exists to
    end."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str
    num_runs: int
    values: list[float]
    run_ids: list[str]
    mean: float
    # Sample standard deviation (``statistics.stdev``, N-1 denominator) — undefined for a single
    # observation, reported as 0.0 rather than raising: a 1-run "summary" is a degenerate but
    # legitimate call (the CLI's own default), and 0.0 spread/stdev is the honest value for it
    # (there is no variation to report), not a masked error.
    stdev: float
    minimum: float
    maximum: float
    spread: float  # maximum - minimum; the number a reader should compare a delta against

    def is_within_spread(self, delta: float) -> bool:
        """True when ``abs(delta)`` does not exceed the run-to-run spread THIS instrument
        measured — i.e. a delta this small cannot be told apart from noise using only these N
        runs. Callers should print this explicitly next to any delta rather than let the reader
        infer it (item 5's own charge: "the tool should say so in its own output")."""
        return abs(delta) <= self.spread


def summarize_repeats(
    *, metric_name: str, values: Sequence[float], run_ids: Sequence[str]
) -> RepeatSummary:
    if not values:
        raise ValueError("summarize_repeats needs at least one run's value")
    if len(values) != len(run_ids):
        raise ValueError(
            f"values ({len(values)}) and run_ids ({len(run_ids)}) must be the same length"
        )
    vals = [float(v) for v in values]
    return RepeatSummary(
        metric_name=metric_name,
        num_runs=len(vals),
        values=vals,
        run_ids=list(run_ids),
        mean=statistics.fmean(vals),
        stdev=statistics.stdev(vals) if len(vals) > 1 else 0.0,
        minimum=min(vals),
        maximum=max(vals),
        spread=max(vals) - min(vals),
    )


async def run_n_times(
    *,
    num_runs: int,
    run_id_prefix: str,
    run_once: Callable[[str], Awaitable[R]],
) -> list[R]:
    """Run ``run_once(run_id)`` ``num_runs`` times, sequentially, each under its OWN run id.

    Sequential (not gathered) on purpose: each call already ingests a real corpus into a real,
    isolated store partition (``corpus.local_memory_for``) and, when the caller's ``run_once``
    passes ``consolidate=True`` through, fires an LLM-driven LTM derivation per run — the exact
    thing whose non-determinism this module exists to surface. Running N of those concurrently
    would additionally contend the shared VM stores/LLM rate limits in a way that changes what is
    being measured (``answer_quality.py``'s own module docstring: a self-inflicted load spike is
    not a system defect). A distinct run id per repeat (``{run_id_prefix}-r{i}``) is what makes
    each one a genuinely independent corpus/partition rather than a second query against the
    first run's already-ingested one.
    """
    if num_runs < 1:
        raise ValueError(f"num_runs must be >= 1, got {num_runs}")
    results: list[R] = []
    for i in range(num_runs):
        results.append(await run_once(f"{run_id_prefix}-r{i}"))
    return results
