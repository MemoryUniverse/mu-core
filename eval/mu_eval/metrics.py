"""Ranking metrics — recall@k, precision@k, MRR@k, nDCG@k over binary relevance.

These are the TEXTBOOK definitions, not invented ones. LoCoMo ships an official *answer* scorer
(an LLM judge — see ``judge.py``) but NO official *retrieval* scorer: neither
``other_repos/MemOS/evaluation/scripts/locomo/`` nor ``other_repos/mem0/evaluation/`` contains a
recall@k / nDCG implementation (verified by grep, 2026-08-29). So the honest split, and the one
this harness reports:

  * The RELEVANCE LABELS are the dataset's own (``qa[i].evidence`` turn ids) — nothing invented.
  * The METRICS are the standard IR definitions below — nothing invented, but explicitly NOT
    "the LoCoMo official retrieval score", because no such thing exists.
  * The OFFICIAL LLM-judge answer accuracy (``judge.py``) is the headline where a judge model
    good enough to pass its own control set is available; lexical overlap is a diagnostic, never
    the score (owner's standing rule, project memory "eval — use official methodology").

Definitions (binary relevance ``rel(d) ∈ {0,1}``, ranked list ``R = [d_1..d_k]``, gold set ``G``):

    recall@k    = |{d ∈ R_k : d ∈ G}| / |G|
    precision@k = |{d ∈ R_k : d ∈ G}| / k
    RR@k        = 1 / rank of the first relevant item in R_k, else 0
    DCG@k       = Σ_{i=1..k} rel(d_i) / log2(i + 1)
    IDCG@k      = Σ_{i=1..min(|G|,k)} 1 / log2(i + 1)
    nDCG@k      = DCG@k / IDCG@k      (0.0 when IDCG@k == 0)

``rank`` is 1-based. A query with an empty gold set contributes nothing and MUST be filtered by
the caller (``LabelledQuery.is_adversarial``) — averaging a 0.0 over unanswerable rows would
depress every number for a reason that has nothing to do with ranking.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict

__all__ = [
    "QueryScores",
    "aggregate",
    "dcg",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]


def _hits(retrieved: Sequence[str], gold: Iterable[str], k: int) -> list[bool]:
    gold_set = set(gold)
    return [doc in gold_set for doc in retrieved[:k]]


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall@k is undefined for an empty gold set — filter the query out")
    found = {doc for doc in retrieved[:k] if doc in gold_set}
    return len(found) / len(gold_set)


def precision_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be >= 1")
    return sum(_hits(retrieved, gold, k)) / k


def mrr_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    for rank, hit in enumerate(_hits(retrieved, gold, k), start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def dcg(hits: Sequence[bool]) -> float:
    return sum(1.0 / math.log2(i + 1) for i, hit in enumerate(hits, start=1) if hit)


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("nDCG@k is undefined for an empty gold set — filter the query out")
    ideal = dcg([True] * min(len(gold_set), k))
    if ideal == 0.0:
        return 0.0
    # Duplicate retrieved ids must not each earn a gain: a system that returns the same gold turn
    # three times has not found three relevant things. Credit each gold id at most once, at its
    # best (earliest) rank.
    seen: set[str] = set()
    hits: list[bool] = []
    for doc in retrieved[:k]:
        hit = doc in gold_set and doc not in seen
        if hit:
            seen.add(doc)
        hits.append(hit)
    return dcg(hits) / ideal


class QueryScores(BaseModel):
    """Per-query metric row — kept per query so the report can slice by LoCoMo category."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    category: int
    gold_size: int
    retrieved: int
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    mrr_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]


def score_query(
    *,
    query_id: str,
    category: int,
    retrieved: Sequence[str],
    gold: Sequence[str],
    ks: Sequence[int],
) -> QueryScores:
    return QueryScores(
        query_id=query_id,
        category=category,
        gold_size=len(set(gold)),
        retrieved=len(retrieved),
        recall_at_k={k: recall_at_k(retrieved, gold, k) for k in ks},
        precision_at_k={k: precision_at_k(retrieved, gold, k) for k in ks},
        mrr_at_k={k: mrr_at_k(retrieved, gold, k) for k in ks},
        ndcg_at_k={k: ndcg_at_k(retrieved, gold, k) for k in ks},
    )


def aggregate(rows: Sequence[QueryScores], ks: Sequence[int]) -> dict[str, dict[int, float]]:
    """Macro-average over queries (each query weighs the same, the LoCoMo convention)."""
    if not rows:
        return {name: dict.fromkeys(ks, 0.0) for name in ("recall", "precision", "mrr", "ndcg")}
    n = len(rows)
    return {
        "recall": {k: sum(r.recall_at_k[k] for r in rows) / n for k in ks},
        "precision": {k: sum(r.precision_at_k[k] for r in rows) / n for k in ks},
        "mrr": {k: sum(r.mrr_at_k[k] for r in rows) / n for k in ks},
        "ndcg": {k: sum(r.ndcg_at_k[k] for r in rows) / n for k in ks},
    }
