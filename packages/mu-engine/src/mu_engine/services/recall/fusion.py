"""Reciprocal Rank Fusion + the ``FusionStrategy`` seam + content-hash dedup.

PORT of ``/home/user/hackathon/memory_universe/shared/retrieval/fusion.py:17``
``reciprocal_rank_fusion`` (rank-based RRF over incomparable channel scores), GENERALISED with a
``key`` extractor so the SAME primitive fuses (a) the in-arm dense+graph channels keyed by
``MemoryItem.id`` and (b) the two federation arms keyed by ``RecallItemView.memory_id``
(recall-service-design §1.3 fuse + §1.6 federate-live). Rank position — not the raw, cross-channel
incomparable score — is what fuses (fusion.py:1-8): a cosine similarity and a graph-hop count are
not on the same scale, only their positions are (CANONICAL §7.9 "one fuse implementation").

Wrapped UNCHANGED behind ``FusionStrategy`` (``reciprocal_rank_fusion`` key) per §1.5; a new fusion
is a ``register()`` call, never an edit to the ranker.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from mu_engine.services.recall.dto import RecallItemView

__all__ = [
    "FusionStrategy",
    "ReciprocalRankFusion",
    "dedup_by_content_hash",
    "reciprocal_rank_fusion",
]

T = TypeVar("T")


def reciprocal_rank_fusion(
    channel_results: Sequence[Sequence[T]],
    *,
    key: Callable[[T], str],
    weights: Sequence[float],
    k: int = 60,
) -> list[tuple[T, float]]:
    """Fuse ranked lists into one weighted RRF ranking (ported fusion.py:17, generalised).

    ``channel_results`` — one ranked list per channel/arm, each already best-first. ``key`` — the
    stable identity of an element (``id`` / ``memory_id``) used to aggregate the same element seen
    in multiple channels. ``weights`` — one non-negative weight per list, same order/length. ``k``
    — the RRF constant (default 60): larger smooths rank aggregation. The union of all elements is
    returned ``(element, fused_score)`` sorted by fused score DESC; an element in only one channel
    still scores (fusion never requires multi-channel agreement). The FIRST occurrence of a key
    (by channel order) is the representative element kept.
    """
    if len(channel_results) != len(weights):
        raise ValueError("channel_results and weights must have the same length")
    if not channel_results:
        return []

    total = sum(weights)
    normalized = [w / total for w in weights] if total > 0 else [0.0 for _ in weights]

    scores: dict[str, float] = {}
    elements: dict[str, T] = {}
    for weight, results in zip(normalized, channel_results, strict=True):
        for rank, element in enumerate(results):
            eid = key(element)
            scores[eid] = scores.get(eid, 0.0) + weight * (1.0 / (k + rank + 1))
            elements.setdefault(eid, element)

    ordered = sorted(scores, key=lambda eid: scores[eid], reverse=True)
    return [(elements[eid], scores[eid]) for eid in ordered]


def dedup_by_content_hash(items: Sequence[RecallItemView]) -> list[RecallItemView]:
    """Collapse duplicate bodies by ``content_hash``, keeping the FIRST (higher-ranked) occurrence
    (federate-live §1.6): a pulled shared→local copy materialised into the PRIVATE plane and its
    still-live SHARED origin surfaced by the shared arm carry the SAME ``content_hash`` (a
    version/dedupe key DISTINCT from the tier-stable id) and must NOT double-count. Order-preserving
    so the RRF ranking is untouched apart from the removed duplicates."""
    seen: set[str] = set()
    out: list[RecallItemView] = []
    for it in items:
        if it.content_hash in seen:
            continue
        seen.add(it.content_hash)
        out.append(it)
    return out


@runtime_checkable
class FusionStrategy(Protocol):
    """The fusion seam (§1.5). ``key`` names the registry entry; ``fuse`` ranks best-first."""

    key: str

    def fuse(
        self,
        channel_results: Sequence[Sequence[T]],
        *,
        id_of: Callable[[T], str],
        weights: Sequence[float],
        k: int,
    ) -> list[tuple[T, float]]: ...


class ReciprocalRankFusion:
    """key="rrf_v1" — wraps :func:`reciprocal_rank_fusion` UNCHANGED (§1.5)."""

    key = "rrf_v1"

    def fuse(
        self,
        channel_results: Sequence[Sequence[T]],
        *,
        id_of: Callable[[T], str],
        weights: Sequence[float],
        k: int,
    ) -> list[tuple[T, float]]:
        return reciprocal_rank_fusion(channel_results, key=id_of, weights=weights, k=k)
