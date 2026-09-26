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
    ks: Sequence[int] | None = None,
) -> list[tuple[T, float]]:
    """Fuse ranked lists into one weighted RRF ranking (ported fusion.py:17, generalised).

    ``channel_results`` — one ranked list per channel/arm, each already best-first. ``key`` — the
    stable identity of an element (``id`` / ``memory_id``) used to aggregate the same element seen
    in multiple channels. ``weights`` — one non-negative weight per list, same order/length. ``k``
    — the RRF constant (default 60): larger smooths rank aggregation. The union of all elements is
    returned ``(element, fused_score)`` sorted by fused score DESC; an element in only one channel
    still scores (fusion never requires multi-channel agreement). The FIRST occurrence of a key
    (by channel order) is the representative element kept.

    ``ks`` — OPTIONAL, one RRF constant PER CHANNEL instead of the single shared ``k`` (AD-273,
    the fusion-arithmetic lane of ADR 0066's graph-tier settlement — `RecallSettings.rrf_k_ltm`'s
    docstring has the full derivation and the measurement). Every channel's contribution is
    ``weight/(k_channel+rank+1)``: a SHARED ``k`` across channels with very different weights
    structurally caps a heavily-discounted channel's best-POSSIBLE rank-0 contribution below a
    heavily-weighted channel's WORST-pool-item contribution, no matter how genuinely relevant
    that channel's top candidate is (`RecallSettings.weight_ltm`'s field docstring has the
    exhaustive proof for the shipped LTM channel, PROVED exhaustively and reproduced end to end
    four times). A per-channel ``k`` lets one channel's decay curve be steeper than another's —
    its best candidate gets a fair shot at the SAME rank-0 score scale as every other channel's
    best candidate, while its weight discount still governs how quickly it falls off past the
    top few (a small ``weight`` over a small ``k`` decays MUCH faster per rank than a large
    ``weight`` over a large ``k``) — without touching any OTHER channel's already-tuned decay.
    Defaults to ``None``: every channel uses the shared ``k`` — BYTE-IDENTICAL to every call site
    that predates this parameter. When given, must be the same length as
    ``channel_results``/``weights``.
    """
    if len(channel_results) != len(weights):
        raise ValueError("channel_results and weights must have the same length")
    if ks is not None and len(ks) != len(channel_results):
        raise ValueError("ks must be the same length as channel_results")
    if not channel_results:
        return []

    total = sum(weights)
    normalized = [w / total for w in weights] if total > 0 else [0.0 for _ in weights]

    scores: dict[str, float] = {}
    elements: dict[str, T] = {}
    for idx, (weight, results) in enumerate(zip(normalized, channel_results, strict=True)):
        channel_k = ks[idx] if ks is not None else k
        for rank, element in enumerate(results):
            eid = key(element)
            scores[eid] = scores.get(eid, 0.0) + weight * (1.0 / (channel_k + rank + 1))
            elements.setdefault(eid, element)

    # AD-317/AD-318 (2026-09-26): a plain `sorted(scores, key=scores.get, reverse=True)` breaks
    # ties on Python's stable sort, which preserves INSERTION order into `scores` above — i.e.
    # whichever order each channel's own ranked list happened to present its candidates in this
    # call. That per-channel order is NOT itself guaranteed stable across separate calls (a vector
    # store's internal tie-break among equal/near-equal distances can vary run to run), so two
    # identical queries could fuse to a different item COMPOSITION even though every score is
    # byte-identical. MEASURED (AD-317, `ranker.py:401` call site, conv-26, n=150, two separately-
    # executed runs, same `limit=20`): 30/150 rows returned a different non-gold item set. This is
    # a TIE-BREAK fix, not a scoring change — it touches no channel's weight/`k`/rank arithmetic,
    # so it is NOT the fusion-lever territory the register already closed (AD-204/262/273/279/289).
    # `-scores[eid]` keeps the primary DESC-by-score order; `eid` (the memory id, content-
    # independent and stable for a given item) is the deterministic secondary key that breaks any
    # tie the same way on every call, on every process, regardless of channel presentation order.
    # CORRECTED (AD-322, 2026-09-26): AD-318 claimed here that "re-running the same 150-row set
    # twice now returns byte-identical item sets (0/150 differ, was 30/150)". IT DOES NOT. Re-run
    # at width 20 on conv-26 with this sort key mutated out as a control in the SAME session:
    # **24/150 item sets still differ WITH this fix, 23/150 WITHOUT it** (ordered lists 134 vs
    # 136). So this fix removes ONE source of nondeterminism — the dict-insertion-order dependence
    # the unit tests pin, which is real and mutation-verified — and the end-to-end nondeterminism
    # has a DIFFERENT, dominant cause that lives UPSTREAM of this function. Ruled out by arms, not
    # by argument: not this tie-break (the control), not read-path reinforcement
    # (`reinforce_on_recall=false` still differs), not index warm-up (rep3-vs-rep4 differs as much
    # as rep1-vs-rep2). Most plausibly ANN candidate order in the vector channel — NOT verified,
    # do not repeat it as fact. What IS invariant across all six arms and 14 repetitions:
    # `gold_in_context` = 79.33 % to the digit, so the nondeterminism only ever moved which
    # NON-gold candidate won a slot. ADR 0097; eval-runs/2026-09-26-ad322-close-verify/.
    ordered = sorted(scores, key=lambda eid: (-scores[eid], eid))
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
        ks: Sequence[int] | None = None,
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
        ks: Sequence[int] | None = None,
    ) -> list[tuple[T, float]]:
        return reciprocal_rank_fusion(channel_results, key=id_of, weights=weights, k=k, ks=ks)
