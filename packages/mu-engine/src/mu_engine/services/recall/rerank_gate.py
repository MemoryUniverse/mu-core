"""``RerankGate`` — the rerank strategy seam (recall-service-design.md §1.5, ADR 0010/0023).

PORT of ``/home/user/hackathon/memory_universe/shared/retrieval/rerank.py:340``
``adaptive_rerank_gate`` (floor + relative-to-top cutoff over rerank scores), generalised the same
way ``fusion.py``'s ``reciprocal_rank_fusion`` already generalises the reference's RRF: the
reference gates ``tuple[MemoryNode, float]`` pairs, this gates ``tuple[RecallItemView, float]``
pairs — same algebra, same two rules, byte-identical cutoff math (verified against the reference
source in this session, CODE-ADOPTION-METHODOLOGY rule 1).

**WHY THIS FILE EXISTS (ACCURACY-PLAN-0831.md item 6 / "wire ModelRouter.rerank over the fused
union").** Verified before this change: ``ModelRouter.rerank`` is fully built
(``providers/model_router.py:265``), a local ``BAAI/bge-reranker-v2-m3`` is configured
(``providers/shipped_settings.py:90``), a ``Task.RERANK`` route exists
(``providers/task_map.py:37``), and ``RecallItemView.rerank_score`` has carried the comment "None
when the rerank gate is dark (deferred this phase)" since the DTO was written — but a grep of
``services/`` and ``mu-local/src`` for ``.rerank(`` found NO caller anywhere. The gate was built
and never plugged in. This module is that plug: :class:`AdaptiveRerankGate` is the ``RerankGate``
the design doc names, ``ranker.py`` is its one caller (inserted between the three-channel RRF fuse
and the STM-floor merge — see that module's ``rank()`` for exactly where and why).

**Dark unless a ``RerankProviderPort`` is configured** — byte-identical to no-rerank (ADR 0010
property 1): with ``reranker=None`` (or ``settings.rerank_enabled=False``, the env-overridable
A/B-comparison escape hatch this file's sibling knobs already use, e.g. ``cross_tier_dedup``),
:meth:`AdaptiveRerankGate.apply` returns its input pool completely unchanged — no wasted model
call, no score attached.

**Empty gate is a FALLBACK, not a failure.** When every candidate in the scored pool falls below
``min_score`` the reference's own docstring names this the HippoRAG-style empty-gate fallback
(``HippoRAG.py:417-419``, cited by recall-service-design.md line 202/246): the caller is meant to
keep its PRE-rerank order rather than return nothing. :meth:`AdaptiveRerankGate.apply` implements
that itself (returns the original ``pool``, not ``[]``) so ``ranker.py`` never has to special-case
an empty rerank result — the STM-floor-protect step downstream (``_merge_floor``) runs identically
either way.

**A reranker call failure degrades the SAME way** — an exhausted rerank model group
(``ModelGroupUnavailableError``, already raised by ``ModelRouter.rerank`` after it emits
``DegradedModeEntered(component="reranker", reason=SURFACE_COMPONENT_DOWN)``, model_router.py:274-
286) is caught here and treated exactly like an empty gate: the pre-rerank pool passes through
unchanged (recall-service-design.md line 551: "Reranker unavailable -> recall reverts to
floor-protected merged"). The degrade event itself was already emitted by the router; this seam
does not need to re-emit it, only to not let the exception propagate and fail the whole recall.

**Rerank cannot widen the set (structural, unchanged by this file).** :meth:`Reranker.score`/
``RerankProviderPort.rerank`` receive only already-authorized content strings pulled from a pool
``RecallService``/``ThreeChannelRecallRanker`` already fused — this gate can narrow that pool
(prune candidates below the adaptive cutoff) or reorder it, but it has no way to introduce a new
id (ADR 0010/0013 "Isolation, verified"; recall-service-design.md line 231). The belt-and-
suspenders authz assert downstream still re-checks every surviving item regardless.

**A pruned member is not necessarily EVICTED.** This gate operates on the fused pool BEFORE the
STM-floor merge (``ranker.py``'s call site), so a protected floor member the gate scores below
``min_score`` and drops is not gone: ``_merge_floor``'s own pre-existing "protected id missing from
the fused list" rescue path (its own docstring: "a defence in depth against a future channel-list
change quietly breaking 'never evicted' — not a case that can fire today") already re-adds it at
the tail from ``floor_views``, the SAME mechanism that already covers a floor member fusion itself
pushed outside the window. This gate does not need its own floor-awareness; the invariant it would
otherwise risk breaking was already made robust to exactly this shape.

**Pool width vs. limit (recall-service-design.md line 202, ADR 0010's mem0-defect fix).** The
reference module's own docstring explains why the caller must feed the gate a pool WIDER than the
final result limit: mem0's call sites pass the SAME ``limit`` to both the vector search and the
reranker's ``top_k``, so the reranker only ever reorders an already-truncated list and can never
improve precision (there is nothing left to prune). ``settings.rerank_pool_size`` (default 20, the
ADR 0023 final combined value) is independent of ``limit`` and of ``channel_pool_size``/
``channel_pool_multiplier`` (the per-CHANNEL fetch width, ACCURACY-PLAN-0831.md §1.4) — it bounds
how many of the (already RRF-fused, best-first) candidates this gate sends to the cross-encoder in
ONE batched forward pass, independent of how wide the upstream channel fetch was. Capped
deliberately for latency, not correctness: ``language-analysis-server.md``'s own budget prices a
cross-encoder forward over <=20 pairs at ~15-40ms against a `recall_e2e_rerank` p95 of <=150ms —
sending the WHOLE fused union (which can be `3 x channel_pool` candidates) to the model on every
recall would blow that budget for no measured benefit past the top slice. Candidates beyond
``rerank_pool_size`` are left in their original fused-RRF order, unscored (``rerank_score`` stays
``None`` on them, exactly as it already does when the gate is dark) — they still compete for a
``limit`` slot downstream via ``_merge_floor``, they are simply not paid for by the rerank model
call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import structlog

from mu_engine.providers._contracts import ModelGroupUnavailableError, RerankProviderPort
from mu_engine.services.recall.dto import RecallItemView

__all__ = ["AdaptiveRerankGate", "RerankGate", "adaptive_rerank_gate"]

_log = structlog.get_logger("mu_engine.services.recall.rerank_gate")


def adaptive_rerank_gate(
    scored: Sequence[tuple[RecallItemView, float]], *, min_score: float, top_fraction: float
) -> list[tuple[RecallItemView, float]]:
    """PORT of ``shared/retrieval/rerank.py:340`` ``adaptive_rerank_gate``, unchanged algebra,
    generalised from ``MemoryNode`` to ``RecallItemView`` (this module's own docstring).

    Structurally the SAME two-rule shape ``dense.py``'s ``_adaptive_relevance_gate`` already uses
    for the MTM channel's own floor/top_fraction gate (recall-service-design.md §1.3 "(2) MTM
    vector") — reused deliberately, not invented (ADR 0009's pattern, applied here to rerank
    scores per ADR 0010): reused as its own function rather than only inline in the gate class
    below so it stays independently unit-testable, exactly as the reference keeps it a bare
    module-level function rather than a method.

    ``scored`` need not already be sorted — this function sorts it itself (the reference's own
    docstring: avoids a latent "caller forgot to sort" bug class its one caller never actually
    exercised).

    1. **Floor.** If even the best-scored candidate is below ``min_score``, no candidate in this
       pool carries real evidence for this query — returns ``[]`` (the caller's HippoRAG-style
       fallback: use the pre-rerank pool instead of an empty result, see
       :meth:`AdaptiveRerankGate.apply`).
    2. **Relative-to-top.** Among candidates that clear the floor, keep only those within
       ``top_fraction`` of this POOL's own top score. Because
       ``cutoff = max(min_score, top_score * top_fraction)`` can never exceed ``top_score`` itself,
       the top-scoring candidate always survives this gate whenever the floor clears — the gate can
       prune everything else, but never the single best match.
    """
    if not scored:
        return []
    ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
    top_score = ranked[0][1]
    if top_score < min_score:
        return []
    cutoff = max(min_score, top_score * top_fraction)
    return [pair for pair in ranked if pair[1] >= cutoff]


@runtime_checkable
class RerankGate(Protocol):
    """The rerank strategy seam (§1.5). Selection by ``settings.recall`` (a future
    ``rerank_gate_registry`` key, mirroring ``recall_registry``/``fusion_registry`` — this phase
    ships exactly one implementation, :class:`AdaptiveRerankGate`, wired directly rather than
    through a registry indirection nothing yet needs, DEV-STANDARDS "no abstraction the codebase
    does not use")."""

    key: str

    async def apply(self, pool: Sequence[RecallItemView], query: str) -> list[RecallItemView]: ...


class AdaptiveRerankGate:
    """key="adaptive_v1" — wraps :class:`~mu_engine.providers._contracts.RerankProviderPort` +
    :func:`adaptive_rerank_gate` (§1.5). See this module's docstring for the full design: dark
    when ``reranker`` is ``None``, capped at ``pool_size`` candidates per call, empty-gate and
    model-unavailable both fall back to the pre-rerank pool, never introduces a new id."""

    key = "adaptive_v1"

    def __init__(
        self,
        reranker: RerankProviderPort | None,
        *,
        min_score: float,
        top_fraction: float,
        pool_size: int,
    ) -> None:
        self._reranker = reranker
        self._min_score = min_score
        self._top_fraction = top_fraction
        self._pool_size = pool_size

    async def apply(self, pool: Sequence[RecallItemView], query: str) -> list[RecallItemView]:
        if self._reranker is None or not pool:
            # DARK (no reranker configured, e.g. `settings.rerank_enabled=False` at the
            # composition root) or nothing to score — byte-identical to no-rerank (ADR 0010
            # property 1). `list(...)` so every caller gets an ordinary list regardless of
            # whether `pool` was already one (mirrors `AdaptiveRerankGate`'s reference apply()).
            return list(pool)

        # Cap the model call at `pool_size` candidates (this module's own docstring, "Pool width
        # vs. limit"): only the top `pool_size` of the already-best-first RRF-fused pool is sent
        # to the cross-encoder; the rest passes through unscored, in its original fused order.
        head = list(pool[: self._pool_size])
        tail = list(pool[self._pool_size :])
        docs = [item.content for item in head]
        try:
            hits = await self._reranker.rerank(query, docs, top_n=None)
        except ModelGroupUnavailableError:
            # The router already emitted `DegradedModeEntered(component="reranker",
            # reason=SURFACE_COMPONENT_DOWN)` before raising this (model_router.py:274-286) — this
            # seam's job is only to not let that failure fail the whole recall (recall-service-
            # design.md line 551: "reverts to floor-protected merged").
            _log.warning("recall.rerank_unavailable", pool_size=len(head))
            return list(pool)

        # A well-behaved backend with `top_n=None` returns exactly one hit per document; a
        # defensive `.get(i, 0.0)` (never a KeyError) means a backend that returns fewer hits than
        # documents leaves the missing ones scored at the gate's own floor rather than crashing —
        # "rerank cannot widen the set" extends naturally to "a document this gate could not score
        # is never invented a passing score".
        scores = {hit.index: hit.score for hit in hits}
        scored = [(item, scores.get(i, 0.0)) for i, item in enumerate(head)]
        gated = adaptive_rerank_gate(
            scored, min_score=self._min_score, top_fraction=self._top_fraction
        )
        if not gated:
            # Empty gate -> HippoRAG fallback: the WHOLE pre-rerank pool (head + tail), not just
            # the head slice — see this module's docstring "Empty gate is a FALLBACK".
            return list(pool)

        reranked_head = [item.model_copy(update={"rerank_score": score}) for item, score in gated]
        return [*reranked_head, *tail]
