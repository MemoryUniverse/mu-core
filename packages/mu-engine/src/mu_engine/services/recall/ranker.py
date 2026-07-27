"""``RecallRanker`` — the per-partition strategy seam (recall-service-design.md §1.3).

``ThreeChannelRecallRanker`` runs the three channels over ONE η partition, fuses MTM+LTM with
rank-based RRF, and merges the STM recency FLOOR so fusion may reorder but NEVER evict a just-said
fact. It is the arm the federation runs TWICE — once per plane — differing only in the injected tier
repositories + the caller identity set (§1.6/§4.2a, CANONICAL §7.9 "one fuse implementation").

Channel behaviour pinned to §1.3:
  * **STM floor** — ``recent(ns, limit)``, unconditional, ``is_floor=True``, ``state='active'`` at
    the adapter (a superseded id is excluded from the window; the floor protects a *valid* recent
    fact, never resurrects a retired one).
  * **MTM dense** — ``semantic(ns, query_vec, ...)`` with the Model-A ``authorized_ids`` +
    ``state='active'`` predicate compiled server-side BEFORE top-k (adapter §3.2). Never pads.
  * **LTM graph (bi-temporal)** — ``graph_recall`` returns only ``m.state='active'`` facts whose
    ``[valid_at, invalid_at)`` interval contains *now* (adapter §4.3), so a conflict-resolved query
    returns the WINNING fact, not the stale one. DETERMINISTIC seed (recorded deviation, no LLM this
    phase — CODE-ADOPTION rule 4): with no entity-extraction pipeline wired yet, the graph arm seeds
    on the whole partition's currently-valid facts (``subject=None``, valid-now, recency-ordered)
    rather than LLM-resolved query entities; RRF lets the query-relevant MTM arm dominate while the
    graph arm contributes still-valid facts. Query-entity seeding folds in behind ``resolve_entity``
    when the extraction pipeline lands — a wiring change, not a shape change.

LTM-store-down is the ONE named in-arm degrade (``DegradeReason.LTM_UNAVAILABLE`` /
``recall_mtm_only``, degradation §_RULES): the graph arm drops, MTM+floor return, the result is
LABELLED. MTM/STM failures have NO degrade row — they re-raise loud (a deny, not a silent partial).
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from mu_contracts.domain.errors import StoreUnavailableError
from mu_contracts.domain.events import DegradeReason
from mu_contracts.domain.model.recall import CallerIdentitySet, Vector
from mu_contracts.ports.time import Clock
from mu_engine.services.recall.dto import (
    RecallChannels,
    RecallItemView,
    RecallResult,
    RecallSettings,
)
from mu_engine.services.recall.fusion import FusionStrategy
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import Scored
from mu_engine.storage.ports import LtmTierRepository, MtmTierRepository, StmTierRepository

__all__ = ["RecallRanker", "ThreeChannelRecallRanker"]


@runtime_checkable
class RecallRanker(Protocol):
    """The strategy seam (§1.3). Selection by ``settings.recall.strategy`` via the registry —
    a new ranker is a ``register()`` call, never an edit to ``RecallService``."""

    key: str

    async def rank(
        self,
        ns: Namespace,
        query: str,
        query_vec: Vector,
        *,
        limit: int,
        channels: RecallChannels,
        caller_identity_set: CallerIdentitySet | None,
    ) -> RecallResult: ...


def _to_view(scored: Scored[MemoryItem], channel: str) -> RecallItemView:
    item = scored.item
    return RecallItemView(
        memory_id=item.id,
        content=item.content,
        content_hash=item.content_hash,
        tier=item.tier,
        channel=channel,
        namespace=item.namespace,
        fused_score=scored.score,
        is_floor=scored.is_floor,
        artifact_ref=item.artifact_ref,
    )


class ThreeChannelRecallRanker:
    """key="rrf_3channel_v1" — the default 3-channel ranked read over one η partition (§1.3)."""

    key = "rrf_3channel_v1"

    def __init__(
        self,
        *,
        stm: StmTierRepository,
        mtm: MtmTierRepository,
        ltm: LtmTierRepository,
        fusion: FusionStrategy,
        settings: RecallSettings,
        clock: Clock,
    ) -> None:
        self._stm = stm
        self._mtm = mtm
        self._ltm = ltm
        self._fusion = fusion
        self._settings = settings
        self._clock = clock

    async def rank(
        self,
        ns: Namespace,
        query: str,
        query_vec: Vector,
        *,
        limit: int,
        channels: RecallChannels,
        caller_identity_set: CallerIdentitySet | None,
    ) -> RecallResult:
        del query  # deterministic graph seed this phase (no LLM entity resolution — see docstring)
        pool = self._settings.channel_pool_size
        floor_limit = self._settings.recency_floor_limit

        # Channels run concurrently under a STRUCTURED-CONCURRENCY TaskGroup (DEV-STANDARDS rule 1):
        # the LTM arm owns its own degrade (``_ltm_channel`` returns a named tuple, never raises),
        # so an LTM outage degrades WITHOUT failing the group. An STM/MTM store-down IS a hard deny:
        # the TaskGroup cancels the siblings (no orphaned in-flight I/O) and surfaces the error. We
        # unwrap the single store error from the ExceptionGroup so the caller sees the domain
        # exception (a deny), not the wrapper. ``CancelledError`` is never caught — it propagates.
        try:
            async with asyncio.TaskGroup() as tg:
                floor_t = tg.create_task(
                    self._stm.recent(ns, limit=floor_limit) if channels.stm else _empty_scored()
                )
                mtm_t = tg.create_task(
                    self._mtm.semantic(
                        ns, query_vec, limit=pool, caller_identity_set=caller_identity_set
                    )
                    if channels.mtm
                    else _empty_scored()
                )
                ltm_t = tg.create_task(
                    self._ltm_channel(ns, pool, caller_identity_set)
                    if channels.ltm
                    else _ltm_ok([])
                )
        except* StoreUnavailableError as eg:
            # STM/MTM store-down is a deny: surface the underlying domain error loud + unwrapped,
            # not the TaskGroup ExceptionGroup wrapper (§5 "re-raise loud, not a silent partial").
            raise eg.exceptions[0] from None

        floor = floor_t.result()
        mtm_hits = mtm_t.result()
        ltm_hits, ltm_degraded = ltm_t.result()

        # Fuse MTM ⊕ LTM by tier-stable id; a cosine score and a graph-hop count fuse by RANK only.
        fused_pairs = self._fusion.fuse(
            [mtm_hits, ltm_hits],
            id_of=lambda s: s.item.id,
            weights=[self._settings.weight_mtm, self._settings.weight_ltm],
            k=self._settings.rrf_k,
        )
        fused_views = [_to_view(scored, _channel_label(scored)) for scored, _score in fused_pairs]

        items = _merge_floor(
            floor_views=[_to_view(s, "stm") for s in floor], fused=fused_views, limit=limit
        )

        ran = RecallChannels(
            stm=channels.stm,
            mtm=channels.mtm,
            ltm=channels.ltm and not ltm_degraded,
        )
        return RecallResult(
            namespace=ns,
            items=items,
            channels_run=ran,
            degraded=DegradeReason.LTM_UNAVAILABLE if ltm_degraded else None,
            generated_at=self._clock.now(),
        )

    async def _ltm_channel(
        self, ns: Namespace, pool: int, caller: CallerIdentitySet | None
    ) -> tuple[list[Scored[MemoryItem]], bool]:
        """LTM graph arm with the ONE named in-arm degrade (LTM_UNAVAILABLE, §5). Returns
        ``(hits, degraded)``; a store-down drops the arm rather than failing the whole recall."""
        try:
            hits = await self._ltm.graph_recall(ns, limit=pool, caller_identity_set=caller)
        except StoreUnavailableError:
            return [], True
        return hits, False


async def _empty_scored() -> list[Scored[MemoryItem]]:
    return []


async def _ltm_ok(hits: list[Scored[MemoryItem]]) -> tuple[list[Scored[MemoryItem]], bool]:
    return hits, False


def _channel_label(scored: Scored[MemoryItem]) -> str:
    return "ltm" if scored.channel.value.startswith("ltm") else "mtm"


def _merge_floor(
    *, floor_views: list[RecallItemView], fused: list[RecallItemView], limit: int
) -> list[RecallItemView]:
    """Merge the STM floor in AFTER fusion (hybrid.py:247): fusion may reorder but never evict a
    floor member. Floor members lead (always recallable), then fused non-duplicates fill up to
    ``limit`` — the floor is protected even when it would push the list past ``limit``."""
    floor_ids = {v.memory_id for v in floor_views}
    tail = [v for v in fused if v.memory_id not in floor_ids]
    room = max(0, limit - len(floor_views))
    return [*floor_views, *tail[:room]]
