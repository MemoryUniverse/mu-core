"""``RecallRanker`` — the per-partition strategy seam (recall-service-design.md §1.3).

``ThreeChannelRecallRanker`` runs the three channels over ONE η partition, fuses MTM+LTM with
rank-based RRF, and merges the STM recency FLOOR so fusion may reorder but NEVER evict a just-said
fact. It is the arm the federation runs TWICE — once per plane — differing only in the injected tier
repositories + the caller identity set (§1.6/§4.2a, CANONICAL §7.9 "one fuse implementation").

Channel behaviour pinned to §1.3:
  * **STM floor** — ``recent(ns, limit)``, ``state='active'`` at the adapter (a superseded id is
    excluded from the window; the floor protects a *valid* recent fact, never resurrects a retired
    one). BUG FIX (data-quality assessment §3.1/#1, 2026-07-31): the candidate pool now FUSES into
    the same RRF pass as MTM/LTM (``weight_stm``) instead of being force-prepended whole — only the
    ``settings.floor_protect_limit`` most-recent candidates are UNCONDITIONALLY protected
    (``is_floor=True``, never evicted); the rest compete on fused rank like any other channel.
    Before this fix, ``recency_floor_limit`` defaulted to the SAME width as the result ``limit``
    (10==10), so a session with >= ``limit`` STM items consumed the ENTIRE result budget and the
    query-relevant MTM/LTM channels never surfaced a single item — every ``recall()`` in that
    session returned the identical, query-blind, insertion-order list (verbatim repro:
    ``docs/tracking/DATA-QUALITY-ASSESSMENT.md`` §3.1).
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

BUG FIX (D4 / conformance D-8, data-quality assessment §3.1/#5, 2026-07-31): the STM/MTM/LTM fuse
above merges by ``MemoryItem.id`` ONLY — the SAME fact recalled under two DIFFERENT ids (its STM
row + an already-promoted MTM/LTM copy sharing ``content_hash``) used to survive as two separate
``RecallItemView`` rows all the way into ``build_context`` ("Coffee-query context contained each
fact twice"). ``_merge_floor`` now runs :func:`~mu_engine.services.recall.fusion.
dedup_by_content_hash` over the floor-protected + fused candidate pool BEFORE the ``limit`` slice
(``settings.cross_tier_dedup``, default on) — the SAME primitive ``RecallService.recall`` already
runs one layer up at the private⊕shared federation seam, applied here to the per-arm candidate set
so a duplicate never occupies one of the ``limit`` result slots in the first place.

D1 STM relevance scoring (data-quality assessment §3.1, floor-fix follow-up to 02fbed9, 2026-07-31):
the ``recency_floor_limit``/``floor_protect_limit`` bound above fixed HOW MANY STM candidates enter
the fuse and HOW MANY are unconditionally protected — but every STM candidate still entered RRF
ordered by RECENCY RANK ONLY (``StmTierRepository.recent`` newest-first), so a targeted query and
a nonsense query in the same session still returned near-identical, STM-dominated lists.
``_score_stm`` now attaches a REAL per-candidate relevance score BEFORE fusion (``stm_scoring``,
"embed": cosine-rank against the SAME query vector the MTM channel already uses; "lexical":
token-overlap fallback needing no embedder; "recency": explicit pre-fix opt-out). The relevance-
ordered list feeds BOTH the RRF fusion channel input (fused rank now reflects relevance, not just
recency) AND the protected-floor DISPLAY order — protection membership (WHICH items can never be
evicted) stays recency-selected (the "never evict a just-said fact" guarantee is unchanged), but the
protected block is now reorderable BY RELEVANCE within itself so a just-said, irrelevant fact no
longer sits at rank 1 ahead of the actual answer.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mu_contracts.domain.errors import StoreUnavailableError
from mu_contracts.domain.events import DegradeReason
from mu_contracts.domain.model.recall import CallerIdentitySet, Vector
from mu_contracts.ports.time import Clock
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.services.recall.dto import (
    RecallChannels,
    RecallItemView,
    RecallResult,
    RecallSettings,
)
from mu_engine.services.recall.fusion import FusionStrategy, dedup_by_content_hash
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import RecallChannel, Scored
from mu_engine.storage.ports import LtmTierRepository, MtmTierRepository, StmTierRepository

__all__ = ["RecallRanker", "StmScoringConfigError", "ThreeChannelRecallRanker"]


class StmScoringConfigError(ValueError):
    """``settings.stm_scoring="embed"`` with no ``EmbeddingPort`` injected into the ranker (§D1).

    Fail-loud misconfiguration — mirrors ``EmbedderConfigError`` (providers/embedding.py): an
    unusable relevance mode never silently degrades to recency-only ordering.
    """


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
        embedder: EmbeddingPort | None = None,
    ) -> None:
        self._stm = stm
        self._mtm = mtm
        self._ltm = ltm
        self._fusion = fusion
        self._settings = settings
        self._clock = clock
        # D1: required only when ``settings.stm_scoring == "embed"`` (the default) — the SAME
        # ``EmbeddingPort`` the composition root already wires into ``RecallService`` (the query is
        # embedded once at that façade boundary, §6-P2/m4); the ranker reuses it to embed STM
        # candidate CONTENT so a cosine score can be computed against the already-embedded query.
        self._embedder = embedder

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
        # `query` (raw text) is used ONLY by the D1 STM relevance scorer below ("lexical" mode) —
        # the LTM graph arm remains the deterministic recency seed this phase (no LLM entity
        # resolution — see docstring); it never reads `query`.
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
                    self._ltm_channel(ns, pool, caller_identity_set, query)
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

        # D1 (§3.1 follow-up to 02fbed9): attach a REAL relevance score to every STM candidate
        # BEFORE fusion — `floor` arrives recency-ordered only (the adapter's ZREVRANGE order);
        # `floor_scored` is the SAME set of candidates re-ordered by `settings.stm_scoring`
        # relevance (embed cosine / lexical overlap / recency no-op). This is what makes the STM
        # channel's RRF rank (below) reflect the QUERY, not just insertion order.
        floor_scored = await self._score_stm(floor, query, query_vec)

        # Fuse STM ⊕ MTM ⊕ LTM by tier-stable id; a recency rank, a cosine score, and a graph-hop
        # count fuse by RANK only (§1.3 "one fuse implementation"). BUG FIX (§3.1/#1): the STM
        # candidate pool used to be excluded from this fusion and force-prepended WHOLE ahead of it
        # (see module docstring) — that made the query-relevant MTM/LTM channels invisible whenever
        # the session held >= `limit` STM items. It now competes on fused rank like any other
        # channel (by RELEVANCE rank since D1, not recency rank); only a small, bounded prefix is
        # still unconditionally protected below.
        settings = self._settings
        fused_pairs = self._fusion.fuse(
            [floor_scored, mtm_hits, ltm_hits],
            id_of=lambda s: s.item.id,
            weights=[settings.weight_stm, settings.weight_mtm, settings.weight_ltm],
            k=settings.rrf_k,
        )
        # `is_floor` from the adapter marks EVERY STM candidate (Scored.is_floor=True on the whole
        # recency pool, redis_stm.py `recent()`) — force it False here so ONLY the explicitly
        # protected prefix below is exempt from eviction; a fused-in STM item that didn't make the
        # protected prefix must compete for its slot exactly like an MTM/LTM hit.
        fused_views = [
            _to_view(scored, _channel_label(scored)).model_copy(update={"is_floor": False})
            for scored, _score in fused_pairs
        ]

        # D1 (b): WHICH items are unconditionally protected stays RECENCY-selected — `floor` (not
        # `floor_scored`) picks the `floor_protect_limit` most-recent candidates, preserving the
        # "never evict a just-said fact" guarantee unchanged. But the block is now REORDERABLE
        # within itself BY RELEVANCE: `floor_scored` is already sorted by relevance desc over the
        # WHOLE STM pool, so filtering it down to the protected ids yields the protected members in
        # relevance order — a just-said, irrelevant fact no longer sits at rank 1 ahead of the
        # actual answer merely because it was said last.
        protect_n = self._settings.floor_protect_limit
        protected_ids = {s.item.id for s in floor[:protect_n]}
        protected_floor_views = [
            _to_view(s, "stm") for s in floor_scored if s.item.id in protected_ids
        ]

        items = _merge_floor(
            floor_views=protected_floor_views,
            fused=fused_views,
            limit=limit,
            cross_tier_dedup=self._settings.cross_tier_dedup,
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
        self, ns: Namespace, pool: int, caller: CallerIdentitySet | None, query: str
    ) -> tuple[list[Scored[MemoryItem]], bool]:
        """LTM graph arm with the ONE named in-arm degrade (LTM_UNAVAILABLE, §5). Returns
        ``(hits, degraded)``; a store-down drops the arm rather than failing the whole recall.

        D-4 (ARCHITECTURE-CONFORMANCE.md "LTM graph arm thin"): ADDS a multi-hop entity-edge
        traversal (:meth:`~mu_engine.storage.ports.GraphStorePort.traverse_entities`) alongside
        the pre-existing flat ``graph_recall`` seed — never REPLACES it (module docstring "LTM
        graph (bi-temporal)" section is otherwise unchanged: the flat seed still runs, still
        contributes still-valid whole-partition facts to RRF). Traversal hits are merged in by
        id, deduped against the flat seed's own hits (a fact both arms surface is not
        double-counted) — the merged list is what the caller fuses into RRF below, so a
        traversal-only hit ("who is Bo's manager?" -> "Ada manages Bo", never surfaced by the
        flat seed if Ada/Bo aren't already in the recency-ordered whole-partition window)
        competes on equal footing with every other LTM hit. ``settings.ltm_max_hops == 0``
        disables the traversal call entirely (flat-only, pre-D6 behavior)."""
        try:
            hits = await self._ltm.graph_recall(ns, limit=pool, caller_identity_set=caller)
        except StoreUnavailableError:
            return [], True
        if self._settings.ltm_max_hops <= 0:
            return hits, False
        try:
            traversal_hits = await self._ltm.traverse_entities(
                ns, query=query, max_hops=self._settings.ltm_max_hops, limit=pool
            )
        except StoreUnavailableError:
            # the flat seed already succeeded above — a traversal-only outage degrades to
            # flat-only LTM results rather than dropping the WHOLE arm (narrower than the
            # named LTM_UNAVAILABLE degrade, which is reserved for the flat seed itself).
            return hits, False
        seen = {s.item.id for s in hits}
        extra = [s for s in traversal_hits if s.item.id not in seen]
        # BUG2 FIX (data-quality re-assessment §3, "fix the ranker LTM arm so traversal hits
        # outrank unrelated same-subject attributes"): RRF fuses channels by RANK (position in
        # this returned list), never by raw `.score` — so unconditionally appending EVERY
        # traversal-only hit AFTER the entire flat seed (the pre-fix code) gave a traversal hit
        # the worst possible rank in this channel no matter how relevant it was, directly
        # contradicting this method's own docstring ("competes on equal footing with every other
        # LTM hit"). Re-sorting the combined list by `.score` — the flat seed's recency-rank score
        # and the traversal arm's hop+predicate-relevance score (`FalkorLtmAdapter.
        # _traverse_entities_impl`) are both already comparable relevance signals — lets a
        # genuinely on-topic traversal hit (e.g. "Ada manages Bo" for "who is Bo's manager?") earn
        # the rank its relevance deserves instead of being structurally buried.
        merged = sorted([*hits, *extra], key=lambda s: -s.score)
        return merged, False

    async def _score_stm(
        self, floor: list[Scored[MemoryItem]], query: str, query_vec: Vector
    ) -> list[Scored[MemoryItem]]:
        """D1 (§3.1): attach a REAL relevance score to every STM candidate, sorted best-first.

        ``floor`` arrives recency-ordered (the adapter's ``recent()``), every item carrying a
        constant channel-native ``.score`` (§1.3 "STM floor" — no relevance signal). This method
        replaces that constant with an actual query-relevance score per ``settings.stm_scoring``
        and returns the SAME candidates re-sorted by it (``.is_floor``/``.channel`` unchanged, only
        ``.score`` + list order differ) — the result both feeds the RRF channel input and the
        protected-floor display order in :meth:`rank`.

          * "recency" — no-op: returns ``floor`` unchanged (explicit pre-fix opt-out, §D1).
          * "lexical" — token-overlap score against the raw ``query`` text; needs no embedder.
          * "embed" (default) — cosine similarity between ``query_vec`` (already embedded once at
            the ``RecallService`` façade, §6-P2/m4) and a fresh embedding of each candidate's
            content, via the SAME ``EmbeddingPort`` the MTM channel is embedded with. Requires an
            embedder injected at construction; ``StmScoringConfigError`` otherwise — a fail-loud
            misconfiguration, never a silent recency fallback.
        """
        if not floor:
            return floor

        mode = self._settings.stm_scoring
        if mode == "recency":
            return floor

        scored: list[tuple[Scored[MemoryItem], float]]
        if mode == "lexical":
            scored = [(s, _lexical_overlap(query, s.item.content)) for s in floor]
        elif mode == "embed":
            if self._embedder is None:
                raise StmScoringConfigError(
                    "RecallSettings.stm_scoring='embed' (the default) requires an EmbeddingPort "
                    "injected into ThreeChannelRecallRanker(embedder=...) at the composition root "
                    "— set stm_scoring='lexical' (no embedder needed) or wire one; this never "
                    "silently falls back to recency-only ordering (DEV-STANDARDS: no silent stubs)."
                )
            vectors = await self._embedder.embed([s.item.content for s in floor])
            scored = [(s, _cosine(query_vec, v)) for s, v in zip(floor, vectors, strict=True)]
        else:  # pragma: no cover - Literal["embed", "lexical", "recency"] makes this unreachable
            raise StmScoringConfigError(f"unknown RecallSettings.stm_scoring {mode!r}")

        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [s.model_copy(update={"score": relevance}) for s, relevance in ranked]


def _lexical_overlap(query: str, content: str) -> float:
    """D1 "lexical" STM relevance score: ``|query_tokens ∩ content_tokens| / |query_tokens|``,
    case-insensitive whitespace tokenization. The minimum-viable fallback (no embedder needed) —
    cheap, deterministic, and enough to make a targeted query diverge from a nonsense one."""
    q_tokens = {t for t in query.lower().split() if t}
    if not q_tokens:
        return 0.0
    c_tokens = {t for t in content.lower().split() if t}
    if not c_tokens:
        return 0.0
    return len(q_tokens & c_tokens) / len(q_tokens)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """D1 "embed" STM relevance score: cosine similarity, 0.0 for either zero-norm vector (never
    divides by zero) — the SAME comparison the MTM channel's dense search is built on."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _empty_scored() -> list[Scored[MemoryItem]]:
    return []


async def _ltm_ok(hits: list[Scored[MemoryItem]]) -> tuple[list[Scored[MemoryItem]], bool]:
    return hits, False


def _channel_label(scored: Scored[MemoryItem]) -> str:
    if scored.channel.value.startswith("ltm"):
        return "ltm"
    if scored.channel is RecallChannel.STM_FLOOR:
        return "stm"
    return "mtm"


def _merge_floor(
    *,
    floor_views: list[RecallItemView],
    fused: list[RecallItemView],
    limit: int,
    cross_tier_dedup: bool,
) -> list[RecallItemView]:
    """Merge the (now BOUNDED, §3.1/#1 bug fix) STM floor in AFTER fusion (hybrid.py:247): fusion
    may reorder but never evict a protected floor member. ``floor_views`` is capped upstream to
    ``settings.floor_protect_limit`` — NOT the whole STM candidate pool — so it always leaves room
    for the fused, query-relevant tail. Floor members lead (always recallable), then fused
    non-duplicates fill up to ``limit``.

    D4 cross-tier dedup (conformance D-8, ``settings.cross_tier_dedup``): the STM/MTM/LTM fuse
    above merges by ``MemoryItem.id`` only, so the SAME fact surfaced from two different tiers
    under two different ids (e.g. its STM row and an already-promoted MTM/LTM copy sharing
    ``content_hash``) would otherwise occupy two of the ``limit`` slots below. Deduping by
    ``content_hash`` HERE — floor members first, so a protected floor row always wins its
    duplicate — before the ``[:limit]`` slice means a dup never crowds out a genuinely distinct
    fact (matches the read-time half of the ``ada_coffee`` double-write finding, DATA-QUALITY-
    ASSESSMENT.md §3.1/#5: "Coffee-query context contained each fact twice"). With the toggle off
    this reduces to the identical ``[*floor_views, *tail][:limit]`` slice the pre-fix code took
    (``floor_views`` is always <= ``limit`` in practice, so slicing the concatenation is equivalent
    to the old two-part ``room``-bounded merge)."""
    floor_ids = {v.memory_id for v in floor_views}
    tail = [v for v in fused if v.memory_id not in floor_ids]
    candidates = [*floor_views, *tail]
    if cross_tier_dedup:
        candidates = dedup_by_content_hash(candidates)
    return candidates[:limit]
