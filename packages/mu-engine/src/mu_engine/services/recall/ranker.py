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
    ``settings.floor_protect_limit`` most-recent candidates are ELIGIBLE for protection
    (``is_floor=True``, never evicted); the rest compete on fused rank like any other channel. T1
    option (c) (``TRACE-0923.md`` §7/§7.1, ``_protected_floor_ids`` below) added a SECOND gate on
    top of that eligibility window: an eligible candidate is only actually protected once its own
    relevance score clears ``settings.floor_protect_min_relevance`` — SHIPPED at ``0.5`` on
    evidence (measured `gold_in_context` sweep, ``dto.py``'s own docstring has the numbers);
    ``-1.0`` (cosine similarity's own true minimum) reproduces the pre-T1c "unconditional within
    the window" behaviour exactly, still reachable via
    ``MU_RECALL__FLOOR_PROTECT_MIN_RELEVANCE=-1``. Before the
    original fix, ``recency_floor_limit`` defaulted to the SAME width as the result ``limit``
    (10==10), so a session with >= ``limit`` STM items consumed the ENTIRE result budget and the
    query-relevant MTM/LTM channels never surfaced a single item — every ``recall()`` in that
    session returned the identical, query-blind, insertion-order list (verbatim repro:
    ``docs/tracking/DATA-QUALITY-ASSESSMENT.md`` §3.1).
  * **STM demoted (AD-250 fix, ADR 0061, NEW)** — ``demoted(ns, limit)``, a SEPARATE channel from
    the STM floor above, over its OWN discoverability index (``StmTierRepository.put_demoted``'s
    own docstring has the full mechanism). A demoted MTM->STM write-ahead copy used to land in
    the SAME bounded recency floor a fresh capture uses, scored by ``item.created_at`` — its
    ORIGINAL age, not the demotion instant — so it re-entered the floor underneath every
    genuinely new turn and dropped out of the window almost immediately, with its MTM point
    already gone: nothing could ever re-recall it to raise ``access_count``, so the ADR 0034
    rescue had a reachable GATE (``promote_stm_mtm``) and no reachable TRIGGER (FAULT-HUNT-0924
    F1 / AD-250, ADR 0058's verify pass). This channel fuses in at the SAME ``weight_stm``
    discount as the ordinary floor, is NEVER ``is_floor``-eligible (a demoted item is not "just
    said" — it competes on relevance alone), and a genuine hit is what
    :meth:`ThreeChannelRecallRanker._reinforce_stm_hits` (below) uses to actually raise
    ``access_count`` toward the rescue gate.
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
recency) AND the protected-floor DISPLAY order — at the time of D1, protection membership (WHICH
items can never be evicted) stayed purely recency-selected and the "never evict a just-said fact"
guarantee was unchanged, with the protected block merely reorderable BY RELEVANCE within itself so
a just-said, irrelevant fact no longer sat at rank 1 ahead of the actual answer. **T1 option (c)
(2026-09-23, ADR 0052) SUPERSEDED the membership half of that sentence**: recency still bounds who
is ELIGIBLE (``floor_protect_limit``), but an eligible candidate is now protected only once its own
relevance score clears ``floor_protect_min_relevance`` — see ``_protected_floor_ids``. The D1
statement above is kept as the record of what D1 itself decided; it is no longer the live
behaviour.

AD-204 channel rank-authority (RETRIEVAL-EVAL-0829.md §5.3 / STATE-AND-DEFECTS-0829.md D3,
2026-08-30, the fix D3 itself named as still open): D1/D3 above fixed WHAT the STM channel's
candidates are ordered/protected by, but every channel still entered the ``fusion.fuse`` call
below at EQUAL weight — and RRF only ever looks at a channel's OWN rank position, never the
breadth of the pool that rank came from. A ten-item, single-session STM recency window and an MTM
ANN search over the WHOLE partition therefore gave their respective rank-0 candidates the exact
same ``1/(k+1)`` vote, even after D1 made the STM candidate genuinely query-scored. MEASURED: with
equal weights the shipped 3-channel fuse was worse than its own MTM channel alone at EVERY cutoff
(recall@1 0.0049 vs 0.1625, a 33x gap). Fix: ``RecallSettings.weight_stm`` default lowered 1.0 ->
0.1, a 10:1 discount against MTM/LTM (see its own docstring in ``dto.py`` for the full measurement,
including why a smaller discount that looked sufficient on a small sample was NOT enough on the
full corpus) — no new mechanism, the weighting knob this module already threaded into
``self._fusion.fuse(...)`` below was simply left at a value nothing had measured. Re-measured:
recall@1/3/5 now land within one query's width of the MTM-alone channel; recall@10 sits ~10% below
it at the SHIPPED ``floor_protect_limit=3``, and that residual is the SEPARATE, already-decided
``floor_protect_limit`` presence guarantee (AD-195) spending result slots on rows fusion ranked
outside the window — confirmed by re-measuring with ``floor_protect_limit=0`` (diagnostic-only),
which matches or exceeds the MTM-alone channel at every cutoff. Not a re-opening of AD-195's
trade-off, only of the rank-authority mismatch this fix's own diagnosis names.

Rerank gate (ACCURACY-PLAN-0831.md item 6, 2026-09-01): the three-channel RRF fuse above now
passes through an :class:`~mu_engine.services.recall.rerank_gate.AdaptiveRerankGate` before the
STM-floor merge (`rank()`'s own inline comment marks the exact insertion point) — the seam
`ModelRouter.rerank` had been built for but never had a caller. See `rerank_gate.py`'s module
docstring for the full design (dark-by-default semantics, empty-gate/model-unavailable fallback,
why the floor's "never evicted" guarantee needed no change here).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import structlog

from mu_contracts.domain.errors import CallerIdentitySetRequiredError, StoreUnavailableError
from mu_contracts.domain.events import DegradeReason
from mu_contracts.domain.model.recall import CallerIdentitySet, Vector
from mu_contracts.ports.time import Clock
from mu_engine.providers._contracts import EmbeddingPort, RerankProviderPort
from mu_engine.services.recall.dto import (
    RecallChannels,
    RecallItemView,
    RecallResult,
    RecallSettings,
)
from mu_engine.services.recall.fusion import FusionStrategy, dedup_by_content_hash
from mu_engine.services.recall.rerank_gate import AdaptiveRerankGate
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import Scored, SparseQuery
from mu_engine.storage.ports import LtmTierRepository, MtmTierRepository, StmTierRepository

__all__ = ["RecallRanker", "StmScoringConfigError", "ThreeChannelRecallRanker"]

_log = structlog.get_logger("mu_engine.services.recall.ranker")


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
        sparse_query: SparseQuery | None = None,
    ) -> RecallResult: ...


def _to_view(
    scored: Scored[MemoryItem], channel: str, *, fused_score: float | None = None
) -> RecallItemView:
    """``fused_score`` defaults to the channel-native ``scored.score`` — the right value for a
    candidate that never went THROUGH fusion (the protected-floor block below, which is
    recency/relevance-selected, not RRF-ranked). A caller handing back an actual
    :func:`~mu_engine.services.recall.fusion.reciprocal_rank_fusion` result MUST pass that value
    explicitly (D2, STATE-AND-DEFECTS-0829.md) — see the ``fused_views`` comprehension below,
    which used to silently drop it via ``for scored, _score in fused_pairs``."""
    item = scored.item
    return RecallItemView(
        memory_id=item.id,
        content=item.content,
        content_hash=item.content_hash,
        tier=item.tier,
        channel=channel,
        namespace=item.namespace,
        fused_score=scored.score if fused_score is None else fused_score,
        is_floor=scored.is_floor,
        artifact_ref=item.artifact_ref,
        turn_seq=item.turn_seq,  # S1b — None on any item written before it existed (dto.py)
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
        reranker: RerankProviderPort | None = None,
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
        # ACCURACY-PLAN-0831.md item 6 (`rerank_gate.py`'s own module docstring has the full
        # design): `reranker=None` (composition root wired nothing) or `settings.rerank_enabled=
        # False` both leave the gate DARK — byte-identical to the pre-rerank behaviour every
        # existing ranker test already exercises.
        self._rerank_gate = AdaptiveRerankGate(
            reranker if settings.rerank_enabled else None,
            min_score=settings.rerank_min_score,
            top_fraction=settings.rerank_top_fraction,
            pool_size=settings.rerank_pool_size,
        )

    async def rank(
        self,
        ns: Namespace,
        query: str,
        query_vec: Vector,
        *,
        limit: int,
        channels: RecallChannels,
        caller_identity_set: CallerIdentitySet | None,
        sparse_query: SparseQuery | None = None,
    ) -> RecallResult:
        # `sparse_query` is the façade-encoded BM25 term weights (mtm-retrieval-design.md §1.3
        # "the M2 resolution"): the RecallService holds the SparseEncoderPort exactly as it holds
        # the EmbeddingPort, encodes the query ONCE at that boundary, and threads the resulting
        # value object down beside `query_vec`. The tier repo still never receives raw text
        # (CANONICAL §6-P2/m4). `None` -> the MTM arm is dense-only, byte-identical to before.
        # `query` (raw text) is used ONLY by the D1 STM relevance scorer below ("lexical" mode) —
        # the LTM graph arm remains the deterministic recency seed this phase (no LLM entity
        # resolution — see docstring); it never reads `query`.
        # FAIL-CLOSED GATE (AD-128; CANONICAL §7.4). On a SHARED η the `to_prefix()` partition
        # separates ORGS, not the MEMBERS of one org — the session slot is caller-supplied — so
        # Model-A is the WHOLE gate and `None` cannot be a legal caller set here. Refuse it HERE,
        # at the one place that knows both the η and the caller set, so a caller that forgot to
        # thread the set never even reaches a tier adapter. This is now the FIRST of two
        # instruments, not the only one (AD-179): every tier adapter also refuses
        # `caller_identity_set=None` on SHARED itself (`qdrant_mtm.py`, `falkor_ltm.py`,
        # `weaviate_mtm.py`, `pgvector_mtm.py`, `chroma_mtm.py` — via
        # `mu_engine.storage.authz.require_shared_caller_identity_set`; the STM tier's
        # `authorized_window`/`authorized_item` always did). Before AD-179 those five adapters
        # instead read `caller_identity_set is None` as "omit the Model-A clause entirely" and
        # this gate was the ONLY thing standing between a forgotten caller set and an UNFILTERED
        # shared read — kept here regardless, because a service-layer refusal that never reaches
        # the store is strictly better than one that does (the convention-at-the-call-sites shape
        # that produced C2, C3 and 7ccc405 is exactly "relying on one layer to remember").
        # An EMPTY set is legal and authorizes NOTHING: `RecallService` coerces a missing shared
        # caller set to `frozenset()` deliberately ("the safe direction, never an over-broad
        # match"), and every arm's predicate denies on it.
        if ns.visibility is Visibility.SHARED and caller_identity_set is None:
            raise CallerIdentitySetRequiredError(
                "recall: a SHARED-η rank requires the Model-A caller identity set "
                "(CANONICAL §7.4) — pass frozenset() to authorize nothing, never None"
            )

        pool = self._effective_pool(limit)
        floor_limit = self._settings.recency_floor_limit
        demoted_limit = self._settings.demoted_floor_limit

        # Channels run concurrently under a STRUCTURED-CONCURRENCY TaskGroup (DEV-STANDARDS rule 1):
        # the LTM arm owns its own degrade (``_ltm_channel`` returns a named tuple, never raises),
        # so an LTM outage degrades WITHOUT failing the group. An STM/MTM store-down IS a hard deny:
        # the TaskGroup cancels the siblings (no orphaned in-flight I/O) and surfaces the error. We
        # unwrap the single store error from the ExceptionGroup so the caller sees the domain
        # exception (a deny), not the wrapper. ``CancelledError`` is never caught — it propagates.
        try:
            async with asyncio.TaskGroup() as tg:
                floor_t = tg.create_task(
                    # AD-128: the STM floor arm receives the caller identity set exactly like the
                    # MTM and LTM arms below. Until the port could express it, this ONE arm ran
                    # unauthorized on the SHARED plane and served a non-member the room's items.
                    self._stm.recent(ns, limit=floor_limit, caller_identity_set=caller_identity_set)
                    if channels.stm
                    else _empty_scored()
                )
                demoted_t = tg.create_task(
                    # AD-250 fix (ADR 0061): the demoted-item channel — a demoted write-ahead
                    # copy lives in its OWN index (`put_demoted`'s docstring), never the ordinary
                    # recency floor above, so it needs its own fetch. Runs whenever the STM floor
                    # does (it is conceptually part of "the STM tier", just a different index) —
                    # `dto.py`'s `demoted_floor_limit` docstring has the full rationale.
                    self._stm.demoted(
                        ns, limit=demoted_limit, caller_identity_set=caller_identity_set
                    )
                    if channels.stm
                    else _empty_scored()
                )
                mtm_t = tg.create_task(
                    self._mtm.semantic(
                        ns,
                        query_vec,
                        limit=pool,
                        caller_identity_set=caller_identity_set,
                        sparse_query=sparse_query,
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
        demoted_hits = demoted_t.result()
        mtm_hits = mtm_t.result()
        ltm_hits, ltm_degraded = ltm_t.result()

        # D1 (§3.1 follow-up to 02fbed9): attach a REAL relevance score to every STM candidate
        # BEFORE fusion — `floor` arrives recency-ordered only (the adapter's ZREVRANGE order);
        # `floor_scored` is the SAME set of candidates re-ordered by `settings.stm_scoring`
        # relevance (embed cosine / lexical overlap / recency no-op). This is what makes the STM
        # channel's RRF rank (below) reflect the QUERY, not just insertion order.
        floor_scored = await self._score_stm(floor, query, query_vec)
        # AD-250 fix (ADR 0061): the SAME relevance scorer, applied to the demoted-channel pool —
        # `_score_stm` scores by CONTENT relevance, which is generic to any STM-sourced candidate
        # list, not specific to the recency floor despite the method's name.
        demoted_scored = await self._score_stm(demoted_hits, query, query_vec)

        # Fuse STM ⊕ MTM ⊕ LTM by tier-stable id; a recency rank, a cosine score, and a graph-hop
        # count fuse by RANK only (§1.3 "one fuse implementation"). BUG FIX (§3.1/#1): the STM
        # candidate pool used to be excluded from this fusion and force-prepended WHOLE ahead of it
        # (see module docstring) — that made the query-relevant MTM/LTM channels invisible whenever
        # the session held >= `limit` STM items. It now competes on fused rank like any other
        # channel (by RELEVANCE rank since D1, not recency rank); only a small, bounded prefix is
        # still unconditionally protected below.
        settings = self._settings
        # AD-250 fix (ADR 0061): `demoted_scored` joins the fuse as a FOURTH channel, at the SAME
        # `weight_stm` discount an ordinary floor candidate gets — it is still "the STM tier",
        # just a different index (`demoted_floor_limit`'s own docstring). Its list AND its
        # weight are OMITTED from the fuse entirely when it is empty (the common case — most
        # namespaces have zero currently-demoted items), rather than passed as an always-present
        # fourth weight slot: `reciprocal_rank_fusion` normalizes by `sum(weights)`, so an EMPTY
        # channel that still occupies a weight slot silently shrinks every OTHER channel's
        # effective share (measured: 1.0/3 -> 1.0/4 at equal weights, a ~25% cut to MTM's
        # contribution on EVERY query, for a channel contributing zero candidates to any of
        # them) — a real, unmeasured ranking-quality regression this fix has no business making
        # for the overwhelming majority of recalls that touch no demoted item at all.
        channel_results: list[Sequence[Scored[MemoryItem]]] = [floor_scored, mtm_hits, ltm_hits]
        channel_weights = [settings.weight_stm, settings.weight_mtm, settings.weight_ltm]
        if demoted_scored:
            channel_results.append(demoted_scored)
            channel_weights.append(settings.weight_stm)
        fused_pairs = self._fusion.fuse(
            channel_results,
            id_of=lambda s: s.item.id,
            weights=channel_weights,
            k=settings.rrf_k,
        )
        # `is_floor` from the adapter marks EVERY STM candidate (Scored.is_floor=True on the whole
        # recency pool, redis_stm.py `recent()`) — force it False here so ONLY the explicitly
        # protected prefix below is exempt from eviction; a fused-in STM item that didn't make the
        # protected prefix must compete for its slot exactly like an MTM/LTM hit.
        #
        # D2 (STATE-AND-DEFECTS-0829.md): `rrf_score` — the value `reciprocal_rank_fusion` actually
        # computed — used to be discarded here (`for scored, _score in fused_pairs`), so
        # `fused_score` carried whatever channel-native score `scored.score` happened to be (a raw
        # MTM cosine, an LTM hop count, ...) instead of the fused rank the field's NAME promises. A
        # negative `fused_score` was observed in the wild, which an RRF score (a sum of positive
        # `1/(k+rank+1)` terms) can never be — proof the field held the wrong number. Pass the real
        # value through explicitly now.
        fused_views = [
            _to_view(scored, _channel_label(scored), fused_score=rrf_score).model_copy(
                update={"is_floor": False}
            )
            for scored, rrf_score in fused_pairs
        ]

        # ACCURACY-PLAN-0831.md item 6 (`rerank_gate.py` module docstring — the full design):
        # the rerank gate runs over the RRF-fused pool, BEFORE the STM-floor merge below —
        # matching recall-service-design.md's own pipeline order ("FusionStrategy.fuse -> ...
        # -> RerankGate.apply", §1 diagram) while relying on `_merge_floor`'s ALREADY-EXISTING
        # "protected id missing from `fused`" rescue path to keep the floor's "never evicted"
        # guarantee intact even for a protected member the gate scores below `min_score` and
        # prunes — see `rerank_gate.py`'s own docstring ("A pruned member is not necessarily
        # EVICTED") for why this needed no change to `_merge_floor` itself. Dark
        # (`settings.rerank_enabled=False` or no reranker injected) returns `fused_views`
        # unchanged, so every pre-existing test of this method is unaffected.
        fused_views = await self._rerank_gate.apply(fused_views, query)

        # S1b — read-time neighbour expansion (TRACE-0923.md §7/§6.2/§5.1; `_expand_neighbors`'s
        # own docstring has the full rationale + graceful-degradation contract). Runs AFTER the
        # rerank gate deliberately: an inserted neighbour is a NEW candidate no model has scored,
        # and it competes for a `limit` slot on its anchor's already-decided position, not on a
        # rerank verdict nobody computed for it. `settings.neighbor_expand_radius=0` (the default)
        # returns `fused_views` unchanged — inert until an operator raises it.
        fused_views = await self._expand_neighbors(
            ns,
            fused_views,
            floor_pool_size=len(floor_scored),
            caller_identity_set=caller_identity_set,
        )

        # D1 (b): WHICH items are unconditionally protected stays RECENCY-selected — `floor` (not
        # `floor_scored`) picks the `floor_protect_limit` most-recent candidates, preserving the
        # "never evict a just-said fact" guarantee unchanged. But the block is now REORDERABLE
        # within itself BY RELEVANCE: `floor_scored` is already sorted by relevance desc over the
        # WHOLE STM pool, so filtering it down to the protected ids yields the protected members in
        # relevance order — a just-said, irrelevant fact no longer sits at rank 1 ahead of the
        # actual answer merely because it was said last.
        protect_n = self._settings.floor_protect_limit
        protected_ids = _protected_floor_ids(
            floor=floor,
            floor_scored=floor_scored,
            protect_n=protect_n,
            min_relevance=self._settings.floor_protect_min_relevance,
        )
        protected_floor_views = [
            _to_view(s, "stm") for s in floor_scored if s.item.id in protected_ids
        ]

        # ADR 0060 (docs/tracking/eval-runs/2026-09-24-ltm-channel-zero-slots.md): "why does the
        # LTM channel win zero slots even when it is populated" — a STRUCTURAL exclusion, not a
        # discount. `weight_ltm=0.1`'s best-possible RRF contribution (rank 0: `0.1/(k+1)`) is
        # PROVABLY smaller than `weight_mtm=1.0`'s worst-pool-item contribution (rank pool-1:
        # `1.0/(k+pool)`) for every shipped `channel_pool_size`/`channel_pool_multiplier`/`rrf_k`
        # combination — so whenever the MTM channel returns >= `limit` candidates (the normal
        # case once a namespace has more than a handful of MTM facts), NO LTM candidate can ever
        # place in the fused top-`limit`, regardless of how many facts the graph tier holds or
        # how genuinely valid they are. Simply raising `weight_ltm` to cross that threshold
        # (measured ~0.68-0.87 at shipped settings) reopens the EXACT collapse `weight_ltm`'s own
        # 2026-08-31 fix was written to close (full-corpus answer-quality 37.2% -> 11.9% at
        # `weight_ltm=1.0` — `dto.py`'s own docstring) — a weight-only fix cannot both let LTM
        # place and keep the query-blind flat seed from flooding.
        #
        # `ltm_protect_limit` (default 0 — see dto.py's field docstring for why the mechanism
        # ships but is not turned on) reuses the SAME "protect membership, rescue at the TAIL if
        # fusion ranked it outside the window" mechanism already built for the STM floor
        # (`_merge_floor` below) instead of granting the channel more rank AUTHORITY: the graph
        # tier's top `ltm_protect_limit` candidates (already rank-ordered — recency-over-
        # currently-valid-facts, `_ltm_channel`) are guaranteed ONE OF the `limit` result slots
        # when the tier has any qualifying candidate, but `_merge_floor`'s rescue always APPENDS
        # a rescued member after the naturally-fused head — never re-orders it ahead of a
        # genuinely relevant hit. This is presence, not priority: it does not touch the ordering
        # property `test_default_settings_rank_the_relevant_mtm_hit_ahead_of_query_blind_ltm_
        # noise` already locks in (a query-blind LTM candidate still never outranks a relevant
        # MTM hit within the window) — it only stops the channel being pinned at exactly zero, IF
        # an operator turns it on. MEASURED at `ltm_protect_limit=1` (matched-width, 383 real
        # LoCoMo queries, 960 distilled facts, `docs/tracking/eval-runs/
        # 2026-09-24-ltm-channel-zero-slots.md`): the mechanism fires exactly as designed (383/383
        # queries got their guaranteed graph slot) and `gold_in_context` DROPS 280->273/383
        # (-1.83pt) — presence without a query-aware seed is a net cost, not merely inert, the
        # same shape the 2026-08-31 `weight_ltm` fix already found at a larger scale. Left at `0`
        # pending an entity-resolved graph seed (`ranker.py` module docstring's own "future work"
        # line); raising it before that would be motion, not progress.
        protected_ltm_views = [
            _to_view(s, "ltm") for s in ltm_hits[: self._settings.ltm_protect_limit]
        ]
        protected_floor_views = [*protected_floor_views, *protected_ltm_views]

        # Shape B — "fetch wider, narrow after expansion" (dto.py's own `neighbor_expand_widen`
        # docstring has the full rationale): widen `_merge_floor`'s working limit so an inserted
        # neighbour (and a protected-floor rescue) has more room to survive the rescue/cross-tier-
        # dedup pass BEFORE the pool is cut back to what the caller actually asked for.
        widen = (
            self._settings.neighbor_expand_widen
            if (self._settings.neighbor_expand_radius > 0 and not self._settings.neighbor_free_ride)
            else 0
        )
        items = _merge_floor(
            floor_views=protected_floor_views,
            fused=fused_views,
            limit=limit + widen,
            cross_tier_dedup=self._settings.cross_tier_dedup,
        )
        if widen:
            items = _narrow_after_expansion(items, limit=limit, neighbor_rescue_budget=widen)

        # Shape A — "free-riding insertion" (dto.py's own `neighbor_free_ride` docstring): a
        # neighbour rides along with the anchor that already earned its slot instead of competing
        # for one of its own. Runs AFTER `_merge_floor`/the widen-narrow step above have already
        # picked the `limit` winners — nothing already selected is ever displaced by this.
        if self._settings.neighbor_expand_radius > 0 and self._settings.neighbor_free_ride:
            items = await self._attach_free_riding_neighbors(
                ns,
                items,
                floor_pool_size=len(floor_scored),
                caller_identity_set=caller_identity_set,
            )

        # AD-250 fix (ADR 0061): the read-stat write-back — runs LAST, over the FINAL `items`
        # (a genuine recall hit is "this made it into what the caller actually got back", not
        # merely "was a channel candidate"), so this never reinforces a fused-out or
        # rerank-pruned STM row. `dto.py`'s `reinforce_on_recall` docstring has the full
        # rationale.
        # AD-259: and the SAME write-back for the MTM channel — the tier the user is still
        # actively using, where the whole point of the usage term lives. ADR 0061 gave the
        # trigger to a memory that had ALREADY been demoted; without this line a memory that is
        # recalled every single day still demotes on the identical schedule as one nobody has
        # touched, because `access_count` is the ONLY salience term a recall can move and nothing
        # moved it (`ports.py`'s `MtmTierRepository.reinforce` docstring has the measurement).
        # One gather over both channels, not two round trips: they are independent writes.
        if self._settings.reinforce_on_recall:
            await asyncio.gather(
                self._reinforce_stm_hits(ns, items), self._reinforce_mtm_hits(ns, items)
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

    def _effective_pool(self, limit: int) -> int:
        """POOL-TRAP FIX (``dto.py``'s ``channel_pool_multiplier`` docstring, ACCURACY-PLAN-0831.md
        §1.4): ``channel_pool_size`` is now a FLOOR on the per-channel fetch width, not the whole
        story — the pool SCALES with whatever ``limit`` this call actually resolved to (the
        caller's explicit limit, or the context-budget-derived one from ``RecallService``), so
        ``dto.py``'s own documented invariant ("per-channel fetch width > limit", ADR 0010) holds
        by construction instead of silently inverting the moment ``limit`` grows past the
        historically-fixed ``channel_pool_size=20``. MEASURED cost of the pre-fix shape
        (RETRIEVAL-EVAL-0829.md §13.1): raising the pool alone, independent of ``limit``, recovered
        recall@10 0.4093 -> 0.4546 (11% relative) — this makes that recovery the DEFAULT behaviour
        rather than something an operator has to remember to configure by hand on every width
        experiment."""
        scaled = math.ceil(limit * self._settings.channel_pool_multiplier)
        return max(self._settings.channel_pool_size, scaled)

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
            # `session_scope` is left at its DEFAULT (`None` = federate every one of the user's
            # sessions), exactly as this ranker already leaves `MtmTierRepository.recall`'s
            # identical parameter at its default. Until `graph_recall` GREW that parameter, the
            # LTM arm had no way to express it and filtered the full session-included namespace —
            # so the MTM arm federated across sessions while the LTM arm, the durable user-scoped
            # tier, stayed locked to the asking session. Same default, same semantics, one fabric.
            hits = await self._ltm.graph_recall(ns, limit=pool, caller_identity_set=caller)
        except StoreUnavailableError:
            return [], True
        if self._settings.ltm_max_hops <= 0:
            return hits, False
        try:
            # C2 FIX: the caller identity set goes to the traversal arm too — it is the SAME
            # `caller` the flat `graph_recall` seed above already receives. The traversal arm
            # DERIVES memory ids from a workspace-wide entity graph, so leaving it off here was a
            # live authorization bypass: SHARED hits came back unfiltered by `m.authorized_ids`
            # AND unfiltered by room.
            traversal_hits = await self._ltm.traverse_entities(
                ns,
                query=query,
                max_hops=self._settings.ltm_max_hops,
                limit=pool,
                caller_identity_set=caller,
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

    async def _expand_neighbors(
        self,
        ns: Namespace,
        fused_views: list[RecallItemView],
        *,
        floor_pool_size: int,
        caller_identity_set: CallerIdentitySet | None,
    ) -> list[RecallItemView]:
        """S1b — read-time neighbour expansion (``docs/tracking/TRACE-0923.md`` §7/§6.2/§5.1; ADR
        pending in ``docs/decisions/``; ``RecallSettings.neighbor_expand_radius``'s own docstring
        has the config-level rationale). §5.1's finding: 32.5% of this repo's own measured
        failures returned a turn within ±1 of the gold and NEVER the gold — the retriever finds
        the right conversational moment and returns the wrong turn of it, because the
        answer-bearing turn is usually a *reply* and the question's vocabulary lives one turn
        earlier. This inserts each fused candidate's ``turn_seq`` neighbours (``MemoryItem.
        turn_seq``, S1b's write-side field, ``local_memory.py::_next_turn_seq_base``) into the
        SAME pool ``_merge_floor`` truncates to ``limit`` — a neighbour competes for a slot
        exactly like the candidate that surfaced it (see the module-level ADR reference for why
        "costs a slot" was chosen over a free-riding wire-contract field this phase does not own).

        **Placement in the ranked list — and the bug an earlier version of this method shipped
        with.** A neighbour is fetched via a RAW STM lookup (this method's own session-window
        fetch below), so it is fundamentally an STM-CHANNEL candidate regardless of which channel
        surfaced its anchor. The first version of this method scored it as ``anchor_score -
        epsilon`` — which, for an MTM-sourced anchor (``weight_mtm=1.0``), put the neighbour at
        very nearly the MTM channel's score SCALE, entirely bypassing the ``weight_stm=0.1``
        discount AD-204 measured and shipped specifically to stop the STM channel from swamping
        MTM (``dto.py``'s own ``weight_stm`` docstring). MEASURED on `mu-dev-vm` real LoCoMo
        (3 conversations, 383 queries): this dropped `gold_in_context` from 257/383 (0.6710) to
        223/383 (0.5822), a net 34-query regression, and inverted the STM/MTM slot share from
        2638/1192 (MTM-dominant, the shipped
        baseline) to 1213/2617 (STM-dominant) — an MTM anchor's high-ranked neighbour was
        crowding out GENUINELY better-ranked MTM candidates the anchor's own channel had earned.

        **The fix: every neighbour scores on the STM channel's OWN weight family, never the
        anchor's.** A neighbour is treated as an UN-RANKED STM candidate entering just past the
        end of the real, already-ranked STM floor pool (``floor_pool_size``, the caller's
        ``len(floor_scored)`` for this query) — ``weight_stm / (rrf_k + floor_pool_size + offset)``,
        ``offset`` the conversational distance from its anchor (so a ±1 neighbour outranks a ±2
        neighbour of the same anchor, both still strictly below every genuinely STM-ranked
        candidate, since ``floor_pool_size + offset > floor_pool_size - 1``, the worst REAL rank
        in the pool). This can never exceed a real STM channel member's score and can never
        approach MTM's scale (bounded by ``weight_stm``, not by whatever channel found the
        anchor) — it restores AD-204's discount instead of routing around it, while still
        entering the SAME RRF competition every other candidate is in (re-sorted below), so a
        neighbour with nothing else competing for its slot still gets a fair look.

        **Graceful degradation (§7's explicit requirement).** An anchor with ``turn_seq=None``
        (any row written before S1b, or by a write path that assigns none) is simply not
        expanded — never treated as ``turn_seq=0``. A ``turn_seq`` COLLISION (two items sharing
        one value — the documented gap in ``local_memory.py::_TURN_SEQ_SCAN_LIMIT``/
        ``RecallSettings.neighbor_expand_session_scan_limit``) resolves to "the first one this
        method's own session-window fetch happens to see", never a raise. A store outage on that
        fetch degrades to NO expansion for this call (returns ``fused_views`` unchanged) — this is
        an enhancement, not one of the channels this ranker's own named degrade contract (§5)
        covers, so it fails OPEN to the pre-S1b result rather than failing the whole recall.
        """
        radius = self._settings.neighbor_expand_radius
        if radius <= 0:
            return fused_views
        # Shape A (dto.py's own `neighbor_free_ride` docstring): free-riding insertion is a
        # SEPARATE mechanism, applied by `rank()` AFTER `_merge_floor` has already picked the
        # winners (`_attach_free_riding_neighbors`) — this "costs a slot" path must not ALSO run,
        # or the same neighbour could both compete here and free-ride there.
        if self._settings.neighbor_free_ride:
            return fused_views
        anchors = [v for v in fused_views if v.turn_seq is not None]
        if not anchors:
            return fused_views
        by_turn_seq = await self._turn_seq_window(ns, caller_identity_set)
        if by_turn_seq is None:
            return fused_views  # enhancement, not a named channel — degrade to no expansion.

        present_ids = {v.memory_id for v in fused_views}
        neighbors: list[RecallItemView] = []
        by_anchor: dict[str, list[RecallItemView]] = {}
        for anchor in anchors:
            anchor_turn_seq = anchor.turn_seq
            if anchor_turn_seq is None:  # pragma: no cover - excluded by `anchors` filter above
                continue
            for view in self._neighbors_of(
                anchor_turn_seq,
                by_turn_seq,
                radius=radius,
                present_ids=present_ids,
                floor_pool_size=floor_pool_size,
            ):
                neighbors.append(view)
                by_anchor.setdefault(anchor.memory_id, []).append(view)
        # PLACEMENT. `fused_views` order is NEVER re-sorted here — VERIFY PASS 2026-09-23: this
        # method used to end with `expanded.sort(key=fused_score)` over the whole merged list,
        # which silently discarded `AdaptiveRerankGate.apply`'s ordering, because the gate records
        # its verdict in `rerank_score` and deliberately leaves `fused_score` at the RRF value
        # (D2/`_to_view`). Switching `neighbor_expand_radius` on therefore switched the reranker
        # off in effect (`test_s1b_expansion_does_not_discard_the_rerank_gates_ordering`).
        #
        # It is NOT enough to append instead: MEASURED (the real-store
        # `test_s1b_neighbor_expansion_int.py` went red), a neighbour's
        # `weight_stm / (rrf_k + floor_pool_size + offset)` score does sometimes exceed a real,
        # weakly-ranked STM candidate's, and appending would silently demote it below one — so the
        # old sort was load-bearing for the mechanism, just too broad. STABLE INSERTION keeps both
        # properties: each neighbour goes before the FIRST existing item scoring strictly lower
        # than it, and the relative order of everything already in the pool is untouched. When the
        # pool is score-ordered (the shipped config — the rerank gate is dark by default) that is
        # byte-identical to the old whole-list sort; when the gate HAS reordered, the gate wins.
        if self._settings.neighbor_expand_placement == "after_anchor":
            # Measured and REJECTED as a default (ARCHITECTURE-DELTAS.md AD-232's own arm H:
            # 246/383 against 281 shipped, with the STM/MTM slot share inverting) — kept as a
            # config-gated arm, not deleted, so the next reader does not re-derive it. Each
            # neighbour sits immediately behind the candidate that surfaced it.
            placed: list[RecallItemView] = []
            for view in fused_views:
                placed.append(view)
                placed.extend(by_anchor.get(view.memory_id, ()))
            return placed
        neighbors.sort(key=lambda v: v.fused_score, reverse=True)
        merged = list(fused_views)
        for view in neighbors:
            at = next(
                (i for i, existing in enumerate(merged) if existing.fused_score < view.fused_score),
                len(merged),
            )
            merged.insert(at, view)
        return merged

    async def _turn_seq_window(
        self, ns: Namespace, caller_identity_set: CallerIdentitySet | None
    ) -> dict[int, Scored[MemoryItem]] | None:
        """The ``turn_seq -> item`` lookup S1b's two expansion mechanisms (costs-a-slot and
        free-riding, both below) share — one session-window fetch, never duplicated per mechanism.
        ``None`` on a store outage (the SAME graceful-degrade contract ``_expand_neighbors``'s own
        docstring documents: an enhancement, not a named channel, so both callers fail OPEN to
        "no expansion" rather than failing the recall)."""
        try:
            window = await self._stm.recent(
                ns,
                limit=self._settings.neighbor_expand_session_scan_limit,
                caller_identity_set=caller_identity_set,
            )
        except StoreUnavailableError:
            return None
        by_turn_seq: dict[int, Scored[MemoryItem]] = {}
        for scored in window:
            seq = scored.item.turn_seq
            if seq is not None and seq not in by_turn_seq:  # first-found wins a collision
                by_turn_seq[seq] = scored
        return by_turn_seq

    def _neighbors_of(
        self,
        anchor_turn_seq: int,
        by_turn_seq: dict[int, Scored[MemoryItem]],
        *,
        radius: int,
        present_ids: set[str],
        floor_pool_size: int,
    ) -> list[RecallItemView]:
        """The ``±radius`` neighbour views for one anchor, scored on the STM channel's OWN weight
        family (``_expand_neighbors``'s own docstring, "the fix" — NEVER derived from the anchor's
        score, so a neighbour can never borrow a higher-weighted channel's scale). ``present_ids``
        is mutated in place (both call sites need a single ACCUMULATING de-dup set across every
        anchor they process, not a per-anchor-local one, so the same physical neighbour is never
        attached twice to two different anchors)."""
        found: list[RecallItemView] = []
        for offset in range(1, radius + 1):
            for seq in (anchor_turn_seq - offset, anchor_turn_seq + offset):
                if seq < 0:
                    continue
                neighbor = by_turn_seq.get(seq)
                if neighbor is None or neighbor.item.id in present_ids:
                    continue
                present_ids.add(neighbor.item.id)
                neighbor_score = self._settings.weight_stm / (
                    self._settings.rrf_k + floor_pool_size + offset
                )
                # `is_floor=False`: the STM adapter's own `recent()` stamps EVERY returned
                # candidate `is_floor=True` unconditionally — a neighbour earns `is_floor=True`
                # only if a later `_merge_floor` pass independently re-stamps it because it ALSO
                # happens to be a protected floor member, never because of how it was fetched.
                view = _to_view(neighbor, "stm", fused_score=neighbor_score).model_copy(
                    update={"is_neighbor": True, "is_floor": False}
                )
                found.append(view)
        return found

    async def _attach_free_riding_neighbors(
        self,
        ns: Namespace,
        items: list[RecallItemView],
        *,
        floor_pool_size: int,
        caller_identity_set: CallerIdentitySet | None,
    ) -> list[RecallItemView]:
        """Shape A — free-riding insertion (``dto.py``'s own ``neighbor_free_ride`` docstring has
        the full design rationale; ADR 0053 "what would need to change before this ships ON" §1).
        Runs AFTER ``_merge_floor`` (and the Shape-B widen/narrow step, when that is also
        configured — the two are mutually exclusive by construction, see ``rank()``) has already
        picked the ``limit`` winners: a neighbour of a SURVIVING item is appended to the returned
        list, never displacing anything already in it. Nothing here can shrink ``items`` or change
        the order/content of what is already present — the only observable effect is a LONGER
        list, exactly the trade `neighbor_free_ride`'s own docstring names ("the downside is
        bounded to prompt length").

        Uses the SAME session-window lookup and STM-family neighbour score as the costs-a-slot
        mechanism (``_turn_seq_window``/``_neighbors_of``) — a free-riding neighbour is not a
        different KIND of candidate, only a differently-PLACED one."""
        anchors = [v for v in items if v.turn_seq is not None]
        if not anchors:
            return items
        by_turn_seq = await self._turn_seq_window(ns, caller_identity_set)
        if by_turn_seq is None:
            return items  # store outage on the expansion-only fetch — degrade to no attachment.
        present_ids = {v.memory_id for v in items}
        radius = self._settings.neighbor_expand_radius
        extras: list[RecallItemView] = []
        for anchor in anchors:
            anchor_turn_seq = anchor.turn_seq
            if anchor_turn_seq is None:  # pragma: no cover - excluded by `anchors` filter above
                continue
            extras.extend(
                self._neighbors_of(
                    anchor_turn_seq,
                    by_turn_seq,
                    radius=radius,
                    present_ids=present_ids,
                    floor_pool_size=floor_pool_size,
                )
            )
        return [*items, *extras]

    async def _reinforce_stm_hits(self, ns: Namespace, items: list[RecallItemView]) -> None:
        """AD-250 fix (ADR 0061): the read-stat write-back a genuine recall hit performs.

        Fires :meth:`~mu_engine.storage.ports.StmTierRepository.reinforce` once per DISTINCT id
        in the FINAL result, de-duped so a row occupying two slots is reinforced exactly once per
        call. Concurrent (``asyncio.gather``), never serialised — a bounded, small number of
        independent writes, not a chain.

        **Every id, not just the ``channel == "stm"`` ones (AD-259 correction).** ADR 0061
        filtered on the channel label, which is the label of the FUSED winner: a memory that
        lives in BOTH tiers — the ordinary case for anything recently ingested, since
        ``WriteStmStage`` and ``DeterministicPromoteStage`` both write it — fuses to ONE view
        under ONE label, so filtering by label silently reinforced only one of the two rows.
        MEASURED: after ten genuine recalls of an item present in both tiers, the Valkey row's
        ``access_count`` was 10 and the Qdrant point's was **0** — and the demotion gate reads
        the Qdrant point. The two copies are DISTINCT ROWS with INDEPENDENT lifecycle gates
        (``scan_for_demotion`` over MTM, ``promote_stm_mtm`` over STM), so a recall of that
        memory is a genuine use of BOTH and each gate needs to see it. Over-inclusion is free
        and safe by contract: ``reinforce`` on a store that does not hold the id is a documented
        no-op returning ``None`` (``ports.py``), never a raise.

        Best-effort by design (this docstring's own contract, ``ports.py``'s ``reinforce``
        docstring): a store outage here degrades to NO reinforcement for this call, exactly like
        every other enhancement in this module (``_expand_neighbors``'s own graceful-degradation
        contract) — it must never fail, or even delay past its own writes, the read it is
        reinforcing. Only ``StoreUnavailableError`` is swallowed (logged); anything else is a
        programming bug and is left to raise — catching it here would hide a real defect behind
        "best effort"."""
        ids = list({v.memory_id for v in items})
        if not ids:
            return
        at = self._clock.now()

        async def _one(memory_id: str) -> None:
            try:
                await self._stm.reinforce(ns, memory_id, at=at)
            except StoreUnavailableError as exc:
                _log.warning(
                    "recall.reinforce_unavailable",
                    ns=ns.to_prefix(),
                    memory_id=memory_id,
                    error=str(exc),
                )

        await asyncio.gather(*(_one(mid) for mid in ids))

    async def _reinforce_mtm_hits(self, ns: Namespace, items: list[RecallItemView]) -> None:
        """AD-259: the MTM twin of :meth:`_reinforce_stm_hits` — same contract, same de-dupe,
        same concurrency, same best-effort degrade, same "every returned id" rule (that method's
        docstring has the reasoning), a different port and one extra guard.

        Kept as a second method rather than folded into one generic helper: the two ports are
        structurally unrelated Protocols and only this one needs the capability check below, so a
        generic version would have to erase the port types AND carry the guard for a caller that
        does not need it — cost with no reuse to show for it at two call sites.
        """
        ids = list({v.memory_id for v in items})
        if not ids:
            return
        # CAPABILITY CHECK, not a type check. `STORE_REGISTRY.build` is typed `-> Any`
        # (`storage/registry.py`), so the object bound to the `vector` role is whatever
        # `MU_STORAGE__VECTOR__BACKEND` selected, and three of the six shipped vector backends
        # (`PgVectorMtmAdapter`/`ChromaMtmAdapter`/`FaissMtmAdapter`) implement no by-id verbs at
        # all — `tier_capabilities.py`'s own module docstring names those three for exactly this
        # reason, and `mu_local/composition.py:413` already gates `set_entity_uids` on the same
        # `hasattr`. Without this line, adding an unconditional by-id write to the READ path
        # would turn every recall on a chroma/faiss/pgvector deployment into an AttributeError —
        # a hot-path crash bought for a best-effort stat write. Degrades silently on purpose:
        # this is an enhancement, and a backend that cannot answer it is a documented shape, not
        # an incident (an outage of a backend that CAN is logged, in `_one` below).
        if not hasattr(self._mtm, "reinforce"):
            return
        at = self._clock.now()

        async def _one(memory_id: str) -> None:
            try:
                await self._mtm.reinforce(ns, memory_id, at=at)
            except StoreUnavailableError as exc:
                _log.warning(
                    "recall.reinforce_mtm_unavailable",
                    ns=ns.to_prefix(),
                    memory_id=memory_id,
                    error=str(exc),
                )

        await asyncio.gather(*(_one(mid) for mid in ids))


def _narrow_after_expansion(
    items: list[RecallItemView], *, limit: int, neighbor_rescue_budget: int
) -> list[RecallItemView]:
    """Shape B's second half — "fetch wider, narrow after expansion" (``dto.py``'s own
    ``neighbor_expand_widen`` docstring). ``items`` was produced by ``_merge_floor`` at a WIDENED
    working limit (``limit + neighbor_expand_widen``); this cuts it back down to the caller's real
    ``limit``, in two stages:

    1. **Never break ``_merge_floor``'s own "a protected floor member is never evicted"
       invariant.** A blind ``items[:limit]`` slice could otherwise push a rescued protected item
       (which ``_merge_floor`` deliberately appends at the END, past the naturally-ranked head)
       back out past the real limit — exactly the guarantee AD-195/ADR 0052 exist to keep.
       ``is_floor`` members are therefore kept unconditionally, regardless of budget.

    2. **The actual "expansion informs ranking" mechanism**: up to ``neighbor_rescue_budget`` of
       the neighbours that survived the WIDENED first cut (``is_neighbor``, not also
       ``is_floor`` — a rescued floor member is already covered by stage 1) are guaranteed a
       final slot too, in their already-established order (best-scored first — ``_expand_
       neighbors``'s stable insertion, never re-sorted here). Unlike Shape A (``neighbor_free_
       ride``, which NEVER displaces anything), this stage DOES displace the weakest ordinary
       (non-floor, non-neighbour) candidates to make room when neither this method nor
       ``_merge_floor``'s own natural ranking gave the neighbour a slot on merit alone — the
       priced trade this shape commits to, capped at ``neighbor_rescue_budget`` displacements so
       the cost is bounded, not unlimited the way an uncapped rescue would be."""
    if len(items) <= limit:
        return items
    # SELECT, then EMIT IN THE INCOMING ORDER. VERIFY 2026-09-24: this used to partition the pool
    # into three buckets and CONCATENATE them (`rest`, then rescued neighbours, then floor
    # members), which is a RE-ORDER of a list every stage upstream treats as a ranking. It is the
    # same defect class AD-232 found one stage earlier (`_expand_neighbors` ending with
    # `expanded.sort(...)`, silently discarding the rerank gate's ordering) and fixed with the same
    # rule: decide membership here, never position. Two concrete inversions it produced: a
    # neighbour is scored at the very BOTTOM of the pool by construction (`weight_stm / (rrf_k +
    # floor_pool_size + offset)`, AD-231) and `_expand_neighbors` appends it LAST, yet it was
    # emitted ahead of the protected just-said fact ADR 0052 exists to keep — and the injector's
    # token budgeter trims from the TAIL, so a tight budget dropped the protected row and kept the
    # speculative one; and `_merge_floor`'s own D3 contract ("a protected member keeps whatever
    # position fusion actually earned it", STATE-AND-DEFECTS-0829.md) was discarded for every
    # protected member on every widened call.
    keep: set[int] = set()
    # (1) every protected floor member, unconditionally — `_merge_floor`'s "never evicted"
    #     guarantee (AD-195/ADR 0052) survives the narrow, exactly as before.
    for i, view in enumerate(items):
        if view.is_floor:
            keep.add(i)
    # (2) up to `neighbor_rescue_budget` neighbours, in their already-established order — and only
    #     while doing so does not push the count past `limit`, so a neighbour can never be the
    #     reason a protected member is evicted (the ordering between (1) and (2) is the priority,
    #     not a position).
    rescued = 0
    for i, view in enumerate(items):
        if rescued >= neighbor_rescue_budget or len(keep) >= limit:
            break
        if view.is_neighbor and not view.is_floor:
            keep.add(i)
            rescued += 1
    # (3) fill the remaining slots with the best-ranked ordinary candidates — the weakest ones are
    #     what a rescued neighbour displaces, capped at `neighbor_rescue_budget` displacements.
    for i, _view in enumerate(items):
        if len(keep) >= limit:
            break
        keep.add(i)
    return [view for i, view in enumerate(items) if i in keep][:limit]


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
    # AD-250 fix (ADR 0061): `.value.startswith("stm")`, not an `is RecallChannel.STM_FLOOR`
    # identity check — `RecallChannel.STM_DEMOTED` (the new demoted-item channel) is a DISTINCT
    # enum member for store-level provenance (its own docstring), but both fold to ONE channel
    # here, exactly as `MTM_HYBRID` already folds into "mtm" alongside `MTM_DENSE`/`MTM_SPARSE`.
    if scored.channel.value.startswith("stm"):
        return "stm"
    return "mtm"


def _protected_floor_ids(
    *,
    floor: list[Scored[MemoryItem]],
    floor_scored: list[Scored[MemoryItem]],
    protect_n: int,
    min_relevance: float,
) -> set[str]:
    """T1 option (c) (``docs/tracking/TRACE-0923.md`` §7/§7.1; ADR pending in
    ``docs/decisions/``): the "never evict a just-said fact" guarantee (AD-195) stays intent-true
    — a just-said fact IS still protected — but it must now also clear a relevance bar, so it no
    longer spends a result slot on a recent item the query has nothing to do with.

    **Why this, and not lowering ``floor_protect_limit``.** §7.1 laid out the honest options: (a)
    keep 3 and pay the cost, (b) lower the default and weaken the guarantee everywhere including
    the live-agent session it was built for, or (c) keep the guarantee's SHAPE (bounded by the
    SAME ``floor_protect_limit`` membership window this always used) and make membership
    CONDITIONAL. This is (c): ``floor`` is still the recency-ordered candidate pool and
    ``protect_n`` still bounds HOW MANY of its most-recent members are even eligible — identical
    to the pre-T1c selection — but a member only clears the gate when its OWN relevance score (in
    ``floor_scored``, the SAME per-candidate score :meth:`ThreeChannelRecallRanker._score_stm`
    already computes for floor re-ordering and for the RRF channel input) is ``>= min_relevance``.
    A member that misses the bar is not specially penalised either — it simply re-enters the
    ordinary fused competition like any other STM candidate that was never protected (§ranker.py
    module docstring, "the rest of the STM candidate pool ... competes in the SAME RRF fusion").

    **What the guarantee now promises** (the ADR's own words, recorded here so the code and the
    decision record cannot drift): *"the most recent STM fact is never evicted from the answer
    window PROVIDED it clears ``min_relevance`` against the asked query — an irrelevant just-said
    aside is no longer owed a slot."* This is a real, deliberate narrowing of AD-195, not a bug —
    AD-195's original promise (protect every one of the top ``floor_protect_limit`` regardless of
    relevance) is what §7's own measurement (3 fixed slots on EVERY query, 30% of the window) says
    is expensive specifically on adversarial benchmarks like LoCoMo where the last few turns are
    rarely the answer; a live agent session's last turns are usually genuinely relevant, so the
    gate is expected to pass them through unchanged there.

    **Default is 0.5, shipped on evidence.** A real `gold_in_context` sweep on `mu-dev-vm`
    recovered essentially all of the measured unconditional-guarantee cost at this bar
    (``RecallSettings.floor_protect_min_relevance``'s own docstring has the full sweep table).
    ``-1.0`` — cosine similarity's own true theoretical minimum (``stm_scoring="embed"``'s real
    range is ``[-1.0, 1.0]``, NOT ``[0.0, 1.0]`` — an earlier ``0.0`` default was live-caught as
    wrong exactly because of this, see that same docstring for the regression) — is the value
    that reproduces the ORIGINAL, fully unconditional AD-195 guarantee, still reachable via
    ``MU_RECALL__FLOOR_PROTECT_MIN_RELEVANCE=-1``.

    **``stm_scoring="recency"`` makes the bar UNIFORM, not conditional.** Under "recency" (D1's
    documented pre-fix opt-out), ``_score_stm`` is a no-op and every candidate in ``floor_scored``
    carries the SAME constant channel-native score (``1.0``) — so ``min_relevance`` degrades to a
    single global on/off switch (protect the top ``protect_n`` unconditionally when
    ``min_relevance <= 1.0``, protect none when it is higher), never a per-candidate filter. This
    is documented rather than special-cased or refused: a deployment that pins ``stm_scoring=
    "recency"`` has already opted out of per-candidate relevance everywhere else in this ranker
    (the floor's own reorder-by-relevance, the RRF channel input), and T1c does not invent a
    relevance signal recency mode deliberately has none of.
    """
    relevance_by_id = {s.item.id: s.score for s in floor_scored}
    return {
        s.item.id
        for s in floor[:protect_n]
        if relevance_by_id.get(s.item.id, s.score) >= min_relevance
    }


def _merge_floor(
    *,
    floor_views: list[RecallItemView],
    fused: list[RecallItemView],
    limit: int,
    cross_tier_dedup: bool,
) -> list[RecallItemView]:
    """Merge the (BOUNDED, §3.1/#1 bug fix) STM floor in AFTER fusion (hybrid.py:247): fusion may
    reorder but never evict a protected floor member. ``floor_views`` is capped upstream to
    ``settings.floor_protect_limit`` — NOT the whole STM candidate pool.

    **D3 (STATE-AND-DEFECTS-0829.md) — floor members no longer lead unconditionally.** Every
    protected member's id is already IN ``fused`` (``floor_scored``, the pool ``floor_views`` is
    drawn from, is one of the three channels ``ThreeChannelRecallRanker.rank`` hands to
    ``fusion.fuse`` above) — the RRF-competed rank fusion already gave it a real signal, not a
    fallback. The pre-fix code discarded that rank for every protected member and force-prepended
    it instead, which is exactly what made the top ``floor_protect_limit`` slots of EVERY result
    the ``floor_protect_limit`` most-recently-written memories, chosen without reference to the
    query: measured on LoCoMo (1,531 labelled queries, RETRIEVAL-EVAL-0829.md §5), ``floor_items =
    4593 = 1531 x 3`` exactly, ``recall@1`` was ``0.0003`` against ``0.1625`` for the MTM channel
    used alone, and disabling the floor outright (non-default) recovered most but not all of that
    gap — the fuse was STILL worse than its own dense channel at every cutoff. §5.3 there names
    the design question this answers: *"what should 'never evict a just-said fact' mean when the
    just-said fact is irrelevant?"* This function answers it as: **a protected member keeps
    whatever position fusion actually earned it; only a member fusion ranked OUTSIDE the returned
    window is rescued — appended at the END, not the front** — so "never evicted" still holds
    (every protected id is guaranteed present) without a query-blind item pre-empting a slot a
    genuinely relevant hit earned. ``is_floor`` stays a MEMBERSHIP flag (which ids are protected),
    never a position instruction: :func:`_to_view` already forces it False on every ``fused``
    entry (a fused-in STM item competes like any other channel once it is not protected), so this
    function re-stamps ``is_floor=True`` onto each protected id's ``fused`` occurrence IN PLACE, at
    whatever rank fusion gave it — the FLAG only, never the whole ``floor_views`` twin, whose
    ``fused_score`` is the STM-native relevance score and not the RRF value D2 requires this field
    to carry — before deciding who needs rescuing.

    D4 cross-tier dedup (conformance D-8, ``settings.cross_tier_dedup``): the STM/MTM/LTM fuse
    above merges by ``MemoryItem.id`` only, so the SAME fact surfaced from two different tiers
    under two different ids (e.g. its STM row and an already-promoted MTM/LTM copy sharing
    ``content_hash``) would otherwise occupy two of the ``limit`` slots below. Deduping by
    ``content_hash`` HERE, over the fused rank order (a rescued protected row can still lose its
    slot to an earlier, better-ranked duplicate — the SAME "the returned window already earned its
    position" principle this function now applies throughout, a deliberate change from the pre-fix
    "floor members first" precedence), means a dup never crowds out a genuinely distinct fact
    (matches the read-time half of the ``ada_coffee`` double-write finding, DATA-QUALITY-
    ASSESSMENT.md §3.1/#5: "Coffee-query context contained each fact twice")."""
    # D2 (STATE-AND-DEFECTS-0829.md), the half the first pass missed: restore MEMBERSHIP only.
    # Substituting the whole `floor_views` twin — which is what this line did first — swapped the
    # RRF value back out for the STM-native relevance score `_to_view(s, "stm")` stamped on it,
    # i.e. it re-introduced D2's exact bug for precisely the protected rows, and it made the
    # returned list carry TWO incomparable score scales (~1e-1 STM against ~1e-2 RRF), which is
    # the shape AD-194 had to teach `PersonaAffinityShaper` to work around. There is nothing else
    # to restore: `fusion.reciprocal_rank_fusion` keeps "the FIRST occurrence of a key (by channel
    # order) as the representative element" and `floor_scored` is channel 0, so a protected
    # member's fused view already carries `channel="stm"` and the same `MemoryItem` — the twin
    # differed in `fused_score` (wrongly) and `is_floor` (rightly) and in nothing else.
    protected_ids = {v.memory_id for v in floor_views}
    natural = [
        v.model_copy(update={"is_floor": True}) if v.memory_id in protected_ids else v
        for v in fused
    ]
    head_ids = {v.memory_id for v in natural[:limit]}
    # Every protected id IS in `fused` (`floor_scored` is one of the three fused channels and RRF
    # returns the union, never a truncation), so the comprehension below finds them all. The
    # `absent` tail is a defence in depth against a future channel-list change quietly breaking
    # "never evicted", not a case that can fire today.
    rescued = [v for v in natural if v.is_floor and v.memory_id not in head_ids]
    present = {v.memory_id for v in natural}
    rescued += [v for v in floor_views if v.memory_id not in present]
    room = max(0, limit - len(rescued))
    candidates = [*natural[:room], *rescued]
    if cross_tier_dedup:
        candidates = dedup_by_content_hash(candidates)
    return candidates[:limit]
