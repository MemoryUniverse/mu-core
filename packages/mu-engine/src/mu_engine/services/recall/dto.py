"""Recall read-path DTOs — the one request/result shape for every recall entry point.

Pins ``recall-service-design.md §1.1`` (RecallQuery/RecallResult/RecallItemView/RecallChannels/
RecallMode) onto the un-collapsed η + the engine ``MemoryItem``. Every DTO is a FROZEN pydantic-v2
value object (DEV-STANDARDS rule 2): created per operation, never shared across tasks, never stored
on a singleton (lifecycle-scoping §2.1).

WHY content is present on ``RecallItemView`` while events are content-free: the ``RecallResult`` is
the IN-PROCESS read payload handed back to the caller — carrying the body is the whole point of a
read. The content-free discipline (CANONICAL §3.1) governs BUSES/logs/metrics, not this return
value; the ranked recall path emits NOTHING on the bus (recall-service-design §1.2 step 5), and the
content-free projection for any downstream event is ``RecallResult.memory_ids``.

RE-HOME NOTE: ``recall-service-design §1.1`` pins these into ``domain/model/recall.py``; they live
here beside the service because the mu-contracts recall module ships only ``Scored``/``SparseQuery``
this phase (same re-home pattern as ``mu_engine.storage.domain.recall``). ``RecallSettings`` is a
config VO taken explicitly at the composition root (the ``PlatformSelectors`` pattern) until the
``settings.recall`` subtree lands — config-sourced at the call site, never hardcoded (rule 3).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mu_contracts.contracts.defaults import DEFAULT_RECALL_LIMIT
from mu_contracts.domain.events import DegradeReason
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine.storage.domain.namespace import Namespace

__all__ = [
    "RecallChannels",
    "RecallItemView",
    "RecallMode",
    "RecallQuery",
    "RecallResult",
    "RecallSettings",
]


class RecallChannels(BaseModel):
    """Which channels this recall runs. A degraded recall drops one (§5)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stm: bool = True
    mtm: bool = True
    ltm: bool = True


class RecallMode(StrEnum):
    """The read shape requested (§1.1). Only ``RANKED`` is shipped this phase; ``ANSWER``/``INJECT``
    (LLM synthesis / additionalContext render) are DEFERRED — no LLM on this path (Azure PARKED)."""

    RANKED = "ranked"
    ANSWER = "answer"
    INJECT = "inject"


class RecallQuery(BaseModel):
    """The one request object for every recall entry point. Immutable, transient (§1.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: Namespace  # η — the tenancy partition (to_prefix scoping, §1.4)
    text: str = Field(min_length=1)
    # ``None`` (the default) means "derive it from the consuming model's context budget"
    # (`RecallService._effective_limit` / `services/recall/width.py`, ACCURACY-PLAN-0831.md item
    # 4) — an explicit int ALWAYS wins over derivation (§ "explicit config still wins"), which is
    # exactly why this is not simply defaulted to `DEFAULT_RECALL_LIMIT` any more: a caller that
    # wants that WIRE default has to say so (the wire-layer `RecallRequest.limit`, `mu_contracts.
    # contracts.requests`, still defaults to `DEFAULT_RECALL_LIMIT` explicitly for API callers
    # that have not opted into derivation — the two defaults are deliberately different layers).
    limit: int | None = Field(default=None, ge=1)
    channels: RecallChannels = RecallChannels()
    mode: RecallMode = RecallMode.RANKED
    persona: str | None = None  # reserved for the DEFERRED answer/inject persona adaptation (§3.2)
    max_tokens: int | None = None  # reserved for the DEFERRED inject budget ceiling (§2.3)
    correlation_id: str | None = None  # threads events + trace across the read

    # Cross-session, per-user memory (ADR 0030; spec §1 ripple table). ``None`` (the new
    # DEFAULT) federates every one of the user's sessions — the PRIVATE-own MTM arm resolves
    # to the truncated user-prefix match (BQ3, ``qdrant_mtm.py:_recall_filter``); a caller
    # that wants the OLD single-session-narrowed behavior sets this to a concrete session id
    # (need not equal ``namespace.session`` — narrows to ANY one of the user's sessions).
    # SHARED rooms IGNORE this field unconditionally: rooms are real walls (session-as-wall
    # stays mandatory), never opted out via this predicate (AC-4.3, §1 S5 test obligation).
    session_scope: str | None = None

    def for_namespace(self, ns: Namespace) -> RecallQuery:
        """A copy re-pointed at ``ns`` — how the service builds the SHARED-arm query from the
        PRIVATE-session query (federate-live §1.6): same text/limit/channels, shared η."""
        return self.model_copy(update={"namespace": ns})


class RecallItemView(BaseModel):
    """One ranked hit. ``content`` is the body fetched from the owning store by id — NEVER read
    from a bus event (§1.1). ``content_hash`` is the cross-arm dedup key (federate-live §1.6)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    content: str
    content_hash: str  # federate-live dedup key: a pulled copy + its shared origin collapse (§1.6)
    tier: MemoryTier
    channel: str  # "stm" | "mtm" | "ltm" — provenance of the hit
    namespace: Namespace  # the η the hit came from (belt-and-suspenders re-assert, §1.4)
    fused_score: float
    # None when the rerank gate is dark (no reranker configured / `rerank_enabled=False`), when
    # the reranker's own model call failed, when the gate's empty-fallback fired, or on any
    # candidate beyond `rerank_pool_size` the gate never sent to the model (`rerank_gate.py`'s own
    # module docstring has the full design — the gate WAS dark unconditionally until it landed).
    rerank_score: float | None = None
    is_floor: bool = False  # STM recency-floor member — reorderable, NEVER evicted (§1.3)
    artifact_ref: str | None = None  # CANONICAL §3/§7.10 (G5): the linked ContextArtifact id


class RecallResult(BaseModel):
    """The ranked read result handed to the caller. The content-free projection for any downstream
    event is :attr:`memory_ids` (§1.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: Namespace
    items: list[RecallItemView]
    channels_run: RecallChannels  # what actually ran (a degrade narrows this)
    degraded: DegradeReason | None = None  # the NAMED degrade taken, else None (§5)
    generated_at: datetime

    @property
    def memory_ids(self) -> list[str]:
        """The content-free projection for events (§1.1 / CANONICAL §3.1)."""
        return [it.memory_id for it in self.items]

    def with_degrade(self, reason: DegradeReason) -> RecallResult:
        """Return this result re-labelled with a NAMED degrade (never a silent partial, §1.6)."""
        return self.model_copy(update={"degraded": reason})


class RecallSettings(BaseModel):
    """Recall tuning knobs — config VO threaded at the composition root (§6; ADR 0023 defaults).

    No knob is hardcoded in the ranker/service (DEV-STANDARDS rule 3): the defaults below are the
    ADR-measured combined configuration, overridable from ``settings.recall`` once that subtree
    lands. ``rrf_k`` matches the ported ``reciprocal_rank_fusion`` constant (hackathon
    ``shared/retrieval/fusion.py``); weights are per-arm (federation) and per-channel (in-arm)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str = "rrf_3channel_v1"  # recall_registry key (§1.3)
    rrf_k: int = Field(default=60, ge=1)  # RRF smoothing constant (fusion.py default)
    recency_floor_limit: int = Field(default=10, ge=0)  # STM floor CANDIDATE pool width (§1.3)
    # Bug fix (data-quality assessment §3.1/#1, 2026-07-31): the floor candidate pool used to be
    # merged in FRONT of the fused MTM/LTM tail UNCONDITIONALLY, and defaulted to the SAME width as
    # the result `limit` (10==10) — so on any session with >= `limit` STM items, the floor consumed
    # the entire result budget and the query-relevant fused channels never got a single slot
    # (every `recall()` in a session became a byte-identical, query-blind insertion-order dump).
    # `floor_protect_limit` bounds how many of the (still recency-ordered) STM candidates are
    # UNCONDITIONALLY protected from eviction — the "never evict a just-said fact" guarantee — while
    # the REST of the STM candidate pool now competes in the SAME RRF fusion as MTM/LTM (§1.3 "one
    # fuse implementation") instead of being force-prepended. This keeps the recency intent (the
    # `floor_protect_limit` most-recent facts are always recallable) without letting the STM channel
    # swamp every other channel's relevance signal.
    floor_protect_limit: int = Field(default=3, ge=0)
    channel_pool_size: int = Field(default=20, ge=1)  # per-channel fetch width > limit (ADR 0010)

    # POOL-TRAP FIX (ACCURACY-PLAN-0831.md §1.4, "channel_pool_size does not scale with limit, so
    # a naive width experiment measures nothing"): `channel_pool_size` above used to be the WHOLE
    # per-channel fetch width, independent of the caller's `limit` — so `dto.py`'s own documented
    # invariant ("per-channel fetch width > limit", ADR 0010) silently inverted at `limit >= 20`,
    # and MEASURED the entire candidate universe was ~30 items (20 dense + 10 recency) regardless
    # of what `limit` asked for (RETRIEVAL-EVAL-0829.md §13.1). `ThreeChannelRecallRanker` now
    # takes the per-channel pool as
    # ``max(channel_pool_size, ceil(limit * channel_pool_multiplier))``
    # — `channel_pool_size` becomes a FLOOR, not the whole story, so the pool follows whatever
    # `limit` the caller (or the width derivation above) actually resolves to. The 2.0 default is
    # not arbitrary: it is exactly the ratio §13.1's own k-curve run used by hand (pool=120 at
    # limit=60) to close the "does the fuse cost recall vs its own dense channel" question for
    # every cutoff up to k=60 — this makes that ratio the DEFAULT instead of something an operator
    # has to remember to set per width experiment.
    channel_pool_multiplier: float = Field(default=2.0, ge=1.0)

    # WIDTH DERIVATION (ACCURACY-PLAN-0831.md item 4, `services/recall/width.py`). A `RecallQuery`
    # with `limit=None` (the DTO default, above) asks `RecallService` to derive the width from the
    # consuming model's own context budget instead of a hardcoded constant — see `width.py`'s
    # module docstring for the formula and for why every constant below is what it is. An explicit
    # `RecallQuery.limit` always overrides every field in this block (§ "explicit config still
    # wins"); `derive_limit_from_budget=False` opts a deployment out entirely (falls back to
    # `DEFAULT_RECALL_LIMIT`, the same wire default this engine always shipped, never a silent 0).
    # Env override: `MU_RECALL__DERIVE_LIMIT_FROM_BUDGET=false`.
    derive_limit_from_budget: bool = Field(default=True)
    # ~370-token mem0 `ANSWER_PROMPT` scaffold overhead (measured — see width.py docstring).
    prompt_reserve_tokens: int = Field(default=370, ge=0)
    # Reserved for the model's own completion — the DEFERRED ANSWER mode's budget, not this
    # (RANKED-only) phase's; kept here because a width big enough to leave the answering model no
    # room to reply is a regression regardless of which mode consumes the recall.
    answer_reserve_tokens: int = Field(default=800, ge=0)
    # ~45 tokens per rendered memory line (measured — see width.py docstring: (823 - 370) / 10).
    tokens_per_memory: float = Field(default=45.0, gt=0.0)
    # Never derive BELOW the wire default — a starved budget still gets a WORKING window, never a
    # crippled one (FULL-LOCAL boundary rule; `width.py`'s own floor-clamp docstring).
    min_derived_limit: int = Field(default=DEFAULT_RECALL_LIMIT, ge=0)
    # Capped at the last width actually shown to help ANSWER accuracy (k=30, §13.2), not merely to
    # raise recall (k=60 was recall-only, never answer-quality-validated) — see width.py's own
    # "why 30, not 60" docstring section for the full citation. Raise this explicitly once a wider
    # answer-quality run reports where the curve turns.
    max_derived_limit: int = Field(default=30, ge=1)

    # AD-204 (channel rank-authority, RETRIEVAL-EVAL-0829.md §5.3 / STATE-AND-DEFECTS-0829.md D3):
    # equal (1.0/1.0/1.0) in-arm weights gave a ten-item, single-session STM recency window the
    # SAME RRF rank authority as an MTM ANN search over the WHOLE partition — rank-based RRF only
    # ever looks at a channel's OWN rank position, never how large or how targeted the pool that
    # rank came from, so the STM channel's best-of-ten (possibly irrelevant) candidate tied the
    # MTM channel's genuine best-of-the-corpus candidate for the SAME `1/(k+1)` vote. MEASURED on
    # LoCoMo (1,531 labelled queries, real Qdrant/Valkey/FalkorDB, `eval/vm_eval.sh baseline`):
    # equal weights made the shipped 3-channel fuse *worse than its own MTM channel alone at every
    # cutoff* (recall@1 0.0049 vs 0.1625 — a 33x gap; recall@10 0.3415 vs 0.4553), even with the
    # floor-position fix (AD-195) already landed.
    #
    # `weight_stm=0.1` (a 10:1 discount against MTM/LTM, both left at 1.0 — only the RATIO after
    # normalization matters, `fusion.py`'s `reciprocal_rank_fusion`) closes it: re-measured on the
    # SAME 1,531 queries, recall@1/3/5 land within one query's width of the MTM-alone channel
    # (0.1608/0.2946/0.3668 vs 0.1625/0.2939/0.3675 — @3 actually edges it), and recall@10 is the
    # ONE cutoff still short (0.4096 vs 0.4553) — but that residual is NOT this defect: re-running
    # with `floor_protect_limit=0` (non-default, diagnostic-only — isolates the fuse from the
    # floor) at the SAME `weight_stm=0.1` gives 0.1608/0.2946/0.3668/0.4559, matching or exceeding
    # the MTM-alone channel at every one of the four cutoffs. The k=10 shortfall at the SHIPPED
    # `floor_protect_limit=3` is the SEPARATE, already-decided "never evict a just-said fact"
    # guarantee (AD-195) spending up to 3 of the 10 returned slots on rows fusion ranked outside
    # the window — a cost AD-195 already priced and chose to keep, not a re-opening of it. This fix
    # closes the rank-authority mismatch RRF had between channels of very different search breadth;
    # it does not, and should not, touch that separate trade-off.
    #
    # Picked 0.1 (a 10:1 discount) over a more extreme value on purpose: 2.0/3.0/4.0 on
    # `weight_mtm` (equivalently ~2:1-4:1 against STM) looked sufficient on a 3-conversation subset
    # (382 queries) — recall@1 rose from 0.0026 to 0.1490 there — but did NOT fully close the gap
    # on the FULL 1,531-query corpus (recall@1 stalled at 0.1426 vs 0.1625 even with
    # `floor_protect_limit=0`, i.e. a small subset can look "saturated" at a ratio the full corpus
    # proves is not yet enough). `weight_stm=0.1` (10:1) was the smallest discount tested that
    # closed the FULL-corpus gap; `0.05`/`0.02` (20:1/50:1) reproduce it to the same 4 decimals, so
    # this is the threshold, not an arbitrarily large number picked past it.
    #
    # `weight_ltm` update (accuracy lane, 2026-08-31, RETRIEVAL-EVAL-0829.md §11): AD-204 above
    # left this at 1.0 because "the LTM channel contributed ZERO items to any top-10 result across
    # the whole measurement run" — true when written, but only because NO eval command in this
    # repo had ever called `LocalMemory.consolidate()` (MTM->LTM DISTILL): the graph tier was
    # UNCONDITIONALLY EMPTY in every baseline/answer-quality run on record, so `weight_ltm` had
    # never been exercised against real content. Fixed the harness gap (`eval/mu_eval/corpus.py`
    # `consolidate=` param) and re-measured with the graph tier actually populated: at the shipped
    # `weight_ltm=1.0`, once populated the LTM channel wins ~30% of result slots (4,603/15,310 in
    # the full 1,531-query LoCoMo run) and recall@10 COLLAPSES (0.4096 with LTM empty -> 0.3329
    # with LTM populated at weight 1.0, -27% relative) — and the SAME collapse reproduces on the
    # real end-to-end metric: `mu_eval answer-quality`, full corpus, gpt-5 answerer+judge, overall
    # accuracy 37.2% (§10, LTM empty) -> **11.9%** with LTM populated at weight 1.0, every category
    # worse including multi-hop (14.4% -> 1.8%). Root cause is EXACTLY AD-204's own diagnosis,
    # replayed for the channel AD-204 couldn't yet test: `graph_recall`'s flat seed is
    # RECENCY-ordered over the WHOLE partition, `subject=None` — query-blind, like the STM window
    # AD-204 already fixed — and it was competing at FULL rank authority (`weight_ltm=1.0`, equal
    # to the query-aware MTM dense search) the moment it had real candidates to rank. Applying
    # AD-204's own already-proven fix pattern here (discount 1.0 -> 0.1, a 10:1 handicap, same
    # ratio as `weight_stm`) drops LTM's slot share to ~0 and recovers recall@10 to 0.4055 — within
    # noise of the LTM-EMPTY baseline (0.4096), i.e. the fix stops the collapse. It does not yet
    # show LTM populated beats LTM empty (that would need the flat seed to stop being query-blind,
    # a bigger change, not attempted here) — only that a populated-but-uncorrected-weight graph
    # tier is actively harmful, and the correction available today is the one already proven for
    # the sibling channel. `weight_stm=0.1`/`weight_ltm=1.0` (the pre-fix combination) is therefore
    # DANGEROUS specifically the day something starts actually writing the graph tier in
    # production (a lifecycle sweep, a manual `consolidate` call) — this was a live, armed
    # regression waiting for exactly that trigger, caught here only because the harness gap that
    # hid it (§11) was closed in the same pass. In-arm recency-channel weight (§1.3 fuse; AD-204).
    weight_stm: float = Field(default=0.1, ge=0.0)
    weight_mtm: float = Field(default=1.0, ge=0.0)  # in-arm dense weight (§1.3 fuse)
    weight_ltm: float = Field(default=0.1, ge=0.0)  # in-arm graph weight (§1.3 fuse; see above)
    weight_private: float = Field(default=1.0, ge=0.0)  # federation: private-arm weight (§1.6)
    weight_shared: float = Field(default=1.0, ge=0.0)  # federation: shared-arm weight (§1.6)

    # D4 read-time cross-tier dedup (CONFIG-AND-DATA-FIX-PLAN.md PART 2 D4; conformance D-8):
    # ``ThreeChannelRecallRanker.rank`` fuses STM-floor + MTM + LTM by ``MemoryItem.id`` only — two
    # DIFFERENT ids carrying the SAME ``content_hash`` (e.g. a fact present in both its STM raw form
    # and an already-promoted copy) survive as two separate ``RecallItemView`` rows and double up in
    # ``build_context`` (DATA-QUALITY-ASSESSMENT.md §3.1/#5 "Coffee-query context contained each
    # fact twice"). ``RecallService.recall`` already runs ``dedup_by_content_hash`` at the
    # PRIVATE⊕SHARED federation seam (fusion.py); this flag gates the SAME primitive applied one
    # layer down, on the per-arm STM/MTM/LTM candidate set, before ``floor_protect_limit`` truncates
    # it — so a duplicate never occupies two of the ``limit`` result slots in the first place. Env
    # override: ``MU_RECALL__CROSS_TIER_DEDUP=false`` reverts to the pre-fix behavior (duplicates
    # allowed through) for A/B comparison (DEV-STANDARDS rule 3).
    cross_tier_dedup: bool = Field(default=True)

    # D1 STM relevance scoring (DATA-QUALITY-ASSESSMENT.md §3.1, floor-fix follow-up to 02fbed9):
    # ``recency_floor_limit``/``floor_protect_limit`` bound HOW MANY STM candidates enter the fuse
    # and HOW MANY are unconditionally protected — but the candidates themselves still carried NO
    # relevance signal of their own; the STM channel entered RRF ordered by RECENCY RANK ONLY
    # (``StmTierRepository.recent`` newest-first). Within one session a targeted query and a
    # nonsense query therefore still surfaced a near-identical STM-dominated list: the floor
    # protected the right COUNT of items but always the same (most-recent) ones, in the same
    # order, regardless of query. ``stm_scoring`` selects the per-candidate relevance mechanism
    # ``ThreeChannelRecallRanker`` applies BEFORE fusing/protecting the STM channel:
    #   * "embed" (DEFAULT) — cosine-rank STM candidate content against the query vector using the
    #     SAME ``EmbeddingPort`` (MiniLM) the MTM channel is already embedded with (embedded once
    #     at the ``RecallService`` façade boundary, §6-P2/m4) — cheapest-correct, no separate model.
    #   * "lexical" — token-overlap score against the raw query text; minimum-viable fallback that
    #     needs no embedder wired (e.g. an embedder-less composition root).
    #   * "recency" — explicit opt-out: PRE-fix behavior, list order stays the adapter's recency
    #     order with no relevance signal (A/B comparison / rollback, DEV-STANDARDS rule 3).
    # Env override: ``MU_RECALL__STM_SCORING=lexical`` (or ``recency``). Selecting "embed" with no
    # embedder injected into the ranker is a FAIL-LOUD misconfiguration (``StmScoringConfigError``),
    # never a silent recency fallback (§5 "re-raise loud, not a silent partial").
    stm_scoring: Literal["embed", "lexical", "recency"] = "embed"

    # D-4 multi-hop LTM traversal arm (ARCHITECTURE-CONFORMANCE.md "LTM graph arm thin";
    # CONFIG-AND-DATA-FIX-PLAN.md PART 2 D6): bounds how many entity-edge hops
    # ``ThreeChannelRecallRanker``'s LTM channel walks (``GraphStorePort.traverse_entities``) to
    # answer a relational query ("who is Bo's manager?") that the flat ``graph_recall`` seed
    # (whole-partition, currently-valid facts — see this module's docstring) cannot: a fact whose
    # SUBJECT differs from the query's own entity mention never surfaces there, only via a
    # traversal FROM the query's mentioned entity over the entity-entity edges
    # ``FalkorLtmAdapter.upsert_fact`` now materializes (B5/B6). ``0`` disables the arm entirely
    # (flat ``graph_recall`` only, pre-D6 behavior — A/B comparison / rollback, DEV-STANDARDS
    # rule 3); the adapter itself further clamps any value > 2 down to 2 (deeper hops risk
    # combinatorial blowup on a shared box, out of this task's scope). Env override:
    # ``MU_RECALL__LTM_MAX_HOPS=0`` (or any int).
    ltm_max_hops: int = Field(default=2, ge=0)

    # Rerank gate (ACCURACY-PLAN-0831.md item 6 / recall-service-design.md §1.5, ADR 0010/0023):
    # `ModelRouter.rerank` (`providers/model_router.py:265`) was fully built — a local
    # `BAAI/bge-reranker-v2-m3` configured, a `Task.RERANK` route registered — and had NO caller
    # anywhere in `services/recall/` until `rerank_gate.py` landed alongside these three fields;
    # `RecallItemView.rerank_score` stayed permanently `None`. These are the SAME three knobs the
    # design doc's `RetrievalWeights` names (`recall-service-design.md:576-578`) and the SAME
    # values ADR 0023 landed on after measurement ("the default configuration is cross_encoder /
    # min_score=0.5 / top_fraction=0.5 / pool_size=20 (ADR 0023 final combined)") — not
    # re-guessed here, carried forward from the decision that already measured them.
    #
    # `rerank_enabled` is the env-overridable A/B escape hatch this file's sibling knobs already
    # use (mirrors `cross_tier_dedup`/`ltm_max_hops=0`): `False` makes `AdaptiveRerankGate` dark
    # regardless of whether a real reranker was injected at the composition root, so a single env
    # var (`MU_RECALL__RERANK_ENABLED=false`) reverts to pre-rerank behavior for comparison
    # without a redeploy.
    #
    # DEFAULT `False`, and it must stay `False` until something actually serves the rerank group.
    # MEASURED 2026-09-01, on real stores, with the gate enabled and both shipped roots injecting
    # `reranker=self.model_router` as they do today:
    #     ModelRouter.rerank -> ModelGroupUnavailableError ("all deployments exhausted"), EVERY call
    #     items carrying a rerank_score: 0 of 30
    #     recall latency: 105 ms -> 4800 ms median, a 46x regression for zero change in results
    # against an SLO of p95 <= 150 ms (observability-design.md:150). The cause is structural, not a
    # VM accident: the rerank group's only deployment is the local endpoint on :8080
    # (`shipped_settings.py:88-90`) and NO compose file in this repo provisions a rerank service, so
    # the group is unavailable on every dev box, every CI runner, every eval run and every
    # FULL-LOCAL install. A default of `True` therefore bought nothing anywhere and cost 4.7 s per
    # recall everywhere. Flip this back the moment a reranker is actually deployed, and re-measure
    # the SLO in the same pass.
    rerank_enabled: bool = Field(default=False)

    # HYBRID MTM — dense ⊕ sparse inside the MTM channel (mtm-retrieval-design.md §1.2/§1.3,
    # `HybridConfig`). Fusion tuning belongs under `RecallSettings`, never `ModelSettings`
    # (§1.2: "IDF needs no model") — the BM25 producer downloads nothing and adds no LLM call to
    # the read path.
    #
    # DARK BY DEFAULT, exactly like `rerank_enabled` above and for the same reason: with
    # `sparse_enabled=False` the composition root wires NO encoder into the MTM adapter, so the
    # collection shape, the write path and the read path are all byte-identical to the dense-only
    # engine every existing test and every committed measurement was taken against. Turning it on
    # is the A/B arm (`MU_RECALL__SPARSE_ENABLED=true`).
    #
    # WHY IT EXISTS AT ALL — the measurement, not a hunch. On the full 1,531-query LoCoMo corpus
    # at `mu-core@0261c6a` (`docs/tracking/K10-VS-K30-RECONCILED-0903.md` §4), 74.7% of wrong
    # answers at k=10 and 59.6% at k=30 never had the gold turn in context: the ceiling is
    # first-stage retrieval, not synthesis. And that first stage is dense-alone — the shipped
    # 3-channel fuse tracks its own MTM dense channel to within 0.5% relative at every cutoff
    # from k=3 to k=60 (`RETRIEVAL-EVAL-0829.md` §13.1), with the LTM graph empty and the STM
    # floor discounted to 0.1. A MiniLM bi-encoder over short conversational turns is weakest on
    # exactly the rare proper nouns, dates and numbers LoCoMo questions turn on, which is the
    # textbook sparse-retrieval strength.
    sparse_enabled: bool = Field(default=False)
    # Registry key carried into `SparseQuery.encoder` as provenance (§1.5:
    # "bm25" | "splade" | "none"). Only "bm25" is implemented in-repo; "splade" is the design's
    # named optional upgrade and would be a new provider, not a new branch here.
    sparse_encoder: str = Field(default="bm25")
    # BM25 knobs — the textbook Robertson/Sparck-Jones values, and the length-normalisation
    # reference FastEmbed's own `Qdrant/bm25` ships. Named and overridable rather than literals
    # buried in the encoder (DEV-STANDARDS rule 3); see `providers/sparse_encoder.py` for why
    # `avg_len` is a constant and not a real corpus average.
    sparse_bm25_k1: float = Field(default=1.2, ge=0.0)
    sparse_bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    sparse_bm25_avg_len: float = Field(default=256.0, gt=0.0)
    sparse_min_token_len: int = Field(default=2, ge=1)
    # ADR 0023 final combined value (recall-service-design.md:576, `rerank.py`'s own
    # `adaptive_rerank_gate` floor rule): the top-scored candidate in the pool must clear this
    # before ANY candidate in the pool is trusted — below it, the whole gate is empty and the
    # caller falls back to the pre-rerank order (HippoRAG-style, `rerank_gate.py`'s own docstring).
    rerank_min_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # ADR 0023 Decision 2 (recall-service-design.md:264): "top_fraction is nondecreasing in the
    # cutoff -> 0.5 is the LOOSER, safer-for-recall setting, not the aggressive one" — a candidate
    # survives if its score is within this fraction of the pool's own top score
    # (`cutoff = max(min_score, top_score * top_fraction)`, `adaptive_rerank_gate`'s own algebra).
    rerank_top_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    # ADR 0010's mem0-defect fix (recall-service-design.md:202): the rerank gate must see a pool
    # WIDER than the final `limit` or there is nothing left for it to prune. Independent of
    # `channel_pool_size`/`channel_pool_multiplier` (the per-CHANNEL fetch width,
    # ACCURACY-PLAN-0831.md §1.4) — this bounds how many of the already-fused, best-first
    # candidates are sent to the cross-encoder in ONE batched forward pass, priced against the
    # `recall_e2e_rerank` p95 budget (`language-analysis-server.md`: <=20 pairs ~15-40ms).
    rerank_pool_size: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def _validate_derived_width_range(self) -> RecallSettings:
        """Fail loud at CONSTRUCTION, not at the first `recall()` call: an inverted
        ``min_derived_limit > max_derived_limit`` is a misconfiguration
        (:func:`~mu_engine.services.recall.width.derive_recall_limit` raises the identical
        ``ValueError`` at call time as a second, defence-in-depth check — this is the earlier,
        friendlier one, since a settings object is typically built once at composition root)."""
        if self.min_derived_limit > self.max_derived_limit:
            raise ValueError(
                "min_derived_limit "
                f"({self.min_derived_limit}) must be <= max_derived_limit "
                f"({self.max_derived_limit})"
            )
        return self
