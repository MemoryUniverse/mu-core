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
    # S1b (TRACE-0923.md §7/§6.2/§5.1, ADR pending): the SAME conversational-order key as
    # `MemoryItem.turn_seq` — `None` for any hit whose underlying item was written before S1b or
    # by a write path that assigns none (this field, like that one, degrades gracefully rather
    # than assuming `0`). Provenance/debugging only — `ranker.py::_expand_neighbors` reads it to
    # find an anchor's neighbours; nothing downstream is REQUIRED to consult it.
    turn_seq: int | None = None
    # True on a `RecallItemView` this ranker added via neighbour expansion rather than because
    # any channel ranked it — provenance only, never consulted by `_merge_floor`'s own logic
    # (a neighbour competes for its `limit` slot exactly like any other candidate once inserted,
    # `_expand_neighbors`'s own docstring: "a neighbour COSTS a slot").
    is_neighbor: bool = False


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
    # AD-250 fix (ADR 0061): the demoted-item channel's OWN pool width — a SEPARATE knob from
    # `recency_floor_limit` above, because the two channels read SEPARATE indices
    # (`StmTierRepository.demoted` vs `recent`, `put_demoted`'s docstring has the full
    # mechanism). A demoted write-ahead copy no longer competes with fresh captures for the
    # ordinary floor's slots, so this bounds how many of THIS namespace's currently-demoted rows
    # (a much smaller, more slowly-growing population than every STM write ever) are eligible to
    # compete in the RRF fuse per query. Defaulted to the SAME width as `recency_floor_limit` —
    # no measurement has yet compared a wider or narrower demoted pool against it; widen this if
    # a namespace routinely holds more than 10 live demoted items and a real query needs to reach
    # past the newest 10 of them. Env override: `MU_RECALL__DEMOTED_FLOOR_LIMIT`.
    demoted_floor_limit: int = Field(default=10, ge=0)
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
    # T1 option (c) (TRACE-0923.md §7/§7.1, ADR pending): `floor_protect_limit` above still bounds
    # WHICH recency-ranked candidates are even ELIGIBLE for protection — this field adds a SECOND,
    # independent gate: an eligible candidate is only actually protected (never-evicted) when its
    # own relevance score (`ranker.py::_score_stm`'s per-candidate score, the SAME one that already
    # reorders the floor and feeds the RRF channel) is >= this bar. See `ranker.py::
    # _protected_floor_ids` for the full rationale, what the AD-195 guarantee now promises, and why
    # `stm_scoring="recency"` makes this a uniform on/off rather than a per-candidate filter.
    # Measured cost this narrows (§7, pooled over 382 LoCoMo queries, real stores): the
    # UNCONDITIONAL guarantee (this field at its default) costs 6.28 pt of `gold_in_context` (9.59
    # on multi-hop) for 24 fixed / 0 regressed — i.e. it is a real, priced trade-off, not free; the
    # whole point of a conditional gate is to keep most of that 24-fixed benefit on a query the
    # protected fact is actually relevant to, while stopping paying the 6.28 pt cost on the queries
    # it is not.
    #
    # DEFAULT SHIPPED AT `0.5`, on evidence (T1 ADR, `docs/tracking/ARCHITECTURE-DELTAS.md`
    # AD-230): a real `gold_in_context` sweep on `mu-dev-vm` (3 LoCoMo conversations, 383 queries,
    # `stm_scoring="embed"`, the shipped default) —
    #     bar        gold_in_context   vs unconditional (257/383, 0.6710)
    #     0.3        272/383  0.7102   +15 queries
    #     0.5        281/383  0.7337   +24 queries  <- SHIPPED
    #     0.7        282/383  0.7363   +25 queries  (within noise of 0.5 — one run, no repeats)
    #     disabled   281/383  0.7337   (floor_protect_min_relevance=1.1, i.e. no candidate ever
    #                                  clears the bar — the "(b) lower floor_protect_limit to 0"
    #                                  option §7.1 named and the owner explicitly did NOT want)
    # `0.5` recovers ESSENTIALLY ALL of the measured gain from disabling the guarantee outright,
    # while — unlike full disable — still protecting a just-said fact whenever it clears a real
    # relevance bar, which is what keeps AD-195's live-agent-session intent alive (§7.1: "probably
    # worth keeping ... LoCoMo is the adversarial case for it, not the typical one"). One run per
    # arm, no repeats (`gic_hybrid.json`'s own measured 3-repeat spread for THIS metric shape was
    # 0.0067 on a similar sample, so 0.5 vs 0.7's 1-query difference is not distinguishable from
    # that noise floor — 0.5 is picked as the more conservative of the two statistically
    # indistinguishable options, not because 0.7 was ruled out).
    #
    # `-1.0` (cosine similarity's own true theoretical minimum — below it no real score, under any
    # `stm_scoring` mode this repo ships, can ever fall) is the value that reproduces AD-195's
    # ORIGINAL, fully unconditional guarantee — set `MU_RECALL__FLOOR_PROTECT_MIN_RELEVANCE=-1` to
    # get it back exactly. **This was not the first default this field shipped with.** `0.0`
    # looked equally inert on the reasoning "a real relevance score is virtually always
    # non-negative" and was WRONG — caught live by `test_persona_composition_int.py::
    # test_a_real_persona_reorders_a_real_recall_and_changes_nothing_else` going red on
    # `mu-dev-vm` against real stores + the real MiniLM embedder: `stm_scoring="embed"` scores by
    # COSINE SIMILARITY (`ranker.py::_cosine`), whose real range is `[-1.0, 1.0]`, not `[0.0,
    # 1.0]` — a floor candidate genuinely orthogonal-to-hostile to the query scores negative, and
    # a `0.0` bar silently un-protected it even though the field's OWN claim at the time was "this
    # default is a no-op". Recorded rather than quietly fixed, because it is the concrete proof
    # that "looks inert" and "is inert" are different claims for a bound whose true range was not
    # checked carefully enough the first time. Env override:
    # `MU_RECALL__FLOOR_PROTECT_MIN_RELEVANCE`.
    floor_protect_min_relevance: float = Field(default=0.5, ge=-1.0)
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
    # ADR 0060 (`docs/tracking/eval-runs/2026-09-24-ltm-channel-zero-slots.md`): "why does the
    # LTM channel win zero slots even when it is populated" (the open question the 2026-08-31
    # `weight_ltm` fix left, and ADR 0058's own verify pass re-raised) turned out to be a
    # STRUCTURAL exclusion, not a discount. `reciprocal_rank_fusion`'s per-channel contribution is
    # `weight/(k+rank+1)` — at shipped `weight_ltm=0.1`/`weight_mtm=1.0`/`rrf_k=60`, the LTM
    # channel's best POSSIBLE contribution (its own rank 0: `0.1/61=0.00164`) is always smaller
    # than the MTM channel's WORST-pool-item contribution (rank `pool-1`: `1.0/(60+pool)`, e.g.
    # `0.0125` at `pool=20`) for every `channel_pool_size`/`channel_pool_multiplier` this repo
    # ships — PROVED exhaustively in `tests/services/test_recall_ranker_unit.py`'s
    # `test_ltm_protect_limit_*` family (a pure-math bisection over the shipped
    # `reciprocal_rank_fusion`) and reproduced end to end: 960 real distilled facts in FalkorDB,
    # 383 real LoCoMo queries, **0** LTM items in **any** fused result (`by_channel` byte-identical
    # to the LTM-empty control).
    #
    # This field is the MECHANISM that fixes the structural exclusion — it reuses the STM floor's
    # own "protect membership, RESCUE AT THE TAIL if fusion ranked the member outside the window"
    # shape (`_merge_floor`, unchanged) for the graph tier's own top `ltm_protect_limit`
    # candidates, so presence no longer requires winning RRF outright. A rescued member is
    # APPENDED after the naturally-fused head, never re-ordered ahead of a genuinely relevant hit
    # — `test_default_settings_rank_the_relevant_mtm_hit_ahead_of_query_blind_ltm_noise` (the
    # regression test the 2026-08-31 fix shipped) keeps passing UNCHANGED at any `ltm_protect_
    # limit` value: a query-blind LTM candidate still never OUTRANKS a relevant MTM hit.
    #
    # **But the mechanism WORKING is not the same question as the mechanism being WORTH shipping
    # on, and MEASURED it is not, yet — so the default stays `0`.** `docs/tracking/eval-runs/
    # 2026-09-24-ltm-channel-zero-slots.md`, matched-width, conv-26/30/41 (383 queries), 960
    # distilled facts:
    #     ltm_protect_limit=0 (structural exclusion, unchanged):
    #         ltm items=0,   gold_in_context=280/383 (0.7311)
    #     ltm_protect_limit=1 (mechanism ON):
    #         ltm items=383, gold_in_context=273/383 (0.7128)
    # The mechanism does exactly what it says — EXACTLY one graph item per query, every query,
    # confirming the rescue fires and is bounded, not probabilistic — and gold_in_context DROPS by
    # 7 queries (-1.83pt) for it. This is not "the channel contributes nothing," it is worse: **the
    # channel contributes NEGATIVE value the moment it is allowed to place at all**, at the SAME
    # root cause AD-204/the 2026-08-31 `weight_ltm` fix already diagnosed — `graph_recall`'s flat
    # seed is query-BLIND (whole-partition, recency-ordered, `subject=None`), so the one slot it
    # is guaranteed is, on average, a worse candidate than whatever query-relevant MTM/STM item it
    # bumped to make room. The earlier `weight_ltm=1.0`->`0.1` fix and this field's default both
    # answer the SAME question the same way: an unconditional, query-blind graph contribution is a
    # net cost proportional to how much of the result budget it spends (30% of slots: -25pt
    # full-corpus answer-quality; 1 of ~10 slots here: -1.83pt gold_in_context) — there is no slot
    # count tested at which this seed's unconditional presence is a net positive.
    #
    # Left at `0` — reproducing today's structural exclusion, now by EXPLICIT, documented,
    # measured choice rather than as an accidental side effect of a weight tuned for a different
    # purpose — until the graph arm's seed stops being query-blind (entity-resolved seeding behind
    # `resolve_entity`, already named as future work in `ranker.py`'s own module docstring: "a
    # bigger change, not attempted here"). The mechanism itself is shipped, tested and
    # mutation-checked so that future work has a bounded, presence-not-priority lever ready the
    # day the seed is worth trusting with even one guaranteed slot — raising this past `0` without
    # first fixing the seed would be motion, not progress, exactly the trap
    # `2026-09-24-verify-graph-tier-contribution.md`'s own closing line named. Env override:
    # `MU_RECALL__LTM_PROTECT_LIMIT`.
    ltm_protect_limit: int = Field(default=0, ge=0)

    # AD-273 (ADR 0066's own "try (a) first" instruction, 2026-09-24) — TRIED, MEASURED, and
    # SETTLED NEGATIVE. This was the one alternative to a protect floor that could still let the
    # graph tier win a slot ON MERIT rather than by guaranteed presence: `weight_ltm`'s docstring
    # above proves the structural exclusion is a SHARED `rrf_k` problem, not (only) a seed-quality
    # problem — `weight/(k+rank+1)` caps a heavily-discounted channel's best-possible (rank 0)
    # contribution below a heavily-weighted channel's worst-pool-item contribution, no matter how
    # relevant that top candidate is. This field is a SEPARATE, smaller RRF constant for the LTM
    # channel only (`fusion.py::reciprocal_rank_fusion`'s `ks` parameter), tried at
    # `ltm_protect_limit=0` — no forced presence, the channel's top candidate simply gets a fair
    # shot at the SAME rank-0 score scale every other channel's top candidate gets, and has to WIN
    # its slot by out-scoring a real MTM/STM candidate.
    #
    # MEASURED (matched-width by construction — every arm returns exactly 3,830 items over the
    # same conv-26/30/41, 383 queries, `ltm_protect_limit=0` throughout — `docs/tracking/
    # eval-runs/2026-09-24-ad273-ltm-fusion-arithmetic.md`):
    #     k_ltm=None (shipped, shared k=60):        ltm=0,   gold_in_context=279/383 (0.7285)
    #     k_ltm=10:                                 ltm=0,   gold_in_context=280/383 (0.7311)
    #     k_ltm=7:                                  ltm=0,   gold_in_context=279/383 (0.7285)
    #     k_ltm=6:                                  ltm=304, gold_in_context=272/383 (0.7102)  -7
    #     k_ltm=5, flat seed (seed_pool=0):         ltm=383, gold_in_context=275/383 (0.7180)  -4
    #     k_ltm=5, content-aware seed (seed_pool=5): ltm=383, gold_in_context=272/383 (0.7102)  -7
    #
    # **The finding, and why it rules the lever out rather than merely under-tuning it.** The
    # transition from "wins nothing" to "wins everything" happens between `k_ltm=7` and `k_ltm=6`
    # — there is no smooth, query-dependent middle where the channel wins ONLY the queries its top
    # candidate is actually relevant to. The root cause is structural, not a tuning miss:
    # `_ltm_channel`'s flat `graph_recall` seed is UNCONDITIONAL — it returns a recency-ordered
    # candidate whenever the namespace holds ANY LTM fact at all, so the channel's rank-0 slot is
    # (almost) never genuinely empty. RRF fuses by RANK POSITION, not by the underlying candidate's
    # actual relevance, so a `k_ltm` small enough to make rank-0's score competitive makes it
    # EQUALLY competitive on every query, relevant or not — the exact "guaranteed one slot
    # regardless of quality" shape `ltm_protect_limit` already produces, just reached through
    # score inflation instead of explicit reservation, and it costs the SAME 4-7 queries either
    # way (compare this table to `ltm_protect_limit`'s own: -6/-7 queries at "forced, one slot").
    # A `k_ltm` weak enough to avoid always-winning (>=7) is, measured, indistinguishable from
    # `None` — it structurally can never place, for the same proof `weight_ltm`'s docstring gives.
    # **There is no value of this field that both wins slots and helps `gold_in_context`.**
    #
    # This settles ADR 0066's open question: the graph tier's problem is not the fusion arithmetic
    # (this field), not the protect floor (`ltm_protect_limit`, tried first), and not the seed
    # (content-aware vs flat — the SAME two points, within noise of each other, at every k tried).
    # It is that `graph_recall`'s unconditional flat seed makes the channel's "does it have a
    # candidate at all" carry no relevance signal, so nothing downstream of it — not weight, not
    # per-channel k, not a floor — can express "only when genuinely relevant" without the seed
    # itself learning to return NOTHING on a query it has no good answer to. `ARCHITECTURE-
    # DELTAS.md` AD-273/AD-274 has the full writeup and the recommendation left to the owner.
    #
    # Left at `None` (the shared `rrf_k`, i.e. the structural exclusion unchanged) — every measured
    # non-default value is a net cost. Env override: `MU_RECALL__RRF_K_LTM` (unset/absent = `None`).
    rrf_k_ltm: int | None = Field(default=None, ge=1)
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

    # AD-250 fix (ADR 0061, `docs/tracking/FAULT-HUNT-0924.md` §1 F1's still-open half): the
    # read-stat write-back `recall-service-design.md` §5.1/line 609 has always claimed happens
    # ("the only mutation [recall] can cause is the read-stat write-back the stores already do
    # idempotently on read") but, until this fix, no `StmTierRepository` adapter actually
    # implemented for any item — `access_count` never rose from a genuine recall, so the ADR 0034
    # rescue (`DemotionService`'s own docstring: "a re-recalled item's raised access_count
    # rescues it") had no real trigger even for an item the new demoted channel could see.
    # `ThreeChannelRecallRanker.rank` now calls `StmTierRepository.reinforce` once per DISTINCT
    # STM-channel item (ordinary floor OR demoted) that survives into the returned
    # `RecallResult` — bumping `access_count` (feeds `SalienceStrategy._usage`, the ONLY upward
    # term the forgetting-curve gates ever see). Default ON — this closes a real defect, not an
    # optional enhancement. Env override: `MU_RECALL__REINFORCE_ON_RECALL=false` for a
    # read-replica/latency-sensitive deployment that wants recall to stay a pure read (the
    # pre-fix behaviour for the ORDINARY floor: never wrong, merely un-rescuable — but for the
    # DEMOTED channel this also means the rescue trigger this whole fix exists for goes dark, so
    # turning it off is a real, named tradeoff, not a free knob).
    reinforce_on_recall: bool = Field(default=True)

    # S1b — read-time neighbour expansion (TRACE-0923.md §7/§6.2/§5.1, ADR pending in
    # `docs/decisions/`). §5.1's own finding: 32.5% of failures return a turn within ±1 of the
    # gold and never the gold (48.4% within ±2), because the answer-bearing turn is usually a
    # REPLY and the question's vocabulary lives in the turn BEFORE it — the retriever finds the
    # right conversational moment and returns the wrong turn of it. `ranker.py::_expand_neighbors`
    # inserts each fused candidate's ±`neighbor_expand_radius` conversational neighbours (by
    # `MemoryItem.turn_seq`, S1b's write-side field — `local_memory.py::_next_turn_seq_base`)
    # into the SAME fused pool `_merge_floor` truncates to `limit`, so a neighbour COMPETES for a
    # result slot exactly like any other candidate (the "does a neighbour cost a slot or ride
    # along with its parent" decision this ADR records: THIS phase chose "costs a slot" — the
    # measurable-today choice, since a free-riding neighbour would need a new field on the
    # WIRE-versioned `mu_contracts` `RecallItemView`, out of this phase's lane; see the ADR).
    # Default 0 = OFF — every existing caller/test is byte-identical until an operator raises it;
    # ±1 is the width §5.1 measured as the single largest gain (41 of 382 queries), ±2 the next
    # (61 of 382) — SET FROM EVIDENCE once the paired gold_in_context sweep is recorded. Env
    # override: `MU_RECALL__NEIGHBOR_EXPAND_RADIUS`.
    neighbor_expand_radius: int = Field(default=0, ge=0)
    # How far back the read-time expansion's OWN STM session-window fetch scans to build the
    # `turn_seq -> item` lookup — mirrors `local_memory.py::_TURN_SEQ_SCAN_LIMIT`'s own documented
    # basis (conv-41, the longest LoCoMo conversation this repo's corpus carries, is 663 turns;
    # set generously above that, NOT tuned to the benchmark) and the SAME documented limit: a
    # session whose STM window exceeds this width will not have every neighbour resolvable, which
    # degrades to "no expansion for that anchor" (graceful, never a crash — see
    # `_expand_neighbors`'s own docstring), not a wrong answer.
    neighbor_expand_session_scan_limit: int = Field(default=4000, ge=1)
    # WHERE an inserted neighbour is placed in the ranked pool. VERIFY PASS 2026-09-23 (AD-232):
    # `"tail"` is the SHIPPED policy S1b was first measured with — a neighbour is scored on the STM
    # channel's own `weight_stm`-discounted family and therefore lands strictly below every real
    # candidate, i.e. at the END of the pool `_merge_floor` then truncates to `limit`. On a corpus
    # where the pool is already `limit`-deep (this eval harness ingests at `importance=0.9`, so
    # nearly every turn is promoted to MTM) that makes the whole mechanism a measured no-op, which
    # is exactly what AD-231 recorded. `"after_anchor"` places each neighbour immediately BEHIND
    # the candidate that surfaced it, so a neighbour of a top-ranked anchor is inside the window
    # whenever its anchor is — still costing a slot (the anchor's tail neighbours push the pool's
    # own tail out), still never re-sorting the pool (so `AdaptiveRerankGate`'s verdict survives),
    # but no longer guaranteed to be truncated away before it can be judged.
    # The ceiling this is aimed at, recomputed independently from
    # `docs/tracking/eval-runs/2026-09-23/trace_*.json` + the raw corpus: of 126 pooled read-path
    # failures over conv-26/30/41, **41 (32.5%) returned a turn within ±1 of the gold and never the
    # gold** (61, 48.4%, within ±2) — i.e. a working ±1 expansion is worth up to +10.7 pt of
    # `gold_in_context` on this sample. Default stays `"tail"` so every number already recorded
    # against S1b keeps its meaning; `MU_RECALL__NEIGHBOR_EXPAND_PLACEMENT=after_anchor`.
    neighbor_expand_placement: Literal["tail", "after_anchor"] = "tail"

    # S1b — TWO FURTHER SHAPES (TRACE-0923.md follow-up, ADR 0053 "what would need to change
    # before this ships ON" §1/§2; ARCHITECTURE-DELTAS.md AD-233's own amendment: "the two live
    # options are the free-riding insertion ... or a wider fetch narrowed after expansion. Nothing
    # else is left"). Both are OFF by default and, unlike `neighbor_expand_placement`, are not
    # alternate values of one knob — a deployment can run neither, either, or (nonsensically) both;
    # `rank()` checks `neighbor_free_ride` first and short-circuits the "costs a slot" mechanism
    # entirely when it is set, so the two never actually compete for the same neighbour.
    #
    # **`neighbor_free_ride`** — a neighbour rides along WITH the anchor that earned its slot
    # instead of consuming one of its own. `_expand_neighbors`'s ordinary "costs a slot" insertion
    # (AD-231/AD-232's own mechanism, above) is SKIPPED entirely when this is `True` — inserting a
    # neighbour into the SAME RRF-competed pool `_merge_floor` truncates was the thing AD-233's own
    # amendment names as unable to pay ("only ~1 in 3 neighbours is gold"), so free-riding runs a
    # SEPARATE step, `_attach_free_riding_neighbors`, strictly AFTER `_merge_floor` has already
    # picked the `limit` winners: for every surviving item carrying a `turn_seq`, its real
    # `±neighbor_expand_radius` neighbours (the SAME session-window lookup and STM-family score as
    # the costs-a-slot mechanism, `weight_stm / (rrf_k + floor_pool_size + offset)` — never derived
    # from the anchor's own score, same AD-231 regression guard) are APPENDED to the returned list,
    # never displacing anything already selected. This means `RecallResult.items` can carry MORE
    # than `limit` entries whenever this is on and at least one surviving item has a real
    # neighbour — the deliberate trade the ADR's own "what would need to change" section names:
    # "the downside is bounded to prompt length" rather than to precision, since nothing already
    # ranked in ever loses its slot to make room for one. A caller that renders a strict `limit`
    # window (e.g. `build_context`'s token budget) is expected to either accept the extra
    # provenance-only context or filter `is_neighbor` rows itself — this field does not change
    # what `limit` MEANS, only whether extras may ride past it.
    # Env override: `MU_RECALL__NEIGHBOR_FREE_RIDE`.
    neighbor_free_ride: bool = False
    # **`neighbor_expand_widen`** — "fetch wider, narrow after expansion": how many EXTRA
    # candidates beyond the caller's real `limit` this shape fetches/considers before cutting back
    # down to `limit`, AND (the same number — one knob, two effects that share a rationale) how
    # many of the neighbours that survive that wider look are then GUARANTEED a final slot. `0`
    # (the default) is byte-identical to the shipped "costs a slot at the real limit" behaviour.
    # At `>0`:
    #   1. `_merge_floor` runs at `limit + neighbor_expand_widen` — giving an inserted neighbour,
    #      and a protected-floor RESCUE, more room to survive the FIRST truncation and the
    #      cross-tier dedup pass inside it, instead of being squeezed out before either ever
    #      considers it.
    #   2. `_narrow_after_expansion` then cuts back to `limit` in two tiers: every `is_floor`
    #      member survives unconditionally (never breaks AD-195/ADR 0052's own guarantee — a
    #      blind `items[:limit]` re-slice of the widened pool could otherwise push a RESCUED
    #      protected item, which `_merge_floor` deliberately appends past the naturally-ranked
    #      head, back out past the real limit); THEN up to `neighbor_expand_widen` of the
    #      neighbours that made the widened cut are ALSO guaranteed a slot, in their
    #      already-established (best-scored-first) order — displacing the WEAKEST ordinary
    #      (non-floor, non-neighbour) candidates if there is not already room. **This is the
    #      priced trade that makes Shape B a distinct mechanism from Shape A
    #      (`neighbor_free_ride`, above), not a second copy of it**: free-riding NEVER displaces
    #      anything (bounded to prompt length); this shape DOES displace up to
    #      `neighbor_expand_widen` of the weakest real candidates to let a neighbour compete for a
    #      genuine slot, capped so the cost stays bounded rather than unlimited.
    # Ignored when `neighbor_expand_radius == 0` (nothing to widen for) or when
    # `neighbor_free_ride` is set (the two mechanisms are alternatives, not composable — free-
    # riding already never costs a slot, so there is nothing for a rescue budget to buy it).
    # Env override: `MU_RECALL__NEIGHBOR_EXPAND_WIDEN`.
    neighbor_expand_widen: int = Field(default=0, ge=0)

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

    # AD-258 fix (content-aware seed, `docs/tracking/FAULT-HUNT-0924.md`; ADR pending): how many
    # of the query's OWN top-ranked MTM dense-vector hits contribute their resolved
    # ``entity_uids`` (D-5 payload backfill, now round-tripping through ``QdrantMapper.from_store``
    # — see that fix's own docstring) into ``traverse_entities``'s hop-1 frontier, ALONGSIDE the
    # pre-existing casefolded-token match (never replacing it). This is the fix ADR 0060's own
    # closing line named as the only lever not yet tried: the graph tier's flat/token seeds are
    # both query-blind or exact-lexical, and every measured attempt to give the channel a
    # guaranteed slot on TOP of that seed made `gold_in_context` worse, never better (ADR 0060,
    # `2026-09-24-ltm-channel-zero-slots.md`). ``ThreeChannelRecallRanker._ltm_channel`` awaits
    # the SAME MTM semantic-search task the MTM channel itself runs (no second embedding call,
    # no second store round trip beyond the traversal it already makes) and harvests
    # ``item.metadata['entity_uids']`` off its top-``ltm_entity_seed_pool`` hits — the entities
    # the MTM channel's own fact-embedding (``"{subject} {predicate} {object}"``,
    # ``pipelines/concrete/ingest.py``'s docstring) already resolved as semantically relevant to
    # the query, whether or not the query names them verbatim. ``0`` disables the seed entirely
    # (pre-fix token-only behavior — A/B comparison, DEV-STANDARDS rule 3).
    #
    # DEFAULT ``0``, same shipped-cautious posture as ``ltm_protect_limit`` above and for the same
    # reason: at ``ltm_protect_limit=0`` the RRF structural exclusion means a better seed changes
    # NOTHING about what a caller gets back (the channel still never wins a fused slot) while
    # still paying a real cost every recall — ``_ltm_channel`` now has to await the MTM channel's
    # own semantic-search task before it can run ``traverse_entities``, serializing work that used
    # to run fully concurrently with MTM. Shipping a non-zero default here would be an unmeasured
    # latency regression for a mechanism inert at the shipped ``ltm_protect_limit``. Raise BOTH
    # together, deliberately, once measured — never this one alone. Env override:
    # ``MU_RECALL__LTM_ENTITY_SEED_POOL``.
    ltm_entity_seed_pool: int = Field(default=0, ge=0)

    # AD-279 — the SELECTIVE seed, and the one lever four prior passes could not pull.
    #
    # `graph_recall` (`falkor_ltm.py::_graph_recall_impl`) is, read literally, `MATCH (m:Memory)
    # WHERE <namespace> AND m.state=active AND still-valid RETURN … ORDER BY m.valid_at DESC
    # LIMIT $limit` with `score = 1/(rank+1)`. There is NO query term in that Cypher at all: the
    # "seed" is the N most RECENTLY-VALID facts in the partition, and it returns them whenever the
    # partition holds any fact — which, on any populated namespace, is every query. So the LTM
    # channel is never empty and its rank-0 slot carries ZERO relevance signal.
    #
    # That is why four independent levers all failed the same way (ADR 0072 has the table):
    # `weight_ltm` (2026-08-31), `ltm_protect_limit` (ADR 0060/0066), the content-aware traversal
    # seed (ADR 0065/AD-262) and per-channel `rrf_k_ltm` (ADR 0069/AD-273) are all DOWNSTREAM of
    # the seed, and RRF fuses by RANK POSITION, never by content — so any lever strong enough to
    # make LTM's rank-0 place at all makes it place on EVERY query, relevant or not, reproducing
    # the identical 4-to-7-query `gold_in_context` cost each time. No arithmetic can make a
    # placement conditional on relevance when the candidate handed to it is unconditional.
    #
    # `False` skips the flat seed entirely, leaving `ThreeChannelRecallRanker._ltm_channel` with
    # only `traverse_entities` — which IS query-conditional: it returns `[]` when no entity in the
    # query (or in the `ltm_entity_seed_pool` MTM seed) matches the entity sub-graph at all
    # (`falkor_ltm.py`'s `if not memory_hop: return []`). The channel then goes SILENT on queries
    # it has no entity-grounded answer to, which is the property every ranking lever was missing.
    # Pair it with `rrf_k_ltm` (the rank authority those silent-when-irrelevant hits need in order
    # to place when they DO fire) — alone it changes nothing, for exactly the structural reason
    # `weight_ltm`'s own docstring proves.
    #
    # DEFAULT `True` — byte-identical to every prior release; this is an A/B lever
    # (DEV-STANDARDS rule 3), not a behaviour change. `ltm_flat_seed=False` together with
    # `ltm_max_hops=0` disables the LTM recall channel outright (both arms off) — a legitimate
    # way to express option (b) of `docs/tracking/GRAPH-TIER-DECISION.md`, and the reason neither
    # value is validated against the other. Env override: `MU_RECALL__LTM_FLAT_SEED=false`.
    ltm_flat_seed: bool = True

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
    #
    # TURNED ON 2026-09-23, on the owner's authorisation, because the A/B arm was measured and it
    # is the single largest retrieval gain this project has recorded:
    #     gold-in-context, conv-26, 149 queries, 3 repeats per arm, ZERO LLM calls
    #       dense only   51.01%   spread 0.0000  (all 149 per-query verdicts identical, 3 reps)
    #       hybrid       67.34%   spread 0.0067
    #     +16.33 pt absolute, +32% relative, ~24x the larger arm's spread
    #     paired on identical queries: 31 fixed, 7 regressed, McNemar p = 1.16e-4
    # Corroborated independently on the full corpus, retrieval-only and therefore free:
    # recall@10 0.4546 -> 0.6138. The dense-only default was the ceiling described above, so
    # leaving it dark would have meant shipping the weaker arm on purpose.
    sparse_enabled: bool = Field(default=True)
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
    # S4 (TRACE-0923.md §5.4/§7): the caller-supplied document text for a dialogue turn carries a
    # leading "Speaker: " label (the eval harness's own `Turn.ingest_text`, and this is the SAME
    # shape MemOS's own LoCoMo ingestion builds, `locomo_ingestion.py:39` — a convention this
    # engine does not control but must not be naive about). Measured on conv-26 (419 turns): the
    # label token `caroline` has document frequency 80.9%, and on the queries where BM25 fails,
    # the speaker name is the ONLY term the query shares with the gold turn at all — the sparse
    # arm is not mis-weighted there, it is empty-handed, weakly favoring every turn the named
    # speaker ever said over turns that actually share content vocabulary. Qdrant's server-side
    # `Modifier.IDF` already down-weights a high-df term correctly (§5.4: `caroline` IDF 0.213 <
    # `the` IDF 0.925), so this is not an IDF defect — it is that the label token is the only
    # match candidate on the failing rows, and it is not informative.
    # `sparse_strip_leading_prefix=True` strips ONE detected "Label: " prefix
    # (`_LEADING_LABEL_PREFIX_RE` in `providers/sparse_encoder.py`) from the text handed to
    # `Bm25SparseEncoder.encode` — the WRITE-side sparse document only. It does NOT touch
    # `MemoryItem.content` (still stored verbatim), the dense embedding text (unchanged — a
    # bi-encoder is not confused by a short label the way a sparse lexical match is; the trace's
    # own §7 S4 entry: "The dense arm may still want it, so treat the sparse document and the
    # dense document as separate concerns"), or `encode_query` (queries are not turn-labelled).
    # A document that tokenises to nothing once its label is stripped degrades to the SAME
    # dense-only-point path `_point_vector` already handles for an empty encode — never a write
    # failure. Default False pending the paired gold_in_context measurement this ADR's evidence
    # promised (`docs/decisions/` S4 ADR) — flip once recorded.
    sparse_strip_leading_prefix: bool = Field(default=False)
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
