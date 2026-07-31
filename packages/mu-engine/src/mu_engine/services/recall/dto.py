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

from pydantic import BaseModel, ConfigDict, Field

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
    limit: int = Field(default=10, ge=1)
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
    rerank_score: float | None = None  # None when the rerank gate is dark (deferred this phase)
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
    weight_stm: float = Field(default=1.0, ge=0.0)  # in-arm recency-channel weight (§1.3 fuse)
    weight_mtm: float = Field(default=1.0, ge=0.0)  # in-arm dense weight (§1.3 fuse)
    weight_ltm: float = Field(default=1.0, ge=0.0)  # in-arm graph weight (§1.3 fuse)
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
