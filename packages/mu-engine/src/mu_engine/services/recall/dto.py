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
    recency_floor_limit: int = Field(default=10, ge=0)  # STM floor width (§1.3)
    channel_pool_size: int = Field(default=20, ge=1)  # per-channel fetch width > limit (ADR 0010)
    weight_mtm: float = Field(default=1.0, ge=0.0)  # in-arm dense weight (§1.3 fuse)
    weight_ltm: float = Field(default=1.0, ge=0.0)  # in-arm graph weight (§1.3 fuse)
    weight_private: float = Field(default=1.0, ge=0.0)  # federation: private-arm weight (§1.6)
    weight_shared: float = Field(default=1.0, ge=0.0)  # federation: shared-arm weight (§1.6)
