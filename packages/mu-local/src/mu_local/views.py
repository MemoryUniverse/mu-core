"""Public result DTOs for the :class:`~mu_local.local_memory.LocalMemory` facade.

Frozen pydantic-v2 value objects (DEV-STANDARDS rule 2) built per call, never shared across tasks.
They are the IN-PROCESS return payloads a LOCAL caller gets back — carrying the body is the whole
point of a read, so ``content`` is present (the content-free discipline governs buses/logs/metrics,
not these return values; CANONICAL §3.1). Each maps 1:1 from an engine result
(``IngestResult`` / ``RecallResult`` / ``DistillReport``) so the facade adds no second behaviour.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ConsolidateView",
    "ContextView",
    "MemoryListView",
    "MemoryRecordView",
    "MemoryWriteResult",
]


class MemoryWriteResult(BaseModel):
    """The fast-return receipt of one ``add`` (maps from ``IngestResult`` — ids/flags only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    content_hash: str
    promoted: bool
    tiers_written: tuple[str, ...]


class MemoryRecordView(BaseModel):
    """One recalled hit (maps from ``RecallItemView``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    content: str
    tier: str  # MemoryTier value — "stm" | "mtm" | "ltm"
    channel: str  # provenance of the hit
    score: float
    is_floor: bool = False  # STM recency-floor member — reorderable, never evicted


class MemoryListView(BaseModel):
    """A ranked list result (from ``RecallResult``); ``degraded`` is the NAMED degrade if any."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[MemoryRecordView]
    degraded: str | None = None

    @property
    def memory_ids(self) -> list[str]:
        return [it.memory_id for it in self.items]


class ContextView(BaseModel):
    """A deterministically-assembled context window (NO LLM synthesis — spec §7 INJECT render)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    items: list[MemoryRecordView]
    degraded: str | None = None


class ConsolidateView(BaseModel):
    """The receipt of one MTM->LTM consolidation sweep (maps from ``DistillReport``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    facts_extracted: int
    added: int
    superseded: int
