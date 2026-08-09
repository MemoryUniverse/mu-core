"""Public result DTOs for the :class:`~mu_local.local_memory.LocalMemory` facade.

Frozen pydantic-v2 value objects (DEV-STANDARDS rule 2) built per call, never shared across tasks.
They are the IN-PROCESS return payloads a LOCAL caller gets back — carrying the body is the whole
point of a read, so ``content`` is present (the content-free discipline governs buses/logs/metrics,
not these return values; CANONICAL §3.1). Each maps 1:1 from an engine result
(``IngestResult`` / ``RecallResult`` / ``DistillReport``) so the facade adds no second behaviour.

``MemoryRecordView`` (the pre-Decision-B per-hit item shape) is RETIRED (build-plan Stage B /
``SDK-BUILD-DECISIONS.md`` Decision B cross-cutting section) — subsumed by the canonical
:class:`~mu_contracts.contracts.recall.RecallItemView`, the single hit-item DTO
``RecallResult.items``/``ContextView.items`` both use everywhere else in this codebase
(``mu_contracts.contracts.recall`` module docstring). ``MemoryListView``/``ContextView`` below keep
their own back-compat shape but now type ``items`` as ``RecallItemView`` too, so they stay
constructible without resurrecting the retired class. ``LocalMemory.get()`` — the one call site that
built a ``MemoryRecordView`` — now returns the canonical full-row
:class:`~mu_contracts.contracts.memory.MemoryResponse` instead (``mu_local/local_memory.py``,
carryover CO-1), since a single point-get is a full row, not a ranked hit.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mu_contracts.contracts.recall import RecallItemView

__all__ = [
    "ConsolidateView",
    "ContextView",
    "MemoryListView",
    "MemoryWriteResult",
]


class MemoryWriteResult(BaseModel):
    """The fast-return receipt of one ``add`` (maps from ``IngestResult`` — ids/flags only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    content_hash: str
    promoted: bool
    tiers_written: tuple[str, ...]


class MemoryListView(BaseModel):
    """A ranked list result (from ``RecallResult``); ``degraded`` is the NAMED degrade if any.

    ``items`` is typed as the canonical :class:`~mu_contracts.contracts.recall.RecallItemView`
    (module docstring — the retired ``MemoryRecordView``'s replacement), not constructed by any
    production call site today (``LocalMemory.recall`` returns the canonical
    :class:`~mu_contracts.contracts.recall.RecallResult` directly); kept as a back-compat shape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: list[RecallItemView]
    degraded: str | None = None

    @property
    def memory_ids(self) -> list[str]:
        return [it.memory_id for it in self.items]


class ContextView(BaseModel):
    """A deterministically-assembled context window (NO LLM synthesis — spec §7 INJECT render).

    ``items`` is typed as the canonical :class:`~mu_contracts.contracts.recall.RecallItemView`
    (module docstring); not constructed by any production call site today (``LocalMemory.context``
    returns the canonical :class:`~mu_contracts.contracts.views.ContextView` directly); kept as a
    back-compat shape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    items: list[RecallItemView]
    degraded: str | None = None


class ConsolidateView(BaseModel):
    """The receipt of one MTM->LTM consolidation sweep (maps from ``DistillReport``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    facts_extracted: int
    added: int
    superseded: int
