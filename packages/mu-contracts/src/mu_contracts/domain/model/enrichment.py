"""S2 write-time enrichment — the durable job + result shapes (ADR-0055; ARCHITECTURE-DELTAS
AD-241; ``docs/superpowers/design/engine-core-spec.md`` §6.4 stage 4).

The owner's shape: ``add()`` stays synchronous and LLM-free (``pipelines/concrete/ingest.py``'s
own "NO LLM on this path" invariant, unchanged by this slice); a captured memory is queued for
enrichment and a background worker performs the LLM call afterward, writing the result back onto
the SAME ``MemoryItem`` (CANONICAL §7.1 id-stability) it already occupies in STM/MTM. Enrichment
only ADDS a representation — it never gates, blocks, or degrades the raw row, which is already
useful and already retrievable the instant ``add()`` returns.

**Content-free by construction (rule 3).** ``EnrichmentJob`` carries only ids/namespace/hashes/
timestamps — never the memory's text. The worker resolves the actual content itself, at drain
time, by reading the durable STM/MTM row through the existing tier ports; nothing here duplicates
memory content into a THIRD store the way the daemon outbox does (``FAULT-HUNT-0924.md`` F4a is
exactly that mistake — a queue that carries content is a second place nothing ever purges).

Mirrors, structurally, the ``LifecycleJob``/``JobHandle`` shape (``mu_contracts/domain/model/
lifecycle.py``) — same "one durable job log, ``job_id``-idempotent, plane-specific adapter"
pattern — but is intentionally a SEPARATE type family. S2's lane owns the ingest path + this new
enrichment worker, explicitly NOT lifecycle internals (the owner's own scoping); reusing
``LifecycleJob``'s table/port would smuggle an unrelated write surface into a file another lane
owns. Two small, honest ports beat one shared one two lanes now have to coordinate on.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EnrichmentJob",
    "EnrichmentJobStatus",
    "EnrichmentPayload",
]


class EnrichmentJobStatus(StrEnum):
    """A queue row's lifecycle (mirrors ``JobStatus``/``RecordState`` precedent)."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"  # permanently dead-lettered after max_attempts — see queue adapter


class EnrichmentPayload(BaseModel, frozen=True):
    """The write-back representation (A-mem shape: PORT of ``MemoryNote.analyze_content``,
    ``other_repos/A-mem/memory_layer.py:308-400`` — one structured LLM pass producing keywords +
    a context sentence + tags; mem0's ADD/UPDATE/DELETE diff loop is deliberately NOT reused here,
    it already lives in ``pipelines/distill.py`` for MTM->LTM consolidation, a different
    question). Additive on ``MemoryItem`` (``storage/domain/memory.py``'s ``enrichment`` field) —
    a memory with ``enrichment=None`` is exactly today's shipped row, fully retrievable.
    """

    model_config = ConfigDict(extra="forbid")

    keywords: tuple[str, ...] = ()
    context: str = ""
    tags: tuple[str, ...] = ()
    model: str = ""
    enriched_at: datetime


class EnrichmentJob(BaseModel, frozen=True):
    """One durable queue row (``EnrichmentQueuePort.submit``). ``job_id`` is the idempotency key
    — deterministic from ``memory_id`` (``enr_{memory_id}``) so a crash-replayed
    ``EnqueueEnrichmentStage`` resubmits the SAME row rather than forking a duplicate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(min_length=1)
    namespace_parts: tuple[str, str, str, str, str]  # Namespace.parts() — CANONICAL §7.3 codec
    memory_id: str = Field(min_length=1)
    content_hash: str = ""  # advisory only — never re-derives identity, see module docstring
    enqueued_at: datetime
