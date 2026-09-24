"""``EnrichmentQueuePort`` / ``EnrichmentExtractorPort`` — S2 write-time enrichment (ADR-0055).

Two narrow seams, DEV-STANDARDS rule 5 (repository pattern) applied to a queue instead of a
store: the ingest path talks to ``EnrichmentQueuePort`` only, never a SQLite/Temporal client
directly; the worker talks to ``EnrichmentExtractorPort`` only, never an ``LLMProviderPort``
directly (that wiring lives one layer down, in ``mu_engine.services.enrich``, mirroring
``FactExtractorPort``/``LlmFactExtractor``).

**Per-plane adapters, one port** (the owner's instruction: "use the durable-execution substrate
the architecture already chose, per plane — do not introduce a third mechanism"): the CLIENT/
FULL-LOCAL adapter is a SQLite-WAL job log (``mu_client.enrichment.sqlite_queue
.SqliteWalEnrichmentQueue`` — the same proven WAL pattern as ``mu_client.outbox.sqlite_outbox
.SqliteOutbox`` and ``mu_client.runners.sqlite_wal.SqliteWalRunner``, a THIRD independent table,
never the lifecycle job log — S2's lane is explicitly not lifecycle internals). The SERVER
adapter is Temporal (ADR 0033's ``TemporalRunner``/``WorkflowRunnerPort`` precedent) and is
**not implemented in this slice** — no Temporal client exists anywhere in this tree yet
(verified: ``grep -rln temporalio`` over every repo returns nothing), so building one here would
be exactly the "silent stub that looks finished" DEV-STANDARDS forbids. Tracked as a named gap
(ARCHITECTURE-DELTAS AD-241) rather than faked. FULL-LOCAL never depends on the gap: the client
adapter needs no server at all.

Bounded, never unbounded (DEV-STANDARDS "Async correctness" — backpressure): ``submit`` returns
``False`` (never raises) when the adapter's pending-row cap is already at capacity, so a write
burst SHEDS enrichment rather than growing the queue file without limit — the caller (the ingest
stage) degrades that into a named, non-fatal ``StageDegraded`` and the raw memory is written
exactly as it always was. This is the ONE way a bounded queue may refuse; every other adapter
method either succeeds or represents a bug (a fail-loud raise), per the closed-exception-set
discipline the rest of this codebase already uses (``LifecycleWorkflowRunnerPort``'s own
``resume_pending``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.enrichment import EnrichmentJob, EnrichmentPayload

__all__ = ["EnrichmentExtractorPort", "EnrichmentQueuePort"]


@runtime_checkable
class EnrichmentQueuePort(Protocol):
    """The durable, restart-safe write-time enrichment job log. Idempotent on ``job.job_id``
    (a redelivered submit of the same id is a no-op — mirrors ``SqliteWalRunner.submit`` /
    ``SqliteOutbox.append``'s ``UNIQUE`` constraint)."""

    async def submit(self, job: EnrichmentJob) -> bool:
        """Durable enqueue; returns fast. ``True`` = accepted (fresh row or an idempotent
        resubmit of an existing one); ``False`` = the bounded queue is at capacity — the ONLY
        non-exceptional refusal this port makes (backpressure, not failure)."""
        ...

    async def claim_batch(self, *, limit: int) -> list[EnrichmentJob]:
        """``PENDING -> RUNNING`` atomically for up to ``limit`` rows, oldest first. A row a
        crashed worker left ``RUNNING`` is reset to ``PENDING`` by :meth:`resume_pending` on the
        next ``open()``, never claimed twice concurrently by two live workers."""
        ...

    async def complete(self, job_id: str) -> None:
        """Mark ``DONE``. Idempotent — completing an already-``DONE``/unknown id is a no-op,
        never a raise (a worker retrying after a crash between the write-back and this call must
        be able to call it again safely)."""
        ...

    async def fail(self, job_id: str, *, error: str, backoff_s: float) -> bool:
        """Record a failed attempt and reschedule ``PENDING`` after ``backoff_s``, UNLESS the
        adapter's own ``max_attempts`` is now exhausted, in which case the row is marked
        ``FAILED`` (permanent — the raw memory stays exactly as useful as before; enrichment
        simply never lands for this row, and this is not silent: the row is inspectable, not
        deleted). Returns ``True`` iff this call was the one that permanently failed the job."""
        ...

    async def resume_pending(self) -> int:
        """Crash-recovery replay on boot/open: any row left ``RUNNING`` is presumed crashed
        mid-job and is reset to ``PENDING``. Returns the count reset. Never double-writes and
        never silently drops a row — dup > loss, the same rule every durable adapter in this
        tree already follows."""
        ...

    async def pending_count(self) -> int:
        """Current ``PENDING`` depth — the backpressure gauge a worker/metric can watch."""
        ...


@runtime_checkable
class EnrichmentExtractorPort(Protocol):
    """One structured LLM pass over a memory's content -> :class:`EnrichmentPayload`. Raises on
    a malformed/empty model response (fail-loud, DEV-STANDARDS: never a silent empty-payload
    fallback) — the caller (``EnrichmentWorker``) is the one place that decides what a failure
    means for the job (retry with backoff, eventually dead-letter), never this seam."""

    async def enrich(self, content: str) -> EnrichmentPayload: ...
