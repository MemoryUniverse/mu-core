"""``InMemoryEnrichmentQueue`` — the in-process ``EnrichmentQueuePort`` adapter (ADR-0055; AD-241).

Mirrors ``platform/adapters/workflow_inline.py``'s ``InlineRunner`` precedent: no durability, no
cross-process crash-resume — the queue lives in one process's memory and is gone when it exits.
This is the right adapter for (a) unit/integration tests of ``EnrichmentWorker``/
``EnqueueEnrichmentStage`` that don't need a real SQLite file, and (b) a daemonless single-call
``mu-local`` session that wants enrichment to happen (via an explicit, caller-driven
``EnrichmentWorker.run_once()``) without owning a WAL file. **Never the FULL-LOCAL daemon's own
default** — the daemon is long-lived and crash-prone the way a durable substrate exists to cover,
so ``mu_client.enrichment.sqlite_queue.SqliteWalEnrichmentQueue`` is the daemon's adapter, not
this one (the same "inline is the test/degrade adapter" split ``LifecycleWorkflowRunnerPort``'s
own ADR draws for its three runners).

Bounded (DEV-STANDARDS: never an unbounded queue) via ``max_pending`` — ``submit`` returns
``False`` once that many rows are ``PENDING``, exactly the backpressure contract
``EnrichmentQueuePort.submit`` documents.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from mu_contracts.domain.model.enrichment import EnrichmentJob, EnrichmentJobStatus

__all__ = ["InMemoryEnrichmentQueue"]


@dataclass
class _Row:
    job: EnrichmentJob
    status: EnrichmentJobStatus = EnrichmentJobStatus.PENDING
    attempts: int = 0
    error: str | None = None


class InMemoryEnrichmentQueue:
    """``EnrichmentQueuePort``. One ``asyncio.Lock`` serializes the dict (mirrors every SQLite
    adapter in this tree serializing its one connection) — cheap correctness over throughput,
    which is fine here: this adapter never crosses a process boundary."""

    def __init__(self, *, max_pending: int = 1000, max_attempts: int = 5) -> None:
        self._rows: dict[str, _Row] = {}
        self._order: list[str] = []
        self._max_pending = max_pending
        self._max_attempts = max_attempts
        self._lock = asyncio.Lock()

    async def submit(self, job: EnrichmentJob) -> bool:
        async with self._lock:
            if job.job_id in self._rows:
                return True  # idempotent resubmit — no-op, existing row untouched
            pending = sum(1 for r in self._rows.values() if r.status is EnrichmentJobStatus.PENDING)
            if pending >= self._max_pending:
                return False
            self._rows[job.job_id] = _Row(job=job)
            self._order.append(job.job_id)
            return True

    async def claim_batch(self, *, limit: int) -> list[EnrichmentJob]:
        async with self._lock:
            claimed: list[EnrichmentJob] = []
            for job_id in self._order:
                if len(claimed) >= limit:
                    break
                row = self._rows[job_id]
                if row.status is EnrichmentJobStatus.PENDING:
                    row.status = EnrichmentJobStatus.RUNNING
                    claimed.append(row.job)
            return claimed

    async def complete(self, job_id: str) -> None:
        async with self._lock:
            row = self._rows.get(job_id)
            if row is not None:
                row.status = EnrichmentJobStatus.DONE

    async def fail(self, job_id: str, *, error: str, backoff_s: float) -> bool:
        del backoff_s  # in-process adapter has no scheduler to honour a delay; next claim is fine
        async with self._lock:
            row = self._rows.get(job_id)
            if row is None:
                return False
            row.attempts += 1
            row.error = error
            if row.attempts >= self._max_attempts:
                row.status = EnrichmentJobStatus.FAILED
                return True
            row.status = EnrichmentJobStatus.PENDING
            return False

    async def resume_pending(self) -> int:
        async with self._lock:
            reset = 0
            for row in self._rows.values():
                if row.status is EnrichmentJobStatus.RUNNING:
                    row.status = EnrichmentJobStatus.PENDING
                    reset += 1
            return reset

    async def pending_count(self) -> int:
        async with self._lock:
            return sum(1 for r in self._rows.values() if r.status is EnrichmentJobStatus.PENDING)
