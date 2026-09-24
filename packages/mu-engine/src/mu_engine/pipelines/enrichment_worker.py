"""``EnrichmentWorker`` — the S2 write-time enrichment worker (ADR-0055; AD-241;
``engine-core-spec.md`` §6.4 stage 4's downstream half).

Drains ``EnrichmentQueuePort`` off the request path: for each claimed job it reads the memory's
CURRENT STM/MTM row by id, calls ``EnrichmentExtractorPort.enrich(content)`` (the ONE LLM call
this whole slice makes), and writes the returned :class:`EnrichmentPayload` back onto the SAME
row via a plain re-``put``/``upsert`` (no new repository verb — CANONICAL §7.1 id-stability: the
id is unchanged, only the additive ``enrichment`` field is set). This is the ONLY place in the
engine that calls an LLM as a side effect of a capture; ``pipelines/concrete/ingest.py`` keeps its
"NO LLM on this path" invariant untouched.

**"Enrichment only adds" — the properties this module has to hold, and how:**

* **Never degrades the unenriched memory.** A slow/failed/timed-out call touches nothing — the
  worker reads the item, and only WRITES if ``enrich()`` returned. A memory the worker has not
  gotten to yet, or gave up on, is byte-identical to one enrichment was never enabled for.
* **Idempotent, restart-safe.** ``claim_batch`` moves a row ``PENDING -> RUNNING``; a worker that
  crashes mid-batch leaves rows ``RUNNING`` that the QUEUE adapter resets on its own next
  ``resume_pending()`` (this module never has to reason about that — it is the adapter's
  contract, exercised by ``SqliteWalEnrichmentQueue``'s own crash-resume test). The write-back
  itself is idempotent too: re-running the SAME job twice (a redelivery) re-derives and re-writes
  the SAME payload shape onto the SAME id — an overwrite, never a duplicate, never a second row.
* **A memory deleted before its enrichment runs is not resurrected.** If neither tier has the id
  any more, the job is ``complete()``-d as a no-op — never an error, never a write.
* **Content-free logging/metering (rule 3).** Every log line below carries ids/counts/latency/
  exception CLASS NAMES only — never ``content``, never a field of ``EnrichmentPayload`` (which is
  LLM-derived text). The queue row itself never held content either (see
  ``mu_contracts.domain.model.enrichment`` module docstring) — this worker is the only place that
  ever reads the verbatim text, and it never logs it.
"""

from __future__ import annotations

import asyncio
import time

import structlog
from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.enrichment import EnrichmentJob
from mu_contracts.domain.model.memory import Namespace
from mu_contracts.ports.enrichment import EnrichmentExtractorPort, EnrichmentQueuePort
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_engine.platform.observability import NoopAuditLog, NoopMetricSink, NoopTracer, TraceScope
from mu_engine.storage.ports import MtmTierRepository, StmTierRepository

__all__ = ["EnrichmentBatchReport", "EnrichmentWorker", "EnrichmentWorkerSettings"]

_log = structlog.get_logger("mu_engine.pipelines.enrichment_worker")

_OP = "enrichment.worker.run_once"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"
_JOB_METRIC = "mu_enrichment_jobs_total"


class EnrichmentWorkerSettings(BaseModel, frozen=True):
    """Central-config knobs (DEV-STANDARDS rule 3). ``batch_size`` bounds how much work one
    ``run_once`` claims — the worker's own half of "a burst of writes must not queue
    unboundedly": the QUEUE bounds how many rows can be PENDING at once (backpressure at enqueue
    time); this bounds how many the worker holds RUNNING/in-flight at once (backpressure at drain
    time), so one call never claims the whole backlog into memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=16, ge=1)
    call_timeout_s: float = Field(default=30.0, gt=0.0)
    retry_backoff_s: float = Field(default=5.0, ge=0.0)


class EnrichmentBatchReport(BaseModel, frozen=True):
    """Content-free summary of one ``run_once`` (safe to log/emit/return whole)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claimed: int = 0
    enriched: int = 0
    skipped_deleted: int = 0
    retried: int = 0
    dead_lettered: int = 0


class EnrichmentWorker:
    """Ports only (DEV-STANDARDS rule 5); fully async. ``run_once`` is the unit the client
    scheduling loop (``mu_client.enrichment.worker_loop``) and the server Temporal activity
    (tracked gap, AD-241) both call — the SAME object either way, mirroring the ``Stage`` protocol
    doc's "identical object runs in-process or as a durable activity" rule."""

    def __init__(
        self,
        *,
        queue: EnrichmentQueuePort,
        extractor: EnrichmentExtractorPort,
        stm: StmTierRepository,
        mtm: MtmTierRepository,
        settings: EnrichmentWorkerSettings | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._queue = queue
        self._extractor = extractor
        self._stm = stm
        self._mtm = mtm
        self._settings = settings or EnrichmentWorkerSettings()
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()

    async def run_once(self) -> EnrichmentBatchReport:
        """Claim one bounded batch and process every job. Never raises for a per-job failure
        (each is isolated and accounted in the report); a genuinely unexpected error inside the
        loop itself (e.g. the queue port raising on ``claim_batch``) DOES propagate — that is a
        substrate failure the caller's scheduling loop must see and back off on, not a per-memory
        enrichment outcome to silently swallow."""
        started = time.perf_counter()
        with self._tracer.span(_OP):
            jobs = await self._queue.claim_batch(limit=self._settings.batch_size)
            enriched = skipped = retried = dead = 0
            for job in jobs:
                outcome = await self._process_one(job)
                if outcome == "enriched":
                    enriched += 1
                elif outcome == "skipped_deleted":
                    skipped += 1
                elif outcome == "retried":
                    retried += 1
                else:
                    dead += 1
                self._metrics.inc(_JOB_METRIC, labels={"outcome": outcome})
            self._metrics.observe(
                _LATENCY_METRIC, time.perf_counter() - started, labels={"operation": _OP}
            )
        report = EnrichmentBatchReport(
            claimed=len(jobs),
            enriched=enriched,
            skipped_deleted=skipped,
            retried=retried,
            dead_lettered=dead,
        )
        # Content-free audit row (ids/enums/counts only — §3.1).
        self._audit.record(
            TraceScope(correlation_id=f"enrichment-batch-{started:.6f}"),
            operation=_OP,
            outcome="ok",
            counts=report.model_dump(),
        )
        return report

    async def _process_one(self, job: EnrichmentJob) -> str:
        """Returns one of ``"enriched"``/``"skipped_deleted"``/``"retried"``/``"dead_lettered"``.
        Every branch ends by calling exactly one of ``complete``/``fail`` on the queue — a job is
        never left ``RUNNING`` by this method (only a genuine process crash leaves that, and
        that is what ``resume_pending`` exists to recover)."""
        ns = Namespace.from_parts(job.namespace_parts)
        job_started = time.perf_counter()
        try:
            stm_item, mtm_item = await asyncio.gather(
                self._stm.get(ns, job.memory_id),
                self._mtm.get(ns, job.memory_id),
            )
        except Exception as exc:  # broad except deliberate: a read failure is a retry, not a crash
            await self._queue.fail(
                job.job_id, error=type(exc).__name__, backoff_s=self._settings.retry_backoff_s
            )
            _log.warning("enrichment_read_failed", job_id=job.job_id, error=type(exc).__name__)
            return "retried"

        if stm_item is None and mtm_item is None:
            # Deleted (or never landed — a race with a concurrent delete) before enrichment ran.
            # "Enrichment only adds": there is nothing to add to any more. Not a failure.
            await self._queue.complete(job.job_id)
            _log.info("enrichment_skipped_deleted", job_id=job.job_id)
            return "skipped_deleted"

        content = (stm_item or mtm_item).content  # type: ignore[union-attr]
        try:
            payload = await asyncio.wait_for(
                self._extractor.enrich(content), timeout=self._settings.call_timeout_s
            )
        except Exception as exc:  # broad except deliberate: model/timeout failure -> retry/dead
            dead = await self._queue.fail(
                job.job_id, error=type(exc).__name__, backoff_s=self._settings.retry_backoff_s
            )
            self._metrics.inc(_ERROR_METRIC, labels={"operation": _OP})
            _log.warning(
                "enrichment_extract_failed",
                job_id=job.job_id,
                error=type(exc).__name__,
                dead_lettered=dead,
            )
            return "dead_lettered" if dead else "retried"

        if stm_item is not None:
            await self._stm.put(stm_item.model_copy(update={"enrichment": payload}))
        if mtm_item is not None:
            await self._mtm.upsert(mtm_item.model_copy(update={"enrichment": payload}))
        await self._queue.complete(job.job_id)
        _log.info(
            "enrichment_written",
            job_id=job.job_id,
            memory_id=job.memory_id,
            latency_ms=round((time.perf_counter() - job_started) * 1000, 1),
        )
        return "enriched"
