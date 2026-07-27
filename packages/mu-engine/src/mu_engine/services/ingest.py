"""``IngestService.remember(activity)`` — the fast-return INGEST entry (engine-core-spec §5).

Drives the CAPTURE->INGEST pipeline (``pipelines/concrete/ingest.py``): ``WriteStmStage`` (STM
durable, id minted once) -> ``DeterministicPromoteStage`` (STM->MTM atomic-fact vector via the REAL
local embedder) -> ``EmitIngestCompletedStage`` (DISTILL trigger). ``HALT_LOUD`` (engine-core-spec
§6.4): a stage failure keeps the durable partial and re-raises — never a silent swallow, never a
compensating rollback of an already-durable STM write.

Ports only (DEV-STANDARDS rule 5); fully async. The writer NEVER waits for conflict detection
(conflict-resolution-async §1). This slice runs the pipeline in-process; the durable-Temporal
dispatcher + bounded-queue backpressure runner land in a later phase.
"""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.events import DomainEvent, MemoryCaptured, MemoryPromoted
from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.pipelines.base import HaltPolicy, Pipeline, PipelineContext, Stage, StageStatus
from mu_engine.pipelines.concrete.ingest import (
    DeterministicPromoteStage,
    EmitIngestCompletedStage,
    IngestActivity,
    WriteStmStage,
    _build_memory_item,
    activity_id_for,
)
from mu_engine.pipelines.errors import StageExecutionError
from mu_engine.pipelines.ledger import StageLedger
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    TraceScope,
)
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.services.settings import IngestSettings
from mu_engine.storage.ports import MtmTierRepository, StmTierRepository

__all__ = ["IngestResult", "IngestService"]

_PIPELINE_NAME = "ingest"
_OP = "ingest.remember"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"


class IngestResult(BaseModel):
    """The fast-return receipt of one ``remember`` (content-free — ids/flags only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    content_hash: str
    promoted: bool
    tiers_written: tuple[str, ...]
    events_emitted: tuple[str, ...]


class IngestService:
    """Application-facing ingest owner (engine-core-spec §5 SHARED/api owner).

    App singleton (composition §11): tier repos + embedder + ledger + clock + bus are injected. The
    concrete pipeline is declared once at construction and reused per ``remember``.
    """

    def __init__(
        self,
        *,
        stm: StmTierRepository,
        mtm: MtmTierRepository,
        embedder: EmbeddingPort,
        bus: EventBusPort,
        ledger: StageLedger,
        clock: Clock,
        settings: IngestSettings | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._settings = settings or IngestSettings()
        # Central observability (DEV-STANDARDS rule 4): span + latency/error metrics + content-free
        # audit on the meaningful op. Sinks default to no-op so the service is testable unwired.
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()
        self._pipeline = Pipeline(
            name=_PIPELINE_NAME,
            halt_policy=HaltPolicy.HALT_LOUD,
            durable=False,
            stages=(
                WriteStmStage(stm=stm, ledger=ledger, clock=clock),
                DeterministicPromoteStage(
                    stm=stm,
                    mtm=mtm,
                    embedder=embedder,
                    settings=self._settings,
                    ledger=ledger,
                    clock=clock,
                ),
                EmitIngestCompletedStage(ledger=ledger, clock=clock),
            ),
        )

    async def remember(self, activity: IngestActivity) -> IngestResult:
        """Ingest one activity: STM durable, deterministic promote, fan-out trigger. Returns on the
        pipeline's durable completion; conflict detection is downstream (never on this path).

        Wrapped in central observability (DEV-STANDARDS rule 4): a content-free span, a latency
        histogram (always) + an error counter (on failure), and a content-free audit row on
        success. ``CancelledError`` propagates and is NOT counted as a failure (it is not one)."""
        correlation_id = activity_id_for(activity)
        started = time.perf_counter()
        with self._tracer.span(_OP, attributes={"pipeline": self._pipeline.name}):
            try:
                result = await self._remember(activity, correlation_id)
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._metrics.inc(_ERROR_METRIC, labels={"operation": _OP})
                raise
            finally:
                self._metrics.observe(
                    _LATENCY_METRIC, time.perf_counter() - started, labels={"operation": _OP}
                )
        # Content-free audit row (ids/enums/counts only — never memory text, §3.1).
        self._audit.record(
            TraceScope(correlation_id=correlation_id),
            operation=_OP,
            outcome="ok",
            tier=result.tiers_written[-1],
            visibility=activity.namespace.visibility.value,
            counts={"tiers_written": len(result.tiers_written)},
        )
        return result

    async def _remember(self, activity: IngestActivity, correlation_id: str) -> IngestResult:
        ctx = PipelineContext(
            pipeline=self._pipeline.name,
            namespace=activity.namespace,
            correlation_id=correlation_id,
            started_at=self._clock.now(),
            state={"activity": activity},
        )
        emitted: list[DomainEvent] = []
        for stage in self._pipeline.stages:
            emitted.extend(await self._run_stage(ctx, stage))

        # Derive the receipt from the EMITTED events (content-free) so a full-replay run — where
        # every stage is a SKIPPED ledger-hit with empty ``produced`` but the recorded events —
        # returns the identical id/flags as the first run.
        captured_ids = [mid for e in emitted if isinstance(e, MemoryCaptured) for mid in e.ids]
        if not captured_ids:
            raise StageExecutionError(_PIPELINE_NAME, "ingest produced no memory id")
        promoted = any(isinstance(e, MemoryPromoted) for e in emitted)
        content_hash = (
            str(ctx.state.get("content_hash") or "")
            or _build_memory_item(activity, at=self._clock.now()).content_hash
        )
        tiers = ("stm", "mtm") if promoted else ("stm",)
        return IngestResult(
            memory_id=captured_ids[0],
            content_hash=content_hash,
            promoted=promoted,
            tiers_written=tiers,
            events_emitted=tuple(type(e).__name__ for e in emitted),
        )

    async def _run_stage(self, ctx: PipelineContext, stage: Stage) -> list[DomainEvent]:
        """Run one stage under HALT_LOUD: merge its state, publish its events AFTER the ledger
        commit, and on failure keep the durable partial + re-raise (engine-core-spec §6.4)."""
        try:
            outcome = await stage.run(ctx)
        except Exception as error:  # HALT_LOUD: durable partial kept; surface loudly, no swallow.
            raise StageExecutionError(stage.name, str(error)) from error
        if outcome.status is StageStatus.FAILED:
            raise StageExecutionError(stage.name, outcome.reason or "stage reported FAILED")
        ctx.state.update(outcome.produced)
        # Events were made durable inside the stage's ledger row (B4); publish them after the
        # commit. A replayed SKIPPED stage re-publishes the recorded events (never empty).
        for event in outcome.events:
            await self._bus.publish(event)
        return list(outcome.events)
