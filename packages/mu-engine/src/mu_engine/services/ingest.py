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

from mu_contracts.domain.events import DomainEvent, MemoryCaptured
from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.pipelines.base import (
    HaltPolicy,
    Pipeline,
    PipelineContext,
    Stage,
    StageOutcome,
    StageStatus,
)
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
        outcomes: dict[str, StageOutcome] = {}
        for stage in self._pipeline.stages:
            outcome = await self._run_stage(ctx, stage)
            outcomes[stage.name] = outcome
            emitted.extend(outcome.events)

        # Derive the receipt from the EMITTED events (content-free) so a full-replay run — where
        # every stage is a SKIPPED ledger-hit with empty ``produced`` but the recorded events —
        # returns the identical id/flags as the first run.
        captured_ids = [mid for e in emitted if isinstance(e, MemoryCaptured) for mid in e.ids]
        if not captured_ids:
            raise StageExecutionError(_PIPELINE_NAME, "ingest produced no memory id")
        promoted = self._promoted_this_call(outcomes)
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

    @staticmethod
    def _promoted_this_call(outcomes: dict[str, StageOutcome]) -> bool:
        """The honest ``promoted``/``tiers_written`` signal for the receipt (F2a receipt-honesty
        fix, Stage-F acceptance — ``tests/acceptance/test_f2a_crash_replay.py``).

        A bare ``any(isinstance(e, MemoryPromoted) for e in emitted)`` is NOT sufficient:
        ``DeterministicPromoteStage`` republishes its OWN recorded event on every content-hash
        ledger-hit (``BaseStage.run``'s "a ledger hit returns the recorded events; SKIPPED
        therefore NEVER means 'no events'" contract, ``pipelines/base.py``) — including when that
        ledger-hit belongs to a DIFFERENT, EARLIER activity: a distinct, independently-issued
        ``add()`` call whose content happens to hash the same (``SurfaceFacade.add()`` mints a
        fresh ``session_offset`` every call, ``facade.py::_fresh_offset`` — no caller-supplied
        idempotency key ever makes two such calls collide on ``WriteStmStage``'s own activity-id
        ledger). Treating "a MemoryPromoted event is present in the emitted list" as "this call
        promoted" therefore claimed ``tiers_written=("stm", "mtm")`` for a call that performed ZERO
        new MTM I/O — a receipt-accuracy defect (the MTM dedup itself was always correct: exactly
        one point per content_hash; only the RECEIPT lied about which call earned it).

        The honest per-call signal distinguishes what actually happened THIS invocation by reading
        each stage's OWN ``StageOutcome.status``, never just the aggregated event list:

        1. ``deterministic_promote`` genuinely ran (``OK``) this call — its own
           ``produced["promoted"]`` is ground truth: ``True`` for a fresh embed+upsert, ``False``
           when the promotion criteria simply were not met (that path is NEVER ledger-gated at all
           — ``DeterministicPromoteStage.idempotency_key`` returns ``""`` when
           ``_promotion_reason`` is ``None``, so it always re-``_execute``s, never SKIPs).
        2. ``deterministic_promote`` SKIPPED (content-hash ledger-hit) AND ``write_stm`` ALSO
           SKIPPED (activity-id ledger-hit) — a genuine full replay of the IDENTICAL prior call
           (same ``session_offset``: a client retry, or the crash-resume case
           ``tests/pipelines/test_crash_replay_resume_int.py`` proves, and the existing
           ``test_remember_is_idempotent_on_replay`` this fix must not regress) — that op already
           completed a real promotion earlier; reporting ``promoted=True`` honestly matches the
           completed state of the op being replayed, not a distinct occurrence.
        3. ``deterministic_promote`` SKIPPED while ``write_stm`` ran fresh (``OK`` — a NEW
           activity/session_offset: a genuinely independent, later ``add()`` whose content happens
           to match an EARLIER, DIFFERENT call's content_hash) — reinforcement, not a new write: no
           MTM I/O happened for THIS memory_id, so ``promoted=False`` /
           ``tiers_written=("stm",)`` is the honest receipt (the F2a finding).
        """
        promote_outcome = outcomes.get(DeterministicPromoteStage.name)
        if promote_outcome is None:
            return False
        if promote_outcome.status is StageStatus.OK:
            return bool(promote_outcome.produced.get("promoted", False))
        write_stm_outcome = outcomes.get(WriteStmStage.name)
        return write_stm_outcome is not None and write_stm_outcome.status is StageStatus.SKIPPED

    async def _run_stage(self, ctx: PipelineContext, stage: Stage) -> StageOutcome:
        """Run one stage under HALT_LOUD: merge its state, publish its events AFTER the ledger
        commit, and on failure keep the durable partial + re-raise (engine-core-spec §6.4).

        Returns the stage's own :class:`StageOutcome` (status + produced), not just its events —
        ``_promoted_this_call`` needs the real per-stage ``OK``-vs-``SKIPPED`` signal, which the
        aggregated event list alone cannot distinguish (a SKIPPED ledger-hit re-publishes the SAME
        event shape an OK run would have produced, see that method's docstring)."""
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
        return outcome
