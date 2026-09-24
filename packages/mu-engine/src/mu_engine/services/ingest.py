"""``IngestService.remember(activity)`` — the fast-return INGEST entry (engine-core-spec §5).

Drives the CAPTURE->INGEST pipeline (``pipelines/concrete/ingest.py``): ``PersistRawArtifactStage``
(NEW, optional — persists the raw activity as a ContextArtifact provenance root, software-arch spec
§6 ``IngestService.ingest`` step 1, l.340) -> ``WriteStmStage`` (STM durable, id minted once,
kind=REFERENCE -> artifact_ref when the artifact stage ran, spec l.341) ->
``DeterministicPromoteStage`` (STM->MTM atomic-fact vector via the REAL local embedder) ->
``EmitIngestCompletedStage`` (DISTILL trigger). ``HALT_LOUD`` (engine-core-spec §6.4): a stage
failure keeps the durable partial and re-raises — never a silent swallow, never a compensating
rollback of an already-durable STM write.

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
    PersistRawArtifactStage,
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
from mu_engine.storage.domain.namespace import Visibility
from mu_engine.storage.ports import ContextRepository, MtmTierRepository, StmTierRepository

__all__ = ["IngestResult", "IngestService"]

_PIPELINE_NAME = "ingest"
_OP = "ingest.remember"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"

# S1b write-side blocker (TRACE-0923.md §7/§6.2, AD-234, `IngestService._ensure_turn_seq`'s own
# docstring): mirrors `mu_local.local_memory._TURN_SEQ_SCAN_LIMIT` 1:1 — the current longest
# LoCoMo conversation this repo's own corpus carries is 663 turns (conv-41); set generously above
# that, NOT tuned to the benchmark. A session whose STM window exceeds this can under-count and
# assign a colliding `turn_seq` — the SAME documented, bounded gap `local_memory.py` already
# carries, not a new one this centralisation introduces.
_TURN_SEQ_SCAN_LIMIT = 4000


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
        artifacts: ContextRepository | None = None,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._settings = settings or IngestSettings()
        # S1b write-side blocker fix (TRACE-0923.md §7/§6.2, AD-234; `_ensure_turn_seq`'s own
        # docstring has the full rationale). Kept ONLY for the turn_seq auto-assign fallback below
        # — every stage that actually writes/reads STM already has its own `stm` reference.
        self._stm = stm
        # Per-namespace `turn_seq` continuation cache + lock, the SAME shape as `mu_local.
        # local_memory.LocalMemory._turn_seq_next` (that class's own `_next_turn_seq_cached`
        # docstring has the "read from the store once per instance, not once per add" rationale
        # this mirrors) — an `asyncio.Lock` per namespace additionally guards the read-then-write
        # race a single-tenant `LocalMemory` never has to worry about but a multi-tenant
        # `IngestService` (this class, shared across every SHARED-plane request on mu-server) does:
        # two concurrent `remember()` calls on the SAME namespace, both missing the cache, must not
        # both compute the same base and hand out a colliding `turn_seq`.
        # KNOWN, DOCUMENTED LIMIT (not hidden — this project's own established discipline for a
        # tracked gap, e.g. AD-234's own `_TURN_SEQ_SCAN_LIMIT` account): both dicts grow by one
        # entry per DISTINCT namespace this instance ever sees and are never evicted. `LocalMemory`
        # carries the identical shape today and it is bounded there by one process's own session
        # count; on mu-server's long-lived, multi-tenant `IngestService` the namespace set is every
        # session of every tenant the process ever serves, so this is unbounded per-process growth
        # over a long uptime — real, but small (one int + one near-empty `asyncio.Lock` per
        # namespace, not per row), and a process restart clears it (the values are re-derived from
        # the store on first touch, never authoritative on their own). Named here as a follow-up
        # (a bounded/LRU cache) rather than built — out of this fix's scope, which is closing the
        # "3 ingest paths, 1 remembers `turn_seq`" gap, not redesigning this cache's eviction.
        self._turn_seq_next: dict[str, int] = {}
        self._turn_seq_locks: dict[str, asyncio.Lock] = {}
        # Central observability (DEV-STANDARDS rule 4): span + latency/error metrics + content-free
        # audit on the meaningful op. Sinks default to no-op so the service is testable unwired.
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()
        # NEW — ``artifacts`` (software-arch spec §6, l.340): optional so every EXISTING caller
        # that constructs an ``IngestService`` without an artifact store (every unit/integration
        # test in this tree today) keeps its byte-identical behaviour — plain PROPOSITION
        # captures, no PersistRawArtifactStage. A caller that DOES thread a real
        # ``ContextRepository`` (the composition roots, ``mu_local``/``mu_engine_server``) gets
        # the stage prepended and every capture becomes kind=REFERENCE targeting a persisted
        # ContextArtifact (this ADDS the reference-capture path; it never removes the
        # proposition/distill one downstream).
        artifact_stage: tuple[Stage, ...] = (
            (PersistRawArtifactStage(artifacts=artifacts, ledger=ledger, clock=clock),)
            if artifacts is not None
            else ()
        )
        self._pipeline = Pipeline(
            name=_PIPELINE_NAME,
            halt_policy=HaltPolicy.HALT_LOUD,
            durable=False,
            stages=(
                *artifact_stage,
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
        activity = await self._ensure_turn_seq(activity)
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

    async def _ensure_turn_seq(self, activity: IngestActivity) -> IngestActivity:
        """S1b write-side blocker (``docs/tracking/TRACE-0923.md`` §7/§6.2, AD-234): auto-assign
        ``turn_seq`` for any caller that leaves it unset, so an ingest path can no longer forget
        it by omission. AD-234 found that only ONE of three ``IngestActivity`` construction sites
        assigned ``turn_seq`` at all — ``mu_local.local_memory.LocalMemory.add`` did (its own
        ``_next_turn_seq_cached``/``_next_turn_seq_base``), ``mu_engine.surface.facade.
        SurfaceFacade.add`` and mu-server's ``SharedMemoryService.add`` did not, and — the delta's
        own words — a row written through either of those two is "permanently unexpandable" by
        S1b's read-time neighbour expansion. That is a call-site convention (every NEW surface has
        to remember to replicate the caching dance), which is exactly the failure shape this
        project's own CLAUDE.md names as the recurring defect pattern (relying on every caller to
        remember one thing). Moving the assignment HERE — the one method every ingest path in this
        tree already funnels through, ``LocalMemory``/``SurfaceFacade``/mu-server's
        ``SharedMemoryService`` alike — makes it structural instead of conventional: a FOURTH
        surface gets a real ``turn_seq`` for free, without knowing this field exists.

        **An explicit caller-supplied ``turn_seq`` always wins and is never recomputed or
        overridden** — this only fills the gap for a caller that expressed no opinion (``None``,
        ``IngestActivity``'s own field default). ``LocalMemory.add`` keeps its own pre-assignment
        exactly as ADR 0053 shipped it (needed there: a multi-message ``add()`` call assigns
        CONSECUTIVE values across one call without a store round-trip per message — see that
        method's own docstring); this fallback exists for every caller that does not do that,
        which today is every other one.

        **Design mirrors ``LocalMemory._next_turn_seq_cached``/``_next_turn_seq_base`` exactly**
        (that pair's own docstrings have the full "read from the store, not an in-process counter,
        because a fresh instance must continue a prior instance's sequence" rationale, and the
        documented ``_TURN_SEQ_SCAN_LIMIT`` gap this inherits verbatim) — read the current STM
        session max ONCE per namespace per ``IngestService`` instance, then increment in memory;
        this is also the AD-234 PERFORMANCE fix applied at the one place every path funnels
        through, not a second unoptimised copy of the pre-fix O(n)-per-add cost.

        **One addition ``LocalMemory`` does not need**: an ``asyncio.Lock`` per namespace. A
        single-tenant ``LocalMemory`` never serves two concurrent ``add()`` calls on the SAME
        session from two different callers; this class is shared across every request on the
        multi-tenant SHARED plane (mu-server), where that IS reachable — without the lock, two
        concurrent ``remember()`` calls on a namespace neither has cached yet could both compute
        the same base and hand out a colliding ``turn_seq``."""
        if activity.turn_seq is not None:
            return activity
        ns = activity.namespace
        key = ns.to_prefix()
        lock = self._turn_seq_locks.setdefault(key, asyncio.Lock())
        async with lock:
            next_seq = self._turn_seq_next.get(key)
            if next_seq is None:
                # Model-A (CANONICAL §7.4): PRIVATE authorizes by partition (`None` is correct
                # there — `IngestActivity`'s own validator forbids `authorized_ids` on PRIVATE);
                # SHARED needs a real set, and an unstamped SHARED write (`authorized_ids=None`,
                # a legal but content-free-to-everyone row, `IngestActivity.authorized_ids`'s own
                # docstring) still must not crash this scan — coerce to the empty set, the SAME
                # "authorize nothing, never over-broad" direction `RecallService`/`ranker.py`
                # already use for exactly this case.
                caller_identity_set = activity.authorized_ids
                if ns.visibility is Visibility.SHARED and caller_identity_set is None:
                    caller_identity_set = frozenset()
                window = await self._stm.recent(
                    ns,
                    limit=_TURN_SEQ_SCAN_LIMIT,
                    caller_identity_set=caller_identity_set,
                )
                seen = [s.item.turn_seq for s in window if s.item.turn_seq is not None]
                next_seq = (max(seen) + 1) if seen else 0
            self._turn_seq_next[key] = next_seq + 1
        return activity.model_copy(update={"turn_seq": next_seq})

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
            or _build_memory_item(
                activity,
                at=self._clock.now(),
                artifact_ref=ctx.state.get("artifact_id"),
                provenance_id=ctx.state.get("artifact_provenance_id"),
            ).content_hash
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
