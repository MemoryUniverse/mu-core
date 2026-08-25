"""``DemotionService`` — the MTM->STM forgetting-curve tier-down move (S1-02; ADR 0034 demotion
half; ``docs/superpowers/design/memory-lifecycle-manager-spec.md`` §7b lines 304-306, AC-1.3
line 435).

**What this is.** The forgetting curve's other half of :class:`~mu_engine.lifecycle.salience.
SalienceStrategy` (S0-08): given a batch of MTM-tier candidates (already gathered by the sweep
loop — the same ``window: Sequence[MemoryItem]`` shape :class:`~mu_engine.pipelines.distill.
DistillPipeline.distill` takes, this module does not itself enumerate MTM), an item whose
``S(m) < demote_mtm`` (``LifecycleSettings.demote_mtm``, default 0.3, spec §16) demotes
MTM->STM. PORT of the MMA reference pattern — ``mma/mma/memory/controller.py``
``_run_demotion:1157`` (the sweep-and-gate loop) + ``atomic_demote_one:189`` (the
write-ahead-then-commit-delete sequencing, "dup > loss") — re-implemented onto this repo's
ports/DTOs/observability conventions, not blind-copied (DEV-STANDARDS "Code-adoption").

**AC-1.3 — the event-shape fix lands HERE, not after (M6).** Today's ``MemoryDemoted`` models
archival (``to_state=State.ARCHIVED``); a tier-down move is NOT an archival and must never
reuse that shape (ADR 0034 consequence: doing so would corrupt the validity-first retention
FSM, ADR 0035, which keys archival off validity, never off a tier move). ``MemoryDemoted.
to_tier`` (S0-04, ``mu_contracts/domain/events.py:323-343``) already ships the additive field
this service is the FIRST real emitter of: every demotion this module performs publishes
``MemoryDemoted(tier=Tier.MTM, to_tier=Tier.STM, to_state=State.ACTIVE, retention=S(m))`` —
``to_state`` is explicitly ``ACTIVE`` (the item is still alive, just moved tier), never the
``ARCHIVED`` default that every pre-existing (archival-only) call site keeps using unchanged.

**Rescue (ADR 0034 "the deliberate feedback loop").** A re-recalled item's raised
``access_count`` (recall bumps it back, e.g. ``distill.py:373``) raises ``use(m)`` and
therefore ``S(m)`` on the next :meth:`DemotionService.demote` pass; an item whose recomputed
score is ``>= demote_mtm`` is skipped entirely (zero writes to either store) — it simply never
leaves MTM. This is the whole rescue mechanic: it is a property of re-evaluating
:class:`SalienceStrategy` fresh every sweep tick, not a separate promotion call this module
makes.

**Clock-injected (spec §19 Rule 1).** Every ``now`` this module could need comes from the
injected ``Clock`` via :meth:`SalienceStrategy.score` — this module makes no
``datetime.now()``/wall-clock call of its own; the one place a "current instant" is stamped
(``stm_copy.updated_at``) also reads ``self._clock.now()``.

**Observability (DEV-STANDARDS rule 4) — matches ``DistillPipeline``'s pattern exactly**
(``mu_engine/pipelines/distill.py`` ``distill``/``_distill``): a content-free span around the
whole sweep, latency (always) + error (on failure) metrics, a content-free audit row on
success, ``CancelledError`` propagates untouched (never counted as a failure).

**The MTM-removal port (CF-2, MLM-STAGE2-CARRYOVER.md — landed).** ``MtmTierRepository``
(``mu_engine/storage/ports.py``) now ships a genuine ``remove(ns, memory_id)`` method alongside
``upsert``/``semantic``/``invalidate``, backed on ``QdrantMtmAdapter`` by a real
``AsyncQdrantClient.delete`` call. ``invalidate`` still specifically models loser-*supersession*
(``state=superseded``, ``superseded_by=<winner-id>``) — this module never calls it for demotion
(there is no "winner" in a forgetting-curve tier-down); ``remove`` is the plain-delete sibling
operation that IS the right shape for this move. :class:`DemotionService` now depends directly
on the real, shared ``MtmTierRepository`` port for that capability.

**RETIRED at Stage-3 integrate.** The ``MtmRemovalPort`` back-compat alias (this file used to
define ``MtmRemovalPort = MtmTierRepository`` and ``mu_engine.lifecycle.__init__`` re-exported it
by name) is gone — no caller ever imported it by name, only by docstring reference. The ad-hoc
local shims this docstring used to flag (``mu_local/composition.py``'s ``_LocalMtmRemovalAdapter``,
``tests/lifecycle/test_manager_int.py``'s ``_NoopMtmRemoval``/``_RealMtmRemoval``,
``tests/lifecycle/test_demotion_int.py``'s ``_RealQdrantRemoval``) are likewise retired — every
call site now passes the real ``QdrantMtmAdapter``/``mtm`` instance directly as ``mtm_remove``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import structlog
from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.events import MemoryDemoted
from mu_contracts.domain.model.memory import State
from mu_contracts.domain.model.memory import Tier as ContractTier
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.pipelines.distill import EventPublisher
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    TraceScope,
)
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.ports import MtmTierRepository, StmTierRepository

__all__ = [
    "DemotionOutcome",
    "DemotionReport",
    "DemotionService",
]

_log = structlog.get_logger("mu_engine.lifecycle.demotion")

_OP = "lifecycle.demote"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"


class DemotionOutcome(BaseModel):
    """One candidate's demotion decision (in-process return value; content-free)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    demoted: bool
    score: float
    reason: str


class DemotionReport(BaseModel):
    """Aggregate outcome for one :meth:`DemotionService.demote` call (content-free counts +
    per-item detail, mirroring :class:`~mu_engine.pipelines.distill.DistillReport`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluated: int
    outcomes: tuple[DemotionOutcome, ...] = ()

    @property
    def demoted(self) -> int:
        return sum(o.demoted for o in self.outcomes)

    @property
    def rescued(self) -> int:
        """Evaluated candidates whose recomputed ``S(m)`` was already ``>= demote_mtm`` — the
        ADR 0034 rescue: zero writes, item simply stays in MTM."""
        return sum(not o.demoted for o in self.outcomes)


class DemotionService:
    """MTM->STM forgetting-curve demotion (spec §7b; ADR 0034). Fully async, cancellation-safe,
    repository-only writes — mirrors :class:`~mu_engine.pipelines.distill.DistillPipeline`'s
    shape and observability wiring exactly.
    """

    def __init__(
        self,
        *,
        stm: StmTierRepository,
        mtm_remove: MtmTierRepository,
        salience: SalienceStrategy | None = None,
        settings: LifecycleSettings | None = None,
        clock: Clock | None = None,
        bus: EventPublisher | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._stm = stm
        self._mtm_remove = mtm_remove
        self._settings = settings or LifecycleSettings()
        self._salience = salience or SalienceStrategy(self._settings.salience)
        self._clock: Clock = clock or SystemClock()
        self._bus = bus
        # Central observability (DEV-STANDARDS rule 4): span + latency/error metrics + content-
        # free audit, same as DistillPipeline. Sinks default no-op so the service is testable
        # unwired.
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()

    async def demote(self, ns: Namespace, candidates: Sequence[MemoryItem]) -> DemotionReport:
        """Evaluate ``candidates`` (already-gathered MTM-tier items — this service does not
        enumerate MTM itself, same contract as ``DistillPipeline.distill``'s ``window``) and
        demote every one whose recomputed ``S(m) < demote_mtm`` MTM->STM.

        Wrapped in central observability exactly like ``DistillPipeline.distill``: a
        content-free span, latency (always) + error (on failure) metrics, a content-free audit
        row on success. ``CancelledError`` propagates and is never counted as a failure.
        """
        started = time.perf_counter()
        with self._tracer.span(_OP, attributes={"ns": ns.to_prefix()}):
            try:
                report = await self._demote(ns, candidates)
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._metrics.inc(_ERROR_METRIC, labels={"operation": _OP})
                raise
            finally:
                self._metrics.observe(
                    _LATENCY_METRIC, time.perf_counter() - started, labels={"operation": _OP}
                )
        self._audit.record(
            TraceScope(correlation_id=ns.to_prefix()),
            operation=_OP,
            outcome="ok",
            tier="mtm",
            visibility=ns.visibility.value,
            counts={
                "evaluated": report.evaluated,
                "demoted": report.demoted,
                "rescued": report.rescued,
            },
        )
        return report

    async def _demote(self, ns: Namespace, candidates: Sequence[MemoryItem]) -> DemotionReport:
        if not self._settings.demotion_enabled:
            _log.info("demotion_disabled", ns=ns.to_prefix())
            return DemotionReport(evaluated=0, outcomes=())

        outcomes: list[DemotionOutcome] = []
        for item in candidates:
            if item.state != MemoryState.ACTIVE:
                continue  # only a live MTM item is a demotion candidate (MMA _run_demotion:1157)
            score = self._salience.score(item, clock=self._clock)
            if score < self._settings.demote_mtm:
                outcome = await self._demote_one(ns, item, score)
            else:
                # Rescue (ADR 0034): S(m) already at/above the gate — zero writes, stays in MTM.
                outcome = DemotionOutcome(
                    memory_id=item.id,
                    demoted=False,
                    score=score,
                    reason="salience_at_or_above_demote_mtm",
                )
            outcomes.append(outcome)

        report = DemotionReport(evaluated=len(outcomes), outcomes=tuple(outcomes))
        _log.info(
            "demotion_sweep_complete",
            ns=ns.to_prefix(),
            evaluated=report.evaluated,
            demoted=report.demoted,
            rescued=report.rescued,
        )
        return report

    async def _demote_one(self, ns: Namespace, item: MemoryItem, score: float) -> DemotionOutcome:
        """Write-ahead-then-commit-delete (PORT of ``atomic_demote_one:189`` — "dup > loss").

        Step 1 — write-ahead: add the STM-tier copy first (id-stable, ``created_at``
        preserved — CANONICAL §7.1). Step 2 — commit: remove the MTM point via the injected
        ``MtmTierRepository.remove`` (CF-2 — the real port, not a local shim). Step 3 — on a
        persistent removal failure: compensating
        rollback (evict the STM write-ahead copy), leaving the item in MTM only (the
        pre-demotion state) rather than silently duplicating it across tiers. If the rollback
        ALSO fails, the item is left in both stores — a genuine, logged dup (never a silent
        loss); no code path here ever deletes a memory outright.
        """
        stm_copy = item.model_copy(deep=True)
        stm_copy.tier = MemoryTier.STM
        stm_copy.updated_at = self._clock.now()

        # Step 1: write-ahead — STM gets the copy before MTM loses its point.
        await self._stm.put(stm_copy)

        # Step 2: commit — remove the MTM point. The injected port owns its own retry/backoff
        # (DEV-STANDARDS rule 7 — a cross-cutting concern is the ADAPTER's decorator, e.g.
        # every real *TierRepository already wraps its I/O in `retry_io`; this service does not
        # duplicate that loop).
        #
        # ⚠ `item.namespace`, NOT the sweep's `ns`. `MtmTierRepository.remove` is namespace-scoped
        # in the ADAPTER (C3) — it deletes a point only from the partition it is handed — and for
        # PRIVATE this sweep's own candidate source, `scan_for_demotion`, is deliberately
        # session-FEDERATED (BQ3/ADR 0030: it matches the session-less user prefix so one sweep
        # sees every one of the user's sessions). So `ns` here is the sweep's session and `item`
        # may legitimately belong to ANOTHER session of the SAME user; handing the adapter `ns`
        # would address the wrong partition, the scoped delete would match nothing, and step 1's
        # write-ahead copy would silently become a permanent cross-tier DUPLICATE — with no
        # exception raised, so not even the rollback below would fire. The item's own namespace is
        # both the correct partition and the one step 1 already wrote to (`stm_copy.namespace`).
        try:
            await self._mtm_remove.remove(item.namespace, item.id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return await self._rollback_after_failed_remove(ns, item, score, exc)

        if self._bus is not None:
            await self._bus.publish(
                MemoryDemoted(
                    namespace=ns,
                    id=item.id,
                    tier=ContractTier.MTM,
                    to_tier=ContractTier.STM,
                    to_state=State.ACTIVE,  # AC-1.3: a real tier-down move, NOT archival
                    retention=score,
                )
            )
        return DemotionOutcome(
            memory_id=item.id, demoted=True, score=score, reason="salience_below_demote_mtm"
        )

    async def _rollback_after_failed_remove(
        self, ns: Namespace, item: MemoryItem, score: float, cause: Exception
    ) -> DemotionOutcome:
        """Step 3 — compensating rollback (mirrors ``atomic_transition``'s rollback branch,
        ``mma/mma/memory/controller.py:158-181``): evict the STM write-ahead copy so the item
        is left in MTM only (unchanged) rather than duplicated. If the rollback itself fails,
        the item is left in both stores — logged loudly, never silently dropped either way."""
        _log.warning("demotion_mtm_remove_failed_rolling_back", memory_id=item.id, error=str(cause))
        try:
            # `item.namespace` for the same reason step 2 uses it (see `_demote_one`): STM is
            # key-prefixed by `Namespace.to_prefix()`, and the write-ahead copy this compensates
            # for was written under `stm_copy.namespace` — the ITEM's. Evicting under the sweep's
            # `ns` would miss a federated cross-session copy and leave the very duplicate this
            # rollback exists to prevent.
            await self._stm.evict(item.namespace, item.id)
        except asyncio.CancelledError:
            raise
        except Exception as rollback_exc:
            self._metrics.inc(_ERROR_METRIC, labels={"operation": _OP, "outcome": "fault_dup"})
            _log.error(
                "demotion_rollback_failed_dup",
                memory_id=item.id,
                remove_error=str(cause),
                rollback_error=str(rollback_exc),
            )
            return DemotionOutcome(
                memory_id=item.id,
                demoted=False,
                score=score,
                reason="mtm_remove_failed_rollback_failed_dup",
            )
        self._metrics.inc(_ERROR_METRIC, labels={"operation": _OP, "outcome": "fault_rolled_back"})
        return DemotionOutcome(
            memory_id=item.id,
            demoted=False,
            score=score,
            reason="mtm_remove_failed_rolled_back",
        )
