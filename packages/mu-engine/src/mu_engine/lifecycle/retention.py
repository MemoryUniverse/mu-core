"""``RetentionService`` — validity-first LTM archival + GC + COLD-slide (S2-01; ADR 0035;
``docs/superpowers/design/memory-lifecycle-manager-spec.md`` §9 (lines 331-364), AC-2.1/AC-2.2
(lines 438-439), §19 Rule 2 item 2 (line 738)).

**The rule (ADR 0035 / spec §9).** An active+valid LTM fact PERSISTS and is NEVER archived for
being un-recalled — validity, not recall-frequency, drives retention:

- **PERMANENT** (``invalid_at is None``) lives forever; only an explicit supersede/delete ever
  changes its state. This service never even considers a PERMANENT/pinned fact for a
  time-driven transition (see :meth:`RetentionService._sweep` — the EPHEMERAL branch is gated
  on ``retention_class is EPHEMERAL`` and the COLD branch on ``retention_class is DURABLE``, so
  a PERMANENT item simply never matches either predicate, at any ``Clock`` advance — AC-2.2).
- **EPHEMERAL** self-expires (``ACTIVE`` -> ``EXPIRED``) once ``clock.now() >= invalid_at +
  ephemeral_grace_s`` — the ``SELF_EXPIRE`` bookkeeping flip spec §9 describes ("recall
  correctness needs only the existing ``invalid_at`` window ``facts_at`` already enforces";
  this flip never changes what ``facts_at``/``graph_recall`` return, it only feeds GC). Driven
  by the INJECTED :class:`~mu_contracts.ports.time.Clock` (spec §19 Rule 1) — never a bare
  ``datetime.now()`` call in this module, the AC-2.1 reproducibility requirement.
- **DURABLE** + importance-gated + long-inactive slides to the COLD sub-tier (``cold=True``,
  state stays ``ACTIVE`` — "archived-but-queryable": ``graph_recall``/``facts_at`` are
  state/validity filters and never look at ``cold``, so a COLD fact keeps surfacing on recall;
  :meth:`reactivate_on_recall` is the reversible flip back). Never PERMANENT/pinned.
- **SUPERSEDED/EXPIRED (dead) facts** are archival/GC candidates — never an ACTIVE non-expired
  fact — GC'd only after BOTH ``gc_history_window_d`` has elapsed since the fact died AND its
  provenance (``SUPERSEDED_BY``) chain HEAD is itself dead (spec §9's mermaid: ``SUPERSEDED -->
  GC : chain head dead AND age>gc_history_window_d``) — a fact whose chain head is still
  ``ACTIVE`` keeps its whole history, no matter how old the individual link.
- **``pinned`` (CANONICAL §7.10/§7.26)** is GC-ineligible unconditionally, honored via
  :meth:`_is_pinned` — see that method's docstring for why it is a ``getattr`` duck-type rather
  than a direct field read (the shipped ``MemoryItem``, S0-06, does not carry ``pinned`` yet).

**The missing LTM query/delete ports (flagged, not papered over — same pattern as
``demotion.py``'s ``MtmRemovalPort``).** ``GraphStorePort``/``FalkorLtmAdapter``
(``mu_engine/storage/ports.py`` + ``storage/adapters/falkor_ltm.py``, sibling tasks' owned
files) ship ``upsert_fact``/``graph_recall``/``facts_at``/``find_conflicts``/``invalidate``/
``resolve_entity`` only — an "as-of" / "still valid" read model, with NO "enumerate everything
in a given ``state``" query and NO real node delete. This service reuses ``upsert_fact`` for
every WRITE it needs (the EXPIRE/COLD flips are just a mutated ``MemoryItem`` re-``MERGE``'d —
``upsert_fact``'s ``memory_json`` carrier already round-trips ``state``/``cold``/
``retention_class`` losslessly, see ``graph_mapper.py``'s ``to_store``), but the two READS
(enumerate-by-state) and the one true DELETE (hard GC) it needs do not exist on that port. This
module depends on a narrow, LOCAL :class:`LtmRetentionStorePort` Protocol (this file) for
exactly those three capabilities — the integrate-phase wiring gap a sibling storage task
(CF-2) should close by promoting a real implementation onto ``FalkorLtmAdapter`` itself (or a
sibling adapter satisfying this Protocol structurally — no import of this module required).
Until that lands, callers construct :class:`RetentionService` with any object satisfying this
narrow Protocol — see this module's test file for a REAL adapter over the live
``mu-dev-falkordb``/``mu-dev-graph``.

**Reference pattern PORTed (SHAPE only, per DEV-STANDARDS code-adoption + this task's own
``reference_repos`` note — quarantine itself stays RESERVED, not built here).** ``mma/mma/
memory/controller.py``'s ``_archive_inactive_ltm:928`` (fetch-then-Python-filter a full
namespace's LTM facts, act only on the ones that pass the gate) and ``_run_archive_gc:997``
(a separate GC pass over already-archived rows, age-gated) — re-implemented onto this repo's
bi-temporal validity-first policy (ADR 0035), never onto MMA's blind time-inactivity gate.

**Clock-injected (spec §19 Rule 1).** Every ``now`` this module could need comes from the
``clock`` parameter :meth:`sweep` receives (``RetentionServicePort.sweep(ns, *, clock)`` —
``manager.py``'s local seam Protocol this class satisfies structurally) — this module makes no
``datetime.now()``/wall-clock call of its own.

**Observability (DEV-STANDARDS rule 4) — matches ``DistillPipeline``/``DemotionService``'s
pattern exactly**: a content-free span around the whole sweep, latency (always) + error (on
failure) metrics, a content-free audit row on success. ``CancelledError`` propagates untouched.
"""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from typing import ClassVar, Literal, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.events import MemoryGarbageCollected
from mu_contracts.domain.model.memory import State as ContractState
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.policy import LifecyclePolicy
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.pipelines.distill import EventPublisher
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    TraceScope,
)
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, RetentionClass
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.ports import GraphStorePort

__all__ = [
    "LtmRetentionStorePort",
    "RetentionOutcome",
    "RetentionReport",
    "RetentionService",
]

_log = structlog.get_logger("mu_engine.lifecycle.retention")

_OP = "lifecycle.retention_sweep"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"

# Archival/GC candidates (spec §9): "acts only on SUPERSEDED/EXPIRED" — never ACTIVE.
_DEAD_STATES = frozenset({MemoryState.SUPERSEDED, MemoryState.EXPIRED})

RetentionAction = Literal["self_expire", "cold_slide", "gc"]


@runtime_checkable
class LtmRetentionStorePort(Protocol):
    """The narrow LTM seam :class:`RetentionService` needs beyond the existing
    ``GraphStorePort`` (see module docstring's "missing LTM query/delete ports" section for
    the full rationale). Three capabilities, none of which exist on ``GraphStorePort`` today:

    * ``facts_by_state`` — enumerate every ``:Memory`` node in ``ns`` whose ``state`` is one of
      ``states`` (a full, namespace-scoped, un-filtered-by-validity read — deliberately NOT
      ``facts_at``, which is an as-of/still-valid read and would never return a dead or
      not-yet-expired-by-clock fact in the first place).
    * ``chain_head_state`` — walk the ``SUPERSEDED_BY`` provenance chain forward from
      ``memory_id`` to its terminal node (the one with no outgoing ``SUPERSEDED_BY`` edge —
      "the chain head") and return THAT node's ``state``. A node with no outgoing edge at all
      is its own head (returns its own state) — the common case for an ``EXPIRED`` fact (a
      self-expire is not a supersession, so it never gains an outgoing edge).
    * ``gc_delete`` — the one true hard-delete this service ever calls (never on an ACTIVE
      fact — see :meth:`RetentionService._sweep`).
    """

    async def facts_by_state(
        self, ns: Namespace, states: frozenset[MemoryState]
    ) -> list[MemoryItem]: ...

    async def chain_head_state(self, ns: Namespace, memory_id: str) -> MemoryState: ...

    async def gc_delete(self, ns: Namespace, memory_id: str) -> None: ...


class RetentionOutcome(BaseModel):
    """One candidate's retention decision (in-process return value; content-free)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    action: RetentionAction
    reason: str


class RetentionReport(BaseModel):
    """Aggregate outcome for one :meth:`RetentionService.sweep` call (content-free counts +
    per-item detail, mirroring :class:`~mu_engine.lifecycle.demotion.DemotionReport`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluated: int
    outcomes: tuple[RetentionOutcome, ...] = ()

    @property
    def acted_on(self) -> int:
        return len(self.outcomes)


class RetentionService:
    """Validity-first LTM retention: EPHEMERAL self-expire, SUPERSEDED/EXPIRED archival + GC
    (provenance-chain-exempt, pin-exempt), COLD slide + reactivate-on-recall (spec §9; ADR
    0035). Satisfies ``manager.py``'s local ``RetentionServicePort`` seam
    (``async def sweep(self, ns: Namespace, *, clock: Clock) -> int``) structurally — no import
    of ``manager.py`` needed or wanted (this module never imports the manager it plugs into).
    """

    def __init__(
        self,
        *,
        ltm: GraphStorePort,
        ltm_retention: LtmRetentionStorePort,
        settings: LifecycleSettings | None = None,
        clock: Clock | None = None,
        bus: EventPublisher | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._ltm = ltm
        self._ltm_retention = ltm_retention
        self._settings = settings or LifecycleSettings()
        self._clock: Clock = clock or SystemClock()
        self._bus = bus
        # The central FSM/pin guard (memory-health §6.1). Stateless + pure, so constructed here:
        # there is exactly one lifecycle policy, not a swappable strategy.
        self._policy = LifecyclePolicy()
        # Central observability (DEV-STANDARDS rule 4): span + latency/error metrics + content-
        # free audit, same as DistillPipeline/DemotionService. Sinks default no-op so the
        # service is testable unwired.
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()

    async def sweep(self, ns: Namespace, *, clock: Clock | None = None) -> int:
        """``RetentionServicePort`` conformance (``manager.py``'s local seam: ``sweep(ns, *,
        clock) -> int`` — count acted on). ``clock`` defaults to the constructor-injected clock
        so this service is directly callable in tests/standalone use; ``MemoryLifecycleManager.
        sweep_user`` always passes its own injected ``self._clock`` explicitly (spec §19 Rule 1
        — every lifecycle decision-maker takes an injected ``Clock``, this one included).

        Wrapped in central observability exactly like ``DemotionService.demote``: a
        content-free span, latency (always) + error (on failure) metrics, a content-free audit
        row on success. ``CancelledError`` propagates and is never counted as a failure.
        """
        active_clock = clock or self._clock
        started = time.perf_counter()
        with self._tracer.span(_OP, attributes={"ns": ns.to_prefix()}):
            try:
                report = await self._sweep(ns, active_clock)
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
            tier="ltm",
            visibility=ns.visibility.value,
            counts={"evaluated": report.evaluated, "acted_on": report.acted_on},
        )
        return report.acted_on

    async def _sweep(self, ns: Namespace, clock: Clock) -> RetentionReport:
        if not self._settings.enabled:
            _log.info("retention_disabled", ns=ns.to_prefix())
            return RetentionReport(evaluated=0)

        retention = self._settings.retention
        now = clock.now()
        outcomes: list[RetentionOutcome] = []
        evaluated = 0

        # ---- pass 1: ACTIVE facts — EPHEMERAL self-expire, DURABLE COLD-slide -------------
        # NEVER touches an ACTIVE non-expired fact (spec §9): the EPHEMERAL branch only fires
        # once now >= invalid_at + grace, and the COLD branch never changes `state` at all.
        active = await self._ltm_retention.facts_by_state(ns, frozenset({MemoryState.ACTIVE}))
        newly_dead: dict[str, MemoryItem] = {}
        for item in active:
            evaluated += 1
            if retention.respect_pins and self._is_pinned(item):
                continue  # CANONICAL §7.10: pinned => GC/archival-ineligible, unconditionally

            if item.retention_class is RetentionClass.EPHEMERAL and item.invalid_at is not None:
                grace = timedelta(seconds=retention.ephemeral_grace_s)
                if now >= item.invalid_at + grace:
                    # CENTRAL GUARD (memory-health §6.1/§6.3). ACTIVE -> EXPIRED is a
                    # retention-class exit, and a pinned item may never take one — CANONICAL §7.10
                    # "never garbage-collected regardless". Asked here, at the flip itself, and
                    # NOT gated on `respect_pins`: a settings flip must not be able to re-enable
                    # an exit CANONICAL calls unconditional. The `respect_pins` pre-filter above
                    # stays as the cheap skip; this is the authority. An automatic sweep treats
                    # the refusal as "skip, keep", never an error escalation (spec §9 line 381).
                    if not self._policy.permits_retention_exit(item, MemoryState.EXPIRED):
                        continue
                    expired = item.model_copy(deep=True)
                    expired.state = MemoryState.EXPIRED
                    expired.updated_at = now
                    await self._ltm.upsert_fact(expired)
                    outcomes.append(
                        RetentionOutcome(
                            memory_id=item.id, action="self_expire", reason="now_ge_invalid_at"
                        )
                    )
                    newly_dead[item.id] = expired
                    continue  # a self-expired item is not also a COLD-slide candidate this tick

            if (
                item.retention_class is RetentionClass.DURABLE
                and not item.cold
                and item.importance_score <= retention.durable_cold_importance_max
            ):
                inactive_for = now - item.updated_at
                if inactive_for >= timedelta(days=retention.durable_cold_after_d):
                    slid = item.model_copy(deep=True)
                    slid.cold = True
                    await self._ltm.upsert_fact(slid)
                    outcomes.append(
                        RetentionOutcome(
                            memory_id=item.id,
                            action="cold_slide",
                            reason="low_importance_inactive",
                        )
                    )

        # ---- pass 2: dead (SUPERSEDED/EXPIRED) facts — GC, chain-head-dead + window gated --
        dead = await self._ltm_retention.facts_by_state(ns, _DEAD_STATES)
        evaluated += len(dead)
        dead_by_id: dict[str, MemoryItem] = {m.id: m for m in dead}
        # a fact this SAME tick's self-expire pass just killed is GC-eligible in this same
        # tick too (no need to wait a full extra sweep cycle for an overdue EPHEMERAL fact).
        for memory_id, item in newly_dead.items():
            dead_by_id.setdefault(memory_id, item)

        for item in dead_by_id.values():
            if retention.respect_pins and self._is_pinned(item):
                continue
            died_at = item.invalid_at or item.updated_at
            if (now - died_at) < timedelta(days=retention.gc_history_window_d):
                continue
            head_state = await self._ltm_retention.chain_head_state(ns, item.id)
            if head_state is MemoryState.ACTIVE:
                continue  # chain head still alive (spec §9 mermaid) — keep the whole history
            # CENTRAL GUARD (memory-health §6.3; CANONICAL §7.10 "and not item.pinned").
            # Deliberately NOT gated on `retention.respect_pins`: that knob is a cheap pre-filter
            # (above), while GC-ineligibility for a pinned item is unconditional in CANONICAL, so
            # the authoritative refusal lives here where no settings value can lift it. Any other
            # caller reaching `gc_delete` without asking this guard is the defect §6 warns about.
            #
            # `permits_gc`, NOT `permits(item, DELETED)`: the item's carried `state` is NOT
            # trustworthy here. `facts_by_state` rebuilds each row from the node's `memory_json`
            # blob while `expire`/`invalidate` flip only the node's `state` PROPERTY, so a dead
            # row still reads `state=ACTIVE` — and an edge-checking guard would raise
            # `IllegalTransitionError` (ACTIVE -> DELETED is not a legal edge) and abort the whole
            # sweep. See `LifecyclePolicy.assert_retention_exit`; staleness reported separately.
            if not self._policy.permits_retention_exit(item, MemoryState.DELETED):
                continue  # skip, keep — never an error escalation (spec §9 line 381)
            # Mapped BEFORE the irreversible delete, deliberately. ``_to_contract_state`` is
            # exhaustive and raises ``KeyError`` on an unmapped member rather than guessing (see
            # ``_STATE_MAP``). Computing it after ``gc_delete`` would make that loud failure
            # DESTRUCTIVE: the fact would already be gone, its ``MemoryGarbageCollected`` would
            # never publish, and the raise would abort the sweep with the remaining dead facts
            # unprocessed. Mapping first means an unmapped state costs us a refused sweep with
            # nothing deleted — loud AND recoverable, which is the whole point of failing loud.
            prior_state = self._to_contract_state(item.state)
            await self._ltm_retention.gc_delete(ns, item.id)
            outcomes.append(
                RetentionOutcome(
                    memory_id=item.id, action="gc", reason="dead_chain_head_dead_window_elapsed"
                )
            )
            if self._bus is not None:
                await self._bus.publish(
                    MemoryGarbageCollected(namespace=ns, id=item.id, prior_state=prior_state)
                )

        report = RetentionReport(evaluated=evaluated, outcomes=tuple(outcomes))
        _log.info(
            "retention_sweep_complete",
            ns=ns.to_prefix(),
            evaluated=report.evaluated,
            acted_on=report.acted_on,
        )
        return report

    async def reactivate_on_recall(self, item: MemoryItem) -> MemoryItem | None:
        """Reactivate-on-recall (ADR 0035; spec §9's ``COLD -> ACTIVE : recall hit`` edge): the
        recall path — owned by another task; the integrate phase wires this call at whichever
        call site assembles a recall result set — calls this whenever a COLD item (``item.cold
        is True``) surfaces in a hit, flipping ``cold`` back to ``False`` via the existing
        ``upsert_fact`` (never a new port method: this is a plain field mutation + re-MERGE,
        exactly like the EXPIRE/COLD-slide flips above).

        No-op (returns ``None``, zero writes) if the item is not actually COLD or
        ``LifecycleSettings.retention.reactivate_on_recall`` is disabled — mirrors
        ``DemotionService``'s rescue-by-re-evaluation shape: a no-op is silent, never an error.
        """
        if not self._settings.retention.reactivate_on_recall or not item.cold:
            return None
        reactivated = item.model_copy(deep=True)
        reactivated.cold = False
        reactivated.updated_at = self._clock.now()
        await self._ltm.upsert_fact(reactivated)
        _log.info(
            "retention_cold_reactivated",
            ns=item.namespace.to_prefix(),
            memory_id=item.id,
        )
        return reactivated

    @staticmethod
    def _is_pinned(item: MemoryItem) -> bool:
        """CANONICAL §7.10/§7.26/§7.17: ``pinned`` is an ALREADY-canonical field-group
        ("GC-ineligibility keys off ``and not item.pinned``", CANONICAL:621), reused here, not
        re-added. ``pinned: bool`` now lives on the shipped ``MemoryItem``
        (``storage/domain/memory.py``, closed by the §7.17 item 4a total-order task,
        2026-08-24). Kept as ``getattr`` duck typing rather than a direct ``item.pinned`` read —
        harmless now that the field exists, and stays defensive against any OTHER structurally
        similar object this Protocol-typed method might see, defaulting to unpinned (the honest,
        conservative default) rather than raising ``AttributeError``.
        """
        return bool(getattr(item, "pinned", False))

    # ``mu_engine.storage.domain.memory.MemoryState`` (engine) and ``mu_contracts.domain.model.
    # memory.State`` (published) are two independent, un-reconciled definitions of the same
    # canonical axis (flagged debt, ``storage/domain/memory.py``'s own "RE-HOME NOTE"; ADR 0049
    # does not merge them, it only closes the one member gap that was producing a wrong
    # published value). Kept as an explicit, EXHAUSTIVE dict rather than a same-string shortcut
    # (e.g. ``ContractState(state.value)``) on purpose: the two enums are separately maintained,
    # so a future engine-side member landing without its published counterpart must fail LOUDLY
    # here — not silently coerce via string parity, and not fall through to a wrong default the
    # way the pre-ADR-0049 ``ARCHIVED`` fallback did. That silent fallback is exactly what let
    # this defect hide (ADR 0049 "Context").
    #
    # ⚠ The guard is RUNTIME-only, and saying so matters: ``mypy --strict`` does not check dict-key
    # exhaustiveness, so adding a seventh ``MemoryState`` member without a row here type-checks
    # and lints clean, then raises ``KeyError`` on the first sweep that meets one. That is louder
    # than the wrong-value-forever it replaces, but it is NOT a compile-time impossibility — do
    # not read this comment as one. The call site maps BEFORE ``gc_delete`` precisely so that
    # runtime failure costs a refused sweep rather than a deleted fact with no event.
    _STATE_MAP: ClassVar[dict[MemoryState, ContractState]] = {
        MemoryState.ACTIVE: ContractState.ACTIVE,
        MemoryState.ARCHIVED: ContractState.ARCHIVED,
        MemoryState.SUPERSEDED: ContractState.SUPERSEDED,
        MemoryState.QUARANTINED: ContractState.QUARANTINED,
        MemoryState.DELETED: ContractState.DELETED,
        MemoryState.EXPIRED: ContractState.EXPIRED,  # ADR 0049 — the fix: no longer ARCHIVED
    }

    @staticmethod
    def _to_contract_state(state: MemoryState) -> ContractState:
        """Map the engine's internal lifecycle state onto the published event-catalog ``State``
        (CANONICAL §7.5's SIX-member axis, ADR 0049) for ``MemoryGarbageCollected.prior_state``.

        EXHAUSTIVE over ``_STATE_MAP``: every current ``MemoryState`` member has an explicit,
        named published counterpart — most saliently ``EXPIRED -> EXPIRED``, since before ADR
        0049 the published enum lacked that member and this method substituted ``ARCHIVED``,
        making every GC event for a self-expired EPHEMERAL fact report a death that never
        happened (ADR 0049 "Context"). A ``MemoryState`` with no entry raises ``KeyError``
        rather than guessing — see the class-level comment on ``_STATE_MAP`` for why that's the
        deliberate choice now that the two enums' members line up 1:1.
        """
        return RetentionService._STATE_MAP[state]
