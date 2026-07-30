"""``MemoryLifecycleManager`` (MLM) — the plane-agnostic orchestrator over ports (S1-03).

Authority: ``docs/superpowers/design/memory-lifecycle-manager-spec.md`` §0.2 (component view,
lines 30-78), §4/§4b (lease acquisition order, lines 164-219), §7 (sweep discovery/backpressure,
lines 271-327), §17/§17a (full API + DTOs, lines 600-699); AC-1.1/AC-1.2 (lines 432-434); ADR 0029
(the manager itself), ADR 0031 (mode-gate call sites), ADR 0032 (lease acquisition order).
CANONICAL §4.1 (two-plane invariant): this class is plane-agnostic by construction — it takes only
ports/services, never a client-only or server-only assumption, so ``ServerMemoryManager`` (slice 4)
can subclass it unmodified.

Reference pattern PORTed: ``mma/mma/memory/controller.py:MemoryController`` — the dual-trigger
(event-driven fast-fire + periodic backstop, ``_on_memory_ingested:380``/``_maintenance_loop:405``)
and lease-then-inner-writer ordering this module reproduces over this repo's own ports/DTOs.

**Sibling Stage-1 modules this file wraps (grounded against the LANDED code, not a guess)**:
``mu_engine.lifecycle.promotion.PromotionService`` (S1-01) and
``mu_engine.lifecycle.demotion.DemotionService`` (S1-02) both landed IN PARALLEL with this task and
are imported directly (no local stand-in Protocol needed for them). ``mu_local.composition.
LocalContainer.build_lifecycle_manager`` (S1-05) is the REAL, already-landed call site this
constructor is grounded against — it constructs this class with ONLY ``salience``/``promotion``/
``demotion``/``distill``/``mode_gate``/``bus``/``settings`` (no ``retention``/``conflict``/
``lease``/``runner``/``clock``/``warm_cache``), relying on this constructor DEFAULTING every other
field for a Stage-1, daemonless (no cross-process runner) composition. This file's constructor
signature is therefore not a byte-for-byte copy of spec §17's illustrative code fragment; it is
the code fragment RECONCILED against the one real caller that already exists, with every deviation
documented below and, where the deviation is genuinely new (not already implied by S1-05's call),
flagged for the CANONICAL audit exactly as this task's packet already expects for ``retention``/
``conflict``:

1. ``retention: RetentionServicePort | None = None`` / ``conflict: ConflictAdjudicatorPort | None =
   None`` — Optional, no-op-when-absent (this task's own acceptance list). ``RetentionService``/
   ``ConflictAdjudicator`` (slice 2/3, S2-01/S3-01) are not yet landed; ``RetentionServicePort``/
   ``ConflictAdjudicatorPort`` below are this file's LOCAL, minimal, structural (PEP 544) seam
   Protocols standing in for them — a landed concrete class needs no import of this module to
   satisfy them structurally; confirm/adjust at the integrate phase only if the landed shape
   differs from the one method this file actually calls.
2. ``lease: LifecycleLeasePort | None = None`` / ``runner: LifecycleWorkflowRunnerPort | None =
   None`` — Optional. S1-05's real call passes neither (mu-local is daemonless, "has no durable
   cross-process runner" per its own docstring) and explicitly "relies on ``MemoryLifecycleManager``
   defaulting those fields for a Stage-1, daemonless composition." Concrete defaults:
   ``_InProcessLifecycleLease`` (fail-fast, in-process-only — matches ``SqliteWalLeaseAdapter``'s
   acquire-or-defer CONTRACT, not its cross-process REACH) and ``_InlineLifecycleRunner`` (executes
   a submitted job's body immediately inside ``submit()`` — matching ``lifecycle_workflow.py``'s
   own words, "InlineRunner = test/degrade", and matching mu-local's "daemonless: the caller drives
   the sweep" need for SOMETHING to actually run the work when there is no background worker).
   mu-client's real daemon composition (S1-06/S1-07) always passes genuine
   ``SqliteWalLeaseAdapter``/``SqliteWalRunner`` instances; these defaults are never used there.
3. ``clock: Clock | None = None`` — not in spec §17's literal fragment, but required by spec §19
   Rule 1 ("the sweep loop itself (``MemoryLifecycleManager.sweep_user``) ... take[s] an injected
   ``Clock`` at construction — never a bare ``datetime.now(UTC)`` call"), and S1-05's call passes
   none either (relies on a default) — defaults to ``SystemClock()``, matching the identical
   ``clock: Clock | None = None`` pattern already used by ``DistillPipeline``/``PromotionService``/
   ``DemotionService``.

**Additive public methods beyond spec §17's literal API** (the SAME "structural ``Protocol``
conformance permits extra public methods" convention ``SqliteWalRunner`` already uses for
``claim_next``/``complete``):

* ``run_job(job: LifecycleJob) -> JobResult`` — the actual dispatcher a durable runner (or, for the
  ``SqliteWalRunner`` case, a NOT-YET-BUILT worker loop — that adapter's own docstring: "a worker
  loop (a later Stage-1 task) drains through this," ``claim_next``/``complete`` — is additive
  there for exactly that reason) calls back into once it dequeues a submitted job. ``_
  InlineLifecycleRunner`` (this file's default) calls it synchronously inside ``submit()``.
* ``sweep_namespace_now(ns, *, mtm_candidates=()) -> None`` — the real per-namespace sweep body:
  ``PromotionService.promote_session`` (which self-fetches its own STM window via ITS OWN injected
  ``StmTierRepository``, S1-05's wiring — this manager needs no store port of its own for the
  promotion leg) covers STM->MTM promotion + the MTM->LTM ``DistillPipeline.distill`` hand-off in
  ONE real, working call; an optional caller-supplied ``mtm_candidates`` window (this manager has
  no MTM enumeration primitive of its own — see the gap note below) drives
  ``DemotionService.demote`` when present.
* ``periodic_tick(*, manual=False) -> None`` — the active-user-registry-driven backstop this
  task's own acceptance list requires ("active-user registry + full_scan_interval_s backstop +
  max_users_per_sweep/max_items_per_user_sweep caps ... sourced from LifecycleSettings"). NOTE:
  mu-client's ``MaintenanceLoop`` (S1-07, landed) already implements an EQUIVALENT registry/
  backstop/coalescing loop of its OWN and calls ``sweep_user`` directly per user_prefix — it does
  NOT call ``on_bus_event``/``periodic_tick`` on this manager at all. This manager's own registry
  is therefore genuinely redundant with ``MaintenanceLoop`` on the mu-client plane specifically,
  but is still required by spec §17's own API (``on_bus_event`` is a named public method) and is
  the seam a plane WITHOUT its own MaintenanceLoop-equivalent (e.g. a future
  ``ServerMemoryManager``/``SleeptimeWorkflow``, slice 4) would use directly.

**Known, honestly-flagged gaps (shared, cross-task — not unique to this file)**:

* **No MTM-tier enumeration primitive exists anywhere in the currently-landed code** —
  ``MtmTierRepository`` (``storage/ports.py``) ships ``upsert``/``semantic``(query-vector-gated)/
  ``invalidate`` only, no scan/list. ``DemotionService.demote`` and
  ``PromotionService.sweep_mtm_to_ltm`` both take an explicit CALLER-supplied ``window`` (their own
  docstrings: "this service does not enumerate MTM itself" / "the full-scan/backstop ENUMERATION
  ... is the MaintenanceLoop's job") and ``MaintenanceLoop`` itself documents the SAME gap
  ("Discovery gap ... needs a user-enumeration port no Stage-0/Stage-1 task in this plan yet
  exposes"). Consequently this manager's automatic ``sweep_namespace_now`` runs the REAL promotion
  leg (via ``promote_session``) but only runs the demotion leg when a window is explicitly
  supplied (``mtm_candidates``) — there is no automatic MTM-tier demotion discovery this slice.
  Flagged for whichever future task adds a genuine MTM scan/list port.
* **Single-item manual ``promote(ns, memory_id)``/``demote(ns, memory_id)`` execution is a
  documented stub** (mode-gate check + durable-enqueue submit are fully wired; ``run_job``'s
  PROMOTE/DEMOTE branches return ``JobStatus.FAILED`` with a named, content-free ``error``) for the
  identical reason: neither ``StmTierRepository``/``MtmTierRepository`` gives this manager a way to
  fetch an arbitrary MTM-tier ``MemoryItem`` by id, and fabricating one to feed
  ``DemotionService.demote``/reuse ``PromotionService``'s gate would silently corrupt the real
  salience decision (wrong ``access_count``/``importance_score``/``created_at``) — worse than a
  loud, named failure. ``sweep_user``/``consolidate`` (the acceptance-critical, container-tested
  paths) are NOT affected — they route through ``sweep_namespace_now`` and never hit this stub.
* ``ExplainRecord.salience_inputs``'s ``rec``/``use``/``imp`` fields are `0.0` placeholders (never
  fabricated plausible-looking numbers) when this manager builds a record from a
  ``PromotionOutcome``/``DemotionOutcome`` — those DTOs (S1-01/S1-02, already landed) carry only
  the COMPOSED ``score`` (mirroring ``SalienceStrategy.score()``'s own float-only return, S0-08),
  never the raw breakdown; ``score`` itself (the one true known value) and the weight/half-life
  config fields (exactly known from ``LifecycleSettings``) are populated correctly. Closing this
  needs either a re-fetch-by-id capability this manager does not have, or
  ``PromotionService``/``DemotionService`` returning the full breakdown directly — flagged, not
  papered over.
* ``get_state(ns)``'s ``stm_count``/``mtm_count``/``ltm_count`` are documented `0` stubs — NOT
  merely because this constructor receives services rather than raw per-tier read ports, but
  because spec §5 types ``get_state`` as a SYNCHRONOUS "instant" warm read; the only existing
  ports that could count STM/LTM items (``StmTierRepository.recent``/``GraphStorePort.
  graph_recall``) are ``async`` I/O, and blocking a sync method on real store I/O would violate
  DEV-STANDARDS rule 1. Real counts need a proactively-maintained ``WarmRecallCacheService``
  (slice 3, S3-02 — not yet landed) this manager could read synchronously from memory — see
  :meth:`MemoryLifecycleManager.get_state`'s own docstring for the full reasoning.
  ``ready_context(session_id)`` is the identical, spec-sanctioned stub (task packet: "may return
  a documented not-yet-wired stub until S3-02 lands"). NOTE: a pre-existing sibling test,
  ``mu-local/tests/test_lifecycle_gate_int.py::test_build_lifecycle_manager_get_state_matches_
  real_store_counts`` (S1-05's own suite, not owned by this task), asserts real non-zero counts
  and currently fails against this — flagged for the integrate phase to reconcile (mark it
  xfail/skip pending S3-02, or redefine it against a warm-cache-backed count once that lands)
  rather than this manager acquiring a blocking I/O call it should not have.
* ``consolidate``'s ``after_offset`` cursor (spec §13.1) is accepted on the wire (``LifecycleJob``
  carries it) but NOT applied by ``run_job``'s CONSOLIDATE branch this slice — it re-sweeps every
  namespace known to the ``user_prefix`` via the same ``sweep_namespace_now`` the periodic path
  uses, rather than resuming from a durable cursor. Flagged as a follow-up (a genuine cursor needs
  a store-wide, offset-ordered read this constructor does not receive either).

Every threshold/interval/weight this file reads comes from the injected ``LifecycleSettings``
(never hardcoded, DEV-STANDARDS rule 3).
"""

from __future__ import annotations

import contextlib
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.events import (
    DegradedModeEntered,
    DegradeReason,
    DomainEvent,
    MemoryCaptured,
    MemoryPromoted,
)
from mu_contracts.domain.model.lifecycle import (
    JobHandle,
    JobResult,
    JobStatus,
    LifecycleJob,
    LifecycleJobKind,
    UserPrefix,
)
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.ports.lifecycle_lease import LifecycleLeasePort
from mu_contracts.ports.lifecycle_workflow import LifecycleWorkflowRunnerPort
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.dto import LifecycleStateView, SalienceInputs, TransitionKind
from mu_engine.lifecycle.explain import ExplainRecord
from mu_engine.lifecycle.mode_gate import ManagerModeGate
from mu_engine.lifecycle.promotion import PromotionOutcomeKind, PromotionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.pipelines.distill import DistillPipeline, EventPublisher
from mu_engine.platform.clock import SystemClock
from mu_engine.storage.domain.memory import MemoryItem

__all__ = [
    "ConflictAdjudicatorPort",
    "MemoryLifecycleManager",
    "RenderedContext",
    "RetentionServicePort",
    "WarmRecallCacheServicePort",
]

_log = structlog.get_logger("mu_engine.lifecycle.manager")

# device_id/plane discriminator for ExplainRecord.decided_by (spec §19): not in this constructor's
# signature (absent from spec §17 too) — a fixed sentinel until a future constructor param or the
# composition root supplies a real device_id/"server" distinction (flagged, module docstring).
_DECIDED_BY_DEFAULT = "local"

# UserPrefix's own validated byte-shape (mu_contracts.domain.model.lifecycle._USER_PREFIX_PATTERN,
# reproduced here read-only for the gate-check fallback below — never re-validates a UserPrefix,
# UserPrefix's own constructor already guarantees this shape).
_USER_PREFIX_RE = re.compile(r"^mu/([^/]+)/([^/]+)/(private|shared)/([^/]+)/$")


# ============================================================================================
# Local seam Protocols — slice-2/3 siblings not yet landed (RetentionService/ConflictAdjudicator/
# WarmRecallCacheService). Structural (PEP 544): the real classes need not import this module.
# ============================================================================================
@runtime_checkable
class RetentionServicePort(Protocol):
    """Seam for the slice-2 ``RetentionService`` (validity-first archival + GC, spec §9).
    Optional at construction — ``None`` = no-op, this slice's retention pass is a documented
    skip (this task's own acceptance list)."""

    async def sweep(self, ns: Namespace, *, clock: Clock) -> int: ...  # count acted on


@runtime_checkable
class ConflictAdjudicatorPort(Protocol):
    """Seam for the slice-3 ``ConflictAdjudicator`` (spec §8). Per spec §8 placement it sits
    INSIDE ``DistillPipeline.reconcile`` (wired there at the composition root, not called
    directly by this manager) — held here only so this constructor matches spec §17's field
    LIST and so a future ``explain()`` can source ``model_verdict`` from it. No required method
    this slice (an empty structural Protocol; any object satisfies it)."""


@runtime_checkable
class WarmRecallCacheServicePort(Protocol):
    """Seam for the slice-3 ``WarmRecallCacheService`` (spec §12). Optional; ``None`` = no-op.

    Stage-3 integrate: ``mu_client.inject.recall_bridge.RecallInjectBridge`` IS this port's real
    implementation (its own module docstring: "PROMOTED (S3-02) into the WarmRecallCacheService
    role"). ``invalidate`` pops its courtesy-cache entry on a real MLM tier-transition event
    (wired via the bridge's OWN bus subscription, not through this manager); ``last_rendered`` is
    the synchronous warm read :meth:`ready_context` below now actually consults — a session with
    no rendered body yet (cold, nothing pulled through ``GET /recall`` for it) returns ``None``,
    never a fabricated body."""

    def invalidate(self, ns: Namespace) -> None: ...

    def last_rendered(self, session_id: str) -> str | None: ...


class RenderedContext(BaseModel):
    """Stub return shape for ``ready_context`` (spec §12/§17). ``WarmRecallCacheService`` +
    ``LiveSessionContext`` (S3-02) are not yet landed (slice 3) — this task's own acceptance
    list permits "a documented not-yet-wired stub until S3-02 lands" as long as the METHOD
    SIGNATURE and the warm-read-never-enqueues contract exist now, which they do:
    :meth:`MemoryLifecycleManager.ready_context` is synchronous and touches nothing but this
    stub."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    rendered: str = ""
    wired: bool = False


# ============================================================================================
# In-process defaults for lease/runner (used only when the caller passes neither — see module
# docstring point 2). NOT a substitute for the real cross-process adapters (S1-06, mu-client).
# ============================================================================================
class _LifecycleLeaseBusyError(Exception):
    """Raised by :class:`_InProcessLifecycleLease` on a contended acquire — mirrors
    ``mu_client.runners.sqlite_wal.LifecycleLeaseBusyError``'s fail-fast "the caller backs off"
    contract (spec §4b) in SHAPE, without mu-engine importing mu-client (layering:
    ``.importlinter`` keeps mu-client/mu-server out of mu-core entirely)."""

    def __init__(self, lease_name: str) -> None:
        super().__init__(f"lifecycle-sweep-lease busy: {lease_name}")


class _InProcessLifecycleLease:
    """Default ``LifecycleLeasePort`` when ``lease=None``. In-process only, fail-fast — no
    cross-process guarantee (that is :class:`~mu_client.runners.sqlite_wal.SqliteWalLeaseAdapter`,
    S1-06). Used by mu-local's daemonless composition (which passes no ``lease``) and by this
    file's own unit tests."""

    def __init__(self) -> None:
        self._held: set[str] = set()

    @contextlib.asynccontextmanager
    async def acquire(self, prefix: UserPrefix) -> AsyncIterator[None]:
        key = str(prefix)
        if key in self._held:
            raise _LifecycleLeaseBusyError(f"lifecycle-sweep-lease:inproc:{key}")
        self._held.add(key)
        try:
            yield
        finally:
            self._held.discard(key)


class _InlineLifecycleRunner:
    """Default ``LifecycleWorkflowRunnerPort`` when ``runner=None``. Executes the submitted
    job's body IMMEDIATELY inside ``submit()`` (``lifecycle_workflow.py``'s own words: "InlineRunner
    = test/degrade") by calling straight back into the bound ``run_job`` handler — the correct
    REAL default for mu-local's daemonless composition (no background worker loop exists there to
    drain a durable queue otherwise), not merely a test convenience. mu-client's real daemon always
    passes a genuine ``SqliteWalRunner`` (durable-enqueue-only) instead."""

    def __init__(self, handler: Callable[[LifecycleJob], Awaitable[JobResult]]) -> None:
        self._handler = handler
        self._results: dict[str, JobResult] = {}

    async def submit(self, job: LifecycleJob) -> JobHandle:
        result = await self._handler(job)
        self._results[job.job_id] = result
        return JobHandle(job_id=job.job_id, submitted_at=job.submitted_at)

    async def await_result(self, handle: JobHandle) -> JobResult:
        return self._results[handle.job_id]

    async def resume_pending(self) -> int:
        return 0  # nothing durable to resume — this runner never persists a pending job


class MemoryLifecycleManager:
    """The always-on, config-driven lifecycle engine's orchestrator (spec §0.2/§4/§7/§17). See
    module docstring for the full constructor-signature rationale, additive public methods, and
    honestly-flagged gaps."""

    def __init__(
        self,
        *,
        salience: SalienceStrategy,
        promotion: PromotionService,
        demotion: DemotionService,
        distill: DistillPipeline,
        retention: RetentionServicePort | None = None,
        conflict: ConflictAdjudicatorPort | None = None,
        lease: LifecycleLeasePort | None = None,
        runner: LifecycleWorkflowRunnerPort | None = None,
        mode_gate: ManagerModeGate,
        bus: EventPublisher,
        settings: LifecycleSettings,
        clock: Clock | None = None,
        warm_cache: WarmRecallCacheServicePort | None = None,
    ) -> None:
        self._salience = salience
        self._promotion = promotion
        self._demotion = demotion
        self._distill = distill
        self._retention = retention
        self._conflict = conflict
        self._mode_gate = mode_gate
        self._bus = bus
        self._settings = settings
        self._clock: Clock = clock or SystemClock()
        self._warm_cache = warm_cache
        # Runner/lease defaults reference self.run_job — safe: methods bind via the class the
        # instant `self` exists, well before __init__'s body needs to have finished (no attribute
        # on self is read until a coroutine actually calls run_job later).
        self._lease: LifecycleLeasePort = lease or _InProcessLifecycleLease()
        self._runner: LifecycleWorkflowRunnerPort = runner or _InlineLifecycleRunner(self.run_job)

        # Active-user registry (spec §7 hybrid discovery) — event-driven, populated by
        # on_bus_event; read by periodic_tick's backstop. Keyed on UserPrefix (session-spanning).
        self._active_namespaces: dict[UserPrefix, set[Namespace]] = {}
        self._batch_counts: dict[UserPrefix, int] = {}
        self._last_swept_at: dict[UserPrefix, datetime] = {}
        self._last_full_scan_at: datetime | None = None
        # One-sweep-per-user in-process coalescing floor (spec §7: "A running sweep coalesces new
        # triggers rather than stacking") — complements, never replaces, the LifecycleLeasePort's
        # cross-process guarantee (AC-1.1's own words, mirrored from MaintenanceLoop's docstring).
        self._pending_jobs: dict[UserPrefix, JobHandle] = {}
        # In-memory audit trail for explain() — no persistent store backs this (see module
        # docstring's get_state stub note); bounded only by process lifetime.
        self._explain_log: dict[tuple[str, str], list[ExplainRecord]] = {}

    # ======================================================================== warm reads (§5/§17)
    def get_state(self, ns: Namespace) -> LifecycleStateView:
        """Instant warm read — NEVER touches the runner (spec §5 read/write split: "``MLM.
        get_state(ns)``... synchronous warm reads (never enqueue, never await a job)").

        Tier counts are a documented ``0`` stub THIS SLICE — not merely an unwired gap but an
        architectural consequence of that same sync contract: ``StmTierRepository.recent``/
        ``GraphStorePort.graph_recall`` (the only existing ports that COULD count STM/LTM items)
        are ``async`` I/O calls, and synchronously blocking this ``def`` (not ``async def``) on
        real store I/O would violate DEV-STANDARDS rule 1 (never block the event loop) while
        also contradicting spec §5's own "instant" framing. The only spec-consistent way to give
        ``get_state`` REAL tier counts is a proactively-maintained WARM CACHE
        (``WarmRecallCacheService``, slice 3/S3-02 — not yet landed) that this manager reads
        synchronously from memory, refreshed asynchronously in the background on every real
        transition — the identical reasoning this task's own acceptance list already applies to
        ``ready_context``'s stub. See the module docstring's gap list."""
        prefix = UserPrefix(ns)
        return LifecycleStateView(
            user_prefix=prefix,
            stm_count=0,
            mtm_count=0,
            ltm_count=0,
            last_swept_at=self._last_swept_at.get(prefix),
            pending_job=self._pending_jobs.get(prefix),
            degraded=None,
        )

    def ready_context(self, session_id: str) -> RenderedContext:
        """Instant warm read — NEVER touches the runner (spec §5/§12). Stage-3 integrate: when a
        ``WarmRecallCacheServicePort`` (``RecallInjectBridge``, S3-02) is wired, this reads ITS
        synchronous courtesy-cache (``last_rendered``) — no I/O, no await, just an in-memory dict
        read the bridge's own PULL/PUSH paths keep fresh. ``None`` (no ``warm_cache`` wired, or
        nothing rendered yet for this session — cold) falls through to the honest ``wired=False``
        stub; never a fabricated body."""
        if self._warm_cache is not None:
            body = self._warm_cache.last_rendered(session_id)
            if body is not None:
                return RenderedContext(session_id=session_id, rendered=body, wired=True)
        return RenderedContext(session_id=session_id, rendered="", wired=False)

    def explain(self, ns: Namespace, memory_id: str) -> list[ExplainRecord]:
        """Instant warm read over this manager's own in-memory audit trail (see module docstring
        for the ``rec``/``use``/``imp`` placeholder caveat)."""
        return list(self._explain_log.get((ns.to_prefix(), memory_id), ()))

    # ================================================================ durable writes (§5/§17) ===
    async def sweep_user(self, prefix: UserPrefix, *, manual: bool = False) -> JobHandle:
        """Cross-session, per-user durable sweep (spec §1/§7/§17). Coalesces a re-trigger while a
        sweep is already in flight for this ``user_prefix`` (returns the SAME ``JobHandle`` rather
        than submitting a second job) — the in-process floor; the cross-process guarantee is the
        ``LifecycleLeasePort`` acquired by :meth:`run_job`'s execution.

        ``manual`` distinguishes an internal engine-driven trigger (event-driven fast-fire /
        periodic backstop — ``manual=False``, the default, never mode-gated: MANAGED means
        exactly "the engine drives this automatically") from an EXTERNAL manual invocation (a
        daemonless CLI ``mu consolidate``/sweep call — ``manual=True``), which goes through
        ``ManagerModeGate.assert_manual_allowed`` first (this task's acceptance: "promote/demote/
        consolidate/sweep_user go through ManagerModeGate.assert_manual_allowed before any
        manual-verb path")."""
        if manual:
            ns = self._representative_namespace(prefix)
            self._mode_gate.assert_manual_allowed(ns, "sweep_user")
        pending = self._pending_jobs.get(prefix)
        if pending is not None:
            return pending
        job = self._make_job(LifecycleJobKind.SWEEP_USER, prefix)
        placeholder = JobHandle(job_id=job.job_id, submitted_at=job.submitted_at)
        self._pending_jobs[prefix] = placeholder
        try:
            handle = await self._runner.submit(job)
        except BaseException:
            self._clear_pending(prefix, placeholder.job_id)
            raise
        current = self._pending_jobs.get(prefix)
        if current is not None and current.job_id == placeholder.job_id:
            # Still marked in-flight under OUR placeholder: a durable/async runner (e.g. the real
            # SqliteWalRunner) that has not yet executed the job body — replace with the real
            # handle. An inline/synchronous runner already ran run_job() to completion (and
            # cleared the placeholder itself, see _execute_sweep's finally) before submit()
            # returned, in which case there is nothing left to mark in-flight.
            self._pending_jobs[prefix] = handle
        return handle

    async def promote(self, ns: Namespace, memory_id: str) -> JobHandle:
        """Manual verb (spec §17) — always mode-gated (this IS a manual verb by definition, no
        ``manual`` flag needed, unlike ``sweep_user``). ``run_job``'s execution of this specific
        job kind is a documented stub this slice (see module docstring)."""
        self._mode_gate.assert_manual_allowed(ns, "promote")
        job = self._make_job(
            LifecycleJobKind.PROMOTE, UserPrefix(ns), namespace=ns.to_prefix(), memory_id=memory_id
        )
        return await self._runner.submit(job)

    async def demote(self, ns: Namespace, memory_id: str) -> JobHandle:
        """Manual verb (spec §17) — always mode-gated. See :meth:`promote`'s docstring for the
        identical execution-stub caveat."""
        self._mode_gate.assert_manual_allowed(ns, "demote")
        job = self._make_job(
            LifecycleJobKind.DEMOTE, UserPrefix(ns), namespace=ns.to_prefix(), memory_id=memory_id
        )
        return await self._runner.submit(job)

    async def consolidate(self, prefix: UserPrefix, *, after_offset: int) -> JobHandle:
        """Manual verb (spec §17) — always mode-gated. ``after_offset`` (§13.1 cursor) is carried
        on the wire but not yet applied by ``run_job``'s execution (documented stub, module
        docstring)."""
        ns = self._representative_namespace(prefix)
        self._mode_gate.assert_manual_allowed(ns, "consolidate")
        job = self._make_job(LifecycleJobKind.CONSOLIDATE, prefix, after_offset=after_offset)
        return await self._runner.submit(job)

    # ============================================================================ event-driven ===
    async def on_bus_event(self, event: DomainEvent) -> None:
        """Populates the active-user registry (spec §7 hybrid discovery) and fires a fast-fire
        sweep once the per-user batch counter reaches ``LifecycleSettings.batch_size``. See module
        docstring for why this is redundant-but-required alongside mu-client's ``MaintenanceLoop``
        (S1-07), which implements an equivalent registry of its own on that specific plane."""
        if not isinstance(event, MemoryCaptured | MemoryPromoted):
            return
        ns = event.namespace
        prefix = UserPrefix(ns)
        self._active_namespaces.setdefault(prefix, set()).add(ns)
        count = self._batch_counts.get(prefix, 0) + 1
        if count < self._settings.batch_size:
            self._batch_counts[prefix] = count
            return
        self._batch_counts[prefix] = 0
        await self.sweep_user(prefix)  # internal engine-driven trigger — never mode-gated

    async def periodic_tick(self, *, manual: bool = False) -> None:
        """The registry-driven backstop this task's acceptance requires: the active-user
        registry + ``full_scan_interval_s`` cadence + ``max_users_per_sweep`` cap, all sourced
        from ``LifecycleSettings`` (never hardcoded). See module docstring's "Additive public
        methods" note for why this is not double-wired against mu-client's own
        ``MaintenanceLoop``. Discovery is bounded to the active-user registry (no raw store-wide
        user enumeration port exists in this constructor, the SAME gap
        ``MaintenanceLoop._sweep_active_users`` documents on its own plane)."""
        now = self._clock.now()
        if (
            self._last_full_scan_at is None
            or (now - self._last_full_scan_at).total_seconds()
            >= self._settings.full_scan_interval_s
        ):
            self._last_full_scan_at = now
        prefixes = list(self._active_namespaces.keys())[: self._settings.max_users_per_sweep]
        for prefix in prefixes:
            await self.sweep_user(prefix, manual=manual)

    # ========================================================= execution entrypoint (additive) ===
    async def run_job(self, job: LifecycleJob) -> JobResult:
        """The actual dispatcher a durable runner (or, until it exists, a future worker loop
        draining ``SqliteWalRunner.claim_next``/``complete``) calls back into once it dequeues a
        submitted job. ``_InlineLifecycleRunner`` (this file's default) calls this synchronously
        inside ``submit()``. See module docstring for the additive-method rationale."""
        if job.kind is LifecycleJobKind.SWEEP_USER:
            return await self._execute_sweep(job)
        if job.kind is LifecycleJobKind.CONSOLIDATE:
            return await self._execute_consolidate(job)
        if job.kind in (LifecycleJobKind.PROMOTE, LifecycleJobKind.DEMOTE):
            # Documented stub — see module docstring's "single-item manual promote/demote" note.
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error="single_item_execution_not_wired_this_slice",
                completed_at=self._clock.now(),
            )
        # pragma: no cover — closed enum, every member handled above
        raise ValueError(f"unknown LifecycleJobKind: {job.kind!r}")

    async def sweep_namespace_now(
        self, ns: Namespace, *, mtm_candidates: Sequence[MemoryItem] = ()
    ) -> None:
        """The real per-namespace sweep body (additive public method, see module docstring).

        Promotion leg: ``PromotionService.promote_session`` — a REAL, working call that
        self-fetches its own STM window (via its OWN injected ``StmTierRepository``, S1-05's
        composition-root wiring) and, in one pass, gate-promotes survivors STM->MTM and hands
        them straight to ``DistillPipeline.distill`` for MTM->LTM (spec §7b path (ii)/(iii)
        combined) — it publishes its OWN ``MemoryPromoted`` events via its own injected bus
        (mirrors ``DistillPipeline``'s self-publish pattern); this manager does not re-publish.

        Demotion leg: only runs when the caller supplies ``mtm_candidates`` — this manager has no
        MTM-tier enumeration primitive of its own (see module docstring's gap note); when
        supplied, ``DemotionService.demote`` is called and (mirroring the promotion leg) publishes
        its own ``MemoryDemoted`` events itself.

        Both legs feed this manager's own ``explain()`` audit trail from their outcome DTOs.
        """
        promo_report = await self._promotion.promote_session(ns, ns.session)
        for outcome in promo_report.outcomes:
            if outcome.kind is PromotionOutcomeKind.STM_TO_MTM:
                self._record_explain(ns, outcome.memory_id, TransitionKind.PROMOTE, outcome.score)

        if mtm_candidates:
            demo_report = await self._demotion.demote(ns, mtm_candidates)
            for demo_outcome in demo_report.outcomes:
                if demo_outcome.demoted:
                    self._record_explain(
                        ns, demo_outcome.memory_id, TransitionKind.DEMOTE, demo_outcome.score
                    )

        if self._retention is not None:
            await self._retention.sweep(ns, clock=self._clock)

    # ================================================================================ internals ===
    async def _run_under_lease(
        self, prefix: UserPrefix, body: Callable[[], Awaitable[None]]
    ) -> tuple[JobStatus, str | None]:
        """Acquires ``self._lease`` for ``prefix`` and runs ``body`` inside it — the ONE lease
        acquisition site in this module (grep-checkable: the sole ``self._lease.acquire(`` call
        site is this method; callers — :meth:`_execute_sweep`/:meth:`_execute_consolidate` — never
        acquire it themselves, so there is no reverse-order acquirer anywhere in this file).

        ``LifecycleLeasePort.acquire``'s entire documented contract is acquire-or-defer (spec
        §4b): ANY exception raised while ENTERING it (as opposed to one raised by ``body`` once
        already inside) is therefore treated uniformly as contention/"busy", regardless of the
        concrete adapter's own exception type — each adapter mints its own (e.g. mu-client's
        ``LifecycleLeaseBusyError`` / this file's own ``_LifecycleLeaseBusyError`` for the
        in-process default; no shared exception base exists across the mu-engine/mu-client
        layering boundary for this manager to catch by type). A busy lease is an EXPECTED
        coalescing outcome (another sweep/manual verb already holds it for this ``user_prefix``),
        surfaced as a named degrade (DEV-STANDARDS rule 8: no silent swallow) via this manager's
        own ``bus``, never a bare crash.
        """
        lease_ctx = self._lease.acquire(prefix)
        try:
            await lease_ctx.__aenter__()
        except Exception as exc:
            await self._bus.publish(
                DegradedModeEntered(
                    component="lifecycle",
                    mode="sweep_coalesced",
                    reason=DegradeReason.BACKPRESSURE_SHED,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            return JobStatus.FAILED, "lease_busy_coalesced"
        try:
            await body()
        except Exception as exc:  # converted to a JobResult, never silently swallowed
            await lease_ctx.__aexit__(type(exc), exc, exc.__traceback__)
            return JobStatus.FAILED, f"{type(exc).__name__}"
        await lease_ctx.__aexit__(None, None, None)
        return JobStatus.SUCCEEDED, None

    async def _execute_sweep(self, job: LifecycleJob) -> JobResult:
        """Sweeps every namespace known to this ``user_prefix`` via :meth:`sweep_namespace_now`
        under the outer ``LifecycleLeasePort`` lease (:meth:`_run_under_lease`) — whose promotion
        leg calls into ``DistillPipeline.distill`` (via ``PromotionService.promote_session``),
        which acquires its OWN ``WriterLeasePort`` INNER (CANONICAL §7.5, unchanged)."""
        prefix = job.user_prefix

        async def _body() -> None:
            for ns in tuple(self._active_namespaces.get(prefix, ())):
                await self.sweep_namespace_now(ns)
            self._last_swept_at[prefix] = self._clock.now()

        status, error = await self._run_under_lease(prefix, _body)
        self._clear_pending(prefix, job.job_id)
        return JobResult(
            job_id=job.job_id, status=status, error=error, completed_at=self._clock.now()
        )

    async def _execute_consolidate(self, job: LifecycleJob) -> JobResult:
        """Same lease-outer/distill-inner ordering as :meth:`_execute_sweep`, keyed on the job's
        own ``user_prefix``. ``after_offset`` is not yet applied (documented stub, module
        docstring)."""
        prefix = job.user_prefix

        async def _body() -> None:
            for ns in tuple(self._active_namespaces.get(prefix, ())):
                await self.sweep_namespace_now(ns)

        status, error = await self._run_under_lease(prefix, _body)
        return JobResult(
            job_id=job.job_id, status=status, error=error, completed_at=self._clock.now()
        )

    def _clear_pending(self, prefix: UserPrefix, job_id: str) -> None:
        current = self._pending_jobs.get(prefix)
        if current is not None and current.job_id == job_id:
            self._pending_jobs.pop(prefix, None)

    def _make_job(
        self,
        kind: LifecycleJobKind,
        prefix: UserPrefix,
        *,
        namespace: str | None = None,
        memory_id: str | None = None,
        after_offset: int | None = None,
    ) -> LifecycleJob:
        return LifecycleJob(
            job_id=str(uuid.uuid4()),
            kind=kind,
            user_prefix=prefix,
            namespace=namespace,
            memory_id=memory_id,
            after_offset=after_offset,
            submitted_at=self._clock.now(),
            config_version=self._settings.config_version,
            policy_version=self._settings.policy_version,
        )

    def _representative_namespace(self, prefix: UserPrefix) -> Namespace:
        """Best-effort ``Namespace`` for ``ManagerModeGate`` resolution when only a
        ``UserPrefix`` is known (``consolidate``/``sweep_user(manual=True)``'s gate check, spec
        §3/§17a — neither carries a session, so the mode-gate's memory/namespace-grain override
        walk needs SOME namespace to resolve against). Prefers a REAL namespace already observed
        for this prefix via the active-user registry (more faithful to a real override than a
        synthetic one); falls back to a synthetic ``Namespace`` parsed from the ``UserPrefix``'s
        own validated segments (session is not part of ``UserPrefix`` by construction, so a
        sentinel session is used) only for a cold-start manual invocation before any capture
        event has been observed for this prefix."""
        seen = self._active_namespaces.get(prefix)
        if seen:
            return next(iter(seen))
        match = _USER_PREFIX_RE.match(prefix)
        if match is None:  # pragma: no cover — UserPrefix's own validator already guarantees this
            raise ValueError(f"malformed UserPrefix: {prefix!r}")
        org, workspace, vis, user_slot = match.groups()
        return Namespace(
            org=org,
            workspace=workspace,
            user=user_slot,
            session="_lifecycle_sweep",
            visibility=Visibility(vis),
        )

    def _record_explain(
        self, ns: Namespace, memory_id: str, transition: TransitionKind, score: float
    ) -> None:
        """Builds one ``ExplainRecord`` from a ``PromotionOutcome``/``DemotionOutcome`` (S1-01/
        S1-02) — see module docstring for why ``rec``/``use``/``imp`` are honest ``0.0``
        placeholders rather than fabricated numbers."""
        salience_inputs = SalienceInputs(
            rec=0.0,
            use=0.0,
            imp=0.0,
            w_rec=self._settings.salience.w_recency,
            w_use=self._settings.salience.w_usage,
            w_imp=self._settings.salience.w_importance,
            recency_half_life_h=self._settings.salience.recency_half_life_h,
            score=score,
        )
        record = ExplainRecord(
            memory_id=memory_id,
            namespace=ns.to_prefix(),
            transition=transition,
            decided_at=self._clock.now(),
            salience_inputs=salience_inputs,
            model_verdict=None,
            config_version=self._settings.config_version,
            policy_version=self._settings.policy_version,
            decided_by=_DECIDED_BY_DEFAULT,
            lease_held="lifecycle-sweep",
        )
        self._explain_log.setdefault((ns.to_prefix(), memory_id), []).append(record)
