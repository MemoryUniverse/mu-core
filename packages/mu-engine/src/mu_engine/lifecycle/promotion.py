"""``PromotionService`` — the THREE promotion paths + pre-TTL rescue (spec §7b, lines 287-311;
ADR 0034 promotion half; spec §14 slice 1, line 420; AC-1.3a, line 436).

```
STM --(i)   immediate      importance>=0.6 / mentions>=2 / add(promote=True)   [EXISTS, untouched]
STM --(iii) periodic       S(m) >= promote_stm_mtm                       [NEW — this module]
STM --TTL-> gone           pre-TTL rescue recomputes salience first      [NEW — this module]
MTM --(ii)  session-bound  full-session recompute -> promote + distill   [NEW trigger, here]
MTM --(iii) periodic       S(m) >= promote_mtm_ltm & age>min_age->distill [NEW — this module]
```

**Path (i) — immediate — is explicitly OUT of this file's scope.** The spec (§7b point 1) says
"KEEP today's deterministic gate ... EXISTS": ``mu_engine.pipelines.concrete.ingest.
DeterministicPromoteStage`` (``importance>=IngestSettings.importance_promote`` OR
``IngestActivity.promote=True``) and ``mu_local.local_memory.LocalMemory.add()`` (which always
sets ``promote=True``, ``local_memory.py:94-125``) are UNCHANGED and NOT re-implemented here —
this docstring, and :func:`assert_immediate_gate_seam_unchanged` below, are the "document/test
the seam" this task's acceptance list asks for; the load-bearing regression coverage for path (i)
already lives in ``tests/services/test_ingest_int.py`` (``test_remember_writes_stm_and_mtm`` /
``test_low_importance_stays_stm_only``), which this task does not duplicate.

**Paths (ii)/(iii) + pre-TTL rescue are what this module adds**, as a composable service over
already-EXISTS pieces — never a second copy of their logic (DEV-STANDARDS rule 6, DRY):

* the STM->MTM write itself PORTs the exact copy-on-write shape ``DeterministicPromoteStage``
  already uses (``ingest.py:227-259``: ``item.model_copy`` with ``tier=MemoryTier.MTM``, then
  ``mtm.upsert`` — the STM copy is left in place, the Redis TTL is its own eviction, never an
  explicit ``evict()`` call here);
* the MTM->LTM leg is a pure **gate** (salience + age) in front of the EXISTS
  ``DistillPipeline.distill`` (``pipelines/distill.py``) — this module decides WHICH items are
  handed to DISTILL, DISTILL decides what happens to each one (mem0 ADD/NOOP/SUPERSEDE/
  SELF_EXPIRE/COEXIST) and emits its OWN ``MemoryPromoted(frm=MTM, to=LTM, reason="distill")``
  (``distill.py:461-467``) — this module never re-emits a second promote event for the same
  transition;
* the pre-TTL rescue window check is a pure function of ``IngestSettings.stm_ttl_s`` (the exact
  TTL ``RedisMapper``/``ValkeyStmAdapter`` set at ``put()`` time, ``redis_mapper.py:31-47``) and
  ``item.created_at`` — no new store-port method is added (this task owns only this one file);
  PORT of the reference pattern ``mma/mma/memory/controller.py:_pre_ttl_sweep:828-874``
  ("ask STM for remaining-TTL candidates, recompute salience NOW, promote survivors") re-expressed
  over the salience-gate + real Valkey TTL floor here instead of a bespoke STM scan method;
* the session-boundary entry point mirrors the fire-and-forget batch shape of
  ``mma/mma/memory/controller.py:_trigger_promotion_for_key:716-738`` (one namespace, best-effort,
  a warning — never a crash — on a downstream failure is NOT adopted here: DEV-STANDARDS rule 8
  wants central, fail-loud exception handling, so this module lets a downstream store failure
  propagate to the caller instead of swallowing it; the daemon-side ``PreCompactPromoter`` HOOK
  WIRING itself (CANONICAL §5.2: PreCompact has ONE owner) is explicitly S1-07's job, not this
  file's — :meth:`PromotionService.promote_session` only exposes the entry point the daemon calls
  on ``Stop``/``SessionEnd``/``PreCompact``/the idle timer.

Salience gate: ``SalienceStrategy.score(item, clock=...)`` (S0-08, ``lifecycle/salience.py``) at
thresholds from ``LifecycleSettings`` (S0-07, ``lifecycle/settings.py``): ``promote_stm_mtm``
(0.7), ``promote_mtm_ltm`` (0.9), ``promote_min_age_h`` (24h), ``pre_ttl_window_s`` (300s).

Observability mirrors ``DistillPipeline.distill`` verbatim (``distill.py:241-273``): a
content-free ``Tracer`` span per operation, ``mu_operation_latency_seconds``/
``mu_operation_errors_total`` (always-observe / on-failure-only, same metric NAMES so the two
pipelines aggregate on one dashboard), a content-free ``AuditLog.record`` on success, plus a
promotion-specific ``mu_lifecycle_promotions_total{frm,to}`` counter (CANONICAL §5.1 —
``MemoryPromoted`` shape unchanged) for every transition this module itself emits.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.errors import MemoryUniverseError
from mu_contracts.domain.events import MemoryPromoted
from mu_contracts.domain.model.memory import Tier
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.pipelines.distill import DistillPipeline, DistillReport, EventPublisher
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    TraceScope,
)
from mu_engine.services.settings import IngestSettings
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.ports import MtmTierRepository, StmTierRepository

__all__ = [
    "PromotionOutcome",
    "PromotionOutcomeKind",
    "PromotionReport",
    "PromotionService",
    "SharedPlaneSweepUnsupportedError",
    "assert_immediate_gate_seam_unchanged",
]

_STM_MTM_OP = "promotion.stm_mtm"
_MTM_LTM_GATE_OP = "promotion.mtm_ltm_gate"
_PRE_TTL_OP = "promotion.pre_ttl_rescue"
_SESSION_OP = "promotion.session_boundary"

# Same metric NAMES as DistillPipeline (distill.py:82-83) — one dashboard, two emitters
# (DEV-STANDARDS rule 4: central observability, not a per-pipeline reinvention).
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"
# Promotion-specific counter (CANONICAL §5.1 MemoryPromoted, tagged by transition).
_PROMOTION_METRIC = "mu_lifecycle_promotions_total"


class SharedPlaneSweepUnsupportedError(MemoryUniverseError):
    """A lifecycle sweep was asked to run over a SHARED η, and this engine cannot yet do it.

    **AD-142.** Raised by :meth:`PromotionService.promote_session`. It refuses instead of
    proceeding, and it refuses HERE — at the top of the verb — rather than letting the failure
    surface from three layers down as
    :class:`~mu_contracts.domain.errors.CallerIdentitySetRequiredError`, which is what happened
    before and which points the reader at the WRONG fix.

    Why the wrong fix. ``promote_session`` reads its window with
    ``StmTierRepository.recent(ns, limit=…)`` and passes no ``caller_identity_set``, because a
    maintenance sweep serves NO caller — there is no principal set that would be honest to pass.
    Since AD-128 that argument is REQUIRED on a SHARED η, so the adapter raises
    ``CallerIdentitySetRequiredError``, whose message reasonably reads as *"supply the caller
    set"*. Supplying a ``SYSTEM``/all-principals sentinel to satisfy it is precisely the
    fail-open shape AD-128 was opened to remove, and it would re-open it on a verb nothing
    audits. So the error is not a missing argument, and this class exists to say so by name.

    Why the sweep is not simply re-pointed at the enumeration verb either — the SECOND, larger
    obligation, found while resolving this row and not previously recorded (see AD-143). The
    read half is genuinely easy: ``recent()`` is a RECALL verb that serves a caller, while
    ``TierEnumerationPort.enumerate_page`` (``storage/tier_capabilities.py:71``, implemented by
    ``RedisStmAdapter``/``InMemoryStmAdapter``) is the partition-walk MAINTENANCE verb, carries
    no Model-A filter by design, and is what a sweep should read through. But the WRITE half is
    not built: ``promote_session``'s second leg hands every survivor to
    ``DistillPipeline.distill``, and ``pipelines/distill.py:646-673`` constructs the distilled
    LTM fact as a BRAND-NEW ``MemoryItem`` whose ``metadata`` is exactly
    ``{"derived_from", "extractor_span"}`` (plus an optional ``valid_at_inferred``). The source
    row's ``authorized_ids`` stamp is DROPPED. On a SHARED η an unstamped row is DENIED by
    ``model_a_permits`` — fail closed, so nothing leaks — which means a shared-plane sweep that
    "worked" would quietly distil every promoted item into an LTM fact that NO member can ever
    read again. Fixing only the read would convert a loud refusal into silent data loss.

    So the refusal stands until BOTH halves land: an enumeration-based maintenance read, and
    stamp carry-through in DISTILL. On a PRIVATE η nothing changes — ``§1`` rule 5 authorizes
    the own partition by key, ``recent()`` needs no caller set there, and this error is never
    raised.
    """


class PromotionOutcomeKind(StrEnum):
    """Per-candidate STM->MTM gate verdict (the MTM->LTM leg's verdicts are DISTILL's own
    ``DistillActionKind``, reused via :attr:`PromotionReport.distilled`, never re-declared here)."""

    STM_TO_MTM = "stm_to_mtm"
    SKIPPED_BELOW_GATE = "skipped_below_gate"


class PromotionOutcome(BaseModel):
    """One STM-window candidate's gate verdict (content-free: id/score/enum/reason label)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: PromotionOutcomeKind
    memory_id: str
    score: float
    reason: str


class PromotionReport(BaseModel):
    """Aggregate outcome of one promotion call. ``distilled`` is populated only on the MTM->LTM
    leg (``sweep_mtm_to_ltm``/``promote_session``) — it is the EXISTS ``DistillReport`` verbatim,
    never re-derived (DRY)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcomes: tuple[PromotionOutcome, ...] = ()
    distilled: DistillReport | None = None

    @property
    def promoted_stm_mtm(self) -> int:
        return sum(o.kind is PromotionOutcomeKind.STM_TO_MTM for o in self.outcomes)

    @property
    def promoted_mtm_ltm(self) -> int:
        return 0 if self.distilled is None else len(self.distilled.actions)


def _age_hours(item: MemoryItem, now: datetime) -> float:
    """Hours since ``item.created_at`` at ``now`` (same shape as ``salience.py``'s own
    ``_recency`` age computation, kept local since that helper is private to that module)."""
    created_at = item.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return (now - created_at).total_seconds() / 3600.0


def assert_immediate_gate_seam_unchanged(settings: IngestSettings) -> None:
    """Documents + guards path (i) (spec §7b point 1): this is a VALUE assertion, not a
    behavioral test — it fails loudly if a future edit to ``IngestSettings`` silently redefines
    the "today's deterministic gate" this module's docstring promises never changed. The actual
    behavioral coverage lives in ``tests/services/test_ingest_int.py`` against the real
    ``DeterministicPromoteStage`` (this module does not re-run that pipeline)."""
    if settings.importance_promote != 0.6 or settings.mention_promote != 2:
        raise AssertionError(
            "IngestSettings deterministic-promotion gate changed under PromotionService's "
            "feet — path (i) is supposed to be EXISTS/unchanged (spec §7b point 1); update this "
            "module's docstring (and re-verify tests/services/test_ingest_int.py) before trusting "
            "this assertion again."
        )


class PromotionService:
    """Paths (ii)/(iii) + pre-TTL rescue (spec §7b). Depends on already-EXISTS ports/pipelines
    only — no new store-port method, no second copy of DISTILL's diff loop.

    ``stm`` is optional and used ONLY by :meth:`promote_session`, which self-fetches its
    session's STM window (the daemon's entry point only carries ``ns``/``session_id`` per the
    spec's own signature, §7b point 2) — ``sweep_stm_to_mtm``/``rescue_pre_ttl`` take an
    explicit, caller-supplied window instead (the full-scan/backstop ENUMERATION across users
    is the ``MaintenanceLoop``'s job, a sibling Stage-1 task, not this file's).
    """

    def __init__(
        self,
        *,
        mtm: MtmTierRepository,
        distill: DistillPipeline,
        salience: SalienceStrategy,
        stm: StmTierRepository | None = None,
        settings: LifecycleSettings | None = None,
        ingest_settings: IngestSettings | None = None,
        clock: Clock | None = None,
        bus: EventPublisher | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._mtm = mtm
        self._distill = distill
        self._salience = salience
        self._stm = stm
        self._settings = settings or LifecycleSettings()
        self._ingest_settings = ingest_settings or IngestSettings()
        self._clock: Clock = clock or SystemClock()
        self._bus = bus
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()

    # ---- path (iii)a — periodic STM->MTM backstop -------------------------------------------
    async def sweep_stm_to_mtm(
        self,
        ns: Namespace,
        window: Sequence[MemoryItem],
        *,
        reason: str = "periodic_sweep",
    ) -> PromotionReport:
        """``S(m) >= promote_stm_mtm`` -> STM->MTM promote (spec §7b point 3). ``window`` is
        caller-supplied (e.g. ``StmTierRepository.recent`` results) — this method is the GATE,
        never the enumeration."""
        report, _ = await self._stm_gate(_STM_MTM_OP, ns, window, reason=reason)
        return report

    # ---- path (iii)b — periodic MTM->LTM backstop (delegates to DistillPipeline) -------------
    async def sweep_mtm_to_ltm(
        self, ns: Namespace, window: Sequence[MemoryItem]
    ) -> PromotionReport:
        """``S(m) >= promote_mtm_ltm AND age > promote_min_age_h`` -> hand the survivor to
        ``DistillPipeline.distill`` (spec §7b point 3) — this method never writes LTM itself."""
        started = time.perf_counter()
        with self._tracer.span(_MTM_LTM_GATE_OP, attributes={"ns": ns.to_prefix()}):
            try:
                survivors = self._select_mtm_ltm_survivors(window)
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._metrics.inc(_ERROR_METRIC, labels={"operation": _MTM_LTM_GATE_OP})
                raise
            finally:
                self._metrics.observe(
                    _LATENCY_METRIC,
                    time.perf_counter() - started,
                    labels={"operation": _MTM_LTM_GATE_OP},
                )
        self._audit.record(
            TraceScope(correlation_id=ns.to_prefix()),
            operation=_MTM_LTM_GATE_OP,
            outcome="ok",
            tier="ltm",
            visibility=ns.visibility.value,
            counts={"candidates": len(window), "survivors": len(survivors)},
        )
        if not survivors:
            return PromotionReport()
        # DISTILL emits its OWN MemoryPromoted(frm=MTM, to=LTM, reason="distill") per winner
        # (distill.py:461-467); this module only meters that its gate handed survivors over.
        self._metrics.inc(
            _PROMOTION_METRIC, labels={"frm": "mtm", "to": "ltm"}, value=len(survivors)
        )
        distilled = await self._distill.distill(ns, survivors)
        return PromotionReport(distilled=distilled)

    def _select_mtm_ltm_survivors(self, window: Sequence[MemoryItem]) -> list[MemoryItem]:
        now = self._clock.now()
        return [
            item
            for item in window
            if self._salience.score(item, clock=self._clock) >= self._settings.promote_mtm_ltm
            and _age_hours(item, now) > self._settings.promote_min_age_h
        ]

    # ---- pre-TTL rescue (spec §7b "Pre-TTL rescue (NEW)") -------------------------------------
    async def rescue_pre_ttl(self, ns: Namespace, window: Sequence[MemoryItem]) -> PromotionReport:
        """Items whose remaining Redis TTL <= ``pre_ttl_window_s`` are recomputed NOW and
        promoted STM->MTM if salient — before the real Redis TTL deletes them. Remaining TTL is
        derived from ``IngestSettings.stm_ttl_s`` (the exact TTL ``RedisMapper.to_store`` set at
        ``put()`` time) and ``item.created_at``, never a new store-port TTL-read method (this
        task owns only this one file). PORT of ``mma/mma/memory/controller.py:_pre_ttl_sweep:
        828-874`` (scan remaining-TTL candidates -> recompute salience -> promote survivors)."""
        now = self._clock.now()
        due = [
            item
            for item in window
            if self._remaining_ttl_s(item, now) <= self._settings.pre_ttl_window_s
        ]
        report, _ = await self._stm_gate(_PRE_TTL_OP, ns, due, reason="pre_ttl_rescue")
        return report

    def _remaining_ttl_s(self, item: MemoryItem, now: datetime) -> float:
        return self._ingest_settings.stm_ttl_s - _age_hours(item, now) * 3600.0

    # ---- path (ii) — session-boundary consolidation --------------------------------------------
    async def promote_session(
        self, ns: Namespace, session_id: str, *, force: bool = False
    ) -> PromotionReport:
        """The daemon-callable entry point on ``Stop``/``SessionEnd``/``PreCompact``/the idle
        timer (spec §7b point 2; CANONICAL §5.2 — PreCompact has ONE owner, the daemon's
        ``PreCompactPromoter``, which now calls THROUGH this method with ``force=True``). Sweeps
        *this session's* STM buffer, recomputes salience with the full session window (no
        ``promote_min_age_h`` gate here — an explicit boundary trigger, not the periodic backstop),
        promotes survivors STM->MTM, then hands every survivor straight to DISTILL for MTM->LTM
        (spec: "promote survivors STM->MTM and distill MTM->LTM" — no additional age gate on this
        path).

        ``force`` (the PreCompact promote-before-delete path, AGENT-INTEGRATION-AUDIT-AND-PLAN.md
        §4 Phase 3): when ``True``, the STM->MTM salience gate (``promote_stm_mtm``) is BYPASSED —
        EVERY item in this session's STM window is promoted regardless of routine salience, because
        the host is about to compact/delete these turns and the whole point is to SAVE the at-risk
        context before it is dropped, not to re-litigate whether each turn was individually salient
        enough for the periodic backstop. The salience ``score`` is still computed and recorded on
        each ``PromotionOutcome`` (honest provenance — the reader sees WHAT would have been gated
        out), only the GATE DECISION is overridden. ``force`` reuses the SAME
        ``_promote_to_mtm``/``_distill.distill`` machinery every other path uses — it is not a
        second, parallel promoter (DEV-STANDARDS rule 6, DRY). ``force=False`` (every pre-existing
        caller: the periodic backstop / ``Stop`` / idle timer via ``sweep_namespace_now``) is
        byte-for-byte the prior gated behaviour.

        **PRIVATE η only (AD-142).** A SHARED η raises
        :class:`SharedPlaneSweepUnsupportedError` before any store is touched — read that class's
        docstring for why the refusal is the honest answer and what two things are owed before it
        can be lifted. Nothing on either shipped composition reaches this branch today
        (``mu-engine-server`` mounts no shared-plane routes; ``mu-local``'s ``_ns`` builds a
        PRIVATE η), so the refusal changes no working path — it replaces a misleading
        ``CallerIdentitySetRequiredError`` raised three layers down with a named one that states
        the real obligation.
        """
        if ns.session != session_id:
            raise ValueError(
                f"promote_session: ns.session={ns.session!r} does not match session_id="
                f"{session_id!r} (η already carries the session slot — pass the SAME session)"
            )
        if self._stm is None:
            raise RuntimeError(
                "promote_session requires an injected StmTierRepository (none was configured on "
                "this PromotionService instance) — never a silent no-op (DEV-STANDARDS rule 8)."
            )
        # AD-142 — refuse a SHARED η HERE, by name, instead of letting the STM port's
        # ``CallerIdentitySetRequiredError`` surface from the read below. See
        # :class:`SharedPlaneSweepUnsupportedError` for why the honest answer is a refusal and
        # not a ``SYSTEM`` caller sentinel, and for the SECOND obligation (DISTILL drops the
        # ``authorized_ids`` stamp) that makes a read-only fix worse than the refusal.
        if ns.visibility is Visibility.SHARED:
            raise SharedPlaneSweepUnsupportedError(
                "promote_session refuses a SHARED namespace: a lifecycle sweep serves no caller, "
                "so it has no honest caller_identity_set for the Model-A gate on "
                "StmTierRepository.recent (AD-128), and DISTILL does not yet carry the "
                "authorized_ids stamp onto the facts it derives — running anyway would write LTM "
                "facts no member could read. Owed, in order: a maintenance read through "
                "TierEnumerationPort.enumerate_page, then stamp carry-through in "
                f"DistillPipeline. (namespace={ns.to_prefix()})"
            )
        window = [
            scored.item
            for scored in await self._stm.recent(ns, limit=self._settings.max_items_per_user_sweep)
        ]
        reason = "precompact_promote_before_delete" if force else "session_boundary"
        stm_report, promoted_items = await self._stm_gate(
            _SESSION_OP, ns, window, reason=reason, force=force
        )
        if not promoted_items:
            return stm_report
        distilled = await self._distill.distill(ns, promoted_items)
        return PromotionReport(outcomes=stm_report.outcomes, distilled=distilled)

    # ---- shared STM->MTM gate + observability (mirrors DistillPipeline.distill, distill.py:
    #      241-273) ------------------------------------------------------------------------------
    async def _stm_gate(
        self,
        op: str,
        ns: Namespace,
        window: Sequence[MemoryItem],
        *,
        reason: str,
        force: bool = False,
    ) -> tuple[PromotionReport, list[MemoryItem]]:
        started = time.perf_counter()
        outcomes: list[PromotionOutcome] = []
        promoted_items: list[MemoryItem] = []
        with self._tracer.span(op, attributes={"ns": ns.to_prefix()}):
            try:
                outcomes, promoted_items = await self._apply_stm_gate(
                    ns, window, reason=reason, force=force
                )
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._metrics.inc(_ERROR_METRIC, labels={"operation": op})
                raise
            finally:
                self._metrics.observe(
                    _LATENCY_METRIC, time.perf_counter() - started, labels={"operation": op}
                )
        self._audit.record(
            TraceScope(correlation_id=ns.to_prefix()),
            operation=op,
            outcome="ok",
            tier="mtm",
            visibility=ns.visibility.value,
            counts={"candidates": len(window), "promoted": len(promoted_items)},
        )
        return PromotionReport(outcomes=tuple(outcomes)), promoted_items

    async def _apply_stm_gate(
        self, ns: Namespace, window: Sequence[MemoryItem], *, reason: str, force: bool = False
    ) -> tuple[list[PromotionOutcome], list[MemoryItem]]:
        now = self._clock.now()
        outcomes: list[PromotionOutcome] = []
        promoted_items: list[MemoryItem] = []
        for item in window:
            score = self._salience.score(item, clock=self._clock)
            # ``force`` (PreCompact promote-before-delete): promote EVERY at-risk turn regardless
            # of routine salience — the host is about to drop these. ``score`` is still computed
            # and recorded below (honest provenance), only the gate DECISION is overridden.
            if force or score >= self._settings.promote_stm_mtm:
                promoted = await self._promote_to_mtm(ns, item, reason=reason, now=now)
                promoted_items.append(promoted)
                outcomes.append(
                    PromotionOutcome(
                        kind=PromotionOutcomeKind.STM_TO_MTM,
                        memory_id=item.id,
                        score=score,
                        reason=reason,
                    )
                )
            else:
                outcomes.append(
                    PromotionOutcome(
                        kind=PromotionOutcomeKind.SKIPPED_BELOW_GATE,
                        memory_id=item.id,
                        score=score,
                        reason="below_promote_stm_mtm",
                    )
                )
        return outcomes, promoted_items

    async def _promote_to_mtm(
        self, ns: Namespace, item: MemoryItem, *, reason: str, now: datetime
    ) -> MemoryItem:
        # Same copy-on-write shape as the EXISTS DeterministicPromoteStage (ingest.py:227-259):
        # the STM copy is left in place (its Redis TTL is its own eviction) — never an explicit
        # evict() call from this gate.
        promoted = item.model_copy(deep=True)
        promoted.tier = MemoryTier.MTM
        promoted.state = MemoryState.ACTIVE
        promoted.updated_at = now
        await self._mtm.upsert(promoted)
        self._metrics.inc(_PROMOTION_METRIC, labels={"frm": "stm", "to": "mtm"})
        if self._bus is not None:
            await self._bus.publish(
                MemoryPromoted(
                    namespace=ns, id=promoted.id, frm=Tier.STM, to=Tier.MTM, reason=reason
                )
            )
        return promoted
