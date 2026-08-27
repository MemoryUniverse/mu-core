"""``MemoryHealthService.assess`` — the READ side (CQRS, side-effect-free).

Authority: ``memory-health-pinning-spec.md`` §5.1 (lines 238-252).

**Read-purity is the contract, not a nicety.** Unlike recall, assessing health MUST NOT reinforce
(no ``access_count``/``last_seen`` write-back) and MUST NOT trigger a tier transition. This class
therefore touches exactly two repository methods — ``enumerate`` and, through the injected reader,
``edges_for`` — both reads; it holds no write port at all, which is what makes the purity
structural rather than a promise. The only thing it ever publishes is the named degrade event,
which §5.1 line 251 requires ("never a silent partial").

**Bounded.** One call = one page of at most ``HealthSettings.page_size`` items, walked through
``MemoryRepository.enumerate`` (spec §3.1) and continued via ``next_cursor``. There is no
unbounded partition scan anywhere on this path, including the conflict lookup, which is bounded
by the page's own ids (``ConflictEdgeReader``'s member-intersection contract).

**Signature delta (recorded).** Spec line 245 writes ``assess(self, ns, *, filter_flags,
cursor)`` while line 248 requires ``TenancyGuard.assert_scope(scope, ns)`` — the ``scope`` the
authz step needs is absent from the signature. It is added as the first parameter here (the same
shape ``PinService`` uses, and the same shape ``DefaultTenancyGuard.assert_scope(scope, ns,
operation)`` actually has: three arguments, ``operation`` required, versus the spec's two).
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

import structlog

from mu_contracts.domain.errors import TierRepositoryUnavailableError
from mu_contracts.domain.events import DegradedModeEntered, DegradeReason
from mu_contracts.domain.model.conflict import ConflictEdges
from mu_contracts.domain.model.health import (
    AT_RISK_FLAGS,
    MemoryHealthEntry,
    MemoryHealthFlag,
    MemoryHealthSummary,
    MemoryHealthView,
)
from mu_contracts.domain.model.memory import MemoryItem, Namespace, State, Tier
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.conflict import ConflictEdgeReader
from mu_contracts.ports.health_assessor import HealthAssessor
from mu_contracts.ports.memory import MemoryRepository
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.security import TenancyGuard
from mu_contracts.ports.time import Clock
from mu_engine.pipelines.distill import EventPublisher
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    TraceScope,
)
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.services.health.settings import HealthSettings

__all__ = ["MemoryHealthService"]

_log = structlog.get_logger("mu_engine.services.health")

_OP = "memory.health_assess"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"

#: The states the health lens reports on (spec §5.1 line 249). ``DELETED`` is excluded because a
#: GC'd item no longer exists to be healthy or not; ``SUPERSEDED``/``EXPIRED`` are excluded
#: because they are settled outcomes, not risks a user can act on.
ASSESSED_STATES: frozenset[State] = frozenset({State.ACTIVE, State.ARCHIVED, State.QUARANTINED})

#: The degraded-read tier subset. CANONICAL §2 names exactly one memory-tier degrade reason —
#: ``LTM_UNAVAILABLE`` — and spec §5.1 line 251 reuses it, so the ONE modelled partial view is
#: "LTM is down, STM+MTM answered".
_REACHABLE_WITHOUT_LTM: frozenset[Tier] = frozenset({Tier.STM, Tier.MTM})

_DEGRADE_COMPONENT = "ltm"
_DEGRADE_MODE = "health_partial"


class MemoryHealthService:
    """The owner-read health projection over one partition (spec §5.1)."""

    def __init__(
        self,
        *,
        repo: MemoryRepository,
        assessor: HealthAssessor,
        conflicts: ConflictEdgeReader,
        settings: HealthSettings | None = None,
        clock: Clock | None = None,
        scope_guard: TenancyGuard | None = None,
        bus: EventPublisher | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._repo = repo
        self._assessor = assessor
        self._conflicts = conflicts
        self._settings = settings or HealthSettings()
        self._clock: Clock = clock or SystemClock()
        self._scope_guard: TenancyGuard = scope_guard or DefaultTenancyGuard()
        self._bus = bus
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()

    async def assess(
        self,
        scope: ClientScope,
        ns: Namespace,
        *,
        filter_flags: frozenset[MemoryHealthFlag] | None = None,
        cursor: str | None = None,
    ) -> MemoryHealthView:
        """One bounded page of ``ns``'s memory health. Mutates nothing.

        Wrapped in the house observability envelope (content-free span + latency/error metrics +
        a counts-only audit row), identically to ``DemotionService.demote`` /
        ``RetentionService.sweep``. ``CancelledError`` propagates and is never counted as a
        failure.
        """
        self._scope_guard.assert_scope(scope, ns, _OP)
        started = time.perf_counter()
        with self._tracer.span(_OP, attributes={"ns": ns.to_prefix()}):
            try:
                view = await self._assess(ns, filter_flags=filter_flags, cursor=cursor)
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
            visibility=ns.visibility.value,
            counts={
                "walked": view.summary.total,
                "surfaced": len(view.entries),
                "pinned": view.summary.pinned_count,
            },
        )
        return view

    async def _assess(
        self,
        ns: Namespace,
        *,
        filter_flags: frozenset[MemoryHealthFlag] | None,
        cursor: str | None,
    ) -> MemoryHealthView:
        now = self._clock.now()
        page, next_cursor, partial = await self._walk(ns, cursor=cursor)
        edges = await self._edges_for(ns, page)

        assessed: list[tuple[MemoryItem, frozenset[MemoryHealthFlag]]] = [
            (item, self._assessor.assess(item, now=now, conflict_edges=edges)) for item in page
        ]
        entries = tuple(
            self._to_entry(item, flags, edges, now=now)
            for item, flags in assessed
            if self._surface(flags, filter_flags)
        )
        return MemoryHealthView(
            namespace=ns,
            summary=self._summarize(assessed),
            entries=entries,
            next_cursor=next_cursor,
            partial=partial,
            generated_at=now,
        )

    async def _walk(
        self, ns: Namespace, *, cursor: str | None
    ) -> tuple[list[MemoryItem], str | None, bool]:
        """One bounded page across every tier, degrading to STM+MTM if a tier is unreachable.

        The retry is deliberately NOT a bare fallback (DEV-STANDARDS rule 8): it narrows to the
        one tier subset CANONICAL models a degrade reason for, emits the NAMED
        ``DegradedModeEntered(LTM_UNAVAILABLE)``, and marks the view ``partial``. If the narrowed
        read also fails there is no modelled degraded path left, and the error propagates loud.
        """
        try:
            page, next_cursor = await self._enumerate(ns, tiers=None, cursor=cursor)
        except TierRepositoryUnavailableError:
            page, next_cursor = await self._enumerate(
                ns, tiers=_REACHABLE_WITHOUT_LTM, cursor=cursor
            )
            await self._emit_degrade(ns)
            return page, next_cursor, True
        return page, next_cursor, False

    async def _enumerate(
        self, ns: Namespace, *, tiers: frozenset[Tier] | None, cursor: str | None
    ) -> tuple[list[MemoryItem], str | None]:
        return await self._repo.enumerate(
            ns,
            states=ASSESSED_STATES,
            tiers=tiers,
            pinned=None,  # the view reports on pinned and unpinned alike
            cursor=cursor,
            limit=self._settings.page_size,
        )

    async def _edges_for(self, ns: Namespace, page: list[MemoryItem]) -> ConflictEdges:
        """Scoped by the AUTHORIZED ``ns`` — the one that passed ``assert_scope`` — never by a
        namespace read off the returned data.

        ``ConflictEdgeReader`` scopes its query by ``to_prefix()`` of whatever ``Namespace`` it is
        handed, so keying it on ``page[0].namespace`` would let a single mis-partitioned row
        redirect the whole page's conflict read into another tenant's prefix. CLAUDE.md rule 4 /
        CANONICAL §1 rule 5 require the scope key to come from the authorized η, and this repo has
        shipped two partition-key defects already (0545119, 1dd023c) — "enumerate can only return
        in-partition items" is not an assumption worth encoding.
        """
        if not page:
            return ConflictEdges()
        return await self._conflicts.edges_for(ns, frozenset(item.id for item in page))

    async def _emit_degrade(self, ns: Namespace) -> None:
        _log.warning("health_view_partial", ns=ns.to_prefix(), reason=DegradeReason.LTM_UNAVAILABLE)
        if self._bus is None:
            return
        await self._bus.publish(
            DegradedModeEntered(
                component=_DEGRADE_COMPONENT,
                mode=_DEGRADE_MODE,
                reason=DegradeReason.LTM_UNAVAILABLE,
            )
        )

    def _surface(
        self,
        flags: frozenset[MemoryHealthFlag],
        filter_flags: frozenset[MemoryHealthFlag] | None,
    ) -> bool:
        """An explicit ``filter_flags`` wins over ``include_healthy``: the caller asked for a
        named slice, so "at-risk only" must not silently subtract from it."""
        if filter_flags is not None:
            return bool(flags & filter_flags)
        if self._settings.include_healthy:
            return True
        return bool(flags & AT_RISK_FLAGS)

    def _to_entry(
        self,
        item: MemoryItem,
        flags: frozenset[MemoryHealthFlag],
        edges: ConflictEdges,
        *,
        now: datetime,
    ) -> MemoryHealthEntry:
        return MemoryHealthEntry(
            memory_id=item.id,
            tier=item.tier,
            state=item.state,
            flags=flags,
            retention=self._assessor.retention(item, now=now),
            salience_score=None if item.salience is None else item.salience.score,
            confidence=edges.confidence_for(item.id),
            last_seen=item.last_seen,
            pinned=item.pinned,
            conflict_with_ids=edges.peers_for(item.id),
        )

    @staticmethod
    def _summarize(
        assessed: list[tuple[MemoryItem, frozenset[MemoryHealthFlag]]],
    ) -> MemoryHealthSummary:
        """Counts over the PAGE THAT WAS WALKED, not the partition.

        Said plainly because it is a real limitation: a partition-wide count would be an
        unbounded scan, which spec §3.1 forbids outright ("NEVER unbounded"). The honest number
        is "of the N items on this page, k are stale"; a caller that needs the partition total
        pages through.
        """
        by_flag: dict[MemoryHealthFlag, int] = {}
        by_tier: dict[Tier, int] = {}
        pinned_count = 0
        retention_unknown = 0
        for item, flags in assessed:
            for flag in flags:
                by_flag[flag] = by_flag.get(flag, 0) + 1
            by_tier[item.tier] = by_tier.get(item.tier, 0) + 1
            if item.pinned:
                pinned_count += 1
            if item.salience is None and item.tier is not Tier.STM:
                # The DECAYING/STALE rules both read `item.salience`; with none recorded they can
                # never fire for this item. Counted so the caller is TOLD the lens was partly
                # blind instead of reading an all-clear that was never computed.
                retention_unknown += 1
        return MemoryHealthSummary(
            total=len(assessed),
            by_flag=by_flag,
            by_tier=by_tier,
            pinned_count=pinned_count,
            retention_unknown=retention_unknown,
        )
