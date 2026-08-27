"""``PinService.pin`` / ``.unpin`` — the WRITE side of pinning.

Authority: ``memory-health-pinning-spec.md`` §5.2 (lines 254-269), whose five ordered steps this
class performs in that order.

**Pin is retention, never access, never relevance** (CANONICAL §7.26 / spec §6.5). Nothing here
touches ``authorized_ids``, no recall filter learns about it, and ``pinned_by`` is an audit
principal only. Failing a pin never changes what anyone can read.

**Step 5 (cross-device convergence) is NOT built here, and that is deliberate — reported, not
silently skipped.** Spec line 269 has ``PinService`` append a
``PrivateDelta(op=SyncOp.PIN, …, origin_device_id, lamport, …)`` through ``PrivateSyncLogPort``.
``SyncOp.PIN``/``UNPIN`` and ``PrivateDelta.pinned`` already exist in the vocabulary, but **the
two required authoring inputs do not exist anywhere in mu-core**: there is no device identity and
no per-device Lamport counter in this repo (a workspace-wide grep finds no producer of
``PrivateDelta`` at all outside a test fixture). Both live in the mu-client daemon's sync client,
which spec §7.2 line 339 itself names as the appender. Fabricating a device id or a Lamport value
here would manufacture a convergence guarantee the system does not have — the exact failure mode
CANONICAL §7.17's determinism argument depends on not happening. So this service performs steps
1-4 and the delta append lands with the daemon sync client that owns the clock.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

import structlog

from mu_contracts.domain.errors import (
    PinAuthorizationError,
    PinLimitExceededError,
    PinTargetNotFoundError,
    PinTargetNotPinnableError,
)
from mu_contracts.domain.events import DomainEvent, MemoryPinned, MemoryUnpinned
from mu_contracts.domain.model.memory import Namespace, State, Visibility
from mu_contracts.domain.model.pin import PinRequest, PinResult
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.memory import MemoryRepository
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.security import TenancyGuard
from mu_contracts.ports.time import Clock
from mu_engine.platform.application.idempotent_write_scope import (
    IdempotentWriteScope,
    MemoryWriteScope,
    WriteStep,
)
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    TraceScope,
)
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.services.pin.settings import PinSettings

__all__ = ["PinService", "ScopeFactory"]

_log = structlog.get_logger("mu_engine.services.pin")

_OP_PIN = "memory.pin"
_OP_UNPIN = "memory.unpin"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"

#: The states a pin may target — ENFORCED in :meth:`PinService._resolve`, not merely declared.
#: Pinning a DELETED/SUPERSEDED/EXPIRED item is meaningless (pin is a RETENTION override and there
#: is nothing left to retain once the item has left) AND harmful: a pinned row is unconditionally
#: GC-ineligible (CANONICAL §7.10), so accepting the pin would strand a dead row in the graph
#: permanently. Doubles as the ``enumerate`` state filter for the §5.2-step-2 pin count, so the
#: bound counts exactly the pins this service is willing to create.
PINNABLE_STATES: frozenset[State] = frozenset({State.ACTIVE, State.ARCHIVED, State.QUARANTINED})

#: A factory producing the write boundary for one mutation (spec §5.2 step 3: "one
#: ``IdempotentWriteScope``"). Injected so a caller that already owns a scope — e.g. a
#: replica-apply carrying ``caused_by_seq`` — supplies its own instead of this service minting a
#: bare one.
ScopeFactory = Callable[[WriteStep], IdempotentWriteScope]


class PinService:
    """Set / clear the ``pinned`` lifecycle-override group, id-stably, across every store the
    item lives in (spec §5.2)."""

    def __init__(
        self,
        *,
        repo: MemoryRepository,
        bus: EventBusPort,
        settings: PinSettings | None = None,
        clock: Clock | None = None,
        scope_guard: TenancyGuard | None = None,
        scope_factory: ScopeFactory | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._repo = repo
        self._bus = bus
        self._settings = settings or PinSettings()
        self._clock: Clock = clock or SystemClock()
        self._scope_guard: TenancyGuard = scope_guard or DefaultTenancyGuard()
        self._scope_factory: ScopeFactory = scope_factory or self._default_scope_factory
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()

    def _default_scope_factory(self, step: WriteStep) -> IdempotentWriteScope:
        return MemoryWriteScope(self._bus, step)

    async def pin(self, scope: ClientScope, ns: Namespace, req: PinRequest) -> PinResult:
        """Steps 1-4 of spec §5.2, in order: authz -> bound -> id-stable cross-store
        set -> event."""
        return await self._observed(_OP_PIN, ns, self._pin(scope, ns, req))

    async def unpin(self, scope: ClientScope, ns: Namespace, memory_id: str) -> PinResult:
        """Symmetric to :meth:`pin` with ``pinned=False``; clears the whole pin group. No
        max-pins bound (unpinning can only reduce the count)."""
        return await self._observed(_OP_UNPIN, ns, self._unpin(scope, ns, memory_id))

    # ------------------------------------------------------------------ the two write paths --
    async def _pin(self, scope: ClientScope, ns: Namespace, req: PinRequest) -> PinResult:
        self._authorize(scope, ns, _OP_PIN)
        target = await self._resolve(ns, req.memory_id, for_pin=True)
        at = self._clock.now()
        if not target:
            await self._assert_within_pin_bound(ns)
        version = await self._commit(
            ns,
            req.memory_id,
            pinned=True,
            at=at,
            by=scope.principal_id,
            reason=req.reason,
            event=MemoryPinned(namespace=ns, id=req.memory_id, by=scope.principal_id),
        )
        return PinResult(memory_id=req.memory_id, pinned=True, pinned_at=at, version=version)

    async def _unpin(self, scope: ClientScope, ns: Namespace, memory_id: str) -> PinResult:
        self._authorize(scope, ns, _OP_UNPIN)
        await self._resolve(ns, memory_id, for_pin=False)
        at = self._clock.now()
        version = await self._commit(
            ns,
            memory_id,
            pinned=False,
            at=at,
            by=scope.principal_id,
            reason=None,
            event=MemoryUnpinned(namespace=ns, id=memory_id, by=scope.principal_id),
        )
        return PinResult(memory_id=memory_id, pinned=False, pinned_at=None, version=version)

    # ------------------------------------------------------------------------- step 1: authz --
    def _authorize(self, scope: ClientScope, ns: Namespace, operation: str) -> None:
        """Partition ownership ONLY (spec §5.2 step 1). Never an ACL op — pin grants no read.

        ``TenancyGuard`` is the same substrate that isolates every store, so a caller who is not
        the partition owner is refused by the identical mechanism that refuses a cross-tenant
        read. The SHARED-origin clause is layered on top and is OFF by default in v1.
        """
        if not self._settings.enabled:
            raise PinAuthorizationError("pinning is not enabled for this deployment")
        self._scope_guard.assert_scope(scope, ns, operation)
        if ns.visibility is Visibility.SHARED:
            self._refuse_shared_origin_pin()

    def _refuse_shared_origin_pin(self) -> None:
        """Spec §5.2 step 1 line 265 makes SHARED-origin pin a CONJUNCTION: refused unless
        ``PinSettings.allow_shared_origin_pin`` **AND** the caller is the item's origin principal
        (resolved from the provenance ``ORIGIN``, §7.10) **or** a workspace admin.

        The first conjunct is a setting and is checked. The second has **no evaluable input on
        this plane**: ``MemoryItem`` carries no origin principal (only an opaque ``provenance_id``
        handle), mu-core has no provenance-ORIGIN reader, and there is no admin role here —
        workspace roles and the governed origin registry both live in ``mu-server``. An
        un-evaluable conjunct is FALSE, never assumed true, so the refusal stands even with the
        flag ON, and the message says which half is missing.

        This is deliberately NOT solved by inventing a resolver port with no implementer (the
        house absence rule): the capability is reported as unbuilt on this plane rather than
        faked. Flipping the flag alone must never be able to grant pin over another member's
        shared item — which is exactly what a single-conjunct check would have done.
        """
        if not self._settings.allow_shared_origin_pin:
            raise PinAuthorizationError("shared-origin pin is not permitted")
        raise PinAuthorizationError(
            "shared-origin pin needs the origin-principal/workspace-admin check, which mu-core "
            "cannot evaluate (no provenance-ORIGIN reader on this plane)"
        )

    # --------------------------------------------------------- step 2: the pin-explosion bound --
    async def _assert_within_pin_bound(self, ns: Namespace) -> None:
        """Refuse once the partition already holds ``max_pins_per_namespace`` pins.

        Counted with ONE bounded ``enumerate`` page whose ``limit`` is the bound itself + 1 — so
        the check terminates in a single round trip, can never scan the partition, and can still
        distinguish "at the bound" from "under it".
        """
        limit = self._settings.max_pins_per_namespace
        pinned_page, _ = await self._repo.enumerate(
            ns,
            states=PINNABLE_STATES,
            tiers=None,
            pinned=True,
            cursor=None,
            limit=limit + 1,
        )
        if len(pinned_page) >= limit:
            raise PinLimitExceededError("namespace pin limit reached")

    # --------------------------------------------------------------- step 3/4: write + event --
    async def _resolve(self, ns: Namespace, memory_id: str, *, for_pin: bool) -> bool:
        """Return whether the target is ALREADY pinned; raise if it does not resolve or is not
        in a pinnable state.

        Non-enumerating denial: the message never echoes the requested id (the
        ``NamespaceIsolationError`` discipline), so a probe cannot use pin to test for existence.

        The ``PINNABLE_STATES`` check applies to ``pin`` only. UNPIN must stay reachable in every
        state — an item that reached a settled exit while pinned (e.g. an EXPLICIT, owner-driven
        supersede, which CANONICAL §7.10 permits) would otherwise be permanently un-GC-able with
        no way to release it.
        """
        item = await self._repo.get(ns, memory_id)
        if item is None:
            raise PinTargetNotFoundError("not found")
        if for_pin and item.state not in PINNABLE_STATES:
            raise PinTargetNotPinnableError(f"state {item.state.value} is not pinnable")
        return item.pinned

    async def _commit(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        pinned: bool,
        at: datetime,
        by: str,
        reason: str | None,
        event: DomainEvent,
    ) -> int:
        """The id-stable cross-store upsert inside one write scope; the event publishes AFTER."""
        version = 0

        async def step() -> None:
            nonlocal version
            version = await self._repo.set_pinned(
                ns, memory_id, pinned, at=at, by=by, reason=reason
            )

        write_scope = self._scope_factory(step)
        write_scope.add(event)
        await write_scope.commit()
        _log.info(
            "memory_pin_applied",
            ns=ns.to_prefix(),
            memory_id=memory_id,
            pinned=pinned,
            version=version,
        )
        return version

    # ------------------------------------------------------------------------ observability --
    async def _observed(
        self, operation: str, ns: Namespace, coro: Coroutine[Any, Any, PinResult]
    ) -> PinResult:
        """The house envelope (DEV-STANDARDS rule 4), identical in shape to
        ``DemotionService.demote``: content-free span, latency always, error on failure,
        counts-only audit on success. ``CancelledError`` propagates, never counted as a failure.
        """
        started = time.perf_counter()
        with self._tracer.span(operation, attributes={"ns": ns.to_prefix()}):
            try:
                result = await coro
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._metrics.inc(_ERROR_METRIC, labels={"operation": operation})
                raise
            finally:
                self._metrics.observe(
                    _LATENCY_METRIC, time.perf_counter() - started, labels={"operation": operation}
                )
        self._audit.record(
            TraceScope(correlation_id=ns.to_prefix()),
            operation=operation,
            outcome="ok",
            visibility=ns.visibility.value,
            counts={"version": result.version},
        )
        return result
