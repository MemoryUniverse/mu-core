"""Composition roots — ``Container`` base + ``LocalContainer`` (platform-layer0-spec §13).

The app-scope root wires the Layer-0 singletons through the registries + builders (DEV-STANDARDS
rule 9: DI at composition roots; no global singletons reached into from deep code). ``startup()`` /
``shutdown()`` delegate to the reused :class:`LifecycleManager` (topological start, health-gated
required deps, reverse-order close).

CONFIG SEAM (parallel-build): the Layer-0 SELECTORS (which bus/workflow backend, which observability
sinks) live in the ``settings.bus`` / ``settings.workflow`` / ``settings.observability`` subtrees
(spec §1.3) that the contracts CONFIG phase is still landing. Until they exist on the ``Settings``
type, the root takes an explicit :class:`PlatformSelectors` — still config-sourced at the call site,
NOT hardcoded. :meth:`Container.from_settings` is the single reader that maps those subtrees to
selectors once they land (tracked gap; no behaviour change to the wiring below).

``SharedContainer`` (real Temporal, managed stores, the SHARED-plane projectors,
tuned strategy substitution) lives in ``mu-server`` (spec §13.2, ADR 0011) and is out of this
package's scope — the physical private/shared boundary is structural (a shared process never imports
a local store adapter).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mu_contracts.config.settings import Settings
from mu_contracts.domain.events import DegradedModeEntered
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.bus import EventBusPort, Subscription
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_contracts.ports.workflow import WorkflowRunnerPort
from mu_engine.platform import adapters as _adapters  # noqa: F401  (runs registration decorators)
from mu_engine.platform.application.unit_of_work import NoWriteUnitOfWork, OutboxUnitOfWork
from mu_engine.platform.clock import build_clock
from mu_engine.platform.degradation import (
    DegradationPolicy,
    SyncStatusSink,
    assert_sync_status_wired,
)
from mu_engine.platform.lifecycle import LifecycleManager, LifecycleSpec, ReadinessReport
from mu_engine.platform.observability import build_audit, build_metrics, build_tracer
from mu_engine.platform.registries import bus_registry, workflow_registry
from mu_engine.platform.tenancy import DefaultTenancyGuard

__all__ = ["Container", "LocalContainer", "PlatformSelectors"]

_LOCAL_WORKFLOW_BACKEND = "inline"  # PINNED on LOCAL (spec §9 M3): no Temporal on the client.


class PlatformSelectors(BaseModel):
    """The Layer-0 wiring selectors — sourced from ``settings.{bus,workflow,observability}`` at the
    call site (the config subtrees, spec §1.3). Required (no defaults) so no backend is silently
    assumed (DEV-STANDARDS rule 3: no hardcoding)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bus_backend: str
    workflow_backend: str
    otel_enabled: bool
    metrics_enabled: bool
    audit_enabled: bool


class Container:
    """The CORE base composition root (spec §13.2). Wires ONLY core/engine Layer-0 adapters."""

    def __init__(self, settings: Settings, selectors: PlatformSelectors) -> None:
        self.settings = settings
        self.selectors = selectors
        self.clock: Clock = build_clock(settings)
        self.tracer: Tracer = build_tracer(enabled=selectors.otel_enabled)
        self.metrics: MetricSink = build_metrics(enabled=selectors.metrics_enabled)
        self.audit: AuditLog = build_audit(enabled=selectors.audit_enabled)
        self.bus: EventBusPort = bus_registry.create(selectors.bus_backend, settings)
        self.workflow: WorkflowRunnerPort = workflow_registry.create(
            selectors.workflow_backend, settings
        )
        self.tenancy = DefaultTenancyGuard()
        self.degradation = DegradationPolicy()
        self._sync_status: SyncStatusSink | None = None
        self._subscriptions: list[Subscription] = []
        self.lifecycle = LifecycleManager(self._lifecycle_specs())
        self._wire_event_handlers()

    # ── wiring ────────────────────────────────────────────────────────────────────────────
    def _lifecycle_specs(self) -> list[LifecycleSpec]:
        """The app-scope adapters the manager owns. The bus is a required ``LifecycleAdapter`` (it
        exposes start/health/close, spec §8.1). Stores/LLM/graph are wired by their own phases."""
        return [LifecycleSpec(name="bus", adapter=self.bus, required=True)]

    def _wire_event_handlers(self) -> None:
        """Subscribe the degrade fan-out (spec §12): operator metric ALWAYS; SYNC-CLASS also to the
        SyncStatusSink (a SYNC-CLASS degrade with no sink wired raises inside ``fan_out``)."""

        async def _on_degraded(event: DegradedModeEntered) -> None:
            self.degradation.fan_out(event, metrics=self.metrics, sync_status=self._sync_status)

        self._subscriptions.append(self.bus.subscribe(DegradedModeEntered, _on_degraded))

    def set_sync_status(self, sink: SyncStatusSink) -> None:
        """Wire the user-facing sync-status projector (SHARED/LOCAL planes call this, spec §12)."""
        self._sync_status = sink

    # ── lifecycle ─────────────────────────────────────────────────────────────────────────
    async def startup(self) -> ReadinessReport:
        return await self.lifecycle.start()

    async def shutdown(self) -> None:
        await self.lifecycle.close()

    # ── scopes (spec §13.1) ─────────────────────────────────────────────────────────────────
    def read_uow(self) -> NoWriteUnitOfWork:
        """The side-effect-free read scope (CQRS-lite, spec §10/§13.1)."""
        return NoWriteUnitOfWork()

    def write_uow(self) -> OutboxUnitOfWork:
        """The transactional write scope (buffer-then-publish, spec §10)."""
        return OutboxUnitOfWork(self.bus)

    def request_scope(self, scope: ClientScope) -> ClientScope:
        """Carry the acting identity for a request (spec §13.1). Layer-0 only threads it; the api
        layer resolves it and the repos pass it to :meth:`DefaultTenancyGuard.assert_scope`."""
        return scope


class LocalContainer(Container):
    """LOCAL plane root (spec §13.2). Pins the ``inline`` workflow runner (M3 — no Temporal on the
    client) regardless of the selector; the local-plane store/daemon adapters are added by the
    ``mu-client`` package (out of ``mu-core`` scope)."""

    def __init__(self, settings: Settings, selectors: PlatformSelectors) -> None:
        super().__init__(settings, selectors)
        # PINNED: the client never runs Temporal (spec §9 M3). Override any selector.
        self.workflow = workflow_registry.create(_LOCAL_WORKFLOW_BACKEND, settings)

    def assert_ready_to_start(self) -> None:
        """A LOCAL plane emits SYNC-CLASS degrades; it must wire a
        SyncStatusSink so they reach the user (spec §12)."""
        assert_sync_status_wired(self._sync_status)
