"""mu-engine Layer-0 platform (platform-layer0-spec.md).

The engine half of Layer-0: the fail-loud :class:`Registry`, content-free observability sinks +
``SafeTraceFields``, cross-cutting decorators (retry/timing/guard), the central exception→envelope
mapping, the persistence-level :class:`DefaultTenancyGuard`, the :class:`DegradationPolicy` two-
consumer fan-out, the application scopes (``UnitOfWork`` / ``IdempotentWriteScope``), the
``LifecycleManager``, the ``Clock`` adapters, the ``inproc``/``inline`` bus+workflow adapters, and
the ``Container`` / ``LocalContainer`` composition roots. All the shared VOCABULARY it consumes
(``Namespace``/``ClientScope``/errors/events/ports/``Settings``) is owned by ``mu-contracts``.
"""

from __future__ import annotations

from mu_engine.platform.clock import FrozenClock, OffsetClock, SystemClock, build_clock
from mu_engine.platform.composition import Container, LocalContainer, PlatformSelectors
from mu_engine.platform.decorators import guard, retry_io, timed
from mu_engine.platform.degradation import (
    DegradationPolicy,
    DegradedDecision,
    SyncStatusSink,
    assert_sync_status_wired,
)
from mu_engine.platform.exceptions import (
    ErrorCode,
    ErrorEnvelope,
    RetryClass,
    classify_error,
    safe_error_response,
    to_envelope,
)
from mu_engine.platform.lifecycle import (
    LifecycleAdapter,
    LifecycleManager,
    LifecycleSpec,
    ReadinessReport,
)
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    SafeTraceFields,
    build_audit,
    build_metrics,
    build_tracer,
    sanitize_label_value,
    sanitize_labels,
)
from mu_engine.platform.registries import (
    algorithm_registry,
    bus_registry,
    provider_registry,
    store_registry,
    workflow_registry,
)
from mu_engine.platform.registry import Registry
from mu_engine.platform.tenancy import DefaultTenancyGuard

__all__ = [
    "Container",
    "DefaultTenancyGuard",
    "DegradationPolicy",
    "DegradedDecision",
    "ErrorCode",
    "ErrorEnvelope",
    "FrozenClock",
    "LifecycleAdapter",
    "LifecycleManager",
    "LifecycleSpec",
    "LocalContainer",
    "NoopAuditLog",
    "NoopMetricSink",
    "NoopTracer",
    "OffsetClock",
    "PlatformSelectors",
    "ReadinessReport",
    "Registry",
    "RetryClass",
    "SafeTraceFields",
    "SyncStatusSink",
    "SystemClock",
    "algorithm_registry",
    "assert_sync_status_wired",
    "build_audit",
    "build_clock",
    "build_metrics",
    "build_tracer",
    "bus_registry",
    "classify_error",
    "guard",
    "provider_registry",
    "retry_io",
    "safe_error_response",
    "sanitize_label_value",
    "sanitize_labels",
    "store_registry",
    "timed",
    "to_envelope",
    "workflow_registry",
]
