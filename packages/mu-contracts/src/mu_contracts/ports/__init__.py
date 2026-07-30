"""Ports — the ``typing.Protocol`` hexagon edges (platform-layer0-spec §0.2; PACKAGING-v2 §1.3).

Repository pattern (DEV-STANDARDS rule 5): the domain talks to these Protocols; concrete store
adapters live behind them in mu-engine. Every port is a structural ``Protocol`` (independently
testable, adapter-swappable). Re-exported with an explicit ``__all__`` (``no_implicit_reexport``).
"""

from mu_contracts.ports.bus import EventBusPort, Handler, Subscription
from mu_contracts.ports.conflict import ConflictEdgeReader
from mu_contracts.ports.device import ClientMetadata, DeviceRegistryPort
from mu_contracts.ports.device_sync import PrivateSyncLogPort
from mu_contracts.ports.governance import ConflictRecordRepository, GrantRepository
from mu_contracts.ports.health import HealthStatus
from mu_contracts.ports.lifecycle_lease import LifecycleLeasePort
from mu_contracts.ports.lifecycle_workflow import LifecycleWorkflowRunnerPort
from mu_contracts.ports.memory import (
    LtmTierRepository,
    MemoryRepository,
    MemoryTierRepository,
    MtmTierRepository,
    StmTierRepository,
)
from mu_contracts.ports.model import EmbeddingPort, LLMProviderPort
from mu_contracts.ports.notification import (
    NotificationPreferenceRepository,
    NotificationRepository,
)
from mu_contracts.ports.observability import (
    AuditLog,
    MetricSink,
    SpanCtx,
    Tracer,
    TurnTraceEvent,
    TurnTraceScope,
)
from mu_contracts.ports.persona import PersonaRepository
from mu_contracts.ports.security import TenancyGuard
from mu_contracts.ports.stores import (
    EdgeSpec,
    GraphNodeRow,
    QdrantPoint,
    RedisRecord,
    RelationalRow,
    RowMapper,
    StoreModel,
)
from mu_contracts.ports.time import Clock
from mu_contracts.ports.workflow import WorkflowHandle, WorkflowRunnerPort

__all__ = [
    "AuditLog",
    "ClientMetadata",
    "Clock",
    "ConflictEdgeReader",
    "ConflictRecordRepository",
    "DeviceRegistryPort",
    "EdgeSpec",
    "EmbeddingPort",
    "EventBusPort",
    "GrantRepository",
    "GraphNodeRow",
    "Handler",
    "HealthStatus",
    "LLMProviderPort",
    "LifecycleLeasePort",
    "LifecycleWorkflowRunnerPort",
    "LtmTierRepository",
    "MemoryRepository",
    "MemoryTierRepository",
    "MetricSink",
    "MtmTierRepository",
    "NotificationPreferenceRepository",
    "NotificationRepository",
    "PersonaRepository",
    "PrivateSyncLogPort",
    "QdrantPoint",
    "RedisRecord",
    "RelationalRow",
    "RowMapper",
    "SpanCtx",
    "StmTierRepository",
    "StoreModel",
    "Subscription",
    "TenancyGuard",
    "Tracer",
    "TurnTraceEvent",
    "TurnTraceScope",
    "WorkflowHandle",
    "WorkflowRunnerPort",
]
