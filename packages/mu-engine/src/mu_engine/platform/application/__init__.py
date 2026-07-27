"""Layer-0 application scopes (platform-layer0-spec §10)."""

from __future__ import annotations

from mu_engine.platform.application.idempotent_write_scope import (
    IdempotentWriteScope,
    MemoryWriteScope,
)
from mu_engine.platform.application.unit_of_work import (
    NoWriteUnitOfWork,
    OutboxUnitOfWork,
    UnitOfWork,
)

__all__ = [
    "IdempotentWriteScope",
    "MemoryWriteScope",
    "NoWriteUnitOfWork",
    "OutboxUnitOfWork",
    "UnitOfWork",
]
