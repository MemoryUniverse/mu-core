"""``IdempotentWriteScope`` — the memory-tier write boundary (platform-layer0-spec §10; CANONICAL
§7.8).

STM/MTM/LTM across Redis/Qdrant/FalkorDB: at-least-once + content-hash upsert. NOT a distributed
transaction — NO cross-store rollback is claimed (the retired "false transactional UnitOfWork"
promise, spec §10 / M6). Atomicity comes from idempotent Temporal activities + per-store outbox
retry, never from this scope. Like :class:`~mu_engine.platform.application.unit_of_work.UnitOfWork`
it buffers events and publishes them AFTER the write — but ``commit`` is best-effort-per-store and a
partial failure is retried by the owning activity, not rolled back here.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from mu_contracts.domain.events import DomainEvent
from mu_contracts.ports.bus import EventBusPort

__all__ = ["IdempotentWriteScope", "MemoryWriteScope"]

#: The store-write step this scope wraps (a content-hash upsert across the memory tiers). Made
#: idempotent by the caller (content-addressed), so at-least-once redelivery is safe.
WriteStep = Callable[[], Awaitable[None]]


@runtime_checkable
class IdempotentWriteScope(Protocol):
    def add(self, event: DomainEvent) -> None: ...
    async def commit(self) -> None: ...


class MemoryWriteScope:
    """Reference impl: run the idempotent write step, then publish buffered events (post-write). No
    rollback — a failed write raises and is retried by the owning activity (spec §10)."""

    def __init__(self, bus: EventBusPort, write_step: WriteStep) -> None:
        self._bus = bus
        self._write_step = write_step
        self._buffer: list[DomainEvent] = []

    def add(self, event: DomainEvent) -> None:
        self._buffer.append(event)

    async def commit(self) -> None:
        # Best-effort idempotent upsert; a failure propagates (fail-loud) for activity-level retry.
        await self._write_step()
        events, self._buffer = self._buffer, []
        for event in events:
            await self._bus.publish(event)
