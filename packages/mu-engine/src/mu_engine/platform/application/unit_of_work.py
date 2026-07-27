"""``UnitOfWork`` — the TRUE transactional boundary (platform-layer0-spec §10; CANONICAL §7.8).

Postgres control/governance path ONLY (grants, provenance ledger, ACL rows). Transactional outbox:
buffered events publish only AFTER the commit — a rollback rolls back the event too (spec §10).

Two sibling scopes:
  * :class:`OutboxUnitOfWork` — the buffer-then-publish reference impl (the real Postgres impl
    lands with the control-plane phase; this one demonstrates the discipline over any
    ``EventBusPort``: nothing is published until ``commit``, ``rollback`` discards the buffer).
  * :class:`NoWriteUnitOfWork` — the read-path scope (CQRS-lite, spec §10): recall is side-effect
    free, so ``add``/``commit`` are FORBIDDEN (raise), not silently ignored.

The memory-tier write boundary is the SEPARATE :class:`~mu_engine.platform.application.\
idempotent_write_scope.IdempotentWriteScope` — the "false transactional promise" for multi-store
memory writes is resolved by keeping the two paths distinct (spec §10, M6).
"""

from __future__ import annotations

from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from mu_contracts.domain.events import DomainEvent
from mu_contracts.ports.bus import EventBusPort

__all__ = ["NoWriteUnitOfWork", "OutboxUnitOfWork", "UnitOfWork"]


@runtime_checkable
class UnitOfWork(Protocol):
    """A transactional scope: buffer events, publish on commit, discard on rollback (spec §10)."""

    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...
    def add(self, event: DomainEvent) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...


class OutboxUnitOfWork:
    """Buffer-then-publish reference impl (spec §10 path a). Events publish only AFTER ``commit``;
    ``rollback`` (or an exception in the ``async with`` body) discards them — a rollback never leaks
    an event on THIS transactional path."""

    def __init__(self, bus: EventBusPort) -> None:
        self._bus = bus
        self._buffer: list[DomainEvent] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            await self.rollback()
        else:
            await self.commit()

    def add(self, event: DomainEvent) -> None:
        self._buffer.append(event)

    async def commit(self) -> None:
        # A real impl commits the Postgres tx HERE, then publishes; the ordering is the contract.
        events, self._buffer = self._buffer, []
        for event in events:
            await self._bus.publish(event)

    async def rollback(self) -> None:
        self._buffer.clear()


class NoWriteUnitOfWork:
    """The read-path scope (CQRS-lite). Recall is side-effect-free — ``add``/``commit`` are
    forbidden so a read handler cannot accidentally emit or persist."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def add(self, event: DomainEvent) -> None:
        del event
        raise RuntimeError("read scope is side-effect-free; add() is forbidden (spec §10)")

    async def commit(self) -> None:
        raise RuntimeError("read scope is side-effect-free; commit() is forbidden (spec §10)")

    async def rollback(self) -> None:
        return None
