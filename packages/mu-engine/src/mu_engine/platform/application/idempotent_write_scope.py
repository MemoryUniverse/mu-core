"""``IdempotentWriteScope`` — the memory-tier write boundary (platform-layer0-spec §10; CANONICAL
§7.8).

STM/MTM/LTM across Redis/Qdrant/FalkorDB: at-least-once + content-hash upsert. NOT a distributed
transaction — NO cross-store rollback is claimed (the retired "false transactional UnitOfWork"
promise, spec §10 / M6). Atomicity comes from idempotent Temporal activities + per-store outbox
retry, never from this scope. Like :class:`~mu_engine.platform.application.unit_of_work.UnitOfWork`
it buffers events and publishes them AFTER the write — but ``commit`` is best-effort-per-store and a
partial failure is retried by the owning activity, not rolled back here.

**The scope also carries the mutation's CAUSATION, and that is a contract, not a convenience**
(``CANONICAL-CONTRACTS.md:689``, §7.14). See :attr:`IdempotentWriteScope.caused_by_seq`.
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
    """The memory-tier write boundary: buffer events, run the idempotent write, publish after.

    ⚠ **``caused_by_seq`` is part of the Protocol, and its absence is what made loop suppression
    unimplementable.** ``CANONICAL-CONTRACTS.md:689`` pins the sync-apply causation *"on the
    mutation's ``IdempotentWriteScope`` context"*: a hub-side mutation that IS the application of
    a ``PrivateDelta`` carries that delta's ``seq`` here, and the private-delta projector
    (appender B, ``mu-server``) **DROPS** such a mutation instead of projecting it back into the
    log. Without the field, applying one delta on the hub re-projects it, the re-projection
    applies, and one write oscillates forever between the engine and the log — an unbounded
    ``seq`` burn that reads as a sync storm rather than as a bug.

    **Why here and not on ``DomainEvent`` (O-34, closed 2026-08-25).** An earlier decision (D-15)
    put the field on the event base so that a bus-subscribed projector could see it. CANONICAL is
    rank-above-all and puts it on the scope — and is also right on the merits: ``DomainEvent`` is
    the base of a ~100-member catalog including LOCAL-plane events that can never have a hub
    ``seq``, so a nullable field there would be a capability the vocabulary claims and the system
    does not have. ADR 0045 then removed the reason to prefer the event anyway: appender B no
    longer subscribes to a bus, it runs inside the mutation's own write transaction, where the
    scope is in hand. **D-15 is superseded; the projector reads the scope.**
    """

    #: ``None`` for an ORIGINAL mutation — the overwhelmingly common case, and the default every
    #: implementation must keep. An ``int`` means *"this mutation is the replica-apply echo of
    #: private-sync-log ``seq`` N"*, and it is set by the ONE caller applying a delta on the hub.
    #: It is a plain ``int``, so it passes the content-free discipline unchanged.
    caused_by_seq: int | None

    def add(self, event: DomainEvent) -> None: ...
    async def commit(self) -> None: ...


class MemoryWriteScope:
    """Reference impl: run the idempotent write step, then publish buffered events (post-write). No
    rollback — a failed write raises and is retried by the owning activity (spec §10)."""

    def __init__(
        self, bus: EventBusPort, write_step: WriteStep, *, caused_by_seq: int | None = None
    ) -> None:
        self._bus = bus
        self._write_step = write_step
        self._buffer: list[DomainEvent] = []
        #: Keyword-only and defaulted to ``None`` so that every existing construction of an
        #: ORIGINAL mutation stays correct without an edit, and so that declaring a mutation to be
        #: a replica-apply echo is always an explicit, greppable act at the one call site that
        #: applies deltas. See :class:`IdempotentWriteScope` for why it lives here at all.
        self.caused_by_seq = caused_by_seq

    def add(self, event: DomainEvent) -> None:
        self._buffer.append(event)

    async def commit(self) -> None:
        # Best-effort idempotent upsert; a failure propagates (fail-loud) for activity-level retry.
        await self._write_step()
        events, self._buffer = self._buffer, []
        for event in events:
            await self._bus.publish(event)
