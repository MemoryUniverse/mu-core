"""EventBusPort — two-plane / two-bus (platform-layer0-spec §8; CANONICAL §4.1).

There is NOT one shared bus: LOCAL and SHARED are separate deployments with separate Redis;
each plane runs its own ``EventBusPort``. Every ``bus.subscribe(...)`` is a subscription to its
OWN plane's bus; a laptop never subscribes to the server's Redis Streams. The envelope is
content-free JSON (ids/hashes/counts/enums/timestamps/namespace-prefix only) — the structural
guard on ``DomainEvent`` (events.py) makes a content-bearing payload unconstructible.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar, runtime_checkable

from mu_contracts.domain.events import DomainEvent
from mu_contracts.ports.health import HealthStatus

__all__ = ["EventBusPort", "Handler", "Subscription"]

E = TypeVar("E", bound=DomainEvent)

# An async, content-free-safe handler for a single event type.
Handler = Callable[[E], Awaitable[None]]


@runtime_checkable
class Subscription(Protocol):
    async def unsubscribe(self) -> None: ...


@runtime_checkable
class EventBusPort(Protocol):
    """A ``LifecycleAdapter`` — the composition root owns its lifetime."""

    async def publish(self, event: DomainEvent) -> None: ...

    def subscribe(self, event_type: type[E], handler: Handler[E]) -> Subscription: ...

    async def start(self) -> None: ...

    async def health(self) -> HealthStatus: ...

    async def close(self) -> None: ...
