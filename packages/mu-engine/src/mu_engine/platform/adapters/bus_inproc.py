"""In-process ``EventBusPort`` adapter (platform-layer0-spec §8.3 — the ``inproc`` backend).

For tests + single-process wiring: an ``asyncio``-native bus that dispatches a published
``DomainEvent`` to every handler subscribed to its type (or a base type). It is a
``LifecycleAdapter`` (start/health/close) so the composition root owns its lifetime (spec §8.1).

This is the test/dev backend; the durable default is the Redis-Streams adapter (spec §8.3,
consumer-group ``XREADGROUP``/``XACK``, ``XAUTOCLAIM`` recovery, ``MINID`` trim) landing with the
storage/bus phase against the real ``mu-dev-cache`` container. The two-plane invariant (separate
Redis per plane, spec §8.2) is a wiring rule the composition roots enforce; a single ``InprocBus``
instance is one plane's bus.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar, cast

from mu_contracts.domain.events import DomainEvent
from mu_contracts.ports.bus import EventBusPort, Handler, Subscription
from mu_contracts.ports.health import HealthStatus
from mu_engine.platform.registries import bus_registry

__all__ = ["InprocBus"]

_AnyHandler = Callable[[DomainEvent], Awaitable[None]]
E = TypeVar("E", bound=DomainEvent)


class _Subscription:
    def __init__(self, bus: InprocBus, event_type: type[DomainEvent], handler: _AnyHandler) -> None:
        self._bus = bus
        self._event_type = event_type
        self._handler = handler

    async def unsubscribe(self) -> None:
        self._bus._remove(self._event_type, self._handler)


class InprocBus:
    """A minimal, deterministic in-process bus. Handlers fire sequentially in subscription order;
    a handler exception propagates to the publisher (fail-loud — no silent swallow)."""

    def __init__(self) -> None:
        self._handlers: dict[type[DomainEvent], list[_AnyHandler]] = {}
        self._lock = asyncio.Lock()
        self._started = False

    async def publish(self, event: DomainEvent) -> None:
        # Snapshot under the lock; dispatch outside it so a handler may (un)subscribe safely.
        async with self._lock:
            targets = [
                handler
                for event_type, handlers in self._handlers.items()
                if isinstance(event, event_type)
                for handler in handlers
            ]
        for handler in targets:
            await handler(event)

    def subscribe(self, event_type: type[E], handler: Handler[E]) -> Subscription:
        typed = cast(_AnyHandler, handler)
        self._handlers.setdefault(event_type, []).append(typed)
        return _Subscription(self, event_type, typed)

    def _remove(self, event_type: type[DomainEvent], handler: _AnyHandler) -> None:
        handlers = self._handlers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)

    async def start(self) -> None:
        self._started = True

    async def health(self) -> HealthStatus:
        return HealthStatus(component="bus", healthy=self._started)

    async def close(self) -> None:
        self._started = False
        self._handlers.clear()


@bus_registry.register("inproc")
def _build_inproc_bus(settings: object) -> EventBusPort:
    del settings  # inproc needs no config
    return InprocBus()
