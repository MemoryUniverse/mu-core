"""Inproc bus + inline workflow adapters (platform-layer0-spec §8.3/§9, §15.5)."""

from __future__ import annotations

import pytest

from mu_contracts.domain.events import DomainEvent
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.adapters.workflow_inline import InlineRunner
from mu_engine.platform.registries import bus_registry, workflow_registry

pytestmark = pytest.mark.unit


class _Ping(DomainEvent):
    n: int


class _Other(DomainEvent):
    k: str


async def test_publish_delivers_to_subscribers() -> None:
    bus = InprocBus()
    await bus.start()
    seen: list[int] = []

    async def handler(event: _Ping) -> None:
        seen.append(event.n)

    bus.subscribe(_Ping, handler)
    await bus.publish(_Ping(n=1))
    await bus.publish(_Other(k="x"))  # not a _Ping subscriber
    await bus.publish(_Ping(n=2))
    assert seen == [1, 2]


async def test_unsubscribe_stops_delivery() -> None:
    bus = InprocBus()
    await bus.start()
    seen: list[int] = []

    async def handler(event: _Ping) -> None:
        seen.append(event.n)

    sub = bus.subscribe(_Ping, handler)
    await bus.publish(_Ping(n=1))
    await sub.unsubscribe()
    await bus.publish(_Ping(n=2))
    assert seen == [1]


async def test_bus_health_reflects_lifecycle() -> None:
    bus = InprocBus()
    assert (await bus.health()).healthy is False
    await bus.start()
    assert (await bus.health()).healthy is True
    await bus.close()
    assert (await bus.health()).healthy is False


def test_inproc_bus_registered() -> None:
    assert bus_registry.is_registered("inproc")


async def test_inline_runner_executes_registered_workflow() -> None:
    runner = InlineRunner()

    async def double(arg: object) -> object:
        assert isinstance(arg, int)
        return arg * 2

    runner.register("double", double)
    handle = await runner.start("double", 21, id="wf-1", task_queue="q")
    assert handle.id == "wf-1"
    assert await handle.result() == 42
    assert await runner.execute("double", 5, id="wf-2", task_queue="q") == 10
    await runner.readiness()  # always ready


async def test_inline_runner_unknown_workflow_raises() -> None:
    runner = InlineRunner()
    with pytest.raises(ValueError, match="unknown inline workflow"):
        await runner.execute("nope", None, id="wf", task_queue="q")


def test_inline_runner_registered() -> None:
    assert workflow_registry.is_registered("inline")
