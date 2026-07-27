"""Application scopes — UnitOfWork vs IdempotentWriteScope (platform-layer0-spec §10)."""

from __future__ import annotations

import pytest

from mu_contracts.domain.events import DomainEvent
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.application.idempotent_write_scope import MemoryWriteScope
from mu_engine.platform.application.unit_of_work import NoWriteUnitOfWork, OutboxUnitOfWork

pytestmark = pytest.mark.unit


class _Ev(DomainEvent):
    tag: str


async def _collect(bus: InprocBus) -> list[str]:
    seen: list[str] = []

    async def handler(event: _Ev) -> None:
        seen.append(event.tag)

    bus.subscribe(_Ev, handler)
    return seen


async def test_outbox_publishes_only_after_commit() -> None:
    bus = InprocBus()
    await bus.start()
    seen = await _collect(bus)
    uow = OutboxUnitOfWork(bus)
    uow.add(_Ev(tag="a"))
    assert seen == []  # buffered, not published
    await uow.commit()
    assert seen == ["a"]  # published post-commit


async def test_outbox_rollback_discards_events() -> None:
    bus = InprocBus()
    await bus.start()
    seen = await _collect(bus)
    uow = OutboxUnitOfWork(bus)
    uow.add(_Ev(tag="x"))
    await uow.rollback()
    await uow.commit()  # nothing buffered now
    assert seen == []


async def test_outbox_context_manager_commits_on_success_rolls_back_on_error() -> None:
    bus = InprocBus()
    await bus.start()
    seen = await _collect(bus)
    async with OutboxUnitOfWork(bus) as uow:
        uow.add(_Ev(tag="ok"))
    assert seen == ["ok"]

    with pytest.raises(RuntimeError):
        async with OutboxUnitOfWork(bus) as uow:
            uow.add(_Ev(tag="bad"))
            raise RuntimeError("boom")
    assert seen == ["ok"]  # the failed transaction leaked no event


async def test_no_write_uow_forbids_writes() -> None:
    uow = NoWriteUnitOfWork()
    with pytest.raises(RuntimeError, match="side-effect-free"):
        uow.add(_Ev(tag="x"))
    with pytest.raises(RuntimeError, match="side-effect-free"):
        await uow.commit()


async def test_idempotent_write_scope_publishes_after_write_step() -> None:
    bus = InprocBus()
    await bus.start()
    seen = await _collect(bus)
    order: list[str] = []

    async def write_step() -> None:
        order.append("write")

    scope = MemoryWriteScope(bus, write_step)
    scope.add(_Ev(tag="m"))
    await scope.commit()
    assert order == ["write"]
    assert seen == ["m"]  # published post-write


async def test_idempotent_write_scope_failed_write_publishes_nothing() -> None:
    bus = InprocBus()
    await bus.start()
    seen = await _collect(bus)

    async def failing_step() -> None:
        raise RuntimeError("store down")

    scope = MemoryWriteScope(bus, failing_step)
    scope.add(_Ev(tag="m"))
    with pytest.raises(RuntimeError):
        await scope.commit()
    assert seen == []  # write failed -> no publish (retried by owning activity)
