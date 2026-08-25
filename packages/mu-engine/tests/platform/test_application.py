"""Application scopes — UnitOfWork vs IdempotentWriteScope (platform-layer0-spec §10)."""

from __future__ import annotations

import pytest

from mu_contracts.domain.events import DomainEvent
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.application.idempotent_write_scope import (
    IdempotentWriteScope,
    MemoryWriteScope,
)
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


# ── `caused_by_seq` — the loop-suppression causation (CANONICAL:689, O-34) ────────────────────
# These three exist because the field is an ABSENCE-shaped contract: the projector that consumes
# it lives in another repo and drops a mutation when the field is set. If the field were prose —
# a docstring on the Protocol and nothing on the class — every consumer would read `None` forever
# and loop suppression would silently never fire. So each test names what breaks it.


async def _noop() -> None:
    return None


async def test_an_ordinary_mutation_carries_no_causation() -> None:
    """Deleting the ``caused_by_seq=None`` default (or making it required) breaks this.

    ``None`` is what makes a mutation ORIGINAL and therefore projectable. It has to be the
    default, or every existing construction of a first-party write would have to opt out of being
    an echo — and the one that forgot would be dropped by the projector and never replicate.
    """
    scope = MemoryWriteScope(InprocBus(), _noop)
    assert scope.caused_by_seq is None


async def test_a_replica_apply_echo_carries_the_seq_it_applied() -> None:
    """Dropping the constructor keyword — or storing it under another name — breaks this.

    This is the only way the projector can tell a hub-side apply from a hub-side original write.
    """
    scope = MemoryWriteScope(InprocBus(), _noop, caused_by_seq=17)
    assert scope.caused_by_seq == 17


def test_a_scope_without_the_causation_does_not_satisfy_the_protocol() -> None:
    """**The gate that makes the two tests above mean something.**

    ``IdempotentWriteScope`` is ``@runtime_checkable``, so this asserts the field is part of the
    RUNTIME contract rather than a comment on it: a scope with ``add``/``commit`` and no
    ``caused_by_seq`` is NOT an ``IdempotentWriteScope``. Removing the field's declaration from
    the Protocol turns this green-and-meaningless, so it is asserted in both directions.
    """

    class _LegacyScope:
        def add(self, event: DomainEvent) -> None: ...

        async def commit(self) -> None: ...

    assert not isinstance(_LegacyScope(), IdempotentWriteScope)
    assert isinstance(MemoryWriteScope(InprocBus(), _noop), IdempotentWriteScope)
