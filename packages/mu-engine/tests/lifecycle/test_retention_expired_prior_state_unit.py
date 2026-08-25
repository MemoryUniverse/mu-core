"""ADR 0049 — ``RetentionService``'s ``_to_contract_state`` publishes the REAL prior state on
``MemoryGarbageCollected``, not a lossy substitute (S2-01 follow-up; CANONICAL §7.5 "SIX
members").

Pure unit tests of isolated logic against an in-memory ``LtmRetentionStorePort`` double — the
same sanctioned DEV-STANDARDS exception ``test_retention_int.py``'s
``test_ac22_pinned_mechanism_never_archived_or_gcd`` already uses (see that file's module
docstring). No real store, no ``pytest.mark.integration`` — fast + local, no external
container.

Covers:
- an ``EXPIRED`` fact GC'd through a REAL ``RetentionService.sweep()`` publishes
  ``MemoryGarbageCollected.prior_state == State.EXPIRED`` — the defect ADR 0049 fixes
  (before the fix this was ``State.ARCHIVED``, a death that never happened).
- a ``SUPERSEDED`` fact GC'd the same way still publishes ``State.SUPERSEDED`` — the neighbour
  the fix must not break.
- ``_to_contract_state(MemoryState.ARCHIVED) == State.ARCHIVED`` — asserted directly against the
  private mapper rather than through ``sweep()``, because ``_DEAD_STATES`` (retention.py:108) is
  ``frozenset({SUPERSEDED, EXPIRED})`` only: an ``ARCHIVED`` item is structurally unreachable by
  the GC pass (``facts_by_state(ns, _DEAD_STATES)`` never returns one), so there is no real-flow
  seam to exercise it through. Calling the ``@staticmethod`` directly is the only way to prove
  the ARCHIVED branch of the mapping still holds after the EXPIRED fix, and is safe here because
  the method takes/returns nothing but plain enum values (no collaborator behaviour hidden
  behind it).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mu_contracts.domain.events import MemoryGarbageCollected
from mu_contracts.domain.model.memory import State as ContractState
from mu_engine.lifecycle.retention import RetentionService
from mu_engine.lifecycle.settings import LifecycleSettings, RetentionSettings
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.domain.memory import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryState,
    MemoryTier,
    RetentionClass,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


class _InMemoryLtmDouble:
    """A REAL, fully-functional in-memory stand-in (DEV-STANDARDS' unit-test exception) —
    mirrors ``test_retention_int.py``'s ``_InMemoryLtmDouble`` exactly (chain head = the item's
    own state; no separate chain modeled, which is fine here since every item under test is its
    own chain head)."""

    def __init__(self, items: dict[str, MemoryItem]) -> None:
        self.items = items

    async def upsert_fact(self, item: MemoryItem) -> None:
        self.items[item.id] = item

    async def facts_by_state(
        self, ns: Namespace, states: frozenset[MemoryState]
    ) -> list[MemoryItem]:
        del ns
        return [i for i in self.items.values() if i.state in states]

    async def chain_head_state(self, ns: Namespace, memory_id: str) -> MemoryState:
        del ns
        return self.items[memory_id].state

    async def gc_delete(self, ns: Namespace, memory_id: str) -> None:
        del ns
        del self.items[memory_id]


def _dead_fact(ns: Namespace, content: str, *, state: MemoryState) -> MemoryItem:
    return MemoryItem(
        content=content,
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.LTM,
        state=state,
        source=MemorySource.USER,
        subject="Ada",
        predicate="fact",
        object=content,
        retention_class=RetentionClass.DURABLE,
        importance_score=0.5,
        valid_at=_T0,
        invalid_at=_T0,  # died long ago
        created_at=_T0,
        updated_at=_T0,
    )


async def test_gc_publishes_the_real_prior_state_for_expired_and_superseded() -> None:
    """ADR 0049's own reachability trace, exercised for real: an EXPIRED item and a SUPERSEDED
    item both sit in ``_DEAD_STATES``, both clear the window + chain-head-dead gates, both reach
    ``self._to_contract_state(item.state)`` at the real GC publish call site (retention.py's
    pass-2 loop) — proving the fix end-to-end through ``RetentionService.sweep()``, not just
    against the mapper in isolation.
    """
    ns = Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )
    expired = _dead_fact(ns, "trip to Berlin ended", state=MemoryState.EXPIRED)
    superseded = _dead_fact(ns, "Ada at Acme (superseded)", state=MemoryState.SUPERSEDED)
    double = _InMemoryLtmDouble({expired.id: expired, superseded.id: superseded})

    # far past any reasonable gc_history_window_d so the window gate never blocks the GC.
    now = _T0 + timedelta(days=1000)
    clock = FrozenClock(now)
    settings = LifecycleSettings(retention=RetentionSettings(gc_history_window_d=0))
    bus = InprocBus()
    await bus.start()
    published: list[MemoryGarbageCollected] = []

    async def _capture(event: MemoryGarbageCollected) -> None:
        published.append(event)

    bus.subscribe(MemoryGarbageCollected, _capture)

    service = RetentionService(
        ltm=double,  # type: ignore[arg-type]
        ltm_retention=double,
        settings=settings,
        clock=clock,
        bus=bus,
    )
    acted_on = await service.sweep(ns, clock=clock)

    assert acted_on == 2
    assert double.items == {}  # both GC'd — real DETACH-DELETE-equivalent on the double
    by_id = {event.id: event for event in published}
    assert len(by_id) == 2
    assert by_id[expired.id].prior_state is ContractState.EXPIRED  # the ADR 0049 defect fix
    assert by_id[superseded.id].prior_state is ContractState.SUPERSEDED  # neighbour, unbroken


def test_to_contract_state_archived_neighbour_unbroken() -> None:
    """``_to_contract_state(MemoryState.ARCHIVED)`` still maps to ``State.ARCHIVED`` after the
    EXPIRED fix. Asserted directly against the private mapper (see module docstring for why: an
    ARCHIVED item is structurally unreachable through ``sweep()``'s GC pass — ``_DEAD_STATES``
    does not include it, so there is no real-flow seam)."""
    assert RetentionService._to_contract_state(MemoryState.ARCHIVED) is ContractState.ARCHIVED


def test_to_contract_state_expired_maps_to_expired_directly() -> None:
    """Direct mapper-level pin of the fixed branch, alongside the real-flow proof above."""
    assert RetentionService._to_contract_state(MemoryState.EXPIRED) is ContractState.EXPIRED
