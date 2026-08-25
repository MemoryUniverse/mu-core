"""Pins the GC-ordering invariant ADR 0049 introduced in ``RetentionService._sweep``'s pass 2
(Task D; ``retention.py`` commit ``616d953``): ``prior_state`` MUST be computed via
``self._to_contract_state(item.state)`` BEFORE ``self._ltm_retention.gc_delete(...)`` runs, never
after.

**Why this needs its own test.** ``_STATE_MAP`` is exhaustive over all six current
``MemoryState`` members (see ``retention.py``'s ``_STATE_MAP`` docstring), so
``_to_contract_state`` cannot actually raise ``KeyError`` today — the ordering fix has no
reachable failure to exercise it through. Without a test pinning the ORDER itself, a future
refactor could silently swap the two statements back to the pre-0049 (destructive) order and
nothing in the suite would fail, because no real ``MemoryState`` member is missing from the map.

**How this test forces the unreachable branch reachable.** It monkeypatches
``RetentionService._STATE_MAP`` (a ``ClassVar[dict[...]]`` — a class attribute, so
``monkeypatch.setattr`` restores it automatically at teardown) to a copy missing the one member
the fixture's dead fact carries — simulating exactly the "engine gains a member, the map lags"
scenario ``_STATE_MAP``'s own docstring calls out, without touching the real ``MemoryState`` enum
(never sanctioned — the task brief is explicit: mutate the map, not the enum).

Pure unit test of isolated logic against an in-memory ``LtmRetentionStorePort`` double — the same
sanctioned DEV-STANDARDS exception ``test_retention_int.py``'s
``test_ac22_pinned_mechanism_never_archived_or_gcd`` and this directory's
``test_retention_expired_prior_state_unit.py`` already use. No real store, no
``pytest.mark.integration`` — fast + local, no external container.

Covers:
- ``test_gc_raises_before_delete_when_mapping_is_unmapped`` — the pin itself: with ``EXPIRED``
  removed from a monkeypatched ``_STATE_MAP``, ``sweep()`` raises ``KeyError`` AND the spy proves
  ``gc_delete`` was never called for the unmapped item. This is the assertion that fails if a
  future refactor reorders the two statements back to compute ``prior_state`` after
  ``gc_delete`` — the delete would happen and only THEN would the (still-monkeypatched) mapping
  raise, so ``gc_delete`` calls would be non-empty and this test would catch it.
- ``test_gc_delete_spy_records_a_call_in_the_success_path`` — non-vacuity proof for the first
  test's negative assertion: the exact same double/item/settings, run WITHOUT the monkeypatch,
  really does append to the spy's call list. This rules out the first test's ``deleted == []``
  passing merely because the spy is wired wrong (e.g. never invoked at all, or invoked on the
  wrong object) rather than because the ordering held.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_engine.lifecycle.retention import RetentionService
from mu_engine.lifecycle.settings import LifecycleSettings, RetentionSettings
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

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


class _SpyLtmDouble:
    """A REAL, fully-functional in-memory ``LtmRetentionStorePort`` double (DEV-STANDARDS'
    sanctioned unit-test exception) — shape mirrors ``test_retention_int.py``'s
    ``_InMemoryLtmDouble`` (including its ``deleted`` spy list), with chain head = the item's own
    state (every item under test is its own chain head, so no separate chain needs modeling).
    """

    def __init__(self, items: dict[str, MemoryItem]) -> None:
        self.items = items
        self.deleted: list[str] = []  # the spy: gc_delete call log, in call order

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
        self.deleted.append(memory_id)
        del self.items[memory_id]


def _dead_fact(ns: Namespace) -> MemoryItem:
    return MemoryItem(
        content="trip to Berlin ended",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.LTM,
        state=MemoryState.EXPIRED,
        source=MemorySource.USER,
        subject="Ada",
        predicate="fact",
        object="trip to Berlin ended",
        retention_class=RetentionClass.DURABLE,
        importance_score=0.5,
        valid_at=_T0,
        invalid_at=_T0,  # died long ago
        created_at=_T0,
        updated_at=_T0,
    )


def _make_service_and_double(
    ns: Namespace, item: MemoryItem
) -> tuple[RetentionService, _SpyLtmDouble, FrozenClock]:
    double = _SpyLtmDouble({item.id: item})
    # far past any reasonable gc_history_window_d so the window gate never blocks the GC, and
    # gc_history_window_d=0 so it never blocks either.
    now = _T0 + timedelta(days=1000)
    clock = FrozenClock(now)
    settings = LifecycleSettings(retention=RetentionSettings(gc_history_window_d=0))
    service = RetentionService(
        ltm=double,  # type: ignore[arg-type]
        ltm_retention=double,
        settings=settings,
        clock=clock,
    )
    return service, double, clock


async def test_gc_raises_before_delete_when_mapping_is_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ns = Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )
    item = _dead_fact(ns)
    service, double, clock = _make_service_and_double(ns, item)

    # Simulate "the map lags a real MemoryState member" (the scenario _STATE_MAP's own docstring
    # warns about) WITHOUT touching the real MemoryState enum: drop just this item's state from a
    # monkeypatched copy of the ClassVar dict. monkeypatch.setattr restores the original dict on
    # teardown regardless of outcome.
    unmapped = {k: v for k, v in RetentionService._STATE_MAP.items() if k is not item.state}
    assert item.state not in unmapped  # sanity: we actually removed the member under test
    monkeypatch.setattr(RetentionService, "_STATE_MAP", unmapped)

    with pytest.raises(KeyError) as exc_info:
        await service.sweep(ns, clock=clock)

    # Constrain WHICH KeyError this is. A bare ``pytest.raises(KeyError)`` would also be satisfied
    # by an incidental dict miss anywhere else in the sweep, which would make the ordering pin
    # below rest on the wrong exception. ``_STATE_MAP[state]`` raises with the missing key itself
    # as ``args[0]``, so this asserts the failure is exactly the mapping lookup we removed — and
    # therefore that the item really did reach pass 2's GC branch.
    assert exc_info.value.args[0] is item.state

    # The pin: gc_delete must NEVER have run for the item whose mapping raised. If a future
    # refactor reorders retention.py's pass 2 back to gc_delete-then-map, this call would have
    # already happened by the time the (still-unmapped) lookup raises, and this assertion fails.
    assert double.deleted == []
    assert item.id in double.items  # the fact must still be there — nothing was deleted


async def test_gc_delete_spy_records_a_call_in_the_success_path() -> None:
    """Non-vacuity proof for the assertion above: the identical double/item/settings, with
    ``_STATE_MAP`` left untouched, really does drive ``gc_delete`` and the spy really does record
    it — so the sibling test's ``deleted == []`` demonstrates the ordering held, not that the spy
    is simply never reached."""
    ns = Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )
    item = _dead_fact(ns)
    service, double, clock = _make_service_and_double(ns, item)

    acted_on = await service.sweep(ns, clock=clock)

    assert acted_on == 1
    assert double.deleted == [item.id]  # the spy DOES record a call when nothing is monkeypatched
    assert double.items == {}
