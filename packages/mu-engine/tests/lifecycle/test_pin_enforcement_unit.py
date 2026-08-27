"""Pin enforcement at the three EXIT sites (memory-health §6.2 / §6.3 / §6.4).

Pure unit tests over in-memory port doubles (the sanctioned DEV-STANDARDS unit exception, the
same shape ``test_retention_gc_ordering_unit.py`` already uses) — no container, no real store,
no ``pytest.mark.integration``.

Each site is asserted in BOTH directions: pinned -> refused, and the byte-identical unpinned
control -> acted on. The control is what makes the negative assertion non-vacuous (a refusal
that also happens when nothing is pinned proves nothing).

**The GC test deliberately runs with ``RetentionSettings.respect_pins=False``.** That knob is a
cheap pre-filter; CANONICAL §7.10 says a pinned item is *"never garbage-collected regardless"*,
so the authoritative refusal must be the central ``LifecyclePolicy`` guard and must survive the
knob being off. Testing with the knob ON would have passed on the pre-filter alone and left the
guard — the thing the whole subsystem rests on — unexercised.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.model.memory import Namespace as ContractNamespace
from mu_engine.lifecycle.conflict import AdjudicationKind, ConflictAdjudicator
from mu_engine.lifecycle.demotion import DemotionService
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


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _item(
    ns: Namespace,
    *,
    pinned: bool,
    tier: MemoryTier = MemoryTier.MTM,
    state: MemoryState = MemoryState.ACTIVE,
    created_at: datetime = _T0,
    memory_id: str = "mem_1",
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        content="the user prefers dark mode",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=tier,
        state=state,
        source=MemorySource.USER,
        subject="user",
        predicate="prefers",
        object="dark mode",
        retention_class=RetentionClass.DURABLE,
        importance_score=0.0,  # with access_count=0 and a very old created_at, S(m) ~ 0
        access_count=0,
        pinned=pinned,
        valid_at=created_at,
        created_at=created_at,
        updated_at=created_at,
    )


# ═══════════════════════════════════════════════════ §6.2 — demotion (MTM -> STM tier move) ══
class _StmSpy:
    def __init__(self) -> None:
        self.put_calls: list[str] = []

    async def put(self, item: MemoryItem) -> None:
        self.put_calls.append(item.id)


class _MtmRemoveSpy:
    def __init__(self) -> None:
        self.removed: list[str] = []

    async def remove(self, ns: Namespace, memory_id: str) -> bool:
        del ns
        self.removed.append(memory_id)
        return True


def _demotion(clock: FrozenClock) -> tuple[DemotionService, _StmSpy, _MtmRemoveSpy]:
    stm, mtm = _StmSpy(), _MtmRemoveSpy()
    service = DemotionService(
        stm=stm,  # type: ignore[arg-type]
        mtm_remove=mtm,  # type: ignore[arg-type]
        settings=LifecycleSettings(),
        clock=clock,
    )
    return service, stm, mtm


async def test_pinned_mtm_item_below_the_demote_gate_is_kept(ns: Namespace) -> None:
    clock = FrozenClock(_T0 + timedelta(days=365))  # S(m) has long since collapsed
    service, stm, mtm = _demotion(clock)

    report = await service.demote(ns, [_item(ns, pinned=True)])

    assert report.demoted == 0
    assert [o.reason for o in report.outcomes] == ["pinned_lifecycle_override"]
    assert stm.put_calls == []  # no write-ahead copy: the item never entered the demote path
    assert mtm.removed == []


async def test_the_same_item_unpinned_is_demoted(ns: Namespace) -> None:
    """Non-vacuity control: identical item, identical clock, pin removed."""
    clock = FrozenClock(_T0 + timedelta(days=365))
    service, stm, mtm = _demotion(clock)

    report = await service.demote(ns, [_item(ns, pinned=False)])

    assert report.demoted == 1
    assert stm.put_calls == ["mem_1"]
    assert mtm.removed == ["mem_1"]


# ══════════════════════════════════════════════════════════════════ §6.3 — garbage collection ══
class _LtmDouble:
    def __init__(self, items: dict[str, MemoryItem]) -> None:
        self.items = items
        self.deleted: list[str] = []

    async def upsert_fact(self, item: MemoryItem) -> None:
        self.items[item.id] = item

    async def facts_by_state(
        self, ns: Namespace, states: frozenset[MemoryState]
    ) -> list[MemoryItem]:
        del ns
        return [i for i in self.items.values() if i.state in states]

    async def chain_head_state(self, ns: Namespace, memory_id: str) -> MemoryState:
        del ns
        return self.items[memory_id].state  # every fact here is its own chain head

    async def gc_delete(self, ns: Namespace, memory_id: str) -> None:
        del ns
        self.deleted.append(memory_id)
        del self.items[memory_id]


def _gc_service(item: MemoryItem) -> tuple[RetentionService, _LtmDouble, FrozenClock]:
    double = _LtmDouble({item.id: item})
    clock = FrozenClock(_T0 + timedelta(days=1000))
    settings = LifecycleSettings(
        # respect_pins OFF on purpose — see this module's docstring. The window gate is opened
        # so the ONLY thing that can still refuse the delete is the central guard.
        retention=RetentionSettings(gc_history_window_d=0, respect_pins=False)
    )
    return (
        RetentionService(
            ltm=double,  # type: ignore[arg-type]
            ltm_retention=double,
            settings=settings,
            clock=clock,
        ),
        double,
        clock,
    )


def _dead(ns: Namespace, *, pinned: bool) -> MemoryItem:
    item = _item(ns, pinned=pinned, tier=MemoryTier.LTM, state=MemoryState.EXPIRED)
    item.invalid_at = _T0
    return item


async def test_pinned_dead_fact_is_never_garbage_collected(ns: Namespace) -> None:
    service, double, clock = _gc_service(_dead(ns, pinned=True))

    acted_on = await service.sweep(ns, clock=clock)

    assert double.deleted == []
    assert "mem_1" in double.items  # invalidate-don't-delete: still there
    assert acted_on == 0


async def test_the_same_dead_fact_unpinned_is_garbage_collected(ns: Namespace) -> None:
    """Non-vacuity control: proves the sweep really does reach ``gc_delete`` here."""
    service, double, clock = _gc_service(_dead(ns, pinned=False))

    acted_on = await service.sweep(ns, clock=clock)

    assert double.deleted == ["mem_1"]
    assert acted_on == 1


class _StaleCarrierLtmDouble(_LtmDouble):
    """Reproduces the SHIPPED FalkorDB behaviour the GC sweep actually sees.

    ``_expire_impl``/``_invalidate_impl`` ``SET m.state = ...`` on the NODE but never rewrite the
    ``m.memory_json`` blob, while ``facts_by_state`` filters on the node property and rebuilds
    each row FROM that blob. A dead row therefore arrives at the sweep still carrying
    ``state=ACTIVE``. This double keeps the two apart the same way.
    """

    def __init__(self, item: MemoryItem, *, node_state: MemoryState) -> None:
        super().__init__({item.id: item})
        self._node_state = node_state

    async def facts_by_state(
        self, ns: Namespace, states: frozenset[MemoryState]
    ) -> list[MemoryItem]:
        del ns
        return [i for i in self.items.values() if self._node_state in states]

    async def chain_head_state(self, ns: Namespace, memory_id: str) -> MemoryState:
        del ns, memory_id
        return self._node_state


async def test_gc_survives_a_dead_row_whose_carried_state_is_stale(ns: Namespace) -> None:
    """REGRESSION: the guard must not turn a GC-able row into a crashed sweep.

    The row is dead on the node and ACTIVE in its carrier. ``ACTIVE -> DELETED`` is not a legal
    edge, so an EDGE-checking guard raised ``IllegalTransitionError`` straight out of
    ``RetentionService.sweep`` and the whole GC pass aborted with nothing collected — a
    regression the pin work introduced, on a path a plain ``SurfaceFacade.delete`` reaches.
    """
    item = _item(ns, pinned=False, tier=MemoryTier.LTM, state=MemoryState.ACTIVE)
    item.invalid_at = _T0
    double = _StaleCarrierLtmDouble(item, node_state=MemoryState.EXPIRED)
    clock = FrozenClock(_T0 + timedelta(days=1000))
    service = RetentionService(
        ltm=double,  # type: ignore[arg-type]
        ltm_retention=double,
        settings=LifecycleSettings(
            retention=RetentionSettings(gc_history_window_d=0, respect_pins=False)
        ),
        clock=clock,
    )

    acted_on = await service.sweep(ns, clock=clock)

    assert double.deleted == ["mem_1"]
    assert acted_on == 1


async def test_a_pinned_stale_carrier_row_is_still_refused(ns: Namespace) -> None:
    """...and the pin still wins on that same stale-carrier path."""
    item = _item(ns, pinned=True, tier=MemoryTier.LTM, state=MemoryState.ACTIVE)
    item.invalid_at = _T0
    double = _StaleCarrierLtmDouble(item, node_state=MemoryState.EXPIRED)
    clock = FrozenClock(_T0 + timedelta(days=1000))
    service = RetentionService(
        ltm=double,  # type: ignore[arg-type]
        ltm_retention=double,
        settings=LifecycleSettings(
            retention=RetentionSettings(gc_history_window_d=0, respect_pins=False)
        ),
        clock=clock,
    )

    await service.sweep(ns, clock=clock)

    assert double.deleted == []


async def test_pinned_ephemeral_fact_is_never_self_expired(ns: Namespace) -> None:
    """The ACTIVE -> EXPIRED retention exit is guarded too (spec §6.1 omits EXPIRED; CANONICAL
    §7.10's "regardless" does not)."""
    item = _item(ns, pinned=True, tier=MemoryTier.LTM)
    item.retention_class = RetentionClass.EPHEMERAL
    item.invalid_at = _T0
    service, double, clock = _gc_service(item)

    await service.sweep(ns, clock=clock)

    assert double.items["mem_1"].state is MemoryState.ACTIVE


# ══════════════════════════════════════════════ §6.4 — conflict: never the automatic loser ══
def _adjudicator(clock: FrozenClock) -> ConflictAdjudicator:
    # router=None -> the deterministic heuristic floor; no LLM, no network.
    return ConflictAdjudicator(router=None, clock=clock)


async def _verdict(
    ns: Namespace, *, winner: MemoryItem, candidate: MemoryItem
) -> tuple[AdjudicationKind, bool, bool]:
    clock = FrozenClock(_T0 + timedelta(days=10))
    adjudicator = _adjudicator(clock)
    contract_ns = ContractNamespace(
        org=ns.org,
        workspace=ns.workspace,
        user=ns.user,
        session=ns.session,
        visibility=ns.visibility.value,
    )
    verdict = await adjudicator.adjudicate(
        ns=contract_ns,
        winner=winner,
        candidate=candidate,
        heuristic_contradicts=True,
        budget=adjudicator.new_budget(),
    )
    return verdict.kind, verdict.apply, verdict.pin_blocked


async def test_a_pinned_target_is_never_the_automatic_supersede_loser(ns: Namespace) -> None:
    """The challenger is NEWER, so the heuristic would SELF_EXPIRE the (pinned) winner."""
    pinned_target = _item(ns, pinned=True, memory_id="pinned_target", created_at=_T0)
    challenger = _item(ns, pinned=False, memory_id="challenger", created_at=_T0 + timedelta(days=1))

    kind, apply, pin_blocked = await _verdict(ns, winner=pinned_target, candidate=challenger)

    assert pin_blocked is True
    assert kind is AdjudicationKind.PENDING  # parked, not applied — the conflict is not lost
    assert apply is False  # nothing is written: the pinned item stays ACTIVE


async def test_the_same_pair_unpinned_does_resolve(ns: Namespace) -> None:
    """Non-vacuity control: without the pin the heuristic really does decide and apply."""
    target = _item(ns, pinned=False, memory_id="target", created_at=_T0)
    challenger = _item(ns, pinned=False, memory_id="challenger", created_at=_T0 + timedelta(days=1))

    kind, apply, pin_blocked = await _verdict(ns, winner=target, candidate=challenger)

    assert pin_blocked is False
    assert kind is AdjudicationKind.SELF_EXPIRE
    assert apply is True


async def test_a_pinned_challenger_does_not_block_a_supersede_it_would_win(
    ns: Namespace,
) -> None:
    """Pin protects the LOSER side only. Here the pinned item is the more recent one, so it is
    the winner of the pair and nothing is being taken from it — the resolution must proceed."""
    older = _item(ns, pinned=False, memory_id="older", created_at=_T0)
    pinned_newer = _item(
        ns, pinned=True, memory_id="pinned_newer", created_at=_T0 + timedelta(days=1)
    )

    kind, apply, pin_blocked = await _verdict(ns, winner=pinned_newer, candidate=older)

    assert pin_blocked is False
    assert kind is AdjudicationKind.SUPERSEDE
    assert apply is True


async def test_a_pinned_candidate_is_never_the_automatic_supersede_loser(ns: Namespace) -> None:
    """The OTHER half of the §6.4 rule, previously untested.

    Here the pinned item is the RESIDENT candidate and the winner was asserted later, so the
    loser is the CANDIDATE — the SUPERSEDE polarity. A rule that only protects the SELF_EXPIRE
    side leaves an automatic supersede of a pinned memory wide open, and no test saw it: mutating
    the SUPERSEDE half away left the whole suite green.
    """
    pinned_candidate = _item(ns, pinned=True, memory_id="pinned_candidate", created_at=_T0)
    newer_winner = _item(
        ns, pinned=False, memory_id="newer_winner", created_at=_T0 + timedelta(days=1)
    )

    kind, apply, pin_blocked = await _verdict(ns, winner=newer_winner, candidate=pinned_candidate)

    assert pin_blocked is True
    assert kind is AdjudicationKind.PENDING
    assert apply is False


async def test_the_same_candidate_unpinned_does_resolve(ns: Namespace) -> None:
    """Non-vacuity control for the SUPERSEDE half."""
    candidate = _item(ns, pinned=False, memory_id="candidate", created_at=_T0)
    newer_winner = _item(
        ns, pinned=False, memory_id="newer_winner", created_at=_T0 + timedelta(days=1)
    )

    kind, apply, pin_blocked = await _verdict(ns, winner=newer_winner, candidate=candidate)

    assert pin_blocked is False
    assert kind is AdjudicationKind.SUPERSEDE
    assert apply is True


# ── the polarity trap: the verdict's `kind` is NOT the writer's direction ────────────────────
class _PolarityStubRouter:
    """Returns a fixed verdict word, so the adjudicator's ``kind`` can be forced to disagree with
    the direction ``DistillPipeline._resolve`` actually writes (which it re-derives from
    ``asserted_later``). ``ConflictAdjudicator`` is the only consumer, so no network, no model."""

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict

    async def generate(self, *args: object, **kwargs: object) -> object:
        from mu_engine.providers._contracts import Completion

        return Completion(
            text=f'{{"verdict": "{self._verdict}", "confidence": 0.9, "reason": "r"}}',
            model_id="stub",
            model_group="stub",
        )


async def _llm_verdict(
    ns: Namespace, *, winner: MemoryItem, candidate: MemoryItem, says: str
) -> tuple[AdjudicationKind, bool, bool]:
    clock = FrozenClock(_T0 + timedelta(days=10))
    adjudicator = ConflictAdjudicator(router=_PolarityStubRouter(says), clock=clock)  # type: ignore[arg-type]
    contract_ns = ContractNamespace(
        org=ns.org,
        workspace=ns.workspace,
        user=ns.user,
        session=ns.session,
        visibility=ns.visibility.value,
    )
    verdict = await adjudicator.adjudicate(
        ns=contract_ns,
        winner=winner,
        candidate=candidate,
        heuristic_contradicts=True,
        budget=adjudicator.new_budget(),
    )
    return verdict.kind, verdict.apply, verdict.pin_blocked


async def test_pin_is_checked_on_the_item_the_writer_will_actually_supersede(
    ns: Namespace,
) -> None:
    """The LLM says ``supersede`` (its own polarity maps the loser to the CANDIDATE), but the
    writer re-derives the direction from ASSERTION recency and will make the WINNER the loser.
    The pin must be consulted on the writer's loser, not the verdict's.

    Reading the loser off ``kind`` here inspects the unpinned candidate, reports
    ``pin_blocked=False``, and a pinned memory is automatically superseded downstream.
    """
    pinned_winner = _item(ns, pinned=True, memory_id="pinned_winner", created_at=_T0)
    later_candidate = _item(
        ns, pinned=False, memory_id="later_candidate", created_at=_T0 + timedelta(days=1)
    )

    kind, apply, pin_blocked = await _llm_verdict(
        ns, winner=pinned_winner, candidate=later_candidate, says="supersede"
    )

    assert pin_blocked is True
    assert kind is AdjudicationKind.PENDING
    assert apply is False


async def test_the_mirror_polarity_trap_is_closed_too(ns: Namespace) -> None:
    """Symmetric case: the LLM says ``self_expire`` (loser = the winner, by its polarity) while
    the writer will supersede the CANDIDATE, which is the pinned one."""
    later_winner = _item(
        ns, pinned=False, memory_id="later_winner", created_at=_T0 + timedelta(days=1)
    )
    pinned_candidate = _item(ns, pinned=True, memory_id="pinned_candidate", created_at=_T0)

    kind, apply, pin_blocked = await _llm_verdict(
        ns, winner=later_winner, candidate=pinned_candidate, says="self_expire"
    )

    assert pin_blocked is True
    assert kind is AdjudicationKind.PENDING
    assert apply is False


async def test_an_llm_verdict_on_an_unpinned_pair_is_untouched(ns: Namespace) -> None:
    """Non-vacuity control for both polarity tests: the same stub router, no pins, still applies."""
    winner = _item(ns, pinned=False, memory_id="winner", created_at=_T0)
    candidate = _item(ns, pinned=False, memory_id="candidate", created_at=_T0 + timedelta(days=1))

    _, apply, pin_blocked = await _llm_verdict(
        ns, winner=winner, candidate=candidate, says="supersede"
    )

    assert pin_blocked is False
    assert apply is True
