"""``RetentionService`` (S2-01; ADR 0035) — REAL ``mu-dev-falkordb`` (mu-dev-graph), ZERO mocks
for every real-container claim (DEV-STANDARDS non-negotiable). Fixtures ``settings``/``uid``/
``make_ns``/``make_item``/``falkor_db``/``ltm`` come from this directory's own ``conftest.py``
(shared with the sibling S1-01/S1-02/S1-03 suites).

Covers S2-01's acceptance list (``.claude/team_analysis`` plan, task id S2-01):

- **Real-container claim** — an EPHEMERAL fact with ``invalid_at`` in the past self-expires
  (``ACTIVE`` -> ``EXPIRED``) and is excluded from ``facts_at(now)``; a PERMANENT fact of the
  same age survives (never touched) — ``test_ephemeral_self_expires_and_permanent_survives``.
- **AC-2.1** — under an ``OffsetClock`` simulating clock skew, the ``now >= invalid_at``
  ``SELF_EXPIRE`` transition fires at the same logical instant relative to ``invalid_at`` as
  under an equivalent-instant clock — skew changes *when* the check runs, never *which* facts
  expire — ``test_ac21_offset_clock_reproduces_the_same_expiry_decision``.
- **AC-2.2** — a PERMANENT fact is never archived/GC'd under an adversarial multi-year
  ``FrozenClock`` jump (REAL FalkorDB) — ``test_ac22_permanent_never_archived_under_multi_year_
  clock_jump`` (property-style: parametrized over several jump magnitudes). The **pinned** half
  of AC-2.2 is a pure unit test of isolated logic (``test_ac22_pinned_mechanism_never_archived_
  or_gcd``, NOT ``pytest.mark.integration``) — DEV-STANDARDS' one sanctioned exception ("mocks
  allowed ONLY in pure unit tests of isolated logic"): the shipped ``MemoryItem`` (S0-06) does
  not carry a ``pinned`` field yet (see ``retention.py``'s own "PIN GAP" docstring note and
  ``storage/domain/memory.py``'s), and a real ``MemoryItem.model_validate_json`` round-trip
  through FalkorDB's ``memory_json`` carrier would silently DROP an ad-hoc ``pinned`` field
  added only on a local test subclass (base ``MemoryItem`` has no ``extra="forbid"`` guard, so
  an unknown key is ignored on read-back) — so the *mechanism* (``RetentionService._is_pinned``
  honored inside ``sweep()``) is proven directly against an in-memory ``LtmRetentionStorePort``
  double instead of a real store round-trip that cannot yet carry the field.
- **Never touches an ACTIVE non-expired fact; acts only on SUPERSEDED/EXPIRED, GC gated on BOTH
  ``gc_history_window_d`` AND provenance-chain-head-dead** — ``test_dead_fact_gc_requires_both_
  window_elapsed_and_chain_head_dead`` (both branches: chain head still ``ACTIVE`` keeps the
  whole chain; chain head dead + window elapsed really ``DETACH DELETE``s the real node).
- **COLD-slides only importance-gated DURABLE items, with reactivate-on-recall** —
  ``test_durable_cold_slide_then_reactivate_on_recall``.
- **``manager.py``'s ``sweep_namespace_now`` calls ``self._retention.sweep(...)`` when
  retention is not None, with ZERO diff to manager.py** — ``test_retention_wired_into_manager_
  sweep_namespace_now``: constructs a REAL ``MemoryLifecycleManager`` with a REAL
  ``RetentionService`` and proves the real FalkorDB node it seeded actually flips ACTIVE->EXPIRED
  purely from calling the manager's own ``sweep_namespace_now`` — the ONLY way that mutation can
  happen is manager.py's existing (S1-03) ``if self._retention is not None: await self._retention.
  sweep(...)`` call site actually firing.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from falkordb.asyncio import FalkorDB

from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.manager import MemoryLifecycleManager
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate, ModePolicyResolver
from mu_engine.lifecycle.promotion import PromotionService
from mu_engine.lifecycle.retention import LtmRetentionStorePort, RetentionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import (
    LifecycleSettings,
    ManagerModeSettings,
    RetentionSettings,
    SalienceSettings,
)
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import FrozenClock, OffsetClock
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter, g_query
from mu_engine.storage.domain.memory import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryState,
    MemoryTier,
    RetentionClass,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


# =================================================================================================
# The REAL LtmRetentionStorePort is now the production ``FalkorLtmAdapter`` itself (S2-01 CF-2
# closed): ``facts_by_state``/``chain_head_state``/``gc_delete`` were promoted off the former
# test-only ``_RealLtmRetentionStore`` in this file onto the adapter, so ``FalkorLtmAdapter``
# structurally satisfies ``LtmRetentionStorePort`` — no test-only store in the production path.
# This fixture hands the SAME live adapter straight through (genuine openCypher, real container).
# =================================================================================================
@pytest.fixture
def ltm_retention(ltm: FalkorLtmAdapter) -> LtmRetentionStorePort:
    return ltm


def _fact(
    ns: Namespace,
    content: str,
    *,
    retention_class: RetentionClass = RetentionClass.DURABLE,
    state: MemoryState = MemoryState.ACTIVE,
    valid_at: datetime,
    invalid_at: datetime | None = None,
    importance: float = 0.5,
    updated_at: datetime | None = None,
    cold: bool = False,
    subject: str = "Ada",
    predicate: str = "fact",
) -> MemoryItem:
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
        subject=subject,
        predicate=predicate,
        object=content,
        retention_class=retention_class,
        cold=cold,
        importance_score=importance,
        valid_at=valid_at,
        invalid_at=invalid_at,
        created_at=valid_at,
        updated_at=updated_at or valid_at,
    )


async def _node_exists(
    falkor_db: FalkorDB, ltm: FalkorLtmAdapter, ns: Namespace, memory_id: str
) -> bool:
    res = await g_query(
        falkor_db.select_graph(ltm.graph_name_for(ns)),
        "MATCH (m:Memory {namespace: $ns, id: $id}) RETURN count(m)",
        {"ns": ns.to_prefix(), "id": memory_id},
    )
    return bool(res and int(res[0][0]) > 0)


# =================================================================================================
# real-container claim: EPHEMERAL self-expires + excluded from facts_at; PERMANENT survives
# =================================================================================================
async def test_ephemeral_self_expires_and_permanent_survives(
    ltm: FalkorLtmAdapter,
    ltm_retention: LtmRetentionStorePort,
    make_ns: Callable[..., Namespace],
) -> None:
    ns = make_ns()
    created = _T0
    now = _T0 + timedelta(days=10)
    clock = FrozenClock(now)

    ephemeral = _fact(
        ns,
        "trip to Berlin ends",
        retention_class=RetentionClass.EPHEMERAL,
        valid_at=created,
        invalid_at=created + timedelta(days=5),  # already past `now`
        subject="Ada",
        predicate="trip_ends",
    )
    permanent = _fact(
        ns,
        "Ada's daughter is Mia",
        retention_class=RetentionClass.PERMANENT,
        valid_at=created,
        invalid_at=None,
        subject="Ada",
        predicate="daughter",
    )
    await ltm.upsert_fact(ephemeral)
    await ltm.upsert_fact(permanent)

    service = RetentionService(ltm=ltm, ltm_retention=ltm_retention, clock=clock)
    acted_on = await service.sweep(ns, clock=clock)
    assert acted_on == 1  # only the EPHEMERAL fact self-expired this tick

    # excluded from facts_at(now) — the EXISTING bi-temporal window (falkor_ltm.py) already
    # enforces this; this proves the state flip did not break/duplicate that read.
    now_facts = {m.id for m in await ltm.facts_at(ns, now)}
    assert ephemeral.id not in now_facts
    assert permanent.id in now_facts  # PERMANENT survives, untouched

    # the EXPIRED bookkeeping flip really landed in the real graph.
    expired = await ltm_retention.facts_by_state(ns, frozenset({MemoryState.EXPIRED}))
    dead = {m.id: m for m in expired}
    assert ephemeral.id in dead
    assert dead[ephemeral.id].state is MemoryState.EXPIRED

    # PERMANENT is still ACTIVE, completely untouched.
    still_active = {
        m.id for m in await ltm_retention.facts_by_state(ns, frozenset({MemoryState.ACTIVE}))
    }
    assert permanent.id in still_active


# =================================================================================================
# AC-2.1 — EPHEMERAL exit reproducible under OffsetClock (skew changes WHEN, never WHICH)
# =================================================================================================
async def test_ac21_offset_clock_reproduces_the_same_expiry_decision(
    ltm: FalkorLtmAdapter,
    ltm_retention: LtmRetentionStorePort,
    make_ns: Callable[..., Namespace],
) -> None:
    ns_a = make_ns(user="skew_direct")
    ns_b = make_ns(user="skew_offset")
    created = _T0
    invalid_at = created + timedelta(hours=1)
    logical_now = invalid_at + timedelta(minutes=30)  # 30 min past close, either way

    fact_a = _fact(
        ns_a,
        "short-lived fact A",
        retention_class=RetentionClass.EPHEMERAL,
        valid_at=created,
        invalid_at=invalid_at,
        subject="A",
        predicate="expires",
    )
    fact_b = _fact(
        ns_b,
        "short-lived fact B",
        retention_class=RetentionClass.EPHEMERAL,
        valid_at=created,
        invalid_at=invalid_at,
        subject="B",
        predicate="expires",
    )
    await ltm.upsert_fact(fact_a)
    await ltm.upsert_fact(fact_b)

    direct_clock = FrozenClock(logical_now)
    skew = timedelta(hours=3)  # a realistic clock-skew magnitude
    skewed_clock = OffsetClock(offset=skew, base=FrozenClock(logical_now - skew))
    assert skewed_clock.now() == direct_clock.now()  # SAME logical instant, different path there

    service_a = RetentionService(ltm=ltm, ltm_retention=ltm_retention, clock=direct_clock)
    service_b = RetentionService(ltm=ltm, ltm_retention=ltm_retention, clock=skewed_clock)

    acted_a = await service_a.sweep(ns_a, clock=direct_clock)
    acted_b = await service_b.sweep(ns_b, clock=skewed_clock)

    assert acted_a == acted_b == 1  # identical decision: WHICH fact expires is skew-invariant
    dead_a = await ltm_retention.facts_by_state(ns_a, frozenset({MemoryState.EXPIRED}))
    dead_b = await ltm_retention.facts_by_state(ns_b, frozenset({MemoryState.EXPIRED}))
    assert {m.id for m in dead_a} == {fact_a.id}
    assert {m.id for m in dead_b} == {fact_b.id}


# =================================================================================================
# AC-2.2 — PERMANENT never archived/GC'd under an adversarial multi-year clock jump (REAL store)
# =================================================================================================
@pytest.mark.parametrize(
    "jump",
    [timedelta(days=1), timedelta(days=400), timedelta(days=365 * 5), timedelta(days=365 * 50)],
    ids=["1d", "400d", "5y", "50y-adversarial"],
)
async def test_ac22_permanent_never_archived_under_multi_year_clock_jump(
    ltm: FalkorLtmAdapter,
    ltm_retention: LtmRetentionStorePort,
    make_ns: Callable[..., Namespace],
    jump: timedelta,
) -> None:
    ns = make_ns(user=f"perm{jump.days}")
    permanent = _fact(
        ns,
        "Ada's daughter is Mia",
        retention_class=RetentionClass.PERMANENT,
        valid_at=_T0,
        invalid_at=None,
        importance=0.01,  # even at the lowest importance, PERMANENT is never COLD/GC-eligible
        subject="Ada",
        predicate="daughter",
    )
    await ltm.upsert_fact(permanent)

    clock = FrozenClock(_T0 + jump)
    # aggressive settings — if PERMANENT exemption were broken, every knob here would trip it.
    settings = LifecycleSettings(
        retention=RetentionSettings(
            durable_cold_after_d=0,
            durable_cold_importance_max=1.0,
            gc_history_window_d=0,
        )
    )
    service = RetentionService(ltm=ltm, ltm_retention=ltm_retention, settings=settings, clock=clock)
    acted_on = await service.sweep(ns, clock=clock)

    assert acted_on == 0
    active = await ltm_retention.facts_by_state(ns, frozenset({MemoryState.ACTIVE}))
    assert len(active) == 1
    assert active[0].id == permanent.id
    assert active[0].cold is False
    assert active[0].state is MemoryState.ACTIVE


# -------------------------------------------------------------------------------------------
# AC-2.2 (pinned half) — pure unit test of isolated logic (see module docstring: the shipped
# MemoryItem does not carry `pinned` yet, so this proves the MECHANISM directly, not via a real
# store round-trip that would silently drop the field).
# -------------------------------------------------------------------------------------------
class _PinnedMemoryItem(MemoryItem):
    """Test-only stand-in for the not-yet-landed canonical `pinned: bool` field (CANONICAL
    §7.10/§7.26) — proves `RetentionService._is_pinned`'s `getattr` duck-typing composes
    correctly the instant that field exists, without editing the shared model."""

    pinned: bool = True


class _InMemoryLtmDouble:
    """A REAL, fully-functional in-memory stand-in — not a mock of an external dependency, a
    pure-Python double for isolated-logic testing (DEV-STANDARDS' explicit exception). Records
    every `upsert_fact`/`gc_delete` call so the test can assert NONE ever touched the pinned
    item."""

    def __init__(self, items: dict[str, MemoryItem]) -> None:
        self.items = items
        self.upserted: list[str] = []
        self.deleted: list[str] = []

    async def upsert_fact(self, item: MemoryItem) -> None:
        self.upserted.append(item.id)
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


async def test_ac22_pinned_mechanism_never_archived_or_gcd() -> None:
    ns = Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )
    pinned_ephemeral = _PinnedMemoryItem(
        content="pinned but time-bound",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.LTM,
        state=MemoryState.ACTIVE,
        retention_class=RetentionClass.EPHEMERAL,
        valid_at=_T0,
        invalid_at=_T0 + timedelta(hours=1),  # long expired
        pinned=True,
    )
    pinned_dead = _PinnedMemoryItem(
        content="pinned but already superseded",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.LTM,
        state=MemoryState.SUPERSEDED,
        retention_class=RetentionClass.DURABLE,
        valid_at=_T0,
        invalid_at=_T0,
        pinned=True,
    )
    double = _InMemoryLtmDouble(
        {pinned_ephemeral.id: pinned_ephemeral, pinned_dead.id: pinned_dead}
    )
    clock = FrozenClock(_T0 + timedelta(days=365 * 50))  # adversarial multi-year jump
    settings = LifecycleSettings(retention=RetentionSettings(gc_history_window_d=0))
    service = RetentionService(ltm=double, ltm_retention=double, settings=settings, clock=clock)  # type: ignore[arg-type]

    acted_on = await service.sweep(ns, clock=clock)

    assert acted_on == 0
    assert double.upserted == []
    assert double.deleted == []
    assert double.items[pinned_ephemeral.id].state is MemoryState.ACTIVE
    assert double.items[pinned_dead.id].state is MemoryState.SUPERSEDED


# =================================================================================================
# GC gated on BOTH gc_history_window_d elapsed AND provenance-chain-head-dead
# =================================================================================================
async def test_dead_fact_gc_requires_both_window_elapsed_and_chain_head_dead(
    ltm: FalkorLtmAdapter,
    ltm_retention: LtmRetentionStorePort,
    falkor_db: FalkorDB,
    make_ns: Callable[..., Namespace],
) -> None:
    ns = make_ns()
    long_ago = _T0
    now = _T0 + timedelta(days=1000)  # far past any reasonable window

    # --- chain: loser -> winner, winner stays ACTIVE (head alive) => loser is NEVER GC'd ---
    loser_alive_head = _fact(
        ns, "Ada at Acme", valid_at=long_ago, subject="Ada", predicate="works_at"
    )
    winner_alive = _fact(
        ns, "Ada at Globex", valid_at=long_ago, subject="Ada", predicate="works_at2"
    )
    await ltm.upsert_fact(loser_alive_head)
    await ltm.upsert_fact(winner_alive)
    await ltm.invalidate(ns, loser_alive_head.id, winner_alive.id, at=long_ago, reason="moved")

    # --- chain: loser -> winner, winner is ITSELF dead (self-expired) => head dead, GC-eligible ---
    loser_dead_head = _fact(
        ns, "Ada at OldCo", valid_at=long_ago, subject="Ada", predicate="works_at3"
    )
    winner_then_dead = _fact(
        ns, "Ada at MidCo", valid_at=long_ago, subject="Ada", predicate="works_at4"
    )
    await ltm.upsert_fact(loser_dead_head)
    await ltm.upsert_fact(winner_then_dead)
    await ltm.invalidate(ns, loser_dead_head.id, winner_then_dead.id, at=long_ago, reason="moved")
    # simulate winner_then_dead itself later expiring (still the chain HEAD — no outgoing edge).
    winner_now_expired = winner_then_dead.model_copy(deep=True)
    winner_now_expired.state = MemoryState.EXPIRED
    await ltm.upsert_fact(winner_now_expired)

    settings = LifecycleSettings(retention=RetentionSettings(gc_history_window_d=30))
    clock = FrozenClock(now)
    service = RetentionService(ltm=ltm, ltm_retention=ltm_retention, settings=settings, clock=clock)
    acted_on = await service.sweep(ns, clock=clock)

    assert acted_on >= 2  # loser_dead_head + winner_now_expired both GC'd
    assert await _node_exists(falkor_db, ltm, ns, loser_alive_head.id)  # head alive: KEPT
    assert await _node_exists(falkor_db, ltm, ns, winner_alive.id)
    assert not await _node_exists(falkor_db, ltm, ns, loser_dead_head.id)  # head dead: GC'd
    assert not await _node_exists(falkor_db, ltm, ns, winner_now_expired.id)


async def test_dead_fact_not_gcd_before_window_elapses(
    ltm: FalkorLtmAdapter,
    ltm_retention: LtmRetentionStorePort,
    falkor_db: FalkorDB,
    make_ns: Callable[..., Namespace],
) -> None:
    ns = make_ns()
    died_at = _T0
    now = _T0 + timedelta(days=1)  # well inside a 30-day window

    stale = _fact(ns, "will supersede shortly", valid_at=died_at, subject="Ada", predicate="p1")
    fresh = _fact(ns, "the winner", valid_at=died_at, subject="Ada", predicate="p1b")
    await ltm.upsert_fact(stale)
    await ltm.upsert_fact(fresh)
    await ltm.invalidate(ns, stale.id, fresh.id, at=died_at, reason="moved")
    # kill the head too so ONLY the window (not chain-head-aliveness) gates this case.
    fresh_dead = fresh.model_copy(deep=True)
    fresh_dead.state = MemoryState.EXPIRED
    await ltm.upsert_fact(fresh_dead)

    settings = LifecycleSettings(retention=RetentionSettings(gc_history_window_d=30))
    clock = FrozenClock(now)
    service = RetentionService(ltm=ltm, ltm_retention=ltm_retention, settings=settings, clock=clock)
    acted_on = await service.sweep(ns, clock=clock)

    assert acted_on == 0
    assert await _node_exists(falkor_db, ltm, ns, stale.id)
    assert await _node_exists(falkor_db, ltm, ns, fresh_dead.id)


# =================================================================================================
# COLD-slide (importance-gated DURABLE) + reactivate-on-recall
# =================================================================================================
async def test_durable_cold_slide_then_reactivate_on_recall(
    ltm: FalkorLtmAdapter,
    ltm_retention: LtmRetentionStorePort,
    make_ns: Callable[..., Namespace],
) -> None:
    ns = make_ns()
    long_ago = _T0
    now = _T0 + timedelta(days=200)

    cold_candidate = _fact(
        ns,
        "trivial low-importance fact",
        retention_class=RetentionClass.DURABLE,
        valid_at=long_ago,
        updated_at=long_ago,  # inactive the whole 200 days
        importance=0.05,
        subject="Ada",
        predicate="likes",
    )
    not_cold_high_importance = _fact(
        ns,
        "high-importance durable fact",
        retention_class=RetentionClass.DURABLE,
        valid_at=long_ago,
        updated_at=long_ago,
        importance=0.9,
        subject="Ada",
        predicate="allergic_to",
    )
    await ltm.upsert_fact(cold_candidate)
    await ltm.upsert_fact(not_cold_high_importance)

    settings = LifecycleSettings(
        retention=RetentionSettings(durable_cold_after_d=180, durable_cold_importance_max=0.2)
    )
    clock = FrozenClock(now)
    service = RetentionService(ltm=ltm, ltm_retention=ltm_retention, settings=settings, clock=clock)
    acted_on = await service.sweep(ns, clock=clock)
    assert acted_on == 1

    active_items = await ltm_retention.facts_by_state(ns, frozenset({MemoryState.ACTIVE}))
    active = {m.id: m for m in active_items}
    assert active[cold_candidate.id].cold is True
    assert active[cold_candidate.id].state is MemoryState.ACTIVE  # archived-but-QUERYABLE
    assert active[not_cold_high_importance.id].cold is False

    # a recall hit on the COLD item reactivates it.
    reactivated = await service.reactivate_on_recall(active[cold_candidate.id])
    assert reactivated is not None
    assert reactivated.cold is False
    after_items = await ltm_retention.facts_by_state(ns, frozenset({MemoryState.ACTIVE}))
    after = {m.id: m for m in after_items}
    assert after[cold_candidate.id].cold is False

    # a second reactivate on an already-hot item is a documented no-op.
    noop = await service.reactivate_on_recall(after[cold_candidate.id])
    assert noop is None


# =================================================================================================
# wiring: manager.py's sweep_namespace_now really calls RetentionService.sweep — ZERO diff to
# manager.py (S1-03 already carries the `if self._retention is not None: await self._retention.
# sweep(...)` call site; this constructs a REAL RetentionService and proves it actually runs).
# =================================================================================================
async def test_retention_wired_into_manager_sweep_namespace_now(
    ltm: FalkorLtmAdapter,
    ltm_retention: LtmRetentionStorePort,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    make_stm: Callable[..., Any],
) -> None:
    ns = make_ns(session="sess-retention")
    now = _T0 + timedelta(days=10)
    clock = FrozenClock(now)
    bus = InprocBus()
    await bus.start()

    ephemeral = _fact(
        ns,
        "trip to Berlin ends",
        retention_class=RetentionClass.EPHEMERAL,
        valid_at=_T0,
        invalid_at=_T0 + timedelta(days=5),  # already past `now`
        subject="Ada",
        predicate="trip_ends",
    )
    await ltm.upsert_fact(ephemeral)

    stm = make_stm()
    mtm_stub = _EmptyMtm()
    distill = DistillPipeline(ltm=ltm, mtm=mtm_stub, clock=clock, bus=bus)  # type: ignore[arg-type]
    promotion = PromotionService(
        mtm=mtm_stub,  # type: ignore[arg-type]
        distill=distill,
        salience=SalienceStrategy(SalienceSettings()),
        stm=stm,
        clock=clock,
        bus=bus,
    )
    retention = RetentionService(ltm=ltm, ltm_retention=ltm_retention, clock=clock, bus=bus)
    manager = MemoryLifecycleManager(
        salience=SalienceStrategy(SalienceSettings()),
        promotion=promotion,
        demotion=DemotionService(
            stm=stm,
            mtm_remove=_NoopMtmRemoval(),
            salience=SalienceStrategy(SalienceSettings()),
            clock=clock,
            bus=bus,
        ),
        distill=distill,
        retention=retention,
        mode_gate=_gate(ManagerMode.HYBRID),
        bus=bus,
        settings=LifecycleSettings(),
        clock=clock,
    )

    # the ONLY thing that can flip this real FalkorDB node ACTIVE->EXPIRED is manager.py's own
    # `if self._retention is not None: await self._retention.sweep(ns, clock=self._clock)`.
    await manager.sweep_namespace_now(ns)

    dead = await ltm_retention.facts_by_state(ns, frozenset({MemoryState.EXPIRED}))
    assert {m.id for m in dead} == {ephemeral.id}


class _EmptyMtm:
    """A REAL, fully-functional empty MTM stand-in for the promotion/distill legs this wiring
    test does not exercise (no STM items are seeded, so nothing ever reaches it) — not a mock
    of a dependency actually under test."""

    async def upsert(self, item: MemoryItem) -> None:  # pragma: no cover — never called here
        raise AssertionError("unexpected MTM write in the retention-wiring test")

    async def semantic(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    async def invalidate(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
        return None


class _NoopMtmRemoval:
    async def remove(self, ns: Namespace, memory_id: str) -> None:
        return None


class _AllowAllResolver:
    def __init__(self, mode: ManagerMode) -> None:
        self._mode = mode

    def resolve(self, ns: object) -> ManagerMode:
        del ns
        return self._mode


def _gate(mode: ManagerMode) -> ManagerModeGate:
    resolver: ModePolicyResolver = _AllowAllResolver(mode)
    return ManagerModeGate(ManagerModeSettings(), resolver)
