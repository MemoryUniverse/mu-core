"""Full tier-lifecycle AUTOMATIC drive — REAL mu-dev-cache (STM) + mu-dev-qdrant (MTM) +
mu-dev-falkordb (LTM), ZERO mocks (DEV-STANDARDS non-negotiable).

This is the S2 tier-lifecycle-completion proof: one AUTOMATIC ``sweep_namespace_now(ns)`` /
``sweep_user`` pass — with NO caller-supplied ``mtm_candidates`` and NO manually-fed retention
window — drives all three transitions on their own against the live stores:

* **PROMOTE** — a salient STM item is promoted STM->MTM->LTM by ``PromotionService``.
* **DEMOTE (the new auto-drive)** — a genuinely-stale MTM point is ENUMERATED by
  ``MtmTierRepository.scan_for_demotion`` (the new bounded Qdrant ``scroll`` primitive) and demoted
  MTM->STM by ``DemotionService``; a FRESH/salient MTM point enumerated in the SAME scan is rescued
  (never demoted). Previously the demotion leg only ran when a window was hand-fed.
* **RETAIN (the new wiring)** — an EPHEMERAL LTM fact past ``invalid_at`` self-expires
  (ACTIVE->EXPIRED) via a REAL ``RetentionService`` over the REAL ``FalkorLtmAdapter``
  (``facts_by_state``/``chain_head_state``/``gc_delete``, promoted onto the production adapter);
  an ACTIVE+valid PERMANENT fact is untouched.

Every asserted outcome is read straight from the real container (Qdrant retrieve/scroll, Valkey
get, FalkorDB facts_by_state) — no in-memory double stands in for a real-container claim.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import AsyncQdrantClient

from mu_contracts.domain.events import (
    DomainEvent,
    MemoryCaptured,
    MemoryDemoted,
    MemoryPromoted,
)
from mu_contracts.domain.model.lifecycle import UserPrefix
from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.manager import MemoryLifecycleManager
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate, ModePolicyResolver
from mu_engine.lifecycle.promotion import PromotionService
from mu_engine.lifecycle.retention import RetentionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings, ManagerModeSettings, SalienceSettings
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryState,
    MemoryTier,
    RetentionClass,
)
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


class _AllowAllResolver:
    def __init__(self, mode: ManagerMode) -> None:
        self._mode = mode

    def resolve(self, ns: object) -> ManagerMode:
        del ns
        return self._mode


def _gate(mode: ManagerMode) -> ManagerModeGate:
    resolver: ModePolicyResolver = _AllowAllResolver(mode)
    return ManagerModeGate(ManagerModeSettings(), resolver)


def _record(sink: list[DomainEvent]) -> Callable[[DomainEvent], Awaitable[None]]:
    async def _handler(event: DomainEvent) -> None:
        sink.append(event)

    return _handler


def _mtm_item(
    ns: Namespace,
    content: str,
    *,
    importance: float,
    access_count: int,
    created_at: datetime,
    dim: int,
) -> MemoryItem:
    return MemoryItem(
        content=content,
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        source=MemorySource.USER,
        importance_score=importance,
        access_count=access_count,
        created_at=created_at,
        embedding=[0.1] * dim,
        embedding_model="test-fixture",
    )


def _ltm_fact(
    ns: Namespace,
    content: str,
    *,
    retention_class: RetentionClass,
    valid_at: datetime,
    invalid_at: datetime | None,
    subject: str,
    predicate: str,
) -> MemoryItem:
    return MemoryItem(
        content=content,
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.LTM,
        state=MemoryState.ACTIVE,
        source=MemorySource.USER,
        subject=subject,
        predicate=predicate,
        object=content,
        retention_class=retention_class,
        importance_score=0.5,
        valid_at=valid_at,
        invalid_at=invalid_at,
        created_at=valid_at,
        updated_at=valid_at,
    )


async def _mtm_point_exists(
    client: AsyncQdrantClient, ns: Namespace, memory_id: str, *, dim: int
) -> bool:
    name = collection_name(ns, dim)
    if not await client.collection_exists(name):
        return False
    points = await client.retrieve(collection_name=name, ids=[point_id(memory_id)])
    return bool(points)


# =================================================================================================
# scan_for_demotion — the bounded MTM enumeration primitive, direct against real Qdrant
# =================================================================================================
async def test_scan_for_demotion_enumerates_active_points_in_plane_partition(
    make_ns: Callable[..., Namespace],
    mtm: QdrantMtmAdapter,
) -> None:
    ns = make_ns()
    stale = _mtm_item(ns, "stale", importance=0.1, access_count=0, created_at=_T0, dim=mtm._dim)
    fresh = _mtm_item(ns, "fresh", importance=1.0, access_count=10, created_at=_T0, dim=mtm._dim)
    await mtm.upsert(stale)
    await mtm.upsert(fresh)

    found = await mtm.scan_for_demotion(ns, limit=100)
    ids = {m.id for m in found}
    assert stale.id in ids  # every ACTIVE point in the partition is a candidate
    assert fresh.id in ids  # (staleness is decided by DemotionService, NOT the scan)


# =================================================================================================
# DEMOTION auto-drive: sweep_namespace_now with NO explicit mtm_candidates still demotes the stale
# point and spares the fresh one — via the new scan_for_demotion enumeration.
# =================================================================================================
async def test_automatic_sweep_demotes_stale_mtm_and_spares_fresh(
    make_ns: Callable[..., Namespace],
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_stm: Callable[..., ValkeyStmAdapter],
) -> None:
    ns = make_ns(session="sess-demote")
    now = _T0 + timedelta(days=10)
    clock = FrozenClock(now)
    stm = make_stm()
    bus = InprocBus()
    await bus.start()
    seen: list[DomainEvent] = []
    bus.subscribe(MemoryDemoted, _record(seen))

    stale = _mtm_item(
        ns, "stale nobody recalled", importance=0.1, access_count=0, created_at=_T0, dim=mtm._dim
    )
    fresh = _mtm_item(
        ns, "hot, salient", importance=1.0, access_count=10, created_at=now, dim=mtm._dim
    )
    await mtm.upsert(stale)
    await mtm.upsert(fresh)

    manager = _build_manager(stm=stm, mtm=mtm, ltm=None, clock=clock, bus=bus)

    # AUTOMATIC: no mtm_candidates supplied — the manager enumerates via scan_for_demotion itself.
    await manager.sweep_namespace_now(ns)

    # REAL Valkey: the stale item demoted MTM->STM (present in STM, tier flipped, id stable).
    stm_copy = await stm.get(ns, stale.id)
    assert stm_copy is not None
    assert stm_copy.tier is MemoryTier.STM
    assert stm_copy.id == stale.id
    # REAL Qdrant: stale point genuinely gone; fresh point untouched.
    assert not await _mtm_point_exists(qdrant_client, ns, stale.id, dim=mtm._dim)
    assert await _mtm_point_exists(qdrant_client, ns, fresh.id, dim=mtm._dim)
    # the fresh/salient item was NOT demoted into STM.
    assert await stm.get(ns, fresh.id) is None
    # REAL bus: exactly one MemoryDemoted (the stale one).
    demoted = [e for e in seen if isinstance(e, MemoryDemoted)]
    assert {e.id for e in demoted} == {stale.id}


# =================================================================================================
# RETENTION auto-drive: sweep_namespace_now with retention WIRED (not None) self-expires the
# EPHEMERAL fact and spares the ACTIVE+valid PERMANENT one — REAL FalkorDB adapter, no test store.
# =================================================================================================
async def test_automatic_sweep_expires_ephemeral_ltm_and_spares_permanent(
    make_ns: Callable[..., Namespace],
    ltm: FalkorLtmAdapter,
    make_stm: Callable[..., ValkeyStmAdapter],
) -> None:
    ns = make_ns(session="sess-retain")
    now = _T0 + timedelta(days=10)
    clock = FrozenClock(now)
    bus = InprocBus()
    await bus.start()

    ephemeral = _ltm_fact(
        ns,
        "trip to Berlin ends",
        retention_class=RetentionClass.EPHEMERAL,
        valid_at=_T0,
        invalid_at=_T0 + timedelta(days=5),
        subject="Ada",
        predicate="trip_ends",
    )
    permanent = _ltm_fact(
        ns,
        "Ada's daughter is Mia",
        retention_class=RetentionClass.PERMANENT,
        valid_at=_T0,
        invalid_at=None,
        subject="Ada",
        predicate="daughter",
    )
    await ltm.upsert_fact(ephemeral)
    await ltm.upsert_fact(permanent)

    manager = _build_manager(stm=make_stm(), mtm=None, ltm=ltm, clock=clock, bus=bus)

    # AUTOMATIC: retention is wired (not None) — the ONLY thing that can flip this real node.
    await manager.sweep_namespace_now(ns)

    expired = {m.id for m in await ltm.facts_by_state(ns, frozenset({MemoryState.EXPIRED}))}
    active = {m.id for m in await ltm.facts_by_state(ns, frozenset({MemoryState.ACTIVE}))}
    assert ephemeral.id in expired
    assert permanent.id in active  # ACTIVE + valid PERMANENT untouched


# =================================================================================================
# FULL lifecycle: ONE automatic sweep_user pass promotes + demotes + retains across all 3 stores.
# =================================================================================================
async def test_one_automatic_sweep_promotes_demotes_and_retains(
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_stm: Callable[..., ValkeyStmAdapter],
) -> None:
    ns = make_ns(session="sess-full")
    now = _T0 + timedelta(days=10)
    clock = FrozenClock(now)
    stm = make_stm()
    bus = InprocBus()
    await bus.start()
    promoted_ev: list[DomainEvent] = []
    demoted_ev: list[DomainEvent] = []
    bus.subscribe(MemoryPromoted, _record(promoted_ev))
    bus.subscribe(MemoryDemoted, _record(demoted_ev))

    # (promote) a salient STM item — importance 1.0, created "now" -> S(m) >= promote_stm_mtm.
    promotable = make_item(
        ns,
        "Ada works at Acme",
        subject="Ada",
        predicate="works_at",
        obj="Acme",
        importance=1.0,
        created_at=now,
    )
    await stm.put(promotable)
    # (demote) a genuinely-stale MTM point (10 days old, importance 0.1, never recalled).
    stale = _mtm_item(
        ns, "stale mtm fact", importance=0.1, access_count=0, created_at=_T0, dim=mtm._dim
    )
    await mtm.upsert(stale)
    # (retain) an EPHEMERAL LTM fact already past its invalid_at, + a PERMANENT survivor.
    ephemeral = _ltm_fact(
        ns,
        "trip ends",
        retention_class=RetentionClass.EPHEMERAL,
        valid_at=_T0,
        invalid_at=_T0 + timedelta(days=5),
        subject="Bo",
        predicate="trip_ends",
    )
    permanent = _ltm_fact(
        ns,
        "Bo's dog is Rex",
        retention_class=RetentionClass.PERMANENT,
        valid_at=_T0,
        invalid_at=None,
        subject="Bo",
        predicate="dog",
    )
    await ltm.upsert_fact(ephemeral)
    await ltm.upsert_fact(permanent)

    manager = _build_manager(stm=stm, mtm=mtm, ltm=ltm, clock=clock, bus=bus)

    # Exactly the MaintenanceLoop-style automatic call: register the ns via a bus event, then sweep
    # the whole user — NO explicit candidates, NO manual verb, NO mode-gate manual path.
    await manager.on_bus_event(MemoryCaptured(namespace=ns, ids=[promotable.id]))
    await manager.sweep_user(UserPrefix(ns))

    # PROMOTE landed in real FalkorDB LTM + published MemoryPromoted.
    facts = await ltm.graph_recall(ns, subject="Ada", limit=10)
    assert any(f.item.object == "Acme" for f in facts)
    assert any(e.id == promotable.id for e in promoted_ev if isinstance(e, MemoryPromoted))

    # DEMOTE: stale MTM point gone, landed in real Valkey STM, MemoryDemoted published.
    assert not await _mtm_point_exists(qdrant_client, ns, stale.id, dim=mtm._dim)
    stm_copy = await stm.get(ns, stale.id)
    assert stm_copy is not None and stm_copy.tier is MemoryTier.STM
    assert any(e.id == stale.id for e in demoted_ev if isinstance(e, MemoryDemoted))

    # RETAIN: ephemeral self-expired, permanent untouched (real FalkorDB read).
    expired = {m.id for m in await ltm.facts_by_state(ns, frozenset({MemoryState.EXPIRED}))}
    active = {m.id for m in await ltm.facts_by_state(ns, frozenset({MemoryState.ACTIVE}))}
    assert ephemeral.id in expired
    assert permanent.id in active


# =================================================================================================
# builder + REAL empty doubles for the legs a given case does not exercise
# =================================================================================================
class _EmptyMtm:
    """REAL empty MTM stand-in for the legs a given case doesn't exercise (isolated-logic double,
    not a mock of a store under test)."""

    async def upsert(self, item: MemoryItem) -> None:  # pragma: no cover
        raise AssertionError("unexpected MTM write")

    async def semantic(self, *a: object, **k: object) -> list[object]:
        return []

    async def invalidate(self, *a: object, **k: object) -> None:  # pragma: no cover
        return None

    async def remove(self, ns: Namespace, memory_id: str) -> None:  # pragma: no cover
        return None

    async def scan_for_demotion(self, ns: Namespace, *, limit: int) -> list[MemoryItem]:
        return []


class _NoLtmDistill:
    """REAL no-op distill for the demotion-only case (no LTM wired; the promotion window is empty,
    so ``PromotionService`` never reaches ``distill.distill`` — this is never actually called)."""

    async def distill(self, *a: object, **k: object) -> object:  # pragma: no cover
        class _R:
            outcomes: tuple[object, ...] = ()

        return _R()


def _build_manager(
    *,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter | None,
    ltm: FalkorLtmAdapter | None,
    clock: FrozenClock,
    bus: InprocBus,
) -> MemoryLifecycleManager:
    """Build a manager over the REAL adapters exactly as ``LocalContainer.build_lifecycle_manager``
    does: ``mtm=`` powers the demotion auto-drive; ``retention=`` a REAL ``RetentionService`` over
    the real ``FalkorLtmAdapter`` when one is supplied (else an honest None)."""
    salience = SalienceStrategy(SalienceSettings())
    mtm_for_services: object = mtm if mtm is not None else _EmptyMtm()
    if ltm is not None:
        distill: object = DistillPipeline(
            ltm=ltm,
            mtm=mtm_for_services,
            clock=clock,
            bus=bus,  # type: ignore[arg-type]
        )
        retention: RetentionService | None = RetentionService(
            ltm=ltm, ltm_retention=ltm, clock=clock, bus=bus
        )
    else:
        distill = _NoLtmDistill()
        retention = None
    promotion = PromotionService(
        mtm=mtm_for_services,  # type: ignore[arg-type]
        distill=distill,  # type: ignore[arg-type]
        salience=salience,
        stm=stm,
        clock=clock,
        bus=bus,
    )
    demotion = DemotionService(
        stm=stm,
        mtm_remove=mtm_for_services,  # type: ignore[arg-type]
        salience=salience,
        clock=clock,
        bus=bus,
    )
    return MemoryLifecycleManager(
        salience=salience,
        promotion=promotion,
        demotion=demotion,
        distill=distill,  # type: ignore[arg-type]
        mtm=mtm,
        retention=retention,
        mode_gate=_gate(ManagerMode.HYBRID),
        bus=bus,
        settings=LifecycleSettings(),
        clock=clock,
    )
