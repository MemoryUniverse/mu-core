"""``DemotionService`` wired onto the REAL ``MtmTierRepository.remove`` port (CF-2,
MLM-STAGE2-CARRYOVER.md) — REAL ``mu-dev-cache`` (Valkey/STM) + REAL ``mu-dev-qdrant`` (MTM),
ZERO mocks.

This is the CF-2-owned sibling of ``tests/lifecycle/test_demotion_int.py`` (S1-02, not this
task's owned file): that suite still passes a local ``_RealQdrantRemoval`` shim as
``mtm_remove``; THIS suite proves the shim is no longer necessary — the very same
``QdrantMtmAdapter`` instance already used for ``upsert``/``semantic`` is passed directly as
``mtm_remove``, exercising ``DemotionService`` end-to-end through the real, shared
``MtmTierRepository.remove`` (a genuine ``AsyncQdrantClient.delete``), not a narrow local
Protocol.

Covers CF-2's acceptance item: "demotion end-to-end via the real port."
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from qdrant_client import AsyncQdrantClient

from mu_contracts.domain.events import DomainEvent, MemoryDemoted
from mu_contracts.domain.model.memory import State
from mu_contracts.domain.model.memory import Tier as ContractTier
from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings, SalienceSettings
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryTier
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id

pytestmark = pytest.mark.integration

_DIM = 8
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class _EventRecorder:
    """A REAL, fully-functional ``EventPublisher`` — an in-memory recorder, not a mock."""

    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


async def _mtm_point_exists(
    client: AsyncQdrantClient, ns: Namespace, memory_id: str, *, dim: int
) -> bool:
    name = collection_name(ns, dim)
    if not await client.collection_exists(name):
        return False
    points = await client.retrieve(collection_name=name, ids=[point_id(memory_id)])
    return bool(points)


async def test_demotion_removes_the_real_mtm_point_via_the_shared_port(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """``DemotionService`` constructed with the REAL ``QdrantMtmAdapter`` as ``mtm_remove`` (no
    local shim, no ``_qdrant``-reaching adapter) really moves MTM->STM: the STM write-ahead
    copy lands in Valkey, the real Qdrant point is genuinely deleted via
    ``MtmTierRepository.remove``, and ``MemoryDemoted(to_tier=STM, to_state=ACTIVE)`` publishes."""
    ns = make_ns()
    item = MemoryItem(
        content="stale fact demoted through the real MtmTierRepository.remove port",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        importance_score=0.1,
        access_count=0,
        created_at=_EPOCH,
        embedding=[0.1] * _DIM,
        embedding_model="test-fixture",
    )
    await mtm.upsert(item)
    assert await _mtm_point_exists(qdrant_client, ns, item.id, dim=_DIM)  # sanity: really landed

    stm = make_stm()
    recorder = _EventRecorder()
    settings = LifecycleSettings()  # demote_mtm=0.3 default
    strategy = SalienceStrategy(SalienceSettings())
    clock = FrozenClock(_EPOCH + timedelta(hours=200))  # far past -> S(m) well below the gate

    # THE POINT OF THIS TEST: mtm_remove=mtm — the SAME real QdrantMtmAdapter instance used for
    # upsert/semantic above, passed directly as the removal port. No local MtmRemovalPort shim.
    service = DemotionService(
        stm=stm,
        mtm_remove=mtm,
        salience=strategy,
        settings=settings,
        clock=clock,
        bus=recorder,
    )

    report = await service.demote(ns, [item])

    assert report.evaluated == 1
    assert report.demoted == 1
    assert report.rescued == 0

    # Real Valkey: the write-ahead copy really landed, tier flipped, id preserved.
    stm_copy = await stm.get(ns, item.id)
    assert stm_copy is not None
    assert stm_copy.tier is MemoryTier.STM
    assert stm_copy.id == item.id

    # Real Qdrant: the MTM point is genuinely gone (real delete via the shared port, not a
    # payload-only supersede — invalidate() would have left it retrievable).
    assert not await _mtm_point_exists(qdrant_client, ns, item.id, dim=_DIM)

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert isinstance(event, MemoryDemoted)
    assert event.namespace == ns
    assert event.id == item.id
    assert event.tier is ContractTier.MTM
    assert event.to_tier is ContractTier.STM
    assert event.to_state is State.ACTIVE  # never the archival-only default


async def test_rescued_item_leaves_the_real_mtm_point_untouched_via_the_real_port(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """A candidate whose recomputed ``S(m) >= demote_mtm`` is rescued: zero Valkey writes, its
    real Qdrant point (upserted through the SAME adapter now wired as the removal port) is left
    completely untouched, no event published."""
    ns = make_ns()
    strategy = SalienceStrategy(SalienceSettings())
    settings = LifecycleSettings()
    clock = FrozenClock(_EPOCH)  # age == 0 -> rec == 1.0 -> maximal score

    fresh = MemoryItem(
        content="fresh, high-salience fact",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        importance_score=1.0,
        access_count=10,
        created_at=_EPOCH,
        embedding=[0.3] * _DIM,
        embedding_model="test-fixture",
    )
    await mtm.upsert(fresh)
    assert await _mtm_point_exists(qdrant_client, ns, fresh.id, dim=_DIM)

    stm = make_stm()
    recorder = _EventRecorder()
    service = DemotionService(
        stm=stm, mtm_remove=mtm, salience=strategy, settings=settings, clock=clock, bus=recorder
    )

    report = await service.demote(ns, [fresh])

    assert report.demoted == 0
    assert report.rescued == 1
    assert await stm.get(ns, fresh.id) is None
    assert await _mtm_point_exists(qdrant_client, ns, fresh.id, dim=_DIM)  # untouched
    assert recorder.events == []
