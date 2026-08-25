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

import socket
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


async def test_federated_sweep_demotes_a_candidate_from_another_session_of_the_same_user(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """The MTM WRITE verbs are namespace-scoped in the adapter (C3) — so this sweep must hand
    ``remove`` the ITEM's namespace, not its own.

    ``LifecycleManager`` sources candidates from ``MtmTierRepository.scan_for_demotion(ns, ...)``,
    which for PRIVATE is deliberately session-FEDERATED (BQ3/ADR 0030: it matches the session-less
    user prefix, so one sweep sees every one of the user's sessions). The sweep's own ``ns``
    therefore names ONE session while a legitimate candidate may belong to ANOTHER — exactly the
    shape reproduced here.

    Before C3 that mismatch was invisible: ``remove`` carried a bare, unsalted point id and
    deleted the point regardless of which partition was named. Now that the delete carries the
    tenancy predicate, passing the sweep's ``ns`` would match ZERO points and — because a scoped
    miss is a silent no-op, not an exception — step 1's STM write-ahead copy would become a
    permanent cross-tier DUPLICATE with the rollback never firing. This test fails if
    ``_demote_one`` regresses to the sweep's ``ns``.
    """
    sweep_ns = make_ns(session="s_sweep")
    item_ns = make_ns(session="s_other")  # same org/workspace/user, DIFFERENT session
    assert sweep_ns.to_prefix() != item_ns.to_prefix()
    assert collection_name(sweep_ns, _DIM) == collection_name(item_ns, _DIM), (
        "precondition failed: the two sessions are in different Qdrant collections, so this "
        "test would not exercise the scoped-write path at all"
    )

    item = MemoryItem(
        content="stale fact belonging to another session of the same user",
        kind=MemoryKind.PROPOSITION,
        namespace=item_ns,
        owner_id=item_ns.user,
        workspace_id=item_ns.workspace,
        session_id=item_ns.session,
        tier=MemoryTier.MTM,
        importance_score=0.1,
        access_count=0,
        created_at=_EPOCH,
        embedding=[0.1] * _DIM,
        embedding_model="test-fixture",
    )
    await mtm.upsert(item)
    assert await _mtm_point_exists(qdrant_client, item_ns, item.id, dim=_DIM)

    stm = make_stm()
    service = DemotionService(
        stm=stm,
        mtm_remove=mtm,
        salience=SalienceStrategy(SalienceSettings()),
        settings=LifecycleSettings(),
        clock=FrozenClock(_EPOCH + timedelta(hours=200)),
        bus=_EventRecorder(),
    )

    report = await service.demote(sweep_ns, [item])

    assert report.demoted == 1
    # The write-ahead copy lands under the ITEM's namespace (STM is key-prefixed by to_prefix()).
    assert await stm.get(item_ns, item.id) is not None
    # And the MTM point is GENUINELY gone — not left behind as a silent cross-tier duplicate.
    assert not await _mtm_point_exists(qdrant_client, item_ns, item.id, dim=_DIM), (
        "the demotion left the MTM point in place while STM already holds the copy — a silent "
        "cross-tier duplicate: the scoped remove was aimed at the sweep's namespace, not the "
        "item's"
    )


async def test_rollback_after_a_failed_remove_evicts_the_copy_from_the_items_own_namespace(
    make_ns: Callable[..., Namespace],
    make_stm: Callable[..., ValkeyStmAdapter],
) -> None:
    """The compensating rollback must evict under the ITEM's namespace, for the same reason the
    commit-delete does.

    STM is key-prefixed by ``Namespace.to_prefix()`` and step 1 writes the copy under
    ``stm_copy.namespace`` — the ITEM's. Under the session-federated sweep (see the test above)
    the service's own ``ns`` names a different session, so evicting under it would silently miss
    the copy and leave the very cross-tier duplicate this rollback exists to prevent, while still
    reporting ``mtm_remove_failed_rolled_back``.

    The removal failure is REAL, not injected: a genuine ``QdrantMtmAdapter`` over a genuine
    ``AsyncQdrantClient`` pointed at a closed local port, so ``remove`` raises a real connection
    error out of the real client (Valkey/STM below is the real ``mu-dev-cache``). Nothing is
    mocked; this branch simply has no other way to be reached.
    """
    sweep_ns = make_ns(session="s_sweep")
    item_ns = make_ns(session="s_other")
    assert sweep_ns.to_prefix() != item_ns.to_prefix()

    with socket.socket() as probe:  # a port nothing is listening on, chosen by the OS
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
    dead_mtm = QdrantMtmAdapter(
        AsyncQdrantClient(url=f"http://127.0.0.1:{dead_port}", check_compatibility=False),
        dim=_DIM,
        store_io_timeout_s=1.0,
    )

    item = MemoryItem(
        content="stale fact whose MTM removal will genuinely fail",
        kind=MemoryKind.PROPOSITION,
        namespace=item_ns,
        owner_id=item_ns.user,
        workspace_id=item_ns.workspace,
        session_id=item_ns.session,
        tier=MemoryTier.MTM,
        importance_score=0.1,
        access_count=0,
        created_at=_EPOCH,
        embedding=[0.1] * _DIM,
        embedding_model="test-fixture",
    )
    stm = make_stm()
    service = DemotionService(
        stm=stm,
        mtm_remove=dead_mtm,
        salience=SalienceStrategy(SalienceSettings()),
        settings=LifecycleSettings(),
        clock=FrozenClock(_EPOCH + timedelta(hours=200)),
        bus=_EventRecorder(),
    )

    report = await service.demote(sweep_ns, [item])

    assert report.demoted == 0
    assert report.outcomes[0].reason == "mtm_remove_failed_rolled_back"
    assert await stm.get(item_ns, item.id) is None, (
        "the rollback reported success while the STM write-ahead copy is still there — it was "
        "evicted under the sweep's namespace instead of the item's"
    )
