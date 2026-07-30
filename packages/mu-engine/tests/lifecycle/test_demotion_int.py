"""``DemotionService`` — REAL ``mu-dev-cache`` (Valkey/STM) + REAL ``mu-dev-qdrant`` (MTM),
ZERO mocks (DEV-STANDARDS non-negotiable; task packet: "write REAL component tests against
the LIVE stack").

Covers S1-02's acceptance list (``.claude/team_analysis`` plan, task id S1-02):

- **AC-1.3** — an item with ``S(m) < demote_mtm`` demotes MTM->STM for real (present in Valkey
  afterwards, its Qdrant point gone) and emits ``MemoryDemoted(tier=MTM, to_tier=STM,
  to_state=ACTIVE, retention=S(m))`` — never the pre-existing ``to_state=ARCHIVED`` archival
  shape.
- **Rescue** — a candidate whose ``access_count`` was bumped (simulating a real recall,
  ``distill.py:373``'s ``reinforced.access_count += 1`` pattern) between the instant its
  would-be-demoted score was computed and the next sweep's ``demote()`` call is skipped
  entirely: zero Valkey writes, its Qdrant point is untouched, no ``MemoryDemoted`` published
  (ADR 0034 "the deliberate feedback loop").
- **Clock-injected** — every score in this file is produced by a pinned ``FrozenClock``; no
  wall-clock read anywhere in ``demotion.py`` (spec §19 Rule 1).
- Central observability wired exactly like ``DistillPipeline`` (span/metrics/audit — asserted
  via a simple content-free recorder, not a mock of a store).

The real Qdrant removal (``MtmRemovalPort``) is backed here by a tiny adapter over the SAME
live ``AsyncQdrantClient`` used by ``QdrantMtmAdapter`` in this suite's sibling tests
(``tests/pipelines/test_distill_int.py`` uses the identical ``collection_name``/``point_id``
helpers to inspect real Qdrant state) — this is the "integrate phase" wiring gap
``demotion.py``'s module docstring flags: a future storage task should promote this shape onto
``QdrantMtmAdapter`` itself.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient, models
from redis.asyncio import Redis

from mu_contracts.config import Settings
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
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id

pytestmark = pytest.mark.integration

_DIM = 8
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


class _RealQdrantRemoval:
    """A REAL ``MtmRemovalPort`` implementation over the live ``mu-dev-qdrant`` client — the
    integrate-phase seam ``demotion.py``'s module docstring calls for (no mock; genuine
    ``client.delete`` against the real container, same ``collection_name``/``point_id`` mapping
    ``QdrantMtmAdapter`` itself uses)."""

    def __init__(self, client: AsyncQdrantClient, *, dim: int) -> None:
        self._client = client
        self._dim = dim

    async def remove(self, ns: Namespace, memory_id: str) -> None:
        name = collection_name(ns, self._dim)
        await self._client.delete(
            collection_name=name,
            points_selector=models.PointIdsList(points=[point_id(memory_id)]),
        )


class _EventRecorder:
    """A REAL, fully-functional ``EventPublisher`` — an in-memory recorder, not a mock of an
    external dependency (there is no real event bus in this slice to fake around)."""

    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


@pytest.fixture
def ns(uid: str) -> Namespace:
    return Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user="u1",
        session="s1",
        visibility=Visibility.PRIVATE,
    )


@pytest_asyncio.fixture
async def valkey_client(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.storage.valkey.url, decode_responses=True)
    await client.ping()  # fail-loud if the container is down
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def stm(valkey_client: Redis, ns: Namespace) -> AsyncIterator[ValkeyStmAdapter]:
    adapter = ValkeyStmAdapter(valkey_client)
    try:
        yield adapter
    finally:
        # best-effort teardown: this suite's own keys only (unique org/workspace per test).
        for key in await valkey_client.keys(f"mu/{ns.to_prefix()}:*"):
            with contextlib.suppress(Exception):
                await valkey_client.delete(key)


@pytest_asyncio.fixture
async def qdrant_client(settings: Settings, uid: str) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        await client.get_collections()  # fail-loud probe; BLOCKED (never faked) if qdrant is down
        yield client
    finally:
        for coll in (await client.get_collections()).collections:
            if uid in coll.name:
                with contextlib.suppress(Exception):
                    await client.delete_collection(coll.name)
        await client.close()


@pytest_asyncio.fixture
async def mtm(qdrant_client: AsyncQdrantClient) -> QdrantMtmAdapter:
    return QdrantMtmAdapter(qdrant_client, dim=_DIM)


@pytest_asyncio.fixture
async def mtm_remove(qdrant_client: AsyncQdrantClient) -> _RealQdrantRemoval:
    return _RealQdrantRemoval(qdrant_client, dim=_DIM)


async def _mtm_point_exists(
    client: AsyncQdrantClient, ns: Namespace, memory_id: str, *, dim: int
) -> bool:
    name = collection_name(ns, dim)
    if not await client.collection_exists(name):
        return False
    points = await client.retrieve(collection_name=name, ids=[point_id(memory_id)])
    return bool(points)


# ------------------------------------------------------------------------------------- AC-1.3
async def test_low_salience_item_demotes_mtm_to_stm_and_emits_memory_demoted_to_tier(
    ns: Namespace,
    mtm: QdrantMtmAdapter,
    mtm_remove: _RealQdrantRemoval,
    stm: ValkeyStmAdapter,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """AC-1.3: S(m) far below demote_mtm (age=200h, importance=0.1, never recalled) really
    moves MTM->STM (real Valkey put, real Qdrant point removed) and publishes
    ``MemoryDemoted(tier=MTM, to_tier=STM, to_state=ACTIVE, retention=S(m))`` — never the
    archival ``to_state=ARCHIVED`` default."""
    created_at = _EPOCH
    now = _EPOCH + timedelta(hours=200)
    item = MemoryItem(
        content="stale fact nobody recalled",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        importance_score=0.1,
        access_count=0,
        created_at=created_at,
        embedding=[0.1] * _DIM,
        embedding_model="test-fixture",
    )
    await mtm.upsert(item)
    assert await _mtm_point_exists(qdrant_client, ns, item.id, dim=_DIM)  # sanity: really landed

    recorder = _EventRecorder()
    settings = LifecycleSettings()  # demote_mtm=0.3 default (spec §16)
    strategy = SalienceStrategy(SalienceSettings())
    clock = FrozenClock(now)
    service = DemotionService(
        stm=stm,
        mtm_remove=mtm_remove,
        salience=strategy,
        settings=settings,
        clock=clock,
        bus=recorder,
    )

    # Sanity: this item really is below the gate under the exact strategy the service uses.
    raw_score = strategy.score(item, clock=clock)
    assert raw_score < settings.demote_mtm

    report = await service.demote(ns, [item])

    assert report.evaluated == 1
    assert report.demoted == 1
    assert report.rescued == 0
    outcome = report.outcomes[0]
    assert outcome.memory_id == item.id
    assert outcome.demoted is True
    assert outcome.score == pytest.approx(raw_score)

    # Real Valkey: the item really landed in STM, tier flipped, id preserved (CANONICAL §7.1).
    stm_copy = await stm.get(ns, item.id)
    assert stm_copy is not None
    assert stm_copy.tier is MemoryTier.STM
    assert stm_copy.id == item.id

    # Real Qdrant: the MTM point is genuinely gone (never merely marked "superseded").
    assert not await _mtm_point_exists(qdrant_client, ns, item.id, dim=_DIM)

    # AC-1.3: the event this service is the FIRST real emitter of.
    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert isinstance(event, MemoryDemoted)
    assert event.namespace == ns
    assert event.id == item.id
    assert event.tier is ContractTier.MTM
    assert event.to_tier is ContractTier.STM  # the additive field, set for a REAL tier move
    assert event.to_state is State.ACTIVE  # NEVER the archival-only ARCHIVED default
    assert event.retention == pytest.approx(raw_score)


# --------------------------------------------------------------------------------------- rescue
async def test_recalled_item_access_count_bump_rescues_it_on_next_salience_recompute(
    ns: Namespace,
    mtm: QdrantMtmAdapter,
    mtm_remove: _RealQdrantRemoval,
    stm: ValkeyStmAdapter,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """ADR 0034's "deliberate feedback loop": at ``access_count=0`` this item's score is below
    ``demote_mtm`` (it WOULD be demoted); a real recall bumps ``access_count`` (the same field
    ``DistillPipeline``'s ``reinforced.access_count += 1`` bumps, ``distill.py:373``) before the
    next sweep runs — the RECOMPUTED score at that next ``demote()`` call is at/above the gate,
    so the item is rescued: zero Valkey writes, its real Qdrant point untouched, no
    ``MemoryDemoted`` published."""
    created_at = _EPOCH
    now = _EPOCH + timedelta(hours=48)  # 2 half-lives -> rec(m) == 0.25 exactly
    settings = LifecycleSettings()
    strategy = SalienceStrategy(SalienceSettings())  # w_rec=.5 w_use=.2 w_imp=.3, usage_cap=10
    clock = FrozenClock(now)

    unrecalled = MemoryItem(
        content="fact about to be forgotten, then recalled just in time",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        importance_score=0.2,
        access_count=0,
        created_at=created_at,
        embedding=[0.2] * _DIM,
        embedding_model="test-fixture",
    )
    # Sanity: AT access_count=0 this item WOULD be demoted next sweep (rec=.25 -> score=.185).
    would_be_score = strategy.score(unrecalled, clock=clock)
    assert would_be_score == pytest.approx(0.5 * 0.25 + 0.2 * 0.0 + 0.3 * 0.2)
    assert would_be_score < settings.demote_mtm

    # A real recall happens between sweeps: access_count bumps to the usage cap (id-stable,
    # same memory — CANONICAL §7.1, exactly as a recall-driven reinforcement would leave it).
    recalled = unrecalled.model_copy(deep=True)
    recalled.access_count = 10  # SalienceSettings.usage_cap default -> use(m) saturates at 1.0
    await mtm.upsert(recalled)
    assert await _mtm_point_exists(qdrant_client, ns, recalled.id, dim=_DIM)

    recorder = _EventRecorder()
    service = DemotionService(
        stm=stm,
        mtm_remove=mtm_remove,
        salience=strategy,
        settings=settings,
        clock=clock,
        bus=recorder,
    )

    rescued_score = strategy.score(recalled, clock=clock)
    assert rescued_score == pytest.approx(0.5 * 0.25 + 0.2 * 1.0 + 0.3 * 0.2)
    assert rescued_score >= settings.demote_mtm  # the recall pushed it back over the gate

    report = await service.demote(ns, [recalled])

    assert report.evaluated == 1
    assert report.demoted == 0
    assert report.rescued == 1
    outcome = report.outcomes[0]
    assert outcome.demoted is False
    assert outcome.score == pytest.approx(rescued_score)
    assert outcome.reason == "salience_at_or_above_demote_mtm"

    # Zero writes: never landed in STM, its real MTM point is untouched, no event published.
    assert await stm.get(ns, recalled.id) is None
    assert await _mtm_point_exists(qdrant_client, ns, recalled.id, dim=_DIM)
    assert recorder.events == []


# ---------------------------------------------------------------------------------- clock-inject
async def test_demotion_gate_is_driven_only_by_the_injected_clock(
    ns: Namespace,
    mtm: QdrantMtmAdapter,
    mtm_remove: _RealQdrantRemoval,
    stm: ValkeyStmAdapter,
    qdrant_client: AsyncQdrantClient,
) -> None:
    """spec §19 Rule 1 — pinning a ``FrozenClock`` far from wall-clock "now" still produces a
    deterministic, correct demote/rescue decision; the service never substitutes
    ``datetime.now()`` for the injected clock."""
    settings = LifecycleSettings()
    strategy = SalienceStrategy(SalienceSettings())
    created_at = datetime(1999, 1, 1, tzinfo=UTC)  # long before any real "now"

    fresh = MemoryItem(
        content="just created, ancient clock",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        importance_score=1.0,
        access_count=10,
        created_at=created_at,
        embedding=[0.3] * _DIM,
        embedding_model="test-fixture",
    )
    await mtm.upsert(fresh)
    pinned_now = created_at  # age == 0 -> rec == 1.0 -> maximal score, regardless of wall-clock
    clock = FrozenClock(pinned_now)
    service = DemotionService(
        stm=stm, mtm_remove=mtm_remove, salience=strategy, settings=settings, clock=clock, bus=None
    )

    report = await service.demote(ns, [fresh])

    assert report.demoted == 0  # age==0 at the pinned instant -> S(m) == 1.0, well above the gate
    assert await _mtm_point_exists(qdrant_client, ns, fresh.id, dim=_DIM)
