"""``TieredMemoryRepository`` against the REAL three-store stack — ZERO mocks.

Real ``mu-dev-cache`` (Valkey, STM) + ``mu-dev-qdrant`` (MTM) + ``mu-dev-falkordb`` (LTM), reached
through the central ``Settings`` tree, never a port literal. DEV-STANDARDS: *"INTEGRATION TESTS
USE THE REAL SYSTEM — ZERO MOCKS. NON-NEGOTIABLE."* If a store is not up, the fixture RAISES —
the suite is BLOCKED, never faked.

This is where the tenancy, bounding and id-stability claims are proven against the code that
actually ships: the Qdrant scroll filter, the Cypher ``WHERE``, the Redis key derivation. The
sibling unit suite proves the router's composition logic; this one proves the three real
predicates underneath it.

Every collection, graph and key created here is namespace-scoped by a per-test ``uid`` and torn
down by name — never by a blanket prefix sweep, because the dev stores are shared with other
lanes.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_contracts.domain.model.memory import Namespace, State, Tier, Visibility
from mu_contracts.domain.model.pin import PinRequest
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.health.assessor import HeuristicV1Assessor
from mu_engine.services.health.service import MemoryHealthService
from mu_engine.services.health.settings import HealthSettings
from mu_engine.services.memory.repository import TieredMemoryRepository
from mu_engine.services.memory.router import TierLeg, TierRouter
from mu_engine.services.pin.service import PinService
from mu_engine.services.pin.settings import PinSettings
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryItem as EngineItem
from mu_engine.storage.domain.memory import MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.mappers.qdrant_mapper import collection_name
from mu_engine.storage.mappers.redis_mapper import RedisMapper

pytestmark = [pytest.mark.integration]

T0 = datetime(2026, 1, 1, tzinfo=UTC)
ACTIVE_ONLY: frozenset[State] = frozenset({State.ACTIVE})
ASSESSED: frozenset[State] = frozenset({State.ACTIVE, State.ARCHIVED, State.QUARANTINED})

#: Tiny deterministic vector dim — this suite proves the STORE predicates (filters, keys, Cypher),
#: not embedding quality, so pulling a real model in would add minutes and prove nothing extra.
_DIM = 8


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


@pytest.fixture
def other_ns(uid: str) -> Namespace:
    """A SECOND tenant inside the same org+workspace — deliberately the hardest case.

    A different org would be separated by the physical partition alone (a different Qdrant
    collection, a different FalkorDB graph), so it would pass even with every filter deleted. Two
    users in ONE org+workspace share the collection and the graph, so only the per-call namespace
    predicate keeps them apart. That is the layer this suite has to exercise.
    """
    return Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user="u2",
        session="s1",
        visibility=Visibility.PRIVATE,
    )


@pytest.fixture
def sibling_session_ns(ns: Namespace) -> Namespace:
    """The SAME user's OTHER session — the hardest MTM case, and the one a user-grain filter misses.

    ``other_ns`` differs by USER, which the Qdrant ``namespace_user_prefix`` key separates all by
    itself; a session-widened filter passes that test. Two SESSIONS of one user share the user
    prefix, so only a full ``to_prefix()`` match keeps them apart. η is six segments with the
    session INCLUDED (``domain/model/memory.py`` lines 137-146), and ``enumerate`` is *"the ONE
    bounded, PAGINATED partition walk"* over that partition.
    """
    return Namespace(
        org=ns.org,
        workspace=ns.workspace,
        user=ns.user,
        session="s2",
        visibility=ns.visibility,
    )


@pytest.fixture
def scope(ns: Namespace) -> ClientScope:
    return ClientScope(
        principal_id=ns.user,
        agent_principal_id=ns.user,
        org_id=ns.org,
        workspace_id=ns.workspace,
        session_id=ns.session,
    )


@pytest_asyncio.fixture
async def valkey_client(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.storage.valkey.url, decode_responses=True)
    await client.ping()  # fail-loud if mu-dev-cache is down
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def stm(
    valkey_client: Redis, ns: Namespace, other_ns: Namespace
) -> AsyncIterator[ValkeyStmAdapter]:
    adapter = ValkeyStmAdapter(valkey_client, mapper=RedisMapper(default_ttl_s=600))
    try:
        yield adapter
    finally:
        # Scoped teardown: only the keys THIS test's namespaces could have created.
        for partition in (ns, other_ns):
            pattern = f"mu/{partition.to_prefix()}:stm:*"
            keys = [key async for key in valkey_client.scan_iter(match=pattern)]
            if keys:
                await valkey_client.delete(*keys)


@pytest_asyncio.fixture
async def qdrant_client(settings: Settings) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=settings.storage.vector.url)
    await client.get_collections()  # fail-loud probe
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def mtm(qdrant_client: AsyncQdrantClient, ns: Namespace) -> AsyncIterator[QdrantMtmAdapter]:
    try:
        yield QdrantMtmAdapter(qdrant_client, dim=_DIM)
    finally:
        # `ns` and `other_ns` share org+workspace, so they share ONE collection — deleting it by
        # its exact derived name removes both tenants' rows and nothing else on the dev store.
        with contextlib.suppress(Exception):
            await qdrant_client.delete_collection(collection_name(ns, _DIM))


@pytest_asyncio.fixture
async def falkor_db(settings: Settings) -> AsyncIterator[FalkorDB]:
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    await db.select_graph("_probe").query("RETURN 1")  # fail-loud probe
    try:
        yield db
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()


@pytest_asyncio.fixture
async def ltm(
    falkor_db: FalkorDB, ns: Namespace, other_ns: Namespace
) -> AsyncIterator[FalkorLtmAdapter]:
    adapter = FalkorLtmAdapter(falkor_db)
    try:
        yield adapter
    finally:
        # Delete THIS test's graphs by exact derived name, never a `mu_g__` prefix sweep: the dev
        # FalkorDB is shared with other lanes and a blanket sweep would destroy their fixtures.
        for partition in (ns, other_ns):
            with contextlib.suppress(Exception):
                await falkor_db.select_graph(adapter.graph_name_for(partition)).delete()


@pytest.fixture
def repo(
    stm: ValkeyStmAdapter, mtm: QdrantMtmAdapter, ltm: FalkorLtmAdapter
) -> TieredMemoryRepository:
    return TieredMemoryRepository(
        router=TierRouter(
            (
                TierLeg(Tier.STM, stm, backend="valkey"),
                TierLeg(Tier.MTM, mtm, backend="qdrant"),
                TierLeg(Tier.LTM, ltm, backend="falkordb"),
            )
        )
    )


def make_item(
    ns: Namespace,
    *,
    memory_id: str,
    tier: MemoryTier,
    state: MemoryState = MemoryState.ACTIVE,
    pinned: bool = False,
) -> EngineItem:
    return EngineItem(
        id=memory_id,
        content=f"content for {memory_id}",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=tier,
        state=state,
        pinned=pinned,
        subject="user",
        predicate="prefers",
        object=memory_id,
        created_at=T0,
        updated_at=T0,
        valid_at=T0,
        embedding=[0.1] * _DIM,
    )


async def seed_all_tiers(
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    partition: Namespace,
    *,
    prefix: str,
    pinned: bool = False,
) -> list[str]:
    ids = [f"{prefix}_stm", f"{prefix}_mtm", f"{prefix}_ltm"]
    await stm.put(make_item(partition, memory_id=ids[0], tier=MemoryTier.STM, pinned=pinned))
    await mtm.upsert(make_item(partition, memory_id=ids[1], tier=MemoryTier.MTM, pinned=pinned))
    await ltm.upsert_fact(
        make_item(partition, memory_id=ids[2], tier=MemoryTier.LTM, pinned=pinned)
    )
    return ids


async def drain(repo: TieredMemoryRepository, partition: Namespace, *, limit: int) -> list[str]:
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(50):
        page, cursor = await repo.enumerate(
            partition, states=ASSESSED, tiers=None, pinned=None, cursor=cursor, limit=limit
        )
        assert len(page) <= limit
        seen.extend(item.id for item in page)
        if cursor is None:
            return seen
    raise AssertionError("enumerate did not terminate within 50 pages")


# ═══════════════════════════════════════════════════════════ TENANCY, against real predicates ══
async def test_enumerate_across_three_real_stores_never_crosses_the_partition(
    repo: TieredMemoryRepository,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    ns: Namespace,
    other_ns: Namespace,
) -> None:
    """Two users in the SAME org+workspace — one Qdrant collection, one FalkorDB graph.

    Only the per-call namespace predicate separates them, so dropping it from any ONE leg
    (the Qdrant ``_resolve_namespace_match`` term, the Cypher ``m.namespace = $ns``, the Redis
    key prefix) surfaces the other tenant's rows here.
    """
    mine = await seed_all_tiers(stm, mtm, ltm, ns, prefix="mine")
    theirs = await seed_all_tiers(stm, mtm, ltm, other_ns, prefix="theirs")

    seen = set(await drain(repo, ns, limit=10))
    assert seen == set(mine)
    assert not (seen & set(theirs)), "a second tenant's memories crossed the real fan-out"

    seen_other = set(await drain(repo, other_ns, limit=10))
    assert seen_other == set(theirs)


async def test_the_graph_tier_filter_separates_two_sessions_sharing_one_falkordb_graph(
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    ns: Namespace,
) -> None:
    """The LTM half of the two-layer isolation, isolated so the PHYSICAL layer cannot mask it.

    ``FalkorLtmAdapter.graph_name_for`` partitions on ``(org, workspace, visibility, user)`` and
    deliberately NOT on session, so two sessions of the same user live in ONE physical graph. That
    makes this the only configuration in which ``m.namespace = $ns`` in the enumerate Cypher is
    load-bearing — for two different USERS the graph name alone already separates them, and a test
    written across users would pass with the predicate deleted (verified: that mutation survives).

    ``storage-pluggable-spec.md`` §6 item 1 requires BOTH layers. This is the test that holds the
    second one honest.
    """
    sibling = Namespace(
        org=ns.org,
        workspace=ns.workspace,
        user=ns.user,
        session="s2",
        visibility=ns.visibility,
    )
    assert ltm.graph_name_for(sibling) == ltm.graph_name_for(
        ns
    ), "fixture precondition: the two sessions must share one physical graph"
    await ltm.upsert_fact(make_item(ns, memory_id="mine_s1", tier=MemoryTier.LTM))
    await ltm.upsert_fact(make_item(sibling, memory_id="other_s2", tier=MemoryTier.LTM))

    repo = TieredMemoryRepository(
        router=TierRouter(
            (
                TierLeg(Tier.STM, stm, backend="valkey"),
                TierLeg(Tier.MTM, mtm, backend="qdrant"),
                TierLeg(Tier.LTM, ltm, backend="falkordb"),
            )
        )
    )
    page, _ = await repo.enumerate(
        ns, states=ASSESSED, tiers=frozenset({Tier.LTM}), pinned=None, cursor=None, limit=20
    )
    ids = {item.id for item in page}
    assert "mine_s1" in ids
    assert "other_s2" not in ids, "the graph-tier namespace predicate did not separate the sessions"


async def test_get_and_set_pinned_refuse_a_foreign_id_on_the_real_stores(
    repo: TieredMemoryRepository,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    ns: Namespace,
    other_ns: Namespace,
) -> None:
    from mu_contracts.domain.errors import PinTargetNotFoundError

    theirs = await seed_all_tiers(stm, mtm, ltm, other_ns, prefix="theirs")
    for foreign_id in theirs:
        assert await repo.get(ns, foreign_id) is None
        with pytest.raises(PinTargetNotFoundError):
            await repo.set_pinned(ns, foreign_id, True, at=T0, by="u1", reason="keep")

    # And nothing was written to the other tenant's rows.
    assert (await mtm.get(other_ns, theirs[1])).pinned is False  # type: ignore[union-attr]
    assert (await ltm.get_fact(other_ns, theirs[2])).pinned is False  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════ BOUNDING, real store cursors ══
async def test_the_paged_walk_covers_the_partition_exactly_once_on_real_stores(
    repo: TieredMemoryRepository,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    ns: Namespace,
) -> None:
    """Real ZSET ranks, real Qdrant scroll offsets, real FalkorDB keyset ids, composed into one
    token and walked at ``limit=2`` — no repeats, no holes."""
    expected: list[str] = []
    for i in range(3):
        expected.extend(await seed_all_tiers(stm, mtm, ltm, ns, prefix=f"m{i}"))

    seen = await drain(repo, ns, limit=2)
    assert sorted(seen) == sorted(expected)
    assert len(seen) == len(set(seen)), "a memory was served twice across real-store pages"


async def test_a_single_real_page_never_exceeds_its_limit(
    repo: TieredMemoryRepository,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    ns: Namespace,
) -> None:
    for i in range(4):
        await seed_all_tiers(stm, mtm, ltm, ns, prefix=f"m{i}")
    page, cursor = await repo.enumerate(
        ns, states=ASSESSED, tiers=None, pinned=None, cursor=None, limit=3
    )
    assert len(page) == 3
    assert cursor is not None


async def test_the_pinned_filter_is_served_by_the_real_server_side_indexes(
    repo: TieredMemoryRepository,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    ns: Namespace,
) -> None:
    """The Qdrant BOOL payload index and the FalkorDB ``:Memory.pinned`` property that
    memory-health-pinning-spec §3.1 line 168 requires by name."""
    pinned_ids = await seed_all_tiers(stm, mtm, ltm, ns, prefix="keep", pinned=True)
    plain_ids = await seed_all_tiers(stm, mtm, ltm, ns, prefix="plain", pinned=False)

    only_pinned, _ = await repo.enumerate(
        ns, states=ASSESSED, tiers=None, pinned=True, cursor=None, limit=20
    )
    assert {i.id for i in only_pinned} == set(pinned_ids)

    only_plain, _ = await repo.enumerate(
        ns, states=ASSESSED, tiers=None, pinned=False, cursor=None, limit=20
    )
    assert {i.id for i in only_plain} == set(plain_ids)


# ═══════════════════════════════════════════ ID STABILITY across three REAL stores (§7.1) ══
async def test_one_pin_lands_under_the_same_id_in_valkey_qdrant_and_falkordb(
    repo: TieredMemoryRepository,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    ns: Namespace,
) -> None:
    """One id, resident in all three real stores at once (the promotion window), pinned once."""
    shared_id = "mem_stable"
    await stm.put(make_item(ns, memory_id=shared_id, tier=MemoryTier.STM))
    await mtm.upsert(make_item(ns, memory_id=shared_id, tier=MemoryTier.MTM))
    await ltm.upsert_fact(make_item(ns, memory_id=shared_id, tier=MemoryTier.LTM))

    version = await repo.set_pinned(ns, shared_id, True, at=T0, by="u1", reason="reference")

    from_stm = await stm.get(ns, shared_id)
    from_mtm = await mtm.get(ns, shared_id)
    from_ltm = await ltm.get_fact(ns, shared_id)
    for stored in (from_stm, from_mtm, from_ltm):
        assert stored is not None, "the pin did not land under this id in one of the stores"
        assert stored.pinned is True
        assert stored.pinned_by == "u1"
        assert stored.pin_reason == "reference"
        assert stored.version == version

    await repo.set_pinned(ns, shared_id, False, at=T0, by="u1", reason=None)
    for getter, arg in ((stm.get, shared_id), (mtm.get, shared_id), (ltm.get_fact, shared_id)):
        cleared = await getter(ns, arg)
        assert cleared is not None
        assert (cleared.pinned, cleared.pinned_at, cleared.pinned_by, cleared.pin_reason) == (
            False,
            None,
            None,
            None,
        )


async def test_a_pin_on_the_real_stores_preserves_the_vector_and_the_graph_edges(
    repo: TieredMemoryRepository,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    ns: Namespace,
) -> None:
    """A pin is a field-group PATCH: it must not cost the MTM embedding or the LTM triple."""
    await mtm.upsert(make_item(ns, memory_id="mem_v", tier=MemoryTier.MTM))
    await ltm.upsert_fact(make_item(ns, memory_id="mem_v", tier=MemoryTier.LTM))

    await repo.set_pinned(ns, "mem_v", True, at=T0, by="u1", reason="keep")

    point = await mtm.get(ns, "mem_v")
    assert point is not None
    assert point.embedding is not None and len(point.embedding) == _DIM
    fact = await ltm.get_fact(ns, "mem_v")
    assert fact is not None
    assert (fact.subject, fact.predicate) == ("user", "prefers")


# ═════════════════════════════════════════ THE FEATURE IS LIVE — real services, real stores ══
async def test_health_and_pin_services_answer_against_the_real_three_store_facade(
    repo: TieredMemoryRepository,
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    ns: Namespace,
    scope: ClientScope,
) -> None:
    """The end-to-end claim: the two services that could not be constructed at all now return
    real answers over Valkey + Qdrant + FalkorDB."""
    ids = await seed_all_tiers(stm, mtm, ltm, ns, prefix="live")

    class _NoEdges:
        async def edges_for(self, partition: Namespace, memory_ids: frozenset[str]) -> object:
            from mu_contracts.domain.model.conflict import ConflictEdges

            return ConflictEdges()

    class _Bus:
        def __init__(self) -> None:
            self.published: list[object] = []

        async def publish(self, event: object) -> None:
            self.published.append(event)

    pin_service = PinService(
        repo=repo,
        bus=_Bus(),  # type: ignore[arg-type]
        settings=PinSettings(max_pins_per_namespace=50),
        clock=FrozenClock(T0),
    )
    pinned = await pin_service.pin(scope, ns, PinRequest(memory_id=ids[1], reason="reference"))
    assert pinned.pinned is True
    assert pinned.version >= 1

    health_settings = HealthSettings()
    health = MemoryHealthService(
        repo=repo,
        assessor=HeuristicV1Assessor(health_settings),
        conflicts=_NoEdges(),  # type: ignore[arg-type]
        settings=health_settings,
        clock=FrozenClock(T0),
    )
    view = await health.assess(scope, ns)
    assert view.partial is False
    assert view.summary.total == len(ids)
    assert view.summary.pinned_count == 1
    assert view.namespace == ns


async def test_the_mtm_leg_is_scoped_to_the_session_not_merely_to_the_user(
    repo: TieredMemoryRepository,
    mtm: QdrantMtmAdapter,
    ns: Namespace,
    sibling_session_ns: Namespace,
) -> None:
    """One leg reading a WIDER key than η is the fan-out hazard HARD CONSTRAINT 1 names.

    ``QdrantMtmAdapter`` has two namespace grains: recall federates the user's sessions
    (ADR 0030's ``namespace_user_prefix``), and the partition walk must not. The STM leg
    (``RedisMapper.memory_key``) and the LTM leg (Cypher ``m.namespace = $ns``) are both
    session-exact, so a user-grain MTM filter would make ONE leg of the same call read outside the
    caller's η — ``MemoryHealthService`` would stamp another session's rows with this η, and
    ``PinService._assert_within_pin_bound`` would count another session's pins against this
    partition's bound.
    """
    await mtm.upsert(make_item(ns, memory_id="mine_s1", tier=MemoryTier.MTM))
    await mtm.upsert(make_item(sibling_session_ns, memory_id="theirs_s2", tier=MemoryTier.MTM))

    seen = await drain(repo, ns, limit=10)

    assert seen == ["mine_s1"], "the MTM leg read outside η — a sibling session's row surfaced"
    # And the by-id path must agree with the walk about what η means; two paths with two answers
    # is itself the defect, independent of which one is wider.
    assert await repo.get(ns, "theirs_s2") is None


async def test_by_artifact_counts_references_within_the_session_partition_only(
    repo: TieredMemoryRepository,
    mtm: QdrantMtmAdapter,
    ns: Namespace,
    sibling_session_ns: Namespace,
) -> None:
    """``by_artifact`` is the artifact GC reference-count authority (memory-layer §2 lines
    312-321): a reference held only by ANOTHER session must not make an artifact read as live
    here, or the artifact is never GC-eligible in either partition."""
    artifact = "art_shared"
    mine = make_item(ns, memory_id="ref_mine", tier=MemoryTier.MTM)
    theirs = make_item(sibling_session_ns, memory_id="ref_theirs", tier=MemoryTier.MTM)
    await mtm.upsert(mine.model_copy(update={"artifact_ref": artifact}))
    await mtm.upsert(theirs.model_copy(update={"artifact_ref": artifact}))

    found = await repo.by_artifact(ns, artifact)

    assert [item.id for item in found] == ["ref_mine"]
