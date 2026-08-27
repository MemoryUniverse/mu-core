"""AD-24 end to end: ``MemoryLifecycleManager.get_state`` returns REAL tier counts, proven against
the REAL mu-dev-* stores with an INDEPENDENT query per tier — ZERO mocks (DEV-STANDARDS).

"Independent" means what it says: every expectation below is cross-checked with a **raw
store client** built in this file (``redis.asyncio.Redis`` / ``AsyncQdrantClient`` /
``falkordb.asyncio``), not with the same repository objects the container uses and never
with a number typed in by hand.
The cache under test never touches a store at all, so the two paths share nothing but the truth.

The corpus is deliberately TIER-DISTINGUISHING — three DIFFERENT non-zero counts, produced by three
different fates in ONE namespace (STM-only ingests, ingests promoted to MTM, a distilled subset that
reaches LTM). A one-item corpus cannot tell a real per-tier counter from one that increments every
tier on every event, and ``>= 1`` cannot catch over-counting; both are assertable only with a corpus
shaped like this one.

Three things are proven that a functional pass alone would not show:

* **η.** A second real user is seeded IN THE SAME PROCESS, on the same bus, and the first user's
  counts must not move. That is the only assertion that actually proves per-tenant scoping — an
  unobserved third user proves nothing about a cache that simply never counted anything.
* **Cold start.** A user this process has never seen reports ``UNOBSERVED``, not a confident ``0``.
* **Drift, asserted as KNOWN-DIVERGENT rather than hidden.** A direct store write publishes no
event,
  so the cache and the store legitimately disagree. The test pins that disagreement so the honest
  limit in ``mu_engine.lifecycle.counts``'s docstring is executable, not a claim.
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
from qdrant_client.http import models as qmodels
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_contracts.domain.events import (
    MemoryCaptured,
    MemoryGarbageCollected,
    MemorySuperseded,
)
from mu_contracts.domain.model.lifecycle import UserPrefix
from mu_engine.lifecycle.counts import CountsBasis
from mu_engine.lifecycle.mode_gate import ManagerMode
from mu_engine.lifecycle.settings import LifecycleSettings, ManagerModeSettings
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.domain.memory import (
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryState,
    MemoryTier,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name
from mu_engine.storage.mappers.redis_mapper import RedisMapper
from mu_local.composition import LifecycleManagerUnavailableError, LocalContainer
from mu_local.config import StorageSettings

pytestmark = pytest.mark.integration

#: Qdrant stores the user prefix WITHOUT ``UserPrefix``'s trailing "/" (``qdrant_mtm.py``'s own
#: ``_user_prefix`` helper), so the independent filter strips it. Verified by scrolling a real
#: point, not assumed.
_QDRANT_USER_PREFIX_KEY = "namespace_user_prefix"

_STM_ONLY = (
    "Ada lives in Paris",
    "Bob drinks tea",
    "Cleo runs marathons",
    "Dan plays cello",
    "Eve builds bikes",
)
_PROMOTED = ("Frank works at Acme", "Gina studies physics", "Hugo owns a boat")
_DISTILLED = _PROMOTED[:2]


def _lifecycle() -> LifecycleSettings:
    return LifecycleSettings(
        manager_mode=ManagerModeSettings(default_mode=ManagerMode.MANUAL.value)
    )


def _ns(uid: str, user: str, session: str = "s1") -> Namespace:
    return Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user=user,
        session=session,
        visibility=Visibility.PRIVATE,
    )


#: Every namespace any test in this file can create — the teardown's ONLY reliable handle on the
#: derived store names. ``collection_name``/``graph_name_for`` HASH the namespace
#: (``mu_mtm__<hash>__private__<dim>``), so the ``uid in name`` substring convention the sibling
#: suites use silently matches nothing for qdrant and falkordb and leaves a collection behind on
#: every run. Verified against the live stores; reported, and fixed here for this file's own
#: fixtures rather than left to accumulate.
_TEST_USERS = ("u1", "u2", "u3")
_TEST_SESSIONS = ("s1", "sA", "sB")


def _all_namespaces(uid: str) -> tuple[Namespace, ...]:
    return tuple(_ns(uid, user, session) for user in _TEST_USERS for session in _TEST_SESSIONS)


async def _teardown(settings: Settings, uid: str) -> None:
    """Drop every qdrant collection / falkordb graph / redis key this run's unique ``uid``
    created — resolved through the SAME name derivations the adapters use, never by substring."""
    wanted_collections = {
        collection_name(ns, dim) for ns in _all_namespaces(uid) for dim in (8, 384, 768, 1024)
    }
    qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        for coll in (await qdrant.get_collections()).collections:
            if uid in coll.name or coll.name in wanted_collections:
                with contextlib.suppress(Exception):
                    await qdrant.delete_collection(coll.name)
    finally:
        await qdrant.close()

    wanted_graphs = {
        FalkorLtmAdapter.graph_name_for(None, ns)  # type: ignore[arg-type]
        for ns in _all_namespaces(uid)
    }
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        for g in await db.list_graphs():
            name = g.decode() if isinstance(g, bytes) else g
            if uid in name or name in wanted_graphs:
                with contextlib.suppress(Exception):
                    await db.select_graph(name).delete()
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()

    redis: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    try:
        keys = [k async for k in redis.scan_iter(match=f"*{uid}*".encode())]
        if keys:
            await redis.delete(*keys)
    finally:
        await redis.aclose()


# ------------------------------------------------------------------ INDEPENDENT store queries --
async def _stm_in_store(settings: Settings, ns: Namespace) -> int:
    """LIVE STM rows: every member of the recency ZSET whose payload key still EXISTS.

    **Not ``ZCARD``, and the difference is the whole point.** The ZSET retains members whose
    payloads have already TTL-expired until a later ``recent()`` sweep prunes them
    (``redis_stm.py:180-183``: *"a TTL-expired member still lingering in the ZSET — drop it,
    self-heal"*). A ``ZCARD`` oracle therefore over-counts in the SAME direction as an add-only
    event cache, which makes the equality below structurally incapable of detecting the dominant
    STM drift this module names as its worst case. Checking the payload keys is the independent
    measurement; it is what ``recent()`` — the real read path — would return.
    """
    redis: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    try:
        ids = await redis.zrange(RedisMapper.recency_key(ns), 0, -1)
        live = 0
        for raw in ids:
            memory_id = raw.decode() if isinstance(raw, bytes) else str(raw)
            if await redis.exists(RedisMapper.memory_key(ns, memory_id)):
                live += 1
        return live
    finally:
        await redis.aclose()


async def _mtm_in_store(settings: Settings, ns: Namespace, dim: int) -> int:
    """Raw Qdrant ``count(exact=True)`` filtered to THIS user's prefix — the same payload key the
    MTM adapter stamps on every point (``qdrant_mtm.py:68``), so the query is per-tenant, not
    per-collection (a collection spans every user of one org+workspace)."""
    qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        result = await qdrant.count(
            collection_name=collection_name(ns, dim),
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=_QDRANT_USER_PREFIX_KEY,
                        match=qmodels.MatchValue(value=str(UserPrefix(ns)).rstrip("/")),
                    )
                ]
            ),
            exact=True,
        )
        return int(result.count)
    finally:
        await qdrant.close()


async def _ltm_in_store(settings: Settings, ns: Namespace) -> int:
    """Raw Cypher against the physical FalkorDB graph this namespace resolves to. ACTIVE only —
    matching the cache's declared ACTIVE-only semantics (``counts.py``, "What the numbers MEAN")."""
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        graph = db.select_graph(FalkorLtmAdapter.graph_name_for(None, ns))  # type: ignore[arg-type]
        result = await graph.query(
            "MATCH (m:Memory) WHERE m.namespace = $ns AND m.state = 'active' RETURN count(m)",
            {"ns": ns.to_prefix()},
        )
        return int(result.result_set[0][0])
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()


@pytest_asyncio.fixture
async def container(settings: Settings, uid: str) -> AsyncIterator[LocalContainer]:
    built = LocalContainer(StorageSettings(), settings=settings, lifecycle=_lifecycle())
    try:
        yield built
    finally:
        await _teardown(settings, uid)
        await built.close()


async def _seed(
    container: LocalContainer, ns: Namespace, texts: tuple[str, ...], *, promote: bool
) -> None:
    for offset, text in enumerate(texts):
        await container.ingest.remember(
            IngestActivity(
                namespace=ns,
                host="mu-local-test",
                session_offset=f"{'p' if promote else 's'}-{offset}",
                kind="user_message",
                text=text,
                promote=promote,
            )
        )


async def test_get_state_counts_match_independent_store_queries_per_tier(
    settings: Settings, uid: str, container: LocalContainer
) -> None:
    """The AD-24 acceptance proof, and the exact scope of what it proves.

    Three different non-zero counts, each equal to a raw-client query against a different store —
    **inside the one envelope where an event delta and a store cardinality coincide**: a single
    process that was already observing the bus before the first byte was written, reading within
    one STM TTL. That is the FULL-LOCAL daemon's normal case and it is worth pinning.

    It is NOT a proof that ``stm_count`` is a cardinality, and this equality must never be quoted
    as one — ``test_a_restarted_process_never_claims_a_cardinality_it_cannot_have`` below pins the
    case where the two legitimately diverge, which is one daemon restart away.
    """
    try:
        manager = container.build_lifecycle_manager()
    except LifecycleManagerUnavailableError as exc:  # pragma: no cover - composition-root gap
        pytest.skip(f"BLOCKED — mu_engine.lifecycle.manager not landed: {exc}")

    ns = _ns(uid, "u1")

    cold = manager.get_state(ns)
    assert cold.counts_basis is CountsBasis.UNOBSERVED, "cold start claimed to have looked"

    await _seed(container, ns, _STM_ONLY, promote=False)
    await _seed(container, ns, _PROMOTED, promote=True)
    recent = [s.item for s in await container.stm.recent(ns, limit=50)]
    await container.distill.distill(ns, [i for i in recent if i.content in _DISTILLED])

    view = manager.get_state(ns)
    in_store = (
        await _stm_in_store(settings, ns),
        await _mtm_in_store(settings, ns, container.mtm._dim),
        await _ltm_in_store(settings, ns),
    )
    from_cache = (view.stm_count, view.mtm_count, view.ltm_count)
    print(f"\nCACHE  stm/mtm/ltm = {from_cache}  basis={view.counts_basis}")  # noqa: T201
    print(f"STORE  stm/mtm/ltm = {in_store}  (raw redis / qdrant / falkordb clients)")  # noqa: T201

    assert view.counts_basis is CountsBasis.EVENT_DELTA
    assert view.counts_observed_since is not None
    assert from_cache == in_store, "get_state disagrees with the real stores"
    assert all(count > 0 for count in from_cache), "the corpus must exercise all three tiers"
    assert len(set(from_cache)) == 3, (
        "the three counts must DIFFER — an equal triple cannot distinguish a per-tier counter from "
        "one that increments every tier on every event"
    )


async def test_a_second_real_user_on_the_same_bus_never_moves_the_first_users_counts(
    settings: Settings, uid: str, container: LocalContainer
) -> None:
    """η, proven with two users that BOTH have real data in the same process on the same bus — not
    with an untouched prefix, which would pass even for a cache that counts nothing."""
    manager = container.build_lifecycle_manager()
    first, second, never = _ns(uid, "u1"), _ns(uid, "u2"), _ns(uid, "u3")

    await _seed(container, first, _STM_ONLY, promote=False)
    before = manager.get_state(first)

    await _seed(container, second, _PROMOTED, promote=True)

    after = manager.get_state(first)
    other = manager.get_state(second)
    assert (after.stm_count, after.mtm_count, after.ltm_count) == (
        before.stm_count,
        before.mtm_count,
        before.ltm_count,
    ), "another user's writes leaked into this user's counts"
    assert after.stm_count == len(_STM_ONLY) == await _stm_in_store(settings, first)
    assert other.stm_count == len(_PROMOTED) == await _stm_in_store(settings, second)
    assert other.mtm_count == await _mtm_in_store(settings, second, container.mtm._dim)
    assert other.mtm_count != after.mtm_count

    cold = manager.get_state(never)
    assert cold.counts_basis is CountsBasis.UNOBSERVED
    assert (cold.stm_count, cold.mtm_count, cold.ltm_count) == (0, 0, 0)


async def test_a_real_ingest_survives_a_broken_count_cache(
    settings: Settings, uid: str, container: LocalContainer
) -> None:
    """``InprocBus.publish`` re-raises a handler's exception into the publisher, so a broken count
    cache on a real ingest path would break the user's ``remember()`` itself. Proven against the
    real stores: the capture completes AND the memory actually lands in Redis."""
    ns = _ns(uid, "u1")

    def _boom(_ev: MemoryCaptured) -> int:
        raise RuntimeError("count fold blew up")

    container.tier_counts.note_captured = _boom  # type: ignore[method-assign]

    await _seed(container, ns, ("Ada lives in Paris",), promote=False)

    assert container.tier_counts.handler_errors >= 1, "the failure must be counted, never silent"
    assert await _stm_in_store(settings, ns) == 1, "the real capture did not land"


async def test_event_delta_counts_drift_from_a_writer_that_publishes_nothing(
    settings: Settings, uid: str, container: LocalContainer
) -> None:
    """KNOWN-DIVERGENT, asserted rather than hidden. A direct tier write publishes no event (the
    same shape as ``TieredMemoryRepository.add``, ``MemoryFacade.update``/``delete`` and STM Redis
    TTL expiry — see ``counts.py``'s drift section), so the cache legitimately falls behind the
    store. If this ever starts matching, either a reconciliation path landed or that writer began
    publishing — both are changes this test should force someone to look at."""
    manager = container.build_lifecycle_manager()
    ns = _ns(uid, "u1")
    await _seed(container, ns, ("Ada lives in Paris",), promote=False)
    assert manager.get_state(ns).stm_count == 1 == await _stm_in_store(settings, ns)

    await container.stm.put(
        MemoryItem(
            id=f"direct-{uuid.uuid4().hex[:8]}",
            content="written straight to the adapter, no bus involved",
            kind=MemoryKind.PROPOSITION,
            tier=MemoryTier.STM,
            namespace=ns,
            owner_id="u1",
            workspace_id=f"ws{uid}",
            session_id="s1",
            source=MemorySource.USER,
        )
    )

    view = manager.get_state(ns)
    in_store = await _stm_in_store(settings, ns)
    assert in_store == 2, "the direct write did not land"
    assert view.stm_count == 1, (
        "an event-derived count that tracked a bus-less write would mean the drift envelope "
        "documented in counts.py is wrong"
    )
    assert view.counts_basis is CountsBasis.EVENT_DELTA, (
        "the basis must still say EVENT_DELTA — it is the caller's only signal that this number "
        "is a delta over observed events rather than a cardinality counted from the store"
    )


async def test_the_cache_is_one_per_container_and_survives_repeat_factory_calls(
    uid: str, container: LocalContainer
) -> None:
    """``build_lifecycle_manager()`` is a factory. Every manager it returns must read the SAME
    cache, and calling it N times must not subscribe N handlers onto a bus whose handler list is
    never pruned."""
    ns = _ns(uid, "u1")
    managers = [container.build_lifecycle_manager() for _ in range(3)]
    await _seed(container, ns, ("Ada lives in Paris",), promote=False)

    counts = {m.get_state(ns).stm_count for m in managers}
    assert counts == {1}, f"repeat factory calls double-counted or diverged: {counts}"
    assert len(container.bus._handlers[MemoryCaptured]) == 1
    assert container.tier_counts.tracked_prefixes == 1
    assert container.tier_counts.counts(UserPrefix(ns)).basis is CountsBasis.EVENT_DELTA


# =============================================== the FIRST-REVIEW blockers, against real stores ==
async def test_a_restarted_process_never_claims_a_cardinality_it_cannot_have(
    settings: Settings, uid: str
) -> None:
    """**The AD-24 blocker, pinned where it was found: a second container over the SAME stores.**

    Every other integration test here builds the container and the corpus in one process, so the
    cache happens to have seen everything and an event delta is indistinguishable from a store
    cardinality. A daemon restart is not exotic — it is the most ordinary event in the deployment —
    and it breaks that coincidence permanently for anything written before the restart.

    The first cut shipped ``EVENT_DERIVED`` ("we looked") and was measured returning
    ``stm=1, basis=event_derived`` for a user with a full store, one capture after restart. This
    test is that measurement, kept: the counts may be a DELTA, the badge must say so, and nothing
    on the wire may read as a cardinality.
    """
    first = LocalContainer(StorageSettings(), settings=settings, lifecycle=_lifecycle())
    ns = _ns(uid, "u1")
    try:
        await _seed(first, ns, _STM_ONLY, promote=False)
        await _seed(first, ns, _PROMOTED, promote=True)
        seeded = first.build_lifecycle_manager().get_state(ns)
        assert seeded.stm_count == len(_STM_ONLY) + len(_PROMOTED)
    finally:
        await first.close()

    in_store = await _stm_in_store(settings, ns)
    assert in_store == len(_STM_ONLY) + len(_PROMOTED), "the corpus did not survive the restart"

    second = LocalContainer(StorageSettings(), settings=settings, lifecycle=_lifecycle())
    try:
        manager = second.build_lifecycle_manager()

        cold = manager.get_state(ns)
        assert (
            cold.counts_basis is CountsBasis.UNOBSERVED
        ), "a fresh process over a populated store must say 'we did not look'"
        assert (cold.stm_count, cold.mtm_count, cold.ltm_count) == (0, 0, 0)

        await _seed(second, ns, ("Ivy sails dinghies",), promote=False)
        after = manager.get_state(ns)
        now_in_store = await _stm_in_store(settings, ns)

        print(f"\nRESTART  cache stm={after.stm_count} basis={after.counts_basis}")  # noqa: T201
        print(f"RESTART  store stm={now_in_store}")  # noqa: T201

        assert after.stm_count == 1, "the delta is what was observed since attach, nothing more"
        assert now_in_store == in_store + 1
        assert after.stm_count < now_in_store, (
            "the premise of this test — if these ever agree, something reconciled and the basis "
            "wording below needs revisiting"
        )
        # The ONE thing that must hold: nothing on the wire claims this 1 is a cardinality.
        assert after.counts_basis is CountsBasis.EVENT_DELTA
        assert after.counts_basis.value == "event_delta"
        assert not after.counts_basis.value.startswith("event_derived")
        # …and `counts_observed_since` is the PROCESS's attach instant, not "when this user was
        # first seen" — it cannot rescue the reading above, which is why the basis carries it.
        assert after.counts_observed_since is not None
        assert after.counts_observed_since > seeded.counts_observed_since  # type: ignore[operator]
    finally:
        await _teardown(settings, uid)
        await second.close()


async def test_an_unattended_removal_sweep_never_flips_a_restarted_process_off_unobserved(
    settings: Settings, uid: str
) -> None:
    """``MemoryGarbageCollected``/``MemorySuperseded``/``MemoryQuarantined`` fire from retention
    and distill with NO user action. The first cut created a bucket for each, so an unattended
    sweep over a restarted daemon turned the honest ``UNOBSERVED`` into a confident ``(0,0,0)`` —
    the claim *"this user has nothing"* — for a user whose stores are full. Published here on the
    container's REAL bus, not a hand-built cache."""
    container = LocalContainer(StorageSettings(), settings=settings, lifecycle=_lifecycle())
    ns = _ns(uid, "u1")
    try:
        manager = container.build_lifecycle_manager()
        assert manager.get_state(ns).counts_basis is CountsBasis.UNOBSERVED

        await container.bus.publish(
            MemoryGarbageCollected(namespace=ns, id="whatever", prior_state=MemoryState.SUPERSEDED)
        )
        await container.bus.publish(
            MemorySuperseded(namespace=ns, loser_id="a", winner_id="b", valid_at=datetime.now(UTC))
        )

        view = manager.get_state(ns)
        assert view.counts_basis is CountsBasis.UNOBSERVED, (
            "an unattended sweep manufactured an 'observed' prefix — AD-24's own lie, reachable "
            "with no user action at all"
        )
        assert container.tier_counts.tracked_prefixes == 0
    finally:
        await _teardown(settings, uid)
        await container.close()


async def test_counts_span_every_session_of_one_user_and_say_so(
    settings: Settings, uid: str, container: LocalContainer
) -> None:
    """``get_state`` takes a SESSION-scoped ``Namespace`` but returns ``UserPrefix``-grained
    counts, because ``UserPrefix`` is the spec's §4b lease grain and the grain
    ``LifecycleStateView.user_prefix`` already reports beside them. That is deliberate — a
    demotion sweep publishes the SWEEP's namespace while deleting an item from ANOTHER session of
    the same user (``demotion.py:283-310``) — but every other test here uses ONE session per user,
    so none of them can see it. Pinned so the semantics are executable, not merely documented."""
    manager = container.build_lifecycle_manager()
    s1, s2 = _ns(uid, "u1", session="sA"), _ns(uid, "u1", session="sB")
    assert UserPrefix(s1) == UserPrefix(s2)

    await _seed(container, s1, _STM_ONLY, promote=False)
    await _seed(container, s2, _PROMOTED, promote=False)

    view1, view2 = manager.get_state(s1), manager.get_state(s2)
    per_session = (await _stm_in_store(settings, s1), await _stm_in_store(settings, s2))

    assert per_session == (len(_STM_ONLY), len(_PROMOTED)), "the two sessions did not both land"
    assert (
        view1.stm_count == view2.stm_count == sum(per_session)
    ), "the counts must span the user, matching the user_prefix reported beside them"
    assert view1.stm_count != per_session[0], (
        "if these ever agree the grain changed to per-session — LifecycleStateView's field docs "
        "and counts.py's tenancy paragraph both have to change with it"
    )
    assert view1.user_prefix == view2.user_prefix == UserPrefix(s1)
