"""A4 structural salience against the REAL mu-dev-falkordb LTM tier — ZERO mocked graph.

Proves the five things a fake store cannot:

1. ``FalkorLtmAdapter`` satisfies the NARROW, READ-ONLY ``LtmCentralityStorePort`` structurally —
   the service needs no new port method, no adapter edit, and no change to any file another lane
   owns.
2. Real degrees come out of the real entity graph, and the projection agrees with the tenancy
   grouping the adapter's own Cypher filters on (a drift guard against
   ``_resolve_memory_namespace_filter``, not a re-statement of it).
3. **The term is not inert.** A real ``DemotionService`` — the exact production service the sweep
   calls, with the real ``SalienceStrategy`` — RESCUES an MTM item it would otherwise demote,
   solely because that item's entities are hubs in the real LTM graph. This is the composition
   production performs, not a hand-built pairing.
4. Namespace scoping is real at the STORE level: two tenants land in two physical FalkorDB graphs
   and neither can inflate the other's degree — and two SHARED rooms that share ONE physical graph
   stay separated by the namespace filter alone.
5. **A refresh never writes.** A fact superseded before the pass stays superseded, and the
   namespace's raw node properties are byte-identical across a refresh. The first cut of A4
   re-``upsert_fact``-ed each item and resurrected exactly such a fact (reproduced live).

**Teardown discipline.** This file NEVER sweeps ``mu_g__*`` (the pattern
``tests/storage/test_graph_falkor_int.py``'s fixture uses would destroy concurrently running
lanes' data on this shared dev instance). It deletes exactly the graphs its own namespaces resolve
to, computed through the real ``graph_name_for``, and touches nothing else. Every test uses a
uuid-unique ``org``, so those names cannot collide with another lane's.

If the container is not up the fixture RAISES — the test is BLOCKED, never faked (DEV-STANDARDS).
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB

from mu_contracts.config import Settings
from mu_engine.lifecycle.centrality import (
    CentralityIndex,
    CentralityService,
    CentralitySettings,
    LtmCentralityStorePort,
    projection_key,
)
from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.adapters.falkor_ltm import (
    FalkorLtmAdapter,
    _resolve_memory_namespace_filter,
)
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryTier, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.integration

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


class _NamespaceFactory:
    """Records every namespace it mints so teardown can target the EXACT physical graphs this test
    occupied, instead of guessing at a name pattern."""

    def __init__(self, uid: str) -> None:
        self._uid = uid
        self.created: list[Namespace] = []

    def __call__(
        self,
        *,
        user: str = "u1",
        session: str = "s1",
        visibility: Visibility = Visibility.PRIVATE,
    ) -> Namespace:
        ns = Namespace(
            org=f"cen{self._uid}",
            workspace=f"ws{self._uid}",
            # CANONICAL §1 rule 4: a SHARED namespace's user slot is ``*``; the ROOM is ``session``.
            user="*" if visibility is Visibility.SHARED else user,
            session=session,
            visibility=visibility,
        )
        self.created.append(ns)
        return ns


@pytest.fixture
def make_ns(uid: str) -> _NamespaceFactory:
    return _NamespaceFactory(uid)


@pytest_asyncio.fixture
async def ltm(settings: Settings, make_ns: _NamespaceFactory) -> AsyncIterator[FalkorLtmAdapter]:
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    await db.select_graph("_probe").query("RETURN 1")  # fail-loud if the container is down
    adapter = FalkorLtmAdapter(db)
    try:
        yield adapter
    finally:
        # ONLY the graphs this test's own uuid-unique namespaces resolve to.
        for name in {adapter.graph_name_for(ns) for ns in make_ns.created}:
            with contextlib.suppress(Exception):  # best-effort teardown
                await db.select_graph(name).delete()


@pytest.fixture
def make_fact() -> Callable[..., MemoryItem]:
    def _make(
        ns: Namespace,
        subject: str,
        obj: str,
        *,
        predicate: str = "uses",
        authorized_ids: Sequence[str] = (),
    ) -> MemoryItem:
        item = MemoryItem(
            content=f"{subject} {predicate} {obj}",
            kind=MemoryKind.PROPOSITION,
            namespace=ns,
            owner_id=ns.user,
            workspace_id=ns.workspace,
            session_id=ns.session,
            subject=subject,
            predicate=predicate,
            object=obj,
            polarity=Polarity.POSITIVE,
            created_at=_EPOCH,
            valid_at=datetime.now(UTC),
        )
        if authorized_ids:
            # Model A ACL lives in ``metadata`` and is promoted onto the node by
            # ``GraphMapper.to_store`` (``storage/mappers/graph_mapper.py:58-59``).
            item.metadata = {**item.metadata, "authorized_ids": list(authorized_ids)}
        return item

    return _make


def _index(**kw: object) -> tuple[CentralityIndex, CentralitySettings]:
    cfg = CentralitySettings(**kw)  # type: ignore[arg-type]
    return CentralityIndex(cfg), cfg


def _service(ltm: FalkorLtmAdapter, index: CentralityIndex, cfg: CentralitySettings):  # type: ignore[no-untyped-def]
    return CentralityService(ltm=ltm, index=index, settings=cfg)


# ------------------------------------------------------------------------------- the seam -------


async def test_the_real_adapter_satisfies_the_narrow_read_only_port(ltm: FalkorLtmAdapter) -> None:
    """No new ``GraphStorePort`` method and no ``storage/adapters/**`` edit: the ONE capability the
    service needs already exists on the shipped adapter."""
    assert isinstance(ltm, LtmCentralityStorePort)


def test_the_projection_key_agrees_with_the_adapters_own_namespace_filter(
    make_ns: _NamespaceFactory,
) -> None:
    """``projection_key`` deliberately RE-STATES ``_resolve_memory_namespace_filter(...,
    session_scope=None)`` rather than importing it (engine-lifecycle must not reach into a storage
    adapter's privates). This is the drift guard that makes the duplication safe."""
    for ns in (
        make_ns(),
        make_ns(user="other"),
        make_ns(session="roomA", visibility=Visibility.SHARED),
    ):
        _, expected = _resolve_memory_namespace_filter(ns, session_scope=None)

        assert projection_key(ns) == expected


# ----------------------------------------------------------------- real degrees, real graph ------


async def test_degrees_are_computed_over_the_real_entity_graph(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    ns = make_ns()
    for i in range(5):
        await ltm.upsert_fact(make_fact(ns, "Postgres", f"Service{i}"))
    index, cfg = _index()

    report = await _service(ltm, index, cfg).refresh(ns)

    assert report.evaluated == 5
    assert report.scored == 5
    assert report.entities == 6  # postgres + 5 services
    assert report.published is True
    assert index.centrality_for(make_fact(ns, "Postgres", "Service0")) == pytest.approx(0.5)
    assert index.centrality_for(make_fact(ns, "Service0", "Postgres")) == pytest.approx(0.5)


async def test_the_projection_federates_the_users_sessions(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    """The ``:Entity`` sub-graph is USER-scoped, session-less (``falkor_ltm.py:146-164``), so a
    fact captured in session A and one in session B belong to ONE degree. A session-scoped read
    would split this hub in half depending on which session ran the sweep."""
    s1, s2 = make_ns(session="s1"), make_ns(session="s2")
    for i in range(3):
        await ltm.upsert_fact(make_fact(s1, "Postgres", f"Early{i}"))
    for i in range(3):
        await ltm.upsert_fact(make_fact(s2, "Postgres", f"Later{i}"))
    index, cfg = _index()

    report = await _service(ltm, index, cfg).refresh(s1)

    assert report.evaluated == 6  # both sessions, one user
    assert index.centrality_for(make_fact(s2, "Postgres", "Later0")) == pytest.approx(0.6)


async def test_only_active_facts_count_superseded_history_never_inflates_a_degree(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    """MU's own principled noise exclusion, and the one that replaces graphify's corpus-specific
    filters: a fact that lost a contradiction is not structural evidence any more."""
    ns = make_ns()
    facts = [make_fact(ns, "Postgres", f"Service{i}") for i in range(6)]
    for f in facts:
        await ltm.upsert_fact(f)
    for loser in facts[:3]:
        await ltm.invalidate(ns, loser.id, facts[5].id, at=datetime.now(UTC), reason="test")
    index, cfg = _index()

    report = await _service(ltm, index, cfg).refresh(ns)

    assert report.evaluated == 3
    assert index.centrality_for(make_fact(ns, "Postgres", "Service5")) == pytest.approx(0.3)


# ------------------------------------------------- the term is NOT inert: a real gate flips ------


async def test_a_real_demotionservice_rescues_an_item_because_of_its_structural_salience(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    """The claim A4 has to earn. Two MTM candidates, identical in recency/usage/importance,
    are fed to the REAL ``DemotionService`` — the same service
    ``MemoryLifecycleManager.sweep_namespace_now`` calls, holding the same
    ``SalienceStrategy``. One is about a hub in the real LTM graph, one is about a leaf.
    Without the projection both are demoted; with it the hub fact is rescued.

    Note which tier each object lives on: the LTM graph supplies the STRUCTURE, the MTM candidates
    are what gets scored. That is the corrected A4 shape — asking the graph about a candidate,
    rather than stamping a scalar onto a fact that is already in the graph and that no gate ever
    scores.
    """
    ns = make_ns()
    for i in range(10):  # a genuine hub: deg(Postgres) = 10
        await ltm.upsert_fact(make_fact(ns, "Postgres", f"Service{i}"))
    await ltm.upsert_fact(make_fact(ns, "Ada", "Tea"))  # a leaf pair

    hub_candidate = make_fact(ns, "Postgres", "Service0")
    leaf_candidate = make_fact(ns, "Ada", "Tea")
    for candidate in (hub_candidate, leaf_candidate):
        candidate.tier = MemoryTier.MTM
        candidate.importance_score = 0.5
        candidate.access_count = 0

    lifecycle = LifecycleSettings()
    # Age both candidates past the half-life so their three-term score sits BELOW demote_mtm —
    # i.e. both are genuinely stale and today's engine demotes both.
    clock = FrozenClock(_EPOCH.replace(year=2026, month=1, day=3))  # 48h -> rec = 0.25
    stm, mtm = _FakeStm(), _FakeMtm()

    without = DemotionService(
        stm=stm,
        mtm_remove=mtm,
        salience=SalienceStrategy(lifecycle.salience),
        settings=lifecycle,
        clock=clock,
    )
    baseline = await without.demote(ns, [hub_candidate, leaf_candidate])
    assert baseline.demoted == 2, "precondition: both are stale enough to demote pre-A4"

    index, cfg = _index()
    await _service(ltm, index, cfg).refresh(ns)
    with_a4 = DemotionService(
        stm=_FakeStm(),
        mtm_remove=_FakeMtm(),
        salience=SalienceStrategy(lifecycle.salience, centrality=index),
        settings=lifecycle,
        clock=clock,
    )
    after = await with_a4.demote(ns, [hub_candidate, leaf_candidate])

    by_id = {o.memory_id: o for o in after.outcomes}
    assert by_id[hub_candidate.id].demoted is False, "a hub fact must be rescued"
    assert by_id[hub_candidate.id].reason == "salience_at_or_above_demote_mtm"
    assert by_id[leaf_candidate.id].demoted is True, "a leaf fact stays stale"


class _FakeStm:
    """The demotion write-ahead sink. Real Redis is another lane's fixture and is irrelevant to the
    gate under test; the LTM GRAPH — the thing A4 actually reads — is real."""

    def __init__(self) -> None:
        self.put_ids: list[str] = []

    async def put(self, item: MemoryItem) -> None:
        self.put_ids.append(item.id)

    async def evict(self, ns: Namespace, memory_id: str) -> None:
        self.put_ids.remove(memory_id)


class _FakeMtm:
    def __init__(self) -> None:
        self.removed: list[str] = []

    async def remove(self, ns: Namespace, memory_id: str) -> None:
        self.removed.append(memory_id)


# ------------------------------------------------------------------------------ tenancy (η) -----


async def test_two_private_tenants_land_in_two_graphs_and_neither_inflates_the_other(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    """CLAUDE.md rule 4, proven at the STORE level. Both tenants use the SAME entity name; a
    projection that crossed the wall would give tenant B's single fact tenant A's degree — a leak
    AND a wrong number."""
    ns_a, ns_b = make_ns(user="ua"), make_ns(user="ub")
    assert ltm.graph_name_for(ns_a) != ltm.graph_name_for(ns_b)  # physical partition

    for i in range(9):
        await ltm.upsert_fact(make_fact(ns_a, "Postgres", f"ServiceA{i}"))
    await ltm.upsert_fact(make_fact(ns_b, "Postgres", "ServiceB0"))
    index, cfg = _index()
    service = _service(ltm, index, cfg)  # ONE service, ONE index, both tenants

    report_a = await service.refresh(ns_a)
    report_b = await service.refresh(ns_b)

    assert report_a.entities == 10
    assert report_b.entities == 2  # NOT 11 — no accumulation across the two passes
    assert index.centrality_for(make_fact(ns_a, "Postgres", "ServiceA0")) == pytest.approx(0.9)
    assert index.centrality_for(make_fact(ns_b, "Postgres", "ServiceB0")) == 0.0


async def test_two_shared_rooms_share_one_physical_graph_and_stay_separated(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    """The SHARED plane, which the first cut never exercised. Both rooms resolve to the SAME
    physical FalkorDB graph, so room isolation rests entirely on the namespace filter — exactly the
    case worth proving on a real store rather than in a fake's dict."""
    room_a = make_ns(session="roomA", visibility=Visibility.SHARED)
    room_b = make_ns(session="roomB", visibility=Visibility.SHARED)
    assert ltm.graph_name_for(room_a) == ltm.graph_name_for(room_b)  # ONE graph, by design

    for i in range(9):
        await ltm.upsert_fact(make_fact(room_a, "Postgres", f"Service{i}"))
    await ltm.upsert_fact(make_fact(room_b, "Postgres", "Elsewhere"))
    index, cfg = _index()
    service = _service(ltm, index, cfg)

    report_a = await service.refresh(room_a)

    assert report_a.evaluated == 9  # room B's fact is NOT in room A's projection
    assert index.centrality_for(make_fact(room_a, "Postgres", "Service0")) == pytest.approx(0.9)
    assert index.centrality_for(make_fact(room_b, "Postgres", "Elsewhere")) is None


async def test_within_one_shared_room_the_degree_spans_every_principals_facts(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    """A STATED LIMITATION, pinned by a test so it cannot change unnoticed.

    A sweep has no caller, so no ``caller_identity_set`` is passed and the projection spans every
    ACTIVE fact in the room regardless of Model-A ``authorized_ids``. Room isolation is exact (the
    test above); PRINCIPAL partitioning inside a room is not attempted. Nothing is returned to any
    principal — no content, no score — but whether a SHARED sweep should compute structure over the
    whole room or per ACL partition is an owner decision, recorded in ARCHITECTURE-DELTAS. If that
    decision lands the other way, THIS test is the one that must change.
    """
    room = make_ns(session="roomA", visibility=Visibility.SHARED)
    for i in range(3):
        await ltm.upsert_fact(make_fact(room, "Postgres", f"Alice{i}", authorized_ids=["alice"]))
    for i in range(3):
        await ltm.upsert_fact(make_fact(room, "Postgres", f"Bob{i}", authorized_ids=["bob"]))
    index, cfg = _index()

    report = await _service(ltm, index, cfg).refresh(room)

    assert report.evaluated == 6
    assert index.centrality_for(make_fact(room, "Postgres", "Alice0")) == pytest.approx(0.6)
    # ...while an ACL-scoped READ of the same room legitimately sees half of it.
    alice_view = await ltm.graph_recall(
        room, limit=50, caller_identity_set=frozenset({"alice"}), session_scope=None
    )
    assert len(alice_view) == 3


# ----------------------------------------------------------------- the pass writes NOTHING ------


async def test_a_refresh_does_not_resurrect_a_superseded_fact(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    """The regression guard for the blocker that forced this module's rewrite.

    The first cut re-``upsert_fact``-ed every item from the pass's own read snapshot, and
    ``_upsert_fact_impl`` SETs the FULL node — ``m.state``, ``m.invalid_at``, ``m.pinned``,
    ``m.version`` — from that snapshot (``falkor_ltm.py:313-332``). A supersession landing during
    the pass was therefore reverted: reproduced live, the victim came back
    ``state='active', invalid_at=''`` with its ``SUPERSEDED_BY`` edge still attached, so loser and
    winner of a resolved contradiction were both ACTIVE at once. Same mechanism reverted
    ``access_count``, ``cold``, and the ``pinned``/``version`` pin group.
    """
    ns = make_ns()
    facts = [make_fact(ns, "Postgres", f"Service{i}") for i in range(4)]
    for f in facts:
        await ltm.upsert_fact(f)
    victim, winner = facts[0], facts[1]
    await ltm.invalidate(ns, victim.id, winner.id, at=datetime.now(UTC), reason="test")
    graph = ltm._graph(ns)  # raw props: the only honest assertion
    props = "m.state, m.invalid_at, m.pinned, m.version, m.access_count"
    before = (await graph.query(f"MATCH (m:Memory) RETURN m.id, {props} ORDER BY m.id")).result_set
    index, cfg = _index()

    await _service(ltm, index, cfg).refresh(ns)

    after = (await graph.query(f"MATCH (m:Memory) RETURN m.id, {props} ORDER BY m.id")).result_set
    assert after == before, "a derived scalar must never rewrite bi-temporal or pin state"
    still_dead = await graph.query(
        "MATCH (m:Memory {id: $i}) RETURN m.state", params={"i": victim.id}
    )
    assert still_dead.result_set == [["superseded"]]
    active_ids = {h.item.id for h in await ltm.graph_recall(ns, limit=50, session_scope=None)}
    assert victim.id not in active_ids


async def test_a_refresh_is_idempotent_and_costs_no_writes_however_often_it_runs(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    """The old shape's write-skip optimisation disappeared exactly on the graphs big enough to
    matter (a sliding recency window changed degrees every pass, so it re-upserted continuously).
    A read-only pass has nothing to amplify."""
    ns = make_ns()
    for i in range(4):
        await ltm.upsert_fact(make_fact(ns, "Postgres", f"Service{i}"))
    index, cfg = _index()
    service = _service(ltm, index, cfg)
    graph = ltm._graph(ns)

    first = await service.refresh(ns)
    snapshot = (await graph.query("MATCH (m:Memory) RETURN m ORDER BY m.id")).result_set
    second = await service.refresh(ns)
    third = await service.refresh(ns)

    assert first.published and second.published and third.published
    assert (await graph.query("MATCH (m:Memory) RETURN m ORDER BY m.id")).result_set == snapshot
    assert index.centrality_for(make_fact(ns, "Postgres", "Service0")) == pytest.approx(0.4)


async def test_a_truncated_pass_publishes_nothing_against_the_real_store(
    ltm: FalkorLtmAdapter, make_ns: _NamespaceFactory, make_fact: Callable[..., MemoryItem]
) -> None:
    """``graph_recall``'s ``ORDER BY m.valid_at DESC LIMIT $limit`` (``falkor_ltm.py:565``) makes an
    over-cap projection "the newest N facts" — a lower bound whose victims depend on sweep timing.
    Withheld, not published: absence renormalises away, a fabricated 0.0 penalises."""
    ns = make_ns()
    for i in range(12):
        await ltm.upsert_fact(make_fact(ns, "Postgres", f"Service{i}"))
    index, cfg = _index(max_facts_per_pass=6)

    report = await _service(ltm, index, cfg).refresh(ns)

    assert report.truncated is True
    assert report.published is False
    assert index.centrality_for(make_fact(ns, "Postgres", "Service0")) is None
