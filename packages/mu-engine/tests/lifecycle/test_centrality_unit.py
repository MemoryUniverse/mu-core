"""``CentralityIndex`` + ``CentralityService`` — the A4 degree-centrality projection
(``lifecycle/centrality.py``).

Isolated logic only (the projection, the degree count, the noise floor, the normalisation, the
absent-vs-zero distinction, the tenancy of the index, the truncation withhold, the RAM bound and
the read deadline) against in-memory fakes. The REAL-FalkorDB proof of the same behaviour — and
the proof that a real production service's gate actually flips — is ``test_centrality_int.py``.

The port under test is graphify's ``god_nodes`` degree count
(``other_repos/graphify/graphify/analyze.py:109-130``, clone f5a3592), so several tests below
assert graphify's own ``networkx.Graph`` semantics explicitly: parallel edges collapse, and a
structurally isolated node is excluded (``analyze.py:87-88``).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from mu_engine.lifecycle.centrality import (
    CentralityIndex,
    CentralityLookup,
    CentralityService,
    CentralitySettings,
    LtmCentralityStorePort,
    projection_key,
)
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored

pytestmark = pytest.mark.unit

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _ns(
    *,
    user: str = "u1",
    org: str = "o1",
    session: str = "s1",
    visibility: Visibility = Visibility.PRIVATE,
) -> Namespace:
    return Namespace(
        org=org,
        workspace="w1",
        # CANONICAL §1 rule 4 (enforced by ``Namespace``): a SHARED namespace's user slot is ``*``
        # — the ROOM is the tenant, and it is carried in ``session``.
        user="*" if visibility is Visibility.SHARED else user,
        session=session,
        visibility=visibility,
    )


def _fact(
    ns: Namespace, subject: str | None, obj: str | None, *, predicate: str = "uses"
) -> MemoryItem:
    return MemoryItem(
        content="c",
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
    )


class FakeLtm:
    """An in-memory store satisfying ``LtmCentralityStorePort`` structurally.

    Keyed by :func:`projection_key`, i.e. the SAME grouping ``graph_recall`` itself filters on for
    ``session_scope=None`` — so a read for one tenant can never see another's facts, exactly as
    ``FalkorLtmAdapter.graph_name_for`` enforces physically.

    ``upsert_fact`` EXISTS here and RAISES. That is the point: this service must never write, and a
    fake that merely lacked the method would let a write slip through as an ``AttributeError``
    somebody could mistake for a wiring bug.
    """

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self.by_key: dict[str, list[MemoryItem]] = {}
        self.read_calls: list[tuple[str, int, str | None, frozenset[str] | None]] = []
        self._delay_s = delay_s

    def seed(self, ns: Namespace, *items: MemoryItem) -> None:
        self.by_key.setdefault(projection_key(ns), []).extend(items)

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        session_scope: str | None = None,
    ) -> list[Scored[MemoryItem]]:
        self.read_calls.append((ns.to_prefix(), limit, session_scope, caller_identity_set))
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        items = self.by_key.get(projection_key(ns), [])[:limit]
        return [
            Scored(item=item, score=1.0, channel=RecallChannel.LTM_GRAPH, rank=i)
            for i, item in enumerate(items)
        ]

    async def upsert_fact(self, item: MemoryItem) -> None:  # pragma: no cover - must never run
        raise AssertionError(
            "CentralityService wrote to the LTM tier. It is a READ-ONLY precompute: "
            "upsert_fact re-SETs state/invalid_at/pinned/version from a stale snapshot "
            "(falkor_ltm.py:313-332) and resurrects superseded facts."
        )


def _service(store: Any, index: CentralityIndex, **kw: Any) -> CentralityService:
    return CentralityService(ltm=store, index=index, settings=CentralitySettings(**kw))


# ---------------------------------------------------------------- the ported degree semantics ---


async def test_degree_is_distinct_neighbours_and_parallel_edges_collapse() -> None:
    """``nx.Graph`` semantics (``analyze.py:115``): ten facts about ONE pair are ONE adjacency.

    Without the collapse those ten facts would mint a fake hub — the "mechanically accumulated
    edges" ``god_nodes``' own docstring excludes.
    """
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, *(_fact(ns, "Ada", "Postgres", predicate=f"p{i}") for i in range(10)))

    await _service(store, index).refresh(ns)

    assert index.centrality_for(_fact(ns, "Ada", "Postgres")) == pytest.approx(0.0)


async def test_a_real_hub_scores_by_its_distinct_neighbour_count() -> None:
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(5)))

    report = await _service(store, index).refresh(ns)

    assert report.entities == 6  # postgres + 5 services
    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) == pytest.approx(0.5)


async def test_the_structural_isolation_floor_is_applied() -> None:
    """``analyze.py:87-88``'s ``G.degree(node) <= 1`` exclusion: a leaf pair is not a hub, and the
    answer is a PRESENT 0.0 (real evidence of peripherality), never an absent ``None``."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, _fact(ns, "Ada", "Tea"))

    await _service(store, index).refresh(ns)

    assert index.centrality_for(_fact(ns, "Ada", "Tea")) == 0.0


async def test_the_floor_is_configurable_and_zero_disables_it() -> None:
    ns = _ns()
    store = FakeLtm()
    store.seed(ns, _fact(ns, "Ada", "Tea"))

    floored = CentralityIndex(CentralitySettings(min_hub_degree=1))
    unfloored = CentralityIndex(CentralitySettings(min_hub_degree=0))
    await _service(store, floored, min_hub_degree=1).refresh(ns)
    await _service(store, unfloored, min_hub_degree=0).refresh(ns)

    assert floored.centrality_for(_fact(ns, "Ada", "Tea")) == 0.0
    assert unfloored.centrality_for(_fact(ns, "Ada", "Tea")) == pytest.approx(0.1)  # deg 1 / cap


async def test_the_score_saturates_at_the_degree_cap() -> None:
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings(degree_cap=5))
    store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(50)))

    await _service(store, index, degree_cap=5).refresh(ns)

    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) == 1.0  # min(50/5, 1), not 10.0


async def test_a_fact_is_as_central_as_its_best_endpoint_not_their_mean() -> None:
    """``god_nodes`` ranks NODES: a fact touching a hub is load-bearing even when its other
    endpoint is a leaf. A mean would let that leaf halve a genuine hub attachment."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(8)))

    await _service(store, index).refresh(ns)

    # deg(postgres)=8 -> 0.8; deg(Svc0)=1 -> floored to 0. max -> 0.8, mean would be 0.4.
    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) == pytest.approx(0.8)


async def test_self_loops_add_no_neighbour() -> None:
    """``nx.Graph`` would count a self-loop as degree 2 — an inflation with no meaning here."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings(min_hub_degree=0))
    store.seed(ns, _fact(ns, "Ada", "ada"), _fact(ns, "Ada", "Bo"))

    report = await _service(store, index, min_hub_degree=0).refresh(ns)

    assert report.entities == 2  # ada + bo, the self-loop contributed nothing
    assert index.centrality_for(_fact(ns, "Ada", "Bo")) == pytest.approx(0.1)  # deg 1, not 3


async def test_entity_identity_is_case_and_whitespace_folded_like_the_graph_tiers_own_key() -> None:
    """``falkor_ltm.py:731`` MERGEs ``:Entity`` on ``name.strip().casefold()``. Any other key here
    would count "Postgres" and " postgres " as two nodes and compute a degree the real graph does
    not have."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(
        ns,
        _fact(ns, "Postgres", "A"),
        _fact(ns, " postgres ", "B"),
        _fact(ns, "POSTGRES", "C"),
    )

    report = await _service(store, index).refresh(ns)

    assert report.entities == 4  # ONE postgres, three leaves
    assert index.centrality_for(_fact(ns, "postgres", "A")) == pytest.approx(0.3)


# ------------------------------------------------------------------- absent vs present zero ----


async def test_an_item_with_no_triple_is_absent_not_zero() -> None:
    """An unstructured capture turn is not IN the entity graph (``falkor_ltm.py:366-367`` skips
    exactly these when materializing edges). The graph has nothing to say about it, so it must not
    be penalised for not being a fact."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, _fact(ns, "Postgres", "A"), _fact(ns, "Postgres", "B"))
    await _service(store, index).refresh(ns)

    assert index.centrality_for(_fact(ns, None, None)) is None
    assert index.centrality_for(_fact(ns, "Postgres", None)) is None
    assert index.centrality_for(_fact(ns, None, "Postgres")) is None


async def test_a_namespace_with_no_projection_is_absent() -> None:
    """FULL-LOCAL with no graph tier configured, a fresh install, a disabled service: every one of
    them must report ABSENT, never a fabricated 0.0."""
    index = CentralityIndex(CentralitySettings())

    assert index.centrality_for(_fact(_ns(), "Ada", "Bo")) is None


async def test_a_disabled_service_reads_nothing_and_publishes_nothing() -> None:
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(5)))

    report = await _service(store, index, enabled=False).refresh(ns)

    assert report.skipped is True
    assert report.published is False
    assert store.read_calls == []  # not merely "wrote nothing" — it did not even READ
    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) is None


# ------------------------------------------------------------------------------- tenancy (η) ---


async def test_one_index_serving_two_tenants_never_crosses_them() -> None:
    """CLAUDE.md rule 4. Both tenants use the SAME entity name; a projection that crossed the wall
    would give tenant B's single fact tenant A's degree — a leak AND a wrong number."""
    ns_a, ns_b = _ns(user="ua"), _ns(user="ub")
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns_a, *(_fact(ns_a, "Postgres", f"SvcA{i}") for i in range(9)))
    store.seed(ns_b, _fact(ns_b, "Postgres", "SvcB0"))

    service = _service(store, index)  # ONE service, ONE index, both tenants
    await service.refresh(ns_a)
    await service.refresh(ns_b)

    assert index.centrality_for(_fact(ns_a, "Postgres", "SvcA0")) == pytest.approx(0.9)
    assert index.centrality_for(_fact(ns_b, "Postgres", "SvcB0")) == 0.0  # NOT 0.9, NOT 1.0


async def test_a_lookup_is_keyed_by_the_items_namespace_not_the_last_refreshed_one() -> None:
    """The mutation this kills: an index that remembered "the current namespace" instead of keying
    every projection would happily score tenant B's item against tenant A's graph."""
    ns_a, ns_b = _ns(user="ua"), _ns(user="ub")
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns_a, *(_fact(ns_a, "Postgres", f"SvcA{i}") for i in range(9)))

    await _service(store, index).refresh(ns_a)

    assert index.centrality_for(_fact(ns_b, "Postgres", "SvcA0")) is None


async def test_private_projections_federate_the_users_sessions() -> None:
    """The ``:Entity`` sub-graph is USER-scoped, session-less (``falkor_ltm.py:146-164``), so a
    fact captured in session A and one captured in session B belong to ONE degree. A session-scoped
    projection would split every hub by whichever session happened to run the sweep."""
    s1, s2 = _ns(session="s1"), _ns(session="s2")
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(s1, *(_fact(s1, "Postgres", f"Early{i}") for i in range(6)))

    await _service(store, index).refresh(s1)

    assert projection_key(s1) == projection_key(s2)
    assert index.centrality_for(_fact(s2, "Postgres", "Later0")) == pytest.approx(0.6)
    assert store.read_calls[0][2] is None  # session_scope=None, deliberately


async def test_shared_rooms_are_exact_and_never_federated() -> None:
    """Rooms are real walls: ``session_scope`` never relaxes SHARED (``falkor_ltm.py:121-122``),
    so two rooms are two projections even for the same org/workspace."""
    room_a = _ns(session="roomA", visibility=Visibility.SHARED)
    room_b = _ns(session="roomB", visibility=Visibility.SHARED)

    assert projection_key(room_a) != projection_key(room_b)
    assert projection_key(room_a) == room_a.to_prefix()

    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(room_a, *(_fact(room_a, "Postgres", f"Svc{i}") for i in range(9)))
    await _service(store, index).refresh(room_a)

    assert index.centrality_for(_fact(room_a, "Postgres", "Svc0")) == pytest.approx(0.9)
    assert index.centrality_for(_fact(room_b, "Postgres", "Svc0")) is None


async def test_a_sweep_forges_no_caller_identity() -> None:
    """A sweep has no caller; inventing one would be an authorization forgery. The SHARED-plane
    consequence (room-exact but not principal-partitioned) is stated in the module docstring and
    escalated to ARCHITECTURE-DELTAS, not silently traded away here."""
    ns = _ns(visibility=Visibility.SHARED, session="roomA")
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, _fact(ns, "Ada", "Bo"))

    await _service(store, index).refresh(ns)

    assert store.read_calls[0][3] is None


# ----------------------------------------------------------- truncation, bounds and deadlines ---


async def test_a_truncated_projection_is_withheld_not_published() -> None:
    """``graph_recall`` orders ``valid_at DESC LIMIT $limit`` (``falkor_ltm.py:565``), so a
    projection over the cap is "the newest N facts" — a systematically UNDER-counted lower bound
    whose victims depend purely on sweep timing. Publishing it would manufacture false
    "peripheral" evidence that PENALISES S(m). Absence renormalises away; a wrong number does not.
    """
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(12)))

    report = await _service(store, index, max_facts_per_pass=6).refresh(ns)

    assert report.truncated is True
    assert report.published is False
    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) is None
    assert len(index) == 0


async def test_a_truncated_pass_leaves_the_previous_projection_untouched() -> None:
    """The withhold must not be a silent wipe either: a namespace that grew past the cap keeps the
    last good projection rather than losing structural salience entirely."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(4)))
    await _service(store, index, max_facts_per_pass=100).refresh(ns)
    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) == pytest.approx(0.4)

    store.seed(ns, *(_fact(ns, "Postgres", f"More{i}") for i in range(20)))
    report = await _service(store, index, max_facts_per_pass=6).refresh(ns)

    assert report.truncated is True
    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) == pytest.approx(0.4)


async def test_exactly_max_facts_per_pass_is_not_truncation() -> None:
    """The off-by-one that the obvious ``len(hits) >= limit`` test gets wrong: a namespace holding
    exactly the cap is complete, and withholding its projection would be a false alarm that costs
    it structural salience forever."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(6)))

    report = await _service(store, index, max_facts_per_pass=6).refresh(ns)

    assert report.truncated is False
    assert report.published is True
    assert report.evaluated == 6
    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) == pytest.approx(0.6)


async def test_the_read_asks_for_one_more_than_the_cap_so_truncation_is_exact() -> None:
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, _fact(ns, "Ada", "Bo"))

    await _service(store, index, max_facts_per_pass=6).refresh(ns)

    assert store.read_calls[0][1] == 7


async def test_the_index_is_bounded_and_evicts_least_recently_refreshed() -> None:
    """DEV-STANDARDS: bounded, never unbounded. On a multi-tenant process the index must not grow
    with the tenant count."""
    store, index = FakeLtm(), CentralityIndex(CentralitySettings(max_namespaces=2))
    namespaces = [_ns(user=f"u{i}") for i in range(3)]
    for ns in namespaces:
        store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(5)))

    service = _service(store, index, max_namespaces=2)
    for ns in namespaces:
        await service.refresh(ns)

    assert len(index) == 2
    assert index.centrality_for(_fact(namespaces[0], "Postgres", "Svc0")) is None  # evicted
    assert index.centrality_for(_fact(namespaces[2], "Postgres", "Svc0")) == pytest.approx(0.5)


async def test_a_refresh_replaces_a_projection_it_never_merges_into_it() -> None:
    """A shrinking graph must not keep phantom edges: a projection is a snapshot of NOW."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(9)))
    service = _service(store, index)
    await service.refresh(ns)
    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) == pytest.approx(0.9)

    store.by_key[projection_key(ns)] = [_fact(ns, "Postgres", "Svc0")]
    await service.refresh(ns)

    assert index.centrality_for(_fact(ns, "Postgres", "Svc0")) == 0.0  # deg 1, floored


async def test_the_read_is_deadlined_and_a_timeout_leaves_the_index_intact() -> None:
    """DEV-STANDARDS timeouts/resilience. A hung graph must not hang a sweep, and it must not
    publish a partial projection on the way out either."""
    ns = _ns()
    store, index = FakeLtm(delay_s=0.5), CentralityIndex(CentralitySettings())
    store.seed(ns, _fact(ns, "Ada", "Bo"))

    with pytest.raises(TimeoutError):
        await _service(store, index, read_timeout_s=0.01).refresh(ns)

    assert len(index) == 0


async def test_cancellation_propagates_and_is_not_counted_as_a_failure() -> None:
    """DEV-STANDARDS rule 1: a cancelled sweep is not an error."""
    ns = _ns()
    store, index = FakeLtm(delay_s=5.0), CentralityIndex(CentralitySettings())
    store.seed(ns, _fact(ns, "Ada", "Bo"))
    errors: list[str] = []

    class _Metrics:
        def inc(self, name: str, *, labels: dict[str, str] | None = None, value: float = 1) -> None:
            errors.append(name)

        def observe(
            self, name: str, value: float, *, labels: dict[str, str] | None = None
        ) -> None: ...

    service = CentralityService(ltm=store, index=index, metrics=_Metrics())  # type: ignore[arg-type]
    task = asyncio.create_task(service.refresh(ns))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert errors == []


# ------------------------------------------------------------------------ contracts and shape ---


async def test_the_service_never_writes_to_the_store() -> None:
    """The load-bearing property of the corrected design. ``FakeLtm.upsert_fact`` raises with the
    reason: a blind re-upsert of a pass-time snapshot re-SETs ``state``/``invalid_at``/``pinned``/
    ``version`` (``falkor_ltm.py:313-332``) and resurrects facts superseded during the pass —
    reproduced live before this module was rewritten."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    store.seed(ns, *(_fact(ns, "Postgres", f"Svc{i}") for i in range(5)))

    report = await _service(store, index).refresh(ns)  # would raise AssertionError if it wrote

    assert report.published is True


def test_the_narrow_port_is_read_only() -> None:
    """Principle of least privilege, asserted structurally: a class that can ONLY read satisfies
    the port, so nothing about the seam invites a write back in."""

    class ReadOnly:
        async def graph_recall(
            self,
            ns: Namespace,
            *,
            subject: str | None = None,
            predicate: str | None = None,
            limit: int,
            caller_identity_set: frozenset[str] | None = None,
            session_scope: str | None = None,
        ) -> list[Scored[MemoryItem]]:
            return []

    assert isinstance(ReadOnly(), LtmCentralityStorePort)


def test_the_index_satisfies_the_synchronous_lookup_seam() -> None:
    assert isinstance(CentralityIndex(CentralitySettings()), CentralityLookup)


async def test_the_report_is_content_free() -> None:
    """CLAUDE.md rule 3: no subject, object, predicate, id or content may appear in anything this
    service emits."""
    ns = _ns()
    store, index = FakeLtm(), CentralityIndex(CentralitySettings())
    secret = _fact(ns, "AdaLovelace", "SecretProject")
    secret.content = "the passphrase is hunter2"
    store.seed(ns, secret, _fact(ns, "AdaLovelace", "Postgres"))

    report = await _service(store, index).refresh(ns)

    serialized = report.model_dump_json()
    for leaked in ("AdaLovelace", "SecretProject", "Postgres", secret.id, secret.content):
        assert leaked not in serialized


def test_the_index_can_forget_a_namespace() -> None:
    ns = _ns()
    index = CentralityIndex(CentralitySettings())
    index.publish(ns, {"ada": frozenset({"bo", "cy"})})
    assert index.centrality_for(_fact(ns, "Ada", "Bo")) == pytest.approx(0.2)

    index.drop(ns)

    assert index.centrality_for(_fact(ns, "Ada", "Bo")) is None
