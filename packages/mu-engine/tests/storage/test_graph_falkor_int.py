"""Graph/LTM adapter — REAL mu-dev-falkordb, ZERO mocks.

Covers: upsert_fact + graph_recall, namespace isolation (both per-user AND per-org — the
multi-org harden, owner directive 2026-07-27), ``find_conflicts``, and the bi-temporal
invalidate-don't-delete guarantee (spec §8.7/§8.9): a superseded fact drops from
``graph_recall`` / ``facts_at(now)`` but SURVIVES in ``facts_at(t_old)``."""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB

from mu_contracts.config import get_settings
from mu_contracts.domain.errors import CallerIdentitySetRequiredError
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.authz import INTERNAL_ENGINE_READ
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.factories import STORE_REGISTRY
from mu_engine.storage.mappers.tenancy import tenant_partition_digest

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def ltm(
    falkor_db: FalkorDB, make_ns: Callable[..., Namespace]
) -> AsyncIterator[FalkorLtmAdapter]:
    yield FalkorLtmAdapter(falkor_db)
    # teardown: delete every graph this test file's adapter could have created. All of them
    # carry the ``mu_g__`` partition prefix (see FalkorLtmAdapter.graph_name_for) regardless of
    # whether org is included, so this catches both the per-user and per-org test graphs.
    for g in await falkor_db.list_graphs():
        name = g.decode() if isinstance(g, bytes) else g
        if name.startswith("mu_g__"):
            with contextlib.suppress(Exception):  # best-effort teardown only
                await falkor_db.select_graph(name).delete()


async def test_upsert_and_graph_recall(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    now = datetime.now(UTC)
    item = make_item(ns, "Ada uses Postgres", subject="Ada", predicate="uses", obj="Postgres")
    item.valid_at = now
    await ltm.upsert_fact(item)
    hits = await ltm.graph_recall(ns, subject="Ada", limit=5)
    assert [h.item.id for h in hits] == [item.id]
    assert hits[0].item.content == item.content  # lossless memory_json carrier round-trip


async def test_shared_graph_recall_with_no_caller_set_fails_closed(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """AD-179 — before this fix, ``graph_recall`` on a SHARED namespace with
    ``caller_identity_set`` OMITTED silently dropped the ``m.authorized_ids`` cypher predicate and
    ran an UNFILTERED SHARED query. It must instead raise, exactly as the STM tier already does
    for the identical input shape, and exactly as this same adapter's own ``traverse_entities``
    now does (test_falkor_traverse_authz_int.py)."""
    room = make_ns(visibility=Visibility.SHARED, session="roomA")
    secret = make_item(
        room,
        "Ada uses Postgres",
        subject="Ada",
        predicate="uses",
        obj="Postgres",
        authorized_ids=["principal-alice"],
    )
    await ltm.upsert_fact(secret)

    with pytest.raises(CallerIdentitySetRequiredError):
        await ltm.graph_recall(room, subject="Ada", limit=5)

    # the one legitimate, EXPLICIT bypass (the lifecycle centrality sweep) must still work, and
    # must still see the fact — it is an engine-internal read, not a per-caller filtered one.
    internal = await ltm.graph_recall(
        room, subject="Ada", limit=5, caller_identity_set=INTERNAL_ENGINE_READ
    )
    assert [h.item.id for h in internal] == [secret.id]


async def test_namespace_isolation(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns_a = make_ns(user="ua")
    ns_b = make_ns(user="ub")  # private graph per (workspace, user)
    a = make_item(ns_a, "A only", subject="X", predicate="p", obj="v")
    a.valid_at = datetime.now(UTC)
    await ltm.upsert_fact(a)
    assert (await ltm.graph_recall(ns_b, subject="X", limit=10)) == []


async def test_find_conflicts(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    now = datetime.now(UTC)
    f1 = make_item(ns, "Ada lives in Paris", subject="Ada", predicate="lives_in", obj="Paris")
    f2 = make_item(ns, "Ada lives in Berlin", subject="Ada", predicate="lives_in", obj="Berlin")
    f1.valid_at = now
    f2.valid_at = now
    await ltm.upsert_fact(f1)
    await ltm.upsert_fact(f2)
    conflicts = await ltm.find_conflicts(ns, "Ada", "lives_in")
    assert {c.id for c in conflicts} == {f1.id, f2.id}


# =================================================================================================
# ADR 0062 / AD-259's still-open half, closed: `GraphStorePort.reinforce`
# =================================================================================================
async def test_reinforce_bumps_access_count_and_updated_at_only(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """Real FalkorDB — payload-only stat write, everything else on the node untouched: the same
    contract `qdrant_mtm.py`'s `_reinforce_impl` docstring pins for MTM, proved here for the
    graph tier's fetch-then-`upsert_fact` shape instead of a Qdrant `set_payload`."""
    ns = make_ns()
    now = datetime.now(UTC)
    item = make_item(ns, "Ada uses Postgres", subject="Ada", predicate="uses", obj="Postgres")
    item.valid_at = now
    await ltm.upsert_fact(item)

    at = now + timedelta(hours=1)
    reinforced = await ltm.reinforce(ns, item.id, at=at)

    assert reinforced is not None
    assert reinforced.access_count == item.access_count + 1
    assert reinforced.updated_at == at
    # untouched by construction: created_at, content, state, the triple, cold.
    assert reinforced.created_at == item.created_at
    assert reinforced.content == item.content
    assert reinforced.state == item.state
    assert (reinforced.subject, reinforced.predicate, reinforced.object) == (
        item.subject,
        item.predicate,
        item.object,
    )

    # persisted, not just returned — a fresh read sees the same bump.
    refetched = await ltm.get_fact(ns, item.id)
    assert refetched is not None
    assert refetched.access_count == item.access_count + 1
    assert refetched.updated_at == at


async def test_reinforce_on_absent_memory_is_a_documented_noop(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
) -> None:
    """The caller passes only ids its own prior read just returned (`ports.py`'s
    `GraphStorePort.reinforce` docstring) — an id absent from this namespace's partition is a
    silent no-op, never a raise, exactly like every other tier's `reinforce`."""
    ns = make_ns()
    result = await ltm.reinforce(ns, "never-existed", at=datetime.now(UTC))
    assert result is None


async def test_bitemporal_invalidate_dont_delete(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t_old = datetime.now(UTC) - timedelta(days=2)
    t_now = datetime.now(UTC)
    loser = make_item(ns, "Ada at Acme", subject="Ada", predicate="works_at", obj="Acme")
    winner = make_item(ns, "Ada at Globex", subject="Ada", predicate="works_at", obj="Globex")
    loser.valid_at = t_old
    winner.valid_at = t_now
    await ltm.upsert_fact(loser)
    await ltm.upsert_fact(winner)

    await ltm.invalidate(ns, loser.id, winner.id, at=t_now, reason="moved-jobs")

    # present-tense recall drops the superseded loser (state + temporal axis).
    recall_ids = {h.item.id for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert loser.id not in recall_ids
    assert winner.id in recall_ids

    # facts_at(now) excludes the loser ...
    now_ids = {m.id for m in await ltm.facts_at(ns, t_now, subject="Ada")}
    assert loser.id not in now_ids
    # ... but history at t_old STILL contains it (invalidate-don't-delete, spec §8.7).
    hist_ids = {m.id for m in await ltm.facts_at(ns, t_old, subject="Ada")}
    assert loser.id in hist_ids

    # AD-269-adjacent fix (found proving AD-269 by running): a BY-ID read must see the SAME
    # supersession `graph_recall`/`facts_at` already do — `get_fact` reads the fact back out of
    # `m.memory_json`, so `invalidate` must rewrite that JSON carrier, not only the promoted
    # `state`/`invalid_at` NODE PROPERTIES the filtered reads use. MUTATION CHECK (run, red,
    # restored): drop the `loser.memory_json = $memory_json` clause from `_invalidate_impl`'s
    # Cypher — this assertion alone goes red (`state == ACTIVE`) while every assertion above it
    # stays green, since those all go through the FILTERED read paths this bug never touched.
    reread = await ltm.get_fact(ns, loser.id)
    assert reread is not None
    assert reread.state is MemoryState.SUPERSEDED, (
        "get_fact returned a stale state after invalidate() — the promoted node property and "
        "the memory_json carrier diverged"
    )
    assert reread.invalid_at == t_now


async def _node_count(falkor_db: FalkorDB, graph_name: str) -> int:
    """Raw node count in a physical FalkorDB graph — proof the partition really exists/is
    populated, independent of the adapter's own (namespace-filtered) read paths."""
    res = await falkor_db.select_graph(graph_name).query("MATCH (m:Memory) RETURN count(m)")
    rows = res.result_set or []
    return int(rows[0][0]) if rows else 0


async def test_multi_org_physical_isolation(
    ltm: FalkorLtmAdapter,
    make_item: Callable[..., MemoryItem],
    falkor_db: FalkorDB,
) -> None:
    """MULTI-ORG HARDEN proof (owner directive 2026-07-27): the physical graph-partition name
    keys on ``(org, workspace, visibility|user)`` — NOT workspace alone. Two orgs sharing the
    SAME workspace id must land in two DIFFERENT physical FalkorDB graphs (not one graph
    filtered by ``namespace``), and a third namespace with the same org but a different
    workspace must be a third distinct graph. A cross-namespace recall returns nothing in
    every direction.
    """
    run = uuid.uuid4().hex[:8]
    shared_workspace = f"w1{run}"

    ns_org_a_w1 = Namespace(
        org=f"orgA{run}",
        workspace=shared_workspace,
        user="alice",
        session="s1",
        visibility=Visibility.PRIVATE,
    )
    ns_org_b_w1 = Namespace(  # SAME workspace name, DIFFERENT org
        org=f"orgB{run}",
        workspace=shared_workspace,
        user="alice",
        session="s1",
        visibility=Visibility.PRIVATE,
    )
    ns_org_a_w2 = Namespace(  # SAME org as the first, DIFFERENT workspace
        org=f"orgA{run}",
        workspace=f"w2{run}",
        user="alice",
        session="s1",
        visibility=Visibility.PRIVATE,
    )

    # the three namespaces must resolve to three DISTINCT physical graph names.
    name_a1 = ltm.graph_name_for(ns_org_a_w1)
    name_b1 = ltm.graph_name_for(ns_org_b_w1)
    name_a2 = ltm.graph_name_for(ns_org_a_w2)
    assert len({name_a1, name_b1, name_a2}) == 3, (name_a1, name_b1, name_a2)
    # D-8: org/workspace are now hashed (tenant_partition_digest), so the raw org slug no
    # longer appears in the name — assert on the digest instead of a `startswith` on raw text.
    assert name_a1 == f"mu_g__{tenant_partition_digest(ns_org_a_w1)}__u_alice"
    assert name_b1 == f"mu_g__{tenant_partition_digest(ns_org_b_w1)}__u_alice"

    a1 = make_item(
        ns_org_a_w1, "Ada uses Postgres", subject="Ada", predicate="uses", obj="Postgres"
    )
    a1.valid_at = datetime.now(UTC)
    await ltm.upsert_fact(a1)

    count_a1 = await _node_count(falkor_db, name_a1)
    count_b1 = await _node_count(falkor_db, name_b1)
    count_a2 = await _node_count(falkor_db, name_a2)
    # PROOF: print the real graph names + node counts — distinct partitions, not one shared
    # graph filtered by property (run with `pytest -s` to see this).
    print(f"\n[multi-org isolation] org=A workspace=w1 -> graph={name_a1!r} nodes={count_a1}")  # noqa: T201
    print(f"[multi-org isolation] org=B workspace=w1 -> graph={name_b1!r} nodes={count_b1}")  # noqa: T201
    print(f"[multi-org isolation] org=A workspace=w2 -> graph={name_a2!r} nodes={count_a2}")  # noqa: T201

    assert count_a1 == 1  # the write physically landed here ...
    assert count_b1 == 0  # ... NOT in org B's same-named workspace
    assert count_a2 == 0  # ... NOR in org A's other workspace

    # cross-namespace recall (the adapter's own read path) returns nothing in every direction.
    assert await ltm.graph_recall(ns_org_b_w1, subject="Ada", limit=10) == []
    assert await ltm.graph_recall(ns_org_a_w2, subject="Ada", limit=10) == []
    # ... but the origin namespace still recalls its own fact.
    own = await ltm.graph_recall(ns_org_a_w1, subject="Ada", limit=10)
    assert [h.item.id for h in own] == [a1.id]


async def test_graph_recall_federates_across_the_users_sessions_by_default(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """REGRESSION — the split-brained recall fabric.

    ``QdrantMtmAdapter.recall`` already federated across a user's sessions whenever
    ``session_scope`` was left at its ``None`` default (ADR 0030 "keep-and-scope"), but
    ``graph_recall`` had NO such parameter and filtered ``m.namespace = $ns`` on the FULL,
    session-included prefix. So the durable, user-scoped LTM tier — the one that is supposed to BE
    the long-term memory — was the single arm that never federated: a fact distilled while the user
    was in session A was invisible from session B, even though ``graph_name_for`` already put both
    sessions' facts in the SAME physical partition.

    Live consequence: an auto-captured turn (default importance -> STM-only, never promoted to MTM)
    reached LTM only through ``consolidate``, and LTM then hid it from every other session — so a
    brand-new agent session recalled nothing at all.
    """
    ns_a = make_ns(user="u1", session="sessionA")
    ns_b = make_ns(user="u1", session="sessionB")
    ns_other = make_ns(user="intruder", session="sessionA")

    await ltm.upsert_fact(
        make_item(ns_a, "Ada lives in Paris", subject="Ada", predicate="lives_in", obj="Paris")
    )

    from_a = {h.item.object for h in await ltm.graph_recall(ns_a, subject="Ada", limit=10)}
    assert from_a == {"Paris"}, "same-session recall regressed"

    from_b = {h.item.object for h in await ltm.graph_recall(ns_b, subject="Ada", limit=10)}
    assert from_b == {"Paris"}, (
        "a DIFFERENT session of the SAME user could not see the fact — the LTM arm is still "
        "session-locked while the MTM arm federates"
    )

    # Federation is per-USER: it must never widen into a cross-user leak.
    from_other = await ltm.graph_recall(ns_other, subject="Ada", limit=10)
    assert from_other == [], "cross-USER leak — federation widened past the user boundary"


async def test_graph_recall_explicit_session_scope_still_narrows_to_one_session(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The federate-live default must remain OPT-OUTABLE: passing a concrete ``session_scope``
    narrows back to exactly one of the user's sessions (the pre-ADR-0030 behaviour), mirroring
    ``MtmTierRepository.recall``'s identical contract."""
    ns_a = make_ns(user="u1", session="sessionA")
    ns_b = make_ns(user="u1", session="sessionB")

    await ltm.upsert_fact(
        make_item(ns_a, "Ada lives in Paris", subject="Ada", predicate="lives_in", obj="Paris")
    )

    narrowed = await ltm.graph_recall(ns_b, subject="Ada", limit=10, session_scope="sessionB")
    assert narrowed == [], "explicit session_scope did not narrow — the opt-out is broken"

    widened = await ltm.graph_recall(ns_b, subject="Ada", limit=10, session_scope="sessionA")
    assert {h.item.object for h in widened} == {
        "Paris"
    }, "session_scope must be able to target ANY of the user's sessions, not only the caller's"


# =================================================================================================
# AD-110 — the two verbs the lazy connect made necessary, against the REAL store
# =================================================================================================
async def test_ping_connects_lazily_and_reports_the_real_store(
    falkor_db: FalkorDB,
) -> None:
    """``ping()`` is the tier's liveness verb, and it must work on an adapter that has NEVER been
    used — that is the whole point of it existing (AD-110).

    Built through the REGISTRY, not by handing in ``falkor_db``: the registry path is the one that
    defers the connect, so an adapter constructed with an already-live client would prove nothing.

    It also must not MATERIALIZE a graph. ``select_graph(name).query("RETURN 1")`` — the obvious
    liveness verb — creates a persistent graph key on the multi-tenant store even for a read-only
    Cypher, outside every ``mu_g__…`` namespace (CLAUDE.md rule 4). The graph list is compared
    before and after for exactly that.
    """
    settings = get_settings()
    adapter: FalkorLtmAdapter = STORE_REGISTRY.build(
        "graph",
        "falkordb",
        host=settings.storage.graph.host,
        port=settings.storage.graph.port,
    )
    assert adapter._db is None, "the registry connected eagerly; AD-110 has regressed"

    def _names(raw: list[object]) -> set[str]:
        return {g.decode() if isinstance(g, bytes) else str(g) for g in raw}

    before = _names(await falkor_db.list_graphs())
    await adapter.ping()
    after = _names(await falkor_db.list_graphs())

    assert adapter._db is not None, "ping() did not connect — it cannot have reached the store"
    assert after == before, (
        f"ping() materialized graph(s) {sorted(after - before)} on the shared store — a liveness "
        "probe must not write, least of all outside every tenant namespace."
    )

    # aclose() on a used adapter really closes; on an unused one it is a no-op, not an error.
    await adapter.aclose()
    fresh: FalkorLtmAdapter = STORE_REGISTRY.build(
        "graph",
        "falkordb",
        host=settings.storage.graph.host,
        port=settings.storage.graph.port,
    )
    await fresh.aclose()  # never connected — must not raise
