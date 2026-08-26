"""pgvector/MTM adapter — REAL mu-dev-pgvector (``pgvector/pgvector:pg16``), ZERO mocks.

Mirrors ``test_vector_qdrant_int.py``'s conformance body 1:1 (spec §8: "run the SAME test body
against every registered backend for a role") so the pgvector alt backend proves the same
correctness invariants Qdrant does: store+retrieve, namespace filter, idempotent (id-stable)
upsert, the ``state='active'`` cross-store supersede drop (spec §8.6), and the SHARED
``authorized_ids`` Model-A completeness hazard (spec §8.3) — here pushed down via a real SQL
``&&`` array-overlap predicate rather than Qdrant's ``MatchAny``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import asyncpg
import pytest
import pytest_asyncio

from mu_contracts.config import Settings
from mu_engine.storage.adapters.pgvector_mtm import PgVectorMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.pgvector_mapper import pgvector_table_name
from mu_engine.storage.mappers.qdrant_mapper import point_id

from .conftest import VECTOR_DIM

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def mtm(settings: Settings) -> AsyncIterator[PgVectorMtmAdapter]:
    adapter = PgVectorMtmAdapter(dsn=settings.storage.pgvector.dsn, dim=VECTOR_DIM)
    yield adapter
    # teardown: drop any tables this test created (unique workspace per test).
    pool = await adapter._ensure_pool()
    async with pool.acquire() as conn:
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE tablename LIKE 'mu_mtm_pgv__%'"
        )
        for row in tables:
            await conn.execute(f'DROP TABLE IF EXISTS "{row["tablename"]}"')
    await adapter.close()


async def test_upsert_and_semantic_recall(
    mtm: PgVectorMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    item = make_item(ns, "vector store fact")
    await mtm.upsert(item)
    hits = await mtm.semantic(ns, item.embedding or [], limit=5)
    assert [h.item.id for h in hits] == [item.id]
    assert hits[0].item.content == item.content  # payload round-trips the record


async def test_idempotent_upsert_is_id_stable(
    mtm: PgVectorMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    item = make_item(ns, "idempotent")
    await mtm.upsert(item)
    await mtm.upsert(item)  # second write must NOT fork (point_id PRIMARY KEY, spec §6 rule 5)
    table = pgvector_table_name(ns, VECTOR_DIM)
    pool = await mtm._ensure_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval(f"SELECT count(*) FROM {table}")  # noqa: S608
    assert count == 1


async def test_namespace_filter(
    mtm: PgVectorMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")  # same table, different η -> mandatory namespace filter
    a = make_item(ns_a, "only in A")
    await mtm.upsert(a)
    hits_b = await mtm.semantic(ns_b, a.embedding or [], limit=10)
    assert hits_b == []


async def test_state_active_supersede_drop(
    mtm: PgVectorMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    loser = make_item(ns, "old truth")
    winner = make_item(ns, "new truth")
    await mtm.upsert(loser)
    await mtm.upsert(winner)
    await mtm.invalidate(ns, loser.id, winner.id, at=datetime.now(UTC), reason="test-supersede")
    hits = await mtm.semantic(ns, loser.embedding or [], limit=10)
    ids = {h.item.id for h in hits}
    assert loser.id not in ids  # dropped by state='active' filter (B1, spec §8.6)
    assert winner.id in ids


async def test_shared_authorized_ids_completeness(
    settings: Settings,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    # SHARED plane: caller authorized for only a sparse subset must still see it (M1 hazard),
    # here enforced by a real SQL `authorized_ids && $n` array-overlap predicate pre-truncation.
    adapter = PgVectorMtmAdapter(dsn=settings.storage.pgvector.dsn, dim=VECTOR_DIM)
    ns = make_ns(visibility=Visibility.SHARED)
    mine = make_item(ns, "authorized fact", authorized_ids=["p_alice", "p_bob"])
    not_mine = make_item(ns, "someone else fact", authorized_ids=["p_carol"])
    await adapter.upsert(mine)
    await adapter.upsert(not_mine)
    caller = frozenset({"p_alice"})
    hits = await adapter.semantic(ns, mine.embedding or [], limit=10, caller_identity_set=caller)
    ids = {h.item.id for h in hits}
    assert mine.id in ids  # authorized-and-relevant surfaces
    assert not_mine.id not in ids  # unauthorized filtered pre-truncation (Model A)
    table = pgvector_table_name(ns, VECTOR_DIM)
    pool = await adapter._ensure_pool()
    async with pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {table}")
    await adapter.close()


async def test_extension_and_table_are_provisioned(settings: Settings) -> None:
    """Fail-loud probe: the `vector` extension must exist on the real mu-dev-pgvector container
    (D3-adjacent "BLOCKED, never mock" — this is the honesty check the data reviewer wants)."""
    conn = await asyncpg.connect(dsn=settings.storage.pgvector.dsn)
    try:
        version = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
        assert version is not None, "the 'vector' extension is not installed on mu-dev-pgvector"
    finally:
        await conn.close()


async def test_invalidate_refuses_another_namespaces_memory(
    mtm: PgVectorMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """C3 on the pgvector backend: the by-id supersede WRITE must carry the tenancy predicate.

    ``pgvector_table_name(ns, dim)`` hashes ``org``+``workspace`` (jointly, since the fix for the
    cross-org collision — see that function's docstring) — still no ``user`` — so
    same-org-same-workspace-different-user still land in the SAME table. ``point_id`` is
    ``uuid5(NAMESPACE_URL, memory_id)``, unsalted by namespace, so a bare
    ``loser_id`` from another principal names a real row of the same table. The adapter must
    apply the exact-equality scope itself (``storage-pluggable-spec.md:483-485``), not rely on a
    caller in another package pre-gating on a read — this test calls ``invalidate`` DIRECTLY.
    """
    victim_ns = make_ns(user="u_victim")
    caller_ns = make_ns(user="u_caller")
    # PRECONDITION: without a shared table this proves nothing about namespace scoping.
    assert pgvector_table_name(victim_ns, VECTOR_DIM) == pgvector_table_name(
        caller_ns, VECTOR_DIM
    ), "precondition failed: the two namespaces are in different pgvector tables"
    assert victim_ns.to_prefix() != caller_ns.to_prefix()
    victim = make_item(victim_ns, "the victim's current truth")
    winner = make_item(caller_ns, "the caller's replacement truth")
    await mtm.upsert(victim)
    await mtm.upsert(winner)

    await mtm.invalidate(
        caller_ns, victim.id, winner.id, at=datetime.now(UTC), reason="cross-tenant-supersede"
    )

    # raw row, not `semantic` — the state column IS the store state.
    pool = await mtm._ensure_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f'SELECT state, payload FROM "{pgvector_table_name(victim_ns, VECTOR_DIM)}" '  # noqa: S608
            "WHERE point_id = $1",
            point_id(victim.id),
        )
    assert row is not None, "a foreign invalidate deleted the victim's row"
    assert row["state"] == MemoryState.ACTIVE.value, (
        "a foreign namespace superseded the victim's memory — it is now dropped from its "
        "owner's active recall"
    )
    assert "superseded_by" not in dict(
        row["payload"]
    ), "a foreign invalidate wrote a supersession edge onto the victim's row"


async def test_invalidate_in_its_own_namespace_still_supersedes(
    mtm: PgVectorMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The control for the test above: without it, an ``invalidate`` hard-wired to a no-op would
    pass the refusal test while breaking every real supersede."""
    ns = make_ns()
    loser = make_item(ns, "old truth")
    winner = make_item(ns, "new truth")
    await mtm.upsert(loser)
    await mtm.upsert(winner)
    at = datetime.now(UTC)

    await mtm.invalidate(ns, loser.id, winner.id, at=at, reason="legitimate-supersede")

    pool = await mtm._ensure_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            f'SELECT state, payload FROM "{pgvector_table_name(ns, VECTOR_DIM)}" '  # noqa: S608
            "WHERE point_id = $1",
            point_id(loser.id),
        )
    assert row is not None, "invalidate must overwrite the row, never delete it"
    assert row["state"] == MemoryState.SUPERSEDED.value
    payload = dict(row["payload"])
    assert payload["superseded_by"] == winner.id
    assert payload["supersede_reason"] == "legitimate-supersede"
    assert payload["invalid_at"] == at.isoformat()
