"""Graph/LTM adapter — REAL mu-dev-falkordb, ZERO mocks.

Covers: upsert_fact + graph_recall, namespace isolation, ``find_conflicts``, and the
bi-temporal invalidate-don't-delete guarantee (spec §8.7/§8.9): a superseded fact drops
from ``graph_recall`` / ``facts_at(now)`` but SURVIVES in ``facts_at(t_old)``."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB

from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def ltm(
    falkor_db: FalkorDB, make_ns: Callable[..., Namespace]
) -> AsyncIterator[FalkorLtmAdapter]:
    yield FalkorLtmAdapter(falkor_db)
    # teardown: delete the per-test graphs (unique workspace per test).
    for g in await falkor_db.list_graphs():
        name = g.decode() if isinstance(g, bytes) else g
        if name.startswith("mu_g__ws"):
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
