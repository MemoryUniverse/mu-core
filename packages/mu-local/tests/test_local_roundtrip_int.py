"""LocalMemory REAL end-to-end round-trip — mu-dev-cache + mu-dev-qdrant + mu-dev-falkordb, REAL
MiniLM embedder, ZERO mocks (DEV-STANDARDS: non-negotiable; the task acceptance for this slice).

``LocalMemory().add(...)`` -> promote -> ``recall(...)`` round-trips real data across
redis (STM) + qdrant (MTM) + falkordb (LTM) on the live containers:

  * ``add`` writes STM (redis) and deterministically promotes to MTM (qdrant) — ``promoted`` +
    ``"mtm"`` in ``tiers_written`` prove the qdrant WRITE;
  * a tier=MTM recall returns qdrant hits — proves the qdrant READ;
  * ``consolidate`` reads the STM window (redis) and writes bi-temporal facts to LTM (falkordb);
  * a tier=LTM recall returns falkordb graph facts — proves the falkordb READ;
  * a full federated recall fuses all three channels, protects the STM recency floor (redis READ),
    and round-trips the original content back to the caller;
  * ``ask`` refuses loudly in heuristic mode (no silent empty synthesis — spec §7, T7).

If a container is down the test RAISES (BLOCKED, never faked).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.storage.domain.memory import MemoryTier
from mu_local import LlmNotConfiguredError, LocalMemory, MemoryListView
from mu_local.config import BackendChoice, StorageSettings
from mu_local.errors import BackendUnavailableError

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"


@pytest_asyncio.fixture
async def mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """A LocalMemory bound to a unique workspace/org so its η partition is isolated; teardown drops
    every qdrant collection / falkordb graph / redis key the run created."""
    memory = LocalMemory(workspace=f"ws{uid}", namespace=f"org{uid}", settings=settings)
    try:
        yield memory
    finally:
        await _teardown(settings, uid)
        await memory.aclose()


async def _teardown(settings: Settings, uid: str) -> None:
    qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        for coll in (await qdrant.get_collections()).collections:
            if uid in coll.name:
                with contextlib.suppress(Exception):
                    await qdrant.delete_collection(coll.name)
    finally:
        await qdrant.close()

    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        for g in await db.list_graphs():
            name = g.decode() if isinstance(g, bytes) else g
            if uid in name:
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


async def _eventually(read: Callable[[], Awaitable[MemoryListView]]) -> MemoryListView:
    """Poll a recall until it returns hits (qdrant applies upserts asynchronously — the store's
    real eventual-consistency model, NOT a masked bug; bounded so a genuine empty still fails)."""
    last = await read()
    for _ in range(40):  # ~8s ceiling
        if last.items:
            return last
        await asyncio.sleep(0.2)
        last = await read()
    return last


async def test_add_promote_recall_round_trips_across_all_three_stores(mem: LocalMemory) -> None:
    # (1) WRITE — add durably lands in STM (redis) and promotes to MTM (qdrant).
    r1 = await mem.add("Ada lives in Paris", user=_USER, session=_SESSION)
    r2 = await mem.add("Ada works at Acme", user=_USER, session=_SESSION)
    assert r1.promoted and r2.promoted, "STM->MTM deterministic promotion did not fire"
    assert "mtm" in r1.tiers_written, "qdrant MTM write not recorded in the receipt"
    assert r1.content_hash and r1.memory_id

    # (2) qdrant READ — a tier=MTM recall returns the dense hits from the vector store.
    mtm_hits = await _eventually(
        lambda: mem.recall(
            "Where does Ada live?", user=_USER, session=_SESSION, tier=MemoryTier.MTM
        )
    )
    assert mtm_hits.items, "qdrant MTM channel returned nothing"
    assert all(it.channel == "mtm" for it in mtm_hits.items)
    assert any("Paris" in it.content for it in mtm_hits.items), "MTM content did not round-trip"

    # (3) falkordb WRITE — consolidate reads the STM window and distills bi-temporal facts to LTM.
    report = await mem.consolidate(user=_USER, session=_SESSION)
    assert report.facts_extracted >= 2, "heuristic extractor found no SPO facts to consolidate"
    assert report.added >= 2, "no facts landed in the LTM graph"

    # (4) falkordb READ — a tier=LTM recall returns the graph facts.
    ltm_hits = await _eventually(
        lambda: mem.recall(
            "Where does Ada live?", user=_USER, session=_SESSION, tier=MemoryTier.LTM
        )
    )
    assert ltm_hits.items, "falkordb LTM channel returned nothing after consolidate"
    assert all(it.channel == "ltm" for it in ltm_hits.items)

    # (5) redis READ + full federation — all three channels fuse, the STM recency floor is
    #     protected, and the original content round-trips back to the caller.
    fused = await mem.recall("What do we know about Ada?", user=_USER, session=_SESSION, limit=10)
    assert fused.degraded is None, "a clean local recall must not degrade (no shared plane)"
    contents = " ".join(it.content for it in fused.items)
    assert "Paris" in contents or "Acme" in contents, "added content missing from federated recall"
    assert any(it.is_floor for it in fused.items), "STM recency-floor (redis) member not protected"

    # (6) search is a behaviour-identical alias for recall (spec verb-alias policy).
    via_alias = await mem.search("What do we know about Ada?", user=_USER, session=_SESSION)
    assert set(via_alias.memory_ids) == set(fused.memory_ids)

    # (7) tenancy isolation — a DIFFERENT user's partition sees none of u1's data.
    other = await mem.recall("What do we know about Ada?", user="u2", session=_SESSION)
    assert not set(other.memory_ids) & set(fused.memory_ids), "cross-user tenancy breach"

    # (8) heuristic mode refuses synthesis loudly — never an empty/degraded answer (spec §7, T7).
    with pytest.raises(LlmNotConfiguredError):
        await mem.ask("Where does Ada live?", user=_USER, session=_SESSION)


def test_unsupported_backend_fails_loud_not_silent() -> None:
    """Selecting a graph backend the STORE_REGISTRY does not ship (e.g. Kùzu — only ``falkordb``
    is registered for the mandatory graph role, storage-rework review FIX-1) raises a NAMED
    ``BackendUnavailableError`` at the composition root — never a silent fallback (spec §7).

    (``vector=faiss`` no longer belongs here: the registry ships it and ``LocalContainer`` now
    routes every kv/vector/graph role through ``STORE_REGISTRY.build`` — FAISS is a supported,
    PRIVATE-plane-only vector backend, not an unsupported one.)"""
    with pytest.raises(BackendUnavailableError):
        LocalMemory(StorageSettings(graph=BackendChoice(backend="kuzu")))
