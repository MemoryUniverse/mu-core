"""AG-2 — ``DeterministicPromoteStage`` idempotency key is namespace-scoped (A3; design §13 item
6d; plan §2 A3).

Pre-fix bug (``ingest.py`` ~:202-214): the promote stage's ledger key was the BARE
``content_hash`` — no η component. Two DIFFERENT users adding IDENTICAL content+triple into the
SAME instance therefore hashed to the SAME ledger key: the first user's promote ran for real and
marked the key completed; the second user's promote hit that same key, returned
``SKIPPED("ledger-hit")``, and replayed the FIRST user's recorded ``MemoryPromoted`` event —
which makes ``IngestResult.promoted`` misleadingly read ``True`` for the second user even though
NO MTM upsert ever happened under their own namespace (``_remember`` derives ``promoted`` purely
from the emitted event types, ``services/ingest.py:178``). The real content silently never lands
in the second user's MTM partition.

This test proves the fix at the source: it drives ``mu_engine.services.ingest.IngestService``
directly (bypassing ``LocalMemory.add()``, which never threads a subject/predicate/object triple)
so the two activities are BYTE-IDENTICAL in content+triple, differing ONLY in ``namespace.user``.
It asserts the real ``MtmTierRepository`` — REAL qdrant, zero mocks — under EACH user's own
namespace, not just the (misleading) ``IngestResult.promoted`` flag: this is what would fail
against the pre-fix bare-``content_hash`` key (bo's real MTM point never appears) and pass once
the key is namespace-scoped.

Same session id is used for both users deliberately (only ``namespace.user`` varies): this
isolates the fix under test to the promote stage's η-scoping rather than incidentally relying on
``WriteStmStage``'s own ``activity_id`` (which already discriminates by ``namespace.session`` +
``session_offset`` — ingest.py:94 — and is NOT the bug this test targets).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.storage.domain.memory import FactObjectKind, MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import Scored
from mu_local import LocalMemory

pytestmark = pytest.mark.integration

# SAME session for both users on purpose (see module docstring) — only η.user differs.
_SESSION = "shared-session"
_CONTENT = "The launch date is 2027-01-01"
_SUBJECT = "ProjectX"
_PREDICATE = "launch_date"
_OBJECT = "2027-01-01"


@pytest_asyncio.fixture
async def mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """A LocalMemory bound to a unique workspace/org so its η partition is isolated from other
    concurrent test runs; teardown drops every qdrant collection / falkordb graph / redis key the
    run created (identical discipline to ``test_local_roundtrip_int.py``)."""
    memory = LocalMemory(workspace=f"wsns{uid}", namespace=f"orgns{uid}", settings=settings)
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


def _activity(ns: Namespace, *, session_offset: str) -> IngestActivity:
    """IDENTICAL content + triple for every caller — only ``ns`` (hence ``ns.user``) varies. A
    fresh, per-user ``session_offset`` so ``WriteStmStage``'s OWN ``activity_id`` never collides
    (that stage already discriminates by namespace.session — this test targets the OTHER stage)."""
    return IngestActivity(
        namespace=ns,
        host="mu-local",
        session_offset=session_offset,
        kind="user_message",
        text=_CONTENT,
        promote=True,  # explicit — bypasses the importance gate, deterministic promotion (AG-2).
        subject=_SUBJECT,
        predicate=_PREDICATE,
        object=_OBJECT,
        object_kind=FactObjectKind.LITERAL,
    )


async def _poll_semantic(
    mem: LocalMemory, ns: Namespace, query_vector: list[float]
) -> list[Scored[MemoryItem]]:
    """Poll a namespace-scoped MTM semantic search until it returns hits (qdrant applies upserts
    asynchronously — real eventual-consistency, not a masked bug; bounded so a genuine miss still
    fails, mirroring ``test_local_roundtrip_int.py``'s ``_eventually``)."""
    hits = await mem._container.mtm.semantic(ns, query_vector, limit=5)
    for _ in range(40):  # ~8s ceiling
        if hits:
            return hits
        await asyncio.sleep(0.2)
        hits = await mem._container.mtm.semantic(ns, query_vector, limit=5)
    return hits


async def test_two_users_identical_content_both_promote_to_their_own_mtm_partition(
    mem: LocalMemory,
) -> None:
    ns_ada = mem._ns("ada", _SESSION)
    ns_bo = mem._ns("bo", _SESSION)

    result_ada = await mem._container.ingest.remember(_activity(ns_ada, session_offset="off-ada"))
    result_bo = await mem._container.ingest.remember(_activity(ns_bo, session_offset="off-bo"))

    # Sanity on the test's own premise: this really IS a content-hash-collision scenario.
    assert result_ada.content_hash == result_bo.content_hash, (
        "test setup invalid — identical content+triple must hash identically"
    )
    assert result_ada.memory_id != result_bo.memory_id, "each user must mint their OWN STM id"

    # The (misleading, pre-fix-passable) receipt flag: both report promoted=True even pre-fix,
    # because a ledger-hit replays the FIRST user's recorded MemoryPromoted event as if it were
    # the second user's own — this is exactly why the assertions below check the REAL store.
    assert result_ada.promoted, "ada's promote did not fire"
    assert result_bo.promoted, "bo's promote did not fire"

    atomic_text = f"{_SUBJECT} {_PREDICATE} {_OBJECT}"
    query_vector = (await mem._container.embedder.embed([atomic_text]))[0]

    hits_ada = await _poll_semantic(mem, ns_ada, list(query_vector))
    hits_bo = await _poll_semantic(mem, ns_bo, list(query_vector))

    # THE durable proof (AG-2): each user's REAL MTM partition must actually hold their own point.
    # Pre-fix (bare content_hash key), bo's promote SKIPPED on ada's ledger completion and never
    # upserted — hits_bo stays empty for the full ~8s poll and this assertion fails.
    assert hits_ada, "ada's MTM point never upserted under her own namespace"
    assert hits_bo, (
        "bo's MTM point never upserted under his own namespace — the cross-user "
        "content_hash-only ledger collision (pre-fix bug) is NOT closed"
    )

    ada_ids = {scored.item.id for scored in hits_ada}
    bo_ids = {scored.item.id for scored in hits_bo}
    assert result_ada.memory_id in ada_ids
    assert result_bo.memory_id in bo_ids

    # 2 DISTINCT MTM points, not one shared/skipped write.
    assert ada_ids.isdisjoint(bo_ids), "expected 2 distinct MTM points — got an id collision"

    # Bonus tenancy check: each user's namespace-scoped semantic search must not surface the
    # OTHER user's point either (the physical to_prefix() partition, independent of the ledger).
    assert result_bo.memory_id not in ada_ids
    assert result_ada.memory_id not in bo_ids
