"""AD-266 D1/D2/D3 VERIFIED BY READING THE STORES, not by asserting on a returned object.

The AD-266 lane closed three findings — ``relevance_score`` and ``last_seen`` written back on a
real recall (D1/D2), and ``mention_count`` bumped on a repeat so ``IngestSettings.mention_promote``
can actually fire (D3) — and proved them at the adapter tier
(``test_reinforce_many_batched_int.py``, ``test_kv_redis_int.py``) and at the mapper tier
(``test_ad281a_read_path_field_projection.py``). What no file asserted is the claim a user cares
about: **after a real ``LocalMemory.add`` + a real ``LocalMemory.recall``, the bytes sitting in
Valkey and in Qdrant carry real values in those fields.** Every layer in between — the ranker's
three reinforce legs, the batched-vs-single preference, the namespace scoping, the fire-and-forget
scheduling — can drop the write-back with every adapter test still green, because no adapter test
goes through the ranker and no ranker test goes to a store.

So this file constructs NOTHING after the ingest. It drives the public verbs, then opens its own
``Redis`` / ``AsyncQdrantClient`` connection and reads the RAW row — ``json.loads`` on the STM blob
(``RedisMapper.to_store`` stores ``item.model_dump_json()`` verbatim) and the Qdrant point payload
— so a field that is right only in the in-memory copy the verb returned cannot pass.

MUTATION CHECK (each run, red, restored — see ADR 0077 §2 for the captured output):
  * ``ranker.py::_reinforce_stm_hits``: drop ``relevance_score=`` from the ``reinforce_many`` call
    -> ``test_a_real_recall_writes_relevance_score...`` goes red on the STM row (the field stays
    ``None`` in Valkey) while every adapter test stays green.
  * ``redis_stm.py:186``: revert ``_bump_if_duplicate``'s ``mention_count + 1`` to ``+ 0``
    -> ``test_saying_the_same_thing_twice...`` goes red on BOTH assertions (the raw row's count and
    the second add's ``promoted``), which is the D3 chain end to end.

REAL ``mu-dev-cache`` (Valkey) + ``mu-dev-qdrant``, REAL MiniLM embedder, ZERO mocks. Run on the
VM (root ``CLAUDE.md`` rule 13):
``infra/mu-vm/vm_test.sh mu-core
packages/mu-local/tests/test_ad266_stat_fields_on_real_stores_int.py``
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_contracts.contracts.recall import RecallResult
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id
from mu_engine.storage.mappers.redis_mapper import RedisMapper
from mu_local import LocalMemory

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"

#: Below ``IngestSettings.importance_promote`` (0.6) ON PURPOSE. The mention arm is the ONLY thing
#: that can promote at this importance, so a promotion in the second add cannot be the importance
#: arm wearing the mention arm's clothes.
_BELOW_IMPORTANCE_GATE = 0.3
#: Above it, for the write-back test, which needs the memory resident in MTM as well as STM.
_ABOVE_IMPORTANCE_GATE = 0.9


@pytest_asyncio.fixture
async def mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    memory = LocalMemory(workspace=f"ws{uid}", namespace=f"org{uid}", settings=settings)
    try:
        yield memory
    finally:
        await _teardown(settings, uid)
        await memory.aclose()


def _ns(uid: str) -> Namespace:
    """The SAME η ``LocalMemory._ns`` builds for (``_USER``, ``_SESSION``) — this is how the raw
    key is addressed without asking the object under test where it put anything."""
    return Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user=_USER,
        session=_SESSION,
        visibility=Visibility.PRIVATE,
    )


async def _teardown(settings: Settings, uid: str) -> None:
    # The MTM collection name is `mu_mtm__{digest(org, workspace)}__{visibility}__{dim}` — a
    # DIGEST, so the `uid in coll.name` match every other integration file in this repo uses
    # NEVER FIRES and every run leaks its collections (AD-295: 293 live collections took Qdrant
    # down with a RocksDB IO error mid-suite on 2026-09-25). Delete by this η's own digest prefix
    # instead; the partition is unique to this test because org/workspace carry `uid`.
    qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        prefix = collection_name(_ns(uid), 0).removesuffix("0")
        for coll in (await qdrant.get_collections()).collections:
            if coll.name.startswith(prefix) or uid in coll.name:
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


async def _raw_stm_row(settings: Settings, ns: Namespace, memory_id: str) -> dict[str, Any]:
    """The STM row as it actually sits in Valkey — no adapter, no mapper round trip."""
    redis: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    try:
        blob = await redis.get(RedisMapper.memory_key(ns, memory_id))
    finally:
        await redis.aclose()
    assert blob is not None, f"no STM row in Valkey for {memory_id}"
    row = json.loads(blob)
    assert isinstance(row, dict)
    return row


async def _raw_mtm_payload(
    settings: Settings, ns: Namespace, memory_id: str
) -> dict[str, Any] | None:
    """The MTM point payload as it actually sits in Qdrant.

    ``collection_name`` needs the embedding dimension, which no ``Settings`` field carries (it is
    the embedder's own property), so the point — whose id is a deterministic, collision-resistant
    derivation of ``memory_id`` (:func:`point_id`) — is looked up across this η's candidate
    collections instead of being told which one to open."""
    qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        prefix = collection_name(ns, 0).removesuffix("0")
        for coll in (await qdrant.get_collections()).collections:
            if not coll.name.startswith(prefix):
                continue
            points = await qdrant.retrieve(collection_name=coll.name, ids=[point_id(memory_id)])
            if points:
                payload = points[0].payload
                return dict(payload) if payload else {}
        return None
    finally:
        await qdrant.close()


async def _eventually(read: Callable[[], Awaitable[RecallResult]]) -> RecallResult:
    last = await read()
    for _ in range(40):  # ~8s ceiling — qdrant applies upserts asynchronously
        if last.items:
            return last
        await asyncio.sleep(0.2)
        last = await read()
    return last


async def _until_positive(
    read: Callable[[], Awaitable[dict[str, Any] | None]], field: str
) -> dict[str, Any]:
    """Poll a RAW row until ``field`` holds a value the write path did NOT put there.

    ``relevance_score``'s stored default is ``0.0`` (``storage/domain/memory.py:209``), not
    ``None`` — the first cut of this test asserted ``is None`` and went red on the precondition,
    which is the only reason this helper waits for a POSITIVE value rather than a non-null one. A
    non-null wait would have been satisfied instantly by the default and proved nothing.

    The write-back is deliberately off the recall
    reply path (ADR 0062/0063 — it must not add latency the caller waits on), so a read taken the
    instant ``recall`` returns can legitimately precede it. Bounded, so a write-back that never
    happens still fails."""
    row = await read()
    for _ in range(50):  # ~10s ceiling
        if row is not None and float(row.get(field) or 0.0) > 0.0:
            return row
        await asyncio.sleep(0.2)
        row = await read()
    assert row is not None, "the row itself never appeared"
    raise AssertionError(f"{field!r} never rose above its default in the stored row: {sorted(row)}")


async def test_a_real_recall_writes_relevance_score_and_last_seen_into_the_stored_rows(
    mem: LocalMemory, settings: Settings, uid: str
) -> None:
    """D1/D2 — after one real recall, the Valkey row AND the Qdrant payload carry real values."""
    before = datetime.now(UTC)
    receipt = await mem.add(
        "Ada lives in Paris", user=_USER, session=_SESSION, importance_score=_ABOVE_IMPORTANCE_GATE
    )
    assert receipt.promoted, "fixture precondition: the memory must be resident in MTM too"
    ns = _ns(uid)

    # The stored row starts with the fields at their defaults — otherwise the assertions below
    # could pass on a value the WRITE path happened to stamp, never proving the RECALL path.
    fresh = await _raw_stm_row(settings, ns, receipt.memory_id)
    assert (
        fresh["relevance_score"] == 0.0
    ), "precondition: the schema default (MemoryItem.relevance_score = 0.0) before any recall"
    assert fresh["access_count"] == 0, "precondition: unread memory"

    hits = await _eventually(
        lambda: mem.recall("Where does Ada live?", user=_USER, session=_SESSION)
    )
    assert any(
        item.memory_id == receipt.memory_id for item in hits.items
    ), "the memory was not recalled, so there was nothing to reinforce"

    stm_row = await _until_positive(
        lambda: _raw_stm_row(settings, ns, receipt.memory_id), "relevance_score"
    )
    assert isinstance(stm_row["relevance_score"], float)
    assert stm_row["relevance_score"] > 0.0, "a placed hit cannot have a zero relevance signal"
    assert stm_row["access_count"] >= 1, "the read counter did not move in Valkey"
    last_seen = datetime.fromisoformat(stm_row["last_seen"])
    assert last_seen >= before, "last_seen in Valkey is not the recall-time stamp"

    mtm_payload = await _until_positive(
        lambda: _raw_mtm_payload(settings, ns, receipt.memory_id), "relevance_score"
    )
    assert isinstance(mtm_payload["relevance_score"], float)
    assert mtm_payload["relevance_score"] > 0.0
    assert mtm_payload["access_count"] >= 1, "the read counter did not move in Qdrant"
    assert datetime.fromisoformat(str(mtm_payload["last_seen"])) >= before


async def test_saying_the_same_thing_twice_bumps_mention_count_and_fires_mention_promote(
    mem: LocalMemory, settings: Settings, uid: str
) -> None:
    """D3 — the third promotion arm, proved the only way it can be: by saying it twice.

    Both adds carry an importance BELOW ``importance_promote`` and neither asks for an explicit
    promote, so ``DeterministicPromoteStage`` has exactly one arm left to fire on."""
    ns = _ns(uid)
    first = await mem.add(
        "Ada prefers oat milk",
        user=_USER,
        session=_SESSION,
        importance_score=_BELOW_IMPORTANCE_GATE,
    )
    assert not first.promoted, (
        "precondition: a single low-importance mention must NOT promote, or the second add's "
        "promotion proves nothing"
    )
    row = await _raw_stm_row(settings, ns, first.memory_id)
    assert row["mention_count"] == 1

    second = await mem.add(
        "Ada prefers oat milk",
        user=_USER,
        session=_SESSION,
        importance_score=_BELOW_IMPORTANCE_GATE,
    )
    assert second.memory_id == first.memory_id, (
        "the repeat did not land on the resident row — STM content-hash dedup (D4) is the "
        "mechanism the mention arm rides on"
    )
    bumped = await _raw_stm_row(settings, ns, first.memory_id)
    assert bumped["mention_count"] == 2, "the repeat did not bump mention_count in Valkey"
    assert second.promoted, (
        "mention_promote (default 2) did not fire on the second mention — the third arm is "
        "unreachable again"
    )
    assert "mtm" in second.tiers_written
    payload = await _raw_mtm_payload(settings, ns, first.memory_id)
    assert payload is not None, "the mention-driven promotion never reached Qdrant"
    assert payload["mention_count"] == 2, "the promoted MTM copy carries a stale mention_count"
