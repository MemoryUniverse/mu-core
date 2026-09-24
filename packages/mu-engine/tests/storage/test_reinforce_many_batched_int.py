"""``reinforce_many`` — the BATCHED read-stat write-back AD-260/ADR 0063 put on the recall hot
path — against REAL ``mu-dev-cache`` (Valkey) and REAL ``mu-dev-qdrant``, ZERO mocks.

**Why this file exists.** ADR 0063 replaced AD-259's per-id ``reinforce`` fan-out with two new
batched verbs (``RedisStmAdapter.reinforce_many``, ``QdrantMtmAdapter.reinforce_many``) that
``ThreeChannelRecallRanker`` now PREFERS on every recall, and shipped them with **no test that
names them**: ``grep -rn reinforce_many packages/mu-engine/tests`` returned nothing at the
verify pass. The only coverage was indirect and single-id — ``test_ad259_reinforce_latency_int.py``
asserts ``access_count > 0`` on ``ids[0]`` alone, and the end-to-end walk carries ONE memory.

That leaves the two defects a batched rewrite actually makes possible invisible, and both are
mutation-proved here (each mutation was run, the named test went red, the line was restored):

1. **Partial application.** ``reinforce_many`` bumping only the first id — or dropping every id
   whose ``retrieve``/pipeline slot mis-zips — passes every pre-existing test. The forgetting
   curve would then keep its remembering half for exactly one memory per recall and silently lose
   it for the other nine, which is AD-259's original defect wearing a batch's clothes.
2. **Cross-namespace reinforcement.** The per-id ``reinforce`` is namespace-scoped in the adapter
   (``test_qdrant_mtm_write_scoping_int.py``'s whole subject, C3/CANONICAL §1 rule 5). The
   batched path re-implements that check independently — a ``retrieve`` by bare point id across a
   collection SHARED by every user of one org+workspace, filtered afterwards by
   ``item.namespace == ns``. Nothing tested that re-implementation, so deleting the comparison
   left every test green while letting one tenant's recall write to another tenant's point.

Run on the VM (root ``CLAUDE.md`` rule 13): ``infra/mu-vm/vm_test.sh mu-core
packages/mu-engine/tests/storage/test_reinforce_many_batched_int.py``.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from redis.asyncio import Redis

from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id
from mu_engine.storage.mappers.redis_mapper import RedisMapper

from .conftest import VECTOR_DIM

pytestmark = pytest.mark.integration

_AT = datetime(2026, 3, 1, tzinfo=UTC)
#: More than one, and enough that "only the first" and "only the last" are both distinguishable
#: from "all of them" — the shape a single-id assertion cannot see.
_N = 5


@pytest_asyncio.fixture
async def mtm(
    qdrant_client: AsyncQdrantClient, qdrant_teardown_collections: Callable[[], list[str]]
) -> AsyncIterator[QdrantMtmAdapter]:
    adapter = QdrantMtmAdapter(qdrant_client, dim=VECTOR_DIM)
    yield adapter
    for name in qdrant_teardown_collections():
        with contextlib.suppress(UnexpectedResponse):
            await qdrant_client.delete_collection(name)


@pytest_asyncio.fixture
async def stm(valkey_client: Redis) -> ValkeyStmAdapter:
    return ValkeyStmAdapter(valkey_client)


async def _raw(client: AsyncQdrantClient, ns: Namespace, memory_id: str) -> dict[str, Any] | None:
    """The point's payload straight out of Qdrant — no adapter, no mapper, no namespace guard.
    Same technique ``test_qdrant_mtm_write_scoping_int.py`` uses, and for the same reason: the
    adapter's own ``get`` applies a namespace refusal, which would mask a cross-namespace WRITE."""
    records = await client.retrieve(
        collection_name=collection_name(ns, VECTOR_DIM),
        ids=[point_id(memory_id)],
        with_payload=True,
    )
    return dict(records[0].payload or {}) if records else None


# -------------------------------------------------------------------------------------------
# 1. Every id, not just the first
# -------------------------------------------------------------------------------------------


async def test_stm_reinforce_many_bumps_every_id_not_just_the_first(
    stm: ValkeyStmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """MUTATION CHECK (run, red): make ``_reinforce_many_impl``'s write loop ``break`` after the
    first queued id — this test fails on item 1 with ``access_count == 0``, while
    ``test_ad259_reinforce_latency_int.py`` (which only ever reads ``ids[0]``) stays green."""
    ns = make_ns(session="reinforce-many-stm")
    items = [make_item(ns, f"batched stm row {i}") for i in range(_N)]
    for item in items:
        await stm.put(item)

    await stm.reinforce_many(ns, [i.id for i in items], at=_AT)

    for n, item in enumerate(items):
        row = await stm.get(ns, item.id)
        assert row is not None, f"item {n} vanished from Valkey"
        assert row.access_count == 1, (
            f"STM item {n} of {_N} was not reinforced (access_count={row.access_count}) — "
            f"reinforce_many applied to only part of the batch"
        )
        assert row.updated_at == _AT
        assert row.created_at == item.created_at, "reinforce_many must not move created_at"


async def test_mtm_reinforce_many_bumps_every_id_not_just_the_first(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """MUTATION CHECK (run, red): truncate ``_reinforce_many_impl``'s ``operations`` list to
    ``operations[:1]`` before ``batch_update_points`` — this test fails on item 1."""
    ns = make_ns(session="reinforce-many-mtm")
    items = [make_item(ns, f"batched mtm point {i}") for i in range(_N)]
    for item in items:
        await mtm.upsert(item)

    await mtm.reinforce_many(ns, [i.id for i in items], at=_AT)

    for n, item in enumerate(items):
        payload = await _raw(qdrant_client, ns, item.id)
        assert payload is not None, f"item {n} vanished from Qdrant"
        assert payload["access_count"] == 1, (
            f"MTM point {n} of {_N} was not reinforced (access_count={payload['access_count']}) "
            f"— reinforce_many applied to only part of the batch"
        )
        assert payload["updated_at"] == _AT.isoformat()
        assert payload["state"] == item.state.value, "a payload-only PATCH must not touch state"
        assert payload["created_at"] == item.created_at.isoformat()


# -------------------------------------------------------------------------------------------
# 2. The batched path re-implements the adapter's tenancy guard — prove the re-implementation
# -------------------------------------------------------------------------------------------


async def test_mtm_reinforce_many_refuses_another_namespaces_memory(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """CANONICAL §1 rule 5 / C3: the tenancy predicate is the ADAPTER's, applied on every write.

    Same shared-collection precondition ``test_qdrant_mtm_write_scoping_int.py`` establishes —
    same org+workspace+visibility, different user slot — so the two points really do live in one
    Qdrant collection and a bare point id really does address the victim's row. This is the only
    configuration in which the ``item.namespace == ns`` comparison inside
    ``_reinforce_many_impl`` is what refuses.

    MUTATION CHECK — and the result is the finding. Isolation here is genuinely TWO-LAYERED
    (``_scoped_point_selector``'s own docstring claims exactly that; this is the first time it
    was demonstrated rather than asserted), so **no single-line mutation turns this test red**:

    * delete ``if item.namespace == ns:`` (keep the dict assignment) -> still GREEN: the foreign
      item reaches ``current_by_point``, but the ``SetPayloadOperation``'s namespace-scoped
      ``filter`` matches zero points, so nothing is written. RUN, confirmed green.
    * replace that same operation's ``filter=_scoped_point_selector(ns, memory_id)`` with a bare
      ``points=[point_id(memory_id)]`` -> still GREEN: the read-side comparison already dropped
      the foreign item, so no operation is ever built for it. RUN, confirmed green.
    * **remove BOTH -> RED**, on exactly this test: the victim's ``access_count`` reaches 1.
      RUN, confirmed red, then restored.

    That is the correct, and reassuring, shape for a tenancy guarantee: either layer alone is
    sufficient, and this test is what notices if the second one is ever also removed.
    """
    victim_ns = make_ns(user="u_victim")
    caller_ns = make_ns(user="u_caller")
    assert collection_name(victim_ns, VECTOR_DIM) == collection_name(
        caller_ns, VECTOR_DIM
    ), "precondition failed: different collections, so this test would prove nothing"
    assert victim_ns.to_prefix() != caller_ns.to_prefix()

    victim = make_item(victim_ns, "the victim's live memory")
    await mtm.upsert(victim)
    before = await _raw(qdrant_client, victim_ns, victim.id)
    assert before is not None and before["access_count"] == 0

    # No raise: a batched best-effort stat write over a set of ids is a documented no-op for any
    # id it cannot legitimately address (`reinforce_many`'s own docstring) — the contract under
    # test is that the victim's row is UNTOUCHED, not that the call explodes.
    await mtm.reinforce_many(caller_ns, [victim.id], at=_AT)

    after = await _raw(qdrant_client, victim_ns, victim.id)
    assert after is not None, "a foreign reinforce_many destroyed the victim's point"
    assert after["access_count"] == 0, (
        "a foreign namespace reinforced the victim's memory through the BATCHED path: the "
        "per-id `reinforce` is adapter-scoped (C3) and its batched twin must be too"
    )
    assert after["updated_at"] == before["updated_at"]


# -------------------------------------------------------------------------------------------
# 3. The retention clock a demoted memory depends on survives the batched path
# -------------------------------------------------------------------------------------------


async def test_stm_reinforce_many_keeps_the_ttl_it_found(
    stm: ValkeyStmAdapter,
    valkey_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """ADR 0054/AD-250: a DEMOTED memory's STM row carries a ~30-day ``demoted_stm_ttl_s``, and
    the recall-rescue trigger reinforces exactly that row. A ``SET`` without ``KEEPTTL`` in the
    batched write pipeline destroys that clock.

    MUTATION CHECK (run, red, then restored) — and the observed failure is worse than the one
    guessed: dropping ``keepttl=True`` from ``_reinforce_many_impl``'s ``write_pipe.set`` does not
    RESET the TTL to the mapper's 1-hour capture default, it **REMOVES it entirely** (measured:
    ``ttl=-1``, Redis's "key exists, no expiry"). So the regression is not a shortened horizon but
    an unbounded one: every demoted memory a user ever recalls becomes immortal in Valkey, growing
    without limit and never demoted again, with nothing anywhere reporting it. No other test in
    the repo covers TTL preservation on the batched path.
    """
    ns = make_ns(session="reinforce-many-ttl")
    long_ttl_s = int(timedelta(days=30).total_seconds())
    items = [make_item(ns, f"demoted row {i}") for i in range(_N)]
    for item in items:
        await stm.put(item, ttl_s=long_ttl_s)

    await stm.reinforce_many(ns, [i.id for i in items], at=_AT)

    for n, item in enumerate(items):
        ttl = await valkey_client.ttl(RedisMapper.memory_key(ns, item.id))
        assert ttl > long_ttl_s - 60, (
            f"item {n}: reinforce_many reset the retention clock (ttl={ttl}s, expected "
            f"~{long_ttl_s}s) — a demoted memory would now die in the capture-buffer hour"
        )


# -------------------------------------------------------------------------------------------
# 4. Absent ids are a no-op, never a raise — the contract the ranker's best-effort leg needs
# -------------------------------------------------------------------------------------------


async def test_reinforce_many_tolerates_absent_ids_mixed_with_present_ones(
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The ranker passes EVERY returned id to BOTH tiers (ADR 0062's correction), so most ids in
    any real batch are absent from at least one store. An absent id must not raise, and — the
    part a zip/index-based batch implementation gets wrong — must not shift the reinforcement
    onto the wrong neighbour."""
    ns = make_ns(session="reinforce-many-absent")
    present = make_item(ns, "a row that really exists in both tiers")
    await stm.put(present)
    await mtm.upsert(present)
    absent_before = make_item(ns, "never written A")
    absent_after = make_item(ns, "never written B")

    ids = [absent_before.id, present.id, absent_after.id]
    await stm.reinforce_many(ns, ids, at=_AT)
    await mtm.reinforce_many(ns, ids, at=_AT)

    row = await stm.get(ns, present.id)
    assert (
        row is not None and row.access_count == 1
    ), "the one present STM row was not reinforced when absent ids surrounded it"
    payload = await _raw(qdrant_client, ns, present.id)
    assert (
        payload is not None and payload["access_count"] == 1
    ), "the one present MTM point was not reinforced when absent ids surrounded it"
    for ghost in (absent_before, absent_after):
        assert await stm.get(ns, ghost.id) is None
        assert await _raw(qdrant_client, ns, ghost.id) is None
