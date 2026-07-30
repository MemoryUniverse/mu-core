"""Vector/MTM adapter — REAL mu-dev-qdrant, ZERO mocks.

Covers: store+retrieve, namespace filter, idempotent (id-stable) upsert, the
``state='active'`` cross-store supersede drop (spec §8.7), and the SHARED ``authorized_ids``
Model-A completeness hazard (spec §8.5)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name

from .conftest import VECTOR_DIM

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def mtm(
    qdrant_client: AsyncQdrantClient, make_ns: Callable[..., Namespace]
) -> AsyncIterator[QdrantMtmAdapter]:
    adapter = QdrantMtmAdapter(qdrant_client, dim=VECTOR_DIM)
    yield adapter
    # teardown: drop any collections this test created (unique workspace per test).
    existing = {c.name for c in (await qdrant_client.get_collections()).collections}
    for name in list(existing):
        if name.startswith("mu_mtm__ws"):
            with contextlib.suppress(Exception):  # best-effort teardown only
                await qdrant_client.delete_collection(name)


async def test_upsert_and_semantic_recall(
    mtm: QdrantMtmAdapter,
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
    mtm: QdrantMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    item = make_item(ns, "idempotent")
    await mtm.upsert(item)
    await mtm.upsert(item)  # second write must NOT fork (uuid5 point id, spec §5 contract 2)
    count = await mtm._qdrant.count(collection_name(ns, VECTOR_DIM))
    assert count.count == 1


async def test_namespace_filter(
    mtm: QdrantMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")  # same collection, different η -> mandatory namespace filter
    a = make_item(ns_a, "only in A")
    await mtm.upsert(a)
    # searching under B must not return A even though they share a collection (spec §3.2).
    # NOTE (S1-04 / BQ3 / ADR 0030): the bare default (session_scope omitted) now federates
    # every one of the user's sessions for PRIVATE-own recall by design — that federation
    # behavior is exhaustively covered by test_qdrant_mtm_session_scope_int.py. This test's
    # actual intent (narrow to exactly one session) still exists and is still exercised by
    # passing session_scope explicitly, which is unchanged from "today"'s exact-match path.
    hits_b = await mtm.semantic(ns_b, a.embedding or [], limit=10, session_scope=ns_b.session)
    assert hits_b == []


async def test_state_active_supersede_drop(
    mtm: QdrantMtmAdapter,
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
    assert loser.id not in ids  # dropped by state='active' filter (B1, spec §8.7)
    assert winner.id in ids


async def test_shared_authorized_ids_completeness(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    # SHARED plane: caller authorized for only a sparse subset must still see it (M1 hazard).
    adapter = QdrantMtmAdapter(qdrant_client, dim=VECTOR_DIM)
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
    await qdrant_client.delete_collection(collection_name(ns, VECTOR_DIM))
