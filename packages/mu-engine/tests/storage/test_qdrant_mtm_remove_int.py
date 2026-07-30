"""``QdrantMtmAdapter.remove`` — REAL ``mu-dev-qdrant``, ZERO mocks (CF-2, MLM-STAGE2-CARRYOVER.md;
DEV-STANDARDS non-negotiable: real deps or BLOCKED, never mock).

Covers CF-2's acceptance item: a point genuinely deleted from ``mu-dev-qdrant`` via
``MtmTierRepository.remove`` (real ``AsyncQdrantClient.delete``), and confirms ``remove`` is a
DIFFERENT operation from ``invalidate`` (loser-supersession: payload overwrite, point stays) —
never a substitute for it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient

from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id

from .conftest import VECTOR_DIM

pytestmark = pytest.mark.integration


@pytest.fixture
async def mtm(qdrant_client: AsyncQdrantClient) -> QdrantMtmAdapter:
    return QdrantMtmAdapter(qdrant_client, dim=VECTOR_DIM)


async def test_remove_deletes_real_point_from_qdrant(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The point genuinely lands, then genuinely disappears — real ``AsyncQdrantClient.delete``,
    not a payload overwrite (contrast with ``invalidate`` below)."""
    ns = make_ns()
    item = make_item(ns, "point that will really be deleted")
    await mtm.upsert(item)

    name = collection_name(ns, VECTOR_DIM)
    try:
        before = await qdrant_client.retrieve(collection_name=name, ids=[point_id(item.id)])
        assert len(before) == 1  # sanity: really landed

        await mtm.remove(ns, item.id)

        after = await qdrant_client.retrieve(collection_name=name, ids=[point_id(item.id)])
        assert after == []  # genuinely gone, not merely marked inactive
    finally:
        await qdrant_client.delete_collection(name)


async def test_remove_on_missing_collection_is_a_noop(
    mtm: QdrantMtmAdapter,
    make_ns: Callable[..., Namespace],
) -> None:
    """A namespace whose collection was never created (nothing was ever upserted): ``remove``
    returns cleanly rather than raising — mirrors ``invalidate``'s own missing-collection guard."""
    ns = make_ns()
    await mtm.remove(ns, "never-existed")  # must not raise


async def test_remove_is_a_different_operation_from_invalidate(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """``invalidate`` supersedes a loser IN PLACE (payload rewritten to ``state=superseded``,
    point stays retrievable); ``remove`` genuinely deletes. The two must never be interchangeable
    (CF-2's explicit constraint: the supersession path stays untouched)."""
    ns = make_ns()
    loser = make_item(ns, "loser fact, will be superseded not deleted")
    winner = make_item(ns, "winner fact")
    await mtm.upsert(loser)
    await mtm.upsert(winner)

    name = collection_name(ns, VECTOR_DIM)
    try:
        await mtm.invalidate(
            ns, loser.id, winner.id, at=datetime.now(UTC), reason="test-supersede"
        )
        # invalidate: the loser point STILL EXISTS, payload flipped to superseded.
        superseded = await qdrant_client.retrieve(
            collection_name=name, ids=[point_id(loser.id)], with_payload=True
        )
        assert len(superseded) == 1
        assert superseded[0].payload is not None
        assert superseded[0].payload["state"] == MemoryState.SUPERSEDED.value
        assert superseded[0].payload["superseded_by"] == winner.id

        # remove: the winner point is genuinely deleted (contrast operation on a fresh id).
        await mtm.remove(ns, winner.id)
        gone = await qdrant_client.retrieve(collection_name=name, ids=[point_id(winner.id)])
        assert gone == []
    finally:
        await qdrant_client.delete_collection(name)
