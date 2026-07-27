"""FAISS/MTM adapter — REAL in-proc faiss index, ZERO mocks.

No container, no library server — FAISS is entirely in-process (spec §1: "in-proc — real"), so
this suite is real end-to-end computation against the actual ``faiss`` C library, just without an
external service to mark ``integration``. Mirrors the conformance body used for the other MTM
backends (store+retrieve, namespace filter, idempotent id-stable upsert, ``state='active'``
supersede drop) PLUS the two FAISS-specific invariants: PRIVATE-plane recall works with
``authorized_ids=None`` (spec §6 invariant 3) and the SHARED plane is refused outright (D3),
matching ``test_registry_unit.py``'s build-time refusal at the adapter's own call boundary
(defense-in-depth, ``faiss_mtm.py`` module docstring).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mu_engine.storage.adapters.faiss_mtm import FaissMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.errors import VectorNotFilterableError

from .conftest import VECTOR_DIM

pytestmark = pytest.mark.unit


@pytest.fixture
def mtm(tmp_path: Path) -> FaissMtmAdapter:
    return FaissMtmAdapter(path=str(tmp_path / "faiss"), dim=VECTOR_DIM)


async def test_upsert_and_semantic_recall(
    mtm: FaissMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    item = make_item(ns, "vector store fact")
    await mtm.upsert(item)
    hits = await mtm.semantic(ns, item.embedding or [], limit=5)
    assert [h.item.id for h in hits] == [item.id]
    assert hits[0].item.content == item.content


async def test_idempotent_upsert_is_id_stable(
    mtm: FaissMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    item = make_item(ns, "idempotent")
    await mtm.upsert(item)
    await mtm.upsert(item)  # second write must NOT fork the FAISS index (remove-then-add, D6)
    _, idx = await mtm._index_for(ns)
    assert idx.index.ntotal == 1


async def test_namespace_filter(
    mtm: FaissMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")  # same partition, different η -> mandatory namespace filter
    a = make_item(ns_a, "only in A")
    await mtm.upsert(a)
    hits_b = await mtm.semantic(ns_b, a.embedding or [], limit=10)
    assert hits_b == []


async def test_state_active_supersede_drop(
    mtm: FaissMtmAdapter,
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
    assert loser.id not in ids
    assert winner.id in ids


async def test_shared_plane_refused(
    mtm: FaissMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    # D3: FAISS is a brute-force backend — refused on SHARED even at the adapter call boundary,
    # defense-in-depth alongside the registry's build-time refusal (test_registry_unit.py).
    ns = make_ns(visibility=Visibility.SHARED)
    item = make_item(ns, "shared attempt")
    with pytest.raises(VectorNotFilterableError):
        await mtm.upsert(item)
    with pytest.raises(VectorNotFilterableError):
        await mtm.semantic(ns, item.embedding or [], limit=5)


async def test_persists_across_adapter_instances(
    tmp_path: Path,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    # in-proc floor still persists to disk (spec §1 embedded floor) — a fresh adapter instance
    # pointed at the same path recovers prior writes.
    path = str(tmp_path / "faiss_persist")
    ns = make_ns()
    item = make_item(ns, "durable fact")
    first = FaissMtmAdapter(path=path, dim=VECTOR_DIM)
    await first.upsert(item)

    second = FaissMtmAdapter(path=path, dim=VECTOR_DIM)
    hits = await second.semantic(ns, item.embedding or [], limit=5)
    assert [h.item.id for h in hits] == [item.id]
