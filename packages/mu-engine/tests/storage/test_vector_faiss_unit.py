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

from mu_engine.storage.adapters.faiss_mtm import FaissMtmAdapter, _int_id
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.errors import VectorNotFilterableError
from mu_engine.storage.mappers.faiss_mapper import faiss_collection_name
from mu_engine.storage.mappers.qdrant_mapper import point_id

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


async def test_invalidate_refuses_another_namespaces_memory(
    mtm: FaissMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """C3 on the FAISS backend: the by-id supersede WRITE must carry the tenancy predicate.

    ``faiss_collection_name(ns, dim)`` hashes ``org``+``workspace`` (jointly, since the fix for the
    cross-org collision — see that function's docstring) + visibility — still no ``user`` — so
    same-org-same-workspace-different-user still land in the SAME index. The docstore key derives
    from ``uuid5(NAMESPACE_URL, memory_id)``, unsalted by namespace,
    so a bare ``loser_id`` from another principal resolves to a real entry of the same docstore.
    FAISS has no server to push a filter to, so the adapter applies the same
    ``payload["namespace"] != to_prefix()`` predicate its own search path uses — in the adapter
    (``storage-pluggable-spec.md:483-485``), not at a call site in another package. This test
    calls ``invalidate`` DIRECTLY, with no read pre-gate.
    """
    victim_ns = make_ns(user="u_victim")
    caller_ns = make_ns(user="u_caller")
    # PRECONDITION: without a shared index this proves nothing about namespace scoping.
    assert faiss_collection_name(victim_ns, VECTOR_DIM) == faiss_collection_name(
        caller_ns, VECTOR_DIM
    ), "precondition failed: the two namespaces are in different FAISS indices"
    assert victim_ns.to_prefix() != caller_ns.to_prefix()
    victim = make_item(victim_ns, "the victim's current truth")
    winner = make_item(caller_ns, "the caller's replacement truth")
    await mtm.upsert(victim)
    await mtm.upsert(winner)

    await mtm.invalidate(
        caller_ns, victim.id, winner.id, at=datetime.now(UTC), reason="cross-tenant-supersede"
    )

    _, idx = await mtm._index_for(victim_ns)
    entry = idx.docstore[_int_id(point_id(victim.id))]
    assert entry["payload"]["state"] == MemoryState.ACTIVE.value, (
        "a foreign namespace superseded the victim's memory — it is now dropped from its "
        "owner's active recall"
    )
    assert (
        "superseded_by" not in entry["payload"]
    ), "a foreign invalidate wrote a supersession edge onto the victim's docstore entry"


async def test_invalidate_in_its_own_namespace_still_supersedes(
    mtm: FaissMtmAdapter,
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

    _, idx = await mtm._index_for(ns)
    payload = idx.docstore[_int_id(point_id(loser.id))]["payload"]
    assert payload["state"] == MemoryState.SUPERSEDED.value
    assert payload["superseded_by"] == winner.id
    assert payload["supersede_reason"] == "legitimate-supersede"
    assert payload["invalid_at"] == at.isoformat()
