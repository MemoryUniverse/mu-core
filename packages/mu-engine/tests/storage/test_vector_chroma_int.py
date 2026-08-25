"""Chroma/MTM adapter — REAL embedded (on-disk) chromadb, ZERO mocks.

No ``mu-dev-*`` container is involved (Chroma's ``PersistentClient`` is embedded, spec §1), but
every call in this suite hits the REAL chromadb library against a real temp directory — nothing is
mocked. Mirrors ``test_vector_qdrant_int.py``'s conformance body (spec §8) so Chroma proves the
same invariants: store+retrieve, namespace filter, idempotent (id-stable) upsert, the
``state='active'`` cross-store supersede drop, and the SHARED ``authorized_ids`` Model-A
completeness hazard — here via the documented WIDEN-then-post-filter path (``chroma_mtm.py``
module docstring) since Chroma has no native array-membership filter.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mu_engine.storage.adapters.chroma_mtm import ChromaMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.chroma_mapper import chroma_collection_name
from mu_engine.storage.mappers.qdrant_mapper import point_id

from .conftest import VECTOR_DIM

# embedded, container-free — real chromadb, zero mocks (see module docstring)
pytestmark = pytest.mark.unit


@pytest.fixture
def mtm(tmp_path: Path) -> ChromaMtmAdapter:
    return ChromaMtmAdapter(path=str(tmp_path / "chroma"), dim=VECTOR_DIM)


async def test_upsert_and_semantic_recall(
    mtm: ChromaMtmAdapter,
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
    mtm: ChromaMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    item = make_item(ns, "idempotent")
    await mtm.upsert(item)
    await mtm.upsert(item)  # second write must NOT fork (uuid5 point id, spec §6 rule 5)
    col = await mtm._collection(ns)
    assert col.count() == 1


async def test_namespace_filter(
    mtm: ChromaMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")  # same collection, different η -> mandatory namespace filter
    a = make_item(ns_a, "only in A")
    await mtm.upsert(a)
    hits_b = await mtm.semantic(ns_b, a.embedding or [], limit=10)
    assert hits_b == []


async def test_state_active_supersede_drop(
    mtm: ChromaMtmAdapter,
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
    assert loser.id not in ids  # dropped by state='active' filter (B1)
    assert winner.id in ids


async def test_shared_authorized_ids_completeness(
    mtm: ChromaMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    # SHARED plane: caller authorized for only a SPARSE subset must still see it (M1 hazard).
    # Chroma has no native array-membership filter (module docstring) — this proves the WIDEN
    # path preserves completeness rather than silently truncating before the authz check.
    ns = make_ns(visibility=Visibility.SHARED)
    mine = make_item(ns, "authorized fact", authorized_ids=["p_alice", "p_bob"])
    not_mine = make_item(ns, "someone else fact", authorized_ids=["p_carol"])
    await mtm.upsert(mine)
    await mtm.upsert(not_mine)
    caller = frozenset({"p_alice"})
    hits = await mtm.semantic(ns, mine.embedding or [], limit=10, caller_identity_set=caller)
    ids = {h.item.id for h in hits}
    assert mine.id in ids  # authorized-and-relevant surfaces
    assert not_mine.id not in ids  # unauthorized filtered before truncation (Model A)


async def test_invalidate_refuses_another_namespaces_memory(
    mtm: ChromaMtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """C3 on the Chroma backend: the by-id supersede WRITE must carry the tenancy predicate.

    ``chroma_collection_name(ns, dim)`` hashes the WORKSPACE + visibility only — no org, no user
    — and the id is ``uuid5(NAMESPACE_URL, memory_id)``, unsalted by namespace, so a bare
    ``loser_id`` from another principal names a real row of the same collection. The adapter must
    apply the exact-equality scope itself (``storage-pluggable-spec.md:483-485``); this test
    calls ``invalidate`` DIRECTLY, with no read pre-gate.
    """
    victim_ns = make_ns(user="u_victim")
    caller_ns = make_ns(user="u_caller")
    # PRECONDITION: without a shared collection this proves nothing about namespace scoping.
    assert chroma_collection_name(victim_ns, VECTOR_DIM) == chroma_collection_name(
        caller_ns, VECTOR_DIM
    ), "precondition failed: the two namespaces are in different Chroma collections"
    assert victim_ns.to_prefix() != caller_ns.to_prefix()
    victim = make_item(victim_ns, "the victim's current truth")
    winner = make_item(caller_ns, "the caller's replacement truth")
    await mtm.upsert(victim)
    await mtm.upsert(winner)

    await mtm.invalidate(
        caller_ns, victim.id, winner.id, at=datetime.now(UTC), reason="cross-tenant-supersede"
    )

    col = await mtm._collection(victim_ns)
    got = col.get(ids=[point_id(victim.id)], include=["metadatas", "documents"])
    metas = got.get("metadatas") or []
    assert metas and metas[0] is not None, "a foreign invalidate removed the victim's row"
    assert metas[0]["state"] == MemoryState.ACTIVE.value, (
        "a foreign namespace superseded the victim's memory — it is now dropped from its "
        "owner's active recall"
    )
    doc = (got.get("documents") or [None])[0]
    assert "superseded_by" not in json.loads(
        doc or "{}"
    ), "a foreign invalidate wrote a supersession edge onto the victim's document"


async def test_invalidate_in_its_own_namespace_still_supersedes(
    mtm: ChromaMtmAdapter,
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

    col = await mtm._collection(ns)
    got = col.get(ids=[point_id(loser.id)], include=["metadatas", "documents"])
    metas = got.get("metadatas") or []
    assert metas and metas[0] is not None, "invalidate must update the row, never delete it"
    assert metas[0]["state"] == MemoryState.SUPERSEDED.value
    payload = json.loads((got.get("documents") or ["{}"])[0] or "{}")
    assert payload["superseded_by"] == winner.id
    assert payload["supersede_reason"] == "legitimate-supersede"
    assert payload["invalid_at"] == at.isoformat()
