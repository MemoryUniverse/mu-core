"""The four by-id MTM **WRITE** verbs must be namespace-scoped IN THE ADAPTER — REAL
``mu-dev-qdrant``, ZERO mocks (C3, ARCHITECTURE-CONFORMANCE audit).

``expire`` / ``invalidate`` / ``set_entity_uids`` / ``remove`` each resolved a point through two
keys, NEITHER of which is namespace-scoped: the collection is
``mu_mtm__{workspace}__{visibility}__{dim}`` (no org, no user — ``qdrant_mapper.collection_name``)
and the point id is ``uuid5(NAMESPACE_URL, memory_id)`` (no namespace salt). A bare ``memory_id``
from another org or another user therefore addressed a REAL point of the same collection, and
``remove`` is a hard ``AsyncQdrantClient.delete``.

It was not a live leak, because every caller pre-gated on the guarded read
(``surface/facade.py`` calls ``mtm.get`` first, and ``_get_impl`` compares namespaces). That is
the whole problem: the safety property lived by CONVENTION in three call sites of ANOTHER package.
``storage-pluggable-spec.md:483-485`` puts it in the ADAPTER — every adapter applies the
exact-equality tenancy predicate unconditionally on every read **and write** — and CANONICAL
§1 rule 5 calls ``to_prefix()`` the tenancy GUARANTEE, "Not a filter". These tests call the write
verbs DIRECTLY, with no ``get`` pre-gate, which is exactly the new-caller / dropped-pre-gate shape
the fix has to survive.

Every test asserts on the RAW Qdrant payload rather than on ``mtm.get``: ``get`` applies its own
namespace refusal, and two of the four written keys (``superseded_by``, ``entity_uids``) are not
``MemoryItem`` fields at all and are dropped by ``QdrantMapper.from_store``. The raw point is the
real store state.

Each verb is covered TWICE — refused across a namespace boundary AND still working within its own.
Without the second half, an implementation that turned all four verbs into no-ops would pass every
refusal test (this project's recorded "passes while proving nothing" failure mode).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.errors import MtmPointAbsentError
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id

from .conftest import VECTOR_DIM

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def mtm(qdrant_client: AsyncQdrantClient) -> AsyncIterator[QdrantMtmAdapter]:
    adapter = QdrantMtmAdapter(qdrant_client, dim=VECTOR_DIM)
    yield adapter
    # teardown: drop any collection this test created (unique workspace per test).
    existing = {c.name for c in (await qdrant_client.get_collections()).collections}
    for name in list(existing):
        if name.startswith("mu_mtm__ws"):
            with contextlib.suppress(Exception):  # best-effort teardown only
                await qdrant_client.delete_collection(name)


async def _raw(client: AsyncQdrantClient, ns: Namespace, memory_id: str) -> dict[str, Any] | None:
    """The point's payload straight out of Qdrant — no adapter, no mapper, no namespace guard."""
    records = await client.retrieve(
        collection_name=collection_name(ns, VECTOR_DIM),
        ids=[point_id(memory_id)],
        with_payload=True,
    )
    return dict(records[0].payload or {}) if records else None


def _shared_collection(victim_ns: Namespace, caller_ns: Namespace) -> None:
    """The load-bearing PRECONDITION.

    If the two namespaces landed in DIFFERENT collections, every refusal test below would pass
    for a reason that has nothing to do with namespace scoping — the write would simply have been
    aimed at another collection. Same workspace, same visibility, different user slot puts them in
    one collection with two different ``to_prefix()`` values, which is the only configuration in
    which the adapter itself is what refuses.
    """
    assert collection_name(victim_ns, VECTOR_DIM) == collection_name(caller_ns, VECTOR_DIM), (
        "precondition failed: the two namespaces are in different Qdrant collections, so this "
        "test would pass without any namespace scoping and would prove nothing"
    )
    assert victim_ns.to_prefix() != caller_ns.to_prefix(), (
        "precondition failed: the two namespaces have the same to_prefix(), so there is no "
        "boundary here to cross"
    )


# --------------------------------------------------------------------------------------
# expire — soft-delete (state='expired' + invalid_at)
# --------------------------------------------------------------------------------------


async def test_expire_refuses_another_namespaces_memory(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    victim_ns = make_ns(user="u_victim")
    caller_ns = make_ns(user="u_caller")
    _shared_collection(victim_ns, caller_ns)
    victim = make_item(victim_ns, "the victim's live memory")
    await mtm.upsert(victim)
    before = await _raw(qdrant_client, victim_ns, victim.id)
    assert before is not None, "the victim is not even stored"
    assert before["state"] == MemoryState.ACTIVE.value

    # Refused AND reported: the three payload verbs raise the typed absence signal rather than
    # succeeding silently, so a caller that depends on "the point wasn't there" (the distill
    # degrade + retry queue) still hears it once the write is namespace-scoped.
    with pytest.raises(MtmPointAbsentError):
        await mtm.expire(caller_ns, victim.id, at=datetime.now(UTC))

    after = await _raw(qdrant_client, victim_ns, victim.id)
    assert after is not None, "a foreign expire made the victim's point disappear entirely"
    assert after["state"] == MemoryState.ACTIVE.value, (
        "a foreign namespace soft-deleted the victim's memory: it is no longer active, so the "
        "mandatory state='active' recall filter now drops it from its owner's recall"
    )
    assert after.get("invalid_at") == before.get(
        "invalid_at"
    ), "a foreign expire stamped invalid_at on the victim's point"


async def test_expire_in_its_own_namespace_still_expires(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The control for :func:`test_expire_refuses_another_namespaces_memory` — without it,
    making ``expire`` an unconditional no-op would pass the refusal test."""
    ns = make_ns()
    item = make_item(ns, "expire me, legitimately")
    await mtm.upsert(item)
    at = datetime.now(UTC)

    await mtm.expire(ns, item.id, at=at)

    after = await _raw(qdrant_client, ns, item.id)
    assert after is not None, "expire must SOFT-delete — the point has to stay (bi-temporal)"
    assert after["state"] == MemoryState.EXPIRED.value
    assert after["invalid_at"] == at.isoformat()


# --------------------------------------------------------------------------------------
# invalidate — id-stable supersede (state='superseded' + superseded_by)
# --------------------------------------------------------------------------------------


async def test_invalidate_refuses_another_namespaces_memory(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    victim_ns = make_ns(user="u_victim")
    caller_ns = make_ns(user="u_caller")
    _shared_collection(victim_ns, caller_ns)
    victim = make_item(victim_ns, "the victim's current truth")
    await mtm.upsert(victim)
    winner = make_item(caller_ns, "the caller's replacement truth")
    await mtm.upsert(winner)
    before = await _raw(qdrant_client, victim_ns, victim.id)
    assert before is not None and before["state"] == MemoryState.ACTIVE.value

    with pytest.raises(MtmPointAbsentError):
        await mtm.invalidate(
            caller_ns, victim.id, winner.id, at=datetime.now(UTC), reason="cross-tenant-supersede"
        )

    after = await _raw(qdrant_client, victim_ns, victim.id)
    assert after is not None, "a foreign invalidate made the victim's point disappear entirely"
    assert after["state"] == MemoryState.ACTIVE.value, (
        "a foreign namespace superseded the victim's memory — it is dropped from its owner's "
        "active recall and now points at a winner the owner cannot even see"
    )
    assert "superseded_by" not in after, (
        f"a foreign invalidate wrote a supersession edge onto the victim's point: "
        f"superseded_by={after.get('superseded_by')!r}"
    )
    assert after.get("invalid_at") == before.get("invalid_at")


async def test_invalidate_in_its_own_namespace_still_supersedes(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The control for :func:`test_invalidate_refuses_another_namespaces_memory`."""
    ns = make_ns()
    loser = make_item(ns, "old truth")
    winner = make_item(ns, "new truth")
    await mtm.upsert(loser)
    await mtm.upsert(winner)
    at = datetime.now(UTC)

    await mtm.invalidate(ns, loser.id, winner.id, at=at, reason="legitimate-supersede")

    after = await _raw(qdrant_client, ns, loser.id)
    assert after is not None, "invalidate must overwrite the payload, never delete the point"
    assert after["state"] == MemoryState.SUPERSEDED.value
    assert after["superseded_by"] == winner.id
    assert after["supersede_reason"] == "legitimate-supersede"
    assert after["invalid_at"] == at.isoformat()


# --------------------------------------------------------------------------------------
# set_entity_uids — payload backfill from the LTM entity resolution (D-5)
# --------------------------------------------------------------------------------------


async def test_set_entity_uids_refuses_another_namespaces_memory(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    victim_ns = make_ns(user="u_victim")
    caller_ns = make_ns(user="u_caller")
    _shared_collection(victim_ns, caller_ns)
    victim = make_item(victim_ns, "the victim's memory awaiting nothing")
    await mtm.upsert(victim)
    before = await _raw(qdrant_client, victim_ns, victim.id)
    assert before is not None and "entity_uids" not in before

    with pytest.raises(MtmPointAbsentError):
        await mtm.set_entity_uids(caller_ns, victim.id, ["ent_from_another_tenant"])

    after = await _raw(qdrant_client, victim_ns, victim.id)
    assert after is not None, "a foreign set_entity_uids made the victim's point disappear"
    assert "entity_uids" not in after, (
        "a foreign namespace grafted its own resolved entity uids onto the victim's point — "
        f"entity_uids={after.get('entity_uids')!r}; that is another tenant's graph identity "
        "written into this tenant's data"
    )


async def test_set_entity_uids_in_its_own_namespace_still_writes(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The control for :func:`test_set_entity_uids_refuses_another_namespaces_memory`."""
    ns = make_ns()
    item = make_item(ns, "backfill my entity uids")
    await mtm.upsert(item)

    await mtm.set_entity_uids(ns, item.id, ["ent_subject", "ent_object"])

    after = await _raw(qdrant_client, ns, item.id)
    assert after is not None
    assert after["entity_uids"] == ["ent_subject", "ent_object"]


# --------------------------------------------------------------------------------------
# remove — the HARD delete
# --------------------------------------------------------------------------------------


async def test_remove_refuses_another_namespaces_memory(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The destructive one: ``remove`` is a real ``AsyncQdrantClient.delete``, so an unscoped
    foreign call does not merely mislabel the victim's memory — it destroys it, with no
    bi-temporal history left behind to recover it from."""
    victim_ns = make_ns(user="u_victim")
    caller_ns = make_ns(user="u_caller")
    _shared_collection(victim_ns, caller_ns)
    victim = make_item(victim_ns, "the victim's only copy")
    await mtm.upsert(victim)
    before = await _raw(qdrant_client, victim_ns, victim.id)
    assert before is not None, "the victim is not even stored"

    await mtm.remove(caller_ns, victim.id)

    after = await _raw(qdrant_client, victim_ns, victim.id)
    assert (
        after is not None
    ), "a foreign namespace HARD-DELETED the victim's memory from Qdrant — irrecoverable"
    assert after["state"] == before["state"]
    assert after["content_hash"] == before["content_hash"]


async def test_remove_in_its_own_namespace_still_deletes(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The control for :func:`test_remove_refuses_another_namespaces_memory` — ``remove`` is the
    demotion tier-down move and MUST still genuinely delete, or the MTM->STM move silently leaves
    a duplicate behind in MTM."""
    ns = make_ns()
    item = make_item(ns, "really delete me")
    await mtm.upsert(item)
    assert await _raw(qdrant_client, ns, item.id) is not None

    await mtm.remove(ns, item.id)

    assert (
        await _raw(qdrant_client, ns, item.id) is None
    ), "remove is a genuine point deletion, not a payload flip"
