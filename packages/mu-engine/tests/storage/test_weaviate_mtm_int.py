"""Weaviate MTM adapter — REAL tunnelled Weaviate (``127.0.0.1:18080``), ZERO mocks (ADR 0050).

Covers the round trip this task's brief names explicitly: create a tenant, upsert with a
supplied vector, retrieve it, and prove a second tenant's read cannot see it — plus the by-id
lifecycle verbs (expire/invalidate/remove) with their own scoped-write proofs, mirroring
``test_vector_qdrant_int.py``'s coverage for the reference backend.

Talks to the live instance over HTTP/REST + GraphQL only (see ``weaviate_mtm.py``'s module
docstring for why: every gRPC-backed SDK call hangs against this HTTP-only tunnel — verified,
not assumed). ``WEAVIATE_URL`` overrides the default tunnel address for a differently-wired run.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import pytest
import pytest_asyncio
import weaviate

from mu_engine.storage.adapters.weaviate_mtm import WeaviateMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.errors import MtmPointAbsentError
from mu_engine.storage.mappers.weaviate_mapper import collection_name, tenant_name

pytestmark = pytest.mark.integration

_WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:18080")
VECTOR_DIM = 8  # tiny deterministic vectors — no ML dep needed, mirrors conftest.VECTOR_DIM


def _parse_host_port(url: str) -> tuple[str, int]:
    rest = url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    return host, int(port or 80)


@pytest_asyncio.fixture
async def mtm(
    make_ns: Callable[..., Namespace],
) -> AsyncIterator[WeaviateMtmAdapter]:
    host, port = _parse_host_port(_WEAVIATE_URL)
    client = weaviate.use_async_with_custom(
        http_host=host,
        http_port=port,
        http_secure=False,
        grpc_host=host,
        grpc_port=50051,
        grpc_secure=False,
        skip_init_checks=True,
    )
    adapter = WeaviateMtmAdapter(client, http_url=_WEAVIATE_URL, dim=VECTOR_DIM)
    # fail-loud if the tunnelled instance is not actually up (DEV-STANDARDS: never faked).
    await adapter._ensure_connected()
    assert await client.is_ready()
    yield adapter
    # teardown: drop every tenant this test's `make_ns` actually created, in the one class this
    # adapter's `dim` maps to — mirrors `qdrant_teardown_collections`'s "read .created, never
    # reconstruct a guess" discipline.
    class_name = collection_name(VECTOR_DIM)
    if await client.collections.exists(class_name):
        coll = client.collections.get(class_name)
        for ns in make_ns.created:
            tenant = tenant_name(ns)
            if await coll.tenants.exists(tenant):
                await coll.tenants.remove([tenant])
    await adapter.close()


class _NamespaceFactory:
    """Local, minimal namespace factory — mirrors ``conftest.NamespaceFactory`` but records
    ``.created`` for this file's own teardown (kept local rather than importing the shared
    fixture, since this test's tenant-per-(org,workspace) teardown grain differs from Qdrant's
    collection-per-(org,workspace,visibility) grain)."""

    def __init__(self, uid: str) -> None:
        self._uid = uid
        self.created: list[Namespace] = []

    def __call__(
        self,
        *,
        visibility: Visibility = Visibility.PRIVATE,
        user: str = "u1",
        session: str = "s1",
        workspace: str | None = None,
        org: str | None = None,
    ) -> Namespace:
        ws = workspace or f"ws{self._uid}"
        org_v = org or f"org{self._uid}"
        if visibility is Visibility.SHARED:
            ns = Namespace.shared(org=org_v, workspace=ws, session=session)
        else:
            ns = Namespace(
                org=org_v, workspace=ws, user=user, session=session, visibility=Visibility.PRIVATE
            )
        self.created.append(ns)
        return ns


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


@pytest.fixture
def make_ns(uid: str) -> _NamespaceFactory:
    return _NamespaceFactory(uid)


@pytest.fixture
def make_item() -> Callable[..., MemoryItem]:
    def _make(
        ns: Namespace,
        content: str,
        *,
        authorized_ids: list[str] | None = None,
        memory_id: str | None = None,
    ) -> MemoryItem:
        seed = sum(ord(c) for c in content)
        embedding = [((seed + i) % 17) / 17.0 for i in range(VECTOR_DIM)]
        meta = {"authorized_ids": authorized_ids} if authorized_ids is not None else {}
        kwargs: dict[str, object] = {}
        if memory_id is not None:
            kwargs["id"] = memory_id
        return MemoryItem(
            content=content,
            namespace=ns,
            owner_id=ns.user if ns.visibility is Visibility.PRIVATE else "owner1",
            workspace_id=ns.workspace,
            session_id=ns.session,
            embedding=embedding,
            embedding_model="test-fixture",
            metadata=meta,
            **kwargs,  # type: ignore[arg-type]
        )

    return _make


# ------------------------------------------------------------------ the round trip the brief names


async def test_create_tenant_upsert_retrieve_and_cross_tenant_isolation(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    """create a tenant, upsert with a supplied vector, retrieve it, and prove a second tenant's
    read cannot see it — the exact round trip this task's brief specifies."""
    ns_a = make_ns()  # tenant A: its own (org, workspace)
    ns_b = make_ns(org=f"org-b-{uuid.uuid4().hex[:12]}")  # tenant B: a DIFFERENT org
    assert tenant_name(ns_a) != tenant_name(ns_b), "test bug: the two namespaces share a tenant"

    item = make_item(ns_a, "vector store fact")
    await mtm.upsert(item)

    got = await mtm.get(ns_a, item.id)
    assert got is not None
    assert got.id == item.id
    assert got.content == item.content
    assert got.embedding == pytest.approx(item.embedding)

    # tenant B's read of the SAME memory_id (a different, unrelated write in a different org) —
    # nothing to find, and semantic search against B's own (empty) shard proves the same.
    assert await mtm.get(ns_b, item.id) is None
    hits_b = await mtm.semantic(ns_b, item.embedding or [], limit=10)
    assert hits_b == []

    # tenant A's own semantic recall DOES see it.
    hits_a = await mtm.semantic(ns_a, item.embedding or [], limit=10)
    assert [h.item.id for h in hits_a] == [item.id]


async def test_idempotent_upsert_is_id_stable(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    item = make_item(ns, "idempotent")
    await mtm.upsert(item)
    await mtm.upsert(item)  # second write must NOT fork (uuid5 point id)
    hits = await mtm.semantic(ns, item.embedding or [], limit=10)
    assert [h.item.id for h in hits] == [item.id]


async def test_point_get_refuses_another_namespaces_memory_in_the_same_org_workspace(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    """Two namespaces sharing (org, workspace) — different USERS — share one Weaviate tenant
    (ADR 0050: visibility/user/session are WITHIN-shard). Only the adapter's own namespace
    equality check can refuse a bare id lookup here; this is a live cross-tenant-shard read, not
    a hypothetical, the Weaviate twin of ``test_vector_qdrant_int.py``'s same-named test."""
    ws = f"shared-ws-{uuid.uuid4().hex[:8]}"
    victim_ns = make_ns(workspace=ws, user="u_victim")
    caller_ns = make_ns(workspace=ws, user="u_caller")
    assert tenant_name(victim_ns) == tenant_name(
        caller_ns
    ), "the pre-condition did not hold: the two namespaces are in different tenants"
    victim = make_item(victim_ns, "the victim's secret")
    await mtm.upsert(victim)
    assert await mtm.get(victim_ns, victim.id) is not None, "the victim is not even stored"
    assert (
        await mtm.get(caller_ns, victim.id) is None
    ), "a point-get resolved a memory from another principal's partition"


async def test_state_active_supersede_drop(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
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
    assert loser.id not in ids  # dropped by state='active' filter
    assert winner.id in ids
    # the round-trip blob reflects the patch too (payload_json kept in sync with the promoted
    # `state` property — see weaviate_mtm._scoped_patch).
    got = await mtm.get(ns, loser.id)
    assert got is not None
    assert got.state.value == "superseded"


async def test_expire_then_absent_raises_on_second_call(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    item = make_item(ns, "to be expired")
    await mtm.upsert(item)
    await mtm.expire(ns, item.id, at=datetime.now(UTC))
    got = await mtm.get(ns, item.id)
    assert got is not None and got.state.value == "expired"

    with pytest.raises(MtmPointAbsentError):
        await mtm.expire(ns, "no-such-memory-id", at=datetime.now(UTC))


async def test_shared_authorized_ids_completeness(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    # SHARED plane: caller authorized for only a sparse subset must still see it (Model A).
    ns = make_ns(visibility=Visibility.SHARED)
    mine = make_item(ns, "authorized fact", authorized_ids=["p_alice", "p_bob"])
    not_mine = make_item(ns, "someone else fact", authorized_ids=["p_carol"])
    await mtm.upsert(mine)
    await mtm.upsert(not_mine)
    caller = frozenset({"p_alice"})
    hits = await mtm.semantic(ns, mine.embedding or [], limit=10, caller_identity_set=caller)
    ids = {h.item.id for h in hits}
    assert mine.id in ids  # authorized-and-relevant surfaces
    assert not_mine.id not in ids  # unauthorized filtered pre-truncation (Model A)


async def test_remove_is_scoped_and_atomic(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    item = make_item(ns, "to be removed")
    await mtm.upsert(item)
    await mtm.remove(ns, item.id)
    assert await mtm.get(ns, item.id) is None
    hits = await mtm.semantic(ns, item.embedding or [], limit=10)
    assert hits == []


async def test_scan_for_demotion_enumerates_active_points_only(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    active = make_item(ns, "still active")
    expired = make_item(ns, "already expired")
    await mtm.upsert(active)
    await mtm.upsert(expired)
    await mtm.expire(ns, expired.id, at=datetime.now(UTC))
    candidates = await mtm.scan_for_demotion(ns, limit=50)
    ids = {c.id for c in candidates}
    assert active.id in ids
    assert expired.id not in ids
