"""Cross-session, per-user recall on Weaviate — REAL tunnelled instance, ZERO mocks (BQ3; ADR
0030; the BLOCKER this file closes).

Before this fix, ``WeaviateMtmAdapter.semantic`` took no ``session_scope`` and always filtered on
the FULL ``namespace`` property (session included) — every PRIVATE recall was silently
session-scoped, and ADR 0030/BQ3 cross-session federation returned nothing from a caller's other
sessions. No error, no degrade signal, just less memory. This file is the Weaviate twin of
``test_qdrant_mtm_session_scope_int.py`` (same acceptance shape, same three-way
``_resolve_namespace_match`` branch), covering:

* AC-int: two ``MemoryItem``s written under the same PRIVATE user in two different sessions are
  BOTH returned by a ``session_scope=None`` recall (the federated default), and only the matching
  one is returned when ``session_scope`` narrows to one of them.
* Federation is per-USER, not per-tenant: unlike Qdrant (one collection per (org, workspace,
  visibility)), Weaviate's tenant is coarser — (org, workspace) ONLY (ADR 0050) — so two
  different PRIVATE users share the SAME physical tenant shard here. ``session_scope=None`` must
  still never leak one user's memories into another's recall.
* AC-4.3-adjacent: a SHARED ``Namespace`` NEVER receives the truncated/opt-in match value,
  regardless of ``session_scope`` — including the misconfigured case of ``session_scope=None`` on
  a SHARED room, which must still behave exactly like the session-scoped SHARED room lookup (no
  cross-room leak).
* The ``user_prefix`` property is a REAL, retrievable, indexed property on the live object — not
  just an in-memory filter value.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
import weaviate

from mu_engine.storage.adapters.weaviate_mtm import (
    USER_PREFIX_PROPERTY,
    WeaviateMtmAdapter,
    _resolve_namespace_match,
    _user_prefix,
)
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import point_id
from mu_engine.storage.mappers.weaviate_mapper import collection_name, tenant_name

pytestmark = pytest.mark.integration

_WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://127.0.0.1:18080")
VECTOR_DIM = 8  # mirrors test_weaviate_mtm_int.py's own tiny deterministic vectors


def _parse_host_port(url: str) -> tuple[str, int]:
    rest = url.split("://", 1)[-1]
    host, _, port = rest.partition(":")
    return host, int(port or 80)


class _NamespaceFactory:
    """Local factory mirroring ``test_weaviate_mtm_int.py``'s own (kept local for this file's own
    teardown grain, per that file's documented rationale)."""

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


@pytest_asyncio.fixture
async def mtm(
    make_ns: _NamespaceFactory,
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
    await adapter._ensure_connected()
    assert await client.is_ready()  # fail-loud if the tunnelled instance is not actually up
    yield adapter
    class_name = collection_name(VECTOR_DIM)
    if await client.collections.exists(class_name):
        coll = client.collections.get(class_name)
        for ns in make_ns.created:
            tenant = tenant_name(ns)
            if await coll.tenants.exists(tenant):
                await coll.tenants.remove([tenant])
    await adapter.close()


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
    ) -> MemoryItem:
        seed = sum(ord(c) for c in content)
        embedding = [((seed + i) % 17) / 17.0 for i in range(VECTOR_DIM)]
        meta = {"authorized_ids": authorized_ids} if authorized_ids is not None else {}
        return MemoryItem(
            content=content,
            namespace=ns,
            owner_id=ns.user if ns.visibility is Visibility.PRIVATE else "owner1",
            workspace_id=ns.workspace,
            session_id=ns.session,
            embedding=embedding,
            embedding_model="test-fixture",
            metadata=meta,
        )

    return _make


# ------------------------------------------------------------------ live federation (AC-int)


async def test_session_scope_none_federates_both_sessions(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    """Same user, two sessions: session_scope=None (the new default) returns BOTH."""
    ns_a = make_ns(session="session-a")
    ns_b = make_ns(session="session-b")  # same org/workspace/user as ns_a — same tenant
    item_a = make_item(ns_a, "fact learned in session A")
    item_b = make_item(ns_b, "fact learned in session B")
    await mtm.upsert(item_a)
    await mtm.upsert(item_b)

    hits = await mtm.semantic(ns_a, item_a.embedding or [], limit=10)
    ids = {h.item.id for h in hits}
    assert item_a.id in ids
    assert item_b.id in ids  # cross-session federation — the whole point of ADR 0030


async def test_session_scope_set_narrows_to_one_session(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    """A concrete session_scope narrows exactly like the pre-federation behavior."""
    ns_a = make_ns(session="session-a")
    ns_b = make_ns(session="session-b")
    item_a = make_item(ns_a, "fact learned in session A only")
    item_b = make_item(ns_b, "fact learned in session B only")
    await mtm.upsert(item_a)
    await mtm.upsert(item_b)

    hits = await mtm.semantic(ns_a, item_a.embedding or [], limit=10, session_scope="session-a")
    ids = {h.item.id for h in hits}
    assert item_a.id in ids
    assert item_b.id not in ids  # narrowed OUT — session_scope excludes session-b

    # narrowing to the OTHER session works even when the query namespace's own `session` differs
    # from the requested `session_scope` ("need not equal ns.session").
    hits_b = await mtm.semantic(ns_a, item_b.embedding or [], limit=10, session_scope="session-b")
    ids_b = {h.item.id for h in hits_b}
    assert item_b.id in ids_b
    assert item_a.id not in ids_b


async def test_different_users_never_cross_federate(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    """Federation is per-USER: session_scope=None must never leak across different users —
    the highest-risk case on Weaviate specifically, since (unlike Qdrant's collection-per-
    (org,workspace,visibility)) two different PRIVATE users here share the SAME physical
    tenant shard (ADR 0050: tenant grain is (org, workspace) only); only the `where` filter
    (namespace / user_prefix) separates them."""
    ws = f"shared-ws-{uuid.uuid4().hex[:8]}"
    ns_u1 = make_ns(workspace=ws, user="u1", session="sX")
    ns_u2 = make_ns(workspace=ws, user="u2", session="sY")
    assert tenant_name(ns_u1) == tenant_name(ns_u2), "test bug: expected the same tenant shard"
    item_u1 = make_item(ns_u1, "u1 private fact")
    item_u2 = make_item(ns_u2, "u2 private fact")
    await mtm.upsert(item_u1)
    await mtm.upsert(item_u2)

    hits = await mtm.semantic(ns_u1, item_u1.embedding or [], limit=10)
    ids = {h.item.id for h in hits}
    assert item_u1.id in ids
    assert item_u2.id not in ids  # different user — never federated


async def test_shared_recall_unaffected_by_session_scope_none(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    """End-to-end guard: a SHARED room recall with session_scope=None (the PRIVATE-own
    federated default) still isolates by room — no cross-room leak from the new default."""
    room_1 = make_ns(visibility=Visibility.SHARED, session="room-1")
    room_2 = make_ns(visibility=Visibility.SHARED, session="room-2")
    in_room_1 = make_item(room_1, "room 1 secret", authorized_ids=["p_alice"])
    in_room_2 = make_item(room_2, "room 2 secret", authorized_ids=["p_alice"])
    await mtm.upsert(in_room_1)
    await mtm.upsert(in_room_2)

    caller = frozenset({"p_alice"})
    hits = await mtm.semantic(
        room_1,
        in_room_1.embedding or [],
        limit=10,
        caller_identity_set=caller,
        session_scope=None,
    )
    ids = {h.item.id for h in hits}
    assert in_room_1.id in ids
    assert in_room_2.id not in ids  # SHARED room-as-wall holds regardless of session_scope


async def test_upsert_stamps_indexed_user_prefix_property(
    mtm: WeaviateMtmAdapter,
    make_ns: _NamespaceFactory,
    make_item: Callable[..., MemoryItem],
) -> None:
    """The truncated user-prefix is a real, retrievable, INDEXED property on the live object —
    verified via a direct REST object read AND a server-side GraphQL where-filter on it."""
    ns = make_ns(session="session-z")
    item = make_item(ns, "indexed field check")
    await mtm.upsert(item)

    resp = await mtm._http.get(
        f"/v1/objects/{mtm._class}/{point_id(item.id)}",
        params={"tenant": tenant_name(ns)},
    )
    resp.raise_for_status()
    props = resp.json()["properties"]
    assert props[USER_PREFIX_PROPERTY] == _user_prefix(ns)
    assert not props[USER_PREFIX_PROPERTY].endswith("session-z")  # session dropped

    hits = await mtm.semantic(ns, item.embedding or [], limit=10, session_scope=None)
    assert item.id in {h.item.id for h in hits}  # the property is actually load-bearing, not dead


# ------------------------------------------------------------------ pure resolver branch (no I/O)


@pytest.mark.parametrize("session_scope", [None, "room-1", "some-other-session"])
def test_shared_never_gets_truncated_match_value(session_scope: str | None) -> None:
    """AC-4.3-adjacent: a SHARED Namespace NEVER resolves to the truncated match value, for ANY
    session_scope — including the misconfiguration case (session_scope=None, the PRIVATE-own
    federated default, on a SHARED room)."""
    ns = Namespace.shared(org="o", workspace="w", session="room-1")
    prop, value = _resolve_namespace_match(ns, session_scope=session_scope)
    assert prop == "namespace"
    assert value == ns.to_prefix()  # always the exact, full, session-included prefix


@pytest.mark.parametrize("session_scope", [None, "session-a", "session-other"])
def test_private_truncated_only_when_scope_is_none(session_scope: str | None) -> None:
    """PRIVATE resolves to the truncated user-prefix ONLY when session_scope is None; any
    concrete session_scope resolves back to a full, session-included match."""
    ns = Namespace(
        org="o", workspace="w", user="u1", session="session-a", visibility=Visibility.PRIVATE
    )
    prop, value = _resolve_namespace_match(ns, session_scope=session_scope)
    if session_scope is None:
        assert prop == USER_PREFIX_PROPERTY
        assert value != ns.to_prefix()
        assert not value.endswith("session-a")
    else:
        assert prop == "namespace"
        assert (
            value
            == Namespace(
                org=ns.org,
                workspace=ns.workspace,
                user=ns.user,
                session=session_scope,
                visibility=ns.visibility,
            ).to_prefix()
        )
