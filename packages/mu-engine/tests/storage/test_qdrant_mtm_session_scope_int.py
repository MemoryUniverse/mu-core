"""Cross-session, per-user recall — REAL mu-dev-qdrant, ZERO mocks (S1-04; ADR 0030; BQ3).

Covers the acceptance items owned by this task:

* AC-int: two ``MemoryItem``s written under the same PRIVATE user in two different sessions
  are BOTH returned by a ``session_scope=None`` recall (the new federated default), and only
  the matching one is returned when ``session_scope`` narrows to one of them.
* AC-4.3-adjacent (pulled forward for local verification; full property-test enumeration is a
  slice-4 gate, §1 S5 test obligation): a SHARED-visibility ``Namespace`` NEVER receives the
  truncated/opt-in match value, regardless of ``session_scope`` — including the misconfigured
  case of ``session_scope=None`` on a SHARED room, which must still behave exactly like the
  session-scoped SHARED room lookup (no cross-room leak).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from qdrant_client import AsyncQdrantClient, models

from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter, _resolve_namespace_match
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility

from .conftest import VECTOR_DIM

pytestmark = pytest.mark.integration


@pytest.fixture
async def mtm(qdrant_client: AsyncQdrantClient) -> QdrantMtmAdapter:
    return QdrantMtmAdapter(qdrant_client, dim=VECTOR_DIM)


async def test_session_scope_none_federates_both_sessions(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """Same user, two sessions: session_scope=None (the new default) returns BOTH."""
    ns_a = make_ns(session="session-a")
    ns_b = make_ns(session="session-b")  # same org/workspace/user as ns_a — same collection
    item_a = make_item(ns_a, "fact learned in session A")
    item_b = make_item(ns_b, "fact learned in session B")
    await mtm.upsert(item_a)
    await mtm.upsert(item_b)
    try:
        hits = await mtm.semantic(ns_a, item_a.embedding or [], limit=10)
        ids = {h.item.id for h in hits}
        assert item_a.id in ids
        assert item_b.id in ids  # cross-session federation — the whole point of ADR 0030
    finally:
        from mu_engine.storage.mappers.qdrant_mapper import collection_name

        await qdrant_client.delete_collection(collection_name(ns_a, VECTOR_DIM))


async def test_session_scope_set_narrows_to_one_session(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """A concrete session_scope narrows exactly like the pre-federation behavior."""
    ns_a = make_ns(session="session-a")
    ns_b = make_ns(session="session-b")
    item_a = make_item(ns_a, "fact learned in session A only")
    item_b = make_item(ns_b, "fact learned in session B only")
    await mtm.upsert(item_a)
    await mtm.upsert(item_b)
    try:
        hits = await mtm.semantic(
            ns_a, item_a.embedding or [], limit=10, session_scope="session-a"
        )
        ids = {h.item.id for h in hits}
        assert item_a.id in ids
        assert item_b.id not in ids  # narrowed OUT — session_scope excludes session-b

        # narrowing to the OTHER user's session works even when the query namespace's own
        # `session` differs from the requested `session_scope` ("need not equal ns.session").
        hits_b = await mtm.semantic(
            ns_a, item_b.embedding or [], limit=10, session_scope="session-b"
        )
        ids_b = {h.item.id for h in hits_b}
        assert item_b.id in ids_b
        assert item_a.id not in ids_b
    finally:
        from mu_engine.storage.mappers.qdrant_mapper import collection_name

        await qdrant_client.delete_collection(collection_name(ns_a, VECTOR_DIM))


async def test_different_users_never_cross_federate(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """Federation is per-USER: session_scope=None must never leak across different users."""
    ns_u1 = make_ns(user="u1", session="sX")
    ns_u2 = make_ns(user="u2", session="sY")  # different user, same workspace/collection
    item_u1 = make_item(ns_u1, "u1 private fact")
    item_u2 = make_item(ns_u2, "u2 private fact")
    await mtm.upsert(item_u1)
    await mtm.upsert(item_u2)
    try:
        hits = await mtm.semantic(ns_u1, item_u1.embedding or [], limit=10)
        ids = {h.item.id for h in hits}
        assert item_u1.id in ids
        assert item_u2.id not in ids  # different user — never federated
    finally:
        from mu_engine.storage.mappers.qdrant_mapper import collection_name

        await qdrant_client.delete_collection(collection_name(ns_u1, VECTOR_DIM))


@pytest.mark.parametrize("session_scope", [None, "room-1", "some-other-session"])
async def test_shared_never_gets_truncated_match_value(
    make_ns: Callable[..., Namespace],
    session_scope: str | None,
) -> None:
    """AC-4.3-adjacent: a SHARED Namespace NEVER resolves to the truncated match value,
    for ANY session_scope — including the misconfiguration case (an operator passing
    session_scope=None, the PRIVATE-own federated default, on a SHARED room)."""
    ns = make_ns(visibility=Visibility.SHARED, session="room-1")
    key, value = _resolve_namespace_match(ns, session_scope=session_scope)
    assert key == "namespace"
    assert value == ns.to_prefix()  # always the exact, full, session-included prefix


@pytest.mark.parametrize("session_scope", [None, "session-a", "session-other"])
async def test_private_truncated_only_when_scope_is_none(
    make_ns: Callable[..., Namespace],
    session_scope: str | None,
) -> None:
    """PRIVATE resolves to the truncated user-prefix ONLY when session_scope is None;
    any concrete session_scope resolves back to a full, session-included match."""
    ns = make_ns(visibility=Visibility.PRIVATE, session="session-a")
    key, value = _resolve_namespace_match(ns, session_scope=session_scope)
    if session_scope is None:
        assert key == "namespace_user_prefix"
        assert value != ns.to_prefix()
        assert not value.endswith("session-a")
    else:
        assert key == "namespace"
        assert value == Namespace(
            org=ns.org,
            workspace=ns.workspace,
            user=ns.user,
            session=session_scope,
            visibility=ns.visibility,
        ).to_prefix()


async def test_shared_recall_unaffected_by_session_scope_none(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """End-to-end guard: a SHARED room recall with session_scope=None (the PRIVATE-own
    federated default) still isolates by room — no cross-room leak from the new default."""
    adapter = QdrantMtmAdapter(qdrant_client, dim=VECTOR_DIM)
    room_1 = make_ns(visibility=Visibility.SHARED, session="room-1")
    room_2 = make_ns(visibility=Visibility.SHARED, session="room-2")
    in_room_1 = make_item(room_1, "room 1 secret", authorized_ids=["p_alice"])
    in_room_2 = make_item(room_2, "room 2 secret", authorized_ids=["p_alice"])
    await adapter.upsert(in_room_1)
    await adapter.upsert(in_room_2)
    try:
        caller = frozenset({"p_alice"})
        hits = await adapter.semantic(
            room_1,
            in_room_1.embedding or [],
            limit=10,
            caller_identity_set=caller,
            session_scope=None,
        )
        ids = {h.item.id for h in hits}
        assert in_room_1.id in ids
        assert in_room_2.id not in ids  # SHARED room-as-wall holds regardless of session_scope
    finally:
        from mu_engine.storage.mappers.qdrant_mapper import collection_name

        await qdrant_client.delete_collection(collection_name(room_1, VECTOR_DIM))


async def test_upsert_stamps_indexed_user_prefix_payload_field(
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The truncated user-prefix is a real, retrievable, INDEXED payload field (not just an
    in-memory filter value) — verified via a server-side scroll filter on it directly."""
    from mu_engine.storage.mappers.qdrant_mapper import collection_name

    ns = make_ns(session="session-z")
    item = make_item(ns, "indexed field check")
    await mtm.upsert(item)
    try:
        name = collection_name(ns, VECTOR_DIM)
        _, user_prefix_value = _resolve_namespace_match(ns, session_scope=None)
        points, _ = await qdrant_client.scroll(
            collection_name=name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="namespace_user_prefix",
                        match=models.MatchValue(value=user_prefix_value),
                    )
                ]
            ),
            limit=10,
        )
        ids = {str(p.id) for p in points}
        from mu_engine.storage.mappers.qdrant_mapper import point_id

        assert point_id(item.id) in ids
    finally:
        await qdrant_client.delete_collection(collection_name(ns, VECTOR_DIM))
