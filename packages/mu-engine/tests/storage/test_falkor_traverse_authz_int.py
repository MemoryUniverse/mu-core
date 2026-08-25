"""C2 REGRESSION — the multi-hop traversal arm's authorization walls. REAL FalkorDB, ZERO mocks.

``GraphStorePort.traverse_entities`` DERIVES ``:Memory`` ids from the workspace-wide ``:Entity``
sub-graph (``_user_scope_prefix``, user slot ``*`` on SHARED — every room of a workspace shares
one entity graph AND one physical FalkorDB partition). Before the C2 fix the hydration that turned
those ids back into facts carried NEITHER a namespace predicate NOR an ``m.authorized_ids``
predicate, so a multi-hop recall returned SHARED facts from any room in the workspace and from any
ACL. That is strictly worse than the MTM leak fixed in ``7ccc405``: that one required the caller to
already KNOW a ``memory_id``; this arm PRODUCES the ids. ``RecallSettings.ltm_max_hops`` defaults
to 2, so the arm is ON by default and nothing downstream re-filters it.

The defect had two INDEPENDENT halves, so this file tests them separately:

1. ``test_traversal_hides_shared_fact_the_caller_is_not_authorized_for`` — the ACL half.
2. ``test_traversal_hides_shared_fact_from_a_different_room`` — the tenancy/room half.
3. ``test_traversal_still_returns_an_authorized_same_room_shared_fact`` — the NEGATIVE CONTROL,
   without which an implementation that returned nothing at all would pass (1) and (2) perfectly.
4. ``test_traversal_on_private_still_walks_across_the_users_own_sessions`` — the deliberate
   PRIVATE/SHARED asymmetry (BUG2's cross-session walk) must NOT be "fixed" away by a later
   reader: on PRIVATE another session is another conversation of the SAME user; on SHARED another
   session is another ROOM, and rooms are real walls.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB

from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.integration

_QUERY = "who is Bo's manager?"  # seeds on the entity "bo"


@pytest_asyncio.fixture
async def ltm(falkor_db: FalkorDB) -> AsyncIterator[FalkorLtmAdapter]:
    yield FalkorLtmAdapter(falkor_db)
    for g in await falkor_db.list_graphs():
        name = g.decode() if isinstance(g, bytes) else g
        if name.startswith("mu_g__"):
            with contextlib.suppress(Exception):  # best-effort teardown only
                await falkor_db.select_graph(name).delete()


async def test_traversal_hides_shared_fact_the_caller_is_not_authorized_for(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """THE ACL HALF. Same room, same entity graph — the walk legitimately REACHES the fact — but
    the caller is not in its ``authorized_ids``. Nothing downstream of this arm re-checks the ACL,
    so this assertion is the only guard that exists."""
    room = make_ns(visibility=Visibility.SHARED, session="roomA")
    secret = make_item(
        room,
        "Ada manages Bo",
        subject="Ada",
        predicate="manages",
        obj="Bo",
        authorized_ids=["principal-alice"],
    )
    await ltm.upsert_fact(secret)

    # Sanity: the ENTITY WALK genuinely reaches this fact for an authorized caller (otherwise
    # the assertion below would pass for the wrong reason — see the negative control too).
    authorized = await ltm.traverse_entities(
        room, query=_QUERY, max_hops=2, limit=10, caller_identity_set=frozenset({"principal-alice"})
    )
    assert [h.item.id for h in authorized] == [secret.id], "precondition: the walk must reach it"

    intruder = await ltm.traverse_entities(
        room,
        query=_QUERY,
        max_hops=2,
        limit=10,
        caller_identity_set=frozenset({"principal-mallory"}),
    )
    assert intruder == [], (
        "AUTHZ BYPASS: a SHARED fact whose authorized_ids does NOT include the caller came back "
        "from a multi-hop traversal — the hydration is missing the m.authorized_ids predicate"
    )


async def test_traversal_hides_shared_fact_from_a_different_room(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """THE TENANCY HALF. Both rooms are the SAME workspace, so they share one physical FalkorDB
    graph AND one ``:Entity`` sub-graph (user slot ``*``) — the walk reaches roomA's fact from
    roomB. Only the ``:Memory`` namespace predicate stops it, and on SHARED that predicate is the
    room-included ``to_prefix()``, UNCONDITIONALLY."""
    room_a = make_ns(visibility=Visibility.SHARED, session="roomA")
    room_b = make_ns(visibility=Visibility.SHARED, session="roomB")
    assert ltm.graph_name_for(room_a) == ltm.graph_name_for(
        room_b
    ), "precondition: both rooms must share ONE physical partition, else this test is vacuous"

    other_rooms_fact = make_item(
        room_a,
        "Ada manages Bo",
        subject="Ada",
        predicate="manages",
        obj="Bo",
        # The SAME caller is authorized — isolating the ROOM wall from the ACL wall.
        authorized_ids=["principal-alice"],
    )
    await ltm.upsert_fact(other_rooms_fact)

    caller = frozenset({"principal-alice"})
    from_a = await ltm.traverse_entities(
        room_a, query=_QUERY, max_hops=2, limit=10, caller_identity_set=caller
    )
    assert [h.item.id for h in from_a] == [
        other_rooms_fact.id
    ], "precondition: own room must see it"

    from_b = await ltm.traverse_entities(
        room_b, query=_QUERY, max_hops=2, limit=10, caller_identity_set=caller
    )
    assert from_b == [], (
        "ROOM LEAK: a SHARED fact captured in a DIFFERENT room came back from a traversal issued "
        "in another room — rooms are real walls; SHARED never gets the PRIVATE cross-session relax"
    )


async def test_traversal_still_returns_an_authorized_same_room_shared_fact(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """NEGATIVE CONTROL. Without this, an implementation that simply returned ``[]`` for every
    SHARED traversal would pass both leak tests above perfectly. A 2-hop walk (Ada -> Bo -> Cy)
    proves the arm is still doing real multi-hop work, not just a 1-hop degenerate case."""
    room = make_ns(visibility=Visibility.SHARED, session="roomA")
    caller_id = "principal-alice"
    one_hop = make_item(
        room,
        "Ada manages Bo",
        subject="Ada",
        predicate="manages",
        obj="Bo",
        authorized_ids=[caller_id, "principal-bob"],
    )
    two_hop = make_item(
        room,
        "Bo manages Cy",
        subject="Bo",
        predicate="manages",
        obj="Cy",
        authorized_ids=[caller_id],
    )
    await ltm.upsert_fact(one_hop)
    await ltm.upsert_fact(two_hop)

    hits = await ltm.traverse_entities(
        room,
        query="who does Ada manage?",
        max_hops=2,
        limit=10,
        caller_identity_set=frozenset({caller_id}),
    )
    ids = {h.item.id for h in hits}
    assert one_hop.id in ids, "the 1-hop authorized same-room fact must still be returned"
    assert two_hop.id in ids, "the 2-hop authorized same-room fact must still be returned"
    assert {h.item.content for h in hits} == {"Ada manages Bo", "Bo manages Cy"}


async def test_traversal_on_private_still_walks_across_the_users_own_sessions(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """THE ASYMMETRY IS DELIBERATE — do not "fix" it. On PRIVATE, a different session is another
    of the SAME user's own conversations, and the entity sub-graph is user-level precisely so a
    relational question asked in session B can walk edges materialized in session A (the BUG2
    scoping fix). The C2 room predicate must NOT narrow this arm; it only closes SHARED.

    The cross-USER wall is asserted too, so "cross-session is allowed" never widens into
    "cross-user is allowed"."""
    sess_a = make_ns(user="u1", session="sessionA")
    sess_b = make_ns(user="u1", session="sessionB")
    intruder_ns = make_ns(user="intruder", session="sessionA")

    fact = make_item(sess_a, "Ada manages Bo", subject="Ada", predicate="manages", obj="Bo")
    await ltm.upsert_fact(fact)

    from_b = await ltm.traverse_entities(sess_b, query=_QUERY, max_hops=2, limit=10)
    assert [h.item.id for h in from_b] == [fact.id], (
        "the PRIVATE cross-session walk regressed — a relational query from another of the "
        "user's OWN sessions must still hydrate the fact (BUG2 scoping fix)"
    )

    from_intruder = await ltm.traverse_entities(intruder_ns, query=_QUERY, max_hops=2, limit=10)
    assert from_intruder == [], "cross-USER leak — PRIVATE federation widened past the user wall"
