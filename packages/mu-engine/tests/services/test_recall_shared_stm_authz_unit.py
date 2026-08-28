"""AD-128 regression — a non-member of a room must not read the room through the STM floor.

The measured defect (real Valkey/Qdrant/FalkorDB, production route): `carol`, authenticated in the
org and never on room `standup`'s roster, POSTed `/v1/memories/recall` naming that session and got
back **2 of the room's shared items**, every one of them `tier=stm channel=stm floor=True`. The MTM
and LTM arms of ``ThreeChannelRecallRanker`` were both Model-A filtered; the STM recency-floor arm
was not — and could not be, because ``StmTierRepository.recent`` had no parameter for the caller
identity set.

These tests assert the CONSEQUENCE, in both directions, because an empty result that is empty for
the wrong reason proves nothing:

* the non-member gets NOTHING of the room (the fix), and
* the member still gets EVERYTHING of the room (the fix is a filter, not an off switch).

Real ``InMemoryStmAdapter`` — the shipped implementation of the port under test, not a fake of it
(a fake that authorized more than the store would hide exactly this defect). The MTM/LTM arms are
inert stubs: this is a unit test of the STM arm's authorization, and those two arms are already
covered by ``test_recall_ranker_unit.py`` and ``test_falkor_traverse_authz_int.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.errors import CallerIdentitySetRequiredError
from mu_contracts.domain.model.authorized_ids import AUTHORIZED_IDS_KEY
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker
from mu_engine.storage.adapters.memory_stm import InMemoryStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import Scored

pytestmark = pytest.mark.unit

ALICE = "prn_alice"
BOB = "prn_bob"
CAROL = "prn_carol"

_ROOM = Namespace.shared(org="acme", workspace="eng", session="standup")
_PRIVATE = Namespace(
    org="acme", workspace="eng", user=ALICE, session="s1", visibility=Visibility.PRIVATE
)
_AT = datetime(2026, 8, 28, 9, 0, tzinfo=UTC)


def _item(
    content: str, *, ns: Namespace, authorized_ids: list[str] | None, seconds: int
) -> MemoryItem:
    at = _AT + timedelta(seconds=seconds)
    metadata = {} if authorized_ids is None else {AUTHORIZED_IDS_KEY: authorized_ids}
    return MemoryItem(
        content=content,
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        state=MemoryState.ACTIVE,
        tier=MemoryTier.STM,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        created_at=at,
        updated_at=at,
        metadata=metadata,
    )


class _EmptyMtm:
    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: object | None = None,
    ) -> list[Scored[MemoryItem]]:
        return []


class _EmptyLtm:
    async def graph_recall(
        self, ns: Namespace, *, limit: int, caller_identity_set: frozenset[str] | None = None
    ) -> list[Scored[MemoryItem]]:
        return []

    async def traverse_entities(
        self,
        ns: Namespace,
        *,
        query: str,
        max_hops: int,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        return []


def _ranker(stm: InMemoryStmAdapter) -> ThreeChannelRecallRanker:
    return ThreeChannelRecallRanker(
        stm=stm,
        mtm=_EmptyMtm(),  # type: ignore[arg-type]
        ltm=_EmptyLtm(),  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        # "recency" scoring keeps this test about AUTHORIZATION only: no embedder, no relevance
        # ordering, so a missing item can only ever be a missing AUTHORIZATION.
        settings=RecallSettings(stm_scoring="recency"),
        clock=FrozenClock(_AT),
    )


async def _room_with_two_shared_memories() -> InMemoryStmAdapter:
    """The demo's exact setup: alice and bob write into room `standup`, roster = {alice, bob}."""
    stm = InMemoryStmAdapter()
    await stm.put(
        _item("the deploy window is friday 4pm", ns=_ROOM, authorized_ids=[ALICE, BOB], seconds=0)
    )
    await stm.put(
        _item(
            "the rollback runbook lives in ops/runbook.md",
            ns=_ROOM,
            authorized_ids=[ALICE, BOB],
            seconds=1,
        )
    )
    return stm


async def _recall(stm: InMemoryStmAdapter, caller: frozenset[str] | None) -> list[str]:
    result = await _ranker(stm).rank(
        _ROOM,
        "deploy",
        [0.0],
        limit=10,
        channels=RecallChannels(stm=True, mtm=False, ltm=False),
        caller_identity_set=caller,
    )
    return [view.content for view in result.items]


async def test_a_non_member_recalls_nothing_of_the_room() -> None:
    """THE DEFECT. carol is on no roster, so no row's `authorized_ids` contains her."""
    stm = await _room_with_two_shared_memories()

    assert await _recall(stm, frozenset({CAROL})) == []


async def test_a_member_still_recalls_everything_of_the_room() -> None:
    """THE POSITIVE CONTROL — the fix must be a filter, not an off switch.

    If this passes only because the room is empty, the negative test above proves nothing; both
    read the SAME two rows out of the SAME adapter.
    """
    stm = await _room_with_two_shared_memories()

    assert sorted(await _recall(stm, frozenset({ALICE}))) == [
        "the deploy window is friday 4pm",
        "the rollback runbook lives in ops/runbook.md",
    ]
    assert len(await _recall(stm, frozenset({BOB}))) == 2


async def test_a_departed_member_recalls_nothing_once_the_stamp_drops_them() -> None:
    """§7.4 offboarding: removal severs by re-stamp, and recall stops returning the point."""
    stm = InMemoryStmAdapter()
    await stm.put(_item("bob was here", ns=_ROOM, authorized_ids=[ALICE], seconds=0))

    assert await _recall(stm, frozenset({BOB})) == []
    assert await _recall(stm, frozenset({ALICE})) == ["bob was here"]


async def test_an_unstamped_shared_row_is_denied_to_everyone() -> None:
    """Fail CLOSED. An unstamped SHARED row is one no governance decision was recorded for.

    Reading absence as "unrestricted" is the fail-open that made AD-128 exploitable in the first
    place — every row in the measured leak was unstamped.
    """
    stm = InMemoryStmAdapter()
    await stm.put(_item("nobody authorized this", ns=_ROOM, authorized_ids=None, seconds=0))

    assert await _recall(stm, frozenset({ALICE})) == []
    assert await _recall(stm, frozenset({CAROL})) == []


async def test_an_empty_caller_set_authorizes_nothing() -> None:
    """`RecallService` coerces a missing SHARED caller set to `frozenset()` deliberately ("the
    safe direction, never an over-broad match"). Assert that direction is real here."""
    stm = await _room_with_two_shared_memories()

    assert await _recall(stm, frozenset()) == []


async def test_a_shared_rank_with_no_caller_set_refuses_rather_than_serving() -> None:
    """`None` on a SHARED η is a WIRING bug, and every tier adapter reads it as "omit the Model-A
    clause". Refused at the ranker so no arm ever runs unfiltered — and refused LOUDLY, because a
    `[]` here would look exactly like an empty room."""
    stm = await _room_with_two_shared_memories()

    with pytest.raises(CallerIdentitySetRequiredError):
        await _recall(stm, None)


async def test_private_recall_is_untouched_by_model_a() -> None:
    """§1 rule 5 / §7.4: the own partition is authorized by its `to_prefix()` key, `None` is the
    correct caller set there, and no stamp is required. The fix must not narrow the private plane
    (which is every mu-client user's entire memory)."""
    stm = InMemoryStmAdapter()
    await stm.put(_item("my own private fact", ns=_PRIVATE, authorized_ids=None, seconds=0))

    result = await _ranker(stm).rank(
        _PRIVATE,
        "fact",
        [0.0],
        limit=10,
        channels=RecallChannels(stm=True, mtm=False, ltm=False),
        caller_identity_set=None,
    )

    assert [view.content for view in result.items] == ["my own private fact"]
