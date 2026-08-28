"""``StmTierRepository.recent`` under Model-A — the port-level half of the AD-128 fix.

``ThreeChannelRecallRanker`` is not the only caller a shared STM window can ever have (AD-102 will
compose ``InProcessSharedRecall``, which ranks the SHARED η through the SAME ranker; the surface
façade and the promotion sweep read the same window). So the authorization is asserted HERE, at the
port, where it is a property of the tier rather than of one call site — the shape the last three
instances of this defect class all lacked (`7079ba8`: *"the safety property lived by CONVENTION in
three call sites of another package"*).

Also covers the WRITE half: §7.4's CONTRACT-TEST OBLIGATION 1 — *"no role id and no session id is
ever written into a point's `authorized_ids` at any stamp/sync/re-stamp site"* — now that ingest is
such a site.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.errors import CallerIdentitySetRequiredError, StampSubjectError
from mu_contracts.domain.model.authorized_ids import (
    AUTHORIZED_IDS_KEY,
    model_a_permits,
    stamp_of,
    validate_stamp_subjects,
)
from mu_engine.pipelines.concrete.ingest import IngestActivity, _build_memory_item
from mu_engine.storage.adapters.memory_stm import InMemoryStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_ROOM = Namespace.shared(org="acme", workspace="eng", session="standup")
_PRIVATE = Namespace(
    org="acme", workspace="eng", user="prn_alice", session="s1", visibility=Visibility.PRIVATE
)
_AT = datetime(2026, 8, 28, 9, 0, tzinfo=UTC)


def _item(content: str, *, ns: Namespace, authorized_ids: list[str] | None) -> MemoryItem:
    return MemoryItem(
        content=content,
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        state=MemoryState.ACTIVE,
        tier=MemoryTier.STM,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        created_at=_AT,
        updated_at=_AT,
        metadata={} if authorized_ids is None else {AUTHORIZED_IDS_KEY: authorized_ids},
    )


# --------------------------------------------------------------------------- the read half
async def test_recent_on_a_shared_namespace_returns_only_rows_the_caller_is_stamped_on() -> None:
    stm = InMemoryStmAdapter()
    await stm.put(_item("mine", ns=_ROOM, authorized_ids=["prn_alice"]))
    await stm.put(_item("theirs", ns=_ROOM, authorized_ids=["prn_bob"]))
    await stm.put(_item("ours", ns=_ROOM, authorized_ids=["prn_alice", "prn_bob"]))

    alice = await stm.recent(_ROOM, limit=10, caller_identity_set=frozenset({"prn_alice"}))
    bob = await stm.recent(_ROOM, limit=10, caller_identity_set=frozenset({"prn_bob"}))
    carol = await stm.recent(_ROOM, limit=10, caller_identity_set=frozenset({"prn_carol"}))

    assert sorted(s.item.content for s in alice) == ["mine", "ours"]
    assert sorted(s.item.content for s in bob) == ["ours", "theirs"]
    assert carol == []


async def test_recent_on_a_shared_namespace_with_no_caller_set_raises() -> None:
    stm = InMemoryStmAdapter()
    await stm.put(_item("mine", ns=_ROOM, authorized_ids=["prn_alice"]))

    with pytest.raises(CallerIdentitySetRequiredError):
        await stm.recent(_ROOM, limit=10)


async def test_recent_on_a_private_namespace_needs_no_caller_set_and_no_stamp() -> None:
    stm = InMemoryStmAdapter()
    await stm.put(_item("my own", ns=_PRIVATE, authorized_ids=None))

    window = await stm.recent(_PRIVATE, limit=10)

    assert [s.item.content for s in window] == ["my own"]


async def test_surviving_rows_are_rank_contiguous() -> None:
    """RRF fuses on POSITION; a gap in the ranks of a filtered window would encode how many rows
    the caller was denied, and would mis-rank the channel besides."""
    stm = InMemoryStmAdapter()
    await stm.put(_item("denied", ns=_ROOM, authorized_ids=["prn_bob"]))
    await stm.put(_item("allowed a", ns=_ROOM, authorized_ids=["prn_alice"]))
    await stm.put(_item("allowed b", ns=_ROOM, authorized_ids=["prn_alice"]))

    window = await stm.recent(_ROOM, limit=10, caller_identity_set=frozenset({"prn_alice"}))

    assert [s.rank for s in window] == [0, 1]


# --------------------------------------------------------------------------- the predicate
@pytest.mark.parametrize(
    ("stamp", "caller", "expected"),
    [
        (frozenset({"prn_a"}), frozenset({"prn_a"}), True),
        (frozenset({"prn_a", "prn_b"}), frozenset({"prn_b"}), True),
        (frozenset({"prn_a"}), frozenset({"prn_b"}), False),
        (frozenset(), frozenset({"prn_a"}), False),  # unstamped item: DENY
        (frozenset({"prn_a"}), frozenset(), False),  # caller resolved to nothing: DENY
        (frozenset({"prn_a"}), None, False),  # nobody resolved a caller: DENY
    ],
)
def test_model_a_permits_denies_every_ambiguous_case(
    stamp: frozenset[str], caller: frozenset[str] | None, expected: bool
) -> None:
    assert model_a_permits(stamp=stamp, caller_identity_set=caller) is expected


def test_a_corrupted_stamp_reads_as_empty_and_therefore_denies() -> None:
    """A bare string would otherwise iterate CHARACTERS, and `"prn_alice"` shares characters with
    almost any caller id — a stamp that decays into a wildcard is worse than no stamp."""
    assert stamp_of({AUTHORIZED_IDS_KEY: "prn_alice"}) == frozenset()
    assert stamp_of({AUTHORIZED_IDS_KEY: None}) == frozenset()
    assert stamp_of({}) == frozenset()
    assert stamp_of(None) == frozenset()


# --------------------------------------------------------------------------- the write half
@pytest.mark.parametrize(
    "subject", ["role_admins", "ses_standup", "sess_standup", "dev_laptop", "device_laptop"]
)
def test_a_role_session_or_device_id_is_refused_at_the_stamp_site(subject: str) -> None:
    """§7.4 CONTRACT-TEST OBLIGATION 1. A role/session token in a stamp is an offboarding hole;
    a device id is an ACL bypass (§7.11)."""
    with pytest.raises(StampSubjectError):
        validate_stamp_subjects([subject])


def test_a_separator_bearing_subject_is_refused() -> None:
    with pytest.raises(StampSubjectError):
        validate_stamp_subjects(["prn_a,prn_b"])


def test_ingest_stamps_the_roster_onto_the_item_it_mints() -> None:
    activity = IngestActivity(
        namespace=_ROOM,
        host="test",
        session_offset="off-1",
        text="the deploy window is friday 4pm",
        authorized_ids=frozenset({"prn_alice", "prn_bob"}),
    )

    item = _build_memory_item(activity, at=_AT)

    assert stamp_of(item.metadata) == frozenset({"prn_alice", "prn_bob"})


def test_a_private_activity_may_not_carry_a_stamp() -> None:
    """§7.4: a PRIVATE item is isolated by `to_prefix()` and never enters an ACL until published —
    a stamp there would be a second, contradicting answer to "who may read this"."""
    with pytest.raises(ValueError, match="SHARED-plane stamp"):
        IngestActivity(
            namespace=_PRIVATE,
            host="test",
            session_offset="off-1",
            text="mine",
            authorized_ids=frozenset({"prn_alice"}),
        )


def test_an_unstamped_shared_activity_writes_a_row_nobody_can_read() -> None:
    """The engine cannot see a roster, so it never invents one. What it guarantees instead is
    that the un-decided case is the CLOSED one."""
    activity = IngestActivity(
        namespace=_ROOM, host="test", session_offset="off-2", text="unstamped"
    )

    item = _build_memory_item(activity, at=_AT)

    assert stamp_of(item.metadata) == frozenset()
    assert not model_a_permits(
        stamp=stamp_of(item.metadata), caller_identity_set=frozenset({"prn_alice"})
    )


# ------------------------------------------------------------- the point-get half (AD-129)
async def test_get_on_a_shared_namespace_denies_a_row_the_caller_is_not_stamped_on() -> None:
    """The UNBOUNDED half of the leak. The window filter did not touch this arm.

    ``recent`` is capped by ``recency_floor_limit`` and ``stm_ttl_s``; a by-id read is capped by
    neither once an id is known — and the shared write's own ``201`` hands the id out. Measured:
    a principal who never joined a room ``GET``-ed a member's memory and received its content
    verbatim, and that read survived the ``recent`` fix untouched.

    **What breaks it:** deleting the ``authorized_item(...)`` wrapper from any STM adapter's
    ``get``.
    """
    stm = InMemoryStmAdapter()
    mine = _item("roster item", ns=_ROOM, authorized_ids=["prn_alice", "prn_bob"])
    theirs = _item("someone else's", ns=_ROOM, authorized_ids=["prn_alice"])
    await stm.put(mine)
    await stm.put(theirs)

    caller = frozenset({"prn_bob"})
    assert await stm.get(_ROOM, mine.id, caller_identity_set=caller) is not None
    assert await stm.get(_ROOM, theirs.id, caller_identity_set=caller) is None


async def test_get_denies_an_unstamped_shared_row_to_everyone() -> None:
    """Fail CLOSED. An unstamped row is one no governance decision was recorded for, and reading
    absence as "unrestricted" is the fail-open that keeps AD-128 alive after the caller set is
    threaded."""
    stm = InMemoryStmAdapter()
    orphan = _item("nobody authorized this", ns=_ROOM, authorized_ids=None)
    await stm.put(orphan)

    assert await stm.get(_ROOM, orphan.id, caller_identity_set=frozenset({"prn_alice"})) is None


async def test_get_on_a_shared_namespace_with_no_caller_set_raises() -> None:
    """``None`` is a wiring bug, not a denial — and it must not be able to look like a miss."""
    stm = InMemoryStmAdapter()
    item = _item("roster item", ns=_ROOM, authorized_ids=["prn_alice"])
    await stm.put(item)

    with pytest.raises(CallerIdentitySetRequiredError):
        await stm.get(_ROOM, item.id)


async def test_get_on_a_private_namespace_needs_no_caller_set_and_no_stamp() -> None:
    """§1 rule 5 / §7.4: the own-partition key IS the authorization; every PRIVATE caller of this
    port — the ingest pipeline's own recovery read included — passes ``None`` and must keep
    working."""
    stm = InMemoryStmAdapter()
    item = _item("my own note", ns=_PRIVATE, authorized_ids=None)
    await stm.put(item)

    assert await stm.get(_PRIVATE, item.id) is not None


async def test_get_of_an_absent_id_and_of_a_denied_id_are_indistinguishable() -> None:
    """Non-enumerating by construction: the caller supplies the id, so a denial that answered
    differently from a miss would turn this verb into a memory-existence oracle over the
    partition — the property ``RoomService._assert_member`` holds for rooms."""
    stm = InMemoryStmAdapter()
    theirs = _item("someone else's", ns=_ROOM, authorized_ids=["prn_alice"])
    await stm.put(theirs)

    caller = frozenset({"prn_carol"})
    assert await stm.get(_ROOM, theirs.id, caller_identity_set=caller) is None
    assert await stm.get(_ROOM, "mem_does_not_exist", caller_identity_set=caller) is None
