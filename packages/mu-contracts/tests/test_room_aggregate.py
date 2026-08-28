"""The room aggregate's PURE invariants (rooms-sessions-subscriptions-spec.md §2, spec:56-104).

No infrastructure, no clock but the one handed in — every assertion here is about the domain
object, which is the point of the aggregate being pure (spec:83).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from mu_contracts.domain.errors import SettingsValidationError
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.room import ParticipantKind
from mu_contracts.domain.model.session import (
    DuplicateParticipantError,
    FloorPolicyKind,
    Participant,
    ParticipantNotInRoomError,
    PresenceState,
    PrivateInRoomError,
    RoomCapacityError,
    RoomClosedError,
    RoomOwnerLease,
    RoomState,
    Session,
    SessionState,
    SharedSessionContext,
    canonical_dedupe_key,
    resolve_floor_policy,
)

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _room(**overrides: object) -> Session:
    base: dict[str, object] = {
        "id": "room-1",
        "org_id": "org-1",
        "workspace_id": "ws-1",
        "namespace_id": "ns-1",
        "created_at": NOW,
    }
    base.update(overrides)
    return Session(**base)  # type: ignore[arg-type]


def _human(principal_id: str) -> Participant:
    return Participant(
        principal_id=principal_id,
        kind=ParticipantKind.HUMAN,
        joined_at=NOW,
        owner_principal_id=principal_id,
    )


# ---- §2.2 roster invariants (spec:80) ---------------------------------------------------------


def test_join_adds_an_active_member() -> None:
    room = _room()
    room.join(_human("p1"), max_participants=10)
    assert [p.principal_id for p in room.active_participants()] == ["p1"]


def test_join_refuses_a_duplicate_active_member() -> None:
    room = _room()
    room.join(_human("p1"), max_participants=10)
    with pytest.raises(DuplicateParticipantError):
        room.join(_human("p1"), max_participants=10)


def test_join_refuses_a_closed_room() -> None:
    room = _room(state=RoomState.CLOSED)
    with pytest.raises(RoomClosedError):
        room.join(_human("p1"), max_participants=10)


def test_capacity_counts_active_members_only() -> None:
    """A room that has churned people through its seats is not full — capacity is a live count,
    not a lifetime quota (see ``Session.join``'s docstring)."""
    room = _room()
    room.join(_human("p1"), max_participants=1)
    room.leave("p1", at=NOW)
    room.join(_human("p2"), max_participants=1)  # must NOT raise
    with pytest.raises(RoomCapacityError):
        room.join(_human("p3"), max_participants=1)


def test_leave_keeps_the_row_for_provenance() -> None:
    room = _room()
    room.join(_human("p1"), max_participants=10)
    room.leave("p1", at=NOW)
    assert room.active_participants() == []
    assert len(room.participants) == 1
    assert room.participants[0].left_at == NOW
    assert room.participants[0].presence is PresenceState.OFFLINE


def test_leave_refuses_a_non_member() -> None:
    with pytest.raises(ParticipantNotInRoomError):
        _room().leave("ghost", at=NOW)


def test_agents_and_bound_agents_of() -> None:
    room = _room()
    room.join(_human("human-1"), max_participants=10)
    room.join(
        Participant(
            principal_id="agt_a",
            kind=ParticipantKind.BOUND_AGENT,
            joined_at=NOW,
            owner_principal_id="human-1",
            binding_id="bind-1",
        ),
        max_participants=10,
    )
    room.join(
        Participant(
            principal_id="agt_b",
            kind=ParticipantKind.SHARED_AGENT,
            joined_at=NOW,
            owner_principal_id="human-2",
        ),
        max_participants=10,
    )
    assert {p.principal_id for p in room.agents()} == {"agt_a", "agt_b"}
    assert [p.principal_id for p in room.bound_agents_of("human-1")] == ["agt_a"]
    assert room.bound_agents_of("human-2") == []


# ---- the write gate (spec:80) -----------------------------------------------------------------


def test_assert_can_post_gates_state_then_membership() -> None:
    room = _room()
    room.join(_human("p1"), max_participants=10)
    assert room.assert_can_post("p1").principal_id == "p1"

    with pytest.raises(ParticipantNotInRoomError):
        room.assert_can_post("p2")

    room.close()
    # Room state is checked BEFORE membership: a member of a closed room gets RoomClosed, not
    # ParticipantNotInRoom, so the caller can tell "the room ended" from "you are not in it".
    with pytest.raises(RoomClosedError):
        room.assert_can_post("p1")


def test_a_paused_room_refuses_posts() -> None:
    """§11's log-unavailable degrade (spec:381): post raises, reads still serve."""
    room = _room()
    room.join(_human("p1"), max_participants=10)
    room.pause()
    with pytest.raises(RoomClosedError):
        room.assert_can_post("p1")


# ---- §2.1 floor policy — FREE_FOR_ALL only, the rest RESERVED (spec:69) ------------------------


def test_free_for_all_is_the_only_resolvable_policy() -> None:
    assert resolve_floor_policy(FloorPolicyKind.FREE_FOR_ALL).kind is FloorPolicyKind.FREE_FOR_ALL


@pytest.mark.parametrize("reserved", [FloorPolicyKind.ROUND_ROBIN, FloorPolicyKind.MODERATED])
def test_a_reserved_policy_fails_loud_and_never_falls_back(reserved: FloorPolicyKind) -> None:
    """A RESERVED policy must NOT silently degrade to free-for-all — that would be an invisible
    turn-taking bypass no log line would ever mention."""
    with pytest.raises(SettingsValidationError):
        resolve_floor_policy(reserved)


def test_a_reserved_policy_is_refused_at_the_post_gate_too() -> None:
    room = _room(floor_policy=FloorPolicyKind.MODERATED)
    room.join(_human("p1"), max_participants=10)
    with pytest.raises(SettingsValidationError):
        room.assert_can_post("p1")


# ---- rooms are SHARED-only (CANONICAL §1 rule 4) ----------------------------------------------


def test_a_private_room_is_a_category_error() -> None:
    with pytest.raises(PrivateInRoomError):
        _room(visibility=Visibility.PRIVATE)


def test_shared_session_context_refuses_a_private_namespace() -> None:
    private = Namespace(
        org="org-1", workspace="ws-1", user="u1", session="room-1", visibility=Visibility.PRIVATE
    )
    with pytest.raises(PrivateInRoomError):
        SharedSessionContext(session_id="room-1", namespace=private)


def test_shared_session_context_namespace_must_be_this_room() -> None:
    other = Namespace.shared(org="org-1", workspace="ws-1", session="room-2")
    with pytest.raises(PrivateInRoomError):
        SharedSessionContext(session_id="room-1", namespace=other)
    ok = SharedSessionContext(
        session_id="room-1",
        namespace=Namespace.shared(org="org-1", workspace="ws-1", session="room-1"),
        head_seq=7,
    )
    assert ok.namespace.user == "*"  # the SHARED zeroing, CANONICAL §1 rule 4


# ---- §2.2 dedupe key (spec:77) ----------------------------------------------------------------


def test_dedupe_key_separates_the_same_text_in_reply_to_different_turns() -> None:
    """spec:77 puts ``reply_to`` in the key precisely so "ok" said twice, threaded differently,
    is two messages rather than one collapsed replay."""
    common = {"author_principal_id": "p1", "room_id": "room-1", "content_hash": "h"}
    assert canonical_dedupe_key(**common, reply_to_seq=1) != canonical_dedupe_key(
        **common, reply_to_seq=2
    )
    assert canonical_dedupe_key(**common, reply_to_seq=None) == canonical_dedupe_key(
        **common, reply_to_seq=None
    )


def test_dedupe_key_separates_authors_and_rooms() -> None:
    assert canonical_dedupe_key(
        author_principal_id="p1", room_id="r1", content_hash="h", reply_to_seq=None
    ) != canonical_dedupe_key(
        author_principal_id="p2", room_id="r1", content_hash="h", reply_to_seq=None
    )
    assert canonical_dedupe_key(
        author_principal_id="p1", room_id="r1", content_hash="h", reply_to_seq=None
    ) != canonical_dedupe_key(
        author_principal_id="p1", room_id="r2", content_hash="h", reply_to_seq=None
    )


def test_dedupe_key_components_cannot_be_shifted_across_the_separator() -> None:
    """The separator is a character the namespace validator forbids inside any component, so
    ``("ab","c")`` can never hash to the same key as ``("a","bc")``."""
    a = canonical_dedupe_key(
        author_principal_id="ab", room_id="c", content_hash="h", reply_to_seq=None
    )
    b = canonical_dedupe_key(
        author_principal_id="a", room_id="bc", content_hash="h", reply_to_seq=None
    )
    assert a != b


# ---- control-plane projection (spec:80/:83) ---------------------------------------------------


def test_paused_projects_to_open_not_closed() -> None:
    """Pausing is a RUNTIME degrade, not a control-plane close: projecting it to CLOSED would make
    a transient store outage look like end-of-session to authorization and billing."""
    room = _room()
    room.pause()
    assert room.to_definition().state is SessionState.OPEN
    room.close()
    assert room.to_definition().state is SessionState.CLOSED


def test_definition_round_trip_preserves_identity_and_roster() -> None:
    room = _room(floor_policy=FloorPolicyKind.FREE_FOR_ALL, last_seq=4)
    room.join(_human("p1"), max_participants=10)
    rebuilt = Session.from_definition(
        room.to_definition(),
        floor_policy=room.floor_policy,
        participants=room.participants,
        last_seq=room.last_seq,
    )
    assert (rebuilt.id, rebuilt.org_id, rebuilt.workspace_id, rebuilt.namespace_id) == (
        room.id,
        room.org_id,
        room.workspace_id,
        room.namespace_id,
    )
    assert rebuilt.last_seq == 4
    assert [p.principal_id for p in rebuilt.active_participants()] == ["p1"]


# ---- §4.2 the lease record (spec:215-222) -----------------------------------------------------


def test_lease_liveness_is_decided_by_the_record_not_by_the_key() -> None:
    lease = RoomOwnerLease(
        room_id="room-1",
        owner_instance_id="i1",
        token=3,
        expires_at=NOW + timedelta(seconds=15),
    )
    assert lease.is_live(at=NOW)
    assert not lease.is_live(at=NOW + timedelta(seconds=16))
    assert lease.published_through_seq == -1  # nothing published yet


def test_a_lease_token_is_never_zero() -> None:
    """Tokens start at 1 so that "no token" and "the first token" can never be confused by a
    falsiness check somewhere downstream."""
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        RoomOwnerLease(room_id="r", owner_instance_id="i", token=0, expires_at=NOW)


# ================================================================================================
# THE TWO CURSORS AND THE VERSION — added by the Phase-4 review-fix lane
# ================================================================================================


def test_the_announce_cursor_starts_behind_the_first_seq() -> None:
    """``announced_through_seq`` defaults to ``-1``, matching ``last_seq`` and an empty room's tail,
    so the very first message (``seq=0``) is unambiguously "not announced yet". A default of ``0``
    would make the first message of every room look announced and would silently reintroduce the
    permanent hole this field exists to close."""
    room = Session(
        id="r", org_id="o", workspace_id="w", namespace_id="n", created_at=datetime.now(UTC)
    )
    assert (room.last_seq, room.announced_through_seq, room.version) == (-1, -1, 0)
    assert room.announced_through_seq < 0


def test_the_cursors_and_the_version_cannot_be_negative_past_the_empty_sentinel() -> None:
    """``ge=-1`` on both cursors and ``ge=0`` on the version: a rewind past the empty-room sentinel
    is not a smaller number, it is a corrupted aggregate."""
    base = {
        "id": "r",
        "org_id": "o",
        "workspace_id": "w",
        "namespace_id": "n",
        "created_at": datetime.now(UTC),
    }
    with pytest.raises(ValidationError):
        Session(**base, announced_through_seq=-2)
    with pytest.raises(ValidationError):
        Session(**base, last_seq=-2)
    with pytest.raises(ValidationError):
        Session(**base, version=-1)


def _agent(principal_id: str, *, owner: str) -> Participant:
    return Participant(
        principal_id=principal_id,
        kind=ParticipantKind.BOUND_AGENT,
        joined_at=NOW,
        owner_principal_id=owner,
        binding_id=f"bind-{principal_id}",
    )


def test_leave_cascades_to_the_agents_the_departing_principal_owns() -> None:
    """**AD-133, measured through the production routes before this line existed.**

    Alice joins, enrols the agent she owns and drives, then leaves through
    ``POST /v1/rooms/{id}/leave``. ``leave`` stamped only HER row; ``is_active_member`` is
    ``left_at is None``; so the agent row stayed ACTIVE and every membership gate — the room
    verbs' ``_assert_member`` and the shared-memory roster gate alike — kept admitting an identity
    Alice still controls. She lost one of the two identities she holds: offboarding was HALVED,
    not severed, while CANONICAL §7.4's M15 bullet is categorical that *"a member who leaves loses
    read access to what was shared"*.

    **What breaks it:** deleting the ``for owned in self.participants`` loop from ``Session.leave``.
    """
    room = _room()
    room.join(_human("alice"), max_participants=10)
    room.join(_agent("agt_alice", owner="alice"), max_participants=10)
    room.join(_human("bob"), max_participants=10)
    room.join(_agent("agt_bob", owner="bob"), max_participants=10)

    room.leave("alice", at=NOW)

    assert [p.principal_id for p in room.active_participants()] == ["bob", "agt_bob"]
    departed = {p.principal_id: p for p in room.participants if not p.is_active_member}
    assert set(departed) == {"alice", "agt_alice"}
    assert departed["agt_alice"].left_at == NOW
    assert departed["agt_alice"].presence is PresenceState.OFFLINE


def test_leave_does_not_touch_agents_owned_by_anyone_else() -> None:
    """The cascade is by OWNERSHIP, not by kind — a shared agent another principal owns keeps its
    seat when an unrelated member departs. Without this the test above would pass against a
    ``leave`` that simply evicted every agent row."""
    room = _room()
    room.join(_human("alice"), max_participants=10)
    room.join(_agent("agt_bob", owner="bob"), max_participants=10)
    room.join(_human("bob"), max_participants=10)

    room.leave("alice", at=NOW)

    assert [p.principal_id for p in room.active_participants()] == ["agt_bob", "bob"]


def test_leave_returns_the_departing_row_not_a_cascaded_one() -> None:
    """The cascade is an EFFECT; the answer is still the caller's own row —
    ``RoomService.leave`` publishes ``ParticipantLeft`` from it."""
    room = _room()
    room.join(_human("alice"), max_participants=10)
    room.join(_agent("agt_alice", owner="alice"), max_participants=10)

    assert room.leave("alice", at=NOW).principal_id == "alice"


def test_a_departed_principal_cannot_leave_twice_even_though_it_owns_rows() -> None:
    """Idempotence direction: the second call refuses on the OWNER's row, before any cascade —
    a re-leave must not silently re-stamp ``left_at`` on rows and move the offboarding timestamp
    forward."""
    room = _room()
    room.join(_human("alice"), max_participants=10)
    room.join(_agent("agt_alice", owner="alice"), max_participants=10)
    room.leave("alice", at=NOW)

    with pytest.raises(ParticipantNotInRoomError):
        room.leave("alice", at=NOW + timedelta(minutes=5))
    assert {p.left_at for p in room.participants} == {NOW}
