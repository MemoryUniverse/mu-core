"""Session / Participant / NamespaceDefinition — the control-plane collaboration objects,
**plus the ROOM RUNTIME aggregate** (``rooms-sessions-subscriptions-spec.md`` §2.1/§2.2/§2.3).

Authority: hackathon ``shared/scope.py:78-106`` (ported shape), storage-schema §2.1
(``sessions`` / ``session_participants`` — the table that PRODUCES ``authorized_ids``,
CANONICAL §7.4 Model A), re-scoped to the un-collapsed org/workspace (§0.1).

``SessionParticipant.left_at`` set = the offboarding re-stamp trigger (CANONICAL §7.4/M15).

--------------------------------------------------------------------------------------------
ROOMS (added by the mu-server Phase 4 foundation lane)
--------------------------------------------------------------------------------------------

``rooms-sessions-subscriptions-spec.md:59`` heads the room enum block
``mu_core/domain/model/session.py`` and ``:74-82`` puts ``Participant``/``Addressing``/``Session``
under that same header, so they live HERE. ``ParticipantKind``/``MessageKind``/``RoomMessage``
already ship next door in ``domain/model/room.py`` and are IMPORTED rather than redeclared — one
vocabulary, never two.

**``Session`` WRAPS ``SessionDefinition``, it does not replace it** (spec:83). ``SessionDefinition``
stays the control-plane value (authorization invariants, ``scope.py:135 assert_authorized``);
``Session`` adds the RUNTIME invariants — roster, room state, floor — and is PURE: every method
here is synchronous, touches no store and no clock it was not handed.

⚠ **THREE DELIBERATE DEVIATIONS from the spec text, each load-bearing:**

1. **``Session`` carries ``org_id``, which spec:79's field list omits.** ADR 0026 un-collapsed org
   from workspace (``CANONICAL-CONTRACTS.md:73``) and every shipped control-plane object carries it
   (``SessionDefinition.org_id`` below; ``sessions.org_id``; ``ClientScope.org_id``). Without it
   ``to_definition()``/``from_definition()`` cannot round-trip and the aggregate cannot be
   tenancy-scoped at all. The spec is pre-ADR-0026 here.
2. **The room error hierarchy is declared in THIS module, and its real home is
   ``mu_contracts/domain/errors.py``** (spec:379). It is here only because the lane that built it
   did not own ``errors.py``; the classes are otherwise exactly §11's. Moving them there and
   re-exporting from here is a mechanical follow-up — see ``ROOM_ERRORS_BELONG_IN`` below.
3. **``FloorPolicyKind`` ships all three members and only ``FREE_FOR_ALL`` has a strategy.**
   ROUND_ROBIN and MODERATED are RESERVED wire vocabulary (spec:69, ``MU-SERVER-BUILD-PLAN.md:29``
   "the enum keeps the other members RESERVED"): selecting one raises at resolve time. There is no
   stub and no ``NotImplementedError`` body — absence is the house rule.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mu_contracts.domain.errors import MemoryUniverseError, SettingsValidationError
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.room import ParticipantKind, RoomMessage

__all__ = [
    "ROOM_ERRORS_BELONG_IN",
    "Addressing",
    "AddressingNotStorableError",
    "DuplicateParticipantError",
    "FloorDeniedError",
    "FloorPolicy",
    "FloorPolicyKind",
    "FreeForAll",
    "MessageTooLongError",
    "NamespaceDefinition",
    "Participant",
    "ParticipantNotInRoomError",
    "PresenceState",
    "PrivateInRoomError",
    "RoomAlreadyExistsError",
    "RoomCapacityError",
    "RoomClosedError",
    "RoomDuplicateError",
    "RoomError",
    "RoomLogConflictError",
    "RoomOwnerLease",
    "RoomState",
    "RoomVersionConflictError",
    "Session",
    "SessionDefinition",
    "SessionParticipant",
    "SessionState",
    "SharedSessionContext",
    "StaleRoomOwnerError",
    "canonical_dedupe_key",
    "resolve_floor_policy",
]


class SessionState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class NamespaceDefinition(BaseModel):
    """A named collaboration grouping inside one workspace (realizes ``namespaces``,
    storage-schema §2.1). Its ``id`` is the η ``.workspace`` grouping component."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    owner_principal_id: str = Field(min_length=1)
    display_name: str | None = Field(default=None, max_length=200)
    allowed_visibilities: frozenset[Visibility] = Field(
        default_factory=lambda: frozenset(Visibility)
    )


class SessionDefinition(BaseModel):
    """One concurrent session under a concrete namespace (realizes ``sessions``,
    storage-schema §2.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    namespace_id: str = Field(min_length=1)
    state: SessionState = SessionState.OPEN
    expires_at: datetime | None = None
    created_at: datetime

    def is_active(self, *, at: datetime | None = None) -> bool:
        now = at or datetime.now(UTC)
        expiry = self.expires_at
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return self.state is SessionState.OPEN and (expiry is None or expiry > now)


class SessionParticipant(BaseModel):
    """A principal's membership in a session — a PRINCIPAL id (Model A, never a role/session
    token). ``left_at`` set triggers the offboarding re-stamp (CANONICAL §7.4/M15)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    joined_at: datetime
    left_at: datetime | None = None


# ==============================================================================================
# ROOMS §2.1 — enums (rooms-sessions-subscriptions-spec.md:56-70)
# ==============================================================================================
#: Where §11 (``rooms-sessions-subscriptions-spec.md:379``) says the room error hierarchy belongs.
#: Stated as a constant rather than only in prose so the move is greppable and so a reader who
#: finds ``RoomClosedError`` here knows immediately that this is a temporary home, not a design.
ROOM_ERRORS_BELONG_IN: Final = "mu_contracts.domain.errors"


class PresenceState(StrEnum):
    """Derived from heartbeats by the ``PresenceTracker`` (spec:65). Presence is ADVISORY and is
    never an authorization input (spec:390) — a room read is authorized by η + Model A, never by
    whether someone looks online."""

    ACTIVE = "active"
    IDLE = "idle"
    AWAY = "away"
    OFFLINE = "offline"


class RoomState(StrEnum):
    """spec:66. ``PAUSED`` is NOT decorative: §11's degrade table (spec:381) makes it the state a
    room enters when its log is unavailable — ``post`` raises, reads still serve. ``SessionState``
    (above) has only OPEN/CLOSED and is the CONTROL-plane state; the two are different axes and
    ``Session.to_definition()`` projects PAUSED down to OPEN because a paused room is still a live
    control-plane session."""

    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"


class FloorPolicyKind(StrEnum):
    """spec:69. ``FREE_FOR_ALL`` is the only member with a strategy — see
    :func:`resolve_floor_policy`."""

    FREE_FOR_ALL = "free_for_all"
    ROUND_ROBIN = "round_robin"
    MODERATED = "moderated"


#: The floor policies this build actually implements. ROUND_ROBIN/MODERATED are RESERVED: they are
#: wire vocabulary (a peer or a later release may name them) and selecting one FAILS LOUD rather
#: than silently degrading to free-for-all, which would be an invisible turn-taking bypass.
IMPLEMENTED_FLOOR_POLICIES: Final = frozenset({FloorPolicyKind.FREE_FOR_ALL})


# ==============================================================================================
# ROOMS §11 — the error hierarchy (spec:379). TEMPORARY HOME — see ROOM_ERRORS_BELONG_IN.
# ==============================================================================================
class RoomError(MemoryUniverseError):
    """Root of the room runtime's typed failures (spec:379)."""


class RoomClosedError(RoomError):
    """The room is CLOSED (or PAUSED) and refuses the mutation. HTTP 409/503 (spec:394)."""


class RoomCapacityError(RoomError):
    """``RoomSettings.max_participants`` reached (spec:349)."""


class DuplicateParticipantError(RoomError):
    """The principal is already an ACTIVE member. Re-join after a ``leave`` is legal and appends a
    NEW row — the old one is retained for provenance (spec:80)."""


class ParticipantNotInRoomError(RoomError):
    """The principal has no active roster row. Non-enumerating at the edge (404, spec:394)."""


class FloorDeniedError(RoomError):
    """The floor policy refused this principal's turn (403, spec:394)."""


class PrivateInRoomError(RoomError):
    """A PRIVATE-visibility scope or body reached the room write path. Raised in §4.1 stage 1,
    BEFORE any store is touched (spec:203) — rooms are SHARED-only, and this is the check that
    keeps the shared plane's privacy invariant true at the room's front door."""


class MessageTooLongError(RoomError):
    """The body is empty, or exceeds ``RoomSettings.max_message_chars`` (spec:349). HTTP 422.

    ⚠ **NOT in spec §11's list — this lane added it.** spec:75 bounds the field as
    ``body: str(1..32_000)`` and spec:349 makes the ceiling configurable, but §11 names no error
    for violating it, so the bound would otherwise be enforced by whatever each caller happened to
    raise. It belongs in §11 alongside :class:`StaleRoomOwnerError`."""


class RoomLogConflictError(RoomError):
    """``expected_seq`` did not match the log's tail — a peer appended first. RETRYABLE: §4.1
    stage 4 re-reads ``tail_seq`` and retries up to ``append_retry_max`` (spec:207)."""


class RoomDuplicateError(RoomError):
    """The same ``dedupe_key`` was already appended. Carries the STORED message so §4.1 stage 4 can
    return it idempotently (spec:207) rather than re-deriving it from a second read."""

    def __init__(self, stored: RoomMessage) -> None:
        super().__init__(f"duplicate room message at seq={stored.seq}")
        self.stored = stored


class RoomAlreadyExistsError(RoomError):
    """``open_room`` was called for a room id that already exists. HTTP 409.

    ⚠ **NOT in spec §11's list — this lane added it, and it needs to go into §11.** The spec
    describes ``open_room`` as if the room could not already be there; without this error the verb
    is a whole-aggregate REPLACE, so a second call ejects the entire roster, resets ``last_seq``
    to ``-1`` while the durable log keeps its rows, re-opens a CLOSED room and destroys the
    departed rows CANONICAL §7.4/M15's offboarding re-stamp depends on. Refusing is the only
    behaviour that cannot silently destroy state; an idempotent re-open belongs with the REST edge
    that has an ``Idempotency-Key`` to decide it with (spec:325)."""


class RoomVersionConflictError(RoomError):
    """A whole-aggregate ``SessionRepository.save`` lost a race — the stored ``Session.version``
    moved under the caller. RETRYABLE: reload, re-apply, save again.

    ⚠ **NOT in spec §11's list — this lane added it.** spec:132 mandates whole-aggregate save; it
    does not mandate a lost-update window, and without a version a concurrent ``join`` and
    ``leave`` are a read-modify-write race in which one of them silently disappears. The careful
    ``expected_seq`` precondition on the ``seq`` path (spec:136) had no counterpart here; this is
    it."""


class AddressingNotStorableError(RoomError):
    """A caller supplied :class:`Addressing` this build cannot persist. HTTP 422.

    ⚠ **This error exists because of a REPORTED gap, not a design choice.** spec:76 lists
    ``addressing: Addressing`` as a ``RoomMessage`` field; the shipped ``RoomMessage``
    (``mu_contracts/domain/model/room.py:36-51``) has no such field and lives in a module this lane
    does not own, so a directed message's ``to_principal_ids`` has nowhere durable to go. Accepting
    one and dropping it would tell the caller a message was delivered to a named recipient when
    nothing recorded who that was — and spec:280 makes the roster/addressing surface the thing an
    agent principal is stamped from. So it is REFUSED until the field lands, rather than accepted
    and lost. ``reply_to_seq`` IS honoured: it is part of spec:77's dedupe key, which is durable.
    """


class StaleRoomOwnerError(RoomError):
    """A fencing token older than the room's current owner token tried to advance owner state.

    ⚠ **NOT in spec §11's list — this lane added it, and it needs to go into §11.** spec:216 pins
    the REQUIREMENT ("a stale owner's store write is *rejected at the store*, not merely
    lease-expired") but names no error for it, and a requirement with no typed failure is a
    requirement that gets implemented as a silent no-op. See :class:`RoomOwnerLease`."""


# ==============================================================================================
# ROOMS §2.2 — value objects (spec:74-82)
# ==============================================================================================
class Addressing(BaseModel):
    """Who a message is for (spec:78). Empty ``to_principal_ids`` = broadcast.

    Addressing is NOT authorization: everyone in the room reads every message (one room = one
    shared partition, spec:268). It selects who is DISPATCHED to, and ``reply_to_seq`` threads the
    conversation — which is why it is part of the dedupe key (spec:77): the same sentence said
    twice in reply to two different turns is two different messages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    to_principal_ids: tuple[str, ...] = ()
    reply_to_seq: int | None = Field(default=None, ge=0)


def canonical_dedupe_key(
    *,
    author_principal_id: str,
    room_id: str,
    content_hash: str,
    reply_to_seq: int | None,
) -> str:
    """spec:77 — ``author + room + content_hash + reply_to``, as a stable hex digest.

    A FUNCTION rather than ``RoomMessage.canonical_dedupe_key()`` (which is where spec:77 puts it)
    because ``RoomMessage`` ships in ``domain/model/room.py``, which this lane does not own — see
    the module docstring. The inputs are exactly §11's four, the separator is a character the
    namespace validator already forbids inside any component, and the digest is content-free: it
    is built from a CONTENT HASH, never from the body.
    """
    material = "\x1f".join(
        (
            author_principal_id,
            room_id,
            content_hash,
            "" if reply_to_seq is None else str(reply_to_seq),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class Participant(BaseModel):
    """A roster row (spec:74). MUTABLE — ``presence`` and ``left_at`` change in place, which is
    exactly why ``Session`` is an aggregate and not a frozen value.

    ``principal_id == agent_principal_id`` for an agent (spec:74): an agent is a first-class
    PRINCIPAL hung off ``owner_principal_id``, carried on ``ClientScope``, and **never on η**
    (CANONICAL §1 rule 2). Do not add an ``agent`` axis anywhere downstream of this field.
    """

    model_config = ConfigDict(extra="forbid")

    principal_id: str = Field(min_length=1)
    kind: ParticipantKind
    display_name: str | None = Field(default=None, max_length=200)
    joined_at: datetime
    left_at: datetime | None = None
    presence: PresenceState = PresenceState.OFFLINE
    #: The human (or parent agent) that owns this participant. Set for agents; for a HUMAN it is
    #: the principal itself, which is the "one-model rule" of spec:243 — "human writes directly"
    #: and "agent writes" are the same code path.
    owner_principal_id: str = Field(min_length=1)
    #: Authorizes a BOUND_AGENT (``scope.py:186`` ``AgentBinding``). ``None`` for everyone else.
    binding_id: str | None = None
    capabilities: frozenset[str] = frozenset()

    @property
    def is_agent(self) -> bool:
        return self.kind in (ParticipantKind.SHARED_AGENT, ParticipantKind.BOUND_AGENT)

    @property
    def is_active_member(self) -> bool:
        return self.left_at is None


class FloorPolicy(Protocol):
    """Turn-taking strategy (spec:37 — domain, app-singleton, stateless)."""

    kind: FloorPolicyKind

    def assert_may_post(self, session: Session, principal_id: str) -> None:
        """Raise :class:`FloorDeniedError` when this principal may not take the floor NOW.

        Membership and room state are already checked by :meth:`Session.assert_can_post` before
        this runs, so a policy only decides TURN-TAKING and never re-decides authorization.
        """
        ...


class FreeForAll:
    """The one implemented floor policy (spec:69/:355). Anyone in the room may post at any time.

    It denies nothing, and that is the honest shape of "free for all" — not a missing check. The
    checks that DO run (open room, active membership) are ``Session.assert_can_post``'s, upstream
    of here, so removing this class would not remove a gate; it would remove the seam at which
    ROUND_ROBIN and MODERATED attach.
    """

    kind: FloorPolicyKind = FloorPolicyKind.FREE_FOR_ALL

    def assert_may_post(self, session: Session, principal_id: str) -> None:
        del session, principal_id  # free-for-all: the floor is never contended


#: The registry, deliberately a plain mapping with ONE entry rather than a decorator-registered
#: table: a registry with an extension mechanism and no second entry is machinery pretending to be
#: a design.
_FLOOR_POLICIES: Final[dict[FloorPolicyKind, FloorPolicy]] = {
    FloorPolicyKind.FREE_FOR_ALL: FreeForAll(),
}


def resolve_floor_policy(kind: FloorPolicyKind) -> FloorPolicy:
    """Strategy lookup — FAIL LOUD on a RESERVED member (spec:69).

    A RESERVED policy MUST NOT fall back to free-for-all: an operator who configures
    ``moderated`` and silently gets a free-for-all room has a turn-taking bypass that no log line
    will ever mention. Raising here means the failure surfaces at settings-load / room-open, which
    is the only place it is cheap.
    """
    policy = _FLOOR_POLICIES.get(kind)
    if policy is None:
        raise SettingsValidationError(
            f"floor policy {kind.value!r} is RESERVED — implemented: "
            + ", ".join(sorted(k.value for k in IMPLEMENTED_FLOOR_POLICIES))
        )
    return policy


# ==============================================================================================
# ROOMS §2.2 — the aggregate (spec:79-83)
# ==============================================================================================
class Session(BaseModel):
    """The room aggregate root: durable identity + roster + floor invariants, **PURE**.

    Every method below is synchronous and side-effect-free apart from mutating this object. All
    durable truth lives in the repositories (``SessionRepository``/``RoomLogRepository``); this
    object rebuilds from them after a crash (spec:198). ``last_seq`` is a HINT — the log is the
    ordering authority and the only thing entitled to answer "what is the tail" (spec:82).

    **``visibility`` is SHARED and cannot be anything else.** A room IS the shared partition
    (``Namespace.shared(...)``, ``user='*'`` — CANONICAL §1 rule 4), so a PRIVATE room is not a
    narrower room, it is a category error. The validator refuses it at construction rather than
    letting §4.1 stage 1 be the only thing standing between a private body and the shared plane.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)  # == room_id
    org_id: str = Field(min_length=1)  # ADR 0026 — see the module docstring, deviation 1
    workspace_id: str = Field(min_length=1)
    namespace_id: str = Field(min_length=1)
    visibility: Visibility = Visibility.SHARED
    state: RoomState = RoomState.OPEN
    floor_policy: FloorPolicyKind = FloorPolicyKind.FREE_FOR_ALL
    participants: list[Participant] = Field(default_factory=list)
    created_at: datetime
    last_seq: int = Field(default=-1, ge=-1)
    #: **The bus announce cursor, and the reason a lost ``RoomMessagePosted`` is recoverable.**
    #:
    #: §4.1's stages 4 and 6 are two different durable systems: the append COMMITS to the room log,
    #: then the event is published on this plane's bus. A crash between them leaves a message
    #: durable at its ``seq`` and never announced — and because the client's retry hits the dedupe
    #: key and returns idempotently, nothing would ever publish it. That is a PERMANENT interior
    #: hole in the bus stream, which carries the same contiguity contract as the frame stream
    #: (CANONICAL:566) and is exactly the failure contiguity dedup cannot absorb.
    #:
    #: This cursor closes it. It is advanced ONLY AFTER a successful publish, so
    #: ``seq > announced_through_seq`` is a durable statement that the announce is still OWED, and
    #: a replay of the same ``dedupe_key`` re-runs the announce instead of returning silently. The
    #: residual is at-most-one DUPLICATE announce (publish succeeded, cursor write did not), which
    #: is the safe direction: a duplicate is absorbable, a hole is not.
    #:
    #: Distinct from ``RoomOwnerLease.published_through_seq``, which is the CENTRIFUGO cursor and
    #: is owner-leased. Two fan-outs, two cursors, deliberately not merged.
    announced_through_seq: int = Field(default=-1, ge=-1)
    #: Optimistic-concurrency guard for the whole-aggregate save spec:132 mandates. Incremented by
    #: the repository on every successful ``save``; a save whose ``expected_version`` no longer
    #: matches the stored row raises :class:`RoomVersionConflictError` instead of silently
    #: reverting whatever a concurrent verb committed in between.
    version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _rooms_are_shared_only(self) -> Session:
        if self.visibility is not Visibility.SHARED:
            raise PrivateInRoomError(
                "a room is the SHARED partition (CANONICAL §1 rule 4); "
                "visibility=private is not a narrower room"
            )
        return self

    # ---- roster reads ------------------------------------------------------------------------

    def active_participants(self) -> list[Participant]:
        return [p for p in self.participants if p.is_active_member]

    def agents(self) -> list[Participant]:
        return [p for p in self.active_participants() if p.is_agent]

    def bound_agents_of(self, owner_principal_id: str) -> list[Participant]:
        return [
            p
            for p in self.agents()
            if p.kind is ParticipantKind.BOUND_AGENT and p.owner_principal_id == owner_principal_id
        ]

    def find_active(self, principal_id: str) -> Participant | None:
        for p in self.active_participants():
            if p.principal_id == principal_id:
                return p
        return None

    # ---- roster mutations (typed failures only — spec:80) ------------------------------------

    def join(self, participant: Participant, *, max_participants: int) -> Participant:
        """RoomClosed / RoomCapacity / DuplicateParticipant (spec:80).

        Capacity counts ACTIVE members only: a room that has churned 400 people through 100 seats
        is not full, and counting departed rows would make ``max_participants`` a lifetime quota
        nobody configured.
        """
        if self.state is not RoomState.OPEN:
            raise RoomClosedError(f"room {self.id} is {self.state.value}")
        if self.find_active(participant.principal_id) is not None:
            raise DuplicateParticipantError(f"{participant.principal_id} is already a member")
        if len(self.active_participants()) >= max_participants:
            raise RoomCapacityError(f"room {self.id} is at capacity ({max_participants})")
        self.participants.append(participant)
        return participant

    def leave(self, principal_id: str, *, at: datetime) -> Participant:
        """ParticipantNotInRoom (spec:80). The row is KEPT, ``left_at`` stamped — provenance for
        every message the principal authored survives the departure, and the same stamp is what
        CANONICAL §7.4/M15 hangs the offboarding re-stamp job on."""
        participant = self.find_active(principal_id)
        if participant is None:
            raise ParticipantNotInRoomError(f"{principal_id} is not an active member")
        participant.left_at = at
        participant.presence = PresenceState.OFFLINE
        return participant

    def close(self) -> None:
        self.state = RoomState.CLOSED

    def pause(self) -> None:
        """§11's log-unavailable degrade (spec:381): ``post`` raises, reads still serve."""
        self.state = RoomState.PAUSED

    # ---- the write gate ----------------------------------------------------------------------

    def assert_can_post(self, principal_id: str) -> Participant:
        """RoomClosed / ParticipantNotInRoom / FloorDenied (spec:80), in that order.

        **Presupposes an already-validated ``ClientScope``** (spec:83): authorization lives in the
        control plane and is not re-decided here, so this can never disagree with it. What it
        decides is RUNTIME eligibility — is the room open, is the caller on the roster, does the
        floor policy grant the turn.
        """
        if self.state is not RoomState.OPEN:
            raise RoomClosedError(f"room {self.id} is {self.state.value}")
        participant = self.find_active(principal_id)
        if participant is None:
            raise ParticipantNotInRoomError(f"{principal_id} is not an active member")
        resolve_floor_policy(self.floor_policy).assert_may_post(self, principal_id)
        return participant

    # ---- control-plane projection (spec:80 "wrap, don't replace") ----------------------------

    def to_definition(self, *, expires_at: datetime | None = None) -> SessionDefinition:
        """Project down to the control-plane value.

        ``PAUSED`` maps to ``SessionState.OPEN``: pausing is a RUNTIME degrade (the log is
        unavailable), not a control-plane close, and projecting it to CLOSED would make a transient
        store outage look like a deliberate end-of-session to authorization and billing.
        """
        return SessionDefinition(
            id=self.id,
            org_id=self.org_id,
            workspace_id=self.workspace_id,
            namespace_id=self.namespace_id,
            state=(SessionState.CLOSED if self.state is RoomState.CLOSED else SessionState.OPEN),
            expires_at=expires_at,
            created_at=self.created_at,
        )

    @classmethod
    def from_definition(
        cls,
        definition: SessionDefinition,
        *,
        floor_policy: FloorPolicyKind = FloorPolicyKind.FREE_FOR_ALL,
        participants: list[Participant] | None = None,
        last_seq: int = -1,
        state: RoomState | None = None,
    ) -> Session:
        """Rebuild the aggregate from the control-plane value + the roster rows.

        ``state`` defaults to the definition's OPEN/CLOSED. PAUSED is never reconstructed from the
        control plane because it is not stored there (see :meth:`to_definition`) — a room that was
        paused by a log outage comes back OPEN and re-pauses the moment the log fails again, which
        is the correct behaviour: the degrade is a property of the store, not of the session.
        """
        return cls(
            id=definition.id,
            org_id=definition.org_id,
            workspace_id=definition.workspace_id,
            namespace_id=definition.namespace_id,
            state=(
                state
                if state is not None
                else (
                    RoomState.CLOSED if definition.state is SessionState.CLOSED else RoomState.OPEN
                )
            ),
            floor_policy=floor_policy,
            participants=list(participants or []),
            created_at=definition.created_at,
            last_seq=last_seq,
        )


# ==============================================================================================
# ROOMS §2.3 — the content-free shared-context descriptor (spec:84-97)
# ==============================================================================================
class SharedSessionContext(BaseModel):
    """A content-free VIEW over the room's shared partition + log + brief (spec:86-97).

    It **grants no access on its own** (spec:97): access is the ``authorized_ids`` stamp
    (CANONICAL §7.4). This object only says WHERE to look — a namespace, the participating agent
    principals stamped at join, the latest brief BY ID, and the log cursor. Bodies are hydrated by
    id at render time, never carried here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)  # == room_id
    #: ``Namespace.shared(...)`` — ``user='*'`` (CANONICAL §1 rule 4). Typed as the η value object
    #: rather than a string so a PRIVATE namespace cannot be smuggled in as one.
    namespace: Namespace
    participant_agent_ids: frozenset[str] = frozenset()
    brief_ref: str | None = None  # ConsolidationCompleted room brief, BY ID
    head_seq: int = Field(default=-1, ge=-1)  # RoomLogRepository tail

    @model_validator(mode="after")
    def _namespace_is_the_shared_partition(self) -> SharedSessionContext:
        """The descriptor names the room's SHARED partition and nothing else.

        Checked here rather than trusted from the caller because this object is what a recall arm
        is handed: a PRIVATE η arriving in a shared-context descriptor would send the shared arm at
        a private partition, and the ``user='*'`` zeroing (CANONICAL §1 rule 4) is the only thing
        that distinguishes them at the key level.
        """
        if self.namespace.visibility is not Visibility.SHARED:
            raise PrivateInRoomError("SharedSessionContext.namespace must be Namespace.shared(...)")
        if self.namespace.session != self.session_id:
            raise PrivateInRoomError("SharedSessionContext.namespace.session must be the room id")
        return self


# ==============================================================================================
# ROOMS §4.2 — the owner lease's durable record (spec:215-222)
# ==============================================================================================
class RoomOwnerLease(BaseModel):
    """One room's ownership record: WHO owns it, at WHICH fencing token, until WHEN, and how far
    that owner has published (spec:215-222, CANONICAL §7.6's durable ``published_through_seq``).

    **The token is the point.** A TTL lease alone cannot satisfy spec:216 ("a stale owner's store
    write is rejected AT THE STORE, not merely lease-expired"): an owner that stalls past its TTL
    and then wakes up still believes it owns the room, and its in-flight cursor advance would land.
    A monotonic token makes that write refusable by the store itself — the stale owner presents an
    older number and is rejected, whatever it believes.

    ``published_through_seq`` lives HERE, with the lease, and not in the log, because it is
    OWNER state: it says how far *the current publisher* has got, and it must survive the owner
    that produced it (CANONICAL:568 — a new owner resumes from ``published_through_seq + 1``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    room_id: str = Field(min_length=1)
    owner_instance_id: str = Field(min_length=1)
    #: Strictly increasing per room, forever. Never reused, never reset — a reset would make an
    #: old owner's token valid again, which is precisely the fence being erected.
    token: int = Field(ge=1)
    expires_at: datetime
    #: -1 = nothing published yet (matching ``Session.last_seq``'s empty-room value).
    published_through_seq: int = Field(default=-1, ge=-1)

    def is_live(self, *, at: datetime) -> bool:
        expiry = self.expires_at
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return expiry > at
