"""Room ports — the Layer-0 Protocol boundary for the S3 shared-context runtime
(``rooms-sessions-subscriptions-spec.md`` §3.1/§3.2, spec:106-160).

**These are OPEN (Apache-2.0) on purpose.** §1's component map puts the room aggregate, its value
objects and its ports in ``mu-core`` on plane *both* (spec:29-31); only the machinery that exists
BECAUSE other people/devices/tenants are involved — the Centrifugo publisher, the REST edge, the
multi-instance owner lease's ADAPTER — is commercial (spec:44-45, ``mu-server/CLAUDE.md``). Defining
the ports here is what lets one aggregate serve both the free single-owner LOCAL realization
(spec:265-277) and the governed SHARED room, without either plane importing the other.

⚠ **WHAT IS DELIBERATELY ABSENT, and why absence rather than a stub** (``DEV-STANDARDS.md``: "no
dead code, no silent stubs that look finished"). Each of these is declared by the spec and has no
caller in this build, so it lands with the lane that gives it one:

* ``RoomLogRepository.follow(room_id, *, from_seq)`` (spec:139) — the OWNER's log follower. Its
  only consumer is the single Centrifugo publisher (spec:213), which is the follow-up lane. On
  Postgres it needs a DEDICATED connection held for the follower's lifetime (``LISTEN/NOTIFY``
  does not fit a pooled ``async_sessionmaker`` seam), so declaring it now would pin a transaction
  shape before the component that has to live with it exists.
* ``PresenceStorePort`` (spec:142-145), ``ServerPushPort`` (spec:152), the ``RoomEvent``
  discriminated union and ``RoomServicePort.subscribe``/``heartbeat`` (spec:114-116) — presence and
  fan-out, i.e. the same follow-up lane.
* ``RoomServicePort.bind_local_agent``/``unbind_local_agent`` (spec:119-120) — bound agents are
  Phase 5. Note when they land: the binding is a RECORD ONLY, no ``HostBridgePort`` crosses
  (CANONICAL §6-P9 / B5).
* ``IngestServicePort`` (spec:147, CANONICAL §6-P6) and ``SharedRecallPort`` (spec:149) — consumed
  by the capture/injection coordinators (spec:33-34), which no lane has built.

⚠ **TWO SIGNATURE DEVIATIONS from spec §3.2, both forced and both load-bearing:**

1. **Every repository verb is η-scoped by ``(org_id, workspace_id, room_id)``, where spec:131-140
   passes a bare ``room_id``/``workspace_id``.** CANONICAL §1 rule 5 makes ``to_prefix()`` — which
   begins at ``org`` — the tenancy guarantee, "not a filter"; a repository keyed on ``room_id``
   alone would make cross-tenant isolation depend on room ids being unguessable, which is exactly
   the substitution rule 5 exists to forbid. ADR 0026 un-collapsed ``org`` from ``workspace``, so
   both segments have to be on the wire.
2. **``append`` takes ``dedupe_key`` as a keyword instead of reading
   ``RoomMessage.canonical_dedupe_key()``** (spec:77). ⚠ **The reason this deviation was recorded
   is GONE; the deviation itself is not, and the difference matters.** ``RoomMessage`` now carries
   spec:76-77's ``id``, ``addressing`` and ``dedupe_key`` (AD-28 item 1), and
   :meth:`mu_contracts.domain.model.room.RoomMessage.canonical_dedupe_key` is the method spec:77
   names. The keyword stays because collapsing it is a signature change to a Protocol whose only
   implementation and only caller (``PostgresRoomLog.append`` / ``RoomService._append_with_retry``)
   live in ``mu-server``, which this lane may not edit: dropping the keyword here would red the
   other repo's type gate without changing a single behaviour. **The collapse is now a mu-server
   edit, not a mu-core one**, and it has a PRECONDITION: ``message.canonical_dedupe_key()`` raises
   unless the message carries its :class:`~mu_contracts.domain.model.room.Addressing`, because
   ``reply_to_seq`` is one of spec:77's four inputs and reading a missing addressing as "not a
   reply" returns a key that differs from the one the row is stored under. So the collapse is
   ``message.dedupe_key or message.canonical_dedupe_key()`` **only once ``RoomService.post`` passes
   ``addressing=`` into the draft** — which is the same one-line mu-server edit AD-28 item (1)
   still waits on. Until then the caller-supplied keyword, derived from the module-level
   :func:`mu_contracts.domain.model.room.canonical_dedupe_key` with the reply target it already
   holds, is the only form that can be right. The dedupe VALUE is identical either way: the
   function and the method compute the same spec:77 digest from the same four inputs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.room import MessageKind, RoomMessage
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.domain.model.session import (
    Addressing,
    FloorPolicyKind,
    Participant,
    RoomOwnerLease,
    Session,
)

__all__ = [
    "RoomClientPort",
    "RoomLogRepository",
    "RoomOwnerLeasePort",
    "RoomServicePort",
    "SessionRepository",
]


@runtime_checkable
class SessionRepository(Protocol):
    """The room aggregate's durable home (spec:130-133). Whole-aggregate save — the roster and the
    room state change together or not at all, so a per-participant write surface would let a
    half-applied join be observable.

    ⚠ **THREE DEVIATIONS from spec:130-133, all closing a lost-update window the spec does not
    mention.** spec:132 mandates the whole-aggregate save and stops there; a whole-aggregate save
    with no precondition is a read-modify-write race, and every verb in this subsystem performs
    one:

    1. **``save`` takes ``expected_version``.** Without it a concurrent ``join`` and ``leave``
       each write the whole object from their own load, and whichever commits second silently
       reverts the other — the ejected participant's next post then fails
       ``ParticipantNotInRoomError`` for no reason a log line explains. The ``seq`` path already
       had exactly this precondition (``expected_seq``, spec:136); the roster path had none.
    2. **``create`` is separate from ``save``.** ``open_room`` through a bare ``save`` is a
       REPLACE: a second call on a live room ejects the roster, resets ``last_seq`` while the
       durable log keeps its rows, and re-opens a CLOSED room. A create-only verb makes that
       unexpressible rather than merely discouraged.

    3. **``advance_seq_cursors`` is a NARROW write, deliberately not a ``save``.** §4.1 stage 5
       sits on the post hot path, and routing a hint update through the whole-aggregate surface is
       what lets a post revert a concurrent join. It is a MONOTONIC max-update of two integers, so
       it commutes: concurrent posts cannot lose each other, and it needs no version.
    """

    async def get(self, org_id: str, workspace_id: str, session_id: str) -> Session | None: ...

    async def create(self, session: Session) -> None:
        """Insert a room that does not exist yet.

        :raises RoomAlreadyExistsError: a row already exists for this ``(org, workspace, id)``.
        """
        ...

    async def save(self, session: Session, *, expected_version: int) -> None:
        """Replace the whole aggregate iff the stored ``version`` is still ``expected_version``,
        then advance ``session.version`` in place so the caller's object matches the row.

        :raises RoomVersionConflictError: the stored version moved — reload and re-apply.
        """
        ...

    async def advance_seq_cursors(
        self, org_id: str, workspace_id: str, session_id: str, *, seq: int
    ) -> None:
        """Monotonically raise ``last_seq`` **and** ``announced_through_seq`` to at least ``seq``.

        Called ONLY after ``RoomMessagePosted`` has been published, which is what makes
        ``seq > announced_through_seq`` a durable statement that the announce is still owed
        (see :attr:`Session.announced_through_seq`). Lowering either cursor is not expressible:
        a rewind would re-announce an arbitrary span of the room.
        """
        ...

    async def list_open(
        self, org_id: str, workspace_id: str, namespace_id: str
    ) -> list[Session]: ...


@runtime_checkable
class RoomLogRepository(Protocol):
    """The append-only per-room event store — **and the sole ``seq`` ordering authority**
    (spec:135-140, CANONICAL §3.2/§7.6).

    ``seq`` is not a counter. It is the property every downstream correctness rule rests on: a
    client applies a frame **only if ``seq == last_applied + 1``** (CONTIGUITY, CANONICAL:566), and
    a monotonic high-water-mark is explicitly REJECTED because it silently drops a missing interior
    ``seq``. So the sequence must be **gap-free and duplicate-free** per room, under concurrent
    posters and under N server instances. A hole here is not a slow client; it is a client that
    stops applying and never resumes.

    An implementation MUST therefore state, in its own docstring, what serializes two concurrent
    appends. "Writes are rare" is not a mechanism.
    """

    async def append(
        self,
        message: RoomMessage,
        *,
        org_id: str,
        workspace_id: str,
        expected_seq: int,
        dedupe_key: str,
    ) -> RoomMessage:
        """Assign ``message`` the next ``seq`` on ``(org_id, workspace_id, message.room_id)``.

        ``expected_seq`` is the caller's optimistic assertion about the tail (spec:136). It is a
        CONTRACT, not a hint: an implementation that ignores it and appends at ``tail+1`` anyway
        would silently reorder a caller that had made a decision from a stale tail.

        :raises RoomLogConflictError: ``expected_seq != tail + 1`` — retryable (spec:207).
        :raises RoomDuplicateError: ``dedupe_key`` already present; carries the STORED message so
            the caller returns it idempotently rather than reading it back.
        """
        ...

    async def get(
        self, org_id: str, workspace_id: str, room_id: str, seq: int
    ) -> RoomMessage | None:
        """The body-materialization path. **CANONICAL:220 names this method by name**: every
        consumer of a ``RoomMessagePosted`` event fetches the full message "from
        ``RoomLogRepository.get(room_id, seq)`` — never from the event". spec:135-140 omits it;
        CANONICAL wins (spec:6)."""
        ...

    async def read(
        self,
        org_id: str,
        workspace_id: str,
        room_id: str,
        *,
        after_seq: int = -1,
        limit: int = 200,
    ) -> list[RoomMessage]:
        """The durable backfill run, in ``seq`` order — the correctness path a client falls to on
        ANY gap (CANONICAL:566, ``GET /v1/rooms/{id}/messages?after_seq=``)."""
        ...

    async def tail_seq(self, org_id: str, workspace_id: str, room_id: str) -> int:
        """The highest assigned ``seq``, or ``-1`` for an empty room."""
        ...


@runtime_checkable
class RoomOwnerLeasePort(Protocol):
    """§4.2's ``room-owner:{room_id}`` lease with a **monotonic fencing token** (spec:215-222).

    **Why a NEW port rather than the shipped leases.** ``LifecycleLeasePort``
    (``mu_contracts/ports/lifecycle_lease.py:54``) and ``WriterLeasePort``
    (``mu_engine/pipelines/distill.py:201``) both yield ``None`` — they are acquire-or-defer locks
    with no token — and spec:216 requires strictly more: *"a stale owner's store write is rejected
    AT THE STORE, not merely lease-expired"*. A lock cannot express that; only a number the store
    can compare can. Changing either shipped port's signature to return a token would alter a
    contract two other subsystems already bind, so this is an additive third port, not a fork.

    **What the lease does NOT govern: ``seq``.** The ordering authority is
    :class:`RoomLogRepository`, whose serialization happens in the STORE. That split is deliberate
    and matches the precedent ``mu_server/application/sync_hub_service.py:27-34`` already set for
    the private sync log ("``seq`` serialization is a different grain entirely"). The consequence
    is stated here because it is the property that makes owner failover survivable: **stages 1-5 of
    §4.1 are lease-INDEPENDENT**, so a lease that expires mid-pipeline costs at most a duplicate
    PUBLISH — never a duplicate ``seq``, never a gap.
    """

    async def acquire(
        self, org_id: str, workspace_id: str, room_id: str, *, instance_id: str
    ) -> RoomOwnerLease | None:
        """Claim the room if it is unowned or the incumbent lease has expired, minting the NEXT
        token. Returns ``None`` when a live owner holds it — a refusal, not an error: on a healthy
        plane exactly one instance wins and the rest are simply not the owner."""
        ...

    async def renew(
        self, lease: RoomOwnerLease, *, org_id: str, workspace_id: str
    ) -> RoomOwnerLease | None:
        """Extend the TTL. Returns ``None`` if this lease is no longer the current one — which is
        how an instance that stalled past its TTL LEARNS it was fenced, instead of assuming."""
        ...

    async def release(self, lease: RoomOwnerLease, *, org_id: str, workspace_id: str) -> None:
        """Give the room up early. A no-op if the lease is already stale — releasing a lease you
        no longer hold must never evict the instance that legitimately took it."""
        ...

    async def read(self, org_id: str, workspace_id: str, room_id: str) -> RoomOwnerLease | None: ...

    async def advance_published_through(
        self, lease: RoomOwnerLease, *, org_id: str, workspace_id: str, through_seq: int
    ) -> RoomOwnerLease:
        """Move the durable ``published_through_seq`` cursor — **the fenced write** (CANONICAL:568).

        This is the one operation that MUST reject a stale owner. A new owner resumes publication
        from ``published_through_seq + 1``; if an expired owner could advance the cursor past frames
        it never published, those frames are never published by anyone and the gap is PERMANENT —
        the one failure in §4.2 that contiguity dedup at the client cannot absorb.

        :raises StaleRoomOwnerError: ``lease.token`` is older than the room's current owner token.
        """
        ...


@runtime_checkable
class RoomServicePort(Protocol):
    """The room application surface (spec:109-123), narrowed to the verbs this build implements —
    see the module docstring for the four groups that are deliberately absent.

    ``RoomClientPort``/``RoomStreamPort`` are NARROWER VIEWS onto one implementation, never second
    implementations (CANONICAL §6-P7, CANONICAL:457).
    """

    async def open_room(self, scope: ClientScope, policy: FloorPolicyKind) -> Session: ...

    async def join(self, scope: ClientScope, participant: Participant) -> Session: ...

    async def leave(self, scope: ClientScope) -> None: ...

    async def post(
        self,
        scope: ClientScope,
        body: str,
        addressing: Addressing,
        kind: MessageKind = MessageKind.UTTERANCE,
        *,
        dedupe_key: str | None = None,
    ) -> RoomMessage: ...

    async def backfill(
        self, scope: ClientScope, after_seq: int, limit: int = 200
    ) -> list[RoomMessage]: ...

    async def roster(self, scope: ClientScope) -> list[Participant]: ...

    async def close_room(self, scope: ClientScope) -> None: ...


@runtime_checkable
class RoomClientPort(Protocol):
    """The daemon's narrow view (CANONICAL §6-P7): backfill + post.

    ``subscribe`` — the third verb spec:125 gives this view — is absent with the rest of the
    fan-out surface. What the daemon holds today is the REST up-channel half, and that is exactly
    what this declares, so no consumer can bind a stream that nothing publishes.
    """

    async def post(
        self,
        scope: ClientScope,
        body: str,
        addressing: Addressing,
        kind: MessageKind = MessageKind.UTTERANCE,
        *,
        dedupe_key: str | None = None,
    ) -> RoomMessage: ...

    async def backfill(
        self, scope: ClientScope, after_seq: int, limit: int = 200
    ) -> list[RoomMessage]: ...
