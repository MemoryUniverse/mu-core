"""Room vocabulary — ParticipantKind / MessageKind / Addressing + the stored RoomMessage.

Authority: CANONICAL §3.2 (the content-free ``RoomMessagePosted`` event carries ``content_hash``
+ ``author_kind``/``kind`` enums — NEVER a body; consumers read the full ``RoomMessage`` from the
``RoomLogRepository`` by ``(room_id, seq)``), rooms/sessions-live-rooms spec:74-78.

``RoomMessage`` is a STORED domain object (read from the durable room log), so it legitimately
carries ``body`` — the content-free discipline binds the BUS/events, not the store (CANONICAL §3.1).
``ParticipantKind`` is UNCHANGED — ``SUBAGENT`` is an ``AgentKind`` projected to
``bound_agent``/``shared_agent``, never a new ``ParticipantKind`` value (CANONICAL §5.7, rooms A1).

--------------------------------------------------------------------------------------------
AD-28 item (1) — ``id`` / ``addressing`` / ``dedupe_key`` land here, and ``Addressing`` moves here
--------------------------------------------------------------------------------------------
``ARCHITECTURE-DELTAS.md`` AD-28 (1): the shipped ``RoomMessage`` had none of spec:76-77's ``id``,
``addressing`` or ``dedupe_key``, so **a DIRECTED post had nowhere durable to record its
``to_principal_ids``** — the recipients were accepted by the service and then lost. They are added
here, and **all three default to ``None`` — "this object does not carry one" — rather than to a
plausible value.** Two forces meet at that decision and point the same way. Mechanically, the six
construction sites live in ``mu-server`` (three of them tests), which this lane may not edit, so a
REQUIRED field would red another repo's build without changing a behaviour. Substantively, the
only writer that exists cannot yet fill any of the three: ``RoomService.post`` builds its draft
without passing the caller's ``Addressing`` on, and ``PostgresRoomLog``'s thirteen-column INSERT
and eight-column SELECT mention neither ``id`` nor the addressing columns. A default that LOOKS
right is therefore a confident wrong answer on the public REST wire — a per-instantiation random
``id`` that changes on every read, or an ``addressing`` claiming broadcast for a message the client
addressed with ``reply_to_seq``. **Absence is the house rule**, and the three fields obey it until
mu-server's appender lands (the exact edits are named on each field). ``None`` here never competes
with a domain value: ``Addressing()`` — empty ``to_principal_ids``, spec:78's broadcast — remains a
real, storable value that a writer states explicitly.

**``Addressing`` is DECLARED here although spec:78 lists it under the
``mu_core/domain/model/session.py`` header.** It is mechanically forced: ``RoomMessage`` now has an
``addressing`` field, ``session.py`` already imports ``RoomMessage`` from this module, and the
reverse edge would be an import cycle. ``session.py`` re-exports both ``Addressing`` and
:func:`canonical_dedupe_key`, so every existing import path (``mu_server.ports`` imports
``Addressing`` from ``...model.session``) is byte-identical. Same package, one declaration, no
duplicate vocabulary.

**``created_at`` vs ``posted_at`` — ``posted_at`` WINS, deliberately** (AD-28 name drift). spec:76
says ``created_at``; the shipped field, the ``room_log`` column, ``PostgresRoomLog._APPEND_SQL``,
``_ROW_COLUMNS`` and ``_to_message`` all say ``posted_at``. Three reasons, in order of weight:
(1) ``Session.created_at`` already exists on the sibling aggregate meaning *when the room was
opened* — two ``created_at`` fields one import apart, one of them meaning "when this sentence was
said", is the kind of collision that gets mis-joined once and then believed; (2) the rename costs
twelve edits in ``mu-server``, three of them in tests this lane may not touch, so a rename landed
here would be a HALF rename — the worst state; (3) ``posted_at`` names the act (a message is
posted; a room is created), which is why the storage lane reached for it independently. The spec
text is the side that changes.

⚠ **That doc change is NOT yet in the design set, and this file may not put it there.** An earlier
cut of this docstring said the ruling was "recorded as a fix-docs delta"; it was not recorded
anywhere — ``docs/superpowers/design/rooms-sessions-subscriptions-spec.md:76`` still reads
``created_at`` and the AD-28 row still stands ⏳ OPEN. Under CLAUDE.md rule 12 an unrecorded
divergence is an unreviewed decision, so the ruling is REPORTED to the orchestrator for
``ARCHITECTURE-DELTAS.md`` together with two others this build makes: ``Addressing`` /
:func:`canonical_dedupe_key` are declared here rather than under spec:78's
``session.py`` header (import cycle — see below), and AD-28 item (3) was answered with NEW
``room_session``/``room_participant`` tables rather than by widening the shipped
``sessions``/``session_participants`` (revision ``b3d47c9a1e02``'s docstring states why).
Asserting a record exists is worse than the missing record: an operator would stop looking.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MAX_BODY_CHARS",
    "MAX_DEDUPE_KEY_CHARS",
    "MESSAGE_ID_PREFIX",
    "Addressing",
    "MessageKind",
    "ParticipantKind",
    "RoomMessage",
    "canonical_dedupe_key",
]

#: **The absolute ceiling on a stored body, spec:76's ``body: str(1..32_000)``.**
#:
#: This is NOT a second copy of ``RoomSettings.max_message_chars`` and the two are not in
#: competition — they are different instruments, and the reconciliation AD-28 asks for is that
#: this one is the RECORD invariant and that one is the DEPLOYMENT POLICY:
#:
#: * ``max_message_chars`` (``mu_server.settings``, default 32 000, ``ge=1``) is per-deployment and
#:   deliberately tunable — a test lowers it to 8, a tenant could lower it to 4 000. It is checked
#:   in ``RoomService.post`` BEFORE the model is built and raises the typed ``MessageTooLongError``
#:   that the REST edge maps to 422. It is authoritative for *what a room accepts*.
#: * ``MAX_BODY_CHARS`` is authoritative for *what the type can hold at all*, for every writer
#:   including ones that never pass through ``RoomService`` (the FREE single-owner LOCAL room,
#:   spec:265-277, has no server-side settings object). It is a domain invariant of the record —
#:   which is why it is a NAMED constant and not a magic literal (``DEV-STANDARDS`` rule 3: the
#:   tunable lives in ``Settings``; an invariant lives with the type it constrains).
#:
#: A policy above this ceiling is a misconfiguration, not a wider room: the model refuses the body
#: and the caller sees ``pydantic.ValidationError`` rather than ``MessageTooLongError``. Keeping
#: ``max_message_chars <= MAX_BODY_CHARS`` is the deployment's obligation.
MAX_BODY_CHARS: Final = 32_000

#: The width of the durable ``room_log.dedupe_key`` column and therefore the ceiling on the
#: client's ``Idempotency-Key`` (spec:325 maps one onto the other). Declared HERE, in the contract
#: package, because three layers need the same number and the incident that produced it was a
#: drift between two of them: an unbounded header reached the column, Postgres answered ``22001``
#: (``StringDataRightTruncation``) — neither an ``IntegrityError`` nor a ``RoomError`` — and one
#: HTTP header was re-labelled a store outage that PAUSED a room. ``mu_engine``'s ``schema.py``
#: imports this constant for the column width, so the model and the DDL cannot drift.
#: ``mu_server.rooms.room_log_pg.MAX_DEDUPE_KEY_CHARS`` still carries its own copy of the value and
#: should collapse onto this one — that repo is owned by another lane right now.
MAX_DEDUPE_KEY_CHARS: Final = 128

#: spec:76 spells the message id ``msg_<uuid>``. The prefix is a constant because the id is a wire
#: value: a reader that has to tell a message id from a room id or a correlation id does it by this
#: prefix, so it may not be re-typed by hand at any site that mints one. :attr:`RoomMessage.id`
#: ENFORCES it, so the constant has a job rather than being documentation.
MESSAGE_ID_PREFIX: Final = "msg_"

#: ``MESSAGE_ID_PREFIX`` + a 36-char canonical UUID = 40; the bound is the column width with room
#: for a longer prefix, never a semantic limit.
_MAX_MESSAGE_ID_CHARS: Final = 160


class ParticipantKind(StrEnum):
    HUMAN = "human"
    SHARED_AGENT = "shared_agent"
    BOUND_AGENT = "bound_agent"


class MessageKind(StrEnum):
    UTTERANCE = "utterance"
    AGENT_RESULT = "agent_result"
    SYSTEM = "system"
    CONTEXT_INJECTED = "context_injected"


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

    Kept as a module-level FUNCTION as well as :meth:`RoomMessage.canonical_dedupe_key` because the
    key has to be computable BEFORE a message object exists: ``RoomService.post`` derives it in
    stage 3 to decide whether the append it is about to attempt is a replay. The inputs are exactly
    spec:77's four, the separator is a character the namespace validator already forbids inside any
    component, and the digest is content-free: it is built from a CONTENT HASH, never from the body.
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


class RoomMessage(BaseModel):
    """The durable room-log record. ``seq`` is the per-room monotonic ordering authority
    (contiguity apply-rule, CANONICAL §7.6). Read from the store by ``(room_id, seq)`` — the
    content-free ``RoomMessagePosted`` event only carries the ``content_hash`` locator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: spec:76's ``msg_<uuid>`` surrogate. **The DURABLE identity of a message is still
    #: ``(org_id, workspace_id, room_id, seq)``** — the room-log primary key and the ordering
    #: authority. This id is the stable handle a dispatch/result/capture chain quotes when it must
    #: name one message without carrying the tenancy triple.
    #:
    #: ⚠ **``None`` means "this object does not carry one", and it is NOT minted at construction.**
    #: An earlier cut defaulted it to ``default_factory=lambda: f"msg_{uuid4()}"``, which made the
    #: id a per-INSTANTIATION random value on a field that is on the public REST wire
    #: (``mu_server.routes.rooms`` declares ``response_model=RoomMessage`` on both message routes):
    #: the shipped hydrator ``PostgresRoomLog._to_message`` passes eight columns and never ``id``,
    #: so the SAME stored message came back with a DIFFERENT id on every read, a 200 replay
    #: disagreed with the 201 it replayed, and two loads of one row were ``!=`` each other. A field
    #: that is always wrong is worse than an absent one — absence is the house rule — so it is
    #: absent until a writer that can durably record it sets it. The ``room_log.id`` column added
    #: by revision ``b3d47c9a1e02`` is nullable for the same reason, and the appender that will
    #: fill both lives in ``mu-server``, which this lane may not edit.
    id: str | None = Field(
        default=None,
        min_length=len(MESSAGE_ID_PREFIX) + 1,
        max_length=_MAX_MESSAGE_ID_CHARS,
        pattern=rf"^{MESSAGE_ID_PREFIX}",
    )
    room_id: str = Field(min_length=1)  # == session_id
    seq: int = Field(ge=0)
    author_principal_id: str = Field(min_length=1)
    author_kind: ParticipantKind
    kind: MessageKind
    correlation_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)  # dedupe / provenance
    #: **The durable home of ``to_principal_ids`` — AD-28 (1).** Before this field a directed post
    #: was accepted and its recipients dropped on the floor, which is why ``RoomService`` had to
    #: refuse one outright (``AddressingNotStorableError``).
    #:
    #: ⚠ **``None`` means "not carried on this object", exactly like :attr:`dedupe_key` — it does
    #: NOT default to ``Addressing()``.** ``Addressing()`` is a real domain value (spec:78: empty
    #: ``to_principal_ids`` IS broadcast, with no reply target), and asserting it is a LIE for the
    #: one writer that exists: ``RoomService.post`` receives the caller's ``Addressing`` and builds
    #: its draft without passing it on, so a client that posts ``reply_to_seq=3`` would read back
    #: ``addressing.reply_to_seq = null`` from a field claiming to be the broadcast case. Until
    #: that service passes ``addressing=`` (one keyword, in ``mu-server``, which this lane may not
    #: edit) and ``PostgresRoomLog`` reads the two columns back, an honest ``None`` is the only
    #: value this field can carry without inventing one.
    addressing: Addressing | None = None
    #: The key the row was actually deduplicated on: the client's ``Idempotency-Key`` when one was
    #: supplied, else :meth:`canonical_dedupe_key`. ``None`` means "not carried on this object" —
    #: either the caller has not chosen one yet (a draft, before ``RoomLogRepository.append``) or
    #: the row was hydrated by a reader that did not select the column. It is deliberately NOT
    #: defaulted to the canonical key: a message claiming a dedupe key that is not the one its row
    #: is stored under would be a confident wrong answer, and this field exists to end exactly that
    #: class of loss.
    dedupe_key: str | None = Field(default=None, min_length=1, max_length=MAX_DEDUPE_KEY_CHARS)
    #: Stored body (legitimate on the STORE object; never on the bus). Bounded per spec:76 — see
    #: :data:`MAX_BODY_CHARS` for why this bound and ``RoomSettings.max_message_chars`` are two
    #: different instruments rather than a duplicated constant.
    body: str = Field(min_length=1, max_length=MAX_BODY_CHARS)
    posted_at: datetime

    def canonical_dedupe_key(self) -> str:
        """spec:77's derived key, computed from THIS message.

        Shares its name with the module-level function on purpose — spec:77 puts the method on
        ``RoomMessage`` and ``RoomService`` needs the same value one stage earlier, so there is one
        definition and two entry points rather than two implementations that can disagree.

        :raises ValueError: when :attr:`addressing` is ``None``. ``reply_to_seq`` is one of
            spec:77's four inputs, so a message that does not carry its addressing cannot compute
            its key — and reading the missing value as "not a reply" would hand back a key that
            silently differs from the one the row is actually stored under, which is the precise
            failure this field exists to end. Use the module-level
            :func:`canonical_dedupe_key` where the reply target is known independently, as
            ``RoomService.post`` does one stage before a message object exists.
        """
        if self.addressing is None:
            raise ValueError(
                "this RoomMessage does not carry its `addressing`, and spec:77's dedupe key is "
                "computed from `reply_to_seq`: pass `addressing=` when building it, or call the "
                "module-level `canonical_dedupe_key()` with the reply target you already know"
            )
        return canonical_dedupe_key(
            author_principal_id=self.author_principal_id,
            room_id=self.room_id,
            content_hash=self.content_hash,
            reply_to_seq=self.addressing.reply_to_seq,
        )
