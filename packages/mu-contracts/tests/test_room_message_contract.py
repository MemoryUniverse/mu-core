"""``RoomMessage``'s AD-28 shape — the three fields spec:76-77 names and the two drifts it resolves.

**What this file exists to prevent:** a directed post whose recipients are accepted and then lost.
That was the shipped behaviour — ``Addressing.to_principal_ids`` reached ``RoomService`` and had
nowhere on the message to go — and it is not a cosmetic gap: the caller was told a message reached
a named recipient while nothing recorded who that was. The durable half of the fix is
``room_log.to_principal_ids`` (``tests/storage/test_room_tables_migration_int.py`` in mu-engine);
this file pins the contract half.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.domain.model.room import (
    MAX_BODY_CHARS,
    MAX_DEDUPE_KEY_CHARS,
    MESSAGE_ID_PREFIX,
    Addressing,
    MessageKind,
    ParticipantKind,
    RoomMessage,
    canonical_dedupe_key,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _message(**overrides: object) -> RoomMessage:
    fields: dict[str, object] = {
        "room_id": "room_1",
        "seq": 0,
        "author_principal_id": "p_alice",
        "author_kind": ParticipantKind.HUMAN,
        "kind": MessageKind.UTTERANCE,
        "correlation_id": "corr_1",
        "content_hash": "h" * 64,
        "body": "hello",
        "posted_at": _NOW,
    }
    fields.update(overrides)
    return RoomMessage(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------- addressing (AD-28 item 1)
def test_a_directed_message_carries_its_recipients() -> None:
    """The defect in one assertion: the message must be able to say WHO it was for."""
    addressing = Addressing(to_principal_ids=("p_bob", "p_carol"), reply_to_seq=7)
    assert _message(addressing=addressing).addressing == addressing


def test_addressing_is_absent_until_a_writer_sets_it_rather_than_claiming_broadcast() -> None:
    """**The blocker this asserts against:** ``addressing`` is on the public REST wire
    (``response_model=RoomMessage``) and the only writer, ``RoomService.post``, builds its draft
    without passing the caller's ``Addressing`` on. Defaulted to ``Addressing()`` the API answers a
    client that posted ``reply_to_seq=3`` with ``addressing.reply_to_seq = null`` under a field
    that claims to be the broadcast case — a knowably false value where absence was available.

    **What breaks it:** restoring ``Field(default_factory=Addressing)``. ``Addressing()`` stays a
    real, storable domain value (spec:78) — the point is that a writer must SAY it.
    """
    assert _message().addressing is None
    assert _message(addressing=Addressing()).addressing == Addressing()


def test_the_dedupe_key_refuses_to_be_computed_without_the_addressing_it_is_built_from() -> None:
    """``reply_to_seq`` is one of spec:77's four inputs. Reading a missing ``addressing`` as "not a
    reply" would return a key that silently differs from the one the row is stored under — the
    exact class of confident wrong answer the field exists to end."""
    with pytest.raises(ValueError, match="addressing"):
        _message().canonical_dedupe_key()


def test_addressing_survives_a_serialization_round_trip() -> None:
    """The field is only worth having if it crosses the wire and the store intact."""
    msg = _message(addressing=Addressing(to_principal_ids=("p_bob",), reply_to_seq=3))
    assert RoomMessage.model_validate(msg.model_dump()) == msg


# ---------------------------------------------------------------- the body bound (spec:76)
def test_a_body_at_the_ceiling_is_accepted_and_one_character_over_is_refused() -> None:
    """spec:76's ``body: str(1..32_000)``. The shipped field was a bare ``str``."""
    assert len(_message(body="x" * MAX_BODY_CHARS).body) == MAX_BODY_CHARS
    with pytest.raises(ValidationError):
        _message(body="x" * (MAX_BODY_CHARS + 1))


def test_an_empty_body_is_not_a_message() -> None:
    with pytest.raises(ValidationError):
        _message(body="")


# ---------------------------------------------------------------- id (spec:76 ``msg_<uuid>``)
def test_a_message_nobody_gave_an_id_carries_none_not_a_freshly_minted_one() -> None:
    """**The blocker this asserts against, in two lines.** ``id`` was
    ``Field(default_factory=lambda: f"msg_{uuid4()}")`` on a model both message routes return as
    their ``response_model``, while the hydrator ``PostgresRoomLog._to_message`` passes eight
    columns and never ``id``. Every read of one stored row therefore minted a NEW id: ``GET`` twice
    gave two ids for one ``(room_id, seq)``, a 200 replay disagreed with the 201 it replayed, and
    two loads of the same row compared unequal on a frozen model.

    **What breaks it:** giving ``id`` any default that is not ``None``.
    """
    a, b = _message(), _message()
    assert a.id is None
    assert a == b, "two loads of one row must be the same message, not two random identities"


def test_an_explicit_id_is_preserved_and_must_carry_the_wire_prefix() -> None:
    """spec:76 spells it ``msg_<uuid>``; a reader tells a message id from a room or correlation id
    by that prefix, so the model enforces it rather than documenting it."""
    assert _message(id="msg_fixed").id == "msg_fixed"
    for rejected in ("", "fixed", MESSAGE_ID_PREFIX):
        with pytest.raises(ValidationError):
            _message(id=rejected)


# ---------------------------------------------------------------- dedupe_key (spec:77)
def test_the_dedupe_key_is_absent_until_something_chooses_one() -> None:
    """Deliberately NOT defaulted to the canonical key: a message claiming a dedupe key that is
    not the one its row is stored under is a confident wrong answer."""
    assert _message().dedupe_key is None
    assert _message(dedupe_key="k" * MAX_DEDUPE_KEY_CHARS).dedupe_key is not None
    with pytest.raises(ValidationError):
        _message(dedupe_key="k" * (MAX_DEDUPE_KEY_CHARS + 1))


def test_the_method_and_the_function_compute_the_same_spec_77_digest() -> None:
    """One definition, two entry points — ``RoomService`` needs the value one stage before a
    message object exists, so the two must never be able to disagree."""
    msg = _message(addressing=Addressing(reply_to_seq=5))
    assert msg.canonical_dedupe_key() == canonical_dedupe_key(
        author_principal_id=msg.author_principal_id,
        room_id=msg.room_id,
        content_hash=msg.content_hash,
        reply_to_seq=5,
    )


def test_the_reply_target_changes_the_dedupe_key() -> None:
    """spec:77 includes ``reply_to``: the same sentence in reply to two different turns is two
    different messages, and the key now reads that from ``addressing``."""
    assert (
        _message(addressing=Addressing(reply_to_seq=1)).canonical_dedupe_key()
        != _message(addressing=Addressing(reply_to_seq=2)).canonical_dedupe_key()
    )


# ---------------------------------------------------------------- the name-drift ruling
def test_posted_at_is_the_name_and_created_at_is_not_an_alias() -> None:
    """AD-28's name drift, resolved one way. Leaving BOTH names live is the failure mode this
    asserts against — ``extra='forbid'`` makes the second name a hard error, not a silent alias."""
    assert _message().posted_at == _NOW
    with pytest.raises(ValidationError):
        RoomMessage(
            room_id="room_1",
            seq=0,
            author_principal_id="p_alice",
            author_kind=ParticipantKind.HUMAN,
            kind=MessageKind.UTTERANCE,
            correlation_id="corr_1",
            content_hash="h" * 64,
            body="hello",
            created_at=_NOW,  # type: ignore[call-arg]
        )
