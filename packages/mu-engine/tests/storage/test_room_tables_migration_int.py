"""Revision ``b3d47c9a1e02`` — REAL Postgres, THROWAWAY databases, ZERO mocks.

**Every test in this file runs the revision against a database it created seconds earlier and
drops when it is done.** That is not tidiness, it is the only way three of the properties below
can be observed at all:

* A test that asserts against a database *already at head* proves nothing about the revision — the
  vacuous-pass this project has recorded four times. An earlier cut of this file upgraded the
  SHARED control plane to ``head`` (a no-op when it is already there) and then asserted; deleting
  a column from the revision left five of its six tests green.
* The DIRTY-database behaviour — a hand-rolled ``room_log``, a present-but-short ``room_session``
  — cannot be set up on the shared control plane at all without breaking whatever else is using
  it, and it is the property the AD-28 brief singled out ("write the migration so it is safe
  against a dirty dev DB").
* The shared control plane at 127.0.0.1:15432 is used CONCURRENTLY by other lanes. Running
  ``alembic downgrade`` against it leaves it a revision short for the duration, and a process
  death mid-test leaves it short for good — the exact drift this repo has already been bitten by.
  The sibling migration tests here still do that; this one does not, and the scratch database is
  why. ``mu-server``'s ``tests/integration/test_room_log_int.py`` already uses the same pattern.

What is proven, and what each proof is FOR
------------------------------------------------------------------------------------------------
1. **The revision chains from the real head**, verified from the script directory rather than
   assumed, and the whole chain applies to an empty database with this revision on top.
2. **``up -> down -> up``** on a clean database: ``downgrade`` really removes what ``upgrade``
   built, so a partial application can be undone.
3. **A hand-rolled ``room_log`` is RECONCILED and its messages survive** — columns, the literally
   named ``ux_roomlog_dedupe``, and ``dedupe_key``'s width — and ``downgrade`` then reverses the
   additive changes WITHOUT dropping the table or its rows. The first cut of this revision dropped
   it: up added three columns, down deleted every message.
4. **A present-but-short ``room_session`` is widened and its cursors are BACKFILLED from the log**,
   with no ``server_default`` left behind. This is the recorded incident shape of AD-28 item (3):
   before the fix, ``alembic_version`` advanced to head while the five columns
   ``mu_server.rooms.session_pg._SELECT_SESSION`` reads stayed missing.
5. **A populated roster missing a NOT NULL column whose value is not derivable STOPS the
   migration** rather than relabelling an agent row ``human``.
6. **A DIRECTED post survives the round trip** — AD-28 item (1) in its durable form.
7. **``(org, workspace, room, seq)`` and ``(org, workspace, room, dedupe_key)`` are unique at the
   DATABASE**, because ``seq`` is the contiguity apply-rule's authority (CANONICAL:566) and a
   duplicate is not a slow client, it is a client that stops applying.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import make_url, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mu_contracts.config import get_settings
from mu_contracts.domain.model.room import (
    MAX_DEDUPE_KEY_CHARS,
    Addressing,
    MessageKind,
    ParticipantKind,
    RoomMessage,
)

pytestmark = pytest.mark.integration

_ALEMBIC_INI = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mu_engine"
    / "storage"
    / "relational"
    / "alembic.ini"
)
_HEAD = "b3d47c9a1e02"  # THE REVISION UNDER TEST — not necessarily the chain head, see below
_PRIOR = "9c41d0b7ae52"  # the revision it Revises

#: ⚠ **``_HEAD`` is the revision under test, NOT "whatever alembic's head is today".**
#:
#: This file used to assert ``script.get_current_head() == _HEAD`` and, after
#: ``scratch.upgrade("head")``, ``version() == _HEAD``. Both were true only until the NEXT
#: migration landed — and one did (``c4a1e07b9d33``, ``acl_entries.permission``), which turned two
#: assertions about *this* revision into a red test about the calendar. A pin that every future
#: migration must come back and edit is not testing the revision; it is testing that no one has
#: written another one. ``test_principal_registry_migration_int.py:45-61`` already records the same
#: lesson in the opposite direction, and uses alembic's ``"head"`` literal for the same reason.
#:
#: What is asserted instead is what this revision actually claims: it chains from ``_PRIOR``, it is
#: still ON the chain reachable from the current head (a rebase that orphaned it would be a real
#: defect), and a fresh ``upgrade head`` — what an operator's first deploy runs — lands on the
#: script directory's own head with this revision's tables and columns present.

_TABLES = ("room_log", "room_session", "room_participant")

#: The three columns this revision adds beyond the shape ``PostgresRoomLog.REQUIRED_DDL`` froze,
#: as ``{column: is_nullable}``. All three are nullable, each for a stated reason (see the revision
#: docstring); none is a "NOT NULL on a populated table" hazard, because on the clean path the
#: table is new and on the reconcile path these three are exactly the nullable ones.
_LATE_COLUMNS = {"id": True, "to_principal_ids": True, "reply_to_seq": True}

#: Every NOT NULL roster column whose value for an already-existing row cannot be derived from
#: anything true. The migration must name ALL of them when it refuses — see the roster test.
_ROSTER_NOT_DERIVABLE = ("kind", "owner_principal_id", "capabilities", "ordinal", "presence")

#: The five columns AD-28 item (3) is about — what ``session_pg._SELECT_SESSION`` reads.
_SESSION_RUNTIME_COLUMNS = frozenset(
    {"visibility", "floor_policy", "last_seq", "announced_through_seq", "version"}
)

#: ``room_log`` exactly as ``mu_server.rooms.room_log_pg.REQUIRED_DDL`` states it — the shape a dev
#: database seeded by hand is in — MINUS ``ux_roomlog_dedupe`` and with a NARROWER ``dedupe_key``,
#: which are the two drifts a hand-rolled table is most likely to carry and the two the reconcile
#: branch has to repair. (Without the constraint, ``PostgresRoomLog._APPEND_SQL``'s ``ON CONFLICT
#: ON CONSTRAINT ux_roomlog_dedupe`` raises ``UndefinedObject`` on the first client retry.)
_LEGACY_ROOM_LOG = """
CREATE TABLE room_log (
    org_id VARCHAR(128) NOT NULL, workspace_id VARCHAR(128) NOT NULL,
    room_id VARCHAR(128) NOT NULL, seq BIGINT NOT NULL,
    author_principal_id VARCHAR(128) NOT NULL, author_kind VARCHAR(32) NOT NULL,
    kind VARCHAR(32) NOT NULL, correlation_id VARCHAR(128) NOT NULL,
    content_hash VARCHAR(128) NOT NULL, dedupe_key VARCHAR(64) NOT NULL,
    body TEXT NOT NULL, posted_at TIMESTAMPTZ NOT NULL, appended_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT pk_room_log PRIMARY KEY (org_id, workspace_id, room_id, seq)
)
"""

#: The recorded incident shape: the table PRESENT while its sibling columns are MISSING.
_SHORT_ROOM_SESSION = """
CREATE TABLE room_session (
    org_id VARCHAR(128) NOT NULL, workspace_id VARCHAR(128) NOT NULL, id VARCHAR(128) NOT NULL,
    namespace_id VARCHAR(128) NOT NULL, state VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT pk_room_session PRIMARY KEY (org_id, workspace_id, id)
)
"""

_SHORT_ROOM_PARTICIPANT = """
CREATE TABLE room_participant (
    org_id VARCHAR(128) NOT NULL, workspace_id VARCHAR(128) NOT NULL,
    room_id VARCHAR(128) NOT NULL, principal_id VARCHAR(128) NOT NULL,
    joined_at TIMESTAMPTZ NOT NULL
)
"""

_LEGACY_ROW = text(
    """
    INSERT INTO room_log (org_id, workspace_id, room_id, seq, author_principal_id, author_kind,
                          kind, correlation_id, content_hash, dedupe_key, body, posted_at,
                          appended_at)
    VALUES ('org_legacy', 'ws_1', 'room_1', :seq, 'p_alice', 'human', 'utterance', 'corr',
            'h', :dedupe_key, 'a message written before the migration existed', now(), now())
    """
)


class _Scratch:
    """One throwaway database, plus the alembic config pointed at it."""

    def __init__(self, name: str, engine: AsyncEngine) -> None:
        self.name = name
        self.engine = engine
        self.cfg = Config(str(_ALEMBIC_INI))
        self.cfg.set_main_option("sqlalchemy.url", str(engine.url.render_as_string(False)))

    async def run(self, statement: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text(statement))

    async def upgrade(self, revision: str = _HEAD) -> None:
        await asyncio.to_thread(command.upgrade, self.cfg, revision)

    async def downgrade(self, revision: str = _PRIOR) -> None:
        await asyncio.to_thread(command.downgrade, self.cfg, revision)

    async def stamp(self, revision: str = _PRIOR) -> None:
        await asyncio.to_thread(command.stamp, self.cfg, revision)

    async def version(self) -> str | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(text("SELECT version_num FROM alembic_version"))).first()
        return None if row is None else str(row[0])

    async def tables(self) -> set[str]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_name = ANY(:names)"
                    ),
                    {"names": list(_TABLES)},
                )
            ).all()
        return {str(r.table_name) for r in rows}

    async def columns(self, table: str) -> dict[str, tuple[bool, int | None, str | None]]:
        """``{column: (is_nullable, char_max_length, column_default)}`` straight from
        ``information_schema`` — never trust the ORM's idea of what it wrote; ask the database."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT column_name, is_nullable, character_maximum_length, column_default "
                        "FROM information_schema.columns WHERE table_name = :t"
                    ),
                    {"t": table},
                )
            ).all()
        return {
            str(r.column_name): (
                r.is_nullable == "YES",
                r.character_maximum_length,
                r.column_default,
            )
            for r in rows
        }

    async def constraints(self, table: str) -> dict[str, str]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT conname, contype FROM pg_constraint "
                        "WHERE conrelid = to_regclass(CAST(:t AS TEXT))"
                    ),
                    {"t": table},
                )
            ).all()
        # `contype` is Postgres `"char"`; asyncpg hands it back as a one-byte `bytes`.
        return {str(r.conname): r.contype.decode() for r in rows}

    async def scalar(self, statement: str) -> object:
        async with self.engine.connect() as conn:
            return (await conn.execute(text(statement))).scalar()


@pytest_asyncio.fixture
async def scratch() -> AsyncIterator[_Scratch]:
    """A database of its own per test, dropped afterwards.

    The DSN comes from the central Settings tree (never a literal); only the database NAME is
    replaced, so the credentials and host stay the ones ``.env.test`` wires.
    """
    control = make_url(get_settings().storage.postgres.dsn)
    name = f"mu_roommig_{uuid4().hex[:12]}"
    admin = create_async_engine(control, isolation_level="AUTOCOMMIT", poolclass=None)
    engine = create_async_engine(control.set(database=name), poolclass=None)
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
        yield _Scratch(name, engine)
    finally:
        await engine.dispose()
        async with admin.connect() as conn:
            await conn.execute(
                text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = :d"),
                {"d": name},
            )
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        await admin.dispose()


@pytest_asyncio.fixture
async def at_head(scratch: _Scratch) -> AsyncIterator[_Scratch]:
    """The revision applied to an EMPTY database, so every data assertion below is against DDL
    this revision built in this test — not against whatever the database already had."""
    await scratch.stamp(_PRIOR)
    await scratch.upgrade(_HEAD)
    yield scratch


# ============================================================ the revision itself
async def test_the_revision_chains_from_the_real_head_and_the_whole_chain_applies(
    scratch: _Scratch,
) -> None:
    """The chain head is VERIFIED from the script directory rather than assumed, and the entire
    chain is then applied to an empty database with this revision on top — which is what an
    operator's first ``alembic upgrade head`` on a fresh deployment actually does.

    **What breaks it:** re-pointing ``down_revision`` at anything but the real previous head, or
    any statement in this revision that a freshly-migrated database rejects.
    """
    script = ScriptDirectory.from_config(scratch.cfg)
    chain_head = script.get_current_head()
    assert chain_head is not None
    assert script.get_revision(_HEAD).down_revision == _PRIOR
    lineage = {revision.revision for revision in script.walk_revisions("base", chain_head)}
    assert _HEAD in lineage, "this revision has been orphaned from the chain reachable from head"

    await scratch.upgrade("head")
    assert await scratch.version() == chain_head
    assert await scratch.tables() == set(_TABLES)

    log = await scratch.columns("room_log")
    for name, nullable in _LATE_COLUMNS.items():
        assert name in log, f"the upgrade did not add {name}: {sorted(log)}"
        assert log[name][0] is nullable, f"{name} has the wrong nullability"
    assert log["dedupe_key"][:2] == (False, MAX_DEDUPE_KEY_CHARS), (
        "dedupe_key's width IS the ceiling the REST edge bounds the Idempotency-Key header to; "
        "when the two drifted, an over-long header became a 22001 that was re-labelled a store "
        "outage and PAUSED a room"
    )
    assert log["body"][0] is False and log["posted_at"][0] is False

    cons = await scratch.constraints("room_log")
    assert cons.get("pk_room_log") == "p"
    assert (
        cons.get("ux_roomlog_dedupe") == "u"
    ), "`ON CONFLICT ON CONSTRAINT ux_roomlog_dedupe` names this constraint literally"
    assert cons.get("ux_roomlog_message_id") == "u"
    assert (await scratch.constraints("room_session")).get("pk_room_session") == "p"
    participant = await scratch.constraints("room_participant")
    assert participant.get("pk_room_participant") == "p"
    assert participant.get("fk_room_participant_session") == "f"

    assert _SESSION_RUNTIME_COLUMNS <= set(await scratch.columns("room_session"))
    assert {"kind", "owner_principal_id", "presence", "binding_id", "capabilities"} <= set(
        await scratch.columns("room_participant")
    )


async def test_up_down_up_on_a_clean_database(scratch: _Scratch) -> None:
    """``downgrade`` really reverses ``upgrade`` when the tables are this revision's own.

    **What breaks it:** dropping any ``op.create_table`` from ``upgrade``, or any drop from
    ``downgrade`` — a ``downgrade`` that does not reverse its ``upgrade`` is how a shared database
    drifts.
    """
    await scratch.stamp(_PRIOR)
    await scratch.upgrade(_HEAD)
    assert await scratch.tables() == set(_TABLES)

    await scratch.downgrade(_PRIOR)
    left = await scratch.tables()
    assert left == set(), f"downgrade left tables it created behind: {sorted(left)}"

    await scratch.upgrade(_HEAD)
    assert await scratch.tables() == set(_TABLES)


# ============================================================ the DIRTY database
async def test_a_hand_rolled_room_log_is_reconciled_and_never_loses_its_messages(
    scratch: _Scratch,
) -> None:
    """The dirty-database property the AD-28 brief made a hard rule, end to end.

    A ``room_log`` seeded by hand from ``PostgresRoomLog.REQUIRED_DDL`` — thirteen columns, a
    narrow ``dedupe_key``, no ``ux_roomlog_dedupe`` — with a message already in it. The upgrade
    must adopt it (all three late columns, the dedupe uniqueness that ``ON CONFLICT ON CONSTRAINT``
    names, the widened key) without touching the row; the downgrade must reverse those additions
    and STILL not touch the row.

    **What breaks it:** reverting the reconcile branch to a no-op (the late columns never arrive);
    reconciling columns but not constraints (``ux_roomlog_dedupe`` stays missing and the first
    client retry raises ``UndefinedObject``); or restoring the unconditional
    ``op.drop_table("room_log")`` in ``downgrade``, which deletes a message this revision never
    wrote.
    """
    await scratch.stamp(_PRIOR)
    await scratch.run(_LEGACY_ROOM_LOG)
    async with scratch.engine.begin() as conn:
        await conn.execute(_LEGACY_ROW, {"seq": 4, "dedupe_key": "legacy-key"})

    await scratch.upgrade(_HEAD)

    assert await scratch.version() == _HEAD
    assert await scratch.scalar("SELECT count(*) FROM room_log") == 1
    log = await scratch.columns("room_log")
    assert set(_LATE_COLUMNS) <= set(log)
    assert log["dedupe_key"][1] == MAX_DEDUPE_KEY_CHARS, "a narrow legacy key was not widened"
    cons = await scratch.constraints("room_log")
    assert cons.get("ux_roomlog_dedupe") == "u", (
        "the reconcile branch skipped the constraint `ON CONFLICT ON CONSTRAINT` names, so every "
        "idempotent replay would raise UndefinedObject"
    )
    assert cons.get("ux_roomlog_message_id") == "u"

    await scratch.downgrade(_PRIOR)

    assert "room_log" in await scratch.tables(), (
        "downgrade dropped a room_log this revision did not create — up added three columns, down "
        "deleted every message"
    )
    assert await scratch.scalar("SELECT count(*) FROM room_log") == 1
    assert set(_LATE_COLUMNS).isdisjoint(
        await scratch.columns("room_log")
    ), "downgrade left the columns it added behind"

    await scratch.upgrade(_HEAD)
    assert set(_LATE_COLUMNS) <= set(await scratch.columns("room_log"))
    assert await scratch.scalar("SELECT count(*) FROM room_log") == 1


async def test_a_short_room_session_is_widened_and_its_cursors_backfilled_from_the_log(
    scratch: _Scratch,
) -> None:
    """**AD-28 item (3)'s recorded incident shape:** the table PRESENT while its sibling columns
    are MISSING. Before this fix the upgrade skipped it silently, ``alembic_version`` advanced to
    head, and the database reported itself fully migrated while every
    ``session_pg._SELECT_SESSION`` would raise ``UndefinedColumn``.

    The backfill is asserted by VALUE, not just by presence: ``last_seq`` /
    ``announced_through_seq`` come from ``MAX(room_log.seq)`` for that room — the ordering
    authority itself — and no ``server_default`` may be left behind afterwards, because a default
    that outlives the migration answers for every future INSERT that forgets the column.

    **What breaks it:** making the reconcile branch cover only ``room_log`` (the shipped defect);
    backfilling the cursors with ``-1``, which re-announces every historical message on the room's
    next post; or leaving the ``server_default`` on instead of dropping it.
    """
    await scratch.stamp(_PRIOR)
    await scratch.run(_LEGACY_ROOM_LOG)
    async with scratch.engine.begin() as conn:
        await conn.execute(_LEGACY_ROW, {"seq": 9, "dedupe_key": "k9"})
    await scratch.run(_SHORT_ROOM_SESSION)
    await scratch.run(
        "INSERT INTO room_session VALUES ('org_legacy', 'ws_1', 'room_1', 'ns_1', 'open', now())"
    )

    await scratch.upgrade(_HEAD)

    session = await scratch.columns("room_session")
    assert _SESSION_RUNTIME_COLUMNS <= set(session), (
        "head was reached with the runtime's columns still missing: "
        f"{sorted(_SESSION_RUNTIME_COLUMNS - set(session))}"
    )
    for column in _SESSION_RUNTIME_COLUMNS:
        assert session[column][0] is False, f"{column} was left nullable"
        assert session[column][2] is None, (
            f"{column} kept a server_default; it would answer for every future INSERT that "
            "forgets the column"
        )
    async with scratch.engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT visibility, floor_policy, last_seq, announced_through_seq, version "
                    "FROM room_session"
                )
            )
        ).one()
    assert row.visibility == "shared"  # spec:79 — a room is SHARED by definition
    assert row.floor_policy == "free_for_all"  # the only IMPLEMENTED_FLOOR_POLICIES member
    assert row.last_seq == 9, "the cursor was not derived from the log's own MAX(seq)"
    assert row.announced_through_seq == 9, (
        "-1 here would re-publish every historical message on the room's next post and "
        "double-count every turn that already reached its clients"
    )
    assert row.version == 0


async def test_a_populated_roster_stops_the_migration_instead_of_inventing_a_participant_kind(
    scratch: _Scratch,
) -> None:
    """A roster row's ``kind`` / ``owner_principal_id`` / ``capabilities`` are not derivable from
    anything true. Defaulting them would relabel an agent as a person — spec:280 makes this
    surface the thing an agent principal is stamped from — and no later error would surface the
    lie. So the migration STOPS, and Postgres's transactional DDL leaves the version row where it
    was, which is a state an operator can still fix.

    **What breaks it:** giving those columns a backfill expression, or dropping the
    ``_NOT_DERIVABLE`` check so they are added with an invented value.
    """
    await scratch.stamp(_PRIOR)
    await scratch.run(_SHORT_ROOM_PARTICIPANT)
    await scratch.run(
        "INSERT INTO room_participant VALUES ('org_legacy', 'ws_1', 'room_1', 'p_alice', now())"
    )

    with pytest.raises(RuntimeError, match="not derivable") as raised:
        await scratch.upgrade(_HEAD)

    # The refusal must name EVERY column it could not derive, not just the first: an operator
    # repairing the table by hand works from this list, and a column quietly given an invented
    # backfill would simply drop out of it.
    assert {"kind", "owner_principal_id", "capabilities", "ordinal"} == {
        name for name in _ROSTER_NOT_DERIVABLE if repr(name) in str(raised.value)
    }
    assert await scratch.version() == _PRIOR, "the version row advanced past a refused upgrade"
    assert "kind" not in await scratch.columns("room_participant")


async def test_an_empty_roster_table_is_reconciled_because_there_is_no_row_to_lie_about(
    scratch: _Scratch,
) -> None:
    """The same short table with NO rows: nothing can be misdescribed, so the columns land NOT
    NULL directly and the FK into ``room_session`` is repaired."""
    await scratch.stamp(_PRIOR)
    await scratch.run(_SHORT_ROOM_PARTICIPANT)

    await scratch.upgrade(_HEAD)

    roster = await scratch.columns("room_participant")
    assert {"kind", "owner_principal_id", "presence", "capabilities", "ordinal"} <= set(roster)
    assert roster["kind"][0] is False
    assert (await scratch.constraints("room_participant")).get("fk_room_participant_session") == "f"


# ============================================================ the data the shape exists to carry
_INSERT = text(
    """
    INSERT INTO room_log (
        org_id, workspace_id, room_id, seq, id, author_principal_id, author_kind, kind,
        correlation_id, content_hash, dedupe_key, to_principal_ids, reply_to_seq, body,
        posted_at, appended_at
    ) VALUES (
        :org_id, :workspace_id, :room_id, :seq, :id, :author_principal_id, :author_kind, :kind,
        :correlation_id, :content_hash, :dedupe_key, :to_principal_ids, :reply_to_seq, :body,
        :posted_at, :appended_at
    )
    """
)

_SELECT = text(
    """
    SELECT id, seq, author_principal_id, author_kind, kind, correlation_id, content_hash,
           dedupe_key, to_principal_ids, reply_to_seq, body, posted_at
      FROM room_log
     WHERE org_id = :org_id AND workspace_id = :workspace_id AND room_id = :room_id
     ORDER BY seq
    """
)

_ORG = "org_roomtest"


def _row(org: str = _ORG, **overrides: object) -> dict[str, object]:
    now = datetime.now(tz=UTC)
    row: dict[str, object] = {
        "org_id": org,
        "workspace_id": "ws_1",
        "room_id": "room_1",
        "seq": 0,
        "id": f"msg_{uuid4()}",
        "author_principal_id": "p_alice",
        "author_kind": ParticipantKind.HUMAN.value,
        "kind": MessageKind.UTTERANCE.value,
        "correlation_id": uuid4().hex,
        "content_hash": "h" * 64,
        "dedupe_key": uuid4().hex,
        "to_principal_ids": None,
        "reply_to_seq": None,
        "body": "hello",
        "posted_at": now,
        "appended_at": now,
    }
    row.update(overrides)
    return row


async def test_a_directed_post_records_who_it_was_for(at_head: _Scratch) -> None:
    """**AD-28 item (1), in its durable form.** Recipients written, read back, rebuilt into the
    ``RoomMessage`` — with the exact ``Addressing`` the author supplied.

    **What breaks it:** removing ``to_principal_ids`` (or ``reply_to_seq``) from the revision.
    That is the pre-fix behaviour: a caller told a message reached ``p_bob``/``p_carol`` while
    nothing recorded who that was.
    """
    addressing = Addressing(to_principal_ids=("p_bob", "p_carol"), reply_to_seq=41)
    async with at_head.engine.begin() as conn:
        await conn.execute(
            _INSERT,
            _row(
                seq=42,
                to_principal_ids=json.dumps(list(addressing.to_principal_ids)),
                reply_to_seq=addressing.reply_to_seq,
            ),
        )

    async with at_head.engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    _SELECT, {"org_id": _ORG, "workspace_id": "ws_1", "room_id": "room_1"}
                )
            )
            .mappings()
            .all()
        )

    assert len(rows) == 1
    row = rows[0]
    stored = RoomMessage(
        id=str(row["id"]),
        room_id="room_1",
        seq=int(row["seq"]),
        author_principal_id=str(row["author_principal_id"]),
        author_kind=ParticipantKind(str(row["author_kind"])),
        kind=MessageKind(str(row["kind"])),
        correlation_id=str(row["correlation_id"]),
        content_hash=str(row["content_hash"]),
        dedupe_key=str(row["dedupe_key"]),
        addressing=Addressing(
            to_principal_ids=tuple(json.loads(str(row["to_principal_ids"]))),
            reply_to_seq=row["reply_to_seq"],
        ),
        body=str(row["body"]),
        posted_at=row["posted_at"],
    )
    assert stored.addressing == addressing
    assert stored.addressing.to_principal_ids == ("p_bob", "p_carol")
    # A message rebuilt from its row is the SAME message, id included — the round trip is what
    # makes `id` a durable handle rather than a value minted afresh on every read.
    reloaded = RoomMessage.model_validate(stored.model_dump())
    assert reloaded == stored
    # The dedupe key is part of spec:77 BECAUSE of reply_to_seq — the round trip has to preserve
    # enough to recompute it.
    assert stored.canonical_dedupe_key() == reloaded.canonical_dedupe_key()


async def test_a_broadcast_row_reads_back_as_the_empty_addressing(at_head: _Scratch) -> None:
    """NULL ``to_principal_ids`` ≡ ``()`` ≡ broadcast (spec:78) — a real domain value, which is
    why the column may be nullable without losing information."""
    async with at_head.engine.begin() as conn:
        await conn.execute(_INSERT, _row(seq=0))
    async with at_head.engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    _SELECT, {"org_id": _ORG, "workspace_id": "ws_1", "room_id": "room_1"}
                )
            )
            .mappings()
            .one()
        )
    assert row["to_principal_ids"] is None
    assert Addressing() == Addressing(to_principal_ids=())


async def test_the_same_seq_cannot_be_written_twice_in_one_room(at_head: _Scratch) -> None:
    """``seq`` is the ordering authority behind the contiguity apply-rule, so duplicate-freedom is
    a property of the DATABASE.

    **What breaks it:** removing ``pk_room_log`` from the revision — the second insert then
    succeeds and one room has two messages claiming turn 7.
    """
    async with at_head.engine.begin() as conn:
        await conn.execute(_INSERT, _row(seq=7))
    with pytest.raises(IntegrityError):
        async with at_head.engine.begin() as conn:
            await conn.execute(_INSERT, _row(seq=7))


async def test_the_same_dedupe_key_cannot_be_written_twice_in_one_room(at_head: _Scratch) -> None:
    """``ux_roomlog_dedupe`` is what makes an idempotent replay resolvable in one statement."""
    key = uuid4().hex
    async with at_head.engine.begin() as conn:
        await conn.execute(_INSERT, _row(seq=0, dedupe_key=key))
    with pytest.raises(IntegrityError):
        async with at_head.engine.begin() as conn:
            await conn.execute(_INSERT, _row(seq=1, dedupe_key=key))


async def test_the_same_seq_in_a_different_tenant_is_a_different_row(at_head: _Scratch) -> None:
    """CANONICAL §1 rule 5: the key BEGINS at ``org_id``. A room id is never a key on its own."""
    other = f"{_ORG}_b"
    async with at_head.engine.begin() as conn:
        await conn.execute(_INSERT, _row(seq=3))
        await conn.execute(_INSERT, _row(other, seq=3))
    async with at_head.engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    _SELECT, {"org_id": other, "workspace_id": "ws_1", "room_id": "room_1"}
                )
            )
            .mappings()
            .all()
        )
    assert len(rows) == 1
