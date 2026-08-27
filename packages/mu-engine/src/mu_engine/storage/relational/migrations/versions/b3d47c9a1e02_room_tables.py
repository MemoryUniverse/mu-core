"""room_log, room_session, room_participant — the S3 room runtime's durable home.

Revision ID: b3d47c9a1e02
Revises: 9c41d0b7ae52
Create Date: 2026-08-27

``ARCHITECTURE-DELTAS.md`` AD-28 items (1), (2) and (3), in ONE revision on purpose — AD-28's own
warning is that ``PostgresRoomLog.REQUIRED_DDL`` "bakes in the missing ``addressing``/``id``
columns, so a storage lane building exactly that shape will need a second migration once (1)
lands". This is that lane, and (1) has landed, so the table is built once with the columns the
contract now has.

Why each object is genuinely missing rather than scope creep
------------------------------------------------------------------------------------------------
* ``room_log`` — ``RoomLogRepository`` is the sole ``seq`` ordering authority and had NO table at
  all (verified on the live control plane: ``to_regclass('public.room_log')`` is NULL). The only
  implementation, ``mu_server.rooms.room_log_pg.PostgresRoomLog``, carried its shape as a
  ``REQUIRED_DDL`` module constant and raised at the database on every call, because migrations
  live here and that repo may not edit them.
* ``room_session`` / ``room_participant`` — ``SessionRepository``'s durable home, likewise absent.
  AD-28 item (3) phrases this as missing COLUMNS on the shipped ``sessions`` /
  ``session_participants``; **this revision deliberately does not widen those two.** They are the
  control-plane home of ``SessionDefinition``, whose ``SessionState`` is ``{OPEN, CLOSED}`` while
  the room's ``RoomState`` also has ``PAUSED`` and ``Session.to_definition()`` projects ``PAUSED``
  *down* to ``OPEN`` — one row carrying both state axes makes that projection lossy in the store,
  and widening a shipped table other components already read is the more expensive and less
  reversible of the two migrations. The rationale is stated at length in ``schema.py`` and matches
  the only existing adapter (``mu_server.rooms.session_pg``), which targets these table names.
* ``room_log.id`` / ``to_principal_ids`` / ``reply_to_seq`` — the three columns AD-28 (1) is about.
  Without ``to_principal_ids`` **a DIRECTED post has nowhere to record who it was for**: the
  service accepted ``Addressing.to_principal_ids`` and then dropped it, which is why
  ``RoomService`` had to refuse directed posts outright rather than lie about delivery.

Safe against a DIRTY dev database — ALL THREE tables, not one
------------------------------------------------------------------------------------------------
The shared dev control plane at 127.0.0.1:15432 has drifted before: a half-applied migration left
a table PRESENT while its sibling columns were MISSING, and the next run failed in a way that read
like a code defect. So ``upgrade()`` does not assume a clean slate. For **each** of the three
tables it creates the table when absent and otherwise RECONCILES it — columns, constraints, and
``dedupe_key``'s width — against the canonical shape declared below.

Reconciling only ``room_log`` (the first cut of this revision did exactly that) is worse than not
reconciling at all: a ``room_session`` that is present-but-short passes silently, the version row
advances to this revision, and the database then *reports itself fully migrated* while lacking the
five columns ``mu_server.rooms.session_pg._SELECT_SESSION`` reads — a state no later revision will
ever revisit, because every later revision is entitled to assume this one ran. That is the exact
incident shape AD-28 item (3) exists to answer, so it is answered for every table this revision
owns.

NOT NULL on a table that already holds rows: BACKFILL, not a server default
------------------------------------------------------------------------------------------------
On the CLEAN path nothing here is a ``NOT NULL`` add to a populated table — all three tables are
new, the CREATE carries the constraint, and a table created in this statement has no rows to
violate it. On the RECONCILE path a ``NOT NULL`` column may have to join rows that already exist,
and this revision **backfills** rather than attaching a ``server_default``:

* a ``server_default`` outlives the migration. It would keep answering for every future INSERT
  that forgets the column — which is how a wrong ``visibility`` or a permanently-zero ``version``
  gets written for years without an error. A backfill touches exactly the rows that exist at
  migration time and leaves the column with NO default, so the application must supply it.
* **A column is backfilled only where the value is DERIVABLE from something already true.** Where
  it is not, ``_reconcile_columns`` REFUSES (see ``_NOT_DERIVABLE``): a roster row's ``kind`` or
  ``owner_principal_id`` cannot be guessed, and a participant silently relabelled ``human`` is a
  provenance lie that no later error would surface. Postgres has transactional DDL, so that
  refusal leaves ``alembic_version`` where it was.

The derivations, each from a stated source: ``last_seq``/``announced_through_seq`` from
``MAX(room_log.seq)`` — the ordering authority itself, and equal to each other because the
alternative (``-1``) re-announces every historical message on the room's next post; ``visibility``
from spec:79's invariant ``visibility = SHARED``; ``floor_policy`` from
``IMPLEMENTED_FLOOR_POLICIES``, whose single member is the only policy a legacy row can have been
running; ``version`` from ``0``, a fresh optimistic-concurrency generation, which is correct
precisely because no caller can be holding an expected version for a row that had no version
column.

The three columns that are NULLABLE are nullable for a different reason each, none of them
"unknown value": ``room_log.id`` — the shipped appender is a fixed thirteen-column INSERT in
another repo and cannot write it, and there is no portable server default (``gen_random_uuid()``
is Postgres-only and this schema binds SQLite and MySQL too), so NULL means "written before the
column existed"; ``room_log.to_principal_ids`` — NULL ≡ ``()`` ≡ BROADCAST, a real domain value
(spec:78); ``room_log.reply_to_seq`` — NULL means "not a reply".

What ``downgrade()`` does, exactly — it never deletes rows it cannot recreate
------------------------------------------------------------------------------------------------
An earlier cut of this file claimed ``downgrade()`` was "symmetrically conditional, so a partial
upgrade can always be undone". **That was false and it destroyed data:** the condition was table
PRESENCE, never PROVENANCE, so on the dirty path — where ``upgrade`` had added three columns to a
``room_log`` it did not create — ``downgrade`` dropped the whole table and every message in it.

The rule now is one sentence, and it is the same rule on both paths: **a table is dropped only
when it holds NO rows.** A table that holds rows keeps them, and only the additive changes that
can be named are reversed (``room_log``'s three late columns and ``ux_roomlog_message_id``, which
exists only because of ``id``). ``pk_room_log`` and ``ux_roomlog_dedupe`` are left on a surviving
table on purpose: they are the shape the pre-revision appender in ``mu-server`` already requires,
and dropping them would break a running server in the name of tidiness.

Two consequences, stated rather than discovered: (a) downgrading a populated deployment leaves the
tables behind at revision ``9c41d0b7ae52`` — which is exactly the state the dev control plane was
in before this revision existed, and re-upgrading reconciles it back, so the cycle is stable;
(b) a ``dedupe_key`` widened from a narrower legacy column is NOT narrowed back, because narrowing
can truncate values this revision never wrote. Those are the only asymmetries and they are both
deliberate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, NamedTuple

import sqlalchemy as sa
from alembic import op

revision: str = "b3d47c9a1e02"
down_revision: str | None = "9c41d0b7ae52"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The dedupe-key column width. A LITERAL, not an import of
#: ``mu_contracts.domain.model.room.MAX_DEDUPE_KEY_CHARS``, because a migration is a frozen
#: historical record: importing a live constant would make this revision build a different table
#: next year than it built today. ``tests/storage/test_room_tables_migration_int.py`` asserts the
#: literal and the constant agree, which is the anti-drift belt without the time-travel.
_DEDUPE_KEY_CHARS: Final = 128

#: Sentinel for ``_Col.backfill``: this column's value for an already-existing row is NOT
#: derivable from anything true, so a reconcile that would have to invent one STOPS instead.
_NOT_DERIVABLE: Final = None


class _Col(NamedTuple):
    """One canonical column, used BOTH to create the table and to reconcile a pre-existing one —
    so the two paths cannot describe different shapes."""

    name: str
    type_: sa.types.TypeEngine[Any]
    nullable: bool = False
    #: SQL expression used to fill this column for rows that already exist when it is added late.
    #: Ignored for a nullable column (NULL is a legal value) and for an empty table (there is
    #: nothing to fill). ``_NOT_DERIVABLE`` on a NOT NULL column makes the reconcile refuse.
    backfill: str | None = _NOT_DERIVABLE


def _column(col: _Col) -> sa.Column[Any]:
    return sa.Column(col.name, col.type_, nullable=col.nullable)


# ------------------------------------------------------------------ the canonical shapes
#: The cursor backfill for a legacy ``room_session`` row: the highest ``seq`` the ROOM LOG holds
#: for that room, or ``-1`` when it holds none. Correlated on the full tenancy triple (CANONICAL §1
#: rule 5 — a room id is never a key on its own), which is also why ``upgrade`` reconciles
#: ``room_log`` before ``room_session``: this expression reads it.
_MAX_ROOM_LOG_SEQ: Final = (
    "COALESCE((SELECT MAX(l.seq) FROM room_log l WHERE l.org_id = room_session.org_id "
    "AND l.workspace_id = room_session.workspace_id AND l.room_id = room_session.id), -1)"
)


_ROOM_SESSION_COLUMNS: Final[tuple[_Col, ...]] = (
    _Col("org_id", sa.String(128)),
    _Col("workspace_id", sa.String(128)),
    _Col("id", sa.String(128)),
    _Col("namespace_id", sa.String(128)),
    # spec:79 — a room is SHARED by definition; there is no other legal value for a legacy row.
    _Col("visibility", sa.String(32), backfill="'shared'"),
    _Col("state", sa.String(32), backfill="'open'"),
    # `IMPLEMENTED_FLOOR_POLICIES` has exactly one member, so a row written before this column
    # existed cannot have been running any other policy.
    _Col("floor_policy", sa.String(32), backfill="'free_for_all'"),
    # From the ordering authority itself, never a guess: the room log IS where `seq` lives.
    _Col("last_seq", sa.BigInteger(), backfill=_MAX_ROOM_LOG_SEQ),
    # Deliberately the SAME value as `last_seq`, not -1: `seq > announced_through_seq` is a durable
    # statement that the announce is still OWED, so -1 would re-publish every historical message on
    # the room's next post and double-count every turn that already reached its clients.
    _Col("announced_through_seq", sa.BigInteger(), backfill=_MAX_ROOM_LOG_SEQ),
    # A fresh optimistic-concurrency generation. Correct rather than convenient: no caller can be
    # holding an expected version for a row that had no version column to read.
    _Col("version", sa.BigInteger(), backfill="0"),
    _Col("created_at", sa.DateTime(timezone=True), backfill="now()"),
)

_ROOM_PARTICIPANT_COLUMNS: Final[tuple[_Col, ...]] = (
    _Col("org_id", sa.String(128)),
    _Col("workspace_id", sa.String(128)),
    _Col("room_id", sa.String(128)),
    _Col("principal_id", sa.String(128)),
    # NOT DERIVABLE, and this is the point of the sentinel: defaulting a roster row to `human`
    # would relabel an agent as a person, and spec:280 makes this surface the thing an agent
    # principal is stamped from. A provenance lie no later error would surface.
    _Col("kind", sa.String(32), backfill=_NOT_DERIVABLE),
    _Col("display_name", sa.String(200), nullable=True),
    _Col("joined_at", sa.DateTime(timezone=True), backfill="now()"),
    _Col("left_at", sa.DateTime(timezone=True), nullable=True),
    _Col("presence", sa.String(32), backfill="'offline'"),  # advisory, never an authz input
    _Col("owner_principal_id", sa.String(128), backfill=_NOT_DERIVABLE),
    _Col("binding_id", sa.String(128), nullable=True),
    _Col("capabilities", sa.Text(), backfill=_NOT_DERIVABLE),
    _Col("ordinal", sa.Integer(), backfill=_NOT_DERIVABLE),
)

_ROOM_LOG_COLUMNS: Final[tuple[_Col, ...]] = (
    _Col("org_id", sa.String(128)),
    _Col("workspace_id", sa.String(128)),
    _Col("room_id", sa.String(128)),
    _Col("seq", sa.BigInteger()),
    _Col("id", sa.String(160), nullable=True),
    _Col("author_principal_id", sa.String(128)),
    _Col("author_kind", sa.String(32)),
    _Col("kind", sa.String(32)),
    _Col("correlation_id", sa.String(128)),
    _Col("content_hash", sa.String(128)),
    _Col("dedupe_key", sa.String(_DEDUPE_KEY_CHARS)),
    _Col("to_principal_ids", sa.Text(), nullable=True),
    _Col("reply_to_seq", sa.BigInteger(), nullable=True),
    _Col("body", sa.Text()),
    _Col("posted_at", sa.DateTime(timezone=True)),
    _Col("appended_at", sa.DateTime(timezone=True)),
)

#: ``{table: (name, columns)}`` — the primary key each table must end up with, whichever path got
#: it there. ``pk_room_log`` is the gap-free/duplicate-free guarantee every contiguity rule
#: downstream rests on, so its absence on a hand-rolled table is a defect to repair, not a variant.
_PRIMARY_KEYS: Final[dict[str, tuple[str, list[str]]]] = {
    "room_session": ("pk_room_session", ["org_id", "workspace_id", "id"]),
    "room_participant": (
        "pk_room_participant",
        ["org_id", "workspace_id", "room_id", "principal_id"],
    ),
    "room_log": ("pk_room_log", ["org_id", "workspace_id", "room_id", "seq"]),
}

#: Spelled literally: ``ON CONFLICT ON CONSTRAINT ux_roomlog_dedupe`` names it, so a hand-rolled
#: table that arrives WITHOUT it turns the idempotent-replay branch into an ``UndefinedObject`` on
#: the first client retry — neither an ``IntegrityError`` nor a ``RoomError``, which is the class
#: of error this project already recorded being re-labelled a store outage that PAUSED a room.
_UNIQUE_CONSTRAINTS: Final[dict[str, tuple[tuple[str, list[str]], ...]]] = {
    "room_log": (
        ("ux_roomlog_dedupe", ["org_id", "workspace_id", "room_id", "dedupe_key"]),
        # The surrogate id is an IDENTITY, so it is unique per room where present at all.
        ("ux_roomlog_message_id", ["org_id", "workspace_id", "room_id", "id"]),
    ),
}

_FK_NAME: Final = "fk_room_participant_session"

#: ``room_log`` columns added AFTER ``PostgresRoomLog.REQUIRED_DDL`` was written. Named once so
#: ``upgrade``'s reconcile branch and ``downgrade`` cannot disagree about what this revision owns.
_ROOM_LOG_LATE_COLUMNS: Final[tuple[str, ...]] = ("id", "to_principal_ids", "reply_to_seq")


# ------------------------------------------------------------------ inspection helpers
def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _existing_tables() -> set[str]:
    return set(_inspector().get_table_names())


def _existing_columns(table: str) -> dict[str, Any]:
    return {c["name"]: c for c in _inspector().get_columns(table)}


def _has_rows(table: str) -> bool:
    # Table names here are module constants, never caller input.
    probe = sa.text(f"SELECT 1 FROM {table} LIMIT 1")  # noqa: S608
    return op.get_bind().execute(probe).first() is not None


# ------------------------------------------------------------------ create
def _create(table: str, columns: Sequence[_Col]) -> None:
    args: list[Any] = [_column(c) for c in columns]
    pk_name, pk_cols = _PRIMARY_KEYS[table]
    args.append(sa.PrimaryKeyConstraint(*pk_cols, name=pk_name))
    for name, cols in _UNIQUE_CONSTRAINTS.get(table, ()):
        args.append(sa.UniqueConstraint(*cols, name=name))
    if table == "room_participant":
        args.append(
            sa.ForeignKeyConstraint(
                ["org_id", "workspace_id", "room_id"],
                ["room_session.org_id", "room_session.workspace_id", "room_session.id"],
                name=_FK_NAME,
                ondelete="CASCADE",
            )
        )
    op.create_table(table, *args)


# ------------------------------------------------------------------ reconcile
def _reconcile_columns(table: str, columns: Sequence[_Col]) -> None:
    present = _existing_columns(table)
    missing = [c for c in columns if c.name not in present]
    if not missing:
        return
    populated = _has_rows(table)
    undecidable = [c.name for c in missing if not c.nullable and c.backfill is _NOT_DERIVABLE]
    if populated and undecidable:
        raise RuntimeError(
            f"cannot reconcile `{table}`: it already holds rows and is missing NOT NULL "
            f"column(s) {sorted(undecidable)} whose value for an existing row is not derivable "
            "from anything true. Inventing one would write a provenance lie no later error would "
            "surface. Empty or repair the table by hand, then re-run this revision."
        )
    for col in missing:
        if col.nullable or not populated:
            op.add_column(table, _column(col))
            continue
        # Add nullable -> backfill the rows that exist NOW -> tighten. No `server_default` is
        # left behind: see the module docstring for why that is the whole point.
        op.add_column(table, sa.Column(col.name, col.type_, nullable=True))
        # Table, column and backfill are all module constants declared above, never caller input.
        fill = f"UPDATE {table} SET {col.name} = {col.backfill} WHERE {col.name} IS NULL"  # noqa: S608
        op.execute(sa.text(fill))
        op.alter_column(table, col.name, nullable=False)


def _reconcile_dedupe_key_width() -> None:
    """A legacy ``room_log`` may carry a NARROWER ``dedupe_key`` than the edge now bounds the
    ``Idempotency-Key`` header to — the drift that produced a ``22001`` and PAUSED a room. Widen
    it; never narrow it (that would truncate values this revision did not write)."""
    column = _existing_columns("room_log").get("dedupe_key")
    if column is None:
        return
    width = getattr(column["type"], "length", None)
    if width is not None and width < _DEDUPE_KEY_CHARS:
        op.alter_column(
            "room_log",
            "dedupe_key",
            type_=sa.String(_DEDUPE_KEY_CHARS),
            existing_nullable=False,
        )


def _reconcile_constraints(table: str) -> None:
    inspector = _inspector()
    pk_name, pk_cols = _PRIMARY_KEYS[table]
    if not inspector.get_pk_constraint(table).get("constrained_columns"):
        op.create_primary_key(pk_name, table, pk_cols)
    existing = {c["name"] for c in inspector.get_unique_constraints(table)}
    for name, cols in _UNIQUE_CONSTRAINTS.get(table, ()):
        if name not in existing:
            op.create_unique_constraint(name, table, cols)
    if table == "room_participant" and not inspector.get_foreign_keys(table):
        op.create_foreign_key(
            _FK_NAME,
            "room_participant",
            "room_session",
            ["org_id", "workspace_id", "room_id"],
            ["org_id", "workspace_id", "id"],
            ondelete="CASCADE",
        )


def _create_or_reconcile(table: str, columns: Sequence[_Col], tables: set[str]) -> None:
    if table not in tables:
        _create(table, columns)
        return
    _reconcile_columns(table, columns)
    if table == "room_log":
        _reconcile_dedupe_key_width()
    _reconcile_constraints(table)


def upgrade() -> None:
    tables = _existing_tables()
    # `room_log` FIRST: `room_session`'s cursor backfill reads MAX(room_log.seq) from it.
    _create_or_reconcile("room_log", _ROOM_LOG_COLUMNS, tables)
    _create_or_reconcile("room_session", _ROOM_SESSION_COLUMNS, tables)
    _create_or_reconcile("room_participant", _ROOM_PARTICIPANT_COLUMNS, tables)


def _strip_room_log_late_columns() -> None:
    """The reverse of the reconcile branch, on a ``room_log`` that still holds messages."""
    if "ux_roomlog_message_id" in {
        c["name"] for c in _inspector().get_unique_constraints("room_log")
    }:
        op.drop_constraint("ux_roomlog_message_id", "room_log", type_="unique")
    present = _existing_columns("room_log")
    for name in _ROOM_LOG_LATE_COLUMNS:
        if name in present:
            op.drop_column("room_log", name)


def downgrade() -> None:
    tables = _existing_tables()
    # Reverse creation order: the participant rows carry the FK into `room_session`.
    if "room_log" in tables:
        if _has_rows("room_log"):
            _strip_room_log_late_columns()
        else:
            op.drop_table("room_log")
    participant_dropped = False
    if "room_participant" in tables and not _has_rows("room_participant"):
        op.drop_table("room_participant")
        participant_dropped = True
    # A `room_session` still referenced by a surviving `room_participant` cannot be dropped, and a
    # populated one is never dropped at all: see the module docstring's one-sentence rule.
    session_free = participant_dropped or "room_participant" not in tables
    if "room_session" in tables and session_free and not _has_rows("room_session"):
        op.drop_table("room_session")
