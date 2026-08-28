"""provenance_ledger becomes the ONE ledger — the union shape (AD-106)

There have been TWO provenance tables for one concept. ``provenance_ledger``
(``storage-schema-rowmapper-spec.md`` §2.6, revision ``46ae4bcc2472``) is the memory-lineage
ledger; ``transfer_provenance`` (``mu-server/src/mu_server/transfer/store_pg.py``, never in this
migration history at all) is the governance plane's own, stood up because the shipped
``ProvenanceAction`` had FOUR members while the transfer FSM needs eight — so four of that
subsystem's actions could not be written here (AD-62, and the adapter's own docstring says so).

AD-62 landed the eight members, which removed the only reason the fork existed.
``governance-transfer-core-spec.md`` §4 and §2.6 both describe ONE append-only ledger per object
stream, so two tables is a divergence with no design behind it.

WHAT THIS REVISION DOES, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
This is the EXPAND half of expand-migrate-contract, and only that half.

It makes ``provenance_ledger`` able to hold a governance event losslessly:

* ``event_id`` — NOT NULL, UNIQUE (``ux_prov_ledger_event``): the append idempotency key;
* ``position`` — the org-scoped ledger-scan order, DB-assigned from a sequence on Postgres;
* the twelve governance event columns, every one NULLABLE because a memory-lineage append has no
  value for them;
* ``memory_id`` becomes NULLABLE, because a governance event's subject is
  ``(object_type, object_id)`` and that object is often not a memory;
* the PRIMARY KEY gains ``org_id``. Two orgs whose streams share an id collide on one history
  today; ``transfer_provenance`` already keyed ``(org_id, stream_id, version)`` and this table
  would inherit the defect the moment a second plane wrote to it;
* ``stream_id`` becomes ``VARCHAR(512)`` and ``version`` becomes ``BIGINT``, matching what the
  governance plane's stream ids and append counters actually need.

  ⚠ ``stream_id`` is a WIDENING against the declarative model (``schema.py`` declared
  ``String(128)``) and a NARROWING against the shipped Postgres DDL — ``46ae4bcc2472:312`` emitted
  a bare ``sa.String()``, i.e. an unbounded ``VARCHAR``, so the model and the database have
  disagreed since the initial revision. Stating it the other way round ("it widens") would be
  half true and would hide the only direction that can fail: a pre-existing row whose stream id
  exceeds 512 characters makes ``ALTER`` raise *"value too long"*. That is loud, names the column,
  and is left to Postgres rather than pre-checked, because the length that matters is the one the
  database enforces after this revision — and 512 is the width the governance plane's own table
  chose for the same values.

It does NOT copy ``transfer_provenance``'s rows here, and it does NOT drop that table. Both
belong to the CONTRACT half, and both are unsafe until ``PostgresProvenanceLedger``
(``mu-server``) re-points its six statements at this table — a repo this revision may not edit.
Dropping a table the shipped governance plane still INSERTs into would take that plane down with
no error naming the cause. The two halves are ordered, not optional: this one is additive and
safe to deploy alone; the second lands WITH the adapter change. **REPORTED.**

WHY ``actor_principal_id`` AND ``occurred_at`` ARE NOT ADDED
-----------------------------------------------------------
They already exist here under this table's names: ``actor_principal_id`` IS ``actor_id`` and
``occurred_at`` IS ``at``. Adding them would create two "who acted" and two "when" columns that
can disagree — the shape a merge is supposed to remove, not introduce. The governance adapter
maps them on the way in.

EXISTING ROWS SURVIVE
---------------------
Every added governance column is nullable, so a pre-existing lineage row is untouched by them.
``event_id`` is NOT NULL and therefore the one column needing a backfill; it is DECIDABLE here
(unlike ``acl_entries.permission`` in ``c4a1e07b9d33``, which was refused loudly because every
candidate default GRANTED something). ``event_id`` grants nothing: it only has to be stable and
distinct, so it is derived as ``sha256(org_id \\x1f stream_id \\x1f version)`` — unique by
construction, because those three columns are the new PRIMARY KEY. The derivation runs in Python
over the alembic connection rather than in SQL, so it is identical on Postgres, MySQL and SQLite.
``position`` is left NULL on backfilled rows: a lineage append that predates this revision was
never part of the governance scan order and inventing a position for it would fabricate an order
that no ledger reader ever observed.

Verified against a real Postgres on a throwaway database: ``up -> down -> up`` with rows present
across the whole cycle (``tests/storage/test_one_provenance_ledger_migration_int.py``).

Revision ID: a71f3c9de205
Revises: c4a1e07b9d33
Create Date: 2026-08-28 00:00:00.000000
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Final

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a71f3c9de205"
down_revision: str | None = "c4a1e07b9d33"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Named once so ``upgrade`` and ``downgrade`` cannot disagree about what this revision owns
#: (the discipline ``b3d47c9a1e02`` states, restated in ``c4a1e07b9d33``).
_TABLE: Final = "provenance_ledger"
_PK: Final = "provenance_ledger_pkey"
_UX_EVENT: Final = "ux_prov_ledger_event"
_IX_POSITION: Final = "ix_prov_ledger_position"
_SEQUENCE: Final = "provenance_ledger_position_seq"

#: jsonb on Postgres, json elsewhere — the same variant ``schema.py`` binds.
_JSONB: Final = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")

#: ``{column: type}`` for every governance field this revision adds. ALL nullable: a
#: memory-lineage append has no value for any of them, and a NOT NULL here would make the
#: lineage half unwritable.
_GOVERNANCE_COLUMNS: Final[dict[str, sa.types.TypeEngine[Any]]] = {
    "object_type": sa.String(length=32),
    "object_id": sa.String(length=128),
    "object_content_hash": sa.String(length=128),
    # 384 = this schema's own `namespace_prefix` width, strictly wider than the 256
    # `transfer_provenance` used, so no existing governance row can fail to fit.
    "origin_namespace_id": sa.String(length=384),
    "grantor_principal_id": sa.String(length=128),
    "grantee_kind": sa.String(length=32),
    "grantee_id": sa.String(length=128),
    "grant_id": sa.String(length=128),
    "packet_id": sa.String(length=128),
    "source_refs": _JSONB,
    "content_hash": sa.String(length=128),
    "cascade_root_grant_id": sa.String(length=128),
}

#: A ledger that has never had a writer should hold nothing; a few thousand rows is still a
#: plausible hand-seeded dev database. Beyond this, a row-by-row Python backfill is the wrong
#: tool and the operator should be told so rather than discovering it as an OOM.
_BACKFILL_ROW_CEILING: Final = 1_000_000
#: ASCII unit separator — cannot appear in an id (every id in this schema is a slug/uuid/hash),
#: so the pre-image is unambiguous and two different triples cannot hash the same.
_SEP: Final = "\x1f"


def _event_id_for(org_id: str, stream_id: str, version: int) -> str:
    """Deterministic, collision-free ``event_id`` for a row that predates the column.

    Unique BY CONSTRUCTION rather than by luck: the pre-image is exactly the new PRIMARY KEY
    ``(org_id, stream_id, version)``, joined on a separator no id can contain.
    """
    pre_image = f"{org_id}{_SEP}{stream_id}{_SEP}{version}".encode()
    return hashlib.sha256(pre_image).hexdigest()


def _backfill_event_ids() -> None:
    bind = op.get_bind()
    count = bind.execute(sa.text(f"SELECT count(*) FROM {_TABLE}")).scalar_one()  # noqa: S608
    if count > _BACKFILL_ROW_CEILING:
        raise RuntimeError(
            f"{_TABLE} holds {count} rows, above this revision's {_BACKFILL_ROW_CEILING} "
            "row-by-row backfill ceiling. `event_id` is derivable "
            "(sha256 of org_id/stream_id/version), so nothing here is undecidable — but derive "
            "it with a set-based statement for a table this size, then re-run."
        )
    rows = bind.execute(
        sa.text(f"SELECT org_id, stream_id, version FROM {_TABLE}")  # noqa: S608
    ).all()
    if not rows:
        return
    bind.execute(
        sa.text(
            f"UPDATE {_TABLE} SET event_id = :event_id "  # noqa: S608
            " WHERE org_id = :org_id AND stream_id = :stream_id AND version = :version"
        ),
        [
            {
                "event_id": _event_id_for(str(r.org_id), str(r.stream_id), int(r.version)),
                "org_id": r.org_id,
                "stream_id": r.stream_id,
                "version": r.version,
            }
            for r in rows
        ],
    )


def upgrade() -> None:
    is_postgres = op.get_bind().dialect.name == "postgresql"

    # (1) widen the two lineage columns the governance plane's own values need.
    op.alter_column(
        _TABLE, "stream_id", type_=sa.String(length=512), existing_type=sa.String(), nullable=False
    )
    op.alter_column(
        _TABLE, "version", type_=sa.BigInteger(), existing_type=sa.Integer(), nullable=False
    )
    # (2) a governance event's subject is (object_type, object_id), not a memory.
    op.alter_column(_TABLE, "memory_id", existing_type=sa.String(), nullable=True)

    # (3) event_id: added NULLABLE, backfilled deterministically, THEN tightened. Adding it NOT
    #     NULL in one step would fail on any populated table, and this backfill is decidable.
    op.add_column(_TABLE, sa.Column("event_id", sa.String(length=128), nullable=True))
    _backfill_event_ids()
    op.alter_column(_TABLE, "event_id", existing_type=sa.String(length=128), nullable=False)
    op.create_unique_constraint(_UX_EVENT, _TABLE, ["event_id"])

    # (4) position — the org-scoped ledger-scan order. DB-assigned on Postgres (the shape
    #     `transfer_provenance` already had, so the governance INSERT needs no change); on every
    #     other dialect the column exists and stays NULL, because MySQL cannot AUTO_INCREMENT a
    #     non-key column. Stated in `ProvenanceLedgerRow`'s docstring as a scope limit.
    op.add_column(_TABLE, sa.Column("position", sa.BigInteger(), nullable=True))
    if is_postgres:
        op.execute(sa.text(f"CREATE SEQUENCE {_SEQUENCE} OWNED BY {_TABLE}.position"))
        op.execute(
            sa.text(
                f"ALTER TABLE {_TABLE} ALTER COLUMN position SET DEFAULT nextval('{_SEQUENCE}')"
            )
        )
    op.create_index(_IX_POSITION, _TABLE, ["org_id", "position"], unique=False)

    # (5) the governance event fields.
    for name, type_ in _GOVERNANCE_COLUMNS.items():
        op.add_column(_TABLE, sa.Column(name, type_, nullable=True))

    # (6) org_id joins the PRIMARY KEY. Adding a column to a PK can never introduce a duplicate,
    #     so this direction is always safe; the reverse is not — see `downgrade`.
    op.drop_constraint(_PK, _TABLE, type_="primary")
    op.create_primary_key(_PK, _TABLE, ["org_id", "stream_id", "version"])


def downgrade() -> None:
    is_postgres = op.get_bind().dialect.name == "postgresql"
    bind = op.get_bind()

    # (6') Narrowing the PK back CAN fail: two orgs may have written the same (stream_id,
    #      version) under the wide key, and there is no honest way to pick a survivor. Refuse
    #      with the count rather than let Postgres report a bare duplicate-key error.
    collisions = bind.execute(
        sa.text(
            f"SELECT count(*) FROM (SELECT stream_id, version FROM {_TABLE} "  # noqa: S608
            " GROUP BY stream_id, version HAVING count(*) > 1) AS d"
        )
    ).scalar_one()
    if collisions:
        raise RuntimeError(
            f"cannot narrow {_TABLE}'s primary key back to (stream_id, version): "
            f"{collisions} (stream_id, version) pairs are shared by more than one org. Those "
            "rows only became legal under the wider key this revision added; deleting one org's "
            "history to make room for another's is not a migration decision. Separate them by "
            "hand, then re-run."
        )
    op.drop_constraint(_PK, _TABLE, type_="primary")
    op.create_primary_key(_PK, _TABLE, ["stream_id", "version"])

    # (5')
    for name in reversed(list(_GOVERNANCE_COLUMNS)):
        op.drop_column(_TABLE, name)

    # (4')
    op.drop_index(_IX_POSITION, table_name=_TABLE)
    if is_postgres:
        op.execute(sa.text(f"ALTER TABLE {_TABLE} ALTER COLUMN position DROP DEFAULT"))
    op.drop_column(_TABLE, "position")
    if is_postgres:
        op.execute(sa.text(f"DROP SEQUENCE IF EXISTS {_SEQUENCE}"))

    # (3')
    op.drop_constraint(_UX_EVENT, _TABLE, type_="unique")
    op.drop_column(_TABLE, "event_id")

    # (2') memory_id goes back to NOT NULL, which fails if a governance row (whose subject is an
    #      object, not a memory) is present. That row cannot be represented in the pre-revision
    #      shape at all, so the refusal is the honest answer, not an obstacle.
    orphans = bind.execute(
        sa.text(f"SELECT count(*) FROM {_TABLE} WHERE memory_id IS NULL")  # noqa: S608
    ).scalar_one()
    if orphans:
        raise RuntimeError(
            f"cannot restore {_TABLE}.memory_id to NOT NULL: {orphans} rows have no memory_id "
            "because their subject is a governance object (object_type/object_id), which the "
            "pre-revision shape cannot express. Move those rows out before downgrading."
        )
    op.alter_column(_TABLE, "memory_id", existing_type=sa.String(), nullable=False)

    # (1')
    long_streams = bind.execute(
        sa.text(f"SELECT count(*) FROM {_TABLE} WHERE length(stream_id) > 128")  # noqa: S608
    ).scalar_one()
    if long_streams:
        raise RuntimeError(
            f"cannot narrow {_TABLE}.stream_id back to 128 characters: {long_streams} rows carry "
            "a longer stream id. Truncating a stream id silently re-parents its history onto a "
            "different stream."
        )
    op.alter_column(
        _TABLE, "version", type_=sa.Integer(), existing_type=sa.BigInteger(), nullable=False
    )
    op.alter_column(
        _TABLE,
        "stream_id",
        type_=sa.String(length=128),
        existing_type=sa.String(length=512),
        nullable=False,
    )
