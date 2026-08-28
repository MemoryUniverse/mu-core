"""usage_event_chain, usage_chain_head — the metering ledger's durable home.

Revision ID: e7c1b4d90a36
Revises: a71f3c9de205
Create Date: 2026-08-28

``ARCHITECTURE-DELTAS.md`` AD-161: the meter is built, proven against real Postgres, and
UNREACHABLE. Mounting it without these tables makes every request spool-and-log, so the mount and
this revision are one change — and this is the half that lives here, because
``mu_server/control_plane/migrate.py:3-4`` records that ``mu-server`` *"does NOT own the migration
history"*.

Why NOT the ``usage_event`` table this schema already ships
------------------------------------------------------------------------------------------------
``usage_event`` / ``usage_outbox`` / ``usage_rollup`` exist (``schema.py:509-566``, revision
``46ae4bcc2472``) and have **zero writers anywhere in the system**. The design's intent is real and
shipped; what was missing is a producer. That table still cannot hold the row the hosted plane's
edge can take, and each reason is a guarantee that would be silently lost by widening it instead
(the full seven are enumerated in ``mu_server.metering.ledger_pg``'s module docstring). The four
that decide it:

* its PRIMARY KEY is ``event_id`` alone and ``ix_usage_ws_seq`` is NON-unique, so two rows may
  share a ``(workspace_id, seq)`` — the append-only promise is unenforceable there, and a chain
  that cannot enforce it only looks verifiable;
* its ``seq`` is keyed per ``workspace_id``, not per ``(org_id, workspace_id)``. ``org`` and
  ``workspace`` are INDEPENDENT assertion claims and the default workspace name is the literal
  ``"shared"``, so two orgs picking the same workspace name would share one ``seq`` authority and
  each other's hash links (CANONICAL §1 rule 5 — a name is never a key on its own);
* its ``product`` column is ``NOT NULL`` and nothing on the hosted plane can compute it. Writing a
  fabricated constant into a billing column is worse than a null;
* it has no head watermark, so a TRUNCATED TAIL is undetectable — a replay of the links alone
  cannot see it.

So this revision adds a **separate, correctly-shaped** pair rather than widening a shipped table
that other components may yet read. ``usage_event`` is deliberately left exactly as it is: it is
the ROUTER's dimension set (``tokens_*``, ``model``, ``role``, ``engine``, ``tier``), which is a
different producer with a different grain, and folding the two is a design decision no migration
gets to make on its own. **REPORTED** — the end state may well be one table.

Every constraint here is a spec line, not a preference
------------------------------------------------------------------------------------------------
* ``pk_usage_event_chain (org_id, workspace_id, seq)`` — §B6's per-workspace monotonic sequence.
  ``PostgresUsageLedger`` appends under ``pg_advisory_xact_lock``; the PRIMARY KEY is what turns a
  lost race into a loud error instead of a silent overwrite.
* ``ux_usage_event_chain_id (org_id, workspace_id, event_id)`` — §B8's idempotency key. The
  adapter's ``IntegrityError`` branch reads the stored row back and mints no second ``seq``; that
  branch is only correct because this constraint exists. Duplicate suppression that lives only in
  application code is one retry away from a double-charged customer.
* ``prev_hash NOT NULL`` — §B6's genesis is a VALUE (``'GENESIS'``), never a ``NULL``, so link 1
  hashes by the same rule as every other link and a verifier needs no special case.
* ``ck_usage_event_chain_quantity CHECK (quantity >= 0)`` — a negative quantity is a credit note,
  and this table rates nothing (§B2: no price anywhere in the capture layer).
* ``usage_chain_head`` — the monotonic head watermark, and not decoration: without it a truncated
  tail leaves no gap and no broken link, so ``verify_chain`` would answer ``ok=True`` for a class
  of tampering it cannot see.
* ``to_prefix VARCHAR(768)`` — a real six-segment η prefix over 128-char components is longer than
  the 128 the shipped ``usage_event`` uses, and the failure mode of a too-narrow billing column is
  an INSERT error, i.e. a LOST billable event, not a truncation.
* **No column can hold text a caller wrote** (CLAUDE.md rule 3): every one is an id, a bounded
  enum value, a non-negative count, a hash or a timestamp. ``mu-server``'s
  ``tests/acceptance/test_metering_content_free.py`` parses the adapter's DDL constant and asserts
  each column maps to a ``UsageEvent`` field, so a free-text column cannot be added on that side
  without a RED test; ``tests/storage/test_usage_meter_chain_migration_int.py`` states the same
  column set independently here.

⚠ NO ORM MODEL DECLARES THESE TWO TABLES, AND THAT IS A REPORTED TRAP, NOT A DESIGN
------------------------------------------------------------------------------------------------
Every other table this chain migrates is also declared in ``schema.py`` — ``room_log``,
``room_session``, ``room_participant`` and ``trust_ledger_entries`` are modelled there even though
only ``mu-server`` ever reads or writes them. These two are not, so ``Base.metadata`` does not
contain them while the migrated database does. Nothing breaks today (``mu-server``'s adapter uses
raw SQL and needs no model, and ``env.py`` declares no ``include_object`` filter), but the next
``alembic revision --autogenerate`` run against a migrated database will compare
model-without-table against database-with-table and propose ``op.drop_table("usage_event_chain")``
— a generated revision that deletes the ledger the invoice is computed from.

**REPORTED, not fixed here:** the fix is two declarative classes in
``mu_engine/storage/relational/schema.py``, which is outside this lane's file ownership, and the
exact shape is stated three times already (this revision, ``mu_server.metering.ledger_pg``'s
``REQUIRED_DDL``, and ``tests/storage/test_usage_meter_chain_migration_int.py``). Adding a fourth
statement blind, in a file another lane may be editing, is how a shape acquires a fifth.

Safe against a database an operator already touched by hand
------------------------------------------------------------------------------------------------
``mu-server``'s ``app.py`` logs ``remedy="apply mu_server.<subsystem>.REQUIRED_DDL"`` on a startup
probe failure — hand-applied DDL is a documented operational path on this plane, not corruption.
So ``upgrade()`` does not assume a clean slate: for EACH table it creates when absent and otherwise
RECONCILES columns, constraints and indexes against the canonical shape declared below.

Reconciling only the table it happens to find is worse than not reconciling at all (the lesson
``b3d47c9a1e02`` records): a ``usage_chain_head`` that is missing while ``usage_event_chain`` is
present would pass silently, the version row would advance, and the database would then *report
itself fully migrated* while the only object that can detect a truncated tail does not exist. No
later revision will ever revisit it, because every later revision is entitled to assume this one
ran.

**A reconcile never invents a billing value.** Every ``NOT NULL`` column here is a measurement or a
tenancy id, and there is no derivable default for *which org was charged* or *who to charge*. A
missing NOT NULL column on a POPULATED table therefore stops the revision with a message naming it
(``_NOT_DERIVABLE``); Postgres's transactional DDL leaves ``alembic_version`` where it was, so the
refusal is safe to retry once a human has looked. The nullable columns are added freely — NULL is
a legal value for each (``correlation_id`` when the edge minted no ``jti``; ``route`` on an
unmatched template; ``device_id`` for a non-device caller; ``latency_bucket`` /
``request_bytes`` / ``response_bytes`` when the measure was not takeable, which the adapter
deliberately records as a null rather than a fabricated ``0``).

What ``downgrade()`` does, exactly — it never deletes a billing row
------------------------------------------------------------------------------------------------
``observability-metering-spec.md`` §A0: *"the invoice is computed from these rows and from nothing
else."* So the rule is one sentence: **the pair is dropped only when it holds nothing.** If either
table has rows, both are left in place and only the version row moves — which is exactly the state
this revision reconciles on the way back up, so the cycle is stable. Dropping a populated ledger to
satisfy symmetry would destroy the only record an invoice can be recomputed from; dropping the
watermark while the chain survives would leave a chain whose truncation is undetectable. The pair
is treated as ONE unit for that second reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, NamedTuple

import sqlalchemy as sa
from alembic import op

revision: str = "e7c1b4d90a36"
down_revision: str | None = "a71f3c9de205"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CHAIN: Final = "usage_event_chain"
_HEAD: Final = "usage_chain_head"

#: Sentinel for ``_Col.backfill``: this column's value for an already-existing row is NOT derivable
#: from anything true, so a reconcile that would have to invent one STOPS instead. EVERY NOT NULL
#: column in this revision carries it — see the module docstring.
_NOT_DERIVABLE: Final = None


class _Col(NamedTuple):
    """One canonical column, used BOTH to create the table and to reconcile a pre-existing one — so
    the two paths cannot describe different shapes."""

    name: str
    type_: sa.types.TypeEngine[Any]
    nullable: bool = False
    backfill: str | None = _NOT_DERIVABLE


def _column(col: _Col) -> sa.Column[Any]:
    return sa.Column(col.name, col.type_, nullable=col.nullable)


# ------------------------------------------------------------------ the canonical shapes
#: Widths are LITERALS, never imports of the live model's constants: a migration is a frozen
#: historical record, and importing a live bound would make this revision build a different table
#: next year than it built today. The anti-drift belt is the independent restatement in
#: ``tests/storage/test_usage_meter_chain_migration_int.py``, not a shared import.
_CHAIN_COLUMNS: Final[tuple[_Col, ...]] = (
    _Col("org_id", sa.String(128)),
    _Col("workspace_id", sa.String(128)),
    _Col("seq", sa.BigInteger()),
    _Col("event_id", sa.String(64)),
    _Col("occurred_at", sa.DateTime(timezone=True)),
    _Col("correlation_id", sa.String(128), nullable=True),
    _Col("principal_id", sa.String(128)),
    _Col("namespace_user", sa.String(128)),
    _Col("session_id", sa.String(128)),
    _Col("visibility", sa.String(16)),
    _Col("to_prefix", sa.String(768)),
    _Col("deployment_mode", sa.String(32)),
    _Col("dimension", sa.String(32)),
    _Col("source", sa.String(32)),
    _Col("quantity", sa.BigInteger()),
    _Col("unit", sa.String(16)),
    _Col("route", sa.String(120), nullable=True),
    _Col("device_id", sa.String(128), nullable=True),
    _Col("latency_bucket", sa.String(16), nullable=True),
    _Col("request_bytes", sa.BigInteger(), nullable=True),
    _Col("response_bytes", sa.BigInteger(), nullable=True),
    _Col("prev_hash", sa.String(64)),
    _Col("row_hash", sa.String(64)),
)

_HEAD_COLUMNS: Final[tuple[_Col, ...]] = (
    _Col("org_id", sa.String(128)),
    _Col("workspace_id", sa.String(128)),
    _Col("seq", sa.BigInteger()),
    _Col("row_hash", sa.String(64)),
    _Col("updated_at", sa.DateTime(timezone=True)),
)

_COLUMNS: Final[dict[str, tuple[_Col, ...]]] = {
    _CHAIN: _CHAIN_COLUMNS,
    _HEAD: _HEAD_COLUMNS,
}

_PRIMARY_KEYS: Final[dict[str, tuple[str, list[str]]]] = {
    _CHAIN: ("pk_usage_event_chain", ["org_id", "workspace_id", "seq"]),
    _HEAD: ("pk_usage_chain_head", ["org_id", "workspace_id"]),
}

_UNIQUE_CONSTRAINTS: Final[dict[str, tuple[tuple[str, list[str]], ...]]] = {
    _CHAIN: (("ux_usage_event_chain_id", ["org_id", "workspace_id", "event_id"]),),
}

#: ``{table: ((name, expression), ...)}``. The expression is SQL because a CHECK is DDL, and it is
#: a module literal — never caller input.
_CHECK_CONSTRAINTS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    _CHAIN: (("ck_usage_event_chain_quantity", "quantity >= 0"),),
}

#: The rater folds by ``(org, workspace, occurred_at)`` and a customer's SELF view adds
#: ``principal_id``; without these a monthly rollup on a busy workspace is a sequential scan of the
#: whole ledger.
_INDEXES: Final[dict[str, tuple[tuple[str, list[str]], ...]]] = {
    _CHAIN: (
        ("ix_usage_event_chain_rollup", ["org_id", "workspace_id", "occurred_at"]),
        (
            "ix_usage_event_chain_principal",
            ["org_id", "workspace_id", "principal_id", "occurred_at"],
        ),
    ),
}


# ------------------------------------------------------------------ inspection helpers
def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def _existing_tables() -> set[str]:
    return set(_inspector().get_table_names())


def _existing_columns(table: str) -> set[str]:
    return {c["name"] for c in _inspector().get_columns(table)}


def _has_rows(table: str) -> bool:
    # Table names here are module constants, never caller input.
    probe = sa.text(f"SELECT 1 FROM {table} LIMIT 1")  # noqa: S608
    return op.get_bind().execute(probe).first() is not None


# ------------------------------------------------------------------ create / reconcile
def _create(table: str) -> None:
    args: list[Any] = [_column(c) for c in _COLUMNS[table]]
    pk_name, pk_cols = _PRIMARY_KEYS[table]
    args.append(sa.PrimaryKeyConstraint(*pk_cols, name=pk_name))
    for name, cols in _UNIQUE_CONSTRAINTS.get(table, ()):
        args.append(sa.UniqueConstraint(*cols, name=name))
    for name, expression in _CHECK_CONSTRAINTS.get(table, ()):
        args.append(sa.CheckConstraint(expression, name=name))
    op.create_table(table, *args)
    for name, cols in _INDEXES.get(table, ()):
        op.create_index(name, table, cols)


def _reconcile_columns(table: str) -> None:
    present = _existing_columns(table)
    missing = [c for c in _COLUMNS[table] if c.name not in present]
    if not missing:
        return
    populated = _has_rows(table)
    undecidable = [c.name for c in missing if not c.nullable and c.backfill is _NOT_DERIVABLE]
    if populated and undecidable:
        raise RuntimeError(
            f"cannot reconcile `{table}`: it already holds rows and is missing NOT NULL "
            f"column(s) {sorted(undecidable)} whose value for an existing row is not derivable "
            "from anything true. Every NOT NULL column here is a measurement or a tenancy id, and "
            "inventing one would attribute a charge to a tenant that did not incur it. Empty or "
            "repair the table by hand, then re-run this revision."
        )
    for col in missing:
        if col.nullable or not populated:
            op.add_column(table, _column(col))
            continue
        # Unreachable while every NOT NULL column is `_NOT_DERIVABLE` (the branch above has already
        # raised). Kept because the alternative is a silent nullable add the day someone gives one
        # of these a backfill: add nullable -> fill the rows that exist NOW -> tighten, leaving NO
        # `server_default` behind for future INSERTs to lean on.
        op.add_column(table, sa.Column(col.name, col.type_, nullable=True))
        fill = f"UPDATE {table} SET {col.name} = {col.backfill} WHERE {col.name} IS NULL"  # noqa: S608
        op.execute(sa.text(fill))
        op.alter_column(table, col.name, nullable=False)


def _reconcile_constraints(table: str) -> None:
    inspector = _inspector()
    pk_name, pk_cols = _PRIMARY_KEYS[table]
    if not inspector.get_pk_constraint(table).get("constrained_columns"):
        op.create_primary_key(pk_name, table, pk_cols)
    existing_unique = {c["name"] for c in inspector.get_unique_constraints(table)}
    for name, cols in _UNIQUE_CONSTRAINTS.get(table, ()):
        if name not in existing_unique:
            op.create_unique_constraint(name, table, cols)
    existing_check = {c["name"] for c in inspector.get_check_constraints(table)}
    for name, expression in _CHECK_CONSTRAINTS.get(table, ()):
        if name not in existing_check:
            op.create_check_constraint(name, table, sa.text(expression))


def _reconcile_indexes(table: str) -> None:
    existing = {ix["name"] for ix in _inspector().get_indexes(table)}
    for name, cols in _INDEXES.get(table, ()):
        if name not in existing:
            op.create_index(name, table, cols)


def _create_or_reconcile(table: str, tables: set[str]) -> None:
    if table not in tables:
        _create(table)
        return
    _reconcile_columns(table)
    _reconcile_constraints(table)
    _reconcile_indexes(table)


def upgrade() -> None:
    tables = _existing_tables()
    for table in (_CHAIN, _HEAD):
        _create_or_reconcile(table, tables)


def downgrade() -> None:
    tables = _existing_tables()
    present = [table for table in (_CHAIN, _HEAD) if table in tables]
    if any(_has_rows(table) for table in present):
        # The one-sentence rule, stated in the module docstring: a populated ledger is the only
        # record an invoice can be recomputed from, and the watermark is what makes a truncated
        # chain detectable. Neither is dropped, and re-upgrading reconciles them back.
        return
    for table in present:
        op.drop_table(table)
