"""Revision ``e7c1b4d90a36`` (AD-161) — REAL Postgres, THROWAWAY databases, ZERO mocks.

Every test here creates its own database, applies the real chain to it, and drops it. That is not
tidiness: a test that asserts against a database already at head proves nothing about the revision
(the vacuous pass this project has now recorded six times), and ``alembic downgrade`` against the
SHARED control plane at 127.0.0.1:15432 leaves it a revision short for every other lane.
``test_room_tables_migration_int.py`` and ``test_one_provenance_ledger_migration_int.py``
established this pattern for exactly those reasons.

What is proven, and what each proof is FOR
------------------------------------------------------------------------------------------------
1. **The whole chain applies with this revision on top**, and afterwards ``usage_event_chain`` and
   ``usage_chain_head`` exist with the exact column set, nullability, widths, constraints and
   indexes ``mu_server.metering.ledger_pg`` reads. That module's statements name
   ``pk_usage_event_chain``, ``ux_usage_event_chain_id`` and every column literally, so a shape
   that differs by one column is a metering plane that raises on its first append — which is
   exactly the state AD-161 is about.
2. **up → down → up on a clean database.** A revision that cannot be undone cannot be deployed
   behind a rollback plan.
3. **The three constraints the ledger's guarantees REST on are enforced by Postgres, not by
   Python.** ``seq`` cannot be reused (the append-only promise), ``event_id`` cannot be duplicated
   (the idempotency promise, i.e. never double-billed), and ``quantity`` cannot be negative (a
   negative quantity is a credit note, and this table rates nothing).
4. **``downgrade`` never deletes a billing row.** The invoice is computed from these rows and from
   nothing else (``observability-metering-spec.md`` §A0), so a populated ledger survives a
   downgrade and the tables are left in place.
5. **A hand-applied HALF of the pair is reconciled, not stepped over.** Operators are told to
   apply ``REQUIRED_DDL`` by hand (``app.py``'s ``remedy=`` lines say so), so the "table PRESENT
   while its sibling is MISSING" state AD-28 was written about is reachable here by an ordinary
   operational path. A revision that skipped the missing half would stamp the version row and
   leave the database *reporting itself fully migrated* while ``usage_chain_head`` — the only
   thing that can see a truncated tail — does not exist.
6. **A reconcile that would have to INVENT a billing value REFUSES**, and leaves
   ``alembic_version`` where it was. Every ``NOT NULL`` column here is a measurement or a tenancy
   id; there is no such thing as a derivable default for "which org was charged".
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Executable, NullPool, make_url, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mu_contracts.config import get_settings

pytestmark = pytest.mark.integration

_ALEMBIC_INI = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mu_engine"
    / "storage"
    / "relational"
    / "alembic.ini"
)
_REVISION = "e7c1b4d90a36"  # THE REVISION UNDER TEST — never assumed to be today's chain head
_PRIOR = "a71f3c9de205"

_CHAIN = "usage_event_chain"
_HEAD = "usage_chain_head"

#: The shape ``mu_server.metering.ledger_pg.REQUIRED_DDL`` states, restated INDEPENDENTLY here as
#: ``{column: (is_nullable, char_max_length)}``.
#:
#: ⚠ Deliberately not imported: ``mu-core`` is the OPEN repo and may not import ``mu-server`` at
#: all (that is the whole boundary rule), and even if it could, a shape that agrees with itself
#: proves nothing. This is the third independent statement of the same table — the adapter's SQL,
#: the migration, and this list — and the whole point is that all three must be edited together.
_CHAIN_COLUMNS: dict[str, tuple[bool, int | None]] = {
    "org_id": (False, 128),
    "workspace_id": (False, 128),
    "seq": (False, None),
    "event_id": (False, 64),
    "occurred_at": (False, None),
    "correlation_id": (True, 128),
    "principal_id": (False, 128),
    "namespace_user": (False, 128),
    "session_id": (False, 128),
    "visibility": (False, 16),
    # 768, not the 128 the shipped `usage_event` table uses: a real six-segment `to_prefix` over
    # 128-char components is longer than 128 chars, and the failure mode of a too-narrow column
    # here is an INSERT error, i.e. a LOST billable event, never a truncation.
    "to_prefix": (False, 768),
    "deployment_mode": (False, 32),
    "dimension": (False, 32),
    "source": (False, 32),
    "quantity": (False, None),
    "unit": (False, 16),
    "route": (True, 120),
    "device_id": (True, 128),
    "latency_bucket": (True, 16),
    "request_bytes": (True, None),
    "response_bytes": (True, None),
    "prev_hash": (False, 64),
    "row_hash": (False, 64),
}

_HEAD_COLUMNS: dict[str, tuple[bool, int | None]] = {
    "org_id": (False, 128),
    "workspace_id": (False, 128),
    "seq": (False, None),
    "row_hash": (False, 64),
    "updated_at": (False, None),
}

#: One complete billing row, written through raw SQL exactly as the adapter writes it. Content-free
#: by construction: ids, bounded enum values, counts, hashes and a timestamp — nothing here is text
#: a caller wrote.
_APPEND = text(
    """
    INSERT INTO usage_event_chain
        (org_id, workspace_id, seq, event_id, occurred_at, correlation_id, principal_id,
         namespace_user, session_id, visibility, to_prefix, deployment_mode, dimension, source,
         quantity, unit, route, device_id, latency_bucket, request_bytes, response_bytes,
         prev_hash, row_hash)
    VALUES (:org_id, 'ws_1', :seq, :event_id, now(), 'corr_1', 'prn_alice', '*', 'ses_1',
            'shared', 'mu/org/ws_1/shared/*/ses_1', 'hosted', :dimension, 'api', :quantity,
            'count', '/v1/memories/recall', 'dev_1', 'fast', 512, 1024, 'GENESIS', :row_hash)
    """
)

#: ``usage_event_chain`` as an operator's hand-application of an EARLIER cut of ``REQUIRED_DDL``
#: would have left it: the three measures the edge learned to take last are absent, and so is the
#: head table. This is the AD-28 incident shape, reached by the path ``app.py``'s ``remedy=`` lines
#: actually send an operator down.
_HALF_APPLIED_CHAIN = """
CREATE TABLE usage_event_chain (
    org_id          VARCHAR(128) NOT NULL,
    workspace_id    VARCHAR(128) NOT NULL,
    seq             BIGINT       NOT NULL,
    event_id        VARCHAR(64)  NOT NULL,
    occurred_at     TIMESTAMPTZ  NOT NULL,
    correlation_id  VARCHAR(128),
    principal_id    VARCHAR(128) NOT NULL,
    namespace_user  VARCHAR(128) NOT NULL,
    session_id      VARCHAR(128) NOT NULL,
    visibility      VARCHAR(16)  NOT NULL,
    to_prefix       VARCHAR(768) NOT NULL,
    deployment_mode VARCHAR(32)  NOT NULL,
    dimension       VARCHAR(32)  NOT NULL,
    source          VARCHAR(32)  NOT NULL,
    quantity        BIGINT       NOT NULL,
    unit            VARCHAR(16)  NOT NULL,
    route           VARCHAR(120),
    prev_hash       VARCHAR(64)  NOT NULL,
    row_hash        VARCHAR(64)  NOT NULL,
    CONSTRAINT pk_usage_event_chain PRIMARY KEY (org_id, workspace_id, seq),
    CONSTRAINT ux_usage_event_chain_id UNIQUE (org_id, workspace_id, event_id)
)
"""

#: The same table missing a NOT NULL column whose value for an already-existing row cannot be
#: derived from anything true. ``principal_id`` is WHO IS CHARGED.
_CHAIN_WITHOUT_PRINCIPAL = """
CREATE TABLE usage_event_chain (
    org_id          VARCHAR(128) NOT NULL,
    workspace_id    VARCHAR(128) NOT NULL,
    seq             BIGINT       NOT NULL,
    event_id        VARCHAR(64)  NOT NULL,
    occurred_at     TIMESTAMPTZ  NOT NULL,
    namespace_user  VARCHAR(128) NOT NULL,
    session_id      VARCHAR(128) NOT NULL,
    visibility      VARCHAR(16)  NOT NULL,
    to_prefix       VARCHAR(768) NOT NULL,
    deployment_mode VARCHAR(32)  NOT NULL,
    dimension       VARCHAR(32)  NOT NULL,
    source          VARCHAR(32)  NOT NULL,
    quantity        BIGINT       NOT NULL,
    unit            VARCHAR(16)  NOT NULL,
    prev_hash       VARCHAR(64)  NOT NULL,
    row_hash        VARCHAR(64)  NOT NULL,
    CONSTRAINT pk_usage_event_chain PRIMARY KEY (org_id, workspace_id, seq)
)
"""

#: A row already in a HAND-APPLIED ``usage_event_chain``, written before the revision ran. It is
#: what makes the reconcile assertions real: an empty table can be repaired by any means.
_HALF_APPLIED_ROW = text(
    """
    INSERT INTO usage_event_chain
        (org_id, workspace_id, seq, event_id, occurred_at, correlation_id, principal_id,
         namespace_user, session_id, visibility, to_prefix, deployment_mode, dimension, source,
         quantity, unit, route, prev_hash, row_hash)
    VALUES ('org_legacy', 'ws_1', 0, 'evt_legacy', now(), 'corr_1', 'prn_alice', '*', 'ses_1',
            'shared', 'mu/org_legacy/ws_1/shared/*/ses_1', 'hosted', 'api_request', 'api', 1,
            'count', '/memories', 'GENESIS', 'h0')
    """
)

#: The same, on the table that never had a ``principal_id`` column at all.
_LEGACY_ROW = text(
    """
    INSERT INTO usage_event_chain
        (org_id, workspace_id, seq, event_id, occurred_at, namespace_user, session_id,
         visibility, to_prefix, deployment_mode, dimension, source, quantity, unit,
         prev_hash, row_hash)
    VALUES ('org_legacy', 'ws_1', 0, 'evt_legacy', now(), '*', 'ses_1', 'shared',
            'mu/org_legacy/ws_1/shared/*/ses_1', 'hosted', 'api_request', 'api', 1, 'count',
            'GENESIS', 'h0')
    """
)


class _Scratch:
    """One throwaway database, plus the alembic config pointed at it."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.cfg = Config(str(_ALEMBIC_INI))
        self.cfg.set_main_option("sqlalchemy.url", str(engine.url.render_as_string(False)))

    async def upgrade(self, revision: str = _REVISION) -> None:
        await asyncio.to_thread(command.upgrade, self.cfg, revision)

    async def downgrade(self, revision: str = _PRIOR) -> None:
        await asyncio.to_thread(command.downgrade, self.cfg, revision)

    async def run(self, statement: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text(statement))

    async def execute(self, statement: Executable, params: dict[str, Any] | None = None) -> None:
        """Typed ``Executable``, not ``object`` + an ignore: the ignore that was here silenced the
        overload check on the very argument these tests vary, so a params dict that no longer
        matched the statement would have type-checked."""
        async with self.engine.begin() as conn:
            await conn.execute(statement, params or {})

    async def version(self) -> str | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(text("SELECT version_num FROM alembic_version"))).first()
        return None if row is None else str(row[0])

    async def exists(self, table: str) -> bool:
        async with self.engine.connect() as conn:
            found = (
                await conn.execute(
                    text("SELECT to_regclass(CAST(:t AS TEXT))"), {"t": f"public.{table}"}
                )
            ).scalar()
        return found is not None

    async def columns(self, table: str) -> dict[str, tuple[bool, int | None]]:
        """``{column: (is_nullable, char_max_length)}`` straight from ``information_schema`` —
        never the ORM's idea of what it wrote; ask the database."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT column_name, is_nullable, character_maximum_length "
                        "FROM information_schema.columns WHERE table_name = :t"
                    ),
                    {"t": table},
                )
            ).all()
        return {
            str(r.column_name): (r.is_nullable == "YES", r.character_maximum_length) for r in rows
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

    async def indexes(self, table: str) -> set[str]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text("SELECT indexname FROM pg_indexes WHERE tablename = :t"), {"t": table}
                )
            ).all()
        return {str(r.indexname) for r in rows}

    async def count(self, table: str) -> int:
        async with self.engine.connect() as conn:
            # `table` is a module constant here, never caller input.
            return int(
                (await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))).scalar() or 0  # noqa: S608
            )


@pytest_asyncio.fixture
async def scratch() -> AsyncIterator[_Scratch]:
    """A database of its own per test, dropped afterwards.

    The DSN comes from the central Settings tree (never a literal); only the database NAME is
    replaced, so the credentials and host stay the ones ``.env.test`` wires.
    """
    control = make_url(get_settings().storage.postgres.dsn)
    name = f"mu_metermig_{uuid4().hex[:12]}"
    admin = create_async_engine(control, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    engine = create_async_engine(control.set(database=name), poolclass=NullPool)
    try:
        async with admin.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{name}"'))
        yield _Scratch(engine)
    finally:
        await engine.dispose()
        async with admin.connect() as conn:
            await conn.execute(
                text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = :d"),
                {"d": name},
            )
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        await admin.dispose()


# ============================================================ the revision itself
async def test_the_revision_chains_from_the_prior_head_and_builds_the_shape_the_ledger_reads(
    scratch: _Scratch,
) -> None:
    """The chain is walked from the script directory rather than assumed, the WHOLE chain is then
    applied to an empty database — what an operator's first ``alembic upgrade head`` does — and the
    resulting tables are compared column by column against the shape the metering adapter's SQL
    names literally.

    **What breaks it:** re-pointing ``down_revision``; dropping any column, width, constraint or
    index from ``upgrade``.
    """
    script = ScriptDirectory.from_config(scratch.cfg)
    chain_head = script.get_current_head()
    assert chain_head is not None
    assert script.get_revision(_REVISION).down_revision == _PRIOR
    lineage = {revision.revision for revision in script.walk_revisions("base", chain_head)}
    assert _REVISION in lineage, "this revision is orphaned from the chain reachable from head"

    await scratch.upgrade("head")
    assert await scratch.version() == chain_head

    assert await scratch.columns(_CHAIN) == _CHAIN_COLUMNS
    assert await scratch.columns(_HEAD) == _HEAD_COLUMNS

    chain_constraints = await scratch.constraints(_CHAIN)
    assert chain_constraints.get("pk_usage_event_chain") == "p", (
        "`PostgresUsageLedger` appends under `pg_advisory_xact_lock` and relies on the PRIMARY KEY "
        "to turn a lost race into an error rather than an overwrite"
    )
    assert (
        chain_constraints.get("ux_usage_event_chain_id") == "u"
    ), "the idempotency key: without it a replayed event mints a second `seq` and double-bills"
    assert chain_constraints.get("ck_usage_event_chain_quantity") == "c"
    assert (await scratch.constraints(_HEAD)).get("pk_usage_chain_head") == "p"

    assert {
        "ix_usage_event_chain_rollup",
        "ix_usage_event_chain_principal",
    } <= await scratch.indexes(_CHAIN), "the rater's monthly fold would be a sequential scan"


async def test_up_down_up_on_a_clean_database(scratch: _Scratch) -> None:
    """``downgrade`` really reverses ``upgrade`` when the tables are this revision's own, and a
    re-``upgrade`` rebuilds them — the rollback plan, executed rather than asserted.

    **What breaks it:** dropping either ``op.create_table`` from ``upgrade``, or either drop from
    ``downgrade``.
    """
    await scratch.upgrade("head")
    assert await scratch.exists(_CHAIN) and await scratch.exists(_HEAD)

    await scratch.downgrade(_PRIOR)
    assert await scratch.version() == _PRIOR
    assert not await scratch.exists(_CHAIN)
    assert not await scratch.exists(_HEAD)

    await scratch.upgrade(_REVISION)
    assert await scratch.version() == _REVISION
    assert await scratch.columns(_CHAIN) == _CHAIN_COLUMNS
    assert await scratch.columns(_HEAD) == _HEAD_COLUMNS


async def test_the_database_enforces_append_only_idempotent_non_negative(
    scratch: _Scratch,
) -> None:
    """The three promises the metering package makes are enforced HERE, by Postgres.

    ``PostgresUsageLedger`` catches ``IntegrityError`` and turns it into an idempotent read-back;
    that branch is only correct because the constraint exists. Application-side duplicate
    suppression on a billing ledger is one retry away from a double-charged customer.

    **What breaks it:** removing ``pk_usage_event_chain``, ``ux_usage_event_chain_id`` or
    ``ck_usage_event_chain_quantity`` from the revision.
    """
    await scratch.upgrade("head")
    row = {
        "org_id": "org_a",
        "seq": 0,
        "event_id": "evt_1",
        "dimension": "recall",
        "quantity": 1,
        "row_hash": "h0",
    }
    await scratch.execute(_APPEND, row)

    with pytest.raises(IntegrityError):  # same seq, different event -> never an overwrite
        await scratch.execute(_APPEND, {**row, "event_id": "evt_2", "row_hash": "h1"})

    with pytest.raises(IntegrityError):  # same event replayed -> never a second seq
        await scratch.execute(_APPEND, {**row, "seq": 1, "row_hash": "h1"})

    # A negative quantity is a credit note, and this table rates nothing.
    with pytest.raises(IntegrityError):
        await scratch.execute(
            _APPEND, {**row, "seq": 2, "event_id": "evt_3", "quantity": -1, "row_hash": "h2"}
        )

    # A DIFFERENT org may reuse both keys: `seq` is per `(org, workspace)`, which is the correction
    # `ledger_pg.py` makes against §B6's "the workspace".
    await scratch.execute(_APPEND, {**row, "org_id": "org_b"})
    assert await scratch.count(_CHAIN) == 2


async def test_downgrade_never_deletes_a_billing_row(scratch: _Scratch) -> None:
    """§A0: *"the invoice is computed from these rows and from nothing else."* So a downgrade on a
    populated ledger leaves the rows — and therefore the tables — in place. The version row still
    moves, which is what makes the state recoverable by re-upgrading.

    **What breaks it:** an unconditional ``op.drop_table`` in ``downgrade`` — the exact defect
    ``b3d47c9a1e02``'s docstring records having shipped once, where the condition was table
    PRESENCE rather than emptiness.
    """
    await scratch.upgrade("head")
    await scratch.execute(
        _APPEND,
        {
            "org_id": "org_a",
            "seq": 0,
            "event_id": "evt_1",
            "dimension": "ingest",
            "quantity": 1,
            "row_hash": "h0",
        },
    )
    await scratch.run(
        "INSERT INTO usage_chain_head (org_id, workspace_id, seq, row_hash, updated_at) "
        "VALUES ('org_a', 'ws_1', 0, 'h0', now())"
    )

    await scratch.downgrade(_PRIOR)
    assert await scratch.version() == _PRIOR
    assert await scratch.exists(_CHAIN), "a downgrade deleted a billing row"
    assert await scratch.count(_CHAIN) == 1
    assert await scratch.exists(
        _HEAD
    ), "dropping the watermark while the chain survives makes a truncated tail invisible"

    # And the state is recoverable: re-upgrading reconciles rather than colliding with the tables
    # the downgrade left behind.
    await scratch.upgrade(_REVISION)
    assert await scratch.count(_CHAIN) == 1


async def test_a_hand_applied_half_of_the_pair_is_reconciled_not_stepped_over(
    scratch: _Scratch,
) -> None:
    """``app.py``'s ``remedy=`` lines tell an operator to apply ``REQUIRED_DDL`` by hand, so a
    database carrying an EARLIER cut of that DDL — and no ``usage_chain_head`` at all — is an
    ordinary operational state, not a corruption.

    A revision that saw ``usage_event_chain`` and stepped over the whole pair would stamp the
    version row and leave the database reporting itself fully migrated while the watermark that
    detects a truncated tail does not exist. No later revision would ever revisit it.

    **What breaks it:** short-circuiting ``upgrade`` on "the table already exists".
    """
    await scratch.upgrade(_PRIOR)
    await scratch.run(_HALF_APPLIED_CHAIN)
    await scratch.execute(_HALF_APPLIED_ROW)

    await scratch.upgrade(_REVISION)

    assert await scratch.version() == _REVISION
    assert await scratch.columns(_CHAIN) == _CHAIN_COLUMNS, "the existing table was not repaired"
    assert await scratch.columns(_HEAD) == _HEAD_COLUMNS, "the missing half was never created"
    assert await scratch.count(_CHAIN) == 1, "the reconcile lost a billing row"
    assert (await scratch.constraints(_CHAIN)).get("ck_usage_event_chain_quantity") == "c"
    assert {
        "ix_usage_event_chain_rollup",
        "ix_usage_event_chain_principal",
    } <= await scratch.indexes(_CHAIN)


async def test_a_reconcile_that_would_invent_a_billing_value_refuses(scratch: _Scratch) -> None:
    """There is no derivable default for *who was charged*. A backfilled ``principal_id`` would
    attribute one tenant's usage to another and no later error would surface it, so the revision
    stops — and Postgres's transactional DDL leaves ``alembic_version`` where it was, which is what
    makes the refusal safe to retry after a human has looked.

    **What breaks it:** giving ``principal_id`` (or any other NOT NULL measure) a backfill
    expression or a ``server_default``.
    """
    await scratch.upgrade(_PRIOR)
    await scratch.run(_CHAIN_WITHOUT_PRINCIPAL)
    await scratch.execute(_LEGACY_ROW)

    with pytest.raises(Exception, match="principal_id"):
        await scratch.upgrade(_REVISION)

    assert await scratch.version() == _PRIOR, "the refusal must not leave the version row advanced"
    assert await scratch.count(_CHAIN) == 1
