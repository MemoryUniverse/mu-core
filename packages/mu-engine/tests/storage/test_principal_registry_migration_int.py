"""Principal-registry migration round trip — REAL Postgres, THROWAWAY database, ZERO mocks.

Verifies build step 6a (`f6fc5f7052cc_principal_registry_org_and_credentials.py`,
mu-server-phase3-devices-sync-spec.md §4b.2): the `principals` extension and the NEW
`principal_credentials` table survive a real round trip against real Postgres, the empty-table
precondition the migration's NOT-NULL-with-no-server-default columns rest on (§4b.2/D) actually
holds, and the one stated literal default — `status` (§4b.2/A) — is backward compatible with a row
written without it (a raw INSERT that never mentions the column, the shape a pre-Phase-3 writer
would produce).

This exercises the actual Alembic revision (upgrade/downgrade functions), not `Base.metadata.
create_all` — the rest of this package's integration fixtures build tables that way (``conftest.
py::pg_engine``), which proves the ORM model is internally consistent but proves nothing about
whether the migration that a real deployment runs actually works.

**AD-202: this file used to run its downgrade/upgrade cycle directly against the SHARED
mu-dev-postgres control plane.** Every `ADD COLUMN` + `DROP COLUMN` round trip permanently
consumes a slot in Postgres's per-table attribute count (`pg_attribute` never reclaims a dropped
column's `attnum`), and this file does four such transitions per run. Repeated runs against the
one shared database accumulated: `principals` reached 1594 attributes, 1592 of them dropped,
against Postgres's 1600-column ceiling, and `alembic upgrade head` started failing with
`TooManyColumnsError` for everyone — six unrelated storage tests then failed on a column that
genuinely did not exist. `test_room_tables_migration_int.py` and
`test_one_provenance_ledger_migration_int.py` already avoided this by giving themselves a
disposable database; this file now does the same, so no number of repeated runs can move the
shared database's attribute count at all.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import NullPool, make_url, text
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
_HEAD = "f6fc5f7052cc"  # THE REVISION UNDER TEST — not necessarily the chain head, see below
_PRIOR = "1ab7f7175baa"  # the revision this one Revises — build step 2c's landed revision

_PRINCIPAL_COLUMNS = {
    "id",
    "kind",
    "org_id",
    "display_name",
    "status",
    "created_at",
    "updated_at",
    "disabled_at",
    "created_by",
    "source",
}
_CREDENTIAL_COLUMNS = {
    "key_id",
    "kind",
    "org_id",
    "workspace_id",
    "principal_id",
    "hashed_secret",
    "display_prefix",
    "label",
    "created_at",
    "last_used_at",
    "expires_at",
    "revoked_at",
    "revoked_by",
    "rotated_from_key_id",
    "issued_by",
}


class _Scratch:
    """One throwaway database, plus the alembic config pointed at it."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.cfg = Config(str(_ALEMBIC_INI))
        self.cfg.set_main_option("sqlalchemy.url", str(engine.url.render_as_string(False)))

    async def upgrade(self, revision: str = _HEAD) -> None:
        await asyncio.to_thread(command.upgrade, self.cfg, revision)

    async def downgrade(self, revision: str = _PRIOR) -> None:
        await asyncio.to_thread(command.downgrade, self.cfg, revision)


@pytest_asyncio.fixture
async def scratch() -> AsyncIterator[_Scratch]:
    """A database of its own per test, dropped afterwards (AD-202).

    The DSN comes from the central Settings tree (never a literal); only the database NAME is
    replaced, so the credentials and host stay the ones ``.env.test`` wires. ``NullPool`` +
    ``prepared_statement_cache_size=0`` because this test runs DDL (upgrade/downgrade) BETWEEN
    queries on the same database, and asyncpg raises ``InvalidCachedStatementError`` when a
    cached plan outlives the schema it was planned against (the same mitigation
    ``test_one_provenance_ledger_migration_int.py`` uses for the same reason).
    """
    control = make_url(get_settings().storage.postgres.dsn)
    name = f"mu_princmig_{uuid4().hex[:12]}"
    admin = create_async_engine(control, isolation_level="AUTOCOMMIT", poolclass=None)
    engine = create_async_engine(
        control.set(database=name),
        poolclass=NullPool,
        connect_args={"prepared_statement_cache_size": 0},
    )
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


async def _columns(engine: AsyncEngine, table: str) -> dict[str, tuple[bool, str | None]]:
    """{column_name: (is_nullable, column_default)} straight from ``information_schema`` — never
    trust the ORM's own idea of what it wrote; ask the real database."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text(
                    "select column_name, is_nullable, column_default "
                    "from information_schema.columns where table_name = :t"
                ),
                {"t": table},
            )
        ).all()
    return {r.column_name: (r.is_nullable == "YES", r.column_default) for r in rows}


async def test_migration_round_trip_and_backward_compatible_defaults(scratch: _Scratch) -> None:
    engine = scratch.engine

    # Bring the throwaway database up to the revision immediately BEFORE this one — the whole
    # chain from empty, not a downgrade from head, since a fresh scratch database has no head to
    # downgrade from. This asserts the empty-table precondition §4b.2/D rests on for real, not
    # merely trusted because it held during manual review.
    await scratch.upgrade(_PRIOR)

    pre_cols = await _columns(engine, "principals")
    assert set(pre_cols) == {"id", "kind"}, (
        "principals must be the original two-column stub immediately before this revision "
        "runs, or the NOT-NULL-with-no-server-default columns below would be unsafe to add"
    )
    async with engine.connect() as conn:
        n = await conn.scalar(text("select count(*) from principals"))
    assert n == 0, (
        "the migration's NOT NULL columns with no server default are safe ONLY because "
        "principals has zero rows at this revision (§4b.2/D) — a non-empty table here means "
        "the upgrade below is expected to fail, not silently succeed"
    )

    # --- upgrade: the new columns + the new table land -----------------------------------
    await scratch.upgrade(_HEAD)

    post_cols = await _columns(engine, "principals")
    assert set(post_cols) == _PRINCIPAL_COLUMNS
    assert post_cols["org_id"] == (False, None)  # NOT NULL, no server default (§4b.2/D)
    assert post_cols["created_by"] == (False, None)
    assert post_cols["source"] == (False, None)
    assert post_cols["status"][0] is False  # NOT NULL
    assert post_cols["status"][1] is not None and "active" in post_cols["status"][1]
    assert post_cols["display_name"] == (True, None)  # nullable, no default
    assert post_cols["disabled_at"] == (True, None)  # nullable, no default

    cred_cols = await _columns(engine, "principal_credentials")
    assert set(cred_cols) == _CREDENTIAL_COLUMNS
    assert cred_cols["hashed_secret"] == (False, None)  # NOT NULL — the UNIQUE column, D-44
    assert cred_cols["label"][1] is not None and "''" in cred_cols["label"][1]

    # --- the backward-compatible default: a row written WITHOUT `status` -----------------
    # (a pre-Phase-3 writer that only ever knew `id`/`kind` still cannot construct a row that
    # omits every new column — org_id/created_at/updated_at/created_by/source are NOT NULL
    # with no default, by design, §4b.2/D — but `status` and `display_name`/`disabled_at`
    # ARE meant to be omittable, and this proves that they actually are.)
    pid = f"prn_migrationtest_{uuid.uuid4().hex[:12]}"
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "insert into principals "
                "(id, kind, org_id, created_at, updated_at, created_by, source) "
                "values (:id, 'service', :org, now(), now(), 'bootstrap', 'bootstrap')"
            ),
            {"id": pid, "org": "org-migration-test"},
        )
    try:
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text("select status, display_name, disabled_at from principals where id = :id"),
                    {"id": pid},
                )
            ).one()
        assert row.status == "active"  # the one stated literal default, §4b.2/A
        assert row.display_name is None
        assert row.disabled_at is None
    finally:
        # Must not survive into the downgrade -> upgrade round trip below: `downgrade` only drops
        # COLUMNS, never rows, so a row left in `principals` with no `org_id` value would make the
        # re-upgrade's `ADD COLUMN org_id ... NOT NULL` fail with a real NotNullViolationError —
        # MEASURED, the first time this file ran against its own scratch database (AD-202): this
        # `finally` was missing, and the round trip failed on exactly that.
        async with engine.begin() as conn:
            await conn.execute(text("delete from principals where id = :id"), {"id": pid})

    # --- org_id's NOT NULL is enforced by the real database, not merely declared in the ORM
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "insert into principals (id, kind, created_at, updated_at, created_by, "
                    "source) values (:id, 'service', now(), now(), 'bootstrap', 'bootstrap')"
                ),
                {"id": f"prn_shouldfail_{uuid.uuid4().hex[:12]}"},
            )

    # --- downgrade -> upgrade: the round trip itself --------------------------------------
    await scratch.downgrade(_PRIOR)
    assert set(await _columns(engine, "principals")) == {"id", "kind"}
    async with engine.connect() as conn:
        still_there = await conn.scalar(text("select to_regclass('public.principal_credentials')"))
    assert still_there is None, "downgrade must drop principal_credentials, not just columns"

    await scratch.upgrade(_HEAD)
    assert set(await _columns(engine, "principals")) == set(post_cols)
    assert set(await _columns(engine, "principal_credentials")) == set(cred_cols)
