"""Revision ``9c41d0b7ae52`` — REAL Postgres, THROWAWAY database, ZERO mocks.

**Why this file exists: the revision was EXECUTED by the suite and never ASSERTED.** Every other
fixture that touches ``private_sync_log`` builds its tables with ``Base.metadata.create_all``
(``conftest.py::pg_engine`` here, and both of ``mu-server``'s scratch-database integration
suites), which proves the ORM model is internally consistent and proves nothing about the DDL a
real deployment runs. ``test_principal_registry_migration_int.py`` drives ``upgrade(cfg, "head")``
in its own scratch database, so this revision's ``upgrade()`` does get *called* there too — but
nothing in that file looks at the shape it produced. A wrong nullability or a missing
``server_default`` would ship green and surface for the first time on an operator's ``alembic
upgrade head``.

⚠ **It genuinely runs the revision rather than reading whatever the shared database already has.**
Asserting against a database that is already at head would pass no matter what the revision says —
the vacuous-pass mode this project has recorded four times. So the test upgrades to the prior
revision, asserts the three columns are absent, upgrades to this one, and asserts the shape the
upgrade built.

**AD-202: this file used to run its downgrade/upgrade cycle directly against the SHARED
mu-dev-postgres control plane.** Every such round trip against a shared table permanently consumes
a slot in Postgres's per-table attribute count (`pg_attribute` never reclaims a dropped column's
`attnum`) — this file's sibling `test_principal_registry_migration_int.py` did the same thing
against `principals` and drove it to 1594 attributes, 1592 dropped, against Postgres's 1600-column
ceiling, breaking `alembic upgrade head` for everyone. This file now gives itself a disposable
database instead, the same pattern `test_room_tables_migration_int.py` and
`test_one_provenance_ledger_migration_int.py` already used.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import NullPool, make_url, text
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
_HEAD = "9c41d0b7ae52"  # THE REVISION UNDER TEST
_PRIOR = "f6fc5f7052cc"  # the revision it Revises

#: The three columns revision ``9c41d0b7ae52`` adds, as
#: ``{name: (is_nullable, default_is_expected)}``. ``appended_at`` is the one carrying a
#: ``server_default``, and it needs one for a reason worth pinning: it is ``NOT NULL`` and the
#: table may already hold rows, so without the default the upgrade fails on any live deployment.
_ADDED = {"resolved_by": True, "caused_by_seq": True, "appended_at": False}


class _Scratch:
    """One throwaway database, plus the alembic config pointed at it."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.cfg = Config(str(_ALEMBIC_INI))
        self.cfg.set_main_option("sqlalchemy.url", str(engine.url.render_as_string(False)))

    async def upgrade(self, revision: str = _HEAD) -> None:
        await asyncio.to_thread(command.upgrade, self.cfg, revision)


@pytest_asyncio.fixture
async def scratch() -> AsyncIterator[_Scratch]:
    """A database of its own per test, dropped afterwards (AD-202).

    The DSN comes from the central Settings tree (never a literal); only the database NAME is
    replaced, so the credentials and host stay the ones ``.env.test`` wires. ``NullPool`` +
    ``prepared_statement_cache_size=0`` because this test runs DDL (upgrade) BETWEEN queries on
    the same database, and asyncpg raises ``InvalidCachedStatementError`` when a cached plan
    outlives the schema it was planned against (the same mitigation
    ``test_one_provenance_ledger_migration_int.py`` uses for the same reason).
    """
    control = make_url(get_settings().storage.postgres.dsn)
    name = f"mu_synclogmig_{uuid4().hex[:12]}"
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
    """``{column_name: (is_nullable, column_default)}`` straight from ``information_schema`` —
    never trust the ORM's idea of what it wrote; ask the real database."""
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


async def test_the_step_2c_remainder_revision_builds_the_shape_it_promises(
    scratch: _Scratch,
) -> None:
    """Upgrade to PRIOR -> assert absent -> upgrade to HEAD -> assert the shape.

    **What breaks it:** dropping any of the three ``op.add_column`` calls from
    ``9c41d0b7ae52.upgrade``; making ``appended_at`` nullable; or removing its
    ``server_default=sa.text("now()")`` — which is the one that would ship green today and then
    abort a real ``alembic upgrade head`` against a populated ``private_sync_log``.
    """
    engine = scratch.engine

    # The chain applied from empty up to — and stopping at — the revision this one Revises, so
    # the assertion below is against exactly what this revision's `upgrade()` itself builds, not
    # whatever an earlier or later migration happened to leave behind.
    await scratch.upgrade(_PRIOR)
    pre = await _columns(engine, "private_sync_log")
    already_present = sorted(set(pre) & set(_ADDED))
    assert not already_present, (
        "the database at the prior revision must not already have the columns this revision "
        f"adds, so the upgrade below asserts a shape it actually built: {already_present}"
    )
    # The columns the PRIOR revisions built must be present.
    assert {"org_id", "principal_id", "seq", "pinned", "resolution_origin"} <= set(pre)

    await scratch.upgrade(_HEAD)
    post = await _columns(engine, "private_sync_log")
    assert set(_ADDED) <= set(post), f"the upgrade did not add every column: {sorted(post)}"
    for name, nullable in _ADDED.items():
        assert post[name][0] is nullable, f"{name} has the wrong nullability"
    assert post["resolved_by"][1] is None
    assert post["caused_by_seq"][1] is None
    assert post["appended_at"][1] is not None and "now()" in post["appended_at"][1], (
        "appended_at is NOT NULL on a table that may already hold rows, so it needs a "
        "server default or the upgrade aborts on a live deployment"
    )
