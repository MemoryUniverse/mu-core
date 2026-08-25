"""Revision ``9c41d0b7ae52`` — REAL mu-dev-postgres, alembic downgrade -> upgrade, ZERO mocks.

**Why this file exists: the revision was EXECUTED by the suite and never ASSERTED.** Every other
fixture that touches ``private_sync_log`` builds its tables with ``Base.metadata.create_all``
(``conftest.py::pg_engine`` here, and both of ``mu-server``'s scratch-database integration
suites), which proves the ORM model is internally consistent and proves nothing about the DDL a
real deployment runs. ``test_principal_registry_migration_int.py`` drives ``upgrade(cfg, "head")``
in its cleanup, so this revision's ``upgrade()`` does get *called* — but nothing looks at the shape
it produced. A wrong nullability or a missing ``server_default`` would ship green and surface for
the first time on an operator's ``alembic upgrade head``.

⚠ **It genuinely runs the revision rather than reading whatever the shared database already has.**
Asserting against a database that is already at head would pass no matter what the revision says —
the vacuous-pass mode this project has recorded four times. So the test downgrades to the prior
revision, asserts the three columns are GONE, upgrades, and asserts the shape the upgrade built.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
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

#: Restore the chain's ACTUAL head, not ``_HEAD`` — the same trap
#: ``test_principal_registry_migration_int.py`` records: the other fixtures here use
#: ``create_all`` (``checkfirst``: it creates missing TABLES, never adds a missing COLUMN), so a
#: database left one revision short stays short and fails a *different file* with no hint why.
_CHAIN_HEAD = "head"

#: The three columns revision ``9c41d0b7ae52`` adds, as
#: ``{name: (is_nullable, default_is_expected)}``. ``appended_at`` is the one carrying a
#: ``server_default``, and it needs one for a reason worth pinning: it is ``NOT NULL`` and the
#: table may already hold rows, so without the default the upgrade fails on any live deployment.
_ADDED = {"resolved_by": True, "caused_by_seq": True, "appended_at": False}


def _alembic_config() -> Config:
    return Config(str(_ALEMBIC_INI))


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(get_settings().storage.postgres.dsn, poolclass=None)
    try:
        yield eng
    finally:
        await eng.dispose()


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
    engine: AsyncEngine,
) -> None:
    """Downgrade -> assert absent -> upgrade -> assert the shape.

    **What breaks it:** dropping any of the three ``op.add_column`` calls from
    ``9c41d0b7ae52.upgrade``; making ``appended_at`` nullable; or removing its
    ``server_default=sa.text("now()")`` — which is the one that would ship green today and then
    abort a real ``alembic upgrade head`` against a populated ``private_sync_log``.
    """
    cfg = _alembic_config()
    try:
        await asyncio.to_thread(command.downgrade, cfg, _PRIOR)
        pre = await _columns(engine, "private_sync_log")
        assert set(pre) & set(_ADDED) == set(), (
            "the downgrade did not drop the columns this revision adds, so the upgrade below "
            f"would assert a shape it did not build: {sorted(set(pre) & set(_ADDED))}"
        )
        # The columns the PRIOR revisions built must survive the downgrade untouched.
        assert {"org_id", "principal_id", "seq", "pinned", "resolution_origin"} <= set(pre)

        await asyncio.to_thread(command.upgrade, cfg, _HEAD)
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
    finally:
        # Put the SHARED database back at the chain head, whatever that is today.
        await asyncio.to_thread(command.upgrade, cfg, _CHAIN_HEAD)
