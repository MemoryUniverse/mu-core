"""Principal-registry migration round trip — REAL mu-dev-postgres, alembic upgrade -> downgrade ->
upgrade, ZERO mocks (DEV-STANDARDS).

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
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
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
_HEAD = "f6fc5f7052cc"
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


def _alembic_config() -> Config:
    # sqlalchemy.url is left BLANK on purpose, matching alembic.ini's own comment: env.py sources
    # the DSN from the same central Settings tree (.env.test) the rest of the suite reads.
    return Config(str(_ALEMBIC_INI))


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    eng = create_async_engine(get_settings().storage.postgres.dsn, poolclass=None)
    try:
        yield eng
    finally:
        await eng.dispose()


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


async def test_migration_round_trip_and_backward_compatible_defaults(engine: AsyncEngine) -> None:
    cfg = _alembic_config()

    try:
        # Land immediately BEFORE this revision, so the empty-table precondition §4b.2/D rests on
        # is asserted for real here, not merely trusted because it held during manual review.
        await asyncio.to_thread(command.downgrade, cfg, _PRIOR)

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
        await asyncio.to_thread(command.upgrade, cfg, _HEAD)

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
                        text(
                            "select status, display_name, disabled_at from principals "
                            "where id = :id"
                        ),
                        {"id": pid},
                    )
                ).one()
            assert row.status == "active"  # the one stated literal default, §4b.2/A
            assert row.display_name is None
            assert row.disabled_at is None
        finally:
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
        await asyncio.to_thread(command.downgrade, cfg, _PRIOR)
        assert set(await _columns(engine, "principals")) == {"id", "kind"}
        async with engine.connect() as conn:
            still_there = await conn.scalar(
                text("select to_regclass('public.principal_credentials')")
            )
        assert still_there is None, "downgrade must drop principal_credentials, not just columns"

        await asyncio.to_thread(command.upgrade, cfg, _HEAD)
        assert set(await _columns(engine, "principals")) == set(post_cols)
        assert set(await _columns(engine, "principal_credentials")) == set(cred_cols)
    finally:
        # Leave the shared dev database at head, whatever happened above — other integration
        # fixtures assume the schema they were built against is the one currently live.
        async with engine.connect() as conn:
            current = await conn.scalar(text("select version_num from alembic_version"))
        if current != _HEAD:
            await asyncio.to_thread(command.upgrade, cfg, _HEAD)
