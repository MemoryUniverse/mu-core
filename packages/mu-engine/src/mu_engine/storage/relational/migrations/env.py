"""Alembic environment — async, schema from ``schema.Base.metadata``, DSN from Settings.

The migration DSN is NOT hardcoded: it flows from the central Settings tree
(``mu_contracts.config.get_settings().storage.postgres.dsn``, DEV-STANDARDS rule 3), so the
same migration runs against the mu-dev-postgres container in dev and the real server in prod.
Runs the async engine via ``connection.run_sync`` (no blocking driver needed — asyncpg only).
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from mu_contracts.config import get_settings
from mu_engine.storage.relational.schema import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _dsn() -> str:
    # sourced from the central Settings boundary (never a literal).
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    return get_settings().storage.postgres.dsn


def run_migrations_offline() -> None:
    context.configure(
        url=_dsn(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection: object) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)  # type: ignore[arg-type]
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_dsn(), poolclass=None)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
