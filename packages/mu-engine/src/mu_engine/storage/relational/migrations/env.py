"""Alembic environment — async, schema from ``schema.Base.metadata``, DSN from Settings.

The migration DSN is NOT hardcoded: it flows from the central Settings tree
(``mu_contracts.config.get_settings().storage.postgres.dsn``, DEV-STANDARDS rule 3), so the
same migration runs against the mu-dev-postgres container in dev and the real server in prod.
Runs the async engine via ``connection.run_sync`` (no blocking driver needed — asyncpg only).

``include_object`` — a NARROW filter, and the narrowness is the point
------------------------------------------------------------------------------------------------
``target_metadata`` is the whole comparison basis for ``alembic revision --autogenerate``, so
anything the migrations create that ``Base.metadata`` does not describe is proposed for DELETION
on the next autogenerate run. The AUTOGENERATE-DROPS-THE-LEDGER audit — the trap revision
``e7c1b4d90a36`` reports on itself, under *"⚠ NO ORM MODEL DECLARES THESE TWO TABLES"* — found
two such classes at once, and they want OPPOSITE fixes:

* **Two migrated TABLES had no declarative model** (``usage_event_chain``/``usage_chain_head``),
  and a real run generated ``op.drop_table('usage_event_chain')`` — a revision that deletes the
  ledger an invoice is computed from. The fix for that is a MODEL, in ``schema.py``, not a filter.
  Filtering a table out would leave the schema module lying about what the database holds and
  would silence every future drift in it as well: no ``ALTER`` on a billing column would ever be
  proposed again. **No table is filtered here, and the regression test asserts that.**
* **Two migrated INDEXES cannot be expressed in ``Base.metadata`` at all** — ``gin_prov_meta`` and
  ``gin_conflict_members`` are Postgres-only GIN indexes on JSON columns, declared as
  ``DDL(...).execute_if(dialect="postgresql")`` ``after_create`` listeners at the bottom of
  ``schema.py`` precisely because MySQL 8 rejects an index on a raw JSON column outright and a
  portable ``Index(...)`` in ``__table_args__`` would break ``create_all`` on that dialect (see
  that module's docstring, item 2). SQLAlchemy has no per-dialect conditional ``Index``, so there
  is no shape that both survives MySQL and appears in the metadata Postgres is compared against.
  Autogenerate therefore proposed ``op.drop_index('gin_prov_meta')`` +
  ``op.drop_index('gin_conflict_members')`` on the same run — MEASURED. That is the textbook case
  for ``include_object``: an object that exists in the database BY DESIGN and is unrepresentable
  in the target metadata.

So the filter is an explicit, named, index-only allowlist. It is deliberately NOT
"skip anything absent from the metadata" — that rule would have hidden the ledger drop itself.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import Any, Final

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from mu_contracts.config import get_settings
from mu_engine.storage.relational.schema import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

#: The ONLY objects autogenerate is allowed to ignore, by exact name. Both are the Postgres-only
#: GIN indexes created by ``schema.py``'s ``after_create`` DDL listeners and by revision
#: ``46ae4bcc2472``; neither can live in ``Base.metadata`` without breaking MySQL DDL. Adding a
#: TABLE to this set is a bug, not a fix — see the module docstring, and the guard in
#: ``tests/storage/test_schema_autogenerate_drift_int.py``.
DDL_ONLY_INDEXES: Final[frozenset[str]] = frozenset({"gin_prov_meta", "gin_conflict_members"})


def include_object(
    object_: Any, name: str | None, type_: str, reflected: bool, compare_to: Any
) -> bool:
    """Exclude ONLY the named Postgres-only GIN indexes; include everything else, always."""
    return not (type_ == "index" and name in DDL_ONLY_INDEXES)


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
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run(connection: object) -> None:
    context.configure(
        connection=connection,  # type: ignore[arg-type]
        target_metadata=target_metadata,
        compare_type=True,
        include_object=include_object,
    )
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
