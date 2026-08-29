"""The schema↔migration drift gate. REAL Postgres, THROWAWAY databases, ZERO mocks.

What went wrong, and why a test is the fix rather than two models
------------------------------------------------------------------------------------------------
``usage_event_chain`` and ``usage_chain_head`` (revision ``e7c1b4d90a36``) were the only migrated
tables with no declarative class in ``mu_engine/storage/relational/schema.py``. ``migrations/
env.py`` hands ``Base.metadata`` to alembic as the comparison target and declared no
``include_object``, so a routine ``alembic revision --autogenerate`` compared model-ABSENT against
database-WITH-TABLE and generated, verbatim, against a real migrated database::

    def upgrade() -> None:
        op.drop_table('usage_chain_head')
        op.drop_index(op.f('ix_usage_event_chain_principal'), table_name='usage_event_chain')
        op.drop_index(op.f('ix_usage_event_chain_rollup'), table_name='usage_event_chain')
        op.drop_table('usage_event_chain')
        op.drop_index(op.f('gin_conflict_members'), table_name='conflict_records', ...)
        op.drop_index(op.f('gin_prov_meta'), table_name='memory_provenance', ...)

Applying that generated revision deletes the ledger the invoice is computed from
(``observability-metering-spec.md`` §A0: *"the invoice is computed from these rows and from
nothing else"*), and drops two Postgres-only GIN indexes along the way. Nobody has to be careless
for this to ship: ``--autogenerate`` is the ORDINARY way to write the next migration, and the
generated file looks like every other generated file.

The two models close today's instance. **This module is the durable half** — it fails the day the
NEXT table is migrated without one, which is the failure mode that actually recurs. It states the
rule three different ways on purpose:

1. :func:`test_every_migrated_table_has_a_declarative_model` — the rule itself, read straight off
   a migrated database's ``information_schema``. It does not care what alembic thinks and would
   still fail if ``env.py`` were later given a filter that hid the table from autogenerate.
2. :func:`test_autogenerate_against_a_migrated_database_proposes_nothing` — the REAL
   ``alembic revision --autogenerate``, into a throwaway version directory that is deleted
   afterwards. Strictly stronger than (1): it also catches a model that DISAGREES with its
   migration, which is worse than no model at all, because the generated op is then a quiet
   ``op.alter_column`` on a billing column instead of an obvious ``op.drop_table``.
3. :func:`test_the_gate_still_detects_an_unmodelled_table` — the control that keeps the other two
   honest. It creates a table with no model and asserts BOTH gates go red. Without it, "no ops
   were proposed" is indistinguishable from "the comparison did not really run" — the vacuous
   pass this project has recorded repeatedly.

Every test creates its own database, applies the real chain, and drops it. Asserting against the
shared control plane at 127.0.0.1:15432 would prove nothing about the chain (it is already at
head) and would leave debris for every other lane —
``test_usage_meter_chain_migration_int.py`` and ``test_room_tables_migration_int.py`` established
the pattern for exactly those reasons.
"""

from __future__ import annotations

import ast
import asyncio
import shutil
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
from mu_engine.storage.relational.schema import Base

pytestmark = pytest.mark.integration

_RELATIONAL = Path(__file__).resolve().parents[2] / "src" / "mu_engine" / "storage" / "relational"
_ALEMBIC_INI = _RELATIONAL / "alembic.ini"
_REAL_VERSIONS = _RELATIONAL / "migrations" / "versions"

#: Alembic's own bookkeeping table. It is created by the migration runner, not by a revision, and
#: it is the ONE table that legitimately has no declarative model — alembic excludes it from
#: autogenerate itself. Spelled out rather than pattern-matched so a future ``mu_*_version`` table
#: cannot slip through on a prefix.
_ALEMBIC_VERSION = "alembic_version"

#: The unmodelled table the control test creates. Prefixed so a stray one in a shared database
#: (there should never be one — every test drops its own database) is identifiable on sight.
_CONTROL_TABLE = "ad170_control_unmodelled"


def _proposed_ops(revision_file: Path) -> list[tuple[str, str | None]]:
    """Every ``op.<name>(...)`` alembic put in the generated ``upgrade()``, as
    ``(op_name, first positional string argument)``.

    Parsed with ``ast`` rather than matched with a regex: a regex over generated DDL matches the
    word ``drop_table`` inside the ``downgrade()`` body and inside comments, and an empty
    revision's body is the single statement ``pass``, which no substring search can distinguish
    from a body that drops a table named in a docstring. The first positional argument is the
    table/index name for every autogenerate-emitted op, and is ``None`` for the shapes that pass
    it by keyword — reported as ``None`` rather than guessed at.
    """
    tree = ast.parse(revision_file.read_text())
    upgrade = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "upgrade"
    )
    ops: list[tuple[str, str | None]] = []
    for node in ast.walk(upgrade):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            continue
        if func.value.id != "op":
            continue
        # `op.f(...)` is alembic's name-quoting helper, never a DDL operation: it appears ONLY as
        # an argument to a real op (`op.drop_index(op.f('gin_prov_meta'), ...)`), so skipping it
        # cannot hide anything and keeps a failure message from listing every name twice.
        if func.attr == "f":
            continue
        first: str | None = None
        if node.args:
            arg = node.args[0]
            # `op.drop_index(op.f('name'), ...)` wraps the name in `op.f(...)`.
            if isinstance(arg, ast.Call) and arg.args:
                arg = arg.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                first = arg.value
        ops.append((func.attr, first))
    return ops


class _Scratch:
    """One throwaway database, plus the alembic config pointed at it."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self.cfg = Config(str(_ALEMBIC_INI))
        self.cfg.set_main_option("sqlalchemy.url", str(engine.url.render_as_string(False)))

    async def upgrade(self, revision: str = "head") -> None:
        # `env.py` calls `asyncio.run()` itself, so the alembic command CANNOT run on this test's
        # event loop — it goes to a worker thread, the same way every other migration test here
        # drives it.
        await asyncio.to_thread(command.upgrade, self.cfg, revision)

    async def autogenerate(self, into: Path) -> list[tuple[str, str | None]]:
        """Run the REAL ``alembic revision --autogenerate`` and return the ops it proposed.

        The generated file goes into ``into`` and is DELETED before returning, so a passing run
        leaves nothing behind and a failing one cannot leave a half-written revision in the real
        ``versions/`` directory where the next ``upgrade`` would pick it up. ``version_locations``
        must still name the real directory, or alembic cannot resolve the current head to chain
        from — it names both, which is exactly what ``--version-path`` does on the command line.
        """
        cfg = Config(str(_ALEMBIC_INI))
        cfg.set_main_option("sqlalchemy.url", str(self.engine.url.render_as_string(False)))
        cfg.set_main_option("version_locations", f"{_REAL_VERSIONS} {into}")
        try:
            await asyncio.to_thread(
                command.revision,
                cfg,
                message="ad170 drift probe",
                autogenerate=True,
                version_path=str(into),
            )
            generated = sorted(into.glob("*.py"))
            assert generated, "alembic --autogenerate produced no revision file at all"
            return [op for path in generated for op in _proposed_ops(path)]
        finally:
            for path in into.glob("*.py"):
                path.unlink()
            shutil.rmtree(into / "__pycache__", ignore_errors=True)

    async def run(self, statement: str) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(text(statement))

    async def table_names(self) -> set[str]:
        """Base tables in ``public``, straight from the catalog — never the ORM's idea of what it
        wrote."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                    )
                )
            ).all()
        return {str(r.table_name) for r in rows}


@pytest_asyncio.fixture
async def scratch() -> AsyncIterator[_Scratch]:
    """A database of its own per test, dropped afterwards.

    The DSN comes from the central Settings tree (never a literal); only the database NAME is
    replaced, so the credentials and host stay the ones ``.env.test`` wires.
    """
    control = make_url(get_settings().storage.postgres.dsn)
    name = f"mu_drift_{uuid4().hex[:12]}"
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


# ============================================================ the rule, stated three ways
async def test_every_migrated_table_has_a_declarative_model(scratch: _Scratch) -> None:
    """A table the migrations create but ``Base.metadata`` does not describe is a table
    ``--autogenerate`` will propose DROPPING. THE gate, and the durable half of the fix."""
    await scratch.upgrade()

    migrated = await scratch.table_names() - {_ALEMBIC_VERSION}
    modelled = set(Base.metadata.tables)

    assert migrated - modelled == set(), (
        f"migrated tables with NO declarative model in schema.py: "
        f"{sorted(migrated - modelled)}. `migrations/env.py` compares the database against "
        "`Base.metadata`, so the next `alembic revision --autogenerate` will emit "
        "`op.drop_table(...)` for each of them — this is how a generated revision came to "
        "propose deleting the metering ledger an invoice is computed from). Add the declarative "
        "class, matching the revision's columns/widths/nullability/constraints EXACTLY."
    )
    assert modelled - migrated == set(), (
        f"declarative models with NO table in a database migrated to head: "
        f"{sorted(modelled - migrated)}. A fresh deployment would come up missing them, and "
        "`--autogenerate` would propose `op.create_table(...)`. Write the migration."
    )


async def test_autogenerate_against_a_migrated_database_proposes_nothing(
    scratch: _Scratch, tmp_path: Path
) -> None:
    """The real ``alembic revision --autogenerate`` against a real migrated database must generate
    an EMPTY revision.

    Strictly stronger than the parity check above, and the difference matters: a model that
    disagrees with its migration by one width or one nullability passes parity and fails here,
    and it is the more dangerous defect — the generated op is a quiet ``op.alter_column`` on a
    billing column rather than a drop anyone would question.
    """
    await scratch.upgrade()

    ops = await scratch.autogenerate(tmp_path)

    assert ops == [], (
        f"`alembic revision --autogenerate` proposed {ops} against a database freshly migrated to "
        "head. The generated revision must be EMPTY: every op here is a difference between "
        "schema.py and the migrations, and applying the file alembic just wrote would enact it. "
        "Fix the model (or the migration) — do NOT silence this by adding the object to "
        "`env.py`'s `DDL_ONLY_INDEXES`, which exists only for the Postgres-only GIN indexes that "
        "cannot be expressed in portable metadata at all."
    )


async def test_the_gate_still_detects_an_unmodelled_table(
    scratch: _Scratch, tmp_path: Path
) -> None:
    """The control. Both gates above must go RED for a table with no declarative model — otherwise
    their green says only that the comparison did not run.

    This is also the guard against "fixing" a future instance with an ``include_object`` filter:
    ``env.py``'s filter is index-only by construction, and if someone widens it to tables, the
    ``drop_table`` assertion here is what notices.
    """
    await scratch.upgrade()
    # Content-free and deliberately trivial: this table exists to be UNMODELLED, nothing else.
    await scratch.run(f"CREATE TABLE {_CONTROL_TABLE} (id VARCHAR(8) NOT NULL PRIMARY KEY)")

    migrated = await scratch.table_names() - {_ALEMBIC_VERSION}
    assert _CONTROL_TABLE in migrated - set(Base.metadata.tables), (
        "the parity gate cannot see a table that is in the database and not in Base.metadata — "
        "its green in the test above proves nothing."
    )

    ops = await scratch.autogenerate(tmp_path)
    assert ("drop_table", _CONTROL_TABLE) in ops, (
        f"`--autogenerate` did NOT propose dropping the unmodelled `{_CONTROL_TABLE}`; it "
        f"proposed {ops}. The autogenerate gate above is therefore vacuous — either the "
        "comparison is not running, or something (an `include_object` filter widened beyond the "
        "two GIN indexes it is allowed to cover) is suppressing table-level differences, which "
        "is precisely what would hide the next unmodelled table."
    )
