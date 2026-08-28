"""Revision ``a71f3c9de205`` (AD-106) — REAL Postgres, THROWAWAY databases, ZERO mocks.

Every test here creates its own database, applies the real chain to it, and drops it. That is
not tidiness: a test that asserts against a database already at head proves nothing about the
revision (the vacuous pass this project has recorded five times), and ``alembic downgrade``
against the SHARED control plane at 127.0.0.1:15432 leaves it a revision short for every other
lane. ``test_room_tables_migration_int.py`` established this pattern for exactly those reasons.

What is proven, and what each proof is FOR
------------------------------------------------------------------------------------------------
1. **The whole chain applies with this revision on top**, and afterwards ``provenance_ledger``
   carries the union shape — every governance column NULLABLE, ``event_id`` NOT NULL and UNIQUE,
   ``memory_id`` NULLABLE, the PRIMARY KEY widened to include ``org_id``.
2. **A pre-existing lineage row SURVIVES up → down → up, and its ``event_id`` is backfilled
   deterministically.** This is the requirement that separates a migration from a rewrite: the
   row is written under the OLD shape, before the revision runs.
3. **``event_id`` is UNIQUE at the DATABASE.** An append-only ledger whose duplicate suppression
   lives only in application code is one retry away from a double-counted history.
4. **A governance event — the row that could not be written here at all, which is why
   ``transfer_provenance`` exists — round-trips**, including a NULL ``memory_id`` and a
   DB-assigned ``position``.
5. **Two orgs may now hold the same ``(stream_id, version)``**, which the pre-revision PRIMARY
   KEY made impossible. That collision is the tenancy defect this table would have inherited the
   moment a second plane wrote to it.
6. **``downgrade`` REFUSES rather than destroys** when the data cannot be represented in the
   narrow shape: a NULL-``memory_id`` governance row, and a cross-org ``(stream_id, version)``
   collision, each stop the downgrade with a message naming the cause.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
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
_REVISION = "a71f3c9de205"  # THE REVISION UNDER TEST — never assumed to be today's chain head
_PRIOR = "c4a1e07b9d33"
_TABLE = "provenance_ledger"

#: Every column this revision adds, and whether it must be nullable. ``event_id`` is the ONE
#: NOT NULL addition, and it is NOT NULL precisely because it is the idempotency key.
_ADDED_COLUMNS = {
    "event_id": False,
    "position": True,
    "object_type": True,
    "object_id": True,
    "object_content_hash": True,
    "origin_namespace_id": True,
    "grantor_principal_id": True,
    "grantee_kind": True,
    "grantee_id": True,
    "grant_id": True,
    "packet_id": True,
    "source_refs": True,
    "content_hash": True,
    "cascade_root_grant_id": True,
}

#: A lineage row in the PRE-revision shape — no ``event_id``, no governance columns. Written
#: BEFORE the revision runs, which is what makes the survival assertions real.
_LEGACY_LINEAGE_ROW = text(
    """
    INSERT INTO provenance_ledger
        (stream_id, version, action, memory_id, org_id, workspace_id, actor_id, at, meta)
    VALUES (:stream_id, :version, 'origin', 'mem_legacy', :org_id, 'ws_1', 'prn_alice', now(),
            CAST('{}' AS JSONB))
    """
)

#: The same lineage row in the POST-revision shape — ``event_id`` supplied by the caller, as a
#: writer on the new schema would.
_LEGACY_LINEAGE_ROW_WITH_EVENT = text(
    """
    INSERT INTO provenance_ledger
        (stream_id, version, action, memory_id, org_id, workspace_id, actor_id, at, meta,
         event_id)
    VALUES (:stream_id, :version, 'origin', 'mem_legacy', :org_id, 'ws_1', 'prn_alice', now(),
            CAST('{}' AS JSONB), :event_id)
    """
)

#: A governance event — the shape ``transfer_provenance`` was stood up to hold. NULL
#: ``memory_id``; the subject is ``(object_type, object_id)``.
_GOVERNANCE_ROW = text(
    """
    INSERT INTO provenance_ledger
        (stream_id, version, action, memory_id, org_id, workspace_id, actor_id, at, meta,
         event_id, object_type, object_id, object_content_hash, origin_namespace_id,
         grantor_principal_id, grantee_kind, grantee_id, grant_id, packet_id, source_refs,
         content_hash, cascade_root_grant_id)
    VALUES (:stream_id, :version, 'shared', NULL, :org_id, 'ws_1', 'prn_alice', now(),
            CAST('{}' AS JSONB), :event_id, 'context_index', 'idx_1', 'h_obj', 'o/ws/u/s',
            'prn_alice', 'user', 'prn_bob', 'grant_1', 'pkt_1', CAST('[]' AS JSONB), 'h_row',
            'grant_root')
    """
)

_SEP = "\x1f"


def _expected_event_id(org_id: str, stream_id: str, version: int) -> str:
    """The revision's own derivation, restated independently here.

    Deliberately NOT imported from the migration module: a backfill that agrees with itself
    proves nothing. This is the value an operator can recompute from the row's own primary key.
    """
    return hashlib.sha256(f"{org_id}{_SEP}{stream_id}{_SEP}{version}".encode()).hexdigest()


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

    async def execute(self, statement: object, params: dict[str, object] | None = None) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(statement, params or {})  # type: ignore[arg-type]

    async def columns(self) -> dict[str, tuple[bool, str | None]]:
        """``{column: (is_nullable, column_default)}`` straight from ``information_schema`` —
        never the ORM's idea of what it wrote; ask the database."""
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT column_name, is_nullable, column_default "
                        "FROM information_schema.columns WHERE table_name = :t"
                    ),
                    {"t": _TABLE},
                )
            ).all()
        return {str(r.column_name): (r.is_nullable == "YES", r.column_default) for r in rows}

    async def primary_key_columns(self) -> list[str]:
        async with self.engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(
                        "SELECT a.attname FROM pg_index i "
                        " JOIN pg_attribute a ON a.attrelid = i.indrelid "
                        "  AND a.attnum = ANY(i.indkey) "
                        " WHERE i.indrelid = to_regclass(CAST(:t AS TEXT)) AND i.indisprimary "
                        " ORDER BY a.attnum"
                    ),
                    {"t": _TABLE},
                )
            ).all()
        return [str(r.attname) for r in rows]

    async def rows(self) -> list[dict[str, object]]:
        async with self.engine.connect() as conn:
            result = await conn.execute(
                text(f"SELECT * FROM {_TABLE} ORDER BY org_id, stream_id, version")  # noqa: S608
            )
            return [dict(m) for m in result.mappings().all()]


@pytest_asyncio.fixture
async def scratch() -> AsyncIterator[_Scratch]:
    """A database of its own per test, dropped afterwards. The DSN comes from the central
    Settings tree (never a literal); only the database NAME is replaced."""
    control = make_url(get_settings().storage.postgres.dsn)
    name = f"mu_provmig_{uuid4().hex[:12]}"
    admin = create_async_engine(control, isolation_level="AUTOCOMMIT", poolclass=None)
    # NullPool + no prepared-statement cache: this file runs DDL (upgrade/downgrade) BETWEEN
    # queries on the same database, and asyncpg raises `InvalidCachedStatementError` when a
    # cached plan outlives the schema it was planned against. Observed, not anticipated — the
    # first run of the up/down/up test failed on exactly that.
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


@pytest_asyncio.fixture
async def at_prior(scratch: _Scratch) -> _Scratch:
    """The chain applied up to — and STOPPING at — the revision before this one, so a row can be
    written in the OLD shape before the revision under test ever runs."""
    await scratch.upgrade(_PRIOR)
    return scratch


# ============================================================ the revision itself
async def test_the_revision_chains_from_the_real_head_and_the_whole_chain_applies(
    scratch: _Scratch,
) -> None:
    """The parent is VERIFIED from the script directory, and the entire chain then applies to an
    empty database with this revision on top — what an operator's first ``upgrade head`` does."""
    script = ScriptDirectory.from_config(scratch.cfg)
    assert script.get_revision(_REVISION).down_revision == _PRIOR
    heads = list(script.get_heads())
    assert len(heads) == 1, f"the migration chain has forked: {heads}"

    await scratch.upgrade("head")

    columns = await scratch.columns()
    for name, must_be_nullable in _ADDED_COLUMNS.items():
        assert name in columns, f"`{name}` is missing — the revision did not add it"
        assert columns[name][0] is must_be_nullable, (
            f"`{name}` nullability is wrong: a governance column that is NOT NULL makes the "
            "memory-lineage half of this table unwritable, and a nullable `event_id` makes the "
            "idempotency key optional."
        )
    assert columns["memory_id"][0] is True, (
        "`memory_id` is still NOT NULL — a governance event's subject is (object_type, "
        "object_id) and often is not a memory at all, so it could not be written here."
    )
    # `position` is DB-assigned on Postgres: the shape `transfer_provenance` already had, kept so
    # the governance adapter's INSERT needs no change when it re-points at this table.
    assert "nextval" in (
        columns["position"][1] or ""
    ), "`position` has no sequence default — the org-scoped ledger scan has no order to walk."
    assert await scratch.primary_key_columns() == ["stream_id", "version", "org_id"], (
        "the PRIMARY KEY does not include org_id: two orgs whose streams share an id collide on "
        "one history."
    )


async def test_a_legacy_lineage_row_survives_up_down_up_with_a_derivable_event_id(
    at_prior: _Scratch,
) -> None:
    """The requirement that separates a migration from a rewrite.

    The row is INSERTed in the pre-revision shape (no ``event_id``, no governance columns), then
    the revision runs. ``event_id`` must be backfilled to a value derivable from the row's own
    primary key — recomputed here independently, so a backfill that merely agrees with itself
    cannot pass. Then down, then up again: still one row, still the same lineage, still the same
    derived id.
    """
    await at_prior.execute(
        _LEGACY_LINEAGE_ROW, {"stream_id": "prov_1", "version": 1, "org_id": "org_a"}
    )

    await at_prior.upgrade()
    after_up = await at_prior.rows()
    assert len(after_up) == 1
    assert after_up[0]["memory_id"] == "mem_legacy"
    assert after_up[0]["event_id"] == _expected_event_id("org_a", "prov_1", 1)
    # A lineage append that predates the revision was never part of the governance scan order;
    # inventing a position for it would fabricate an order no reader ever observed.
    assert after_up[0]["position"] is None

    await at_prior.downgrade()
    mid = await at_prior.rows()
    assert len(mid) == 1, "downgrade destroyed the row it was supposed to leave alone"
    assert mid[0]["memory_id"] == "mem_legacy"
    assert "event_id" not in mid[0], "downgrade left `event_id` behind; it is not reversible"

    await at_prior.upgrade()
    after_second_up = await at_prior.rows()
    assert len(after_second_up) == 1
    assert after_second_up[0]["event_id"] == _expected_event_id("org_a", "prov_1", 1), (
        "the backfill is not deterministic — the same row got a different id on the second "
        "application, so an id an operator recorded once no longer resolves."
    )


async def test_event_id_is_unique_at_the_database(scratch: _Scratch) -> None:
    """Application-level duplicate suppression on an append-only ledger is one retry away from a
    double-counted history, so the constraint has to be in the DDL."""
    await scratch.upgrade()
    await scratch.execute(
        _GOVERNANCE_ROW, {"stream_id": "s_1", "version": 1, "org_id": "org_a", "event_id": "ev_1"}
    )
    with pytest.raises(IntegrityError):
        await scratch.execute(
            _GOVERNANCE_ROW,
            {"stream_id": "s_2", "version": 1, "org_id": "org_a", "event_id": "ev_1"},
        )


async def test_a_governance_event_round_trips_including_a_null_memory_id_and_a_position(
    scratch: _Scratch,
) -> None:
    """The row that could not be written to this table at all — which is the entire reason
    ``transfer_provenance`` exists (AD-62/AD-106)."""
    await scratch.upgrade()
    await scratch.execute(
        _GOVERNANCE_ROW, {"stream_id": "s_1", "version": 1, "org_id": "org_a", "event_id": "ev_1"}
    )
    rows = await scratch.rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["memory_id"] is None
    assert row["object_type"] == "context_index"
    assert row["object_id"] == "idx_1"
    assert row["cascade_root_grant_id"] == "grant_root"
    assert row["action"] == "shared"  # one of AD-62's four previously-inexpressible actions
    assert isinstance(row["position"], int), (
        "`position` was not assigned by the database — the ledger scan (read_all(org, "
        "from_position)) has nothing to order by."
    )


async def test_two_orgs_may_hold_the_same_stream_id_and_version(at_prior: _Scratch) -> None:
    """The tenancy defect the pre-revision PRIMARY KEY carried. Before ``org_id`` joined the key
    this INSERT pair was impossible, which means two orgs' histories were competing for one row
    slot the moment a second plane wrote here.

    Seeded at ``_PRIOR`` and inserted one org at a time: under the OLD key the second INSERT
    would have collided, so the revision is what makes this state reachable — and the backfill
    is what has to keep the two rows' ``event_id`` values apart.
    """
    await at_prior.execute(
        _LEGACY_LINEAGE_ROW, {"stream_id": "prov_shared", "version": 1, "org_id": "org_a"}
    )
    await at_prior.upgrade()
    await at_prior.execute(
        _LEGACY_LINEAGE_ROW_WITH_EVENT,
        {
            "stream_id": "prov_shared",
            "version": 1,
            "org_id": "org_b",
            "event_id": _expected_event_id("org_b", "prov_shared", 1),
        },
    )
    rows = await at_prior.rows()
    assert [r["org_id"] for r in rows] == ["org_a", "org_b"]
    assert (
        len({r["event_id"] for r in rows}) == 2
    ), "the two orgs' backfilled event ids collided — the derivation is not org-scoped."


# ============================================================ downgrade refuses, never destroys
async def test_downgrade_refuses_a_governance_row_rather_than_dropping_its_subject(
    scratch: _Scratch,
) -> None:
    """A NULL ``memory_id`` cannot be expressed in the narrow shape. Refusing is the honest
    answer; making the column NOT NULL by inventing a memory id, or by deleting the row, would
    silently lose a governance fact."""
    await scratch.upgrade()
    await scratch.execute(
        _GOVERNANCE_ROW, {"stream_id": "s_1", "version": 1, "org_id": "org_a", "event_id": "ev_1"}
    )
    with pytest.raises(RuntimeError, match="memory_id"):
        await scratch.downgrade()
    assert len(await scratch.rows()) == 1, "the refused downgrade still damaged the data"


async def test_downgrade_refuses_a_cross_org_stream_collision(at_prior: _Scratch) -> None:
    """Those rows only became legal under the wider key this revision added. Picking a survivor
    is not a migration decision, so the downgrade stops and says how many pairs are affected."""
    await at_prior.execute(
        _LEGACY_LINEAGE_ROW, {"stream_id": "prov_shared", "version": 1, "org_id": "org_a"}
    )
    await at_prior.upgrade()
    await at_prior.execute(
        _LEGACY_LINEAGE_ROW_WITH_EVENT,
        {"stream_id": "prov_shared", "version": 1, "org_id": "org_b", "event_id": "ev_b"},
    )
    with pytest.raises(RuntimeError, match="primary key"):
        await at_prior.downgrade()
    assert len(await at_prior.rows()) == 2, "the refused downgrade still damaged the data"
