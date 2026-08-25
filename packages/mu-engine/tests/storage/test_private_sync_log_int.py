"""``private_sync_log`` — REAL mu-dev-postgres round-trip proof for the §7.17 item 4a additive
columns (``pinned``, ``resolution_origin``), added by Alembic revision ``1ab7f7175baa`` on top
of the initial schema (``46ae4bcc2472``). ZERO mocks (DEV-STANDARDS non-negotiable).

Why this table, not a repository: no ``PrivateDelta`` <-> ``PrivateSyncLogRow`` repository/mapper
is wired yet (only the SQLAlchemy schema exists) — this test proves the RELATIONAL MIRROR itself
carries the two new fields losslessly, which is what CANONICAL:777 ("every field is on the delta
itself") requires for `mu_engine.services.conflict.order.total_order_key` to ever be computable
against a candidate set reloaded from Postgres (the hub's §7.5 re-evaluation path).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncEngine

from mu_engine.storage.relational.schema import PrivateSyncLogRow

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


async def test_pinned_and_resolution_origin_round_trip(pg_engine: AsyncEngine, uid: str) -> None:
    org_id = f"org{uid}"
    principal_id = f"principal{uid}"
    async with pg_engine.begin() as conn:
        await conn.execute(
            insert(PrivateSyncLogRow).values(
                org_id=org_id,
                workspace_id=f"ws{uid}",
                principal_id=principal_id,
                seq=1,
                origin_device_id="dev_a",
                op="upsert",
                memory_id="mem_1",
                content_hash="hash_a",
                tier="ltm",
                valid_at=_T0,
                valid_at_inferred=False,
                lamport=1,
                occurred_at=_T0,
                provenance_id="prov_1",
                pinned=True,
                resolution_origin="manual",
            )
        )
    async with pg_engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    select(PrivateSyncLogRow).where(
                        PrivateSyncLogRow.org_id == org_id,
                        PrivateSyncLogRow.principal_id == principal_id,
                        PrivateSyncLogRow.seq == 1,
                    )
                )
            )
            .mappings()
            .first()
        )
    assert row is not None
    assert row["pinned"] is True
    assert row["resolution_origin"] == "manual"


async def test_pinned_defaults_false_and_resolution_origin_defaults_null(
    pg_engine: AsyncEngine, uid: str
) -> None:
    """Additive-migration backward compatibility: a row written WITHOUT the two new columns
    (as every pre-existing row was, before revision ``1ab7f7175baa``) gets ``pinned=False`` /
    ``resolution_origin=None`` — never a NOT NULL failure, never a fabricated value."""
    org_id = f"org{uid}"
    principal_id = f"principal{uid}"
    async with pg_engine.begin() as conn:
        await conn.execute(
            insert(PrivateSyncLogRow).values(
                org_id=org_id,
                workspace_id=f"ws{uid}",
                principal_id=principal_id,
                seq=1,
                origin_device_id="dev_a",
                op="upsert",
                memory_id="mem_1",
                content_hash="hash_b",
                tier="ltm",
                valid_at=_T0,
                valid_at_inferred=False,
                lamport=1,
                occurred_at=_T0,
                provenance_id="prov_1",
                # pinned / resolution_origin deliberately omitted
            )
        )
    async with pg_engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    select(PrivateSyncLogRow).where(
                        PrivateSyncLogRow.org_id == org_id,
                        PrivateSyncLogRow.principal_id == principal_id,
                        PrivateSyncLogRow.seq == 1,
                    )
                )
            )
            .mappings()
            .first()
        )
    assert row is not None
    assert row["pinned"] is False
    assert row["resolution_origin"] is None


async def test_the_step_2c_remainder_columns_round_trip(pg_engine: AsyncEngine, uid: str) -> None:
    """``resolved_by`` / ``caused_by_seq`` / ``appended_at`` — the rest of build step 2c's D-28
    column list, added by revision ``9c41d0b7ae52`` on top of ``f6fc5f7052cc``.

    Each is here for a reason the two tests above do not cover:

    * ``caused_by_seq`` is **already a field on the shipped wire delta**
      (``PrivateDelta.caused_by_seq`` — *"replica-apply echo of log seq N; projector DROPS it"*).
      Without the column it round-trips to ``None``, and a replica-apply echo becomes
      indistinguishable on read-back from an original write — the one distinction appender B's
      loop suppression rests on (**CANONICAL:689**: absent it, *"one write oscillates forever
      between the engine and the log"*).
    * ``resolved_by`` is ``CANONICAL-CONTRACTS.md:538``'s attribution half of a manual resolution,
      landing together with the matching ``PrivateDelta.resolved_by`` field so the column has a
      real source rather than nothing to put in it.
    * ``appended_at`` is the hub's receive time and the ONLY possible basis for the retention
      window: ``private_sync_log_retention_s`` is a duration, and a pruner over a table with no
      timestamp cannot be written at all. It has **no reader today — that is O-32** — and this
      test is what keeps it from silently becoming un-writable too.

    **What breaks it:** ``alembic downgrade -1``.
    """
    org_id = f"org{uid}"
    principal_id = f"principal{uid}"
    async with pg_engine.begin() as conn:
        await conn.execute(
            insert(PrivateSyncLogRow).values(
                org_id=org_id,
                workspace_id=f"ws{uid}",
                principal_id=principal_id,
                seq=1,
                origin_device_id="dev_a",
                op="supersede",
                memory_id="mem_1",
                content_hash="hash_c",
                tier="ltm",
                valid_at=_T0,
                valid_at_inferred=False,
                lamport=3,
                occurred_at=_T0,
                provenance_id="prov_1",
                winner_id="mem_win",
                loser_id="mem_lose",
                resolution_origin="manual",
                resolved_by="prn_amir",
                caused_by_seq=41,
                appended_at=_T0,
            )
        )
    async with pg_engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    select(PrivateSyncLogRow).where(
                        PrivateSyncLogRow.org_id == org_id,
                        PrivateSyncLogRow.principal_id == principal_id,
                        PrivateSyncLogRow.seq == 1,
                    )
                )
            )
            .mappings()
            .first()
        )
    assert row is not None
    assert row["resolved_by"] == "prn_amir"
    assert row["caused_by_seq"] == 41
    assert row["appended_at"] == _T0


async def test_caused_by_seq_and_resolved_by_default_to_null(
    pg_engine: AsyncEngine, uid: str
) -> None:
    """The additive columns are nullable-safe: a delta that is not an echo and was not manually
    resolved writes neither, and reads back ``None`` for both rather than a fabricated value.

    ⚠ ``None`` for ``caused_by_seq`` is the LOAD-BEARING default — it is what "this mutation is an
    original write, project it" means to appender B. A column that defaulted to ``0`` would read as
    an echo of ``seq`` 0 and the projector would drop every original write in the system.
    """
    org_id = f"org{uid}"
    principal_id = f"principal{uid}"
    async with pg_engine.begin() as conn:
        await conn.execute(
            insert(PrivateSyncLogRow).values(
                org_id=org_id,
                workspace_id=f"ws{uid}",
                principal_id=principal_id,
                seq=2,
                origin_device_id="dev_a",
                op="upsert",
                memory_id="mem_2",
                content_hash="hash_d",
                tier="ltm",
                valid_at=_T0,
                valid_at_inferred=False,
                lamport=1,
                occurred_at=_T0,
                provenance_id="prov_1",
                appended_at=_T0,
                # resolved_by / caused_by_seq deliberately omitted
            )
        )
    async with pg_engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    select(PrivateSyncLogRow).where(
                        PrivateSyncLogRow.org_id == org_id,
                        PrivateSyncLogRow.principal_id == principal_id,
                        PrivateSyncLogRow.seq == 2,
                    )
                )
            )
            .mappings()
            .first()
        )
    assert row is not None
    assert row["resolved_by"] is None
    assert row["caused_by_seq"] is None
