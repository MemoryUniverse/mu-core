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
