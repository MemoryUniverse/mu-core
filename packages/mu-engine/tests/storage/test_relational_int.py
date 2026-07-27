"""Relational control-plane adapter — REAL mu-dev-postgres, ZERO mocks.

Covers: idempotent content-addressed sync (``ux_prov_chash``, spec §8.8), the content-free
guarantee at the REAL column level (spec §8.3), namespace-scoped listing, and audit append."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from mu_engine.storage.adapters.relational_control import RelationalControlPlaneAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace

pytestmark = pytest.mark.integration


async def test_sync_provenance_idempotent(
    pg_engine: AsyncEngine,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = RelationalControlPlaneAdapter(pg_engine)
    ns = make_ns()
    item = make_item(ns, "sync me")
    await adapter.sync_provenance(item)
    await adapter.sync_provenance(item)  # retried push -> ON CONFLICT DO UPDATE, no double row
    async with pg_engine.connect() as conn:
        n = await conn.scalar(
            text(
                "select count(*) from memory_provenance "
                "where org_id=:o and workspace_id=:w and content_hash=:h"
            ),
            {"o": ns.org, "w": item.workspace_id, "h": item.content_hash},
        )
    assert n == 1
    row = await adapter.get_provenance(item.workspace_id, item.id)
    assert row is not None and row["provenance_id"] == item.provenance_id


async def test_columns_are_content_free(
    pg_engine: AsyncEngine,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = RelationalControlPlaneAdapter(pg_engine)
    ns = make_ns()
    item = make_item(ns, "Zerg rushes at dawn", subject="Zerg", predicate="rushes", obj="dawn")
    await adapter.sync_provenance(item)
    async with pg_engine.connect() as conn:
        row = (
            (
                await conn.execute(
                    text("select * from memory_provenance where memory_id=:m"),
                    {"m": item.id},
                )
            )
            .mappings()
            .first()
        )
    assert row is not None
    blob = " ".join(str(v) for k, v in row.items() if k != "content_hash").lower()
    # NO memory text ever lands in a Postgres column (CANONICAL §3 / spec §8.3).
    for leak in ("zerg rushes at dawn", "rushes", "dawn"):
        assert leak not in blob


async def test_namespace_scoped_listing(
    pg_engine: AsyncEngine,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = RelationalControlPlaneAdapter(pg_engine)
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")
    await adapter.sync_provenance(make_item(ns_a, "in A"))
    await adapter.sync_provenance(make_item(ns_b, "in B"))
    rows_a = await adapter.list_by_namespace(ns_a.to_prefix(), limit=50)
    prefixes = {r["namespace_prefix"] for r in rows_a}
    assert prefixes == {ns_a.to_prefix()}  # never returns B's rows


async def test_audit_append(pg_engine: AsyncEngine, make_ns: Callable[..., Namespace]) -> None:
    adapter = RelationalControlPlaneAdapter(pg_engine)
    ns = make_ns()
    await adapter.append_audit(
        org_id=ns.org,
        workspace_id=ns.workspace,
        actor_id="u1",
        action="memory.write",
        target_id="mem_x",
        success=True,
        payload={"count": 1},  # content-free
    )
    async with pg_engine.connect() as conn:
        n = await conn.scalar(
            text("select count(*) from audit_log where org_id=:o and workspace_id=:w"),
            {"o": ns.org, "w": ns.workspace},
        )
    assert n == 1
