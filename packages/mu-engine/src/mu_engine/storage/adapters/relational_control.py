"""Relational control-plane adapter — SQLAlchemy 2.x async (``storage-pluggable §3.1``).

Implements ``ControlPlaneRepository`` over the content-free schema of ``schema.py`` (spec
§2). The SAME schema binds to Postgres (asyncpg) and SQLite (aiosqlite); this adapter uses
dialect-aware ``INSERT ... ON CONFLICT DO UPDATE`` so ``sync_provenance`` is idempotent on
``ux_prov_chash`` (spec §2.4) — a retried ``SyncWorkflow`` push never double-writes.

Content-free (spec §0): ``RelationalMapper.to_store`` drops content by construction, so no
column ever receives memory text. Fully async; timeouts + cancellation-safe session scope.
No hand-rolled per-dialect SQL beyond the one portable upsert seam (spec §2.1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Executable, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.mappers.relational_mapper import RelationalMapper
from mu_engine.storage.relational.schema import AuditLogRow, MemoryProvenance

__all__ = ["RelationalControlPlaneAdapter"]


class RelationalControlPlaneAdapter:
    """Content-free relational mirror + control plane over a real async engine."""

    def __init__(self, engine: AsyncEngine, *, mapper: RelationalMapper | None = None) -> None:
        self._engine = engine
        self._mapper = mapper or RelationalMapper()

    async def sync_provenance(self, item: MemoryItem) -> str:
        """Idempotent upsert of the content-free MemoryItem mirror (spec §2.4)."""
        row = self._mapper.to_store(item)
        values: dict[str, Any] = {"memory_id": item.id, **row.cols}
        dialect = self._engine.dialect.name
        conflict_key = ("workspace_id", "content_hash")
        stmt: Executable
        if dialect == "postgresql":
            pg = pg_insert(MemoryProvenance).values(**values)
            stmt = pg.on_conflict_do_update(
                index_elements=list(conflict_key),
                set_={c: pg.excluded[c] for c in row.cols if c not in conflict_key},
            )
        elif dialect == "sqlite":
            sq = sqlite_insert(MemoryProvenance).values(**values)
            stmt = sq.on_conflict_do_update(
                index_elements=list(conflict_key),
                set_={c: sq.excluded[c] for c in row.cols if c not in conflict_key},
            )
        else:  # pragma: no cover - other dialects reserved (spec §3.1 D7)
            raise NotImplementedError(f"unsupported relational dialect: {dialect}")
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
        return item.id

    async def get_provenance(self, workspace_id: str, memory_id: str) -> dict[str, Any] | None:
        table = MemoryProvenance.__table__
        stmt = select(table).where(
            table.c.workspace_id == workspace_id, table.c.memory_id == memory_id
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            row = result.mappings().first()
        return dict(row) if row is not None else None

    async def list_by_namespace(self, namespace_prefix: str, *, limit: int) -> list[dict[str, Any]]:
        # scoped by to_prefix() — tenancy isolation (spec §2 tenancy rule).
        table = MemoryProvenance.__table__
        stmt = select(table).where(table.c.namespace_prefix == namespace_prefix).limit(limit)
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            rows = result.mappings().all()
        return [dict(r) for r in rows]

    async def append_audit(
        self,
        *,
        workspace_id: str,
        actor_id: str,
        action: str,
        target_id: str | None,
        success: bool,
        payload: dict[str, Any] | None = None,
    ) -> None:
        # content-free audit row (spec §2.9): ids/hashes/counts/enums only.
        stmt = insert(AuditLogRow).values(
            ts=datetime.now(UTC),
            workspace_id=workspace_id,
            actor_id=actor_id,
            action=action,
            target_id=target_id,
            success=success,
            payload=payload or {},
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
