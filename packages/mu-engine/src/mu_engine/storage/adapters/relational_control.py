"""Relational control-plane adapter — SQLAlchemy 2.x async (``storage-pluggable §3.1``).

Implements ``ControlPlaneRepository`` over the content-free schema of ``schema.py`` (spec
§2). The SAME schema binds to Postgres (asyncpg), SQLite (aiosqlite), and MySQL/MariaDB
(asyncmy — owner stage 2026-07-27, "dev's stack is MySQL", spec §3.1); this adapter uses
dialect-aware ``INSERT ... ON CONFLICT``/``ON DUPLICATE KEY UPDATE`` so ``sync_provenance`` is
idempotent on ``ux_prov_chash`` (spec §2.4) — a retried ``SyncWorkflow`` push never
double-writes on ANY of the three dialects.

Content-free (spec §0): ``RelationalMapper.to_store`` drops content by construction, so no
column ever receives memory text. Fully async; timeouts + cancellation-safe session scope.
No hand-rolled per-dialect SQL beyond the one portable upsert seam (spec §2.1) — MySQL's
``ON DUPLICATE KEY UPDATE`` keys off the table's ``UNIQUE`` constraint (matching
``ux_prov_chash``) rather than the Postgres/SQLite ``index_elements=`` form (MySQL's dialect
has no equivalent kwarg — this is the ONE named per-dialect divergence, degrade **D7**:
Postgres partial-index/GIN efficiency is lost on MySQL, never correctness, spec §7).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Executable, insert, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.mappers.relational_mapper import RelationalMapper
from mu_engine.storage.relational.schema import AuditLogRow, MemoryProvenance

__all__ = ["RelationalControlPlaneAdapter"]

# Constructor DEFAULT only (DEV-STANDARDS rule 3: no hardcoded constant lives in adapter LOGIC).
# The live value is DI-threaded from the central Settings tree (``PostgresSettings``/
# ``MySQLSettings`` ``.store_io_timeout_s``) by the ``STORE_REGISTRY`` factory
# (``mu_engine.storage.factories._build_postgres``/``_build_mysql``); a bare
# ``RelationalControlPlaneAdapter(engine)`` (e.g. in a unit test) still gets a sane, named
# default rather than a silent unconfigured 0/None.
_DEFAULT_STORE_IO_TIMEOUT_S = 15.0


class RelationalControlPlaneAdapter:
    """Content-free relational mirror + control plane over a real async engine.

    Every external SQL call is wrapped by :func:`retry_io` (transient-only retry/backoff +
    per-attempt timeout) so no store call is unbounded (MAJOR-4); the writes are idempotent
    (``ON CONFLICT DO UPDATE`` / append at a monotonic grain) so a retry never double-writes.
    The timeout is a per-instance, DI-threaded ``retry_io`` wrapper (never a class-level
    decorator baking in a module constant) so ``store_io_timeout_s`` is genuinely tunable.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        mapper: RelationalMapper | None = None,
        store_io_timeout_s: float = _DEFAULT_STORE_IO_TIMEOUT_S,
    ) -> None:
        self._engine = engine
        self._mapper = mapper or RelationalMapper()
        self._retry = retry_io(timeout_s=store_io_timeout_s)

    async def sync_provenance(self, item: MemoryItem) -> str:
        return await self._retry(self._sync_provenance_impl)(item)

    async def _sync_provenance_impl(self, item: MemoryItem) -> str:
        """Idempotent upsert of the content-free MemoryItem mirror (spec §2.4)."""
        row = self._mapper.to_store(item)
        values: dict[str, Any] = {"memory_id": item.id, **row.cols}
        dialect = self._engine.dialect.name
        # dedupe grain = the ratified (org, workspace, content_hash) (ADR 0026), matching the
        # ux_prov_chash unique index — a retried SyncWorkflow push never double-writes.
        conflict_key = ("org_id", "workspace_id", "content_hash")
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
        elif dialect == "mysql":
            # MySQL has no `index_elements=` upsert-target form — `ON DUPLICATE KEY UPDATE`
            # fires off ANY unique-key collision, so it targets `ux_prov_chash` implicitly
            # (D7: the one named per-dialect divergence, never a correctness loss — spec §7).
            my = mysql_insert(MemoryProvenance).values(**values)
            stmt = my.on_duplicate_key_update(
                **{c: my.inserted[c] for c in row.cols if c not in conflict_key}
            )
        else:  # pragma: no cover - other dialects reserved (spec §3.1 D7)
            raise NotImplementedError(f"unsupported relational dialect: {dialect}")
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
        return item.id

    async def get_provenance(self, workspace_id: str, memory_id: str) -> dict[str, Any] | None:
        return await self._retry(self._get_provenance_impl)(workspace_id, memory_id)

    async def _get_provenance_impl(
        self, workspace_id: str, memory_id: str
    ) -> dict[str, Any] | None:
        table = MemoryProvenance.__table__
        stmt = select(table).where(
            table.c.workspace_id == workspace_id, table.c.memory_id == memory_id
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            row = result.mappings().first()
        return dict(row) if row is not None else None

    async def list_by_namespace(self, namespace_prefix: str, *, limit: int) -> list[dict[str, Any]]:
        return await self._retry(self._list_by_namespace_impl)(namespace_prefix, limit=limit)

    async def _list_by_namespace_impl(
        self, namespace_prefix: str, *, limit: int
    ) -> list[dict[str, Any]]:
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
        org_id: str,
        workspace_id: str,
        actor_id: str,
        action: str,
        target_id: str | None,
        success: bool,
        payload: dict[str, Any] | None = None,
    ) -> None:
        return await self._retry(self._append_audit_impl)(
            org_id=org_id,
            workspace_id=workspace_id,
            actor_id=actor_id,
            action=action,
            target_id=target_id,
            success=success,
            payload=payload,
        )

    async def _append_audit_impl(
        self,
        *,
        org_id: str,
        workspace_id: str,
        actor_id: str,
        action: str,
        target_id: str | None,
        success: bool,
        payload: dict[str, Any] | None = None,
    ) -> None:
        # content-free audit row (spec §2.9): ids/hashes/counts/enums only. org_id is the
        # ratified tenancy root (ADR 0026), distinct from the workspace grouping.
        stmt = insert(AuditLogRow).values(
            ts=datetime.now(UTC),
            org_id=org_id,
            workspace_id=workspace_id,
            actor_id=actor_id,
            action=action,
            target_id=target_id,
            success=success,
            payload=payload or {},
        )
        async with self._engine.begin() as conn:
            await conn.execute(stmt)
