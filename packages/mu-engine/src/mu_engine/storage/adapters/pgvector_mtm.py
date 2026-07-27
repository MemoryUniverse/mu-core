"""pgvector vector adapter — an alternate MTM tier repository (``storage-pluggable-spec.md §3.3``).

PORT of mem0's PGVector store (``other_repos/mem0/mem0/vector_stores/pgvector.py:147-244`` — the
table DDL + HNSW/DiskANN index + ``<=>`` cosine-distance ``ORDER BY`` search) and
``other_repos/mem0/mem0/configs/vector_stores/pgvector.py:6-21`` (the ``hnsw``/``diskann`` knobs),
UPGRADED from mem0's raw ``psycopg`` connection pool + sync driver to a fully async ``asyncpg``
pool (DEV-STANDARDS rule 1: every I/O path is ``async``/``await``) with the ``pgvector`` codec
(``pgvector.asyncpg.register_vector``) so a plain ``list[float]`` binds directly to the ``vector``
column — no manual ``'[...]'::vector`` literal formatting, unlike mem0's raw-SQL approach.

Filterable AND SHARED-safe (spec §3.3 table — "pgvector | yes (SQL WHERE + HNSW) | yes"):
``namespace``/``state``/``authorized_ids`` are REAL scalar/array columns (never buried inside the
``payload`` jsonb) so every SHARED search evaluates them INSIDE the SQL ``WHERE`` clause — a
GIN-indexed ``&&`` (array-overlap) predicate for ``authorized_ids`` — strictly BEFORE the
``ORDER BY ... LIMIT`` truncation (the M1 completeness hazard, spec §6 invariant 2). ``invalidate``
is the id-stable cross-store supersede: an ``UPDATE`` that flips the ``state`` column and merges
the payload jsonb, never a delete (spec §6 invariant 4) — idempotent on the id-stable ``point_id``
primary key (spec §6 invariant 5).

Table/index identifiers are NEVER built from raw caller input (see
``mu_engine.storage.mappers.pgvector_mapper.pgvector_table_name`` — a SHA-256 hash of the
workspace), so the f-string DDL/DML below is injection-safe by construction; each such statement
carries a ``# noqa: S608`` with that justification inline.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored, SparseQuery
from mu_engine.storage.mappers.pgvector_mapper import PgVectorMapper, pgvector_table_name
from mu_engine.storage.mappers.qdrant_mapper import point_id
from mu_engine.storage.ports import QdrantPoint

__all__ = ["PgVectorMtmAdapter"]

# Constructor DEFAULTS only (DEV-STANDARDS rule 3: no hardcoded constant lives in adapter LOGIC).
# The live values are DI-threaded from the central Settings tree (``PgVectorSettings``) by the
# ``STORE_REGISTRY`` factory (``mu_engine.storage.factories._build_pgvector``); a bare
# ``PgVectorMtmAdapter(dsn=..., dim=...)`` (e.g. in a unit test) still gets sane, named defaults.
_DEFAULT_STORE_IO_TIMEOUT_S = 10.0
_DEFAULT_POOL_MIN_SIZE = 1
_DEFAULT_POOL_MAX_SIZE = 5


class PgVectorMtmAdapter:
    """Implements ``MtmTierRepository`` over a real asyncpg + pgvector connection pool."""

    filterable = True  # SHARED-safe (spec §3.3): namespace/state/authorized_ids are SQL WHERE

    def __init__(
        self,
        *,
        dsn: str,
        dim: int,
        hnsw: bool = True,
        min_size: int = _DEFAULT_POOL_MIN_SIZE,
        max_size: int = _DEFAULT_POOL_MAX_SIZE,
        store_io_timeout_s: float = _DEFAULT_STORE_IO_TIMEOUT_S,
    ) -> None:
        self._dsn = dsn
        self._dim = dim
        self._hnsw = hnsw
        self._min_size = min_size
        self._max_size = max_size
        self._mapper = PgVectorMapper(dim=dim)
        self._pool: asyncpg.Pool | None = None  # type: ignore[no-any-unimported] # asyncpg ships no stubs
        self._ensured: set[str] = set()
        # per-instance, DI-threaded retry wrapper (never a class-level decorator baking in a
        # module constant) so `store_io_timeout_s` is genuinely tunable from Settings.
        self._retry = retry_io(timeout_s=store_io_timeout_s)

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:  # type: ignore[no-any-unimported]
        # bind the pgvector codec (list[float] <-> `vector` column) + a jsonb codec so `payload`
        # round-trips as a plain dict, never a str the caller must re-parse.
        # (asyncpg ships no stubs — same rationale as falkor_ltm.py's `db: FalkorDB` ignore)
        await register_vector(conn)
        await conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog", format="text"
        )

    async def _ensure_pool(self) -> asyncpg.Pool:  # type: ignore[no-any-unimported]
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                init=self._init_connection,
            )
        return self._pool

    async def close(self) -> None:
        """Release the connection pool (DEV-STANDARDS resource management)."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _ensure_table(self, ns: Namespace) -> str:
        table = pgvector_table_name(ns, self._dim)
        if table in self._ensured:
            return table
        pool = await self._ensure_pool()
        index_kind = "hnsw" if self._hnsw else "ivfflat"
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    point_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    state TEXT NOT NULL,
                    authorized_ids TEXT[] NOT NULL DEFAULT '{{}}',
                    embedding vector({self._dim}) NOT NULL,
                    payload JSONB NOT NULL
                )
                """
            )
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{table}_ns_idx" ON {table} (namespace)'
            )
            await conn.execute(f'CREATE INDEX IF NOT EXISTS "{table}_state_idx" ON {table} (state)')
            await conn.execute(
                f'CREATE INDEX IF NOT EXISTS "{table}_auth_gin" ON {table} '
                "USING gin (authorized_ids)"
            )
            # ivfflat requires rows before it can be built usefully; hnsw does not. Best-effort:
            # a missing vector index degrades to a full-scan, never a correctness loss (D7-style).
            try:
                await conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "{table}_vec_idx" ON {table} '
                    f"USING {index_kind} (embedding vector_cosine_ops)"
                )
            except asyncpg.PostgresError:
                pass
        self._ensured.add(table)
        return table

    async def upsert(self, item: MemoryItem) -> None:
        return await self._retry(self._upsert_impl)(item)

    async def _upsert_impl(self, item: MemoryItem) -> None:
        table = await self._ensure_table(item.namespace)
        row = self._mapper.to_store(item)
        payload = row.payload
        namespace = str(payload.get("namespace", item.namespace.to_prefix()))
        state = str(payload.get("state", item.state.value))
        authorized_ids = [str(a) for a in (payload.get("authorized_ids") or [])]
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {table} (point_id, namespace, state, authorized_ids, embedding, payload)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (point_id) DO UPDATE SET
                    namespace = EXCLUDED.namespace,
                    state = EXCLUDED.state,
                    authorized_ids = EXCLUDED.authorized_ids,
                    embedding = EXCLUDED.embedding,
                    payload = EXCLUDED.payload
                """,  # noqa: S608 -- `table` is hash-derived (pgvector_table_name), never raw input
                row.point_id,
                namespace,
                state,
                authorized_ids,
                row.vector,
                payload,
            )

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
    ) -> list[Scored[MemoryItem]]:
        return await self._retry(self._semantic_impl)(
            ns,
            query_vector,
            limit=limit,
            caller_identity_set=caller_identity_set,
            sparse_query=sparse_query,
        )

    async def _semantic_impl(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
    ) -> list[Scored[MemoryItem]]:
        table = pgvector_table_name(ns, self._dim)
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table)
            if not exists:
                return []
            params: list[Any] = [query_vector, ns.to_prefix(), MemoryState.ACTIVE.value]
            where = "namespace = $2 AND state = $3"
            # Model A — SHARED only; PRIVATE is isolated by to_prefix() + the plane split.
            if ns.visibility is Visibility.SHARED and caller_identity_set is not None:
                params.append(list(caller_identity_set))
                where += f" AND authorized_ids && ${len(params)}"
            params.append(limit)
            rows = await conn.fetch(
                f"""
                SELECT point_id, embedding, payload, 1 - (embedding <=> $1) AS score
                FROM {table}
                WHERE {where}
                ORDER BY embedding <=> $1
                LIMIT ${len(params)}
                """,  # noqa: S608 -- `table` is hash-derived (pgvector_table_name), never raw input
                *params,
            )
        out: list[Scored[MemoryItem]] = []
        for rank, r in enumerate(rows):
            payload = dict(r["payload"])
            # the pgvector codec decodes `embedding` into a `pgvector.Vector`, not a bare
            # list/ndarray (pgvector.asyncpg.register_vector) — `.to_list()` is the documented
            # unwrap.
            raw_vector = r["embedding"]
            vector = raw_vector.to_list() if hasattr(raw_vector, "to_list") else list(raw_vector)
            item = self._mapper.from_store(
                QdrantPoint(
                    point_id=str(r["point_id"]),
                    vector=[float(v) for v in vector],
                    sparse=None,
                    payload=payload,
                    collection=table,
                )
            )
            out.append(
                Scored(
                    item=item,
                    score=float(r["score"]),
                    channel=RecallChannel.MTM_DENSE,
                    rank=rank,
                )
            )
        return out

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        return await self._retry(self._invalidate_impl)(
            ns, loser_id, winner_id, at=at, reason=reason
        )

    async def _invalidate_impl(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        table = pgvector_table_name(ns, self._dim)
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            exists = await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table)
            if not exists:
                return
            # id-stable supersede write (spec §3.3): flip state + merge payload, vector intact.
            await conn.execute(
                f"""
                UPDATE {table}
                SET state = $2,
                    payload = payload || $3
                WHERE point_id = $1
                """,  # noqa: S608 -- `table` is hash-derived (pgvector_table_name), never raw input
                point_id(loser_id),
                MemoryState.SUPERSEDED.value,
                {
                    "state": MemoryState.SUPERSEDED.value,
                    "invalid_at": at.isoformat(),
                    "superseded_by": winner_id,
                    "supersede_reason": reason,
                },
            )
