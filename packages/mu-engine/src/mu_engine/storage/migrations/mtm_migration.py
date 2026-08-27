"""Blue-green migration off a legacy-named Qdrant MTM collection (AD-2/AD-6).

Follows the recipe `docs/superpowers/design/mtm-retrieval-design.md` §1.7 already specifies for
the dense-only -> hybrid migration, applied here to the naming defect instead: **create the
new-named ("green") collection, re-upsert, verify, THEN drop the old ("blue") one — never mutate
a collection in place.** Dense vectors are READ BACK and WRITTEN VERBATIM — never re-embedded
(§1.7: "Dense vectors are reused, never recomputed"). Idempotent (Qdrant `upsert` is upsert-by-id,
so re-running after a partial/interrupted run just re-writes the same points) and therefore safe
to resume by simply re-invoking it.

Per-POINT target resolution, not per-collection (the hard part the owning brief calls out): one
legacy collection can hold points for MULTIPLE distinct `(org, workspace)` pairs — that is exactly
the AD-6 collision this migration exists to undo — so each point is routed independently via
:func:`~mu_engine.storage.migrations.naming.resolve_tenancy_from_mtm_payload`. A point whose
tenancy cannot be recovered is SKIPPED and counted, never guessed; the source collection is
dropped only once EVERY point in it has been migrated (zero unresolved) and the drop is verified
by a post-upsert count on every target.

Dry-run by default (`apply=False`): resolves and reports, writes/deletes nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import structlog
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mu_engine.storage.adapters.qdrant_mtm import _KEYWORD_INDEXES
from mu_engine.storage.migrations.naming import discover_legacy_mtm_collection_names
from mu_engine.storage.migrations.planning import plan_mtm_migration

__all__ = [
    "MtmCollectionMigrationResult",
    "migrate_all_legacy_mtm_collections",
    "migrate_mtm_collection",
]

_log = structlog.get_logger("mu_engine.storage.migrations.mtm_migration")

# Bounds one scroll round-trip / one upsert batch — never materializes an unbounded page (the
# shared-box RAM guard DEV-STANDARDS calls out), independent of a collection's total size.
_PAGE_SIZE = 256


class MtmCollectionMigrationResult(BaseModel, frozen=True):
    """Content-free (CLAUDE.md rule 3): counts and partition NAMES only, no payload content."""

    source_collection: str
    dim: int
    found: int
    migrated: int
    unresolved: int
    target_collections: tuple[str, ...]
    applied: bool
    dropped_source: bool


async def _scroll_all_points(
    client: AsyncQdrantClient, collection: str
) -> AsyncIterator[tuple[str, dict[str, object], list[float]]]:
    """Every ``(point_id, payload, dense_vector)`` in ``collection``, paginated. The vector is
    read back verbatim here and nowhere re-embedded downstream — the §1.7 "no re-embed" rule."""
    offset: models.ExtendedPointId | None = None
    while True:
        records, offset = await client.scroll(
            collection_name=collection,
            limit=_PAGE_SIZE,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        for rec in records:
            payload: dict[str, object] = dict(rec.payload or {})
            raw_vector = rec.vector if isinstance(rec.vector, list) else []
            vector = [float(v) for v in raw_vector if isinstance(v, int | float)]
            yield str(rec.id), payload, vector
        if offset is None:
            return


async def _ensure_target_collection(client: AsyncQdrantClient, name: str, dim: int) -> None:
    """Create the green collection with the SAME payload-index catalog the live adapter creates
    (:data:`mu_engine.storage.adapters.qdrant_mtm._KEYWORD_INDEXES`, imported rather than
    re-typed — DRY). A no-op if the collection already exists (idempotent / resumable)."""
    if await client.collection_exists(name):
        return
    await client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    for field in _KEYWORD_INDEXES:
        await client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


async def migrate_mtm_collection(
    client: AsyncQdrantClient, source_collection: str, *, apply: bool
) -> MtmCollectionMigrationResult:
    """Migrate one legacy collection. ``apply=False`` (the default at the CLI layer) resolves
    every point and reports what WOULD happen; nothing is written or dropped."""
    info = await client.get_collection(source_collection)
    vectors_config = info.config.params.vectors
    if not isinstance(vectors_config, models.VectorParams):
        # A named/sparse vector config is not a shape this (pre-hybrid) legacy naming era ever
        # produced (§1.7: the hybrid collection is a SEPARATE, current-named migration this
        # module does not own) — refuse rather than guess a dimension.
        _log.warning("mtm_migration.unsupported_vectors_config", collection=source_collection)
        return MtmCollectionMigrationResult(
            source_collection=source_collection,
            dim=0,
            found=0,
            migrated=0,
            unresolved=0,
            target_collections=(),
            applied=False,
            dropped_source=False,
        )
    dim = vectors_config.size

    payload_by_id: dict[str, dict[str, object]] = {}
    vector_by_id: dict[str, list[float]] = {}
    async for point_id, payload, vector in _scroll_all_points(client, source_collection):
        payload_by_id[point_id] = payload
        vector_by_id[point_id] = vector

    plan = plan_mtm_migration(
        source_collection=source_collection, dim=dim, points=payload_by_id.items()
    )
    _log.info(
        "mtm_migration.planned",
        source_collection=source_collection,
        found=len(payload_by_id),
        resolved=plan.resolved_count,
        unresolved=plan.unresolved_count,
        targets=len(plan.targets),
    )

    if not apply:
        return MtmCollectionMigrationResult(
            source_collection=source_collection,
            dim=dim,
            found=len(payload_by_id),
            migrated=0,
            unresolved=plan.unresolved_count,
            target_collections=tuple(plan.targets),
            applied=False,
            dropped_source=False,
        )

    for target, point_ids in plan.targets.items():
        await _ensure_target_collection(client, target, dim)
        points = [
            models.PointStruct(id=pid, vector=vector_by_id[pid], payload=payload_by_id[pid])
            for pid in point_ids
        ]
        for start in range(0, len(points), _PAGE_SIZE):
            await client.upsert(collection_name=target, points=points[start : start + _PAGE_SIZE])

    verified = True
    for target, point_ids in plan.targets.items():
        result = await client.count(collection_name=target, exact=True)
        if result.count < len(point_ids):
            verified = False
            _log.warning(
                "mtm_migration.verify_short",
                target_collection=target,
                expected_at_least=len(point_ids),
                found=result.count,
            )

    dropped_source = False
    if verified and plan.unresolved_count == 0:
        await client.delete_collection(source_collection)
        dropped_source = True
        _log.info("mtm_migration.dropped_source", source_collection=source_collection)
    else:
        _log.info(
            "mtm_migration.source_kept",
            source_collection=source_collection,
            verified=verified,
            unresolved=plan.unresolved_count,
        )

    return MtmCollectionMigrationResult(
        source_collection=source_collection,
        dim=dim,
        found=len(payload_by_id),
        migrated=plan.resolved_count,
        unresolved=plan.unresolved_count,
        target_collections=tuple(plan.targets),
        applied=True,
        dropped_source=dropped_source,
    )


async def migrate_all_legacy_mtm_collections(
    client: AsyncQdrantClient, *, apply: bool
) -> list[MtmCollectionMigrationResult]:
    """Discover every legacy-named MTM collection live in ``client`` and migrate each one."""
    listing = await client.get_collections()
    legacy = discover_legacy_mtm_collection_names(c.name for c in listing.collections)
    results = []
    for name in legacy:
        results.append(await migrate_mtm_collection(client, name, apply=apply))
    return results
