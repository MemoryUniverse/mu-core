"""Qdrant vector adapter — the MTM tier repository (``storage-pluggable §3.3``).

PORT of the point shape + payload-index catalog + the pre-truncation recall filter from
``/home/user/hackathon/memory_universe/shared/stores/mtm_qdrant.py:90,127,195-222``,
re-homed via :class:`QdrantMapper`.

The load-bearing rule (spec §3.2, M1 authz hazard): every SHARED search evaluates
``namespace`` + ``state='active'`` + ``authorized_ids`` (Model A) INSIDE the ANN traversal,
BEFORE top-k truncation. ``invalidate`` is the id-stable cross-store supersede (spec §3.3):
overwrite the loser's payload with ``state='superseded'`` so the ``state='active'`` filter
drops it. Fully async (``AsyncQdrantClient``); dimension comes from the live embedder.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored, SparseQuery
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper, collection_name, point_id
from mu_engine.storage.ports import QdrantPoint

__all__ = ["QdrantMtmAdapter"]

# payload fields promoted to server-side indexes (spec §3.2).
_KEYWORD_INDEXES = (
    "namespace",
    "session_id",
    "state",
    "visibility",
    "authorized_ids",
    "current_tier",
    "owner_id",
    "content_hash",
    "artifact_ref",
)


class QdrantMtmAdapter:
    """Implements ``MtmTierRepository`` over a real Qdrant connection."""

    def __init__(self, client: AsyncQdrantClient, *, dim: int) -> None:
        self._qdrant = client
        self._dim = dim
        self._mapper = QdrantMapper(dim=dim)
        self._ensured: set[str] = set()

    async def _ensure_collection(self, ns: Namespace) -> str:
        name = collection_name(ns, self._dim)
        if name in self._ensured:
            return name
        if not await self._qdrant.collection_exists(name):
            await self._qdrant.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
            )
            for field in _KEYWORD_INDEXES:
                await self._qdrant.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        self._ensured.add(name)
        return name

    async def upsert(self, item: MemoryItem) -> None:
        name = await self._ensure_collection(item.namespace)
        row = self._mapper.to_store(item)
        await self._qdrant.upsert(
            collection_name=name,
            points=[models.PointStruct(id=row.point_id, vector=row.vector, payload=row.payload)],
        )

    def _recall_filter(
        self, ns: Namespace, caller_identity_set: frozenset[str] | None
    ) -> models.Filter:
        # compiled recall filter (spec §3.2), applied server-side pre-truncation.
        must: list[models.Condition] = [
            models.FieldCondition(key="namespace", match=models.MatchValue(value=ns.to_prefix())),
            models.FieldCondition(
                key="state", match=models.MatchValue(value=MemoryState.ACTIVE.value)
            ),
        ]
        # Model A — SHARED only; PRIVATE is isolated by to_prefix() + the plane split.
        if ns.visibility is Visibility.SHARED and caller_identity_set is not None:
            must.append(
                models.FieldCondition(
                    key="authorized_ids",
                    match=models.MatchAny(any=list(caller_identity_set)),
                )
            )
        return models.Filter(must=must)

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
    ) -> list[Scored[MemoryItem]]:
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return []
        hits = await self._qdrant.query_points(
            collection_name=name,
            query=query_vector,
            query_filter=self._recall_filter(ns, caller_identity_set),
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )
        out: list[Scored[MemoryItem]] = []
        for rank, hit in enumerate(hits.points):
            raw = hit.vector if isinstance(hit.vector, list) else []
            vector = [float(v) for v in raw if isinstance(v, int | float)]
            payload: dict[str, Any] = hit.payload or {}
            item = self._mapper.from_store(
                QdrantPoint(
                    point_id=str(hit.id),
                    vector=vector,
                    sparse=None,
                    payload=payload,
                    collection=name,
                )
            )
            out.append(
                Scored(
                    item=item,
                    score=float(hit.score),
                    channel=RecallChannel.MTM_DENSE,
                    rank=rank,
                )
            )
        return out

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        # id-stable supersede write (spec §3.3): stamp state='superseded' + invalid_at on the
        # loser point, vector + rest of payload intact. Idempotent (id-stable point_id).
        name = collection_name(ns, self._dim)
        if not await self._qdrant.collection_exists(name):
            return
        await self._qdrant.set_payload(
            collection_name=name,
            payload={
                "state": MemoryState.SUPERSEDED.value,
                "invalid_at": at.isoformat(),
                "superseded_by": winner_id,
                "supersede_reason": reason,
            },
            points=[point_id(loser_id)],
        )
