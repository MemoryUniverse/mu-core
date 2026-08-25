"""``QdrantMapper`` — MemoryItem <-> QdrantPoint (MTM).

PORT of the payload build + vector-null-out from
``/home/user/hackathon/memory_universe/shared/stores/mtm_qdrant.py:195-222`` and the
``uuid5`` point-id scheme (``mtm_qdrant.py:32``).

- point id = ``uuid5(NAMESPACE_URL, memory.id)`` — deterministic (id-stability, spec §5
  contract 2); raw ``MemoryItem.id`` preserved in payload for reversibility.
- payload = ``MemoryItem.to_dict()`` minus the embedding, plus the flattened indexed filter
  keys (spec §3.2): ``namespace`` (η scope), ``state`` (B1), ``authorized_ids`` (Model A,
  SHARED only), ``owner_id``, ``content_hash``, ``artifact_ref``, ``current_tier``.
"""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.ports import QdrantPoint

__all__ = ["NAMESPACE_PAYLOAD_KEY", "QdrantMapper", "collection_name", "point_id"]

# The ONE payload/metadata key every MTM backend stores ``Namespace.to_prefix()`` under, defined
# HERE (the mapper that writes it) so the adapters that must scope a by-id write to a namespace
# spell it once instead of four times: Qdrant's keyword-indexed payload field, Chroma's flat
# metadata key, FAISS's docstore payload key. (pgvector promotes it to a real SQL column of the
# same name, declared in its own DDL.)
NAMESPACE_PAYLOAD_KEY = "namespace"


def point_id(memory_id: str) -> str:
    """Deterministic Qdrant point id from the tier-stable memory id (mtm_qdrant.py:32)."""
    return str(uuid5(NAMESPACE_URL, memory_id))


def collection_name(ns: Namespace, dim: int) -> str:
    """Coarse physical partition ``mu_mtm__{workspace}__{visibility}__{dim}`` (spec §3.1)."""
    return f"mu_mtm__{ns.workspace}__{ns.visibility.value}__{dim}"


class QdrantMapper:
    """Implements ``RowMapper[QdrantPoint]`` (spec §5)."""

    def __init__(self, *, dim: int) -> None:
        self.dim = dim

    def to_store(self, item: MemoryItem) -> QdrantPoint:
        payload = item.to_dict()
        payload.pop("embedding", None)  # vector nulled out of payload (mtm_qdrant.py:213-214)
        # flattened indexed filter keys (spec §3.2)
        payload[NAMESPACE_PAYLOAD_KEY] = item.namespace.to_prefix()
        payload["namespace_parts"] = list(item.namespace.parts())
        payload["state"] = item.state.value
        payload["visibility"] = item.namespace.visibility.value
        payload["current_tier"] = item.tier.value
        payload["owner_id"] = item.owner_id
        payload["content_hash"] = item.content_hash
        payload["session_id"] = item.session_id
        if item.artifact_ref is not None:
            payload["artifact_ref"] = item.artifact_ref
        # Model A authz: SHARED points carry authorized_ids; PRIVATE carries none (spec §3.2).
        if item.namespace.visibility is Visibility.SHARED:
            authorized = item.metadata.get("authorized_ids")
            payload["authorized_ids"] = list(authorized) if authorized else []
        vector = item.embedding if item.embedding is not None else [0.0] * self.dim
        return QdrantPoint(
            point_id=point_id(item.id),
            vector=list(vector),
            sparse=None,
            payload=payload,
            collection=collection_name(item.namespace, self.dim),
        )

    def from_store(self, row: QdrantPoint) -> MemoryItem:
        payload: dict[str, Any] = dict(row.payload)
        # strip the flattened index keys that shadow canonical fields (namespace overflow).
        parts = payload.pop("namespace_parts", None)
        if parts is not None:
            payload["namespace"] = parts
        for k in ("current_tier", "authorized_ids"):
            payload.pop(k, None)
        item = MemoryItem.from_dict(payload)
        if row.vector and any(v != 0.0 for v in row.vector):
            item.embedding = list(row.vector)
        return item
