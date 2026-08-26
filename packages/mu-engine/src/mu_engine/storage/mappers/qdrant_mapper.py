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
from mu_engine.storage.mappers.tenancy import tenant_partition_digest
from mu_engine.storage.ports import QdrantPoint

__all__ = [
    "NAMESPACE_PAYLOAD_KEY",
    "QdrantMapper",
    "collection_name",
    "point_id",
    "tenant_partition_digest",
]

# The ONE payload/metadata key every MTM backend stores ``Namespace.to_prefix()`` under, defined
# HERE (the mapper that writes it) so the adapters that must scope a by-id write to a namespace
# spell it once instead of four times: Qdrant's keyword-indexed payload field, Chroma's flat
# metadata key, FAISS's docstore payload key. (pgvector promotes it to a real SQL column of the
# same name, declared in its own DDL.)
NAMESPACE_PAYLOAD_KEY = "namespace"


def point_id(memory_id: str) -> str:
    """Deterministic Qdrant point id from the tier-stable memory id (mtm_qdrant.py:32)."""
    return str(uuid5(NAMESPACE_URL, memory_id))


# `tenant_partition_digest` used to be DEFINED here and imported by the three other vector
# mappers (chroma/faiss/pgvector) — a reviewer flagged that as a layering smell (nothing about the
# digest is Qdrant-specific). D-8 (the LTM graph tier's copy of the same raw-join collision) made
# reaching into this module from a fifth caller (`falkor_ltm.py`, a GRAPH adapter) worse, not
# better, so the helper moved to the neutral `mu_engine.storage.mappers.tenancy` module — see that
# module for the full derivation/collision-resistance docstring. Re-imported (not re-derived) and
# kept in `__all__` here so `from mu_engine.storage.mappers.qdrant_mapper import
# tenant_partition_digest` still resolves for any existing caller of this module.


def collection_name(ns: Namespace, dim: int) -> str:
    """Coarse physical partition for ``(org, workspace, visibility, dim)`` — see
    :func:`tenant_partition_digest` for what the digest derives and why it is
    collision-resistant rather than collision-resistant. ``visibility`` and ``dim`` stay in the
    clear since they carry no caller-controlled text and cannot participate in the join
    ambiguity :func:`tenant_partition_digest` closes.
    """
    return f"mu_mtm__{tenant_partition_digest(ns)}__{ns.visibility.value}__{dim}"


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
