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
from mu_engine.storage.errors import MissingEmbeddingError
from mu_engine.storage.mappers.tenancy import tenant_partition_digest
from mu_engine.storage.ports import QdrantPoint

__all__ = [
    "NAMESPACE_PAYLOAD_KEY",
    "QdrantMapper",
    "collection_name",
    "payload_str_list",
    "point_id",
    "tenant_partition_digest",
]


def payload_str_list(value: object) -> list[str]:
    """Narrow ONE ``dict[str, object]`` payload slot to ``list[str]``, or refuse it.

    AD-184 re-homed the StoreModel DTOs to ``mu_contracts.ports.stores``, and the contracts shape
    — the stricter of the two, which is why it won — types ``QdrantPoint.payload`` as
    ``dict[str, object]`` where ``mu_engine``'s deleted copy said ``dict[str, Any]``. ``Any``
    silences every read of that dict; ``object`` does not, which is the point of the stricter
    shape and also why three sites went red under ``mypy --strict`` the moment the duplicate was
    deleted (this helper is the fix; it was NOT part of the AD-184 change as first written, and
    that CI gate was red on the branch).

    A non-sequence RAISES rather than coercing to ``[]`` — deliberately, because the one caller
    that matters is the ``authorized_ids`` stamp on a SHARED write (``pgvector_mtm``,
    ``weaviate_mtm``). Reading a malformed stamp as "the empty list" would write a row nobody can
    read, silently; reading it as "no filter" would be worse. The old ``[str(a) for a in ...]``
    raised ``TypeError`` on the same input, so this preserves the behaviour exactly while making
    the narrowing legible. Content-free: the message names the slot, never its value.
    """
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(v) for v in value]
    raise TypeError(
        f"payload slot expected a sequence, got {type(value).__name__} — "
        "a malformed indexed key is never silently read as empty"
    )


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
        # D1 (STATE-AND-DEFECTS-0829.md): a silent `[0.0] * dim` substitution used to sit here —
        # cosine against an all-zero vector is 0 forever, and it hid a whole write path
        # (LifecyclePromotionService._promote_to_mtm) never embedding at all. Fail loud instead
        # (DEV-STANDARDS rule 8): a caller that reaches this write path with no embedding has a
        # bug to fix at the call site, never a vector this mapper should fabricate.
        if item.embedding is None:
            raise MissingEmbeddingError(
                f"QdrantMapper.to_store: memory {item.id!r} (ns={item.namespace.to_prefix()!r}) "
                "has embedding=None — refusing to silently substitute a zero vector into the "
                "vector tier (D1). Embed before upserting."
            )
        vector = item.embedding
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
        # AD-258 fix-impl (verified by running, `test_qdrant_mtm_write_scoping_int.py`'s own
        # module docstring named this as known-dropped): `set_entity_uids` (D-5) PATCHes a
        # top-level ``entity_uids`` payload key onto an already-written point — it is not a
        # ``MemoryItem`` field, so ``MemoryItem.model_validate`` (default Pydantic ``extra=
        # "ignore"``) silently discarded it on every read, making the LTM entity resolution
        # write-only: never reachable through ``semantic()``/``get()`` despite the write
        # succeeding every time. Mirrors ``to_store``'s own ``authorized_ids`` precedent (a
        # payload-only indexed key gets moved to/from ``MemoryItem.metadata`` at the mapper
        # seam) — restored into ``metadata['entity_uids']`` here so the ranker's content-aware
        # LTM seed (AD-258, `ranker.py::_ltm_channel`) can actually read the entity uids the
        # graph tier already resolved for this fact instead of always seeing an empty list.
        entity_uids = payload.pop("entity_uids", None)
        if entity_uids is not None:
            metadata = dict(payload.get("metadata") or {})
            metadata["entity_uids"] = entity_uids
            payload["metadata"] = metadata
        item = MemoryItem.from_dict(payload)
        if row.vector and any(v != 0.0 for v in row.vector):
            item.embedding = list(row.vector)
        return item
