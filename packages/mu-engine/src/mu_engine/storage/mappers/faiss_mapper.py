"""``FaissMapper`` — MemoryItem <-> the generic vector StoreModel (``QdrantPoint``).

Same DRY rationale as the other two new vector mappers: DELEGATES payload projection to
:class:`~mu_engine.storage.mappers.qdrant_mapper.QdrantMapper` (one projection, many
partition-naming schemes, ``storage-schema-rowmapper-spec.md §5``). FAISS has no server-side
collection concept — the "collection" name doubles as the in-proc index key AND the on-disk file
basename (``storage-pluggable-spec.md §3.3``: "in-proc — real").
"""

from __future__ import annotations

from hashlib import sha256

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper
from mu_engine.storage.ports import QdrantPoint

__all__ = ["FaissMapper", "faiss_collection_name"]


def faiss_collection_name(ns: Namespace, dim: int) -> str:
    """Deterministic filesystem-safe in-proc index key for the (workspace, visibility, dim)
    partition (PRIVATE-plane only, D3 — never used for a SHARED namespace)."""
    digest = sha256(ns.workspace.encode("utf-8")).hexdigest()[:16]
    return f"mu_mtm_faiss__{digest}__{ns.visibility.value}__{dim}"


class FaissMapper:
    """Implements ``RowMapper[QdrantPoint]`` (spec §5) — reuses ``QdrantMapper``'s payload shape."""

    def __init__(self, *, dim: int) -> None:
        self.dim = dim
        self._base = QdrantMapper(dim=dim)

    def to_store(self, item: MemoryItem) -> QdrantPoint:
        row = self._base.to_store(item)
        return row.model_copy(
            update={"collection": faiss_collection_name(item.namespace, self.dim)}
        )

    def from_store(self, row: QdrantPoint) -> MemoryItem:
        return self._base.from_store(row)
