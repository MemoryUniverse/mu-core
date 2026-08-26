"""``FaissMapper`` — MemoryItem <-> the generic vector StoreModel (``QdrantPoint``).

Same DRY rationale as the other two new vector mappers: DELEGATES payload projection to
:class:`~mu_engine.storage.mappers.qdrant_mapper.QdrantMapper` (one projection, many
partition-naming schemes, ``storage-schema-rowmapper-spec.md §5``). FAISS has no server-side
collection concept — the "collection" name doubles as the in-proc index key AND the on-disk file
basename (``storage-pluggable-spec.md §3.3``: "in-proc — real").
"""

from __future__ import annotations

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper
from mu_engine.storage.mappers.tenancy import tenant_partition_digest
from mu_engine.storage.ports import QdrantPoint

__all__ = ["FaissMapper", "faiss_collection_name"]


def faiss_collection_name(ns: Namespace, dim: int) -> str:
    """Deterministic filesystem-safe in-proc index key for the (org, workspace, visibility, dim)
    partition (PRIVATE-plane only, D3 — never used for a SHARED namespace) — uses the shared
    :func:`~mu_engine.storage.mappers.tenancy.tenant_partition_digest` (org+workspace hashed
    jointly, so two orgs sharing a workspace slug get DIFFERENT physical indexes/files; CANONICAL
    §1 rule 6; the org-missing form was the tracked defect — ``ARCHITECTURE-CONFORMANCE.md``
    §8/§10.4)."""
    return f"mu_mtm_faiss__{tenant_partition_digest(ns)}__{ns.visibility.value}__{dim}"


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
