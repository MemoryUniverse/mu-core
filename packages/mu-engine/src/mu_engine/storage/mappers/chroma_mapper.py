"""``ChromaMapper`` — MemoryItem <-> the generic vector StoreModel (``QdrantPoint``).

Same DRY rationale as :mod:`~mu_engine.storage.mappers.pgvector_mapper`: DELEGATES payload
projection to :class:`~mu_engine.storage.mappers.qdrant_mapper.QdrantMapper` (one projection, many
partition-naming schemes, ``storage-schema-rowmapper-spec.md §5``). PORT of mem0's Chroma store
shape (``other_repos/mem0/mem0/vector_stores/chroma.py:23-74`` — collection-per-tenant,
``metadatas``/``documents`` split).

Chroma collection names have their OWN charset/length rule (3-63 chars, alnum + ``.``/``_``/``-``,
must start and end alnum, no consecutive dots) distinct from Qdrant's — so, like pgvector, the
``org``+``workspace`` pair is hashed jointly (via the shared
:func:`~mu_engine.storage.mappers.qdrant_mapper.tenant_partition_digest` helper — see its
docstring for what the digest derives and why it is collision-resistant rather than
collision-resistant) rather than embedded raw.
"""

from __future__ import annotations

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper, tenant_partition_digest
from mu_engine.storage.ports import QdrantPoint

__all__ = ["ChromaMapper", "chroma_collection_name"]


def chroma_collection_name(ns: Namespace, dim: int) -> str:
    """Deterministic Chroma-legal collection name for the (org, workspace, visibility, dim)
    partition — uses the shared :func:`tenant_partition_digest` (org+workspace hashed jointly, so
    two orgs sharing a workspace slug land in DIFFERENT physical collections; CANONICAL §1 rule 6;
    the org-missing form was the tracked defect — ``ARCHITECTURE-CONFORMANCE.md`` §8/§10.4)."""
    return f"mu-mtm-chroma-{tenant_partition_digest(ns)}-{ns.visibility.value}-{dim}"


class ChromaMapper:
    """Implements ``RowMapper[QdrantPoint]`` (spec §5) — reuses ``QdrantMapper``'s payload shape."""

    def __init__(self, *, dim: int) -> None:
        self.dim = dim
        self._base = QdrantMapper(dim=dim)

    def to_store(self, item: MemoryItem) -> QdrantPoint:
        row = self._base.to_store(item)
        return row.model_copy(
            update={"collection": chroma_collection_name(item.namespace, self.dim)}
        )

    def from_store(self, row: QdrantPoint) -> MemoryItem:
        return self._base.from_store(row)
