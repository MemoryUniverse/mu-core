"""``ChromaMapper`` — MemoryItem <-> the generic vector StoreModel (``QdrantPoint``).

Same DRY rationale as :mod:`~mu_engine.storage.mappers.pgvector_mapper`: DELEGATES payload
projection to :class:`~mu_engine.storage.mappers.qdrant_mapper.QdrantMapper` (one projection, many
partition-naming schemes, ``storage-schema-rowmapper-spec.md §5``). PORT of mem0's Chroma store
shape (``other_repos/mem0/mem0/vector_stores/chroma.py:23-74`` — collection-per-tenant,
``metadatas``/``documents`` split).

Chroma collection names have their OWN charset/length rule (3-63 chars, alnum + ``.``/``_``/``-``,
must start and end alnum, no consecutive dots) distinct from Qdrant's — so, like pgvector, the
workspace is hashed rather than embedded raw (safe regardless of what characters
``Namespace._FORBIDDEN_NS_CHARS`` happens to still allow through).
"""

from __future__ import annotations

from hashlib import sha256

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper
from mu_engine.storage.ports import QdrantPoint

__all__ = ["ChromaMapper", "chroma_collection_name"]


def chroma_collection_name(ns: Namespace, dim: int) -> str:
    """Deterministic Chroma-legal collection name for the (workspace, visibility, dim) partition."""
    digest = sha256(ns.workspace.encode("utf-8")).hexdigest()[:16]
    return f"mu-mtm-chroma-{digest}-{ns.visibility.value}-{dim}"


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
