"""``PgVectorMapper`` — MemoryItem <-> the generic vector StoreModel (``QdrantPoint``).

DELEGATES the payload-projection logic to :class:`~mu_engine.storage.mappers.qdrant_mapper.
QdrantMapper` — the payload SHAPE every vector engine stores is backend-agnostic
(``storage-schema-rowmapper-spec.md §5``: one common field set every ``RowMapper`` round-trips;
``QdrantPoint`` is already generic — ``point_id``/``vector``/``sparse``/``payload``/``collection``
— nothing in its shape is Qdrant-specific). Only the physical PARTITION NAME differs per vector
engine (``storage-pluggable-spec.md §3.3``, mem0 ``VectorStoreFactory`` — one adapter per engine,
one shared projection). Reusing :class:`QdrantMapper` instead of re-deriving the payload keeps
that projection in ONE place (DEV-STANDARDS rule 6, DRY).

PORT of mem0's pgvector table shape (``other_repos/mem0/mem0/vector_stores/pgvector.py:156-161``
— ``id``/``vector``/``payload`` columns) — the naming scheme below is OUR OWN addition (mem0 lets
the caller pick a raw ``collection_name`` string and interpolates it directly into DDL, which is
NOT safe here: ``Namespace`` only forbids separator characters
(``mu_contracts.domain.model.memory._FORBIDDEN_NS_CHARS``), not arbitrary SQL metacharacters, and a
Postgres table name cannot be bound as a query parameter). The table name is therefore a
deterministic SHA-256 hash of the workspace — never the raw string — so it is injection-safe by
construction regardless of what characters a workspace id legally contains.
"""

from __future__ import annotations

from hashlib import sha256

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper
from mu_engine.storage.ports import QdrantPoint

__all__ = ["PgVectorMapper", "pgvector_table_name"]


def pgvector_table_name(ns: Namespace, dim: int) -> str:
    """Deterministic, injection-safe Postgres identifier for the (workspace, visibility, dim)
    physical partition (mirrors ``qdrant_mapper.collection_name``'s partition grain, spec §3.1)."""
    digest = sha256(ns.workspace.encode("utf-8")).hexdigest()[:16]
    return f"mu_mtm_pgv__{digest}__{ns.visibility.value}__{dim}"


class PgVectorMapper:
    """Implements ``RowMapper[QdrantPoint]`` (spec §5) — reuses ``QdrantMapper``'s payload shape."""

    def __init__(self, *, dim: int) -> None:
        self.dim = dim
        self._base = QdrantMapper(dim=dim)

    def to_store(self, item: MemoryItem) -> QdrantPoint:
        row = self._base.to_store(item)
        return row.model_copy(update={"collection": pgvector_table_name(item.namespace, self.dim)})

    def from_store(self, row: QdrantPoint) -> MemoryItem:
        return self._base.from_store(row)
