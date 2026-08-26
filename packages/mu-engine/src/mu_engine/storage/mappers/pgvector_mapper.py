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
Postgres table name cannot be bound as a query parameter). The table name is therefore the shared
:func:`~mu_engine.storage.mappers.tenancy.tenant_partition_digest` — a deterministic SHA-256
hash of ``org``+``workspace`` jointly (see that function's docstring for why the two are joined on
``":"`` before hashing, and why the resulting digest is collision-resistant rather than
collision-resistant) — never the raw string — so it is injection-safe by construction regardless of
what characters an org/workspace id legally contains.
"""

from __future__ import annotations

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper
from mu_engine.storage.mappers.tenancy import tenant_partition_digest
from mu_engine.storage.ports import QdrantPoint

__all__ = ["PgVectorMapper", "pgvector_table_name"]


def pgvector_table_name(ns: Namespace, dim: int) -> str:
    """Deterministic, injection-safe Postgres identifier for the (org, workspace, visibility, dim)
    physical partition (mirrors ``qdrant_mapper.collection_name``'s partition grain) — uses the
    shared :func:`~mu_engine.storage.mappers.tenancy.tenant_partition_digest` (org+workspace
    hashed jointly, so two orgs sharing a workspace slug get DIFFERENT physical tables; CANONICAL
    §1 rule 6; the org-missing form was the tracked defect —
    ``ARCHITECTURE-CONFORMANCE.md`` §8/§10.4)."""
    return f"mu_mtm_pgv__{tenant_partition_digest(ns)}__{ns.visibility.value}__{dim}"


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
