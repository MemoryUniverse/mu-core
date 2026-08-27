"""``WeaviateMapper`` — MemoryItem <-> the generic vector StoreModel (``QdrantPoint``).

ADR 0050 (``docs/decisions/0050-weaviate-shared-plane-vector-tier.md``): Weaviate, in native
multi-tenancy mode, is the SHARED-plane MTM vector backend. DELEGATES the payload-projection
logic to :class:`~mu_engine.storage.mappers.qdrant_mapper.QdrantMapper` — the SAME DRY rationale
:class:`~mu_engine.storage.mappers.pgvector_mapper.PgVectorMapper` already uses: the payload
SHAPE every vector engine stores is backend-agnostic (``storage-schema-rowmapper-spec.md §5``),
only the physical PARTITION differs per engine.

**The partition shape here is genuinely different from every other registered vector backend —
read this before touching either name function below.** Qdrant/pgvector/chroma/faiss all
partition by *collection-per-namespace* (``{digest}__{visibility}__{dim}``, one physical
collection/table per ``(org, workspace, visibility, dim)``). Weaviate's own native multi-tenancy
makes a **tenant** the physical shard (ADR 0050 Context), so this mapper splits the same grain
across TWO names instead of one:

* :func:`collection_name` — the ONE Weaviate class every namespace of a given ``dim`` SHARES
  (``MuMtm{dim}``). No ``org``/``workspace``/``visibility`` in it at all: those live in the
  tenant, not the class.
* :func:`tenant_name` — the physical shard INSIDE that class, one per ``(org, workspace)`` pair
  (:func:`~mu_engine.storage.mappers.tenancy.tenant_partition_digest`). This is the ADR's
  explicit instruction: *"The tenant name MUST use ``tenant_partition_digest(ns)``."*

``visibility`` is deliberately NOT part of either name — the ADR calls it out as a WITHIN-shard
component, enforced by the adapter's mandatory ``visibility`` equality filter on every read/write,
exactly like Qdrant enforces ``user``/``session`` within its own (coarser than per-user) collection.
"""

from __future__ import annotations

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import QdrantMapper
from mu_engine.storage.mappers.tenancy import tenant_partition_digest
from mu_engine.storage.ports import QdrantPoint

__all__ = ["WeaviateMapper", "collection_name", "tenant_name"]


def collection_name(dim: int) -> str:
    """The ONE Weaviate class every namespace of this ``dim`` shares.

    Deliberately carries NO caller-controlled text (``org``/``workspace``/``visibility`` all live
    in :func:`tenant_name` instead) — a Weaviate class name must start with an uppercase letter
    and contain only ``[A-Za-z0-9_]`` (GraphQL type-name rules); a fixed ``"MuMtm"`` prefix plus a
    plain integer satisfies that unconditionally, with no escaping/validation needed because
    nothing caller-supplied ever reaches it.
    """
    return f"MuMtm{dim}"


def tenant_name(ns: Namespace) -> str:
    """The Weaviate tenant — i.e. the physical shard (ADR 0050 Context: *"each shard is
    dedicated to holding data for a single tenant"*) — ``ns`` lands in.

    **MUST be :func:`~mu_engine.storage.mappers.tenancy.tenant_partition_digest`, never a literal
    ``f"{org}__{workspace}"`` join.** ADR 0050 names this exact defect (its own D-1 delta still
    proposes the unsafe join, and this function overrides it): ``_`` is not in
    ``Namespace._FORBIDDEN_NS_CHARS``, so ``org="a__b"/workspace="c"`` and ``org="a"/workspace=
    "b__c"`` would collide under a literal join — two different orgs sharing one physical Weaviate
    tenant/shard, the exact cross-tenant leak ``tenant_partition_digest`` exists to close (see that
    function's docstring for the full derivation and the proven collision it closes). Hashing on
    ``":"`` (forbidden inside a single ``org``/``workspace`` value — ``_FORBIDDEN_NS_CHARS``) makes
    the join point unambiguous BEFORE it is hashed, which a literal ``"__"`` join is not.
    """
    return tenant_partition_digest(ns)


class WeaviateMapper:
    """Implements ``RowMapper[QdrantPoint]`` (spec §5) — reuses ``QdrantMapper``'s payload shape,
    the SAME delegation pattern ``PgVectorMapper``/``ChromaMapper``/``FaissMapper`` already use.

    ``QdrantPoint.collection`` carries the Weaviate CLASS name only (:func:`collection_name`) —
    the tenant is a separate, per-call parameter on the Weaviate client (``with_tenant(...)``),
    not part of any name this mapper produces, so it is resolved by the adapter via
    :func:`tenant_name`, not threaded through the generic ``StoreModel`` shape.
    """

    def __init__(self, *, dim: int) -> None:
        self.dim = dim
        self._base = QdrantMapper(dim=dim)

    def to_store(self, item: MemoryItem) -> QdrantPoint:
        row = self._base.to_store(item)
        return row.model_copy(update={"collection": collection_name(self.dim)})

    def from_store(self, row: QdrantPoint) -> MemoryItem:
        return self._base.from_store(row)
