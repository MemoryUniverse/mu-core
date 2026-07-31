"""Storage ports — the ``typing.Protocol`` edges the domain talks to (DEV-STANDARDS rule 5).

Repository pattern throughout: services depend on these Protocols, never on a store client.
Every port is fully async (DEV-STANDARDS rule 1). Shapes follow
``storage-schema-rowmapper-spec.md §5`` (RowMapper/StoreModel), §1.4 (ConflictEdgeReader),
and ``storage-pluggable-spec.md §2`` (tier repos).

RE-HOME NOTE: CANONICAL pins these ports into ``mu-contracts/ports/``; defined here because
that package is a scaffold this phase (empty ``ports/__init__.py``).
"""

from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from mu_engine.storage.domain.conflict import ConflictEdges
from mu_engine.storage.domain.entity import EntityResolution
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import Scored, SparseQuery

__all__ = [
    "ConflictEdgeReader",
    "ControlPlaneRepository",
    "EdgeSpec",
    "GraphNodeRow",
    "GraphStorePort",
    "LtmTierRepository",
    "MtmTierRepository",
    "QdrantPoint",
    "RedisRecord",
    "RelationalRow",
    "RowMapper",
    "StmTierRepository",
    "StoreModel",
]


# ------------------------------------------------------------------ StoreModels (spec §5)
class RedisRecord(BaseModel, frozen=True):
    """Redis STM row — ``blob`` is ``MemoryItem`` JSON (LOSSLESS)."""

    key: str
    ttl_s: int | None
    blob: str


class QdrantPoint(BaseModel, frozen=True):
    """Qdrant MTM point — vector nulled out of the payload."""

    point_id: str
    vector: list[float]
    sparse: dict[str, Any] | None
    payload: dict[str, Any]
    collection: str


class EdgeSpec(BaseModel, frozen=True):
    """One openCypher edge to MERGE alongside a graph node."""

    rel_type: str
    props: dict[str, Any] = {}
    target_labels: tuple[str, ...] = ()
    target_merge_key: dict[str, Any] = {}


class GraphNodeRow(BaseModel, frozen=True):
    """FalkorDB LTM node + its edges."""

    labels: tuple[str, ...]
    merge_key: dict[str, Any]
    props: dict[str, Any]
    edges: tuple[EdgeSpec, ...] = ()


class RelationalRow(BaseModel, frozen=True):
    """A content-free relational row (spec §0 — ids/hashes/enums/counts/timestamps only)."""

    table: str
    pk: dict[str, Any]
    cols: dict[str, Any]


StoreModel = RedisRecord | QdrantPoint | GraphNodeRow | RelationalRow

SM = TypeVar("SM", RedisRecord, QdrantPoint, GraphNodeRow, RelationalRow)


# ------------------------------------------------------------------ RowMapper (spec §5)
@runtime_checkable
class RowMapper(Protocol, Generic[SM]):
    """The ``to_store``/``from_store`` seam — the ONLY place field-projection lives (spec §5).

    Contract (spec §5): round-trip fidelity, id-stability, tenancy via ``to_prefix()``,
    first-class ``artifact_ref``/``embedding_ref``/``provenance_id`` (never JSON overflow).
    """

    def to_store(self, item: MemoryItem) -> SM: ...

    def from_store(self, row: SM) -> MemoryItem: ...


# ------------------------------------------------------------------ tier repositories
class StmTierRepository(Protocol):
    """KV / STM tier (``storage-pluggable §1``). Recency floor + TTL + chash dedup."""

    async def put(self, item: MemoryItem) -> None: ...

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None: ...

    async def recent(self, ns: Namespace, *, limit: int) -> list[Scored[MemoryItem]]: ...

    async def evict(self, ns: Namespace, memory_id: str) -> None: ...


class MtmTierRepository(Protocol):
    """Vector / MTM tier (``storage-pluggable §2.3``; filter-before-truncation for SHARED)."""

    async def upsert(self, item: MemoryItem) -> None: ...

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: SparseQuery | None = None,
    ) -> list[Scored[MemoryItem]]: ...

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: Any, reason: str
    ) -> None: ...

    async def remove(self, ns: Namespace, memory_id: str) -> None:
        """Plain point deletion — NOT supersession (spec §7b demotion; CF-2, MLM-STAGE2-
        CARRYOVER.md). ``invalidate`` models a loser being superseded by a winner (payload
        overwritten with ``state=superseded``, the point stays); ``remove`` genuinely deletes
        the point (a forgetting-curve tier-down move has no "winner"). Distinct operations —
        never substitute one for the other."""
        ...


class GraphStorePort(Protocol):
    """Graph / LTM tier — bi-temporal KG (``storage-pluggable §2.2``; graph MANDATORY)."""

    async def upsert_fact(self, item: MemoryItem) -> None: ...

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]: ...

    async def facts_at(
        self, ns: Namespace, at: Any, *, subject: str | None = None
    ) -> list[MemoryItem]: ...

    async def find_conflicts(
        self, ns: Namespace, subject: str, predicate: str
    ) -> list[MemoryItem]: ...

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: Any, reason: str
    ) -> None: ...

    async def mark_conflict(self, ns: Namespace, a_id: str, b_id: str, *, at: Any) -> None:
        """Tag two STILL-ACTIVE facts ``CONFLICTS_WITH`` each other — no state/``invalid_at``
        write to either side (D3, spec §8 "never fabricate"). Used for a verdict that could NOT
        be auto-applied (a genuinely undecidable ``PENDING`` bi-temporal tie, or a MANUAL-policy-
        withheld ``SUPERSEDE``/``SELF_EXPIRE``) so the conflict is still visible on the GRAPH
        itself, not only in the adjudicator's side-channel ``ConflictRecord`` inbox. Distinct from
        ``invalidate`` (which ALSO merges a ``CONFLICTS_WITH`` edge, but only as a byproduct of
        flipping the loser to ``state=superseded``) — this is the bare, standalone edge for the
        both-stay-active case."""
        ...

    async def resolve_entity(self, ns: Namespace, name: str) -> EntityResolution: ...


# alias — the LTM tier repo IS the graph store port (spec §5 tree)
LtmTierRepository = GraphStorePort


# ------------------------------------------------------------------ relational control plane
class ControlPlaneRepository(Protocol):
    """Content-free relational mirror / control plane (spec §2).

    ``sync_provenance`` is the idempotent (``ON CONFLICT DO UPDATE``) sync target keyed by
    ``ux_prov_chash`` (spec §2.4).
    """

    async def sync_provenance(self, item: MemoryItem) -> str: ...

    async def get_provenance(self, workspace_id: str, memory_id: str) -> dict[str, Any] | None: ...

    async def list_by_namespace(
        self, namespace_prefix: str, *, limit: int
    ) -> list[dict[str, Any]]: ...

    async def append_audit(
        self,
        *,
        org_id: str,
        workspace_id: str,
        actor_id: str,
        action: str,
        target_id: str | None,
        success: bool,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


# ------------------------------------------------------------------ ConflictEdgeReader (spec §1.4)
class ConflictEdgeReader(Protocol):
    """Bounded, content-free conflict-adjacency projection (spec §1.4).

    Loads ONLY conflict rows whose member set intersects ``memory_ids`` (the health-view
    page), never a full-partition scan; scoped by ``to_prefix()``.
    """

    async def edges_for(self, ns: Namespace, memory_ids: frozenset[str]) -> ConflictEdges: ...
