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

from mu_engine.storage.domain.artifact import ContextArtifact
from mu_engine.storage.domain.conflict import ConflictEdges
from mu_engine.storage.domain.entity import EntityResolution
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.domain.recall import Scored, SparseQuery

__all__ = [
    "ConflictEdgeReader",
    "ContextRepository",
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

    async def put(self, item: MemoryItem) -> str:
        """Write ``item``, returning the RESIDENT memory id (add() return-idempotency fix,
        DATA-QUALITY-REASSESSMENT §3 "add() idempotency" / the D4 report).

        Normally ``item.id`` (a fresh write). On a write-time content-hash dedup hit (D4) — the
        namespace already holds a DIFFERENT, still-resident row for this exact ``content_hash`` —
        the store bumps THAT row's recency/TTL instead of forking a second physical entry, and
        this returns THAT row's id, not ``item.id``. Every implementation MUST return the id of
        whichever row is now physically resident under ``item.content_hash`` in this namespace, so
        a caller (``WriteStmStage``) can re-stamp its own id onto the SAME identity the store
        actually kept, instead of minting+returning an id the store never held (CANONICAL §7.1
        id-stability applied to the dedup path)."""
        ...

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None: ...

    async def recent(self, ns: Namespace, *, limit: int) -> list[Scored[MemoryItem]]: ...

    async def evict(self, ns: Namespace, memory_id: str) -> None: ...


class MtmTierRepository(Protocol):
    """Vector / MTM tier (``storage-pluggable §2.3``; filter-before-truncation for SHARED)."""

    async def upsert(self, item: MemoryItem) -> None: ...

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        """Point-get ONE MTM point by id from ``ns``'s partition (``None`` if absent) — the vector-
        tier twin of :meth:`StmTierRepository.get`, added for the TARGETED single-memory lifecycle
        verbs (``promote`` MTM->LTM / ``demote`` MTM->STM / ``update`` / ``delete``) which must
        LOCATE a memory in its current tier before acting on it (the sweep-oriented
        ``PromotionService``/``DemotionService`` take a caller-supplied window and never resolve a
        bare id). A real store read (``AsyncQdrantClient.retrieve`` on ``QdrantMtmAdapter``), NOT a
        semantic search (no query vector) and NOT filtered by ``state`` — a superseded/expired point
        is still returned so ``delete``/``update`` can act on it idempotently."""
        ...

    async def expire(self, ns: Namespace, memory_id: str, *, at: Any) -> None:
        """Soft-delete ONE MTM point (``delete`` verb, invalidate-don't-delete): flip its payload
        ``state`` to ``expired`` + stamp ``invalid_at=at`` so the mandatory ``state='active'``
        recall filter drops it from active recall, while the point itself STAYS (bi-temporal
        history — never a hard point deletion). Distinct from :meth:`invalidate` (which models a
        loser SUPERSEDED by a *winner*: ``state=superseded`` + ``superseded_by=<winner-id>``) — a
        plain user delete has no winner, so it must not fabricate a supersession edge. Distinct from
        :meth:`remove` (a genuine point deletion, for the demotion tier-down move). A payload-only
        PATCH (``set_payload``), the SAME primitive ``invalidate`` uses — vector untouched."""
        ...

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

    async def scan_for_demotion(self, ns: Namespace, *, limit: int) -> list[MemoryItem]:
        """Enumerate up to ``limit`` ACTIVE MTM points in ``ns``'s plane partition as
        forgetting-curve DEMOTION candidates (spec §7b; the MTM-enumeration primitive the
        automatic sweep needs to feed ``DemotionService.demote`` — previously the flagged
        "no MTM-tier enumeration primitive exists" gap, ``manager.py``/``maintenance.py``).

        A REAL, BOUNDED store read — NOT a query-vector semantic search (``semantic`` needs a
        vector and top-k truncates) and NOT a scan-everything foot-gun: it filters server-side
        to the plane's own partition (the SAME namespace/user-prefix match ``semantic`` compiles
        for recall) AND ``state='active'`` BEFORE paging, capped at ``limit``. The staleness
        decision itself is NOT made here — every returned item is re-scored by
        ``DemotionService`` against ``SalienceStrategy`` (the Ebbinghaus gate), so a fresh/salient
        item enumerated here is rescued, never demoted. Ordering is store-native (unordered);
        the cap bounds RAM on a shared box, it is not a "most-stale-first" priority read."""
        ...


class GraphStorePort(Protocol):
    """Graph / LTM tier — bi-temporal KG (``storage-pluggable §2.2``; graph MANDATORY)."""

    async def upsert_fact(self, item: MemoryItem) -> None: ...

    async def get_fact(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        """Point-get ONE ``:Memory`` LTM node by id from ``ns``'s graph partition (``None`` if
        absent) — the graph-tier twin of :meth:`StmTierRepository.get`/:meth:`MtmTierRepository.
        get`, added for the TARGETED lifecycle verbs (``update``/``delete``) which must LOCATE a
        fact before superseding/expiring it. A real ``MATCH (m:Memory {namespace, id})`` returning
        ``m.memory_json`` (``FalkorLtmAdapter``), NOT filtered by ``state``/validity — a superseded/
        expired fact is still returned so the verb can act on it idempotently. This is the by-ID
        point-read the bi-temporal read models (``facts_at``/``graph_recall``) never exposed."""
        ...

    async def expire(self, ns: Namespace, memory_id: str, *, at: Any) -> None:
        """Soft-delete ONE ``:Memory`` LTM node (``delete`` verb, invalidate-don't-delete): stamp
        ``state='expired'`` + ``invalid_at=at`` so every mandatory read filter
        (``state='active' AND (invalid_at='' OR invalid_at>now)``) drops it from active recall,
        while the node + its edges STAY on the graph (bi-temporal history) — NEVER the hard
        ``DETACH DELETE`` :meth:`gc_delete` performs (that is the retention sweep's GC of an
        ALREADY-dead, window-elapsed, chain-head-dead fact). Distinct from :meth:`invalidate`
        (loser SUPERSEDED_BY a winner) — a plain delete has no winner. Also closes the fact's own
        entity-entity edge (bi-temporal parity), exactly as :meth:`invalidate` does."""
        ...

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        # Mirrors `MtmTierRepository`'s own federate-live semantics (ADR 0030 "keep-and-scope"):
        # `None` (the DEFAULT) federates every one of the user's sessions; an explicit session id
        # narrows to that one; SHARED ignores it entirely. Absent here until now, which left the
        # LTM arm session-locked while the MTM arm federated — a split-brained recall fabric.
        session_scope: str | None = None,
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

    async def traverse_entities(
        self,
        ns: Namespace,
        *,
        query: str,
        max_hops: int,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        """Multi-hop entity-edge traversal (D-4, ARCHITECTURE-CONFORMANCE.md "LTM graph arm
        thin"): seeds on entity names found in ``query`` and walks the entity-entity edges
        ``upsert_fact`` materializes (B5/B6), up to ``max_hops`` (1-2), returning the underlying
        ``:Memory`` fact(s) each traversed edge traces back to. See
        ``FalkorLtmAdapter.traverse_entities`` for the full contract (seed matching, bi-temporal
        exclusion of superseded edges, hop-distance scoring).

        ``caller_identity_set`` carries the SAME Model-A caller PRINCIPAL-id set ``graph_recall``
        takes (CANONICAL-CONTRACTS.md §7.4) and is load-bearing on the SHARED plane: this arm
        DERIVES memory ids from a workspace-wide entity graph, so without it the hydration has
        nothing to filter ``m.authorized_ids`` against and returns ``:Memory`` rows from any room
        and any ACL in the workspace. The parameter exists because the PORT must be able to
        express the caller set — an implementation that ignores it on SHARED is an authorization
        bypass, not an optimization."""
        ...

    async def by_artifact(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]:
        """Reverse provenance lookup: every LTM ``:Memory`` node REFERENCES-linked to the
        ``:Artifact`` node ``artifact_id`` (software-arch spec §5 ``ContextRepository``
        docstring note + ``mu_contracts.ports.memory.MemoryTierRepository.by_artifact`` — "the
        FIRST-CLASS reverse lookup ... never a scan"). Traverses FROM the merged ``:Artifact``
        node via the existing ``REFERENCES`` edge this module's ``_upsert_fact_impl`` already
        writes whenever ``item.artifact_ref`` is set — never a ``:Memory``-label scan."""
        ...


# alias — the LTM tier repo IS the graph store port (spec §5 tree)
LtmTierRepository = GraphStorePort


# ------------------------------------------------------------------ ContextRepository (§5)
class ContextRepository(Protocol):
    """The provenance-root store port (software-arch spec §5, l.260-263): persists the RAW
    ingested activity as a :class:`~mu_engine.storage.domain.artifact.ContextArtifact` —
    step 1 of ``IngestService.ingest`` (spec §6, l.340) — before the STM capture memory (step 2)
    is minted as ``kind=REFERENCE`` pointing at it via ``artifact_ref``.

    ``open`` (spec l.262, a streaming ``AsyncReadable`` read of the body) is DEFERRED — this
    minimal-correct slice exposes ``get_blob`` (a bounded whole-body read) instead; see
    ``adapters/content_fs.py``'s module docstring for the flagged simplification and the
    full ``content_git.py`` (versioned, worktree-merge, "ported from Letta Context
    Repositories" — spec l.437) this is a floor beneath, not a replacement for.
    """

    async def put(self, art: ContextArtifact, blob: bytes) -> ContextArtifact:
        """Persist ``blob`` under ``art``'s locator; return the stored (possibly re-hashed)
        handle. Content-addressed + idempotent: re-``put``-ting the SAME ``(namespace, id,
        content_hash)`` is a no-op overwrite, never a duplicate."""
        ...

    async def get(self, ns: Namespace, artifact_id: str) -> ContextArtifact | None:
        """Hydrate the content-free metadata handle by id — never the body (CANONICAL §3.1)."""
        ...

    async def get_blob(self, ns: Namespace, artifact_id: str) -> bytes | None:
        """Hydrate the BODY by id (the bounded floor beneath spec l.262's streaming ``open``)."""
        ...


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
