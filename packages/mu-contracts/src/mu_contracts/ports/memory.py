"""Memory repository ports — the tier repositories + the application-facing façade.

Authority: memory-layer-design.md §2, CANONICAL §6-P2 (the ``MemoryRepository`` façade over the
engine-internal ``Stm/Mtm/Ltm`` tier repos), §7.1 (id stability), §7.4 (Model-A ``authorized_ids``
is the CALLER principal-id set, never a memory-id set), §7.5 (``invalidate`` cross-store supersede
+ ``state='active'`` hot-read floor). Repository pattern (DEV-STANDARDS rule 5): the domain talks
to these Protocols; concrete Redis/Qdrant/FalkorDB adapters live behind them in mu-engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.entity import EntityResolution
from mu_contracts.domain.model.memory import MemoryItem, Namespace, Tier
from mu_contracts.domain.model.recall import CallerIdentitySet, Scored, SparseQuery, Vector

__all__ = [
    "LtmTierRepository",
    "MemoryRepository",
    "MemoryTierRepository",
    "MtmTierRepository",
    "StmTierRepository",
]


@runtime_checkable
class MemoryTierRepository(Protocol):
    """Common tier-repo edge (memory-layer §2). ``by_artifact`` is the FIRST-CLASS reverse lookup
    (artifact → referencing memories), backed by the ``artifact_ref`` index; never a scan."""

    tier: Tier

    async def add(self, item: MemoryItem) -> None: ...

    async def get(self, ns: Namespace, id: str) -> MemoryItem | None: ...

    async def upsert(self, item: MemoryItem) -> None: ...

    async def delete(self, ns: Namespace, id: str) -> bool: ...

    async def by_artifact(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]: ...


@runtime_checkable
class StmTierRepository(MemoryTierRepository, Protocol):
    async def recent(self, ns: Namespace, *, limit: int) -> list[MemoryItem]:  # recency floor
        ...

    async def set_ttl(self, ns: Namespace, id: str, ttl_s: int) -> None: ...


@runtime_checkable
class MtmTierRepository(MemoryTierRepository, Protocol):
    async def semantic(
        self,
        ns: Namespace,
        query_vec: Vector,
        *,
        k: int,
        authorized_ids: CallerIdentitySet,
        sparse_query: SparseQuery | None = None,
    ) -> list[Scored[MemoryItem]]:
        """Model A (CANONICAL §7.4): ``authorized_ids`` is the CALLER PRINCIPAL-id set, filtered
        server-side via ``MatchAny`` — never a memory-id read-set, never a role/session token.
        ``state='active'`` is added on every prefetch arm + the outer filter (§7.5)."""
        ...

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        """The id-stable payload upsert stamping ``state='superseded'`` + ``invalid_at`` on an
        MTM-resident loser (CANONICAL §7.5 B1) — critical for an un-promoted loser where the
        graph-only ``LtmTierRepository.invalidate`` is a no-op. Never deletes."""
        ...


@runtime_checkable
class LtmTierRepository(MemoryTierRepository, Protocol):
    async def graph_recall(
        self, ns: Namespace, query_vec: Vector, *, k: int, authorized_ids: CallerIdentitySet
    ) -> list[Scored[MemoryItem]]: ...

    async def facts_at(self, ns: Namespace, *, at: datetime) -> list[MemoryItem]:
        """Bi-temporal as-of read; never returns an edge whose ``invalid_at`` is set at ``at``."""
        ...

    async def resolve_entity(self, ns: Namespace, name: str) -> EntityResolution: ...

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None: ...


@runtime_checkable
class MemoryRepository(Protocol):
    """The application-facing façade (CANONICAL §6-P2). ``semantic`` embeds via ``EmbeddingPort``
    and threads the resolved ``authorized_ids`` before delegating to the tier repos; ``by_artifact``
    fans across tiers. Read-path calls take a no-write UoW (recall is side-effect-free)."""

    async def add(self, item: MemoryItem) -> None: ...

    async def get(self, ns: Namespace, id: str) -> MemoryItem | None: ...

    async def semantic(
        self, ns: Namespace, query: str, *, k: int, authorized_ids: CallerIdentitySet
    ) -> list[Scored[MemoryItem]]: ...

    async def by_artifact(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]: ...
