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
from mu_contracts.domain.model.memory import MemoryItem, Namespace, State, Tier
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

    # ---- Lifecycle-override (pin) additions — memory-health §3.1 (spec lines 160-177) --------

    async def set_pinned(
        self,
        ns: Namespace,
        id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None = None,
    ) -> int:
        """Id-stable upsert of the whole ``pinned`` field-group, returning the new version.

        Keyed by the TIER-STABLE ``MemoryItem.id`` (CANONICAL §7.1) and applied across every store
        the item resides in — the same cross-store, id-keyed shape as ``invalidate`` (§7.5), and
        for the same reason: a pin set at any tier must survive promotion/demotion, so it can
        never be keyed on the tier. ``by`` is the pinning principal (audit only — NEVER an authz
        principal, CANONICAL §7.4); ``reason`` is a short named classification, never memory text.

        Unpin passes ``pinned=False``, which clears ``pinned_at``/``pinned_by``/``pin_reason``.
        A missing id raises ``PinTargetNotFoundError`` — never a silent no-op.
        """
        ...

    async def enumerate(
        self,
        ns: Namespace,
        *,
        states: frozenset[State],
        tiers: frozenset[Tier] | None,
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        """The ONE bounded, PAGINATED partition walk (spec §3.1 lines 171-176).

        Shared by the demotion sweep and ``MemoryHealthService.assess`` so neither invents its own
        scan. ``pinned=False`` lets a sweep skip pinned items cheaply; ``pinned=None`` = do not
        filter on pin. Returns ``(page, next_cursor)`` with ``len(page) <= limit`` and
        ``next_cursor is None`` iff the walk is exhausted. **NEVER unbounded** — an implementation
        that ignores ``limit`` violates this port.
        """
        ...


@runtime_checkable
class StmTierRepository(MemoryTierRepository, Protocol):
    async def get(
        self,
        ns: Namespace,
        id: str,
        *,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> MemoryItem | None:
        """Point-get, Model-A authorized on a SHARED η (CANONICAL §7.4).

        Widens ``MemoryTierRepository.get`` with the OPTIONAL caller set — optional so every
        PRIVATE-plane caller is unchanged, required in effect on a SHARED η where ``None`` raises
        ``CallerIdentitySetRequiredError`` and a row the caller is not stamped on is returned as a
        MISS. This tier overrides it because this tier is the one a SHARED plane point-gets
        through, and the parameter's absence was a measured leak (ARCHITECTURE-DELTAS AD-129)."""
        ...

    async def recent(
        self,
        ns: Namespace,
        *,
        limit: int,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> list[MemoryItem]:
        """The recency floor. ``caller_identity_set`` is Model-A (CANONICAL §7.4) and is REQUIRED
        on a SHARED η: every returned row's exploded ``authorized_ids`` stamp must intersect it,
        an unstamped row is DENIED, and ``None`` raises ``CallerIdentitySetRequiredError``. On a
        PRIVATE η it is ``None`` — the own partition is authorized by ``to_prefix()`` (§1 rule 5).

        The parameter exists because its ABSENCE was a live SHARED-plane leak: the MTM/LTM arms of
        recall were Model-A filtered and this one could not express the caller set at all
        (ARCHITECTURE-DELTAS AD-128)."""
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

    async def set_pinned(
        self,
        ns: Namespace,
        id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None = None,
    ) -> int:
        """Fan the id-stable pin upsert across every tier the id lives in via ``TierRouter``
        (spec §3.1 line 179). Returns the new version."""
        ...

    async def enumerate(
        self,
        ns: Namespace,
        *,
        states: frozenset[State],
        tiers: frozenset[Tier] | None,
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        """The façade's bounded partition walk, fanning across tiers via ``TierRouter``."""
        ...
