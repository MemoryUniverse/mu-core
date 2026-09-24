"""Storage ports — the ``typing.Protocol`` edges the domain talks to (DEV-STANDARDS rule 5).

Repository pattern throughout: services depend on these Protocols, never on a store client.
Every port is fully async (DEV-STANDARDS rule 1). Shapes follow
``storage-schema-rowmapper-spec.md §5`` (RowMapper/StoreModel), §1.4 (ConflictEdgeReader),
and ``storage-pluggable-spec.md §2`` (tier repos).

**StoreModels + RowMapper are NOT defined here** (ARCHITECTURE-DELTAS **AD-184**, fix-impl,
2026-08-30). Until that fix this module declared its OWN, incompatible, second copy of
``RedisRecord``/``QdrantPoint``/``EdgeSpec``/``GraphNodeRow``/``RelationalRow``/``RowMapper`` —
``mu_contracts.ports.stores`` published one shape (``QdrantPoint.sparse`` optional,
``extra="forbid"``) while every mapper and adapter in THIS package imported a second, looser one
(``sparse`` required with no default, no ``extra`` guard) — the exact two-spellings failure
ADR 0047 exists to prevent, reproduced inside one repo (it already cost a full day once, across
repos: WIRE-CONFORMANCE-FINDING-0821.md). They are now imported from ``mu_contracts.ports.stores``
— the STRICTER of the two shapes — and re-exported here so every existing
``from mu_engine.storage.ports import QdrantPoint`` (etc.) import keeps working unchanged. Do NOT
redeclare any of them in this module again; the one legal home is ``mu-contracts`` (CANONICAL
pins storage-vocabulary DTOs there — pure wire shapes both planes must agree on).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, TypeVar

from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_contracts.ports.stores import (
    EdgeSpec,
    GraphNodeRow,
    QdrantPoint,
    RedisRecord,
    RelationalRow,
    RowMapper,
    StoreModel,
)
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

# Retained for any downstream annotation spelled ``SM`` against this module (none in-tree today —
# verified: no concrete mapper parameterizes ``RowMapper[...]`` at runtime, every mapper satisfies
# it structurally). Bound to the imported union rather than re-declared against the four classes
# individually so it cannot silently re-diverge from it.
SM = TypeVar("SM", bound=StoreModel)


# ------------------------------------------------------------------ tier repositories
class StmTierRepository(Protocol):
    """KV / STM tier (``storage-pluggable §1``). Recency floor + TTL + chash dedup."""

    async def put(self, item: MemoryItem, *, ttl_s: int | None = None) -> str:
        """Write ``item``, returning the RESIDENT memory id (add() return-idempotency fix,
        DATA-QUALITY-REASSESSMENT §3 "add() idempotency" / the D4 report).

        Normally ``item.id`` (a fresh write). On a write-time content-hash dedup hit (D4) — the
        namespace already holds a DIFFERENT, still-resident row for this exact ``content_hash`` —
        the store bumps THAT row's recency/TTL instead of forking a second physical entry, and
        this returns THAT row's id, not ``item.id``. Every implementation MUST return the id of
        whichever row is now physically resident under ``item.content_hash`` in this namespace, so
        a caller (``WriteStmStage``) can re-stamp its own id onto the SAME identity the store
        actually kept, instead of minting+returning an id the store never held (CANONICAL §7.1
        id-stability applied to the dedup path).

        ``ttl_s`` (FAULT-HUNT-0924 F1 fix, ADR 0054): an explicit TTL override for THIS write,
        replacing whatever default the adapter's own mapper/construction-time setting would
        otherwise stamp (e.g. ``RedisMapper.default_ttl_s``). ``None`` (every pre-existing caller)
        preserves the adapter's prior default byte-for-byte — this parameter is purely additive.
        The one caller that passes it today is ``DemotionService``'s write-ahead STM copy
        (``LifecycleSettings.demoted_stm_ttl_s``): a demoted memory is not a fresh, unprocessed
        capture, so it must not silently inherit the capture-buffer TTL (``IngestSettings.
        stm_ttl_s``) a real ingest write uses."""
        ...

    async def get(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> MemoryItem | None:
        """One keyed row, or ``None``.

        ``caller_identity_set`` is the Model-A CALLER PRINCIPAL set (CANONICAL §7.4) and carries
        the SAME per-visibility contract as :meth:`recent`: ignored on a PRIVATE η (§1 rule 5
        authorizes the own partition by key), REQUIRED on a SHARED one, where a row whose exploded
        ``authorized_ids`` stamp does not intersect it — an unstamped row included — is returned as
        ``None``. ``None`` on a SHARED η raises
        :class:`~mu_contracts.domain.errors.CallerIdentitySetRequiredError`.

        A denial is a MISS, deliberately: the caller supplies the id, so "you may not read this"
        and "no such row" must be indistinguishable or this verb is an existence oracle over the
        partition. The parameter exists because its ABSENCE was the UNBOUNDED half of AD-128 — the
        recency floor is capped by ``recency_floor_limit`` and ``stm_ttl_s``, a by-id read by
        neither (ARCHITECTURE-DELTAS **AD-129**).
        """
        ...

    async def recent(
        self,
        ns: Namespace,
        *,
        limit: int,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> list[Scored[MemoryItem]]:
        """The recency-floor window over ``ns``, newest-first, at most ``limit`` rows.

        ``caller_identity_set`` is the Model-A CALLER PRINCIPAL set (CANONICAL §7.4), and it is
        REQUIRED on a SHARED η — the port grew it because it could not express it, which is the
        whole of AD-128: the MTM and LTM arms of ``ThreeChannelRecallRanker`` both received the
        caller set and the STM floor arm could not, so a non-member of a room read the room's
        items by naming its session. (The same shape as the C2 fix: *"the port must be able to
        express the caller set"*.)

        Contract, per visibility:

        * **PRIVATE η** — ``None`` is correct and expected: the whole own-partition is authorized
          by the ``to_prefix()`` key (§1 rule 5 / §7.4 *"PRIVATE items are isolated by
          ``Namespace.to_prefix()`` partitioning, not via authorized_ids"*). Any set passed here
          is ignored.
        * **SHARED η** — every returned row MUST satisfy
          :func:`~mu_contracts.domain.model.authorized_ids.model_a_permits`: its exploded
          ``authorized_ids`` stamp intersects this set. A row with no stamp is DENIED (fail
          closed — an unstamped row is one no governance decision was recorded for). ``None``
          raises :class:`~mu_contracts.domain.errors.CallerIdentitySetRequiredError`; it is never
          an empty result and never an unfiltered one.

        The filter is applied over the ALREADY-BOUNDED ``limit`` window, so a caller may receive
        FEWER than ``limit`` rows on a SHARED η — a floor is a floor, not a quota, and no
        implementation may widen the window to refill it (that would be the over-fetch §7.4
        rejects with Model B).
        """
        ...

    async def evict(self, ns: Namespace, memory_id: str) -> None: ...

    async def put_demoted(self, item: MemoryItem, *, ttl_s: int, at: datetime) -> str:
        """The demotion write-ahead verb (AD-250 fix, ADR 0061).

        `FAULT-HUNT-0924.md` F1 named it and ADR 0058's verify pass proved it: raising the
        demoted copy's TTL (ADR 0054) fixed WHEN it disappears, not WHETHER anything could ever
        find it again. `DemotionService`'s write-ahead copy used to go through :meth:`put`,
        landing in the SAME recency ZSET a fresh capture uses, scored by `item.created_at` — the
        item's ORIGINAL age, not the demotion instant. A demoted memory was demoted precisely
        for being old/unused, so it re-entered the recency floor underneath every genuinely new
        turn and dropped out of `recent(limit=recency_floor_limit)` (default 10) as soon as ten
        newer STM writes existed in that session — almost immediately. Its MTM point is already
        gone by then, so nothing could ever re-recall it to raise `access_count`, so the
        recall-rescue ADR 0034 describes had a reachable GATE (`promote_stm_mtm`) and no
        reachable TRIGGER.

        This is the fix: a demoted item gets its OWN discoverability index
        (:meth:`demoted`/``RedisMapper.demoted_key``), scored by ``at`` (the DEMOTION instant,
        never `item.created_at`), that a fresh capture can never push it out of — because fresh
        captures never enter this index. It is conceptually the "cold, still-in-KV" sub-tier the
        fault hunt itself named: not a member of "what was just said" (never `is_floor`-eligible
        — `ports.py`'s `recent` docstring's protection guarantee is untouched), but still a real,
        relevance-competing STM-channel candidate (`ThreeChannelRecallRanker.rank` fuses it in
        at `weight_stm`, the SAME discount an ordinary floor candidate gets). `ttl_s` is REQUIRED
        (never a silent capture-buffer default — this verb has exactly one caller and it always
        knows its own retention window, `LifecycleSettings.demoted_stm_ttl_s`)."""
        ...

    async def demoted(
        self,
        ns: Namespace,
        *,
        limit: int,
        caller_identity_set: CallerIdentitySet | None = None,
    ) -> list[Scored[MemoryItem]]:
        """The demoted-item channel (AD-250 fix, ADR 0061) — :meth:`put_demoted`'s read side,
        newest-demoted-first, at most ``limit`` rows. Same Model-A contract as :meth:`recent`
        (AD-128): ``caller_identity_set`` is ignored on PRIVATE (own-partition key authorizes),
        REQUIRED on SHARED (a missing set raises
        :class:`~mu_contracts.domain.errors.CallerIdentitySetRequiredError`; a denied row is
        excluded, never included unstamped)."""
        ...

    async def reinforce(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        at: datetime,
        relevance_score: float | None = None,
    ) -> MemoryItem | None:
        """The read-stat write-back a genuine recall hit performs (AD-250 fix, ADR 0061).

        `recall-service-design.md` §5.1/line 609 has always CLAIMED this: *"the only mutation
        [recall] can cause is the read-stat write-back the stores already do idempotently on
        read"* — cited against specific file:lines that, on inspection, implemented no such
        thing for any ``StmTierRepository`` adapter: nothing ever incremented ``access_count`` on
        a recall hit, for ANY item, demoted or not (the only place ``access_count`` was ever
        written in this engine was ``DistillPipeline``'s identical-content LTM reconciliation, a
        DISTILL-time event, never a recall one). So the ADR 0034 / ``DemotionService`` rescue
        narrative — "a re-recalled item's raised ``access_count`` rescues it" — had no real
        trigger, even for an item :meth:`recent`/:meth:`demoted` could genuinely see. This method
        is that trigger, implemented for real: called by
        :class:`~mu_engine.services.recall.ranker.ThreeChannelRecallRanker` once per DISTINCT id
        that actually made it into a returned ``RecallResult``. (ADR 0061 shipped this filtered
        to ``channel == "stm"``; AD-259 widened it to every returned id — a memory living in BOTH
        tiers fuses to ONE view under ONE label, so the label filter silently reinforced only one
        of its two independently-gated rows. Over-inclusion is free: see the ``None`` contract
        below.)

        Increments ``access_count`` by 1 and refreshes ``updated_at`` to ``at`` — MUST leave
        ``item.created_at`` untouched (the field :meth:`~mu_engine.lifecycle.salience.
        SalienceStrategy._recency` reads; this is a read-STAT write-back, not a re-capture) and
        MUST preserve the row's REMAINING ttl (a recall must never reset a demoted item's
        ``demoted_stm_ttl_s`` retention clock back to a fresh one — the store-level analogue of
        ``KEEPTTL``, the same discipline :meth:`set_pinned` already uses for the same reason).
        Deliberately does NOT touch either discoverability index's score — :meth:`demoted`'s own
        window does not shrink from unrelated fresh-capture activity the way :meth:`recent`'s
        does (``put_demoted``'s docstring), so there is nothing here that needs re-scoring to
        stay reachable.

        ``None`` if ``memory_id`` is absent (already TTL-expired, evicted, or never existed) — a
        no-op, never a raise: the caller passes only ids ITS OWN prior read just returned, so an
        absence here means the row expired in the narrow window between that read and this write,
        which is exactly the race :meth:`recent`'s own self-heal branch already treats as
        ordinary. Best-effort by contract — a caller MUST NOT let a failure here fail the read it
        is reinforcing (the same "enhancement, not a named channel" degrade discipline
        ``ranker.py``'s neighbour expansion already uses).

        **AD-266 fix — ``relevance_score``.** ``None`` (the default) leaves the stored
        ``relevance_score`` untouched, exactly as every call site before this fix behaved. When
        the caller passes the query-relevance score this hit was actually found at (the fused or
        reranked score ``ThreeChannelRecallRanker`` already computed for the same hit —
        ``recall-service-design.md`` §5.1's "``relevance=score``" — the prototype's
        ``stm_redis.py:323-341`` had always done this), it is written back alongside
        ``access_count``/``last_seen`` in the SAME payload PATCH, never a second round trip. Also
        stamps ``last_seen=at`` (the D2 fix — a distinct field from ``updated_at``, see
        ``storage/domain/memory.py``'s ``last_seen`` docstring)."""
        ...


class MtmTierRepository(Protocol):
    """Vector / MTM tier (``storage-pluggable §2.3``; filter-before-truncation for SHARED)."""

    async def upsert(self, item: MemoryItem) -> None: ...

    async def reinforce(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        at: datetime,
        relevance_score: float | None = None,
    ) -> MemoryItem | None:
        """The read-stat write-back a genuine recall hit performs on a LIVE MTM point (AD-259).

        The exact twin of :meth:`StmTierRepository.reinforce`, for the tier the user is actually
        still using. ADR 0061 gave the demoted STM copy a real trigger; this gives the same
        trigger to the memory that has not been demoted YET — which is the half
        ``memory-layer §6.2`` and ``recall-service-design.md`` §5.1 have always described as the
        forgetting curve's "remembering" side, and which §5.1 cited ``mtm_qdrant.py:388`` for.
        MEASURED: that file contained the string ``access_count`` **zero** times, and after ADR
        0061 the only adapters reinforcing anything were the three STM ones — so every MTM
        memory demoted on the identical schedule whether the user recalled it a hundred times or
        never. ``SalienceStrategy``'s usage term (``w_usage=0.2``, ``usage_cap=10``) was the
        difference between S=0.2725 (DEMOTE) and S=0.4725 (KEEP) at importance 0.70/age 72h, and
        nothing could ever move it.

        Increments ``access_count`` by 1 and refreshes ``updated_at`` to ``at``. MUST leave
        ``created_at`` untouched (``SalienceStrategy._recency``'s input — this is a read-STAT
        write-back, not a re-capture) and MUST NOT touch the stored vector, the ``state``, or any
        bi-temporal field: it is a payload-only PATCH, the same shape :meth:`expire` /
        :meth:`set_pinned` already use.

        ``None`` if ``memory_id`` is absent from ``ns``'s partition — a no-op, never a raise, for
        the same reason :meth:`StmTierRepository.reinforce` gives: the caller passes only ids its
        own prior read just returned. Best-effort by contract — a caller MUST NOT let a failure
        here fail the read it is reinforcing.

        **AD-266 fix — ``relevance_score`` + ``last_seen``, same contract as
        :meth:`StmTierRepository.reinforce`** (that method's docstring has the full rationale):
        ``relevance_score=None`` leaves the stored value untouched; a real score is written back
        in the same payload PATCH as ``access_count``/``updated_at``/``last_seen`` — closing the
        gap AD-266 found (``mtm_qdrant.py:375-399`` wrote it, ``qdrant_mtm.py:538-545`` did not)."""
        ...

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

    async def reinforce(
        self,
        ns: Namespace,
        memory_id: str,
        *,
        at: datetime,
        relevance_score: float | None = None,
    ) -> MemoryItem | None:
        """The LTM twin of :meth:`MtmTierRepository.reinforce` (ADR 0062 / AD-259 named this the
        one still-missing half: ``recall-service-design.md`` §5.1 describes a ``COLD -> ACTIVE``
        reactivate-on-recall edge (spec §9) built on "the existing access_count/last_seen
        write-back" — but until this method existed, ``LtmTierRepository`` had no ``reinforce``
        at all, so ``RetentionService`` had nothing to read and the edge had no trigger. NAMED,
        not fixed, is what ADR 0062 left this as; this closes it.

        Same contract as :meth:`MtmTierRepository.reinforce`: increments ``access_count`` by 1
        and refreshes ``updated_at`` to ``at``. MUST leave ``created_at``, the stored
        ``content``/``subject``/``predicate``/``object`` triple, ``state``, ``cold`` and every
        bi-temporal field untouched by this call itself — it is a read-STAT write-back, not a
        re-capture and not the COLD flip (see below). ``None`` if ``memory_id`` is absent from
        ``ns``'s graph partition — a no-op, never a raise, for the same reason
        :meth:`StmTierRepository.reinforce` gives: the caller passes only ids its own prior read
        just returned. Best-effort by contract — a caller MUST NOT let a failure here fail the
        read it is reinforcing.

        **Reactivation is deliberately NOT folded into this write.** ``reinforce`` only bumps the
        stat; :meth:`~mu_engine.lifecycle.retention.RetentionService.reactivate_on_recall` is the
        method that actually flips a COLD fact back to non-COLD (``upsert_fact``-based, gated on
        ``LifecycleSettings.retention.reactivate_on_recall``), and it already existed — unwired —
        before this method did. ``RetentionService._sweep`` reads the freshly-bumped
        ``updated_at`` on its NEXT pass over a COLD fact (the same "no longer long-inactive"
        test the cold-slide itself uses, mirrored) and calls that method — exactly the
        recall-writes-a-stat / lifecycle-sweep-reads-it split every other tier's reinforcement
        already follows (STM's write-back rescues a DEMOTED item only on the NEXT
        ``promote_stm_mtm`` sweep, never inline; MTM's rescues only on the NEXT
        ``scan_for_demotion`` sweep). Keeping this port method a pure stat bump — no
        ``LifecycleSettings`` import here, no ``cold`` mutation here — keeps the storage layer
        from depending downward on the lifecycle layer that owns the policy over it.

        **AD-266 fix — ``relevance_score`` + ``last_seen``**, same optional-and-additive contract
        as :meth:`MtmTierRepository.reinforce`: ``None`` leaves ``relevance_score`` untouched;
        ``last_seen`` is always stamped to ``at`` alongside ``updated_at``."""
        ...

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
        seed_entity_uids: Sequence[str] | None = None,
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
        bypass, not an optimization.

        ``seed_entity_uids`` (AD-258, content-aware seed) ADDS a second, non-lexical seed source
        to the frontier's FIRST hop, alongside the casefolded-token match against
        ``canonical_name`` this arm has always done — it never replaces the token match (a query
        that does name its entity in plain text keeps matching exactly as before). The caller
        (``ThreeChannelRecallRanker._ltm_channel``) resolves this list from the query's OWN
        top-ranked MTM dense-vector hits' ``entity_uids`` payload (the entities the MTM channel's
        fact-embedding — ``"{subject} {predicate} {object}"``, ``pipelines/concrete/ingest.py``'s
        own docstring — already resolved as semantically relevant to the query), so an entity the
        query never names verbatim (a paraphrase, a pronoun, a follow-up question) can still seed
        the frontier — the exact gap named in ``docs/tracking/FAULT-HUNT-0924.md``/ADR 0060's own
        "entity-resolved seeding behind ``resolve_entity``" future-work note. ``None`` or empty
        reproduces the pre-fix, token-only seed exactly (A/B comparison, DEV-STANDARDS rule 3)."""
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

    async def delete(self, ns: Namespace, artifact_id: str) -> bool:
        """Remove the metadata handle for ``artifact_id`` (FAULT-HUNT-0924.md F4b: this port
        shipped ``put``/``get``/``get_blob`` and no delete path at all — "no port to add one
        to"). Returns ``True`` when a handle was found and removed, ``False`` when it was already
        absent (idempotent — a retried delete is a no-op, never an error).

        **Reference-aware, per ``ContextArtifact.retention`` (``mu_contracts.domain.model.
        artifact.Retention``, ``RetentionPolicy.REFERENCE_COUNTED``):** the underlying
        content-addressed BLOB may still be the SAME bytes another live ``ContextArtifact`` (a
        distinct id, identical ``content_hash`` — e.g. two captures of identical text) points at,
        so an implementation deletes the blob only when no other handle in this namespace still
        references its ``content_hash``. The caller is responsible for the ONE ref-count question
        this port cannot answer on its own — whether any ``MemoryItem.artifact_ref`` across the
        tiers still points at THIS ``artifact_id`` (``MemoryTierRepository.by_artifact`` is that
        authority, ``artifact.py``'s own ``Retention`` docstring: "authority = by_artifact()").
        Calling this before that check is deleting a still-referenced artifact's handle."""
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
