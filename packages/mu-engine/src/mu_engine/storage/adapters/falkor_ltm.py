"""FalkorDB graph adapter — the LTM tier repository (``storage-pluggable §3.4``).

Graph is MANDATORY (owner override 1): there is no ``sqlfold``/``none`` LTM mode.

PORT of the ``:Memory`` MERGE, ``SUPERSEDED_BY``/``CONFLICTS_WITH`` lifecycle, and the
``find_conflicts``/``facts_at``/``resolve_entity`` logic from
``/home/user/hackathon/memory_universe/shared/stores/graph_falkor.py:98,131,908-952,259``,
re-implemented as openCypher over the FalkorDB async client.

DELIBERATE DEVIATION (recorded, CODE-ADOPTION rule 4): ``storage-pluggable §2.2`` pins the
openCypher **query dialect** + Graphiti's ``GraphDriver`` as the engine seam. This adapter
emits openCypher directly through the lightweight ``falkordb`` async client rather than
vendoring ``graphiti-core`` — the query dialect (the portable part) is identical; only the
driver object differs. ``graphiti-core`` is NOT installed at this phase to keep the
dependency/RAM footprint down on the shared box; the ``GraphDriver`` swap is a constructor
change (the adapter already isolates all Cypher here). The bi-temporal semantics
(``valid_at``/``invalid_at`` invalidate-don't-delete) match Graphiti's model exactly.

Mandatory read filters on every traversal (spec §4.3): ``m.namespace = $ns`` (exact) AND
``m.state = 'active'`` AND ``(m.invalid_at = '' OR m.invalid_at > $now)``; SHARED adds the
Model-A ``ANY(id IN $caller ...)`` predicate.

OWNER DECISIONS (2026-07-27, AUTHORITATIVE):
1. Graph = FalkorDB, KEPT. This adapter is OUR OWN — it ports Graphiti's bi-temporal
   *pattern* (invalidate-don't-delete, ``valid_at``/``invalid_at``), it does NOT import
   ``graphiti_core``. It sits behind :class:`~mu_engine.storage.ports.GraphStorePort` +
   ``StoreRegistry`` exactly like the vector/kv/relational roles (registered in
   ``mu_engine.storage.factories`` under ``(graph, "falkordb")`` — see that module's
   docstring for the EXTENSION SEAM a future ``neo4j``/``kuzu``/``ladybug``/``neptune``
   driver would register under, with ZERO change to this file or to the engine).
2. MULTI-ORG HARDEN — the physical FalkorDB graph-partition name is keyed on
   ``(org, workspace, visibility|user)``, NOT workspace alone (see :meth:`graph_name_for`).
   Two orgs sharing a workspace id therefore land in DIFFERENT physical graphs; FalkorDB's
   native multi-graph-per-tenant design (``select_graph``) is preserved — never collapsed to
   one shared graph. The ``m.namespace = $ns`` (org-scoped ``to_prefix()``) property filter on
   every query stays as mandatory defense-in-depth ON TOP of that physical partition, not
   instead of it.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import structlog
from falkordb.asyncio import FalkorDB
from qdrant_client.http.exceptions import UnexpectedResponse

from mu_contracts.ports.time import Clock
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.entity import EntityCandidate, EntityResolution
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored
from mu_engine.storage.mappers.graph_mapper import GraphMapper

__all__ = ["EntityUidsSink", "FalkorLtmAdapter"]

_log = structlog.get_logger("mu_engine.storage.adapters.falkor_ltm")


@runtime_checkable
class EntityUidsSink(Protocol):
    """D-5 (ARCHITECTURE-CONFORMANCE.md; CONFIG-AND-DATA-FIX-PLAN.md PART 2 D6): the ONE narrow
    method ``FalkorLtmAdapter`` needs from the MTM tier to backfill ``entity_uids`` onto an
    already-promoted point once this fact's subject/object entities are resolved at LTM upsert
    time — deliberately NOT added to the shared ``MtmTierRepository`` Protocol (``ports.py``):
    every OTHER vector backend (pgvector/chroma/faiss) would then need a matching
    implementation for a capability only ``QdrantMtmAdapter`` implements today. A structural
    (``runtime_checkable``) Protocol keeps this strictly additive/optional — a composition root
    that never wires ``mtm_entity_sink=`` (or wires a non-Qdrant vector backend) leaves this
    whole code path a no-op, byte-identical to before this feature existed.
    """

    async def set_entity_uids(
        self, ns: Namespace, memory_id: str, entity_uids: list[str]
    ) -> None: ...


def _sanitize_rel_type(predicate: str) -> str:
    """A canonicalized predicate (D5, ``services/extract.py::_canonicalize_predicate``) -> a
    SAFE openCypher relationship-type token. Cypher has no parameterized relationship type
    (unlike node labels/props) — the sanitized token is interpolated directly into the query
    STRING below, so this allowlists to ``[A-Za-z0-9_]`` only (never passes predicate text
    through verbatim) and uppercases for the conventional Cypher rel-type casing
    (``manages`` -> ``MANAGES``). An empty/all-punctuation predicate (should not happen post-D5
    canonicalization, but never trust that blindly) falls back to the named ``RELATED_TO``
    catch-all rather than emitting an invalid/empty rel-type token."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", predicate.strip()).strip("_")
    cleaned = cleaned.upper()
    if not cleaned:
        return "RELATED_TO"
    if cleaned[0].isdigit():
        cleaned = f"REL_{cleaned}"
    return cleaned


# Constructor DEFAULTS only (DEV-STANDARDS rule 3: no hardcoded constant lives in adapter
# LOGIC). The live values are DI-threaded in from the central Settings tree
# (``mu_contracts.config.FalkorDBSettings``) by the ``STORE_REGISTRY`` factory
# (``mu_engine.storage.factories._build_falkordb``); a bare ``FalkorLtmAdapter(db)`` (e.g. in
# a unit test) still gets a sane, named default rather than a silent unconfigured 0/None.
_DEFAULT_SHORTLIST_SIZE = 5
_DEFAULT_SIMILARITY_THRESHOLD = 0.84  # graph_falkor.py resolve_entity deterministic band
# Per-attempt I/O budget (DEV-STANDARDS async sharpener: "timeouts on every external call").
_DEFAULT_STORE_IO_TIMEOUT_S = 10.0


class FalkorLtmAdapter:
    """Implements ``GraphStorePort``/``LtmTierRepository`` over a real FalkorDB connection.

    ALL domain time flows through the injected :class:`Clock` port in UTC (MAJOR-3): the
    bi-temporal ``invalid_at > $now`` graph filter must be UTC-correct — a local-tz ``now``
    (host is +03:00) would mis-compare against the UTC ISO strings stored on the nodes.
    Every external openCypher call is wrapped by :func:`retry_io` (transient-only retry/backoff
    + a per-attempt timeout) so no store call is unbounded (MAJOR-4). The retry timeout, the
    ``resolve_entity`` shortlist size, and its similarity band are constructor parameters
    (DI-threaded from Settings by the factory), never module-level literals baked into the
    decorated method — this lets ONE instance be tuned without touching the class.
    """

    def __init__(  # type: ignore[no-any-unimported]  # falkordb ships no stubs
        self,
        db: FalkorDB,
        *,
        mapper: GraphMapper | None = None,
        clock: Clock | None = None,
        shortlist_size: int = _DEFAULT_SHORTLIST_SIZE,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        store_io_timeout_s: float = _DEFAULT_STORE_IO_TIMEOUT_S,
        mtm_entity_sink: EntityUidsSink | None = None,
    ) -> None:
        self._db = db
        self._mapper = mapper or GraphMapper()
        self._clock: Clock = clock or SystemClock()
        self._shortlist_size = shortlist_size
        self._similarity_threshold = similarity_threshold
        # a per-instance retry wrapper (not a class-level decorator) so `store_io_timeout_s`
        # is genuinely DI-threaded per instance, not fixed at import time.
        self._retry = retry_io(timeout_s=store_io_timeout_s)
        # D-5 (ARCHITECTURE-CONFORMANCE.md, entity_uids MTM payload): optional structural seam
        # (see :class:`EntityUidsSink`) — `None` (the default) is a no-op, byte-identical to
        # before this feature existed; a composition root wires the live ``QdrantMtmAdapter``
        # here via :meth:`set_mtm_entity_sink` once both the vector + graph roles are built.
        self._mtm_entity_sink: EntityUidsSink | None = mtm_entity_sink

    def set_mtm_entity_sink(self, sink: EntityUidsSink | None) -> None:
        """Post-construction wiring hook (D-5): the composition root builds the ``vector`` role
        BEFORE the ``graph`` role (``mu_local.composition.LocalContainer`` — ``self.mtm`` then
        ``self.ltm``, in that order), so the sink cannot be a constructor arg supplied by the
        SAME ``STORE_REGISTRY.build("graph", ...)`` call that constructs this adapter — the
        registry factory only sees the ``graph`` role's own config. A duck-typed
        (``EntityUidsSink``-shaped) already-built vector adapter is handed in here instead, right
        after both roles exist. Never required: a composition root that skips this call, or
        wires a non-Qdrant vector backend, leaves ``upsert_fact``'s entity_uids write a no-op.
        """
        self._mtm_entity_sink = sink

    def graph_name_for(self, ns: Namespace) -> str:
        """The physical FalkorDB graph a namespace resolves to — pure, no I/O.

        Partition key = ``(org, workspace, visibility|user)`` (multi-org harden, owner
        directive 2026-07-27): ``mu_g__{org}__{workspace}__shared`` for the SHARED plane,
        ``mu_g__{org}__{workspace}__{user}`` for PRIVATE. Public so tests/observability can
        assert on — and print — the exact physical partition a namespace lands in without
        duplicating this computation (e.g. proving two orgs sharing a workspace id land in
        two distinct graphs).
        """
        if ns.visibility is Visibility.SHARED:
            return f"mu_g__{ns.org}__{ns.workspace}__shared"
        return f"mu_g__{ns.org}__{ns.workspace}__{ns.user}"

    def _graph(self, ns: Namespace) -> Any:
        return self._db.select_graph(self.graph_name_for(ns))

    async def upsert_fact(self, item: MemoryItem) -> None:
        return await self._retry(self._upsert_fact_impl)(item)

    async def _upsert_fact_impl(self, item: MemoryItem) -> None:
        g = self._graph(item.namespace)
        row = self._mapper.to_store(item)
        props = row.props
        set_authorized = ", m.authorized_ids = $authorized_ids" if "authorized_ids" in props else ""
        cypher = (
            "MERGE (m:Memory {namespace: $namespace, id: $id}) "
            "SET m.subject = $subject, m.predicate = $predicate, m.object = $object, "
            "m.object_kind = $object_kind, m.polarity = $polarity, m.state = $state, "
            "m.valid_at = $valid_at, m.invalid_at = $invalid_at, "
            "m.content_hash = $content_hash, m.artifact_ref = $artifact_ref, "
            "m.provenance_id = $provenance_id, m.content = $content, m.memory_json = $memory_json"
            f"{set_authorized}"
        )
        await g.query(cypher, params=dict(props))
        if item.artifact_ref:
            await g.query(
                "MATCH (m:Memory {namespace: $ns, id: $id}) "
                "MERGE (a:Artifact {namespace: $ns, id: $art}) "
                "MERGE (m)-[:REFERENCES]->(a)",
                params={"ns": props["namespace"], "id": item.id, "art": item.artifact_ref},
            )
        await self._materialize_entity_edge(g, item, props)

    async def _materialize_entity_edge(
        self, g: Any, item: MemoryItem, props: dict[str, Any]
    ) -> None:
        """B5/B6 (ARCHITECTURE-CONFORMANCE.md Lens B): MERGE a subject/object ``:Entity`` pair +
        a canonical-predicate-typed edge between them, alongside the ``:Memory`` node above —
        the actual traversable-KG materialization ``resolve_entity`` (below) has always been
        able to READ but nothing ever WROTE (0 ``:Entity`` nodes, 0 entity-entity edges pre-fix).

        SCOPE DECISION (recorded, not gated on ``item.object_kind``): every extracted fact's
        ``object_kind`` defaults to ``FactObjectKind.LITERAL`` today (no code path in
        ``services/extract.py`` ever sets ``ENTITY``) — gating this on
        ``object_kind is ENTITY`` would make the feature permanently dead code given the
        current extractor. Both subject and object are materialized as ``:Entity`` nodes
        unconditionally (a literal like "Thursday" becomes a lightweight entity node too) so
        the graph is genuinely traversable NOW; ``object_kind`` itself is untouched on the
        ``:Memory`` node (still LITERAL/ENTITY as extracted) — this is a graph-projection
        decision only, not a change to the domain model's own literal/entity distinction.

        Skipped when subject/predicate/object aren't all present (unstructured/partial items
        never reach here with a real triple). Entity identity = ``(namespace, canonical_name)``
        where ``canonical_name`` is the SAME ``.strip().casefold()`` key ``resolve_entity``
        already matches on (case-insensitive, D3 casing-fix precedent) — so a name resolved
        here is immediately visible to the next ``resolve_entity``/``_merge_entity`` call.
        """
        if not (item.subject and item.predicate and item.object):
            return
        subj_uid = await self._merge_entity(item.namespace, g, item.subject)
        obj_uid = await self._merge_entity(item.namespace, g, item.object)
        if not subj_uid or not obj_uid:
            return
        rel_type = _sanitize_rel_type(item.predicate)
        ns_prefix = props["namespace"]
        # MERGE keyed on (subj entity, obj entity, rel TYPE) — idempotent re-upsert of the SAME
        # fact refreshes this SAME edge's timestamps/memory_id; a functional-supersession
        # (different object -> different `obj_uid`) lands on a DIFFERENT edge instead of
        # colliding with the one it supersedes (see `_invalidate_entity_edge` for the loser-side
        # bi-temporal close). Relationship TYPE cannot be parameterized in openCypher — `rel_type`
        # is sanitized to `[A-Za-z0-9_]` by `_sanitize_rel_type` before this interpolation, never
        # raw predicate text (Cypher-injection floor).
        edge_cypher = (
            f"MATCH (s:Entity {{namespace: $ns, entity_uid: $subj_uid}}), "
            f"(o:Entity {{namespace: $ns, entity_uid: $obj_uid}}) "
            f"MERGE (s)-[r:{rel_type}]->(o) "
            "SET r.memory_id = $mid, r.predicate = $predicate, "
            "r.valid_at = $valid_at, r.invalid_at = $invalid_at"
        )
        await g.query(
            edge_cypher,
            params={
                "ns": ns_prefix,
                "subj_uid": subj_uid,
                "obj_uid": obj_uid,
                "mid": item.id,
                "predicate": item.predicate,
                "valid_at": props["valid_at"],
                "invalid_at": props["invalid_at"],
            },
        )
        # Mutable in-place update (``MemoryItem.model_config = ConfigDict(frozen=False)``) — the
        # SAME object reference the caller (e.g. ``DistillPipeline._resolve``) holds, so this is
        # visible to the caller with no port-signature/return-type change to `upsert_fact`.
        item.metadata = {**item.metadata, "entity_uids": [subj_uid, obj_uid]}
        await self._backfill_mtm_entity_uids(item, [subj_uid, obj_uid])

    async def _merge_entity(self, ns: Namespace, g: Any, name: str) -> str:
        """Resolve-or-create an ``:Entity`` node for ``name`` — reuses ``resolve_entity``'s OWN
        deterministic exact/alias-match query (+ its ``similarity_threshold``/``shortlist_size``
        config) as instructed, never a second, divergent resolution mechanism: a resolved hit's
        ``entity_uid`` is reused verbatim; an unresolved name mints a fresh ``entity_uid`` and
        MERGEs a new node keyed on the SAME ``(namespace, canonical_name)`` identity
        ``resolve_entity`` matches on. Calls the PRIVATE ``_resolve_entity_impl`` directly (not
        the public, ``retry_io``-wrapped ``resolve_entity``) to avoid double-wrapping retries —
        this method already runs inside `_upsert_fact_impl`'s own single retry envelope."""
        name = name.strip()
        if not name:
            return ""
        resolution = await self._resolve_entity_impl(ns, name)
        if resolution.entity_uid is not None:
            return resolution.entity_uid
        canonical = resolution.canonical_name  # already .strip().casefold()'d
        uid = f"ent_{uuid4().hex}"
        cypher = (
            "MERGE (e:Entity {namespace: $ns, canonical_name: $canon}) "
            "ON CREATE SET e.entity_uid = $uid, e.aliases = [$raw], e.alias_keys = [$canon] "
            "ON MATCH SET e.aliases = "
            "CASE WHEN $raw IN e.aliases THEN e.aliases ELSE e.aliases + $raw END "
            "RETURN e.entity_uid AS uid"
        )
        rows = await g_query(
            g, cypher, {"ns": ns.to_prefix(), "canon": canonical, "uid": uid, "raw": name}
        )
        return str(rows[0][0]) if rows else uid

    async def _backfill_mtm_entity_uids(self, item: MemoryItem, entity_uids: list[str]) -> None:
        """D-5: push ``entity_uids`` onto the already-promoted MTM (Qdrant) point through the
        optional :class:`EntityUidsSink` seam. The MTM point is keyed by the INGEST id
        (CANONICAL §7.1 id-stability) — a STRUCTURED fact keeps its tier-stable id STM->MTM->LTM
        (LTM node id == MTM point id); an EXTRACTED fact mints a fresh LTM node id and records
        its source ingest id in ``metadata['derived_from']`` (mirrors
        ``DistillPipeline._mtm_point_id`` verbatim — same rule, independently applied here since
        this adapter has no reference to that pipeline). A no-op when no sink is wired."""
        if self._mtm_entity_sink is None:
            return
        derived = item.metadata.get("derived_from")
        mtm_point_memory_id = str(derived) if derived else item.id
        try:
            await self._mtm_entity_sink.set_entity_uids(
                item.namespace, mtm_point_memory_id, entity_uids
            )
        except UnexpectedResponse:
            # write-after-read visibility lag (same NAMED degrade precedent as
            # `DistillPipeline._invalidate_mtm_guarded`) — the point isn't visible in Qdrant yet.
            # Best-effort forward-compat backfill: never fails the LTM write itself, no retry
            # queue (unlike the invalidate guard) since a MISSED entity_uids backfill is a
            # degraded-but-safe outcome (the fact + entity edge are still correct in the graph).
            _log.warning(
                "mtm_entity_uids_backfill_point_absent",
                ns=item.namespace.to_prefix(),
                memory_id=mtm_point_memory_id,
            )

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        return await self._retry(self._graph_recall_impl)(
            ns,
            subject=subject,
            predicate=predicate,
            limit=limit,
            caller_identity_set=caller_identity_set,
        )

    async def _graph_recall_impl(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        prefix = ns.to_prefix()
        where = [
            "m.namespace = $ns",
            "m.state = $active",
            "(m.invalid_at = '' OR m.invalid_at > $now)",
        ]
        params: dict[str, Any] = {
            "ns": prefix,
            "active": MemoryState.ACTIVE.value,
            "now": self._clock.now().isoformat(),  # injected Clock, UTC (MAJOR-3)
            "limit": limit,
        }
        if subject is not None:
            where.append("m.subject = $subject")
            params["subject"] = subject
        if predicate is not None:
            where.append("m.predicate = $predicate")
            params["predicate"] = predicate
        if ns.visibility is Visibility.SHARED and caller_identity_set is not None:
            where.append("ANY(cid IN $caller WHERE cid IN m.authorized_ids)")
            params["caller"] = list(caller_identity_set)
        cypher = (
            f"MATCH (m:Memory) WHERE {' AND '.join(where)} "
            "RETURN m.memory_json AS mj ORDER BY m.valid_at DESC LIMIT $limit"
        )
        res = await g_query(self._graph(ns), cypher, params)
        out: list[Scored[MemoryItem]] = []
        for rank, record in enumerate(res):
            item = MemoryItem.model_validate_json(record[0])
            out.append(
                Scored(
                    item=item, score=1.0 / (rank + 1), channel=RecallChannel.LTM_GRAPH, rank=rank
                )
            )
        return out

    async def facts_at(
        self, ns: Namespace, at: datetime, *, subject: str | None = None
    ) -> list[MemoryItem]:
        return await self._retry(self._facts_at_impl)(ns, at, subject=subject)

    async def _facts_at_impl(
        self, ns: Namespace, at: datetime, *, subject: str | None = None
    ) -> list[MemoryItem]:
        # bi-temporal as-of read: valid_at <= at AND (invalid_at empty OR invalid_at > at).
        at_iso = at.isoformat()
        where = [
            "m.namespace = $ns",
            "m.valid_at <> ''",
            "m.valid_at <= $at",
            "(m.invalid_at = '' OR m.invalid_at > $at)",
        ]
        params: dict[str, Any] = {"ns": ns.to_prefix(), "at": at_iso}
        if subject is not None:
            where.append("m.subject = $subject")
            params["subject"] = subject
        cypher = f"MATCH (m:Memory) WHERE {' AND '.join(where)} RETURN m.memory_json AS mj"
        res = await g_query(self._graph(ns), cypher, params)
        return [MemoryItem.model_validate_json(r[0]) for r in res]

    async def find_conflicts(self, ns: Namespace, subject: str, predicate: str) -> list[MemoryItem]:
        return await self._retry(self._find_conflicts_impl)(ns, subject, predicate)

    async def _find_conflicts_impl(
        self, ns: Namespace, subject: str, predicate: str
    ) -> list[MemoryItem]:
        # active facts sharing (subject, predicate) — the conflict access pattern
        # (graph_falkor.py find_conflicts; spec §2.5 ix_fact_conflict mirror).
        #
        # DEFECT-1 FIX (real-path verify gate, live-reproduced): the collision match is
        # case-INSENSITIVE on subject (`toLower(m.subject) = toLower($subject)`), matching
        # `resolve_entity`'s own `name.strip().casefold()` precedent two methods below — a real
        # SLM extraction of the SAME entity across two turns is NOT guaranteed to return the same
        # surface casing (observed: "ada"/"tuesday" on v1, "Ada"/"Thursday" on v2 for the SAME
        # flight fact), and an exact-string `m.subject = $subject` silently drops the collision,
        # so the v2 fact is treated as a brand-new (subject, predicate) pair (ADD) instead of a
        # functional-supersession candidate — the exact defect this fixes. `predicate` is left
        # exact: predicate canonicalization is `extract.py`'s job (D-7's canonical_predicate_map)
        # and is out of scope here — this is subject-collision only (full entity normalization is
        # D6/later, per the remediation scope). `toLower()` on both sides (not just one) so an
        # already-lowercased stored subject still matches a differently-cased query subject and
        # vice versa; leaves entity identity (the stored `m.subject` value itself, used by
        # `graph_recall`/`facts_at` and returned in `memory_json`) untouched — this only widens
        # the MATCH predicate for collision detection, it never rewrites what's stored.
        cypher = (
            "MATCH (m:Memory) WHERE m.namespace = $ns "
            "AND toLower(m.subject) = toLower($subject) "
            "AND m.predicate = $predicate AND m.state = $active "
            "AND (m.invalid_at = '' OR m.invalid_at > $now) RETURN m.memory_json AS mj"
        )
        params = {
            "ns": ns.to_prefix(),
            "subject": subject,
            "predicate": predicate,
            "active": MemoryState.ACTIVE.value,
            "now": self._clock.now().isoformat(),  # injected Clock, UTC (MAJOR-3)
        }
        res = await g_query(self._graph(ns), cypher, params)
        return [MemoryItem.model_validate_json(r[0]) for r in res]

    async def invalidate(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        return await self._retry(self._invalidate_impl)(
            ns, loser_id, winner_id, at=at, reason=reason
        )

    async def _invalidate_impl(
        self, ns: Namespace, loser_id: str, winner_id: str, *, at: datetime, reason: str
    ) -> None:
        # invalidate-don't-delete (spec §4.2 lifecycle): stamp loser state='superseded' +
        # invalid_at, MERGE (old)-[:SUPERSEDED_BY]->(new) + (old)-[:CONFLICTS_WITH]->(new).
        g = self._graph(ns)
        at_iso = at.isoformat()
        await g.query(
            "MATCH (loser:Memory {namespace: $ns, id: $loser}) "
            "SET loser.state = $superseded, loser.invalid_at = $at "
            "WITH loser MATCH (winner:Memory {namespace: $ns, id: $winner}) "
            "MERGE (loser)-[s:SUPERSEDED_BY]->(winner) SET s.created_at = $at, s.reason = $reason "
            "MERGE (loser)-[c:CONFLICTS_WITH]->(winner) SET c.created_at = $at",
            params={
                "ns": ns.to_prefix(),
                "loser": loser_id,
                "winner": winner_id,
                "superseded": MemoryState.SUPERSEDED.value,
                "at": at_iso,
                "reason": reason,
            },
        )
        await self._invalidate_entity_edge(g, ns, loser_id, at_iso)

    async def _invalidate_entity_edge(
        self, g: Any, ns: Namespace, loser_id: str, at_iso: str
    ) -> None:
        """B5/B6 bi-temporal parity: close the LOSER's own entity-entity edge (if any) alongside
        the ``:Memory`` node's own ``invalid_at`` stamp above, so a multi-hop traversal
        (:meth:`traverse_entities`) never returns a superseded relation as current.

        Guarded by ``r.memory_id = $loser_id`` (no relationship-type filter needed — the edge
        carries its OWN originating fact id regardless of predicate) rather than re-deriving the
        loser's subject/predicate/object and re-resolving its entities: `_upsert_fact_impl`
        ALWAYS writes the winner to the graph BEFORE `DistillPipeline._resolve` calls
        `invalidate()` (both call sites) — when winner and loser share the exact same
        (subject, predicate, object) entity-edge triple (a same-object polarity contradiction,
        e.g. "Ada likes coffee" vs "Ada doesn't like coffee"), the winner's own upsert has
        ALREADY overwritten that shared edge's `memory_id` to `winner_id` by the time this runs —
        the `r.memory_id = $loser_id` guard means this MATCH then finds nothing and is a correct
        no-op (never re-invalidating the edge out from under the winner it now represents). A
        functional supersession (different object -> a genuinely DIFFERENT edge) still carries
        `memory_id = loser_id` untouched, so this correctly closes THAT edge.
        """
        await g.query(
            "MATCH (:Entity {namespace: $ns})-[r {memory_id: $loser_id}]->"
            "(:Entity {namespace: $ns}) SET r.invalid_at = $at",
            params={"ns": ns.to_prefix(), "loser_id": loser_id, "at": at_iso},
        )

    async def mark_conflict(self, ns: Namespace, a_id: str, b_id: str, *, at: datetime) -> None:
        return await self._retry(self._mark_conflict_impl)(ns, a_id, b_id, at=at)

    async def _mark_conflict_impl(
        self, ns: Namespace, a_id: str, b_id: str, *, at: datetime
    ) -> None:
        # bare CONFLICTS_WITH edge between two facts that BOTH remain active (D3, spec §8) —
        # no state/invalid_at write to either side, unlike `_invalidate_impl`'s bundled edge.
        g = self._graph(ns)
        await g.query(
            "MATCH (a:Memory {namespace: $ns, id: $a}) "
            "WITH a MATCH (b:Memory {namespace: $ns, id: $b}) "
            "MERGE (a)-[c:CONFLICTS_WITH]->(b) "
            "SET c.created_at = $at, c.reason = $reason",
            params={
                "ns": ns.to_prefix(),
                "a": a_id,
                "b": b_id,
                "at": at.isoformat(),
                "reason": "pending_adjudication",
            },
        )

    async def resolve_entity(self, ns: Namespace, name: str) -> EntityResolution:
        return await self._retry(self._resolve_entity_impl)(ns, name)

    async def _resolve_entity_impl(self, ns: Namespace, name: str) -> EntityResolution:
        # PORT graph_falkor.py resolve_entity: deterministic match OR bounded shortlist.
        canonical = name.strip().casefold()
        cypher = (
            "MATCH (e:Entity) WHERE e.namespace = $ns AND "
            "(e.canonical_name = $canon OR $canon IN e.alias_keys) "
            "RETURN e.entity_uid AS uid, e.canonical_name AS cn, e.aliases AS al LIMIT $k"
        )
        res = await g_query(
            self._graph(ns),
            cypher,
            {"ns": ns.to_prefix(), "canon": canonical, "k": self._shortlist_size},
        )
        candidates = tuple(
            EntityCandidate(
                entity_uid=str(r[0]),
                canonical_name=str(r[1]),
                aliases=tuple(r[2] or ()),
                similarity=1.0,
            )
            for r in res
        )
        if len(candidates) == 1 and candidates[0].similarity >= self._similarity_threshold:
            return EntityResolution(
                canonical_name=candidates[0].canonical_name,
                entity_uid=candidates[0].entity_uid,
                candidates=candidates,
            )
        return EntityResolution(canonical_name=canonical, entity_uid=None, candidates=candidates)

    async def by_artifact(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]:
        return await self._retry(self._by_artifact_impl)(ns, artifact_id)

    async def _by_artifact_impl(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]:
        # Reverse provenance lookup (GraphStorePort.by_artifact, storage/ports.py): traverses
        # FROM the merged (:Artifact) node via the REFERENCES edge `_upsert_fact_impl` already
        # writes whenever `item.artifact_ref` is set (module docstring OWNER DECISION 1's same
        # MERGE-on-write pattern) — a graph traversal off an indexed MERGE key, never a
        # `:Memory`-label scan (mu_contracts.ports.memory.MemoryTierRepository.by_artifact
        # docstring: "never a scan").
        cypher = (
            "MATCH (a:Artifact {namespace: $ns, id: $art})<-[:REFERENCES]-(m:Memory) "
            "RETURN m.memory_json AS mj"
        )
        res = await g_query(self._graph(ns), cypher, {"ns": ns.to_prefix(), "art": artifact_id})
        return [MemoryItem.model_validate_json(r[0]) for r in res]

    async def traverse_entities(
        self, ns: Namespace, *, query: str, max_hops: int, limit: int
    ) -> list[Scored[MemoryItem]]:
        """D-4 (ARCHITECTURE-CONFORMANCE.md "LTM graph arm thin"): the multi-hop traversal arm —
        answers a relational query ("who is Bo's manager?") by seeding on entity NAMES found in
        ``query`` and walking the entity-entity edges :meth:`_materialize_entity_edge` writes,
        UP TO ``max_hops`` (clamped to 1-2: deeper hops risk combinatorial blowup on a shared
        box and are out of this task's scope). Returns the underlying ``:Memory`` fact(s) each
        hop's edge traces back to (via the edge's own ``memory_id`` property), never the bare
        entity nodes — the caller (:class:`~mu_engine.services.recall.ranker.
        ThreeChannelRecallRanker`) fuses these into the SAME ``Scored[MemoryItem]`` shape
        ``graph_recall`` returns.

        Seed matching reuses ``resolve_entity``'s OWN exact/case-insensitive ``canonical_name``
        convention (D3 casing-fix precedent) — never a new fuzzy-similarity mechanism: a query
        token matches an entity iff ``token.casefold() == e.canonical_name`` (which is itself
        already ``.strip().casefold()``'d at write time). A BFS over 1-hop Cypher MATCHes (never
        FalkorDB's variable-length ``-[*1..N]-`` path syntax, whose per-edge property binding is
        not something this adapter risks depending on) so each hop is a plain, well-understood
        query. Bi-temporal: an edge whose ``invalid_at`` has passed (superseded, per
        :meth:`_invalidate_entity_edge`) is excluded from every hop, so a stale relation never
        resurfaces via traversal.
        """
        return await self._retry(self._traverse_entities_impl)(
            ns, query=query, max_hops=max_hops, limit=limit
        )

    async def _traverse_entities_impl(
        self, ns: Namespace, *, query: str, max_hops: int, limit: int
    ) -> list[Scored[MemoryItem]]:
        tokens = {t.casefold() for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 1}
        if not tokens:
            return []
        g = self._graph(ns)
        ns_prefix = ns.to_prefix()
        now_iso = self._clock.now().isoformat()
        hops = max(1, min(max_hops, 2))

        frontier = sorted(tokens)
        seen_entities: set[str] = set(frontier)
        memory_hop: dict[str, int] = {}  # memory_id -> smallest hop distance it was found at
        for depth in range(1, hops + 1):
            if not frontier:
                break
            cypher = (
                "MATCH (seed:Entity {namespace: $ns})-[r]-(other:Entity {namespace: $ns}) "
                "WHERE seed.canonical_name IN $frontier "
                "AND (r.invalid_at = '' OR r.invalid_at > $now) AND r.memory_id IS NOT NULL "
                "RETURN DISTINCT r.memory_id AS mid, other.canonical_name AS ocn"
            )
            rows = await g_query(
                g, cypher, {"ns": ns_prefix, "frontier": frontier, "now": now_iso}
            )
            next_frontier: list[str] = []
            for mid, ocn in rows:
                mid_s = str(mid)
                if mid_s not in memory_hop:
                    memory_hop[mid_s] = depth
                ocn_s = str(ocn)
                if ocn_s not in seen_entities:
                    seen_entities.add(ocn_s)
                    next_frontier.append(ocn_s)
            frontier = next_frontier

        if not memory_hop:
            return []
        fetch_cypher = (
            "MATCH (m:Memory {namespace: $ns}) WHERE m.id IN $ids "
            "AND m.state = $active AND (m.invalid_at = '' OR m.invalid_at > $now) "
            "RETURN m.id AS id, m.memory_json AS mj"
        )
        rows = await g_query(
            g,
            fetch_cypher,
            {
                "ns": ns_prefix,
                "ids": list(memory_hop),
                "active": MemoryState.ACTIVE.value,
                "now": now_iso,
            },
        )
        scored: list[Scored[MemoryItem]] = []
        for mid, mj in rows:
            hop = memory_hop.get(str(mid), hops)
            item = MemoryItem.model_validate_json(mj)
            scored.append(
                Scored(item=item, score=1.0 / hop, channel=RecallChannel.LTM_GRAPH, rank=0)
            )
        scored.sort(key=lambda s: -s.score)
        return [
            s.model_copy(update={"rank": rank}) for rank, s in enumerate(scored[:limit])
        ]


async def g_query(graph: Any, cypher: str, params: dict[str, Any]) -> list[list[Any]]:
    """Run an openCypher read and return the raw result rows (result_set)."""
    result = await graph.query(cypher, params=params)
    return list(result.result_set or [])
