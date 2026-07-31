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

from datetime import datetime
from typing import Any

from falkordb.asyncio import FalkorDB

from mu_contracts.ports.time import Clock
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.decorators import retry_io
from mu_engine.storage.domain.entity import EntityCandidate, EntityResolution
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored
from mu_engine.storage.mappers.graph_mapper import GraphMapper

__all__ = ["FalkorLtmAdapter"]

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
    ) -> None:
        self._db = db
        self._mapper = mapper or GraphMapper()
        self._clock: Clock = clock or SystemClock()
        self._shortlist_size = shortlist_size
        self._similarity_threshold = similarity_threshold
        # a per-instance retry wrapper (not a class-level decorator) so `store_io_timeout_s`
        # is genuinely DI-threaded per instance, not fixed at import time.
        self._retry = retry_io(timeout_s=store_io_timeout_s)

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


async def g_query(graph: Any, cypher: str, params: dict[str, Any]) -> list[list[Any]]:
    """Run an openCypher read and return the raw result rows (result_set)."""
    result = await graph.query(cypher, params=params)
    return list(result.result_set or [])
