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
3. D-8 FIX (2026-08-27): ``graph_name_for`` originally built its name by RAW string join
   (``mu_g__{org}__{workspace}__...``). ``_`` is not in ``Namespace._FORBIDDEN_NS_CHARS``, so
   ``org="acme__eu", workspace="ws"`` and ``org="acme", workspace="eu__ws"`` produced the
   IDENTICAL graph name — the same raw-join ambiguity already fixed in the four MTM vector
   mappers (``mu_engine.storage.mappers.tenancy.tenant_partition_digest``), independently
   present here because this adapter built its own name rather than reusing that helper. This
   OUTRANKS the vector-tier version of the same defect: CANONICAL §7.4 authorizes the PRIVATE
   graph-recall arm by physical partition ALONE (no payload-filter backstop), so a name
   collision here is unmediated — the property filter that saves the vector tier from a
   collision does not exist for this tier's PRIVATE reads. Fixed by reusing
   ``tenant_partition_digest`` (``org``+``workspace`` hashed jointly on an unambiguous
   ``"org:workspace"`` join, since ``":"`` IS forbidden in both) — see that function's
   docstring for the full derivation. The per-user PRIVATE segment stays a readable suffix
   (see :meth:`graph_name_for`'s own docstring) — CANONICAL §1 rule 6 as CLARIFIED 2026-08-26
   holds a per-user graph to be rule 5's partition applied, not a new physical store, so
   per-user separation is conformant and this fix PRESERVES it rather than folding it into the
   digest. NOT a data migration: existing graphs under the old (collidable) names are orphaned
   by this change — recorded as an open gap, not remediated here.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
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
from mu_engine.storage.errors import MtmPointAbsentError
from mu_engine.storage.mappers.graph_mapper import GraphMapper
from mu_engine.storage.mappers.tenancy import tenant_partition_digest
from mu_engine.storage.tier_capabilities import with_pin_group

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


def _resolve_memory_namespace_filter(
    ns: Namespace, *, session_scope: str | None
) -> tuple[str, str]:
    """The ``(cypher_predicate, param_value)`` pair the ``:Memory`` reads filter on — the GRAPH
    twin of ``qdrant_mtm._resolve_namespace_match``, and deliberately the same three-way rule.

    FEDERATE-LIVE GAP FIX (live-reproduced): ``QdrantMtmAdapter`` already honoured
    ``session_scope=None`` by relaxing to the session-less USER prefix (ADR 0030 "keep-and-scope"
    — federate every one of the user's sessions, the DEFAULT), but ``graph_recall``/``facts_at``
    took no ``session_scope`` at all and filtered ``m.namespace = $ns`` on the FULL, session-
    included ``to_prefix()``. So federate-live recall was split-brained: the MTM arm went
    cross-session while the LTM arm — the durable, user-scoped tier that is supposed to BE the
    long-term memory — stayed locked to whichever session asked. The physical partition
    (``graph_name_for``: org/workspace/visibility/user, no session) already held every session's
    facts side by side; only this read filter hid them. The same reasoning ``_user_scope_prefix``
    below already applies to the ``:Entity`` sub-graph, never extended to ``:Memory`` until now.

    * SHARED -> exact ``to_prefix()``, UNCONDITIONALLY (rooms are real walls; ``session_scope``
      never relaxes SHARED — identical to the Qdrant rule).
    * PRIVATE, ``session_scope is None`` -> ``STARTS WITH`` the user prefix + ``/`` (the trailing
      separator makes it an exact segment boundary, never a sibling-prefix collision).
    * PRIVATE, explicit ``session_scope`` -> exact, rebuilt with that session (narrows to ANY one
      of the user's sessions, not only the caller's own).
    """
    if ns.visibility is Visibility.SHARED:
        return "m.namespace = $ns", ns.to_prefix()
    if session_scope is None:
        return "m.namespace STARTS WITH $ns", f"{_user_scope_prefix(ns)}/"
    scoped = (
        ns
        if session_scope == ns.session
        else Namespace(
            org=ns.org,
            workspace=ns.workspace,
            user=ns.user,
            session=session_scope,
            visibility=ns.visibility,
        )
    )
    return "m.namespace = $ns", scoped.to_prefix()


def _user_scope_prefix(ns: Namespace) -> str:
    """BUG2 FIX (data-quality re-assessment §3 "MULTI-HOP TRAVERSAL DOES NOT WALK", scoping
    caveat): the USER-level scope key for the ``:Entity`` sub-graph (subject/object nodes +
    entity-entity edges, B5/B6) — the SAME shape as ``Namespace.to_prefix()`` MINUS the trailing
    ``session`` segment, deliberately. A real-world entity ("Ada", "Bo") and the relation between
    two of them are USER-level concepts, not session-scoped ones: the same person is the same
    graph node regardless of which conversation turn — and which SESSION — mentioned them. Before
    this fix every ``:Entity``/entity-edge was tagged with the FULL session-included
    ``ns.to_prefix()``, so the SAME person mentioned in two different sessions minted TWO
    disconnected entity nodes, and a query issued from session B could never walk an edge
    materialized while the caller was in session A — even though the PHYSICAL FalkorDB graph
    (``graph_name_for``, keyed on (org, workspace, visibility, user) — no session) already holds
    both sessions' facts in the exact same partition. This is exactly the re-assessment's
    "traversal filters on the FULL η-prefix INCLUDING session" caveat: physical isolation was
    already per-user; the property-level scope just hadn't caught up. ``:Memory`` nodes
    (``graph_recall``/``facts_at``/``find_conflicts``/session-scoped reads) are UNCHANGED — this
    narrower scope applies ONLY to the entity sub-graph the multi-hop traversal arm walks."""
    user_slot = "*" if ns.visibility is Visibility.SHARED else ns.user
    return "/".join(("mu", ns.org, ns.workspace, ns.visibility.value, user_slot))


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
# Safety bound on the SUPERSEDED_BY provenance-chain walk (:meth:`FalkorLtmAdapter.
# chain_head_state`) — a real chain is a handful of links; this only guards against a corrupt
# cyclic ``SUPERSEDED_BY`` edge causing an unbounded loop, never a normal-path limit.
_MAX_CHAIN_HOPS = 64
# Hard per-call cap on the bounded partition walk (:meth:`FalkorLtmAdapter.enumerate_page`),
# applied as a Cypher ``LIMIT`` on top of the caller's own ``limit``. Structural guard, not a
# tunable: ``facts_by_state`` next door is the un-paged read this one exists to replace, and the
# whole point of the replacement is that no single graph round trip can materialize a partition.
_MAX_ENUMERATE_PAGE = 512


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
        db: FalkorDB | None = None,
        *,
        db_factory: Callable[[], FalkorDB] | None = None,
        mapper: GraphMapper | None = None,
        clock: Clock | None = None,
        shortlist_size: int = _DEFAULT_SHORTLIST_SIZE,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        store_io_timeout_s: float = _DEFAULT_STORE_IO_TIMEOUT_S,
        mtm_entity_sink: EntityUidsSink | None = None,
    ) -> None:
        """AD-110: ``db`` and ``db_factory`` are the EAGER and the LAZY way to supply the
        connection, and exactly one of them must be given.

        ``FalkorDB.__init__`` runs a SYNCHRONOUS cluster-detection probe
        (``Is_Cluster`` -> a fresh blocking ``redis.Redis(...).info(section="server")``, even on
        the ``falkordb.asyncio`` class — ``falkordb/asyncio/cluster.py``). A composition root
        builds its stores from inside the ASGI ``lifespan`` coroutine, so an EAGER ``db=`` there
        runs that probe ON the event loop thread and stalls every other task for as long as it
        takes — bounded by ``socket_timeout`` since the previous fix, but a stall is a stall
        (DEV-STANDARDS async sharpener: *"no blocking/sync I/O in the event loop"*).

        ``db_factory`` is therefore what :func:`mu_engine.storage.factories._build_falkordb`
        passes: construction does ZERO I/O, and the probe happens on first USE, inside
        :meth:`_ensure_db`'s ``asyncio.to_thread`` — off the loop, once, under a lock, and still
        bounded by the same ``store_io_timeout_s`` budget (``retry_io``'s per-attempt
        ``asyncio.wait_for``). ``db=`` is retained because an integration test that already holds
        a live client has nothing to defer and no event loop to protect.

        A failed lazy connect leaves ``self._db`` unset, so the NEXT call retries it: a FalkorDB
        that is down at startup and up a minute later heals, which an eager probe cannot do
        (it takes the whole process down before ``/health`` exists to report it).
        """
        if (db is None) == (db_factory is None):
            raise ValueError(
                "FalkorLtmAdapter needs exactly one of `db` (an already-connected client) or "
                "`db_factory` (a zero-argument callable connected lazily, off the event loop) — "
                f"got db={'set' if db is not None else 'None'}, "
                f"db_factory={'set' if db_factory is not None else 'None'}"
            )
        self._db = db
        # ONE non-optional seam, so :meth:`_ensure_db` has no unreachable ``None`` branch to
        # narrow away: an eagerly-supplied client becomes a factory that returns it (never
        # called, since ``self._db`` is already set and the fast path returns first).
        self._db_factory: Callable[[], FalkorDB] = db_factory or (lambda: db)  # type: ignore[no-any-unimported]
        # Guards the one-time lazy connect. Created here (not lazily) so two concurrent first
        # calls cannot each mint their own lock and both run the probe.
        self._db_lock = asyncio.Lock()
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
        directive 2026-07-27): ``mu_g__{digest}__shared`` for the SHARED plane,
        ``mu_g__{digest}__u_{user}`` for PRIVATE, where ``digest`` is
        :func:`~mu_engine.storage.mappers.tenancy.tenant_partition_digest` — the SAME
        ``org``+``workspace`` SHA-256 digest the four MTM vector mappers already build their
        collection/table names from (DEV-STANDARDS rule 6, DRY: one collision-resistant
        derivation, not a second one re-typed here). Public so tests/observability can assert
        on — and print — the exact physical partition a namespace lands in without duplicating
        this computation (e.g. proving two orgs sharing a workspace id land in two distinct
        graphs).

        D-8 FIX: the previous implementation joined ``org``/``workspace`` RAW
        (``mu_g__{org}__{workspace}__...``) — ``_`` is not forbidden in either segment
        (``Namespace._FORBIDDEN_NS_CHARS``), so ``org="acme__eu", workspace="ws"`` and
        ``org="acme", workspace="eu__ws"`` produced the IDENTICAL name, physically colliding
        two different orgs' graphs. Hashing the two together (on the unambiguous ``":"``-joined
        pre-image ``tenant_partition_digest`` builds) removes that ambiguity regardless of what
        ``_`` patterns a caller's org/workspace slug contains. See the module docstring's
        OWNER DECISION 3 for why this outranks the (already-fixed) vector-tier version of the
        same defect.

        The PRIVATE ``user`` segment deliberately stays a RAW, readable suffix — never folded
        into the digest — for two reasons: (1) CANONICAL §1 rule 5 pins ``org``+``workspace`` as
        the two tenancy segments the physical partition must key on; rule 6 as CLARIFIED
        2026-08-26 holds that a per-user graph on the PRIVATE plane IS rule 5's partition
        applied, not a "new physical" store needing its own collision-resistant treatment — so
        widening the digest to cover ``user`` would over-apply a fix scoped to the tenancy
        segments, not the per-user one. (2) it costs nothing to keep collision-free anyway: the
        digest is a FIXED 16-hex-character string that can never itself contain ``_`` or
        collide with a different ``(org, workspace)`` pair's digest at any real tenant count
        (see :func:`tenant_partition_digest`'s own "collision-RESISTANT, not
        collision-resistant" precision note), so no ``user`` value — however many ``_``
        characters it contains — can ever cross the ``__`` boundary into ambiguity. The ``u_``
        prefix on the user segment (rather than appending ``user`` bare) exists only to keep the
        PRIVATE and SHARED branches themselves from colliding when a caller's literal ``user``
        happens to equal the string ``"shared"`` — a residual ambiguity the PRE-D-8 code already
        carried (``mu_g__{org}__{workspace}__shared`` for SHARED vs. a PRIVATE user literally
        named ``"shared"`` produced the same string) and which this fix closes as a zero-cost
        side effect of touching this method, not a new scope item.
        """
        digest = tenant_partition_digest(ns)
        if ns.visibility is Visibility.SHARED:
            return f"mu_g__{digest}__shared"
        return f"mu_g__{digest}__u_{ns.user}"

    async def _ensure_db(self) -> Any:
        """The FalkorDB client, connecting OFF the event loop on first use (AD-110).

        ``asyncio.to_thread`` is what moves the vendor constructor's blocking cluster probe off
        the loop; the ``asyncio.Lock`` + the second ``is not None`` check inside it make the
        connect happen exactly once even when N coroutines race the first call. On failure
        ``self._db`` stays ``None`` and the exception propagates to ``retry_io`` (a connection
        error is RETRYABLE), so the next call tries again rather than caching a dead client.
        """
        db = self._db
        if db is not None:
            return db
        async with self._db_lock:
            if self._db is None:
                self._db = await asyncio.to_thread(self._db_factory)
            return self._db

    async def _graph(self, ns: Namespace) -> Any:
        db = await self._ensure_db()
        return db.select_graph(self.graph_name_for(ns))

    async def ping(self) -> None:
        """Cheapest liveness verb for this tier — connect (lazily, off the loop) and PING.

        Public because AD-110 made the previous way of asking unusable. A composition root's
        health probe used to reach the raw client by private attribute path
        (``mu-server/src/mu_server/composition_shared.py:386`` probes ``self.ltm`` at
        ``"_db.connection"``), which resolved only because the client was built eagerly inside
        the factory. It is not built until first use any more, so that path resolves to ``None``
        and the probe reports the tier ``error`` on a perfectly healthy store — see this method's
        entry in the AD-110 report for the exact companion edit that is owed.

        Deliberately a ``PING`` on the underlying connection and NOT ``select_graph(name).query``:
        a read-only Cypher against a literal probe name MATERIALIZES that graph key on the
        multi-tenant store, outside every ``mu_g__…`` namespace — measured, and already recorded
        in the composition root that stopped doing it.

        NOT wrapped in ``self._retry``: a health probe reports what is true NOW, and a retry loop
        would report the state of the store several backoffs ago. The caller owns the bound.
        """
        db = await self._ensure_db()
        await db.connection.ping()

    async def aclose(self) -> None:
        """Release the underlying connection, if one was ever opened (AD-110).

        A lazily-connected adapter that was never used has nothing to close, and says so by doing
        nothing — the same posture a composition root's LIFO teardown already takes for an
        unresolvable client. The reason this exists as a METHOD is the mirror of :meth:`ping`'s:
        a teardown registry that resolved the raw client ONCE, at construction, now resolves
        ``None`` and silently registers no closer at all, so the socket outlives the process's
        shutdown sequence.
        """
        db = self._db
        if db is None:
            return
        await db.connection.aclose()

    async def upsert_fact(self, item: MemoryItem) -> None:
        return await self._retry(self._upsert_fact_impl)(item)

    async def _upsert_fact_impl(self, item: MemoryItem) -> None:
        g = await self._graph(item.namespace)
        row = self._mapper.to_store(item)
        props = row.props
        set_authorized = ", m.authorized_ids = $authorized_ids" if "authorized_ids" in props else ""
        cypher = (
            "MERGE (m:Memory {namespace: $namespace, id: $id}) "
            "SET m.subject = $subject, m.predicate = $predicate, m.object = $object, "
            "m.object_kind = $object_kind, m.polarity = $polarity, m.state = $state, "
            "m.valid_at = $valid_at, m.invalid_at = $invalid_at, "
            "m.content_hash = $content_hash, m.artifact_ref = $artifact_ref, "
            "m.provenance_id = $provenance_id, m.content = $content, "
            # The pin group's filterable half. `GraphMapper.to_store` promotes `pinned`/`version`
            # onto the node props precisely so a sweep can reach them from a Cypher WHERE
            # (memory-health-pinning-spec §3.1 line 168) — but this SET clause is an explicit
            # allowlist, so a prop the mapper produces and this list omits is simply never
            # written. That is not hypothetical: with the mapper updated and this line not,
            # `m.pinned` read back as NULL on every node and `enumerate(pinned=True)` returned
            # nothing from the graph tier while looking perfectly healthy.
            "m.pinned = $pinned, m.version = $version, "
            "m.memory_json = $memory_json"
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
        # BUG2 FIX: the entity-edge MATCH scopes on the USER-level prefix (see
        # `_user_scope_prefix`'s docstring) — NOT `props["namespace"]` (the `:Memory` node's own
        # FULL session-included prefix) — matching the scope `_merge_entity` just MERGEd `s`/`o`
        # under, two lines up.
        ns_prefix = _user_scope_prefix(item.namespace)
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
        # BUG2 FIX: USER-level scope (see `_user_scope_prefix`), not `ns.to_prefix()` — an entity
        # is a per-USER concept, deduped across every session that user has, not re-minted per
        # session.
        rows = await g_query(
            g, cypher, {"ns": _user_scope_prefix(ns), "canon": canonical, "uid": uid, "raw": name}
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
        except (UnexpectedResponse, MtmPointAbsentError):
            # write-after-read visibility lag (same NAMED degrade precedent as
            # `DistillPipeline._invalidate_mtm_guarded`) — the point isn't visible in Qdrant yet.
            # `MtmPointAbsentError` is the namespace-scoped adapter's typed spelling of the same
            # thing (C3): once the by-id payload write carries the tenancy predicate, a miss is a
            # silent wire-level success, so Qdrant's raw 404 no longer fires on its own.
            # Best-effort forward-compat backfill: never fails the LTM write itself, no retry
            # queue (unlike the invalidate guard) since a MISSED entity_uids backfill is a
            # degraded-but-safe outcome (the fact + entity edge are still correct in the graph).
            _log.warning(
                "mtm_entity_uids_backfill_point_absent",
                ns=item.namespace.to_prefix(),
                memory_id=mtm_point_memory_id,
            )

    async def get_fact(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return await self._retry(self._get_fact_impl)(ns, memory_id)

    async def _get_fact_impl(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        # Point-get by id (targeted lifecycle verbs: update/delete must LOCATE a fact first).
        # A plain ``MATCH (m:Memory {namespace, id})`` returning ``m.memory_json`` — NOT filtered
        # by state/validity (a superseded/expired fact is still returned so the verb acts on it
        # idempotently), unlike ``graph_recall``/``facts_at`` (bi-temporal still-valid read models).
        res = await g_query(
            await self._graph(ns),
            "MATCH (m:Memory {namespace: $ns, id: $id}) RETURN m.memory_json AS mj",
            {"ns": ns.to_prefix(), "id": memory_id},
        )
        if not res:
            return None
        return MemoryItem.model_validate_json(res[0][0])

    async def expire(self, ns: Namespace, memory_id: str, *, at: datetime) -> None:
        return await self._retry(self._expire_impl)(ns, memory_id, at=at)

    async def _expire_impl(self, ns: Namespace, memory_id: str, *, at: datetime) -> None:
        # Soft-delete (``delete`` verb, invalidate-don't-delete): stamp state='expired' +
        # invalid_at so every mandatory read filter (state='active' AND (invalid_at='' OR
        # invalid_at>now)) drops it, while the node + edges STAY (bi-temporal history) — NOT the
        # hard ``DETACH DELETE`` ``gc_delete`` performs. NO ``SUPERSEDED_BY`` edge (a plain delete
        # has no winner, unlike ``_invalidate_impl``). Closes the fact's own entity-entity edge too
        # (bi-temporal parity), reusing ``_invalidate_entity_edge``.
        g = await self._graph(ns)
        at_iso = at.isoformat()
        await g.query(
            "MATCH (m:Memory {namespace: $ns, id: $id}) "
            "SET m.state = $expired, m.invalid_at = $at",
            params={
                "ns": ns.to_prefix(),
                "id": memory_id,
                "expired": MemoryState.EXPIRED.value,
                "at": at_iso,
            },
        )
        await self._invalidate_entity_edge(g, ns, memory_id, at_iso)

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        session_scope: str | None = None,
    ) -> list[Scored[MemoryItem]]:
        return await self._retry(self._graph_recall_impl)(
            ns,
            subject=subject,
            predicate=predicate,
            limit=limit,
            caller_identity_set=caller_identity_set,
            session_scope=session_scope,
        )

    async def _graph_recall_impl(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        session_scope: str | None = None,
    ) -> list[Scored[MemoryItem]]:
        ns_predicate, ns_value = _resolve_memory_namespace_filter(ns, session_scope=session_scope)
        where = [
            ns_predicate,
            "m.state = $active",
            "(m.invalid_at = '' OR m.invalid_at > $now)",
        ]
        params: dict[str, Any] = {
            "ns": ns_value,
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
        res = await g_query(await self._graph(ns), cypher, params)
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
        res = await g_query(await self._graph(ns), cypher, params)
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
        res = await g_query(await self._graph(ns), cypher, params)
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
        g = await self._graph(ns)
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
        # BUG2 FIX: USER-level scope (see `_user_scope_prefix`) — matches the scope the edge was
        # MERGEd under in `_materialize_entity_edge`, not the `:Memory` node's session-scoped
        # `ns.to_prefix()`.
        await g.query(
            "MATCH (:Entity {namespace: $ns})-[r {memory_id: $loser_id}]->"
            "(:Entity {namespace: $ns}) SET r.invalid_at = $at",
            params={"ns": _user_scope_prefix(ns), "loser_id": loser_id, "at": at_iso},
        )

    async def mark_conflict(self, ns: Namespace, a_id: str, b_id: str, *, at: datetime) -> None:
        return await self._retry(self._mark_conflict_impl)(ns, a_id, b_id, at=at)

    async def _mark_conflict_impl(
        self, ns: Namespace, a_id: str, b_id: str, *, at: datetime
    ) -> None:
        # bare CONFLICTS_WITH edge between two facts that BOTH remain active (D3, spec §8) —
        # no state/invalid_at write to either side, unlike `_invalidate_impl`'s bundled edge.
        g = await self._graph(ns)
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
        # BUG2 FIX: USER-level scope (see `_user_scope_prefix`) — must match `_merge_entity`'s OWN
        # write scope, or every resolve MERGE-mints a duplicate entity node per session.
        res = await g_query(
            await self._graph(ns),
            cypher,
            {"ns": _user_scope_prefix(ns), "canon": canonical, "k": self._shortlist_size},
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
        res = await g_query(
            await self._graph(ns), cypher, {"ns": ns.to_prefix(), "art": artifact_id}
        )
        return [MemoryItem.model_validate_json(r[0]) for r in res]

    # ------------------------------------------------------------- LtmRetentionStorePort (S2-01)
    # The three validity-first retention capabilities ``mu_engine.lifecycle.retention.
    # RetentionService`` needs BEYOND ``GraphStorePort`` (that Protocol is an as-of/still-valid
    # read model with no enumerate-by-state query and no hard delete — see ``retention.py``'s
    # "missing LTM query/delete ports" note). PROMOTED here from the integrate-phase seam
    # ``tests/lifecycle/test_retention_int.py::_RealLtmRetentionStore`` proved against the live
    # mu-dev-falkordb — the SAME openCypher, now a REAL production adapter method (no test-only
    # store in the production path). ``FalkorLtmAdapter`` therefore structurally satisfies the
    # ``LtmRetentionStorePort`` Protocol (no import of the lifecycle layer needed — PEP 544).
    async def facts_by_state(
        self, ns: Namespace, states: frozenset[MemoryState]
    ) -> list[MemoryItem]:
        return await self._retry(self._facts_by_state_impl)(ns, states)

    async def _facts_by_state_impl(
        self, ns: Namespace, states: frozenset[MemoryState]
    ) -> list[MemoryItem]:
        # A full, namespace-scoped, un-filtered-by-validity enumeration of every :Memory node
        # whose ``state`` is one of ``states`` (deliberately NOT ``facts_at`` — that is an
        # as-of/still-valid read and would never return a dead or not-yet-expired-by-clock fact
        # the retention sweep must act on).
        cypher = (
            "MATCH (m:Memory) WHERE m.namespace = $ns AND m.state IN $states "
            "RETURN m.memory_json AS mj"
        )
        res = await g_query(
            await self._graph(ns),
            cypher,
            {"ns": ns.to_prefix(), "states": [s.value for s in states]},
        )
        return [MemoryItem.model_validate_json(r[0]) for r in res]

    # ---------------------------------------------------- TierEnumerationPort / TierPinPort --
    async def enumerate_page(
        self,
        ns: Namespace,
        *,
        states: frozenset[MemoryState],
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        """LTM's half of the bounded partition walk (``TierEnumerationPort``).

        **This is the bounded answer to :meth:`facts_by_state`, which is not one.** That method's
        own comment calls it "a full, namespace-scoped ... enumeration": ``MATCH (m:Memory) WHERE
        m.namespace = $ns AND m.state IN $states RETURN m.memory_json`` with no ordering, no
        ``SKIP`` and no ``LIMIT``. It is correctly tenant-scoped and still exactly the shape spec
        §3.1 forbids for a user-facing walk. It is deliberately left untouched —
        ``RetentionService`` depends on the un-paged form — and this method is what the health
        view and the pin bound use instead.

        Paged by a KEYSET cursor on ``m.id`` (``ORDER BY m.id`` + ``m.id > $cursor``), not
        ``SKIP``/``OFFSET``. A keyset walk costs the same on page 1000 as on page 1 and, more
        importantly, cannot silently skip or repeat rows when a concurrent write shifts the
        offset underneath a long walk — which for a health view is the difference between "these
        are your memories" and "these are some of your memories".

        ``coalesce(m.pinned, false)`` rather than a bare ``m.pinned = $pinned``: the property was
        promoted onto the node only when the pin group landed, so nodes written before that carry
        no ``pinned`` at all. In Cypher a missing property is ``null``, and ``null = false`` is
        ``null``, not ``true`` — so an un-coalesced predicate would silently drop every
        pre-existing fact from an ``enumerate(pinned=False)`` sweep.
        """
        return await self._retry(self._enumerate_page_impl)(
            ns, states=states, pinned=pinned, cursor=cursor, limit=limit
        )

    async def _enumerate_page_impl(
        self,
        ns: Namespace,
        *,
        states: frozenset[MemoryState],
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        if limit <= 0:
            return [], None
        capped = min(limit, _MAX_ENUMERATE_PAGE)
        # `m.namespace = $ns` is the property-level tenancy predicate; the physical graph is
        # ALREADY partitioned per (org, workspace, visibility|user) by `graph_name_for`, so this
        # is the second of the two layers `storage-pluggable-spec.md §6` item 1 requires — never
        # a filter standing alone.
        where = ["m.namespace = $ns", "m.state IN $states"]
        params: dict[str, Any] = {
            "ns": ns.to_prefix(),
            "states": [s.value for s in states],
            "limit": capped,
        }
        if pinned is not None:
            where.append("coalesce(m.pinned, false) = $pinned")
            params["pinned"] = pinned
        if cursor is not None:
            where.append("m.id > $cursor")
            params["cursor"] = cursor
        cypher = (
            f"MATCH (m:Memory) WHERE {' AND '.join(where)} "
            "RETURN m.memory_json AS mj, m.id AS id ORDER BY m.id LIMIT $limit"
        )
        rows = await g_query(await self._graph(ns), cypher, params)
        items = [MemoryItem.model_validate_json(r[0]) for r in rows]
        # A SHORT page means the walk is exhausted; a FULL page means there may be more. Erring
        # toward one extra empty page is correct here — claiming exhaustion on a full page would
        # silently truncate the caller's view of its own partition.
        next_cursor = str(rows[-1][1]) if len(rows) == capped and rows else None
        return items, next_cursor

    async def set_pinned(
        self,
        ns: Namespace,
        memory_id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None,
    ) -> int | None:
        """LTM's half of the id-stable cross-store pin upsert (``TierPinPort``)."""
        return await self._retry(self._set_pinned_impl)(
            ns, memory_id, pinned, at=at, by=by, reason=reason
        )

    async def _set_pinned_impl(
        self,
        ns: Namespace,
        memory_id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None,
    ) -> int | None:
        # `{namespace: $ns, id: $id}` is the node-level tenancy predicate, inside the MATCH — a
        # foreign memory_id matches no node and the SET touches nothing, the graph twin of
        # `qdrant_mtm._scoped_point_selector`. Read-then-write (not a blind SET) because the
        # version is derived from the CURRENT record and because `memory_json` is the lossless
        # carrier every read reconstructs from: writing the promoted `pinned` property without
        # rewriting `memory_json` would leave the node's filterable view and its authoritative
        # view disagreeing about whether the item is pinned.
        current = await self._get_fact_impl(ns, memory_id)
        if current is None:
            return None  # not resident in THIS tier's partition.
        updated = with_pin_group(current, pinned=pinned, at=at, by=by, reason=reason)
        await (await self._graph(ns)).query(
            "MATCH (m:Memory {namespace: $ns, id: $id}) "
            "SET m.pinned = $pinned, m.version = $version, m.memory_json = $mj",
            params={
                "ns": ns.to_prefix(),
                "id": memory_id,
                "pinned": updated.pinned,
                "version": updated.version,
                "mj": updated.model_dump_json(),
            },
        )
        return updated.version

    async def chain_head_state(self, ns: Namespace, memory_id: str) -> MemoryState:
        return await self._retry(self._chain_head_state_impl)(ns, memory_id)

    async def _chain_head_state_impl(self, ns: Namespace, memory_id: str) -> MemoryState:
        # Walk the ``SUPERSEDED_BY`` provenance chain forward from ``memory_id`` to its terminal
        # node (the one with no outgoing ``SUPERSEDED_BY`` edge — "the chain head") and return
        # THAT node's state (spec §9 mermaid: GC only when "chain head dead"). A node with no
        # outgoing edge is its own head (the common EXPIRED case — a self-expire gains no edge).
        graph = await self._graph(ns)
        prefix = ns.to_prefix()
        current_id = memory_id
        for _ in range(_MAX_CHAIN_HOPS):
            hop = await g_query(
                graph,
                "MATCH (:Memory {namespace: $ns, id: $id})-[:SUPERSEDED_BY]->(next:Memory) "
                "RETURN next.id AS id",
                {"ns": prefix, "id": current_id},
            )
            if not hop:
                head = await g_query(
                    graph,
                    "MATCH (m:Memory {namespace: $ns, id: $id}) RETURN m.state AS state",
                    {"ns": prefix, "id": current_id},
                )
                return MemoryState(head[0][0])
            current_id = str(hop[0][0])
        raise RuntimeError(f"chain_head_state: exceeded {_MAX_CHAIN_HOPS} hops (cycle?)")

    async def gc_delete(self, ns: Namespace, memory_id: str) -> None:
        return await self._retry(self._gc_delete_impl)(ns, memory_id)

    async def _gc_delete_impl(self, ns: Namespace, memory_id: str) -> None:
        # The one true HARD delete the retention sweep ever calls — never on an ACTIVE fact
        # (RetentionService gates this on SUPERSEDED/EXPIRED + window + chain-head-dead). Real
        # ``DETACH DELETE`` (drops the node AND its edges) against the live graph, no mock.
        await (await self._graph(ns)).query(
            "MATCH (m:Memory {namespace: $ns, id: $id}) DETACH DELETE m",
            params={"ns": ns.to_prefix(), "id": memory_id},
        )

    async def traverse_entities(
        self,
        ns: Namespace,
        *,
        query: str,
        max_hops: int,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
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

        AUTHZ (C2 fix): the ids this arm derives come from the WORKSPACE-wide ``:Entity``
        sub-graph (``_user_scope_prefix``, user slot ``*`` on SHARED), so the ``:Memory``
        hydration below MUST re-impose both walls ``graph_recall`` already imposes — the
        ``_resolve_memory_namespace_filter`` room predicate AND, on SHARED, the
        ``m.authorized_ids`` Model-A ACL predicate built from ``caller_identity_set``
        (CANONICAL-CONTRACTS.md:529, governance-policy-spec.md:124). Both are SERVER-SIDE and
        PRE-truncation (CANONICAL-CONTRACTS.md:508-510), never a post-filter on the returned
        rows. Before this fix the hydration carried NEITHER, so a multi-hop recall returned facts
        from any room in the workspace and from any ACL — a bypass strictly worse than the MTM
        leak fixed in ``7ccc405``, which at least required the caller to already KNOW a memory id.
        """
        return await self._retry(self._traverse_entities_impl)(
            ns,
            query=query,
            max_hops=max_hops,
            limit=limit,
            caller_identity_set=caller_identity_set,
        )

    async def _traverse_entities_impl(
        self,
        ns: Namespace,
        *,
        query: str,
        max_hops: int,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        tokens = {t.casefold() for t in re.findall(r"[A-Za-z0-9]+", query) if len(t) > 1}
        if not tokens:
            return []
        g = await self._graph(ns)
        # BUG2 FIX (scoping): the entity/edge frontier walk scopes on the USER-level prefix (see
        # `_user_scope_prefix`'s docstring), not the session-included `ns.to_prefix()` — so a
        # relational query issued from ANY session for this user walks entity edges materialized
        # from EVERY session, not just the one the query happens to be running in.
        user_ns_prefix = _user_scope_prefix(ns)
        now_iso = self._clock.now().isoformat()
        hops = max(1, min(max_hops, 2))
        query_stems = {_stem(t) for t in tokens}

        frontier = sorted(tokens)
        seen_entities: set[str] = set(frontier)
        # memory_id -> (smallest hop distance, whether ITS predicate lexically matches a query
        # word) — BUG2 FIX (c): the predicate-relevance signal below lets a genuinely on-topic
        # 2-hop relation ("owns", for "what does Bo own?") outrank an unrelated 1-hop same-subject
        # attribute (e.g. Ada's own favorite-coffee edge), which pure hop-distance alone cannot.
        memory_hop: dict[str, int] = {}
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
                g, cypher, {"ns": user_ns_prefix, "frontier": frontier, "now": now_iso}
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
        # C2 FIX (authorization bypass): the hydration below re-imposes the SAME two walls
        # `_graph_recall_impl` imposes, because the ids above were derived from the WORKSPACE-wide
        # entity sub-graph (`_user_scope_prefix`, user slot `*` on SHARED) and therefore prove
        # nothing about the room or the ACL of the `:Memory` they point at.
        #
        # `_resolve_memory_namespace_filter` is used rather than `_user_scope_prefix` precisely
        # because it already encodes the asymmetry this arm needs, and the asymmetry is
        # DELIBERATE on both sides:
        #   * PRIVATE, `session_scope is None` (the only mode this arm calls it in) -> `STARTS
        #     WITH` the session-less USER prefix, so the BUG2 cross-session walk is PRESERVED
        #     exactly: a relational query issued from session B still hydrates a fact captured in
        #     session A. That was the whole point of the BUG2 scoping fix and is not narrowed.
        #   * SHARED -> exact `to_prefix()` (room-included), UNCONDITIONALLY — "rooms are real
        #     walls" (this module's own `_resolve_memory_namespace_filter` docstring). SHARED must
        #     NOT get the PRIVATE relaxation: a different session on the SHARED plane is a
        #     different ROOM, not another of the user's own conversations.
        ns_predicate, ns_value = _resolve_memory_namespace_filter(ns, session_scope=None)
        fetch_where = [
            ns_predicate,
            "m.id IN $ids",
            "m.state = $active",
            "(m.invalid_at = '' OR m.invalid_at > $now)",
        ]
        fetch_params: dict[str, Any] = {
            "ns": ns_value,
            "ids": list(memory_hop),
            "active": MemoryState.ACTIVE.value,
            "now": now_iso,
        }
        # The Model-A ACL clause — the SAME form `_graph_recall_impl` builds, reused verbatim
        # rather than a second dialect of the same rule.
        if ns.visibility is Visibility.SHARED and caller_identity_set is not None:
            fetch_where.append("ANY(cid IN $caller WHERE cid IN m.authorized_ids)")
            fetch_params["caller"] = list(caller_identity_set)
        fetch_cypher = (
            f"MATCH (m:Memory) WHERE {' AND '.join(fetch_where)} "
            "RETURN m.id AS id, m.memory_json AS mj"
        )
        rows = await g_query(g, fetch_cypher, fetch_params)
        scored: list[Scored[MemoryItem]] = []
        for mid, mj in rows:
            hop = memory_hop.get(str(mid), hops)
            item = MemoryItem.model_validate_json(mj)
            # BUG2 FIX (a)/(c): a flat `1/hop` score alone ranks any 1-hop same-subject attribute
            # (e.g. Ada's OWN favorite-coffee edge) above a genuinely on-topic 2-hop relation, and
            # ties every 1-hop hit together regardless of whether it actually answers the query's
            # relation ("manager"/"owns"/...). `_predicate_matches_query` adds a flat bonus when
            # the fact's OWN predicate lexically relates to a query word — this is what makes the
            # genuinely-asked-about relation ("Ada MANAGES Bo" for "who is Bo's manager?") rank
            # above noise at the SAME hop distance, and even above a CLOSER but irrelevant hop.
            relevance_bonus = 1.0 if _predicate_matches_query(item.predicate, query_stems) else 0.0
            scored.append(
                Scored(
                    item=item,
                    score=(1.0 / hop) + relevance_bonus,
                    channel=RecallChannel.LTM_GRAPH,
                    rank=0,
                )
            )
        scored.sort(key=lambda s: -s.score)
        return [s.model_copy(update={"rank": rank}) for rank, s in enumerate(scored[:limit])]


_STEM_SUFFIXES = ("ing", "ers", "er", "es", "ed", "s")


def _stem(word: str) -> str:
    """BUG2 FIX (data-quality re-assessment §3, ranking item (c)): a crude common-suffix strip —
    NOT a real stemmer, no NLP dependency — just enough to equate a query's natural-language
    relation word with the canonicalized predicate string it maps to ("manager"/"manages"/
    "managed" -> "manag"; "owns"/"owner"/"owning" -> "own"). Only strips when the remainder is
    still at least 3 characters (``len(w) > len(suffix) + 2``) so short/unrelated words never
    over-strip into an accidental collision (e.g. "is" is returned unchanged)."""
    for suffix in _STEM_SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _predicate_matches_query(predicate: str | None, query_stems: set[str]) -> bool:
    """BUG2 FIX (c): True iff this fact's OWN predicate lexically relates to a word actually
    typed in the query (via :func:`_stem`) — the traversal-relevance signal that lets "Ada MANAGES
    Bo" outrank an unrelated same-subject attribute for "who is Bo's MANAGER?", and "Bo OWNS Q3
    report" outrank one for "what does Bo OWN?". A canonicalized predicate can be multi-word
    (``project_deadline``, ``lives_in``) — every underscore-segment is stemmed and checked
    independently so a compound predicate still matches on its meaningful segment."""
    if not predicate:
        return False
    for segment in re.split(r"[_\s]+", predicate.casefold()):
        if len(segment) < 3:
            continue
        seg_stem = _stem(segment)
        if any(
            seg_stem == qs or seg_stem.startswith(qs) or qs.startswith(seg_stem)
            for qs in query_stems
            if len(qs) >= 3
        ):
            return True
    return False


async def g_query(graph: Any, cypher: str, params: dict[str, Any]) -> list[list[Any]]:
    """Run an openCypher read and return the raw result rows (result_set)."""
    result = await graph.query(cypher, params=params)
    return list(result.result_set or [])
