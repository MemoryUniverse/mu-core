"""``CentralityService`` + ``CentralityIndex`` — the structural-salience precompute (A4).

**What this is.** A fourth, *structural* input to the sweep-gate score ``S(m)``
(``lifecycle/salience.py``, spec §6): how connected a candidate fact's entities are inside the
namespace's own LTM knowledge graph. A fact whose subject or object is a hub — a "god node" — is
structurally load-bearing and is worth promoting/keeping; a fact hanging off two leaves is not.

```
deg(e)    = number of DISTINCT other entities adjacent to e in ns's entity graph
deg_hub(e)= deg(e) if deg(e) > min_hub_degree else 0     # structural-isolation floor
raw(m)    = max(deg_hub(subject(m)), deg_hub(object(m))) # as central as its best endpoint
cen(m)    = min(raw(m) / degree_cap, 1.0)                # saturating, [0, 1]
```

**Why it cannot live inside ``score()``.** ``SalienceStrategy.score`` is pure and SYNCHRONOUS and
runs once per item on the sweep path; centrality is a whole-graph property that needs I/O. So
:class:`CentralityService` computes the namespace's entity-degree projection ASYNCHRONOUSLY, once
per namespace per sweep, and publishes it into a :class:`CentralityIndex`; ``score()`` then reads
``cen(m)`` out of that index with a plain dict lookup. Same shape the lifecycle spec already uses
for ``access_count`` (written by the recall path, read for free by ``SalienceStrategy._usage``,
spec §6 line 267): the expensive part happens elsewhere, the score path stays pure.

**The value is NOT written back onto the item, and that is a corrected design decision.** The
first cut of A4 persisted ``cen(m)`` into ``MemoryItem.metadata`` and re-``upsert_fact``-ed the
item, on the reasoning that this mirrors ``relevance_score``. Three findings killed that shape,
each verified against the live dev FalkorDB before this rewrite:

1. *It landed on the wrong tier, so the term was inert.* ``upsert_fact`` writes an LTM ``:Memory``
   node, but every production caller of ``SalienceStrategy.score`` scores an STM or an MTM item —
   ``promotion.py:388`` (STM window), ``promotion.py:270`` (MTM->LTM gate), ``demotion.py:236``
   (``MtmTierRepository.scan_for_demotion`` candidates). Nothing scores an LTM-resident item, and
   nothing copies LTM ``metadata`` back into the Qdrant payload (``falkor_ltm.py:442-457``
   backfills ``entity_uids`` and nothing else). A persisted-on-LTM value could therefore never be
   read by any gate. Asking the graph about a *candidate for* promotion is also the semantically
   right question; asking it about a fact already in the graph is not.
2. *It resurrected superseded facts.* ``_upsert_fact_impl`` (``falkor_ltm.py:313-332``) SETs the
   FULL node — ``m.state``, ``m.invalid_at``, ``m.pinned``, ``m.version``, ``m.memory_json`` —
   from whatever snapshot it is handed. A pass reads the namespace once, then writes each item
   back with an ``await`` in between, so any ``invalidate``/``expire``/``set_pinned`` landing in
   that window is reverted. Reproduced live: a fact superseded between the read and the write-back
   came back ``state='active', invalid_at=''`` and was visible in ACTIVE ``graph_recall`` again,
   with its ``SUPERSEDED_BY`` edge still attached — loser and winner of a resolved contradiction
   both active at once. This is verbatim the defect class ``pipelines/distill.py:727-736`` already
   documents and forbids ("never trust the reconcile-time snapshot for a write").
3. *It was a 4x un-batched write amplification on the tier user recall reads.* Each changed fact
   cost a ``:Memory`` MERGE plus ``_materialize_entity_edge``'s two ``_merge_entity`` calls and an
   edge MERGE (``falkor_ltm.py:343-440``), re-stamped ``r.valid_at``/``r.invalid_at`` on entity
   edges, rewrote ``item.metadata['entity_uids']``, and could reach into the MTM/Qdrant tier
   through ``EntityUidsSink`` — all from a purely derived scalar.

A per-namespace in-memory projection has none of those properties: it is read-only against the
store, it lands on exactly the items being scored whatever tier they live in, and a stale entry is
at worst a slightly-old structural hint (unlike a stale ``state``, which is a correctness defect).
Persisting the scalar remains desirable for provenance/explain and cross-process reuse, but only
through a NARROW single-field write that does not exist on any port today — see the module report
/ ARCHITECTURE-DELTAS for the follow-up.

**Ported from (CODE-ADOPTION-METHODOLOGY.md — actual source, read on disk).**
``other_repos/graphify/graphify/analyze.py::god_nodes`` lines 109-130, clone
``f5a3592882ad54e5394c5cd5391786a589110bd1`` (the exact commit
``docs/superpowers/design/research-graphify-adoption.md`` names, and the exact line range its
§4/§6 A4 item cites). What actually ports, verbatim in substance:

* ``degree = dict(G.degree())`` (``analyze.py:115``) on a ``networkx.Graph`` — a SIMPLE graph, so
  parallel edges collapse and ``degree`` is the DISTINCT-neighbour count. Reproduced here with
  ``collections.defaultdict(set)``: no NetworkX dependency is taken, because the ported line is a
  degree count and nothing else. (``networkx`` is importable in this workspace only TRANSITIVELY,
  via ``torch`` — ``uv.lock``; depending on it would be a reproducibility violation, and
  ``research-graphify-adoption.md:264-265`` in any case forbids the graspologic half of that stack
  inside ``mu-core``.)
* The structural-isolation exclusion ``G.degree(node_id) <= 1`` (``analyze.py:87-88``) — the ONE
  noise clause in ``god_nodes``' filter chain that is corpus-agnostic. It appears here as the
  configurable :attr:`CentralitySettings.min_hub_degree` floor.

**What deliberately does NOT port, and is NOT cited to graphify.** ``_is_file_node``
(``analyze.py:63-89``), ``_is_json_key_node`` (``:100-106``), ``_is_concept_node`` and
``_BUILTIN_NOISE_LABELS`` (``:11-29``) every one key on source filenames, file extensions, AST
label shapes, or language builtins. MU's ``:Entity`` graph has none of those — its nodes are
``canonical_name``/``entity_uid``/``aliases`` (``storage/adapters/falkor_ltm.py:427-433``). A claim
that those filters ported would be a fabricated citation. MU's own principled exclusion is stronger
and is applied instead: the projection reads only ACTIVE, still-bi-temporally-valid facts, so
superseded/expired history never inflates a degree.

**Two further DELIBERATE deviations from the research doc, both recorded in the design delta.**

1. *Plane.* ``research-graphify-adoption.md:264-265`` and ``:328`` put A4 in a SERVER-SIDE
   consolidation worker, "not in ``mu-core`` or ``mu-local``". It lives in ``mu-core`` here. The
   server placement was driven by the graspologic/Leiden/betweenness half of that proposal (A5);
   plain degree needs neither. Keeping it server-side would gate a piece of single-user engine
   QUALITY behind the server, which the project boundary rule forbids outright ("do NOT gate engine
   quality behind the server" — ``CLAUDE.md``). Betweenness and Leiden stay where the research doc
   put them; degree comes local.
2. *Output shape.* ``god_nodes`` returns a ranked top-N LIST of hub nodes (``analyze.py:123-129``).
   Salience needs a per-item scalar in [0, 1] on EVERY item. The normalisation
   (``min(raw / degree_cap, 1.0)``) is OURS, not graphify's — graphify never normalises. It
   deliberately reuses the saturating shape ``SalienceStrategy._usage`` already ratified for
   ``use(m)`` rather than minting a second normalisation idiom (DEV-STANDARDS rule 6, DRY).

**Namespace scoping (CLAUDE.md rule 4 — the hard one).** A projection is built ONLY from
:meth:`LtmCentralityStorePort.graph_recall`'s return for the ONE namespace being refreshed, and is
filed in the index under :func:`projection_key` — the SAME grouping ``graph_recall`` itself filters
on (``falkor_ltm.py:104-143``): the exact room prefix on SHARED, the session-less user prefix on
PRIVATE. A lookup for an item recomputes that key from the item's OWN namespace, so an item can
only ever be scored against the projection of its own tenant. A degree computed across tenants
would be both a leak and a wrong number. ``session_scope=None`` is passed deliberately: the
``:Entity`` sub-graph and its edges are tagged with the session-LESS user prefix
(``falkor_ltm.py:146-164``) while ``:Memory`` nodes carry the full session-included prefix, so a
session-scoped read would compute a session-local degree over a user-scoped graph and would
silently vary by which session happened to run the sweep.

**SHARED plane, principal granularity — a stated limitation, escalated not buried.** No
``caller_identity_set`` is passed: a sweep genuinely has no caller, and inventing one would be an
authorization forgery. Room isolation therefore holds exactly (verified live: a second room's facts
never enter another room's projection), but WITHIN a room the degree is computed over every ACTIVE
fact regardless of Model-A ``authorized_ids``. That is defensible — the sweep is the engine acting
on the room's own data, no principal is reading, and no score or content is returned to anyone —
but it is a real design question the owner has not ruled on, so it is recorded as an open item in
ARCHITECTURE-DELTAS rather than settled here. ``storage/ports.py:285-286``'s "an implementation that
ignores it on SHARED is an authorization bypass" governs ``traverse_entities`` HYDRATING rows *for
a caller*; this path has no caller and returns no rows.

**Truncation withholds; it never publishes a lower bound.** ``graph_recall`` orders
``valid_at DESC LIMIT $limit`` (``falkor_ltm.py:565``), so a namespace larger than
``max_facts_per_pass`` would yield degrees computed over "the newest N facts" — systematically
UNDER-counted, and varying with sweep timing. Publishing those would manufacture false "peripheral"
evidence that penalises S(m). So a truncated pass publishes NOTHING: the namespace keeps whatever
projection it already had, ``cen`` stays ABSENT if it had none, the report says ``truncated``, and a
warning is logged. Absence renormalises away (``salience.py``); a wrong number does not.

**Content-free discipline (CLAUDE.md rule 3).** Subjects, objects, predicates and content never
reach a log, span, metric or audit row. Only counts, the namespace prefix, and enum-ish labels.

**Bounded (DEV-STANDARDS: never unbounded).** The projection read is capped by
``max_facts_per_pass`` and wrapped in an ``asyncio.timeout``; the index holds at most
``max_namespaces`` projections, evicting least-recently-refreshed first. Cancellation-correct:
``CancelledError`` propagates untouched and is never counted as a failure (DEV-STANDARDS rule 1).
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    TraceScope,
)
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import Scored

__all__ = [
    "CentralityIndex",
    "CentralityLookup",
    "CentralityReport",
    "CentralityService",
    "CentralitySettings",
    "LtmCentralityStorePort",
    "projection_key",
]

_log = structlog.get_logger("mu_engine.lifecycle.centrality")

_OP = "lifecycle.centrality_refresh"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"


def projection_key(ns: Namespace) -> str:
    """The tenancy key one projection covers — the SAME grouping ``graph_recall`` filters on.

    A DELIBERATE re-statement of ``falkor_ltm._resolve_memory_namespace_filter(ns,
    session_scope=None)``'s param value (``falkor_ltm.py:104-143``), not an import of it: this
    module is engine-lifecycle and must not reach into a storage ADAPTER's privates. The two are
    drift-guarded by a test that asserts this function agrees with the adapter's own helpers for
    both visibilities.

    * SHARED -> the exact room prefix. Rooms are real walls; ``session_scope`` never relaxes SHARED.
    * PRIVATE -> the session-less user prefix + ``/``. The trailing separator makes it an exact
      segment boundary (``/`` is in ``_FORBIDDEN_NS_CHARS``,
      ``mu_contracts/domain/model/memory.py:96``, so no sibling-prefix collision is expressible).
    """
    if ns.visibility is Visibility.SHARED:
        return ns.to_prefix()
    return "/".join(("mu", ns.org, ns.workspace, ns.visibility.value, ns.user)) + "/"


@runtime_checkable
class LtmCentralityStorePort(Protocol):
    """The narrow LTM seam :class:`CentralityService` needs — READ-ONLY, and already present on
    ``GraphStorePort``/``FalkorLtmAdapter``, so the real adapter satisfies this STRUCTURALLY with
    no import of this module and no edit to ``storage/ports.py`` or ``storage/adapters/**`` (both
    owned by other lanes).

    Mirrors the precedent ``lifecycle/retention.py``'s ``LtmRetentionStorePort`` set, with one
    difference worth stating: retention's capabilities did NOT exist on the port and had to be
    added; this ONE does, so this Protocol is a NARROWING (principle of least privilege — this
    service can enumerate and can do NOTHING else to the graph, in particular it cannot write),
    not a request for new capability.

    A degree aggregate query (``MATCH (e:Entity {namespace:$ns})-[r]-() RETURN e, count(r)``) would
    be strictly cheaper than enumerating facts and counting in Python, and would remove
    ``max_facts_per_pass`` truncation entirely — but it would have to live in
    ``storage/adapters/falkor_ltm.py``, another lane's file. REPORTED as a follow-up, not reached
    across for; the Python projection is correct today and is bounded.
    """

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        session_scope: str | None = None,
    ) -> list[Scored[MemoryItem]]: ...


class CentralitySettings(BaseModel):
    """Structural-salience knobs. No value here is ever a call-site literal (DEV-STANDARDS rule 3);
    the whole subtree reaches the env as ``MU_LIFECYCLE__CENTRALITY__*`` once a composition root
    threads ``LifecycleSettings`` (the same path ``tests/config/test_engine_settings_unit.py``
    already proves for ``…__SALIENCE__*``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: OFF makes :meth:`CentralityService.refresh` a no-op that publishes nothing. Every item then
    #: keeps ``cen(m)`` ABSENT, and ``S(m)`` is byte-for-byte today's three-term score — so
    #: disabling this feature is genuinely free, not a degraded mode.
    enabled: bool = True

    #: The degree at which ``cen(m)`` saturates to 1.0. Deliberately the same value and the same
    #: saturating shape as ``SalienceSettings.usage_cap`` (10): both answer "how much of this
    #: unbounded count is enough to count as maximal?", and using one number for one idiom is DRY.
    #: An entity with 10+ distinct neighbours in a personal memory graph is a genuine hub.
    degree_cap: int = Field(default=10, ge=1)

    #: Structural-isolation floor — ``graphify/analyze.py:87-88``'s ``G.degree(node) <= 1``
    #: exclusion, made configurable. An entity at or below this degree is a leaf, not a hub, and
    #: contributes NO structural credit (``deg_hub = 0``). ``0`` disables the floor.
    min_hub_degree: int = Field(default=1, ge=0)

    #: Shared-box RAM guard on the projection read (DEV-STANDARDS: bounded, never unbounded), and
    #: the mandatory ``limit`` ``graph_recall`` requires. Mirrors
    #: ``LifecycleSettings.max_items_per_user_sweep``'s own bound and default. A pass that EXCEEDS
    #: this cap publishes nothing at all — see the module docstring's truncation paragraph.
    max_facts_per_pass: int = Field(default=2_000, ge=1)

    #: How many namespaces' projections :class:`CentralityIndex` retains before evicting the
    #: least-recently-refreshed. Bounds the index's RAM on a multi-tenant process.
    max_namespaces: int = Field(default=256, ge=1)

    #: Deadline on the ONE store read this service makes (DEV-STANDARDS timeouts/resilience). A
    #: timeout leaves the previous projection in place and is reported as a failed refresh; it
    #: never publishes a partial projection.
    read_timeout_s: float = Field(default=30.0, gt=0.0)


class CentralityReport(BaseModel):
    """One namespace's refresh outcome — content-free by construction (counts only, no ids, no
    entity names, no memory text)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Facts read back from the namespace's graph this pass.
    evaluated: int = 0
    #: Facts that carried a real (subject, object) pair and therefore contributed an edge.
    scored: int = 0
    #: Distinct entities in the projection.
    entities: int = 0
    #: ``True`` iff the projection was PUBLISHED into the index (and is therefore now readable by
    #: ``SalienceStrategy``). ``False`` for a disabled or truncated pass.
    published: bool = False
    #: ``True`` iff the namespace holds MORE than ``max_facts_per_pass`` facts. The projection is
    #: then a lower bound and is deliberately DISCARDED rather than published.
    truncated: bool = False
    #: ``True`` iff the service was disabled and nothing was read.
    skipped: bool = False


def _entity_key(name: str) -> str:
    """The entity identity a projection groups on — ``.strip().casefold()``, byte-identical to the
    ``canonical_name`` key the graph tier itself MERGEs ``:Entity`` nodes on
    (``falkor_ltm.py:731`` ``canonical = name.strip().casefold()``). Using any other key here would
    count "Postgres" and "postgres" as two nodes and compute a degree the real graph does not
    have."""
    return name.strip().casefold()


def _hub_degree(adjacency: Mapping[str, frozenset[str]], key: str, min_hub_degree: int) -> int:
    """``deg(e)`` with ``graphify/analyze.py:87-88``'s structural-isolation exclusion applied: an
    entity at or below ``min_hub_degree`` is a leaf, not a hub, and contributes 0."""
    degree = len(adjacency.get(key, ()))
    return degree if degree > min_hub_degree else 0


def _project(facts: Sequence[MemoryItem]) -> dict[str, frozenset[str]]:
    """The undirected simple-graph projection — ``networkx.Graph`` semantics, reproduced.

    ``defaultdict(set)`` is what makes this a SIMPLE graph: many facts between the same two
    entities (``Ada uses Postgres`` / ``Ada likes Postgres``) collapse to ONE adjacency, exactly as
    ``nx.Graph`` collapses parallel edges before ``G.degree()`` counts them (``analyze.py:115``).
    Without that collapse, ten facts about one pair would mint a fake hub — the very "mechanically
    accumulated edges" ``god_nodes``' docstring excludes.

    Self-loops (subject and object resolving to the same entity) are dropped: they add no
    neighbour, and ``nx.Graph`` would count them as degree 2 — an inflation with no structural
    meaning here.
    """
    adjacency: dict[str, set[str]] = defaultdict(set)
    for item in facts:
        if not (item.subject and item.object):
            continue
        a = _entity_key(item.subject)
        b = _entity_key(item.object)
        if not a or not b or a == b:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    return {key: frozenset(neighbours) for key, neighbours in adjacency.items()}


@runtime_checkable
class CentralityLookup(Protocol):
    """The SYNCHRONOUS, I/O-free seam ``SalienceStrategy`` reads ``cen(m)`` through.

    Deliberately the whole contract: one pure function of the item. Anything that needs to await,
    to open a socket, or to know the wall clock is on the WRONG side of this seam and belongs in
    :class:`CentralityService`.
    """

    def centrality_for(self, item: MemoryItem) -> float | None: ...


class CentralityIndex:
    """The per-namespace entity-degree projections, and the pure lookup ``score()`` uses.

    Bounded: at most ``max_namespaces`` projections, evicted least-recently-refreshed first
    (``OrderedDict`` as an LRU, the same idiom the platform uses elsewhere). Publishing a
    namespace REPLACES its projection wholesale — a projection is a snapshot, never merged into,
    so a shrinking graph cannot leave phantom edges behind.

    A projection has NO expiry, and that is intentional: staleness here degrades a structural HINT
    (this entity was a hub as of the last sweep), never a correctness field like ``state`` or
    ``invalid_at``. Giving the lookup a clock would also destroy ``score()``'s AC-0.1 property of
    being a pure function of ``(item, clock.now())`` by making it depend on a second, hidden clock.
    Freshness is a property of the sweep cadence and is the refresher's job.
    """

    __slots__ = ("_by_key", "_settings")

    def __init__(self, settings: CentralitySettings | None = None) -> None:
        self._settings = settings or CentralitySettings()
        self._by_key: OrderedDict[str, dict[str, frozenset[str]]] = OrderedDict()

    def publish(self, ns: Namespace, adjacency: dict[str, frozenset[str]]) -> None:
        """Install ``ns``'s freshly-computed projection, evicting the LRU entry if over budget."""
        key = projection_key(ns)
        self._by_key.pop(key, None)
        self._by_key[key] = adjacency
        while len(self._by_key) > self._settings.max_namespaces:
            self._by_key.popitem(last=False)

    def drop(self, ns: Namespace) -> None:
        """Forget ``ns``'s projection — the lookup then reports ABSENT for its items again."""
        self._by_key.pop(projection_key(ns), None)

    def __len__(self) -> int:
        return len(self._by_key)

    def centrality_for(self, item: MemoryItem) -> float | None:
        """``cen(m)`` for one item, or ``None`` for "the graph has nothing to say".

        ``None`` (ABSENT) and ``0.0`` (PRESENT and peripheral) are different answers and are kept
        different on purpose — see :meth:`~mu_engine.lifecycle.salience.SalienceStrategy.score`:

        * ``None`` — either no projection has been refreshed for this item's namespace (a fresh
          install, a disabled service, a truncated pass, an LTM tier that is not configured at
          all), or the item carries no (subject, object) triple, so it is not IN the entity graph
          (an unstructured capture turn; ``falkor_ltm.py:366-367`` skips exactly these when
          materializing entity edges). Neither is evidence of peripherality, so neither may
          penalise the item.
        * ``0.0`` — the graph WAS consulted and this fact's endpoints are both at or below the hub
          floor. That is real evidence, and it lowers S(m) as such.

        ``max`` over the two endpoints, not mean: ``god_nodes`` ranks NODES, and a fact touching a
        hub is structurally load-bearing even when its other endpoint is a leaf. Averaging would
        let one leaf halve a genuine hub attachment.

        Purely a dict lookup and float arithmetic: no I/O, no await, no clock. That is what lets
        the sweep gate call it once per item without doing per-item graph I/O.
        """
        if not (item.subject and item.object):
            return None
        adjacency = self._by_key.get(projection_key(item.namespace))
        if adjacency is None:
            return None
        cfg = self._settings
        raw = max(
            _hub_degree(adjacency, _entity_key(item.subject), cfg.min_hub_degree),
            _hub_degree(adjacency, _entity_key(item.object), cfg.min_hub_degree),
        )
        return min(raw / cfg.degree_cap, 1.0)


class CentralityService:
    """Refresh ONE namespace's LTM entity-degree projection and publish it into the index.

    Stateless across calls apart from the index it writes into: the adjacency map is built fresh
    inside :meth:`refresh` from that one namespace's read and is never accumulated. See the module
    docstring's namespace-scoping paragraph for why that is a tenancy requirement, not a style
    choice.
    """

    def __init__(
        self,
        *,
        ltm: LtmCentralityStorePort,
        index: CentralityIndex,
        settings: CentralitySettings | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self._ltm = ltm
        self._index = index
        self._settings = settings or CentralitySettings()
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()

    async def refresh(self, ns: Namespace) -> CentralityReport:
        """Project ``ns``'s entity graph and publish it, or report why it was not published.

        Wrapped in the central observability envelope ``RetentionService.sweep`` already
        established: a content-free span, latency (always) + error (on failure) metrics, a
        content-free audit row. ``CancelledError`` propagates and is never counted as a failure
        (DEV-STANDARDS rule 1).
        """
        started = time.perf_counter()
        with self._tracer.span(_OP, attributes={"ns": ns.to_prefix()}):
            try:
                report = await self._refresh(ns)
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._metrics.inc(_ERROR_METRIC, labels={"operation": _OP})
                raise
            finally:
                self._metrics.observe(
                    _LATENCY_METRIC, time.perf_counter() - started, labels={"operation": _OP}
                )
        self._audit.record(
            TraceScope(correlation_id=ns.to_prefix()),
            operation=_OP,
            outcome=_outcome(report),
            tier="ltm",
            visibility=ns.visibility.value,
            counts={
                "evaluated": report.evaluated,
                "scored": report.scored,
                "entities": report.entities,
            },
        )
        return report

    async def _refresh(self, ns: Namespace) -> CentralityReport:
        cfg = self._settings
        if not cfg.enabled:
            _log.info("centrality_disabled", ns=ns.to_prefix())
            return CentralityReport(skipped=True)

        facts, truncated = await self._read_projection(ns, cfg)
        if truncated:
            # Publishing a lower bound would manufacture false "peripheral" evidence that
            # penalises S(m), and which items got under-counted would depend purely on sweep
            # timing (`graph_recall` orders `valid_at DESC`). Withhold instead: the namespace
            # keeps whatever projection it already had, and an item with none stays ABSENT.
            _log.warning(
                "centrality_projection_truncated_withheld",
                ns=ns.to_prefix(),
                limit=cfg.max_facts_per_pass,
            )
            return CentralityReport(evaluated=len(facts), truncated=True)

        adjacency = _project(facts)
        self._index.publish(ns, adjacency)
        return CentralityReport(
            evaluated=len(facts),
            scored=sum(1 for item in facts if item.subject and item.object),
            entities=len(adjacency),
            published=True,
        )

    async def _read_projection(
        self, ns: Namespace, cfg: CentralitySettings
    ) -> tuple[list[MemoryItem], bool]:
        """Every ACTIVE, still-valid fact in ``ns`` — the ONLY store call this service makes, and
        the only source a projection is ever built from.

        Reads ``max_facts_per_pass + 1`` deliberately: ``len(hits) > max_facts_per_pass`` is then
        an EXACT truncation test, where the more obvious ``len(hits) >= limit`` reports truncation
        falsely for a namespace holding exactly ``limit`` facts.

        ``subject``/``predicate`` are left ``None`` so this is the full namespace enumeration;
        ``session_scope=None`` federates the user's sessions to match the user-grained ``:Entity``
        sub-graph (module docstring); no ``caller_identity_set`` is passed because a sweep has no
        caller — see the module docstring's SHARED-plane paragraph.
        """
        probe = cfg.max_facts_per_pass + 1
        async with asyncio.timeout(cfg.read_timeout_s):
            hits = await self._ltm.graph_recall(ns, limit=probe, session_scope=None)
        if len(hits) > cfg.max_facts_per_pass:
            return [hit.item for hit in hits[: cfg.max_facts_per_pass]], True
        return [hit.item for hit in hits], False


def _outcome(report: CentralityReport) -> str:
    if report.skipped:
        return "skipped"
    if report.truncated:
        return "truncated"
    return "ok"
