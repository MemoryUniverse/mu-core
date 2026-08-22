"""D6 real-path verify (REMEDIATION, ARCHITECTURE-CONFORMANCE.md Lens B rows B5/B6/D-4/D-5) —
proves LTM is an actual TRAVERSABLE knowledge graph, not just a flat ``:Memory`` bag, over the
REAL public ``add()``/``consolidate()``/``recall()`` path (embedded ``LocalContainer`` -> real
mu-dev-redis/qdrant/falkordb, offline MiniLM embedder). ZERO mocks/store fakes.

Before this fix: 0 ``:Entity`` nodes, 0 entity-entity edges, ``entity_uids`` DEAD on the MTM
payload — confirmed live in ``ARCHITECTURE-CONFORMANCE.md`` (Lens B). This proves, via a DIRECT
``GRAPH.QUERY`` (bypassing every port-level filter):

  (a) ``:Entity`` nodes exist for "Ada"/"Bo" after ``add("Ada manages Bo")`` + ``consolidate()``;
  (b) an ``(Ada)-[:MANAGES]->(Bo)`` edge exists, bi-temporal (``valid_at``/``invalid_at``);
  (c) a multi-hop traversal (``GraphStorePort.traverse_entities``, the NEW LTM recall arm,
      D-4) for "who is Bo's manager?" returns the underlying "Ada manages Bo" fact — was a
      structural 0/1 miss (the flat ``graph_recall`` seed never resolves a query's OWN entity
      mention, only a whole-partition recency scan);
  (d) a functional-predicate supersession (``works_at``) invalidates the LOSER's entity edge
      too — a stale relation never resurfaces via traversal, mirroring the ``:Memory`` node's
      own invalidate-don't-delete guarantee;
  (e) an extracted proposition's ``provenance_id`` now traces back to its source STM item
      (the context-provenance follow-up one-liner, ``DistillPipeline._fact_to_item``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient

from mu_contracts.config import Settings
from mu_engine.storage.adapters.falkor_ltm import _user_scope_prefix
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id
from mu_local import LocalMemory

from .test_local_roundtrip_int import _teardown

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"


@pytest_asyncio.fixture
async def mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """Heuristic mode (no LLM profile) — the SAME deterministic ``HeuristicSpoExtractor`` every
    other DISTILL/consolidate real-path test in this repo exercises; entity/edge materialization
    is orthogonal to WHICH extractor produced the fact (D3's real-SLM proof already covers the
    LLM-adjudication surface, separately)."""
    memory = LocalMemory(workspace=f"wsd6{uid}", namespace=f"orgd6{uid}", settings=settings)
    try:
        yield memory
    finally:
        await _teardown(settings, f"d6{uid}")
        await memory.aclose()


async def test_entities_and_edge_materialize_and_multihop_traversal(
    mem: LocalMemory, settings: Settings
) -> None:
    container = mem._container
    ns = Namespace(
        org=mem._org,
        workspace=mem._workspace,
        user=_USER,
        session=_SESSION,
        visibility=Visibility.PRIVATE,
    )

    receipt = await mem.add("Ada manages Bo", user=_USER, session=_SESSION, importance_score=0.9)
    await mem.add("Bo owns the Q3 report", user=_USER, session=_SESSION, importance_score=0.9)
    report = await mem.consolidate(user=_USER, session=_SESSION)
    assert report.facts_extracted >= 2, "heuristic extractor found no SPO facts to consolidate"
    assert report.added >= 2, "no facts landed in the LTM graph"

    falkor = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        graph_name = container.ltm.graph_name_for(ns)  # type: ignore[attr-defined]
        g = falkor.select_graph(graph_name)

        # BUG2 FIX (data-quality re-assessment §3): `:Entity` nodes + entity-entity edges are now
        # scoped on the USER-level prefix (`_user_scope_prefix`), not the session-included
        # `ns.to_prefix()` — an entity is a per-USER concept, deduped across every session, so
        # these ground-truth GRAPH.QUERY assertions must key on the SAME scope the adapter writes.
        user_ns = _user_scope_prefix(ns)

        # (a) :Entity nodes exist for Ada + Bo (direct GRAPH.QUERY, bypassing every port filter).
        ent_res = await g.query(
            "MATCH (e:Entity) WHERE e.namespace = $ns RETURN e.canonical_name, e.entity_uid",
            params={"ns": user_ns},
        )
        entities = {row[0]: row[1] for row in ent_res.result_set}
        print(f"\n[D6-VERIFY] :Entity nodes: {entities}")  # noqa: T201
        assert "ada" in entities, ":Entity node for 'Ada' was never materialized"
        assert "bo" in entities, ":Entity node for 'Bo' was never materialized"

        # (b) (Ada)-[:MANAGES]->(Bo) edge exists, bi-temporal, currently active.
        edge_res = await g.query(
            "MATCH (s:Entity {namespace: $ns, canonical_name: 'ada'})"
            "-[r]->(o:Entity {namespace: $ns, canonical_name: 'bo'}) "
            "RETURN type(r), r.valid_at, r.invalid_at, r.memory_id",
            params={"ns": user_ns},
        )
        edges = edge_res.result_set
        print(f"[D6-VERIFY] Ada->Bo edges: {edges}")  # noqa: T201
        assert edges, "no entity-entity edge (Ada)->(Bo) was materialized"
        rel_type, valid_at, invalid_at, edge_mid = edges[0]
        assert rel_type == "MANAGES", f"expected a MANAGES rel type, got {rel_type!r}"
        assert valid_at, "entity edge missing its bi-temporal valid_at stamp"
        assert invalid_at == "", "a freshly-written entity edge must be currently ACTIVE"

        # (c) multi-hop traversal: "who is Bo's manager?" -> the underlying Ada/Bo fact.
        hits = await container.ltm.traverse_entities(
            ns, query="who is Bo's manager?", max_hops=2, limit=10
        )
        contents = [h.item.content for h in hits]
        print(f"[D6-VERIFY] traverse_entities('who is Bo's manager?') -> {contents}")  # noqa: T201
        assert any("Ada" in c and "manages" in c and "Bo" in c for c in contents), (
            "multi-hop traversal for 'who is Bo's manager?' did not surface the Ada/Bo fact "
            f"(0/1 miss) — got {contents!r}"
        )

        # (e) provenance one-liner: the extracted "Ada manages Bo" proposition traces back to
        #     its SOURCE STM item's own provenance_id — never a fresh, disconnected
        #     `prov_<newly-minted-ltm-node-id>` (what `MemoryItem.model_post_init` would mint if
        #     `_fact_to_item` didn't copy it explicitly).
        source_item = await container.stm.get(ns, receipt.memory_id)
        assert source_item is not None, "source STM item missing"
        prov_res = await g.query(
            "MATCH (m:Memory {namespace: $ns}) "
            "WHERE toLower(m.subject) = 'ada' AND m.predicate = 'manages' "
            "RETURN m.id, m.provenance_id",
            params={"ns": ns.to_prefix()},
        )
        rows = prov_res.result_set
        print(f"[D6-VERIFY] Ada/manages LTM node id/provenance_id: {rows}")  # noqa: T201
        assert rows, "extracted Ada/manages fact never landed in the graph"
        ltm_node_id, ltm_provenance_id = rows[0]
        assert (
            ltm_node_id != receipt.memory_id
        ), "sanity: the extracted proposition mints its OWN id (derived_from tracks the source)"
        assert ltm_provenance_id == source_item.provenance_id, (
            "extracted proposition's provenance_id does not trace back to its source STM item "
            f"(got {ltm_provenance_id!r}, source has {source_item.provenance_id!r})"
        )
    finally:
        await falkor.connection.aclose()


async def test_functional_supersede_invalidates_entity_edge(
    mem: LocalMemory, settings: Settings
) -> None:
    """``works_at`` is already one of ``DistillSettings.functional_predicates`` by default
    (``pipelines/distill.py``) — a different object for the SAME (subject, predicate) is a
    genuine supersession. Proves the LOSER's entity-entity edge closes (``invalid_at`` set)
    alongside its ``:Memory`` node, and a post-supersede traversal never returns the stale
    relation as current."""
    container = mem._container
    ns = Namespace(
        org=mem._org,
        workspace=mem._workspace,
        user=_USER,
        session=_SESSION,
        visibility=Visibility.PRIVATE,
    )
    falkor = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        graph_name = container.ltm.graph_name_for(ns)  # type: ignore[attr-defined]
        g = falkor.select_graph(graph_name)
        user_ns = _user_scope_prefix(ns)  # BUG2 FIX — see the other test's comment above

        await mem.add("Ada works at Acme", user=_USER, session=_SESSION, importance_score=0.9)
        report1 = await mem.consolidate(user=_USER, session=_SESSION)
        assert report1.added >= 1

        before = await g.query(
            "MATCH (s:Entity {namespace: $ns, canonical_name: 'ada'})"
            "-[r:WORKS_AT]->(o:Entity {namespace: $ns, canonical_name: 'acme'}) "
            "RETURN r.invalid_at",
            params={"ns": user_ns},
        )
        assert (
            before.result_set and before.result_set[0][0] == ""
        ), "Ada/Acme entity edge must be ACTIVE before the superseding add"

        await mem.add("Ada works at Globex", user=_USER, session=_SESSION, importance_score=0.9)
        report2 = await mem.consolidate(user=_USER, session=_SESSION)
        print(f"\n[D6-VERIFY supersede] report2: {report2}")  # noqa: T201
        assert report2.superseded >= 1, "expected the Acme fact to be superseded by Globex"

        after_loser = await g.query(
            "MATCH (s:Entity {namespace: $ns, canonical_name: 'ada'})"
            "-[r:WORKS_AT]->(o:Entity {namespace: $ns, canonical_name: 'acme'}) "
            "RETURN r.invalid_at",
            params={"ns": user_ns},
        )
        after_winner = await g.query(
            "MATCH (s:Entity {namespace: $ns, canonical_name: 'ada'})"
            "-[r:WORKS_AT]->(o:Entity {namespace: $ns, canonical_name: 'globex'}) "
            "RETURN r.invalid_at",
            params={"ns": user_ns},
        )
        print(f"[D6-VERIFY supersede] Ada->Acme (loser) invalid_at rows: {after_loser.result_set}")  # noqa: T201
        print(  # noqa: T201
            f"[D6-VERIFY supersede] Ada->Globex (winner) invalid_at rows: {after_winner.result_set}"
        )
        assert after_loser.result_set, "loser entity edge disappeared entirely (should be RETAINED)"
        assert after_loser.result_set[0][0] != "", (
            "superseded fact's entity edge was NOT invalidated — a stale relation would still "
            "read as current"
        )
        assert (
            after_winner.result_set and after_winner.result_set[0][0] == ""
        ), "winner entity edge must be currently ACTIVE"

        # a post-supersede traversal for "who does Ada work for?" must surface Globex, never the
        # stale Acme relation as current.
        hits = await container.ltm.traverse_entities(
            ns, query="who does Ada work for?", max_hops=2, limit=10
        )
        contents = [h.item.content for h in hits]
        print(f"[D6-VERIFY supersede] traverse_entities post-supersede -> {contents}")  # noqa: T201
        assert any(
            "Globex" in c for c in contents
        ), "current Globex relation missing from traversal"
        assert not any(
            "Acme" in c and "not" not in c.lower() for c in contents
        ), "superseded Acme relation resurfaced via traversal"
    finally:
        await falkor.connection.aclose()


async def test_entity_uids_backfilled_onto_mtm_payload(
    mem: LocalMemory, settings: Settings
) -> None:
    """D-5 (entity_uids DEAD -> live): once ``upsert_fact`` resolves "Ada"/"Bo" against the LTM
    graph, the already-promoted MTM (Qdrant) point backing the SOURCE ingest id carries
    ``entity_uids`` in its payload — proven with a direct Qdrant point ``retrieve()``, never
    through a recall-shaped read."""
    container = mem._container
    ns = Namespace(
        org=mem._org,
        workspace=mem._workspace,
        user=_USER,
        session=_SESSION,
        visibility=Visibility.PRIVATE,
    )
    receipt = await mem.add("Ada manages Bo", user=_USER, session=_SESSION, importance_score=0.9)
    assert receipt.promoted, "the source message must promote STM->MTM for this proof"
    await mem.consolidate(user=_USER, session=_SESSION)

    qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        dim = container.embedder.dimension
        name = collection_name(ns, dim)
        points = await qdrant.retrieve(
            collection_name=name, ids=[point_id(receipt.memory_id)], with_payload=True
        )
        assert points, "MTM point for the source ingest id is missing"
        payload = points[0].payload or {}
        print(f"\n[D6-VERIFY entity_uids] MTM payload entity_uids: {payload.get('entity_uids')}")  # noqa: T201
        entity_uids = payload.get("entity_uids")
        assert (
            isinstance(entity_uids, list) and len(entity_uids) == 2
        ), f"entity_uids was never backfilled onto the MTM payload: {payload.get('entity_uids')!r}"
    finally:
        await qdrant.close()
