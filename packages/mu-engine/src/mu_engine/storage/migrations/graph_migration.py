"""Blue-green migration off a legacy-named FalkorDB graph (AD-8).

Same shape as ``mtm_migration.py`` (§1.7's recipe, applied to the naming defect): create the
new-named ("green") graph, re-MERGE every node whose tenancy resolves, re-create the edges
between migrated nodes, verify counts, THEN drop the old ("blue") graph — never mutate a graph in
place. Idempotent (every write is a Cypher ``MERGE`` keyed on the node's own stable identity, never
a bare ``CREATE``), so re-running after a partial/interrupted run is safe.

**Per-NODE target resolution**, not per-graph — the same reason as the MTM module: AD-8's raw
``org__workspace`` join is exactly what let two different orgs collide into ONE physical graph, so
this migration cannot trust the source graph's name to mean any one tenant. Each node is routed
independently via :func:`~mu_engine.storage.migrations.naming.resolve_tenancy_from_graph_props`. A
node whose tenancy cannot be recovered is SKIPPED and counted, never guessed.

**Three node shapes, three identities** (`falkor_ltm.py`'s own MERGE keys, reused so a copied
node round-trips to the SAME identity a live write would produce):
``:Memory``/``:Artifact`` -> ``(namespace, id)``; ``:Entity`` -> ``(namespace, entity_uid)``
(the adapter's own live writes MERGE ``:Entity`` on ``(namespace, canonical_name)``, but
``entity_uid`` is a 1:1 stable alias of that same identity minted once at first resolution
(`falkor_ltm.py::_merge_entity`) — using it here keeps this module's own reporting/logging
content-free, since ``canonical_name`` is extracted memory content and ``entity_uid`` is a
structural id, CLAUDE.md rule 3). A node matching neither shape has no identity this migration can
safely dedupe by and is treated as unresolved rather than blindly ``CREATE``-d (which would not be
idempotent on a re-run).

Edges are copied in a second pass, matched on the SAME two identities, with the relationship TYPE
re-validated through the adapter's own :func:`~mu_engine.storage.adapters.falkor_ltm.
_sanitize_rel_type` before being interpolated into a query (Cypher has no parameterized rel type;
never trust a value read back from storage as safe to inline without re-checking it).

Dry-run by default (`apply=False`): resolves and reports, writes/deletes nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import structlog
from falkordb.asyncio import FalkorDB
from pydantic import BaseModel

from mu_engine.storage.adapters.falkor_ltm import _sanitize_rel_type, g_query
from mu_engine.storage.migrations.naming import (
    discover_legacy_graph_names,
    resolve_tenancy_from_graph_props,
)
from mu_engine.storage.migrations.planning import plan_graph_migration

__all__ = ["GraphMigrationResult", "migrate_all_legacy_graphs", "migrate_graph"]

_log = structlog.get_logger("mu_engine.storage.migrations.graph_migration")


class GraphMigrationResult(BaseModel, frozen=True):
    """Content-free (CLAUDE.md rule 3): counts and graph NAMES only, never node/edge content."""

    source_graph: str
    nodes_found: int
    nodes_migrated: int
    nodes_unresolved: int
    edges_found: int
    edges_migrated: int
    edges_unresolved: int
    target_graphs: tuple[str, ...]
    applied: bool
    dropped_source: bool


def _node_display_key(internal_id: int, labels: Sequence[str], props: Mapping[str, Any]) -> str:
    """A content-free bucket key for the plan/report — a structural id only, never the entity's
    ``canonical_name`` (extracted memory content)."""
    label = labels[0] if labels else "Node"
    if props.get("id") is not None:
        return f"{label}:{props['id']}"
    if props.get("entity_uid") is not None:
        return f"{label}:{props['entity_uid']}"
    return f"{label}:node{internal_id}"


def _node_pattern(
    alias: str, labels: Sequence[str], props: Mapping[str, Any]
) -> tuple[str, dict[str, Any]] | None:
    """A ``(alias:Label {namespace: $.., <identity-prop>: $..})`` Cypher fragment + its params,
    or ``None`` when neither identity shape (``id`` or ``entity_uid``) is present — the caller
    must then refuse to MERGE/MATCH this node rather than fall back to an unstable pattern."""
    label_clause = ":".join(labels) if labels else ""
    label_part = f":{label_clause}" if label_clause else ""
    if props.get("id") is not None:
        key_prop, key_val = "id", props["id"]
    elif props.get("entity_uid") is not None:
        key_prop, key_val = "entity_uid", props["entity_uid"]
    else:
        return None
    ns_param = f"{alias}_namespace"
    key_param = f"{alias}_{key_prop}"
    pattern = f"({alias}{label_part} {{namespace: ${ns_param}, {key_prop}: ${key_param}}})"
    return pattern, {ns_param: props.get("namespace"), key_param: key_val}


async def _merge_node(graph: Any, labels: Sequence[str], props: Mapping[str, Any]) -> bool:
    """MERGE one copied node into the target graph, keyed on its own stable identity — a re-run
    over the same source node re-writes the SAME target node, never a duplicate."""
    built = _node_pattern("n", labels, props)
    if built is None:
        return False
    clause, params = built
    cypher = f"MERGE {clause} SET n += $props"
    await graph.query(cypher, params={**params, "props": dict(props)})
    return True


async def _merge_edge(
    graph: Any,
    a_labels: Sequence[str],
    a_props: Mapping[str, Any],
    b_labels: Sequence[str],
    b_props: Mapping[str, Any],
    rel_type: str,
    rel_props: Mapping[str, Any],
) -> bool:
    """Re-create one ``(a)-[rel_type]->(b)`` edge in the target graph, matching both endpoints
    by their already-migrated identity. Both endpoints are asserted (``MATCH``, not ``MERGE``) —
    the node pass must have already migrated them; a missing endpoint means something upstream is
    wrong and this returns ``False`` rather than silently creating a bare stub node."""
    a_built = _node_pattern("a", a_labels, a_props)
    b_built = _node_pattern("b", b_labels, b_props)
    if a_built is None or b_built is None:
        return False
    a_clause, a_params = a_built
    b_clause, b_params = b_built
    safe_rel = _sanitize_rel_type(rel_type)
    cypher = f"MATCH {a_clause}, {b_clause} " f"MERGE (a)-[r:{safe_rel}]->(b) SET r += $rel_props"
    await graph.query(cypher, params={**a_params, **b_params, "rel_props": dict(rel_props)})
    return True


async def migrate_graph(db: FalkorDB, source_graph: str, *, apply: bool) -> GraphMigrationResult:  # type: ignore[no-any-unimported]
    """Migrate one legacy graph. ``apply=False`` (the default at the CLI layer) resolves every
    node/edge and reports what WOULD happen; nothing is written or dropped."""
    source = db.select_graph(source_graph)
    node_rows = await g_query(source, "MATCH (n) RETURN id(n), labels(n), properties(n)", {})
    nodes: list[tuple[int, tuple[str, ...], dict[str, Any], str]] = []
    for internal_id, labels, props in node_rows:
        labels_t = tuple(labels)
        props_d = dict(props)
        key = _node_display_key(internal_id, labels_t, props_d)
        nodes.append((internal_id, labels_t, props_d, key))

    plan = plan_graph_migration(
        source_graph=source_graph, nodes=((key, props) for _id, _labels, props, key in nodes)
    )
    _log.info(
        "graph_migration.planned",
        source_graph=source_graph,
        nodes_found=len(nodes),
        nodes_resolved=plan.resolved_count,
        nodes_unresolved=plan.unresolved_count,
        targets=len(plan.targets),
    )

    edge_rows = await g_query(
        source,
        "MATCH (a)-[r]->(b) RETURN id(a), labels(a), properties(a), "
        "id(b), labels(b), properties(b), type(r), properties(r)",
        {},
    )

    if not apply:
        return GraphMigrationResult(
            source_graph=source_graph,
            nodes_found=len(nodes),
            nodes_migrated=0,
            nodes_unresolved=plan.unresolved_count,
            edges_found=len(edge_rows),
            edges_migrated=0,
            edges_unresolved=0,
            target_graphs=tuple(plan.targets),
            applied=False,
            dropped_source=False,
        )

    target_by_internal_id: dict[int, str] = {}
    nodes_migrated = 0
    for internal_id, labels, props, _key in nodes:
        tenancy = resolve_tenancy_from_graph_props(props)
        if tenancy is None:
            continue
        target_name = tenancy.target_graph_name()
        ok = await _merge_node(db.select_graph(target_name), labels, props)
        if ok:
            target_by_internal_id[internal_id] = target_name
            nodes_migrated += 1

    edges_migrated = 0
    edges_unresolved = 0
    for a_id, a_labels, a_props, b_id, b_labels, b_props, rel_type, rel_props in edge_rows:
        target_a = target_by_internal_id.get(a_id)
        target_b = target_by_internal_id.get(b_id)
        if target_a is None or target_b is None or target_a != target_b:
            # either endpoint never resolved, or (should not happen in well-formed data — every
            # edge-creating write in falkor_ltm.py scopes both endpoints to ONE `ns`) the two
            # endpoints resolved to DIFFERENT tenants. Never bridge two target graphs with one
            # edge — that is exactly the cross-tenant merge this migration exists to prevent.
            edges_unresolved += 1
            continue
        ok = await _merge_edge(
            db.select_graph(target_a),
            tuple(a_labels),
            dict(a_props),
            tuple(b_labels),
            dict(b_props),
            rel_type,
            dict(rel_props),
        )
        if ok:
            edges_migrated += 1
        else:
            edges_unresolved += 1

    verified = True
    for target_name in {*target_by_internal_id.values()}:
        expected = sum(1 for t in target_by_internal_id.values() if t == target_name)
        result = await g_query(db.select_graph(target_name), "MATCH (n) RETURN count(n)", {})
        found = int(result[0][0]) if result else 0
        if found < expected:
            verified = False
            _log.warning(
                "graph_migration.verify_short",
                target_graph=target_name,
                expected_at_least=expected,
                found=found,
            )

    dropped_source = False
    if verified and plan.unresolved_count == 0 and edges_unresolved == 0:
        await source.delete()
        dropped_source = True
        _log.info("graph_migration.dropped_source", source_graph=source_graph)
    else:
        _log.info(
            "graph_migration.source_kept",
            source_graph=source_graph,
            verified=verified,
            nodes_unresolved=plan.unresolved_count,
            edges_unresolved=edges_unresolved,
        )

    return GraphMigrationResult(
        source_graph=source_graph,
        nodes_found=len(nodes),
        nodes_migrated=nodes_migrated,
        nodes_unresolved=plan.unresolved_count,
        edges_found=len(edge_rows),
        edges_migrated=edges_migrated,
        edges_unresolved=edges_unresolved,
        target_graphs=tuple(plan.targets),
        applied=True,
        dropped_source=dropped_source,
    )


async def migrate_all_legacy_graphs(db: FalkorDB, *, apply: bool) -> list[GraphMigrationResult]:  # type: ignore[no-any-unimported]
    """Discover every legacy-named graph live in ``db`` and migrate each one."""
    names = await db.list_graphs()
    legacy = discover_legacy_graph_names(str(n) for n in names)
    results = []
    for name in legacy:
        results.append(await migrate_graph(db, name, apply=apply))
    return results
