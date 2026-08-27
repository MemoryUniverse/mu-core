"""Pure planning layer over :mod:`naming`'s per-item tenancy recovery — deliberately split out
of the I/O modules (``mtm_migration.py`` / ``graph_migration.py``) so the "does this point/node
resolve, and to which target partition" decision is testable with plain fakes (dicts), with no
Qdrant/FalkorDB client in the loop at all. The I/O modules call these functions once per scrolled
page; nothing here ever opens a connection.

Content-free (CLAUDE.md rule 3): a plan carries POINT IDS / NODE DISPLAY KEYS and PARTITION NAMES
only — never payload/prop content.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from mu_engine.storage.migrations.naming import (
    resolve_tenancy_from_graph_props,
    resolve_tenancy_from_mtm_payload,
)

__all__ = ["GraphMigrationPlan", "MtmMigrationPlan", "plan_graph_migration", "plan_mtm_migration"]


class MtmMigrationPlan(BaseModel, frozen=True):
    """The outcome of resolving one PAGE (or the whole) of a legacy MTM collection's points."""

    source_collection: str
    dim: int
    targets: dict[str, tuple[str, ...]]  # target collection name -> point ids routed there
    unresolved_point_ids: tuple[str, ...]  # could not recover (org, workspace) — SKIPPED

    @property
    def resolved_count(self) -> int:
        return sum(len(ids) for ids in self.targets.values())

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved_point_ids)


def plan_mtm_migration(
    *,
    source_collection: str,
    dim: int,
    points: Iterable[tuple[str, Mapping[str, Any]]],
) -> MtmMigrationPlan:
    """Route each ``(point_id, payload)`` pair to its recovered target collection, or to the
    unresolved bucket when :func:`~mu_engine.storage.migrations.naming.
    resolve_tenancy_from_mtm_payload` returns ``None``. Pure — takes payload dicts, not a client;
    a page-at-a-time caller can invoke this once per scrolled page and merge the resulting plans."""
    targets: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for point_id, payload in points:
        tenancy = resolve_tenancy_from_mtm_payload(payload)
        if tenancy is None:
            unresolved.append(point_id)
            continue
        target = tenancy.target_mtm_collection_name(dim)
        targets.setdefault(target, []).append(point_id)
    return MtmMigrationPlan(
        source_collection=source_collection,
        dim=dim,
        targets={name: tuple(ids) for name, ids in targets.items()},
        unresolved_point_ids=tuple(unresolved),
    )


class GraphMigrationPlan(BaseModel, frozen=True):
    """The outcome of resolving one PAGE (or the whole) of a legacy graph's nodes. ``targets``
    keys are physical FalkorDB graph names; values are content-free node DISPLAY keys
    (``"{label}:{id-or-canonical_name}"`` — see ``graph_migration.py``'s node-key builder), never
    the node's own content-bearing properties."""

    source_graph: str
    targets: dict[str, tuple[str, ...]]
    unresolved_node_keys: tuple[str, ...]

    @property
    def resolved_count(self) -> int:
        return sum(len(keys) for keys in self.targets.values())

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved_node_keys)


def plan_graph_migration(
    *,
    source_graph: str,
    nodes: Iterable[tuple[str, Mapping[str, Any]]],
) -> GraphMigrationPlan:
    """Route each ``(node_display_key, props)`` pair to its recovered target graph, or to the
    unresolved bucket when :func:`~mu_engine.storage.migrations.naming.
    resolve_tenancy_from_graph_props` returns ``None``. Pure — takes prop dicts, not a client."""
    targets: dict[str, list[str]] = {}
    unresolved: list[str] = []
    for node_key, props in nodes:
        tenancy = resolve_tenancy_from_graph_props(props)
        if tenancy is None:
            unresolved.append(node_key)
            continue
        target = tenancy.target_graph_name()
        targets.setdefault(target, []).append(node_key)
    return GraphMigrationPlan(
        source_graph=source_graph,
        targets={name: tuple(keys) for name, keys in targets.items()},
        unresolved_node_keys=tuple(unresolved),
    )
