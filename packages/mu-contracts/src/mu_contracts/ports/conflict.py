"""ConflictEdgeReader — the bounded, content-free conflict adjacency reader.

Authority: storage-schema-rowmapper-spec.md §1.4. Reads ``conflict_records`` WHERE
``namespace_prefix=$ns AND member_ids && $memory_ids AND state != DISMISSED`` — scoped by
``to_prefix()`` and bounded by the health-view page's ``memory_ids`` (member-intersection),
NEVER a full-partition scan; the assessor stays pure/deterministic (memory-health §3.3).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.conflict import ConflictEdges
from mu_contracts.domain.model.memory import Namespace

__all__ = ["ConflictEdgeReader"]


@runtime_checkable
class ConflictEdgeReader(Protocol):
    async def edges_for(self, ns: Namespace, memory_ids: frozenset[str]) -> ConflictEdges: ...
