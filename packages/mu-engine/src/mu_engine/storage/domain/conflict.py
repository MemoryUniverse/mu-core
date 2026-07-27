"""``ConflictEdges`` + ``ConflictEdgeRow`` + ``ConflictState`` — the conflict adjacency view.

PIN of ``storage-schema-rowmapper-spec.md §1.4`` (lines 106-136). A read-only, content-free
projection over ``conflict_records`` (§2.9) + the graph ``CONFLICTS_WITH``/``SUPERSEDED_BY``
edges (§4.3): a bounded snapshot loaded once per health-view page, then queried per-item in
memory (no per-item DB round-trip inside the pure assessor).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

__all__ = ["ConflictEdgeRow", "ConflictEdges", "ConflictState"]


class ConflictState(StrEnum):
    """Conflict lifecycle (engine-core §7.6)."""

    DETECTED = "detected"
    AUTO_RESOLVED = "auto_resolved"
    MANUAL_PENDING = "manual_pending"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    REOPENED = "reopened"


class ConflictEdgeRow(BaseModel, frozen=True):
    """One adjacency fact, content-free (spec §1.4 lines 111-116)."""

    memory_id: str
    peer_ids: frozenset[str]
    conflict_id: str
    state: ConflictState
    pin_blocked: bool


class ConflictEdges(BaseModel, frozen=True):
    """Immutable per-namespace conflict adjacency snapshot handed to the pure assessor."""

    rows_by_memory: dict[str, ConflictEdgeRow] = {}

    def unresolved_for(self, memory_id: str) -> bool:
        """True if the memory is in a live-unresolved conflict (memory-health §4)."""
        row = self.rows_by_memory.get(memory_id)
        return row is not None and row.state in {
            ConflictState.DETECTED,
            ConflictState.MANUAL_PENDING,
            ConflictState.REOPENED,
        }

    def pin_blocked_for(self, memory_id: str) -> bool:
        """True if a pin is blocked by an unresolved conflict (memory-health §5.2)."""
        row = self.rows_by_memory.get(memory_id)
        return row is not None and row.pin_blocked
