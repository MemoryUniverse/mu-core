"""Conflict vocabulary — ConflictRecord (content-free) + the ConflictEdges adjacency view.

Authority: engine-core-spec.md §7.6 (``ConflictRecord``, ``ConflictState``, the sticky
``resolution_origin``), storage-schema-rowmapper-spec.md §1.4 (``ConflictEdges`` /
``ConflictEdgeRow``, the bounded content-free projection handed to the pure ``HealthAssessor``).

``conflict_id = sha256(ns.to_prefix | sorted(member_ids) | predicate_key)[:24]`` is idempotent.
A ``resolution_origin="manual"`` winner is STICKY — a later automatic contradicting delta
``REOPEN``s the conflict, it never silently flips (CANONICAL §7.5 X5 / conflict-async §7).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.memory import Namespace

__all__ = [
    "ConflictEdgeRow",
    "ConflictEdges",
    "ConflictRecord",
    "ConflictState",
    "ResolutionOrigin",
]


class ConflictState(StrEnum):
    """The conflict lifecycle (engine-core §7.6). ``ConflictLifecyclePolicy`` enforces legal
    transitions (illegal → ``IllegalConflictTransitionError``)."""

    DETECTED = "detected"
    AUTO_RESOLVED = "auto_resolved"
    MANUAL_PENDING = "manual_pending"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    REOPENED = "reopened"


class ResolutionOrigin(StrEnum):
    """Who/what decided the winner. ``MANUAL`` is sticky in per-replica re-derivation
    (CANONICAL §7.5 X5)."""

    AUTO = "auto"
    MANUAL = "manual"
    SYSTEM_DEGRADED = "system_degraded"


class ConflictRecord(BaseModel):
    """The content-free conflict record (engine-core §7.6). Holds member ids / hashes / enums /
    counts / timestamps only — never the conflicting text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conflict_id: str = Field(min_length=1)  # sha256(ns.to_prefix|sorted(member_ids)|predicate)[:24]
    namespace: Namespace
    member_ids: tuple[str, ...] = Field(min_length=1)  # the conflicting MemoryItem ids
    predicate_key: str = Field(min_length=1)
    method: str = Field(min_length=1)  # detection method (e.g. "nli", "bitemporal")
    detected_confidence: float = Field(ge=0.0, le=1.0)
    proposed_winner_id: str | None = None
    state: ConflictState = ConflictState.DETECTED
    resolution_kind: str | None = None  # SUPERSEDE | REFINE | REINSTATE | ... (named)
    resolved_winner_id: str | None = None
    resolution_origin: ResolutionOrigin | None = None
    resolved_by: str | None = None  # principal id (manual) — content-free
    policy_snapshot: dict[str, str] = Field(default_factory=dict)  # content-free scalar snapshot
    detected_at: datetime
    resolved_at: datetime | None = None
    superseded_valid_at: datetime | None = None  # the loser's bi-temporal close, when resolved
    #: True iff the automatic winner-picker was REFUSED because the item it would have made the
    #: loser is ``pinned`` (memory-health §6.4; CANONICAL §7.17 item 4a(b) — a pinned item is
    #: never the auto-supersede/quarantine loser). The pinned item stays ACTIVE and the conflict
    #: is PARKED rather than lost; ``ConflictEdgeRow.pin_blocked`` projects this onto the
    #: health-view page, where it surfaces as ``MemoryHealthFlag.CONFLICTING``.
    pin_blocked: bool = False


class ConflictEdgeRow(BaseModel):
    """One adjacency fact, content-free (storage-schema §1.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str = Field(min_length=1)
    peer_ids: frozenset[str]  # the other members of the conflict
    conflict_id: str = Field(min_length=1)
    state: ConflictState
    pin_blocked: bool = False  # a pin is blocked by this unresolved conflict (memory-health §5.2)
    #: The conflict's ``detected_confidence``, carried onto the adjacency row so the pure
    #: ``HealthAssessor`` can raise ``LOW_CONFIDENCE`` without a per-item DB round-trip.
    #: Confidence is a property of the CONFLICT (``ConflictRecord.detected_confidence`` /
    #: ``AdjudicationVerdict.confidence``), never of the memory — there is no ``MemoryItem
    #: .confidence`` field and memory-health §4 line 228's ``item.confidence`` has no referent.
    detected_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ConflictEdges(BaseModel):
    """Immutable per-namespace conflict adjacency snapshot handed to the pure assessor
    (storage-schema §1.4). Loaded once per health-view page, then queried per-item in memory —
    no per-item DB round-trip inside the assessor, which stays pure/deterministic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows_by_memory: dict[str, ConflictEdgeRow] = Field(default_factory=dict)

    def unresolved_for(self, memory_id: str) -> bool:
        row = self.rows_by_memory.get(memory_id)
        return row is not None and row.state in {
            ConflictState.DETECTED,
            ConflictState.MANUAL_PENDING,
            ConflictState.REOPENED,
        }

    def pin_blocked_for(self, memory_id: str) -> bool:
        row = self.rows_by_memory.get(memory_id)
        return row is not None and row.pin_blocked

    def confidence_for(self, memory_id: str) -> float | None:
        """The conflict confidence attached to ``memory_id``, or ``None`` if it is in no
        conflict (or the row carries none). The source of ``MemoryHealthEntry.confidence``."""
        row = self.rows_by_memory.get(memory_id)
        return None if row is None else row.detected_confidence

    def peers_for(self, memory_id: str) -> tuple[str, ...]:
        """The other members of ``memory_id``'s conflict, sorted for determinism — ids only
        (content-free even here: the pair is a link, not text). Empty when unconflicted."""
        row = self.rows_by_memory.get(memory_id)
        return () if row is None else tuple(sorted(row.peer_ids))
