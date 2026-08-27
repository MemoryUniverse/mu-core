"""``PendingConflictEdgeReader`` — the shipped :class:`ConflictEdgeReader` over the real
conflict inbox.

Authority: ``memory-health-pinning-spec.md`` §3.3 / §4 line 229 (``conflict_edges
.unresolved_for`` / ``.pin_blocked_for``) · ``storage-schema-rowmapper-spec.md`` §1.4
(``ConflictEdges``/``ConflictEdgeRow``, the bounded content-free projection).

**Why this exists.** ``ConflictEdges`` is the ONLY input that can raise ``CONFLICTING`` and
``LOW_CONFIDENCE`` in :class:`~mu_engine.services.health.assessor.HeuristicV1Assessor`, and
``ConflictRecord.pin_blocked`` — the §6.4 "a new fact contradicts a memory you pinned" signal that
``ConflictAdjudicator`` writes — reaches a user only through this projection. Without an
implementer both were fed exclusively by test literals: the flag existed, nothing could ever set
it from real data. This adapter closes that seam using the ONE bounded query the shipped
``ConflictRecordRepository`` port offers.

**Scope, stated plainly (a real limitation, not a rounding error).**
``ConflictRecordRepository.pending(ns)`` returns the ``MANUAL_PENDING`` inbox only. That is
exactly the set ``ConflictAdjudicator._park`` writes — every withheld verdict, pin-blocked ones
included — so the §6.4 surface is complete. It is NOT every state
``ConflictEdges.unresolved_for`` recognises: a ``DETECTED`` or ``REOPENED`` record would also be
"unresolved", and no port method can currently fetch those. Reported as a port gap rather than
worked around with an unbounded scan (spec §3.1: "NEVER unbounded"). When
``ConflictRecordRepository`` gains a member-intersection query, this class narrows to it and the
assessor is untouched.

**Content-free.** Ids, enums, a confidence float and a boolean — the conflicting text never
enters the projection, exactly as ``ConflictRecord`` itself never carries it.
"""

from __future__ import annotations

from mu_contracts.domain.model.conflict import ConflictEdgeRow, ConflictEdges, ConflictRecord
from mu_contracts.domain.model.memory import Namespace
from mu_contracts.ports.governance import ConflictRecordRepository

__all__ = ["PendingConflictEdgeReader"]


class PendingConflictEdgeReader:
    """Project the namespace's parked conflict records onto one health-view page."""

    def __init__(self, records: ConflictRecordRepository) -> None:
        self._records = records

    async def edges_for(self, ns: Namespace, memory_ids: frozenset[str]) -> ConflictEdges:
        """One adjacency row per page member that appears in a parked conflict.

        Scoped by the caller's ``ns`` (the repository keys on ``to_prefix()``) and bounded by the
        page's own ids — the member-intersection contract of ``ConflictEdgeReader``. An empty page
        does no I/O at all.

        A memory in SEVERAL parked conflicts collapses to ONE row, because ``ConflictEdges`` is a
        ``dict[str, ConflictEdgeRow]``. The winner is chosen deterministically — most recently
        detected, ``conflict_id`` breaking a timestamp tie — so two replicas assessing the same
        data render the same view, and a ``pin_blocked`` record is never hidden behind an
        arbitrary dict-ordering choice: see :meth:`_prefer`.
        """
        if not memory_ids:
            return ConflictEdges()
        chosen: dict[str, ConflictRecord] = {}
        for record in await self._records.pending(ns):
            members = frozenset(record.member_ids)
            for memory_id in members & memory_ids:
                incumbent = chosen.get(memory_id)
                if incumbent is None or self._prefer(record, incumbent):
                    chosen[memory_id] = record
        return ConflictEdges(
            rows_by_memory={
                memory_id: ConflictEdgeRow(
                    memory_id=memory_id,
                    peer_ids=frozenset(record.member_ids) - {memory_id},
                    conflict_id=record.conflict_id,
                    state=record.state,
                    pin_blocked=record.pin_blocked,
                    detected_confidence=record.detected_confidence,
                )
                for memory_id, record in chosen.items()
            }
        )

    @staticmethod
    def _prefer(candidate: ConflictRecord, incumbent: ConflictRecord) -> bool:
        """True iff ``candidate`` should represent the memory instead of ``incumbent``.

        A ``pin_blocked`` record wins outright: it is the one the owner must act on (§6.4), and
        losing it to a merely-newer ordinary conflict would drop the *"a new fact contradicts a
        memory you pinned"* signal entirely. Otherwise: newer ``detected_at``, then the larger
        ``conflict_id`` — total and deterministic, never insertion order.
        """
        if candidate.pin_blocked != incumbent.pin_blocked:
            return candidate.pin_blocked
        if candidate.detected_at != incumbent.detected_at:
            return candidate.detected_at > incumbent.detected_at
        return candidate.conflict_id > incumbent.conflict_id
