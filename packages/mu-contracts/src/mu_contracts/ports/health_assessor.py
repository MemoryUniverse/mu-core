"""``HealthAssessor`` — the pure, registered health-classification strategy port.

Authority: ``docs/superpowers/design/memory-health-pinning-spec.md`` §3.3 (lines 185-196).

**Module-name deviation (recorded, not silent).** Spec line 188 puts this Protocol in
``ports/health.py``. That module already exists in this package and owns an unrelated concept —
``HealthStatus``, the ``LifecycleAdapter`` readiness DTO (platform-layer0-spec §8.1). Two
different "health" vocabularies in one module would make every import site ambiguous, so the
memory-health strategy port lives here under an unambiguous name. The spec needs the amendment.

**Purity is the contract.** ``assess`` is synchronous, takes its instant as an argument (never
calls a clock), performs no I/O, and reads a pre-loaded ``ConflictEdges`` snapshot rather than a
reader port — so a full flag matrix is unit-testable with zero infra, exactly like every other
memory-layer strategy. An implementation that awaited anything would make
``MemoryHealthService.assess``'s CQRS read-purity unverifiable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.conflict import ConflictEdges
from mu_contracts.domain.model.health import MemoryHealthFlag
from mu_contracts.domain.model.memory import MemoryItem

__all__ = ["HealthAssessor"]


@runtime_checkable
class HealthAssessor(Protocol):
    """Classify ONE item into health flags. Registered under :attr:`key` in ``health_registry``
    (mu-engine), fail-loud on an unknown key — a future ``learned_v1`` registers without editing
    :class:`~mu_engine.services.health.service.MemoryHealthService`."""

    key: str

    def assess(
        self, item: MemoryItem, *, now: datetime, conflict_edges: ConflictEdges
    ) -> frozenset[MemoryHealthFlag]: ...

    def retention(self, item: MemoryItem, *, now: datetime) -> float:
        """R(Δt) for ``item`` — the SAME number the ``DECAYING`` band is cut from, exposed so the
        view can show the user what the assessor decided on rather than a second, re-derived
        value (memory-health §4 line 214: *"the health view surfaces the same retention value
        this decision reads"*)."""
        ...
