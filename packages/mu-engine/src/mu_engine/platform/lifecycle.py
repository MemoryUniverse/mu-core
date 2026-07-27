"""``LifecycleManager`` — topological start / health-gated deps / reverse-order rollback
(platform-layer0-spec §13.2; reused verbatim from platform-design §3.2).

Contract (spec §13.2):
  * start in TOPOLOGICAL order (deterministic tie-break);
  * a ``required=True`` dep that fails ``start()`` OR reports ``healthy=False`` ROLLS BACK — every
    adapter started so far is closed in reverse order — then the failure re-raises (fail-loud);
  * a ``required=False`` dep that fails DEGRADES (named in :attr:`ReadinessReport.degraded`) rather
    than blocking startup (e.g. a missing optional MTM);
  * a dep that is healthy-but-``degraded`` starts AND is reported degraded (contracts
    ``HealthStatus.degraded``);
  * ``close()`` closes the started adapters in REVERSE order.

Fully async + cancellation-safe (DEV-STANDARDS rule 1): a ``CancelledError`` during start triggers
the same reverse-order rollback and propagates.

The readiness contract is the contracts DTO ``mu_contracts.ports.health.HealthStatus`` (spec §8.1);
this module owns the composition-side ``LifecycleAdapter`` shape + the manager.

Home: ``mu-engine`` (``bootstrap/lifecycle.py`` per spec §0.2; placed under ``platform/`` here).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from mu_contracts.ports.health import HealthStatus

__all__ = [
    "LifecycleAdapter",
    "LifecycleManager",
    "LifecycleSpec",
    "ReadinessReport",
]


@runtime_checkable
class LifecycleAdapter(Protocol):
    """Anything the manager starts/health-checks/closes (bus, stores, workflow runner). Matches the
    ``EventBusPort`` lifecycle triple (spec §8.1) so ports plug in directly."""

    async def start(self) -> None: ...
    async def health(self) -> HealthStatus: ...
    async def close(self) -> None: ...


class LifecycleSpec(BaseModel):
    """One managed component + its policy. ``depends_on`` names specs that must start first."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    adapter: LifecycleAdapter
    required: bool = True
    depends_on: tuple[str, ...] = ()


class ReadinessReport(BaseModel):
    """The outcome of ``start()``: what came up, and which components degraded (spec §13.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    started: tuple[str, ...]
    degraded: tuple[str, ...]


def _topo_order(specs: dict[str, LifecycleSpec]) -> list[str]:
    """Deterministic Kahn topological sort by ``depends_on`` (stable, sorted tie-break)."""
    indeg: dict[str, int] = dict.fromkeys(specs, 0)
    for spec in specs.values():
        for dep in spec.depends_on:
            if dep not in specs:
                raise ValueError(f"lifecycle dep not registered: {spec.name} -> {dep}")
            indeg[spec.name] += 1
    ready = sorted(name for name, deg in indeg.items() if deg == 0)
    order: list[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for other in sorted(specs):
            if name in specs[other].depends_on:
                indeg[other] -= 1
                if indeg[other] == 0:
                    ready.append(other)
        ready.sort()
    if len(order) != len(specs):
        raise ValueError("lifecycle dependency cycle detected")
    return order


class LifecycleManager:
    """Owns the ordered start/stop of the app-scope adapters (spec §13.2)."""

    def __init__(self, specs: list[LifecycleSpec]) -> None:
        by_name: dict[str, LifecycleSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"duplicate lifecycle spec: {spec.name}")
            by_name[spec.name] = spec
        self._specs = by_name
        self._order = _topo_order(by_name)
        self._started: list[str] = []

    async def start(self) -> ReadinessReport:
        """Start in topological order; roll back on a required failure; degrade an optional one."""
        degraded: list[str] = []
        try:
            for name in self._order:
                spec = self._specs[name]
                try:
                    await spec.adapter.start()
                    status = await spec.adapter.health()
                except Exception:
                    if spec.required:
                        raise
                    degraded.append(name)
                    continue
                if status.healthy:
                    self._started.append(name)
                    if status.degraded:
                        degraded.append(name)
                elif spec.required:
                    raise RuntimeError(f"required component unhealthy: {name}")
                else:
                    degraded.append(name)
        except BaseException:
            # required-dep failure OR cancellation -> reverse-order rollback, then propagate.
            await self._close_started()
            raise
        return ReadinessReport(started=tuple(self._started), degraded=tuple(degraded))

    async def close(self) -> None:
        """Close started adapters in reverse order (spec §13.2)."""
        await self._close_started()

    async def _close_started(self) -> None:
        while self._started:
            name = self._started.pop()
            await self._specs[name].adapter.close()
