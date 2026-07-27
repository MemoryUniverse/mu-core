"""HealthStatus — the shared readiness contract every LifecycleAdapter reports.

Authority: platform-layer0-spec §8.1 (``EventBusPort.health() -> HealthStatus``), §13.2
(``LifecycleManager`` health-gated ``required`` deps). A tiny content-free DTO so a container
can topologically gate startup and surface degraded optional deps."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["HealthStatus"]


class HealthStatus(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    component: str = Field(min_length=1)
    healthy: bool
    degraded: bool = False
    detail: str | None = None  # named diagnostic; never memory content
