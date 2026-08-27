"""``HealthSettings`` — the memory-health central-config subtree.

Authority: ``memory-health-pinning-spec.md`` §8 (lines 348-366). Field names and defaults are the
spec's verbatim, plus :attr:`HealthSettings.curve` (see
:mod:`mu_engine.services.health.forgetting` for why that had to be introduced).

DEV-STANDARDS rule 3: no threshold, band edge, page size or TTL is ever a literal at a call site.
**No model field** — health calls no LLM and no embedder, and saying so here is deliberate: it
bars a future model dependency from creeping in (spec §8 line 366 / §3.2 line 183).

"Tracked seam" convention (same as ``LifecycleSettings``): a plain frozen ``BaseModel`` taken as
an explicit constructor argument. CANONICAL §7.27 lists ``HealthSettings``/``PinSettings`` as
required ``Settings`` root siblings; that composition-root wiring (and the ``MU_HEALTH__`` env
prefix it implies) is not done here and is reported as an outstanding delta.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mu_engine.services.health.forgetting import ForgettingCurveSettings

__all__ = ["HealthSettings"]


class HealthSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: ``health_registry`` key; fail-loud on a miss (spec §3.3 / memory-layer §10).
    strategy: str = "heuristic_v1"
    #: ``last_seen`` older than this AND a low recency component -> STALE. BOTH are required
    #: (spec line 352: *"AND recency component below this"*) so a merely old but still-reinforced
    #: memory is never called stale.
    stale_after_h: float = Field(default=168.0, gt=0.0)
    stale_recency: float = Field(default=0.2, ge=0.0, le=1.0)
    #: Upper edge of the DECAYING band; the lower edge is ``curve.demote_retention``.
    stale_retention_band: float = Field(default=0.6, ge=0.0, le=1.0)
    #: C < this -> LOW_CONFIDENCE. Kept in sync with ``ConflictSettings.refine_confidence``.
    low_confidence_below: float = Field(default=0.5, ge=0.0, le=1.0)
    #: The bounded ``enumerate`` window — the view is NEVER an unbounded partition scan.
    page_size: int = Field(default=50, ge=1)
    #: Default surface is at-risk only.
    include_healthy: bool = False
    #: Health-view warm-cache TTL. Declared here because it is the projection's knob; the cache
    #: itself is a surface-layer concern and is not built in this slice.
    warm_ttl_s: float = Field(default=300.0, gt=0.0)

    curve: ForgettingCurveSettings = Field(default_factory=ForgettingCurveSettings)
