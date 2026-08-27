"""The Ebbinghaus forgetting-curve CONSTANTS — the one home for the curve's knobs.

Authority: ``memory-health-pinning-spec.md`` §4 (lines 200-232) + §8 (line 365).

**Spec gap, recorded rather than papered over.** §8 line 365 says the curve constants
(``demote_retention`` / ``min_strength`` / ``base_strength``) are *"read from ``DemotionSettings``
(memory-layer §13)"* so that health *"does NOT redefine the curve constants"*. **There is no
``DemotionSettings`` in this codebase, and none of the three knobs exists anywhere.** The shipped
demotion gate is ``LifecycleSettings.demote_mtm`` over
:class:`~mu_engine.lifecycle.salience.SalienceStrategy`, which is a *recency half-life on
``created_at``* — a different curve on a different clock basis from the Ebbinghaus retention
``R(Δt) = exp(-Δt/S)`` measured from the last scoring instant. So the spec's one-line thesis
(line 7, *"nothing new is scored"*) and §4 line 204 (*"the health lens does NOT re-implement the
curve"*) are not satisfiable as written: the curve had to be introduced.

It is introduced HERE, once, so that when the demotion strategy migrates to the Ebbinghaus form
(memory-layer §6.2) it consumes this same subtree rather than a second copy — which is what §8's
"one place the curve lives" intent actually requires. Reported for a spec amendment.

Convention: a plain frozen ``BaseModel`` "tracked seam", identical to
``mu_engine.lifecycle.settings.LifecycleSettings`` and ``mu_engine.services.settings
.IngestSettings`` — not yet a ``Settings`` root sibling (that wiring is a composition-root task;
CANONICAL §7.27 requires it and it is reported as an outstanding delta).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ForgettingCurveSettings"]


class ForgettingCurveSettings(BaseModel):
    """Constants of ``R(Δt) = exp(-Δt_days / max(S, min_strength))`` (spec §4).

    Ported from MemoryBank ``forget_memory.py`` — the full PORT-FROM citation, the two deliberate
    deviations, and the upstream precedence bug live on
    :meth:`~mu_contracts.domain.model.memory.SalienceComponents.strength_retention`, which is the
    single implementation of the curve. This class carries only its knobs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Starting strength S (in DAYS) of a once-seen memory. Upstream's literal default is
    #: ``memory_strength = dialog.get('memory_strength', 1)`` (``forget_memory.py:106``) with a
    #: ``/5`` folded into the curve body (``:36``); deviation 1 folds that constant into the
    #: STRENGTH UNITS instead, which is what makes this value 5.0 rather than 1.
    base_strength: float = Field(default=5.0, gt=0.0)
    #: Divisor floor. Guards ``exp(-Δt/S)`` against a zero/negative recorded strength — a
    #: division-by-zero or an inverted curve, both of which would be silent data corruption.
    min_strength: float = Field(default=1.0, gt=0.0)
    #: ``R < this`` is what the sweep will act on. The lower edge of the ``DECAYING`` band, so
    #: the band means exactly "what the next sweep would archive" (spec §4 line 214).
    demote_retention: float = Field(default=0.3, ge=0.0, le=1.0)
