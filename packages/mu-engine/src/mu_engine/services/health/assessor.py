"""``heuristic_v1`` — the default :class:`~mu_contracts.ports.health_assessor.HealthAssessor`.

Authority: ``memory-health-pinning-spec.md`` §4 (lines 216-232) — the flag rules are that
pseudocode, line for line, with the substitutions the shipped model forces (each one flagged
below rather than silently made).

PURE: no I/O, no clock call, no store, no model. ``now`` arrives as an argument and the conflict
adjacency arrives pre-loaded as a :class:`~mu_contracts.domain.model.conflict.ConflictEdges`
snapshot, so the whole flag matrix is unit-testable with zero infra.

Substitutions forced by the shipped model (all reported for spec amendment):

* **``item.confidence`` does not exist.** Spec line 228 reads ``item.confidence``; there is no
  such field on either ``MemoryItem``. Confidence is a property of the CONFLICT
  (``ConflictRecord.detected_confidence``), so it is read from
  ``conflict_edges.confidence_for(id)``.
* **``item.salience`` is optional and often absent.** ``SalienceComponents`` is computed per
  recall/sweep and never stored, so ``retention`` and the ``STALE`` recency half can be
  genuinely unknown. Unknown resolves to "no decay claim" (``retention = 1.0``) and "no STALE
  flag" — the conservative direction: the lens never invents a risk it cannot evidence.
"""

from __future__ import annotations

from datetime import datetime

from mu_contracts.config.settings import Settings
from mu_contracts.domain.model.conflict import ConflictEdges
from mu_contracts.domain.model.health import MemoryHealthFlag
from mu_contracts.domain.model.memory import MemoryItem, State, Tier
from mu_contracts.ports.health_assessor import HealthAssessor
from mu_engine.platform.registry import Registry
from mu_engine.services.health.settings import HealthSettings

__all__ = ["HEURISTIC_V1", "HeuristicV1Assessor", "health_registry"]

HEURISTIC_V1 = "heuristic_v1"

#: Retention of an item the curve cannot speak about: STM (a TTL window, not a decay curve) and
#: any item carrying no recorded salience strength. 1.0 = "nothing has decayed", the honest
#: conservative reading — never a fabricated decay.
_NO_DECAY = 1.0

_SECONDS_PER_HOUR = 3600.0


class HeuristicV1Assessor:
    """The deterministic flag rules (spec §4 lines 219-230)."""

    key = HEURISTIC_V1

    def __init__(self, settings: HealthSettings) -> None:
        self._settings = settings

    def retention(self, item: MemoryItem, *, now: datetime) -> float:
        """R(Δt) — delegated to
        :meth:`~mu_contracts.domain.model.memory.SalienceComponents.strength_retention`, the ONE
        place the curve lives (spec §4 line 232). This method never re-codes it."""
        if item.tier is Tier.STM or item.salience is None:
            return _NO_DECAY
        return item.salience.strength_retention(now, min_strength=self._settings.curve.min_strength)

    def assess(
        self, item: MemoryItem, *, now: datetime, conflict_edges: ConflictEdges
    ) -> frozenset[MemoryHealthFlag]:
        s = self._settings
        flags: set[MemoryHealthFlag] = set()

        if item.pinned:
            # An OVERRIDE marker, not a risk: the exit flags below are still computed and shown,
            # so the user can see what would have happened (spec §4 line 222).
            flags.add(MemoryHealthFlag.PINNED)

        if item.state is State.ARCHIVED:
            flags.add(MemoryHealthFlag.ARCHIVED)

        if item.tier is not Tier.STM and item.state is State.ACTIVE:
            retention = self.retention(item, now=now)
            if s.curve.demote_retention <= retention < s.stale_retention_band:
                flags.add(MemoryHealthFlag.DECAYING)
            if self._is_stale(item, now=now):
                flags.add(MemoryHealthFlag.STALE)

        confidence = conflict_edges.confidence_for(item.id)
        if item.state is State.QUARANTINED or (
            confidence is not None and confidence < s.low_confidence_below
        ):
            flags.add(MemoryHealthFlag.LOW_CONFIDENCE)

        if conflict_edges.unresolved_for(item.id) or conflict_edges.pin_blocked_for(item.id):
            flags.add(MemoryHealthFlag.CONFLICTING)

        return frozenset(flags)

    def _is_stale(self, item: MemoryItem, *, now: datetime) -> bool:
        """Age AND low recency — both required (spec line 227 / §8 line 352).

        Returns ``False`` when no salience has ever been computed for the item: the recency half
        of the conjunction is then unknown, and an unknown conjunct must not be read as True.
        """
        if item.salience is None:
            return False
        age_h = (now - item.last_seen).total_seconds() / _SECONDS_PER_HOUR
        return age_h > self._settings.stale_after_h and (
            item.salience.recency < self._settings.stale_recency
        )


#: The strategy substitution seam (spec §3.3 line 196; memory-layer §10). Fail-loud on an unknown
#: key and on a duplicate registration — inherited from :class:`Registry`, not re-implemented.
#:
#: NOTE on the factory signature: :class:`Registry` factories take the central ``Settings`` root,
#: but ``HealthSettings`` is not yet a sibling field on it (see
#: :mod:`mu_engine.services.health.settings`), so the registered factory builds the subtree's own
#: defaults. Once the composition root wires ``Settings.health`` this becomes
#: ``settings.health`` with no change to callers. Production wiring injects an explicitly
#: configured assessor into :class:`~mu_engine.services.health.service.MemoryHealthService`
#: directly (DI); the registry is the SUBSTITUTION seam, not the only construction path.
health_registry: Registry[HealthAssessor] = Registry("health")


@health_registry.register(HEURISTIC_V1)
def _build_heuristic_v1(settings: Settings) -> HealthAssessor:
    del settings  # HealthSettings is not yet a Settings sibling — see the note above.
    return HeuristicV1Assessor(HealthSettings())
