"""``SalienceStrategy`` — the sweep gate score S(m) (spec §6, ADR 0034).

```
S(m)   = w_rec*rec(m) + w_use*use(m) + w_imp*imp(m)        # weights sum to 1
rec(m) = exp(-ln2 * age_hours(m) / recency_half_life_h)    # Ebbinghaus half-life decay
use(m) = min(access_count(m) / usage_cap, 1)
imp(m) = importance_score(m)
```

``rel`` (relevance) is DROPPED off this sweep-path score entirely (spec §6 resolves DRAFT §9
Q1 — the owner's answer is option (a): drop ``rel``, re-weight the remaining three to sum to 1).
``relevance_score`` (``storage/domain/memory.py:128``) is written back and consumed only at
recall time where a query exists; this module never reads it.

Weights + half-life + usage-cap are NEVER a literal here (DEV-STANDARDS rule 3) — they come from
``SalienceSettings`` (S0-07, ``mu_engine.lifecycle.settings``), injected at construction.

Clock-injected (spec §19 Rule 1 — ``SalienceStrategy.score(item, *, clock: Clock)``, verbatim
signature): the instant used for ``age_hours`` is whatever the caller's ``Clock`` returns from
``clock.now()`` — this module makes NO ``datetime.now()``/wall-clock call of its own anywhere,
matching the determinism boundary ``platform/clock.py``'s own docstring states for the rest of
the platform. A ``FrozenClock`` pins the instant for deterministic tests (AC-0.1); an
``OffsetClock`` simulates skew (elsewhere in the MLM, not this pure-math module).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.settings import SalienceSettings
from mu_engine.storage.domain.memory import MemoryItem

__all__ = ["SalienceStrategy"]

_LN2 = math.log(2)


class SalienceStrategy:
    """Pure, side-effect-free sweep-gate scorer (spec §6).

    Takes its ``SalienceSettings`` at construction (the "tracked seam" — never a call-site
    literal) and the ``Clock`` per call (spec §19 Rule 1's exact signature), so one instance is
    reusable, unmodified, across every sweep tick.
    """

    def __init__(self, settings: SalienceSettings) -> None:
        self._settings = settings

    def score(self, item: MemoryItem, *, clock: Clock) -> float:
        """S(m) at ``clock.now()`` — a pure function of ``(item, clock.now())`` (AC-0.1)."""
        now = clock.now()
        s = self._settings
        rec = self._recency(item, now)
        use = self._usage(item)
        imp = item.importance_score
        return s.w_recency * rec + s.w_usage * use + s.w_importance * imp

    def _recency(self, item: MemoryItem, now: datetime) -> float:
        """rec(m) = exp(-ln2 * age_hours(m) / recency_half_life_h) — Ebbinghaus decay.

        ``age_hours`` is measured from ``created_at`` (the MMA reference form,
        ``mma/mma/utils/helpers.py:247`` ``time_decay`` — same shape, ``Clock``-driven ``now``
        instead of a direct ``datetime.now()`` call, per spec §19 Rule 1).
        """
        created_at = item.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        age_hours = (now - created_at).total_seconds() / 3600.0
        half_life_h = self._settings.recency_half_life_h
        return math.exp(-_LN2 * age_hours / half_life_h)

    def _usage(self, item: MemoryItem) -> float:
        """use(m) = min(access_count(m) / usage_cap, 1) — access_count EXISTS (memory.py:129)."""
        usage_cap = self._settings.usage_cap
        if usage_cap <= 0:
            return 1.0 if item.access_count > 0 else 0.0
        return min(item.access_count / usage_cap, 1.0)
