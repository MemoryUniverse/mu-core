"""``SalienceStrategy`` — the sweep gate score S(m) (spec §6, ADR 0034; A4 amendment).

```
base(m) = w_rec*rec(m) + w_use*use(m) + w_imp*imp(m)       # the ratified three, sum to 1
S(m)    = base(m)                                          # cen ABSENT
        = (1 - w_cen)*base(m) + w_cen*cen(m)               # cen PRESENT
rec(m)  = exp(-ln2 * age_hours(m) / recency_half_life_h)   # Ebbinghaus half-life decay
use(m)  = min(access_count(m) / usage_cap, 1)
imp(m)  = importance_score(m)
cen(m)  = min(max(deg_hub(subject), deg_hub(object)) / degree_cap, 1)   # A4, may be ABSENT
```

**The fourth term (A4 structural salience).** ``cen(m)`` is how connected this fact's entities are
inside the namespace's own LTM graph — graphify's "god node" degree centrality
(``other_repos/graphify/graphify/analyze.py:109-130``) turned into a per-item scalar. It is a GRAPH
property, so it is NOT computed here: ``lifecycle/centrality.py``'s async ``CentralityService``
projects the namespace's entity graph once per sweep into a ``CentralityIndex``, and this module
reads ``cen(m)`` out of that index with a pure dict lookup (``CentralityLookup.centrality_for``).
That is the same shape ``access_count`` already uses (written by the recall path, read here by
:meth:`_usage` — spec §6 line 267), and it is what keeps this function pure and synchronous: a
sweep must never do per-item graph I/O.

**Why the fourth term is a BLEND and not a fourth entry in a renormalising weighted mean.** The
obvious formulation — four weights summing to 1, divided by the sum of the weights whose terms are
present — is mathematically identical to this one but NOT identical in floating point on the path
that matters. ``(0.45*rec + 0.18*use + 0.27*imp)/0.90`` differs from ``0.5*rec + 0.2*use +
0.3*imp`` in the last ulp for 46,942 of 112,211 grid points over
the ``rec``/``use``/``imp`` grid, and 20 of those points CROSS one of the three
absolute gates:
e.g. ``rec=0.35, use=0.4, imp=0.15`` scores exactly ``0.3`` the old way and ``0.29999999999999993``
the renormalised way, which is the difference between being rescued by ``score < demote_mtm`` and
being DEMOTED. Since ``cen`` is absent for every item on an install with no centrality service
wired, that would have been a silent behaviour change for every FULL-LOCAL user, forever. The blend
form makes the absent branch return ``base`` UNTOUCHED — bit-for-bit today's three-term score, not
approximately it — while the present branch expands to exactly the same effective four-term weight
vector ``(0.45, 0.18, 0.27, 0.10)`` that sums to 1 and preserves the ratified 5 : 2 : 3 ratio.

**Absence never penalises.** ``cen(m)`` is ABSENT whenever no centrality projection covers this
item's namespace (every fresh install, every install with no graph tier configured, a disabled
service, a truncated pass) or the item carries no (subject, object) triple. Had the absent term
simply contributed 0.0, the achievable maximum would have fallen to 0.90 for every such item —
silently tightening ``promote_stm_mtm=0.7``/``promote_mtm_ltm=0.9``/``demote_mtm=0.3`` (ABSOLUTE
gates calibrated against a score whose maximum is 1.0) and making FULL-LOCAL promote less and
demote more. That is precisely the "crippled baseline" the project's boundary rule forbids.

A PRESENT ``cen(m) == 0.0`` is different, and deliberately so: the graph WAS consulted and this
fact is peripheral. It scores as evidence, and it lowers S. The break-even is exact — an item's
score is unchanged iff its centrality equals its own three-term score, rises above that, and falls
below it.

``rel`` (relevance) is DROPPED off this sweep-path score entirely (spec §6 resolves DRAFT §9 Q1 —
the owner's answer is option (a): drop ``rel``, re-weight the remaining terms to sum to 1). The A4
amendment does not re-open that decision. ``relevance_score``
(``storage/domain/memory.py:173``) is specified as written back and consumed only at recall time
where a query exists; this module never reads it.

Weights + half-life + caps are NEVER a literal here (DEV-STANDARDS rule 3) — they come from
``SalienceSettings`` (S0-07, ``mu_engine.lifecycle.settings``), injected at construction.

Clock-injected (spec §19 Rule 1 — ``SalienceStrategy.score(item, *, clock: Clock)``, verbatim
signature): the instant used for ``age_hours`` is whatever the caller's ``Clock`` returns from
``clock.now()`` — this module makes NO ``datetime.now()``/wall-clock call of its own anywhere,
matching the determinism boundary ``platform/clock.py``'s own docstring states for the rest of the
platform. A ``FrozenClock`` pins the instant for deterministic tests (AC-0.1); an ``OffsetClock``
simulates skew (elsewhere in the MLM, not this pure-math module).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.centrality import CentralityLookup
from mu_engine.lifecycle.settings import SalienceSettings
from mu_engine.storage.domain.memory import MemoryItem

__all__ = ["SalienceStrategy"]

_LN2 = math.log(2)


class SalienceStrategy:
    """Pure, side-effect-free sweep-gate scorer (spec §6).

    Takes its ``SalienceSettings`` at construction (the "tracked seam" — never a call-site
    literal), an OPTIONAL ``CentralityLookup`` (the A4 seam; ``None`` = the three-term score,
    unchanged), and the ``Clock`` per call (spec §19 Rule 1's exact signature), so one instance is
    reusable, unmodified, across every sweep tick.
    """

    def __init__(
        self, settings: SalienceSettings, *, centrality: CentralityLookup | None = None
    ) -> None:
        self._settings = settings
        self._centrality = centrality

    def score(self, item: MemoryItem, *, clock: Clock) -> float:
        """S(m) at ``clock.now()`` — a pure function of ``(item, clock.now(), projection)``.

        Synchronous and I/O-free by contract: the first three terms are already ON the item, and
        ``cen(m)`` is a dict lookup into a projection some OTHER coroutine refreshed. Nothing here
        awaits, opens a socket, or reads a wall clock.

        With no ``CentralityLookup`` wired — or with one that has no projection for this item's
        namespace, or an item with no (subject, object) triple — the fourth term is ABSENT and the
        return value is ``base``: bit-for-bit the pre-A4 three-term score, not an approximation of
        it. See the module docstring for why that exactness is load-bearing rather than pedantic.
        """
        s = self._settings
        base = (
            s.w_recency * self._recency(item, clock.now())
            + s.w_usage * self._usage(item)
            + s.w_importance * item.importance_score
        )
        cen = self._centrality.centrality_for(item) if self._centrality is not None else None
        if cen is None:
            return base
        # Convex combination: with base and cen both in [0, 1] and w_centrality in [0, 1] (both
        # enforced by SalienceSettings), the result is in [0, 1] by construction. The clamp is a
        # last-ulp guard on that guarantee, not a substitute for it — AC-1's unit-interval bound
        # is a property of THIS function, never of one particular settings object.
        blended = (1.0 - s.w_centrality) * base + s.w_centrality * cen
        return min(max(blended, 0.0), 1.0)

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
        """use(m) = min(access_count(m) / usage_cap, 1) — access_count EXISTS (memory.py:174).

        The working precedent for the whole A4 shape: a value computed on ANOTHER path (recall),
        carried on the item, and read here for free by a pure function.
        """
        usage_cap = self._settings.usage_cap
        if usage_cap <= 0:
            return 1.0 if item.access_count > 0 else 0.0
        return min(item.access_count / usage_cap, 1.0)
