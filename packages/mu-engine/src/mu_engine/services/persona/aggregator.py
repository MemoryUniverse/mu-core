"""Stage 1 — ``weighted_slot_v1``, the DETERMINISTIC trait aggregator (``persona-design.md`` §2.2).

Authority: spec §2.2 (lines 100-107). Spec line 107 is the contract this module is built to
satisfy and the one its tests attack: *"No model call — this is arithmetic over already-extracted
facts, so it is unit-testable with an injected ``Clock``."*

PURE, and structurally so: this module imports no router, no ``Task``, no provider, no store, no
clock and no ``asyncio``. ``now`` arrives as an argument. Given the same evidence set it returns
the same slots in any input order — the winner key is a TOTAL order ending in ``MemoryItem.id``,
so two candidates can never tie and let iteration order decide.

Scoring (spec line 104) — *"the highest-confidence, most-reinforced, most-recent value
(``confidence * f(mention_count, access_count) * recency``), reusing the salience recency curve
(``0.5 ** (Δt / half_life)``, memory-layer §5)"*:

* ``confidence`` — :attr:`PersonaEvidence.tag_confidence` (MemOS ``confidence_score``,
  ``OR/MemOS/src/memos/mem_reader/memory.py:114``).
* ``f(mention_count, access_count)`` — ``1 + w*ln(1 + mention + access)``. **The spec does not
  define f** (reported); the saturating form is chosen because MemoryBank's reinforcement is a
  flat +1 per recall (``OR/MemoryBank/memory_bank/memory_retrieval/forget_memory.py:69``, in
  ``update_memory_when_searched`` at ``:63``), so a linear term would grow without bound and let
  volume beat confidence. ``w`` is ``PersonaSettings.reinforcement_weight``.
* ``recency`` — ``0.5 ** (Δt_h / half_life_h)`` measured from ``MemoryItem.last_seen``, and only
  for SUBJECTIVE slots. Objective slots do not decay (§3.3 line 167), so their recency term is
  1.0 and "most recent" survives only as the tie-break.

  *Why the curve is written here and not delegated:* the identical half-life shape already exists
  at ``mu_engine/lifecycle/salience.py:72`` (``exp(-ln2 * age_h / half_life_h)``, which is
  ``0.5 ** (age_h/half_life_h)``), but ``SalienceStrategy`` is not reusable for this: it takes the
  OTHER ``MemoryItem`` (``mu_engine.storage.domain.memory``), anchors on ``created_at`` rather
  than ``last_seen``, and returns the composite ``S(m)``, not a bare recency. The shape is kept
  byte-identical to that line so the two cannot drift.

  ``MemoryItem.salience`` is deliberately NOT read. It is optional and frequently absent (the
  reason ``health/assessor.py:17-20`` records), and a slot whose recency silently became "unknown"
  would be scored against a fabricated 1.0.

Decay drop (§3.3 line 167): a subjective candidate whose recency has fallen below
``PersonaSettings.subjective_drop_below_recency`` is discarded BEFORE the winner is chosen, so a
stale mood is forgotten rather than carried forever. If every candidate for a slot decays, the
slot is absent from the result — that is the intended forgetting, not an error.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from mu_contracts.config.settings import Settings
from mu_contracts.domain.model.persona import PersonaSlot, SlotValue
from mu_engine.platform.registry import Registry
from mu_engine.services.persona.evidence import SUBJECTIVE_SLOTS, PersonaEvidence
from mu_engine.services.persona.settings import WEIGHTED_SLOT_V1, PersonaSettings

__all__ = [
    "TraitAggregator",
    "WeightedSlotV1Aggregator",
    "persona_aggregator_registry",
]

_SECONDS_PER_HOUR = 3600.0

#: Recency of a slot the curve does not apply to: every OBJECTIVE slot (§3.3 line 167 — an
#: occupation does not fade). 1.0 = "nothing has decayed", asserted rather than assumed.
_NO_DECAY = 1.0


@runtime_checkable
class TraitAggregator(Protocol):
    """Spec §6 line 217: ``aggregate(items, *, now) -> dict[PersonaSlot, SlotValue]``.

    SYNCHRONOUS on purpose. An ``async def`` here would make an I/O-bearing or model-bearing
    implementation registrable under ``persona_aggregator_strategy`` without anything noticing;
    a sync signature makes spec line 107's "no model call" a property of the TYPE, not a promise
    in a docstring.
    """

    def aggregate(
        self, evidence: Sequence[PersonaEvidence], *, now: datetime
    ) -> dict[PersonaSlot, SlotValue]: ...


class WeightedSlotV1Aggregator:
    """The default deterministic aggregator (spec §2.2)."""

    key = WEIGHTED_SLOT_V1

    def __init__(self, settings: PersonaSettings) -> None:
        self._settings = settings

    def aggregate(
        self, evidence: Sequence[PersonaEvidence], *, now: datetime
    ) -> dict[PersonaSlot, SlotValue]:
        """Bucket by slot, score, drop the decayed, pick the total-order winner (spec lines
        102-105). Pure: no I/O, no clock call, no model, no randomness."""
        buckets: dict[PersonaSlot, list[tuple[float, PersonaEvidence]]] = {}
        for ev in evidence:
            scored = self._score(ev, now=now)
            if scored is None:  # decayed past the drop floor (§3.3 line 167)
                continue
            buckets.setdefault(ev.slot, []).append((scored, ev))
        return {slot: self._resolve(slot, cands) for slot, cands in sorted(buckets.items())}

    # ------------------------------------------------------------------------------ scoring
    def _score(self, ev: PersonaEvidence, *, now: datetime) -> float | None:
        """``confidence * f(mention, access) * recency`` (spec line 104), or ``None`` when a
        subjective candidate has decayed past the drop floor."""
        recency = self._recency(ev, now=now)
        if ev.slot in SUBJECTIVE_SLOTS and recency < self._settings.subjective_drop_below_recency:
            return None
        return ev.tag_confidence * self._reinforcement(ev) * recency

    def _recency(self, ev: PersonaEvidence, *, now: datetime) -> float:
        """``0.5 ** (Δt_h / half_life_h)`` for a subjective slot; ``1.0`` for an objective one."""
        if ev.slot not in SUBJECTIVE_SLOTS:
            return _NO_DECAY
        age_h = (now - ev.item.last_seen).total_seconds() / _SECONDS_PER_HOUR
        if age_h <= 0.0:
            # Not yet elapsed, or a clock that went backwards: nothing has decayed. Same
            # conservative reading as `SalienceComponents.strength_retention` (memory.py:253).
            return _NO_DECAY
        return math.pow(0.5, age_h / self._settings.subjective_half_life_h)

    def _reinforcement(self, ev: PersonaEvidence) -> float:
        """``f(mention_count, access_count)`` — see the module docstring for why it saturates."""
        touches = ev.item.mention_count + ev.item.access_count
        return 1.0 + self._settings.reinforcement_weight * math.log1p(touches)

    # ------------------------------------------------------------------------------ resolution
    def _resolve(
        self, slot: PersonaSlot, candidates: list[tuple[float, PersonaEvidence]]
    ) -> SlotValue:
        """The winner + its provenance. ``supersession, not deletion`` (§3.3 line 168): the losing
        values simply do not enter the live slot, while their memories keep full bi-temporal
        history in the tiers."""
        winner = max(candidates, key=self._order)[1]
        half_life = self._settings.subjective_half_life_h if slot in SUBJECTIVE_SLOTS else None
        return SlotValue(
            value=winner.value,
            confidence=winner.tag_confidence,
            support_ids=self._support_ids(candidates),
            # WHEN the slot was last EVIDENCED, not when this rebuild ran — `rebuilt_at` on the
            # profile already carries the latter, and conflating them would make every slot look
            # freshly re-asserted after any rebuild and defeat §3.3's decay.
            updated_at=winner.item.last_seen,
            decay_half_life_h=half_life,
        )

    @staticmethod
    def _order(candidate: tuple[float, PersonaEvidence]) -> tuple[float, datetime, str]:
        """The TOTAL order that makes the pick order-independent: score, then recency, then the
        unique ``MemoryItem.id``. Without the id term two equally-scored candidates would tie and
        the answer would depend on input order — which is exactly what
        ``test_persona_aggregator_unit.py``'s determinism test exists to forbid."""
        score, ev = candidate
        return (score, ev.item.last_seen, ev.item.id)

    def _support_ids(self, candidates: list[tuple[float, PersonaEvidence]]) -> tuple[str, ...]:
        """Provenance for the slot (spec line 105), highest-scoring first, capped and deduped.

        Sorted by the same total order as the winner pick, so the head of ``support_ids`` is
        always the winning memory and the cap drops the LEAST evidential ids, deterministically.
        """
        ranked = sorted(candidates, key=self._order, reverse=True)
        return tuple(_dedupe(ev.item.id for _, ev in ranked))[: self._settings.support_ids_limit]


def _dedupe(ids: Iterable[str]) -> list[str]:
    """Order-preserving dedupe — one memory evidencing a slot twice is one support id."""
    seen: set[str] = set()
    out: list[str] = []
    for value in ids:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


#: The Stage-1 substitution seam (spec lines 242, 245; memory-layer §10). Fail-loud on an unknown
#: key and on a duplicate registration — inherited from :class:`Registry`, not re-implemented.
#:
#: Spec line 242 writes ``AdapterRegistry[str, TraitAggregator]``. No such type exists in this
#: repo; the shipped equivalent is the single-parameter ``Registry[T]``
#: (``mu_engine/platform/registry.py:42``), whose factory signature is ``Callable[[Settings], T]``.
#: Reported for spec amendment to ``Registry[TraitAggregator]``.
#:
#: NOTE on the factory signature, same as ``health_registry``: ``Registry`` factories take the
#: central ``Settings`` root, but ``PersonaSettings`` is not yet a sibling field on it (see
#: :mod:`mu_engine.services.persona.settings`), so the registered factory builds the subtree's own
#: defaults. Production wiring injects an explicitly configured aggregator into
#: :class:`~mu_engine.services.persona.service.PersonaService` directly (DI); the registry is the
#: SUBSTITUTION seam (spec line 245's ``ocean_v1`` ablation), not the only construction path.
persona_aggregator_registry: Registry[TraitAggregator] = Registry("persona_aggregator")


@persona_aggregator_registry.register(WEIGHTED_SLOT_V1)
def _build_weighted_slot_v1(settings: Settings) -> TraitAggregator:
    del settings  # PersonaSettings is not yet a Settings sibling — see the note above.
    return WeightedSlotV1Aggregator(PersonaSettings())


def slots_changed(
    before: Mapping[PersonaSlot, SlotValue], after: Mapping[PersonaSlot, SlotValue]
) -> int:
    """How many slots differ between two profiles — the content-free count carried by
    ``PersonaUpdated.slots_changed`` (spec §4 line 177).

    Counts an added, a removed (a decay drop IS a change the warm cache must see) and a
    value-changed slot alike. Compares only ``value``, never the whole ``SlotValue``: a rebuild
    that merely re-evidences the same value with a fresher ``updated_at`` has not changed the
    persona, and reporting it as a change would make ``PersonaUpdated`` invalidate every warm
    inject bundle on every tick.
    """
    keys = set(before) | set(after)
    return sum(
        1
        for key in keys
        if (before[key].value if key in before else None)
        != (after[key].value if key in after else None)
    )
