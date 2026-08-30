"""``PersonaAffinityShaper`` — §5.2's topic-affinity prior, and the ONLY place persona changes
what a caller sees (``persona-design.md`` §5.2, §5.4).

**Why this module exists.** ``PersonaSettings.affinity_weight`` shipped with no consumer: its own
comment said so ("its consumer is not built in this slice"). A persona that is written, versioned,
decayed and erased but never changes a single byte a caller reads is a field that round-trips, not
a capability. §5.2 is where persona earns its place.

**What it does, in the exact words of §5.2.** *"Persona supplies a relevance prior, applied inside
fusion/rerank on the authorized candidate pool… Magnitude is bounded by
``PersonaSettings.affinity_weight`` (default small) so persona re-orders but cannot dominate
semantic evidence."* And §5.4 rule 4(c) says what the observable consequence must be: *"two callers
with opposite personas over the same PRIVATE partition get the SAME candidate set, differing only
in ORDER and answer voice."* So this module re-orders, and does nothing else. It cannot add a hit,
cannot drop one, and does not rewrite ``fused_score`` (that number is the ranker's testimony about
semantic evidence; overwriting it would make persona lie in the ranker's voice).

**PLACEMENT DELTA, recorded rather than made silently.** §5.2 says the nudge is *"folded into the
RRF weights (``fusion.py:17``) as a rank prior"*. It is not, and it cannot be: ``FusionStrategy
.fuse`` takes ONE weight PER CHANNEL, not a per-candidate prior — the delta ``affinity_weight``'s
own comment already recorded before this module existed. Implementing it one layer OUT, over the
result of the whole ranked read, is strictly stronger for the property §5.2 cares about most:

    *"the affinity boost is applied AFTER the server-side ``MatchAny(authorized_ids)`` +
    ``state='active'`` filter has run inside filterable-HNSW. Persona reweights the SURVIVORS; it
    never participates in the ``query_filter``."*

Inside fusion that ordering is a convention a future edit can break. Out here it is structural:
this class is handed a finished result and has no reference to a store, a filter, a ranker or an
authorized-id resolver to break it with.

**The §5.4 firewall, held by CONSTRUCTION.** This module names no recall type, no authz type and
no vector-store type — the ban ``test_persona_boundary_unit.test_the_persona_package_imports_no_
recall_authz_or_vector_module`` enforces over the whole package, and this module is deliberately
written to keep passing it rather than to be excused from it. The seams below are STRUCTURAL
Protocols in the house style (``RetentionServicePort``, ``ConflictAdjudicationPort``,
``PersonaSynthesisPort``): the narrow slice persona depends on, declared here, satisfied verbatim
by ``mu_engine.services.recall.service.RecallService`` and its DTOs without either side importing
the other. What persona can see of a hit is exactly three fields — an id, a body and a score. It
cannot see a namespace, an ``authorized_ids`` set or a filter, so it cannot participate in one.

**Read by KEY, never searched.** The affinity terms come from ``PersonaRepository.get(ns)`` — a
load by key on the contracts port (§3.2 line 161: *"persona is never vector-searched; it is loaded
by key"*), taking a namespace and nothing else. It receives no query, no candidate set and no
caller identity, so it cannot become a filter condition even by accident (§5.4 rule 2).

**RECORDED DELTA — the warm read.** §4.6 (preload) says the query-time persona read is a WARM one:
the daemon holds the persona-lensed bundle pre-rendered and ``PersonaUpdated`` invalidates it. No
warm-cache service exists on either composition root in this repo, so this shaper performs the
keyed load per recall. That is correct-but-unwarmed: with the shipped ``InMemoryPersonaRepository``
it is a dict lookup, and with a durable adapter it is one keyed read that the §4.6 warm cache is
supposed to absorb. Flagged for the day that cache lands, not silently absorbed.

**Fail-SOFT is the right failure here, and it is named.** A persona read that raises must not take
down a recall: persona is voice and relevance, and the correct answer without it is the ranker's
own order — the same result the user got before persona existed. So a failed read degrades to the
UNSHAPED result and emits ``DegradedModeEntered``; it never returns a half-shaped list and never
swallows the event. Refusing the recall instead would let an optional subsystem fail a mandatory
one, which is the opposite of the boundary rule.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

import structlog

from mu_contracts.domain.events import DegradedModeEntered, DegradeReason
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.persona import PersonaProfile, PersonaSlot
from mu_contracts.ports.persona import PersonaRepository
from mu_engine.pipelines.distill import EventPublisher
from mu_engine.services.persona.settings import PersonaSettings
from mu_engine.services.persona.store import persona_key

__all__ = [
    "AFFINITY_SLOTS",
    "PersonaAffinityShaper",
    "PersonaShapedRankedRead",
    "RankedHit",
    "RankedRead",
]

_log = structlog.get_logger("mu_engine.services.persona.shaping")

#: The slots §5.2 names as the affinity source, verbatim: *"the ``followed_topic``/``preference``/
#: ``expertise`` slots give a small per-candidate boost"*. NOT every slot: ``response_style`` or
#: ``interaction_pace`` describe HOW to speak (§5.1's job), not WHAT the user cares about, and
#: letting them move a ranking would make persona shape relevance with voice.
AFFINITY_SLOTS: frozenset[PersonaSlot] = frozenset(
    {PersonaSlot.FOLLOWED_TOPIC, PersonaSlot.PREFERENCE, PersonaSlot.EXPERTISE}
)

_DEGRADE_COMPONENT = "persona"
_DEGRADE_MODE = "persona_shaping_unavailable"
#: The same closest-existing reason ``PersonaService`` records: CANONICAL §2's ``DegradeReason``
#: union has no persona member and this lane does not own ``domain/events.py``.
_DEGRADE_REASON = DegradeReason.LLM_UNAVAILABLE_HEURISTIC

#: Word-ish token split for affinity terms. Deliberately not a tokenizer: a persona slot value is
#: a short canonical phrase, and matching it needs word boundaries, not linguistics.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@runtime_checkable
class RankedHit(Protocol):
    """The four fields persona is allowed to see of a candidate — and no others.

    Satisfied verbatim by both shipped ``RecallItemView`` classes. There is deliberately no
    ``namespace``, no ``authorized_ids``, no ``state`` and no ``channel`` here: what a shaper
    cannot read, a shaper cannot filter on (§5.4 rule 2).

    ``is_floor`` is here because the ranked read this decorator wraps is **not one sorted list**
    — see :meth:`PersonaAffinityShaper._reorder`. It is a protection flag and a display-block
    marker, carrying no tenancy, identity or authorization meaning, so reading it costs the §5.4
    firewall nothing.
    """

    @property
    def memory_id(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def fused_score(self) -> float: ...

    @property
    def is_floor(self) -> bool: ...


H = TypeVar("H", bound=RankedHit)


@runtime_checkable
class RankedRead(Protocol):
    """The narrow slice of the ranked read path this decorator wraps.

    ``scope``/``q``/the result are ``Any`` ON PURPOSE, and it is the point rather than a shortcut:
    naming ``ClientScope`` would be harmless but naming ``RecallQuery``/``RecallResult`` would put
    ``mu_engine.services.recall`` in persona's import graph, which §5.4 rules 1-2 forbid and the
    package's own boundary test enforces. The only members :class:`PersonaShapedRankedRead` ever
    touches are ``q.namespace``, ``result.items`` and ``result.model_copy`` — each asserted at
    runtime by ``test_persona_shaping_unit`` against the REAL shipped DTOs, so the loss of static
    typing here is bought back by a test that would fail the day either shape changes.
    """

    async def recall(self, scope: Any, q: Any) -> Any: ...

    async def recall_degraded(
        self, scope: Any, q: Any, *, without: set[str], reason: DegradeReason
    ) -> Any: ...


class PersonaAffinityShaper:
    """§5.2's topic-affinity prior over an already-authorized, already-ranked candidate list."""

    def __init__(
        self,
        *,
        repo: PersonaRepository,
        settings: PersonaSettings | None = None,
        bus: EventPublisher | None = None,
    ) -> None:
        self._repo = repo
        self._settings = settings or PersonaSettings()
        self._bus = bus

    async def shape(self, ns: Namespace, hits: Sequence[H]) -> Sequence[H]:
        """Return ``hits`` re-ordered by the persona prior — the SAME set, never a different one.

        Returns the input object unchanged (identity, not a copy) when there is nothing to do:
        persona disabled, a SHARED namespace, no profile, no affinity terms, or a read that
        failed. The caller uses that identity to skip rebuilding its result DTO.
        """
        if not self._settings.enabled or ns.visibility is not Visibility.PRIVATE or len(hits) < 2:
            return hits
        profile = await self._load(ns)
        if profile is None:
            return hits
        terms = self._terms(profile)
        if not terms:
            return hits
        return self._reorder(hits, terms)

    # ------------------------------------------------------------------------------ internals
    async def _load(self, ns: Namespace) -> PersonaProfile | None:
        try:
            return await self._repo.get(ns)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._degrade(ns, detail=type(exc).__name__)
            return None

    def _terms(self, profile: PersonaProfile) -> Mapping[str, float]:
        """``term -> weight``, where weight is the slot's own confidence (§5.2: the boost is
        per-candidate and confidence-scaled, so an uncertain slot moves a hit less than a certain
        one). Terms below ``affinity_min_confidence``/``affinity_min_term_chars`` are dropped —
        see those settings for why a weak or tiny term is worse than no term."""
        terms: dict[str, float] = {}
        for slot, value in profile.slots.items():
            if (
                slot not in AFFINITY_SLOTS
                or value.confidence < self._settings.affinity_min_confidence
            ):
                continue
            for word in _WORD.findall(value.value.casefold()):
                if len(word) < self._settings.affinity_min_term_chars:
                    continue
                terms[word] = max(terms.get(word, 0.0), value.confidence)
        return terms

    def _reorder(self, hits: Sequence[H], terms: Mapping[str, float]) -> Sequence[H]:
        """Sort by ``fused_score * (1 + affinity_weight * match)`` — STABLE, so two hits with no
        persona signal keep the ranker's own relative order exactly.

        ``match`` is the strongest matching term's confidence, in ``[0, 1]``, so the multiplier is
        in ``[1, 1 + affinity_weight]``: with the default 0.15 a persona-matching hit can gain at
        most 15 %, which is the "re-orders but cannot dominate semantic evidence" bound spec line
        194 asks for, stated as an inequality rather than a hope.

        **WITHIN EACH BLOCK, IN PLACE — because the list this decorator receives is not one
        sorted list, and since D3 it is not two contiguous blocks either.** The two blocks carry
        scores on two DIFFERENT scales: an STM query-relevance score (``_score_stm``, order ~1e-1)
        for a protected floor member versus an RRF rank score (``fusion.py``, order ~1e-2) for a
        fused hit. A single sort over the whole list would compare incomparable numbers.

        ⚠ **The block-CONCATENATION this method used to do is now a defect of its own (D3,
        STATE-AND-DEFECTS-0829.md).** It built ``[*floor_block, *fused_block]``, which was
        faithful to the ``_merge_floor`` of the time (*"floor members lead, then the fused tail"*)
        — and that leading floor is exactly what LoCoMo measured as ``recall@1 = 0.0003`` against
        ``0.1625`` for the same corpus's dense channel alone: the top ``floor_protect_limit`` slots
        of every result were the most-recently-written memories, chosen without reference to the
        query. ``ranker.py:_merge_floor`` and ``recall/service.py:_protect_floor`` were both fixed
        to keep a protected member at whatever rank fusion actually earned it (rescuing only a
        member ranked outside the window, at the END), so ``is_floor`` entries now arrive
        INTERLEAVED. Re-concatenating them here would have silently restored the query-blind
        prefix one decorator downstream of the fix, on every deployment that wires a model —
        ``build_persona`` returns a wiring whenever a router exists, and both composition roots
        then wrap ``self.recall`` with it.

        So the sort is **positional**: the floor subsequence and the fused subsequence are each
        sorted on their own scale, then written back into the SLOTS THEY CAME FROM. Persona
        re-orders within a block, as §5.2 asks; which slots are floor slots stays the ranker's
        decision, as ``is_floor``'s own docstring says it must (*"a protection flag and a display-
        block marker"*, never a position instruction).

        **Found by consequence, not by reading.** The first version of this method did sort
        globally. The integration test that asserts a real persona re-orders a real recall caught
        it: the order changed while the persona-matching hit did not move — the change was the
        global re-sort of a list that was never in score order, and it happened even when the
        persona matched NOTHING at all. A relevance prior that perturbs a read it does not match
        is not a prior, it is a bug with a config knob.
        """
        weight = self._settings.affinity_weight
        reordered = list(hits)
        for slots in (
            [i for i, h in enumerate(hits) if h.is_floor],
            [i for i, h in enumerate(hits) if not h.is_floor],
        ):
            block = self._sort_block([hits[i] for i in slots], terms, weight)
            for slot, hit in zip(slots, block, strict=True):
                reordered[slot] = hit
        if all(a is b for a, b in zip(hits, reordered, strict=True)):
            return hits  # nothing moved — hand back the identical object, not a copy
        return reordered

    @staticmethod
    def _sort_block(block: Sequence[H], terms: Mapping[str, float], weight: float) -> list[H]:
        """One block, sorted by the boosted score, STABLE on the block's own incoming order."""
        scored = [
            (hit.fused_score * (1.0 + weight * _match(hit.content, terms)), index, hit)
            for index, hit in enumerate(block)
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [row[2] for row in scored]

    async def _degrade(self, ns: Namespace, *, detail: str) -> None:
        _log.warning(
            "persona_shaping_degraded", ns=persona_key(ns), reason=_DEGRADE_REASON, detail=detail
        )
        if self._bus is None:
            return
        await self._bus.publish(
            DegradedModeEntered(
                component=_DEGRADE_COMPONENT,
                mode=_DEGRADE_MODE,
                reason=_DEGRADE_REASON,
                detail=detail,
            )
        )


def _match(content: str, terms: Mapping[str, float]) -> float:
    """The strongest affinity term present in ``content``, or ``0.0``.

    Whole-word, case-folded. STRONGEST, not summed: summing would let a slot value made of three
    common words out-shout ``affinity_weight`` and turn a bounded nudge into a dominating term —
    the exact thing §5.2 bounds the magnitude to prevent.
    """
    best = 0.0
    for word in _WORD.findall(content.casefold()):
        weight = terms.get(word)
        if weight is not None and weight > best:
            best = weight
    return best


class PersonaShapedRankedRead:
    """A ``RankedRead`` decorator that applies §5.2 to the result of the one it wraps.

    A DECORATOR, not an edit to the recall service, and that is a design statement rather than a
    convenience: it makes "persona runs strictly AFTER the authorized top-k" (§5.2's hard
    boundary) a fact about the call graph. Everything the recall service does — the two arms, RRF
    fusion, content-hash dedup, floor protection and the belt ``authorized_ids`` re-check — has
    already finished before this object sees anything, and this object holds no reference through
    which it could reach back into any of it.
    """

    def __init__(self, *, inner: RankedRead, shaper: PersonaAffinityShaper) -> None:
        self._inner = inner
        self._shaper = shaper

    async def recall(self, scope: Any, q: Any) -> Any:
        return await self._shape(q, await self._inner.recall(scope, q))

    async def recall_degraded(
        self, scope: Any, q: Any, *, without: set[str], reason: DegradeReason
    ) -> Any:
        """The named channel-subset read, shaped by the SAME rule.

        Forwarded rather than inherited-by-omission on purpose: ``RecallService.recall_degraded``
        calls its OWN ``self.recall``, so a decorator that forwarded only ``recall`` would leave
        every degraded read unshaped and persona would silently stop mattering exactly when a
        channel is down. It has no caller in any ``src/`` tree today; the day it gets one, it
        behaves like its sibling.
        """
        result = await self._inner.recall_degraded(scope, q, without=without, reason=reason)
        return await self._shape(q, result)

    async def _shape(self, q: Any, result: Any) -> Any:
        items: Sequence[RankedHit] = result.items
        shaped = await self._shaper.shape(q.namespace, items)
        if shaped is items:
            return result  # nothing moved: the ranker's own object, untouched
        return result.model_copy(update={"items": list(shaped)})
