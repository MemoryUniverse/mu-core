"""``PartitionPersonaEvidenceReader`` — the production ``PersonaEvidenceReader``
(``persona-design.md`` §1 + §2.2 line 103).

**This module closes the blocker :mod:`mu_engine.services.persona.evidence` recorded.** That
docstring says spec line 103 wants the slot tag produced by "a cheap ``models.classify_model``
classifier … at *capture* time, **cached on the item**", that ``MemoryItem`` is ``extra="forbid"``
with nowhere to cache it, and that the two ways out — *re-run the classifier every rebuild* or
*invent a keyword heuristic the spec does not specify* — were "both worse than an honest gap". At
the time that was right. It is not right any more: with nothing implementing this port the whole
subsystem was DEAD, the ``<persona>`` inject section was budgeted 15 % of the context window for a
string that could never be non-empty, and an honest gap that ships as a product claim stops being
honest.

So this module takes the FIRST of those two ways out and removes the objection to it rather than
accepting it:

* **The classifier is spec line 103's classifier**, routed on ``Task.CLASSIFY`` →
  ``ModelSettings.classify_model`` (``providers/task_map.py:36``). No keyword heuristic is
  invented here; a slot tag is a model verdict or it does not exist.
* **The cost objection is answered by moving the CACHE, not by dropping it.** Spec line 103's
  requirement is *"one small call amortised over ingest, not re-run every rebuild"* — a statement
  about how often the model runs, and only incidentally about where the answer is kept. The answer
  is kept here, in :class:`PartitionPersonaEvidenceReader`'s per-item tag cache keyed on
  ``MemoryItem.id``, so a memory is classified **once per process, not once per rebuild**. The
  only fidelity delta against the spec is that the cache is beside the item instead of on it, and
  therefore does not survive a restart. Recorded, not hidden (ARCHITECTURE-DELTAS).
* **The cache holds the TAG, never the evidence row.** ``PersonaEvidence`` carries the whole
  ``MemoryItem``, and ``mention_count``/``access_count``/``last_seen`` are exactly the fields
  ``WeightedSlotV1Aggregator`` scores reinforcement and decay from (``aggregator.py:107-127``).
  Caching a built ``PersonaEvidence`` would freeze a memory's reinforcement at whatever it was the
  first time it was classified and quietly defeat §3.3's decay. The tag — slot, value, confidence
  — is the only part that is a function of the CONTENT, which is the only part that cannot change
  for a given id (supersession mints a new id; it never rewrites one).

**No model wired ⇒ no reader, by the house ABSENCE rule.** This class takes a REQUIRED tagger. A
deployment with no ``ModelRouter`` composes no persona at all and says so by name at the
composition root (the same decision ``LocalContainer`` already makes for ``health``/``pin`` on a
binding that cannot walk a partition). A reader that returned "no evidence" because it had no
classifier would be a stub answering honestly-shaped nothing, and every rebuild above it would
read as "this user has no persona yet" forever. Fail CLOSED and VISIBLE beats fail quiet.

**Tenancy.** Both reads take the AUTHORIZED ``ns`` — the one that already passed
``TenancyGuard.assert_scope`` in ``PersonaService`` — and neither widens it: the partition walk is
bounded and namespace-scoped, and ``PersonaService._own_partition_only`` (``service.py:496``)
independently REFUSES any row this reader hands back from outside the user's own PRIVATE grain.
That belt is the reason this module is allowed to depend on a repository at all.

**Off the hot path.** The only callers of this port are ``PersonaService.rebuild`` /
``PersonaService.refresh``, both sleep-time (``service.py:31-38``). The model this module holds is
therefore unreachable from ``remember()`` and from recall — a property of WHERE the port is
called, enforced by ``test_persona_boundary_unit``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.memory import MemoryItem, Namespace, State, Tier
from mu_contracts.domain.model.persona import PersonaSlot
from mu_engine.providers._contracts import Message, MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.services.persona.evidence import PersonaEvidence
from mu_engine.services.persona.settings import PersonaSettings
from mu_engine.services.persona.store import assert_private
from mu_engine.services.persona.synthesizer import PersonaSynthesisPort

__all__ = [
    "ClassifierSlotTagger",
    "PartitionPersonaEvidenceReader",
    "PersonaPartitionReader",
    "PersonaSlotTagger",
    "SlotTag",
]

_log = structlog.get_logger("mu_engine.services.persona.reader")

#: The tiers a durable trait may be read from. STM is excluded for the SAME reason
#: ``PersonaService._INCREMENTAL_TIERS`` excludes it (``service.py:110-113``): an STM row is a TTL
#: window, not a durable trait, and a persona built from it would learn and forget on the STM
#: clock instead of on §3.3's decay curve. One rule, stated in two places that must agree — this
#: one is asserted against the other in ``test_persona_reader_unit``.
_EVIDENCE_TIERS: frozenset[Tier] = frozenset({Tier.MTM, Tier.LTM})

#: Only live rows evidence a persona. A SUPERSEDED/ARCHIVED/QUARANTINED memory is a fact the
#: engine has already decided no longer holds (invalidate-don't-delete, memory-layer §7.3), and
#: §3.3's supersession rule would be defeated if the loser kept voting for its slot.
_EVIDENCE_STATES: frozenset[State] = frozenset({State.ACTIVE})


class SlotTag(BaseModel):
    """One classifier verdict about ONE memory, WITHOUT the memory.

    This is the cacheable half of a :class:`~mu_engine.services.persona.evidence.PersonaEvidence`
    — see the module docstring for why the other half must not be cached. ``value`` is
    memory-derived CONTENT and carries the same discipline ``PersonaEvidence.value`` does: in
    process only, never a log, span, audit row, bus event or meter (CLAUDE.md rule 3).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    slot: PersonaSlot
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class PersonaSlotTagger(Protocol):
    """Spec line 103's classifier, as a port.

    Batched on purpose: the unit of work is a page of the partition, not a memory, so a rebuild
    costs a bounded number of model calls rather than one per row. An implementation MUST return
    a mapping keyed by ``MemoryItem.id`` and MUST NOT invent an id it was not given.
    """

    async def tag(self, items: Sequence[MemoryItem]) -> Mapping[str, tuple[SlotTag, ...]]: ...


@runtime_checkable
class PersonaPartitionReader(Protocol):
    """The NARROW slice of ``mu_contracts.ports.memory.MemoryRepository`` this reader depends on.

    Declared structurally, in the ``ConflictAdjudicationPort``/``PersonaSynthesisPort`` style, for
    two reasons. (1) DEV-STANDARDS rule 6: persona needs a bounded partition walk and a by-id
    fetch, not ``add``/``set_pinned``/``semantic``. (2) ``MemoryRepository.semantic`` takes a
    ``CallerIdentitySet`` — the authorization vocabulary §5.4 rule 1 bars persona from naming at
    all. Depending on the full façade would put that type in persona's import graph; depending on
    this slice makes it structurally unreachable. ``TieredMemoryRepository`` satisfies it verbatim.
    """

    async def enumerate(
        self,
        ns: Namespace,
        *,
        states: frozenset[State],
        tiers: frozenset[Tier] | None,
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]: ...

    async def get(self, ns: Namespace, id: str) -> MemoryItem | None: ...


# ------------------------------------------------------------------------------ the classifier
#: What each slot MEANS, in one clause, for the classifier prompt. **A gloss per slot, because
#: the bare vocabulary was measured not to be enough**: given only the names, the deployed model
#: answered with `"follows"`, `"topic"` and `"coffee"` — words lifted from the statement — and
#: `_parse_row` discarded three rows out of four. The glosses are the smallest thing that made it
#: copy names off the list instead of inventing them. `test_persona_reader_unit` asserts this
#: table covers `PersonaSlot` EXACTLY, so a new slot cannot ship with no gloss.
_SLOT_GLOSS: dict[PersonaSlot, str] = {
    PersonaSlot.EXPERTISE: "a subject the user is skilled in",
    PersonaSlot.FOLLOWED_TOPIC: "a subject the user follows or reads about",
    PersonaSlot.GOAL: "something the user is trying to achieve",
    PersonaSlot.HOBBY: "a leisure activity the user does",
    PersonaSlot.INFORMATION_DENSITY: "how much detail the user wants",
    PersonaSlot.INTERACTION_PACE: "how fast the user wants to move",
    PersonaSlot.LANGUAGE: "a natural language the user speaks",
    PersonaSlot.LANGUAGE_STYLE: "the register the user writes in",
    PersonaSlot.NICKNAME: "what the user likes to be called",
    PersonaSlot.OCCUPATION: "the user's job",
    PersonaSlot.PERSONALITY: "a stable character trait of the user",
    PersonaSlot.PREFERENCE: "something the user likes or chooses",
    PersonaSlot.RESPONSE_STYLE: "how the user wants answers written",
    PersonaSlot.ROLE_PREFERENCE: "the role the user wants the assistant to take",
}

#: The classifier prompt. **MEASURED against the deployed model, not composed** — see the
#: docstring of :class:`ClassifierSlotTagger` for the before/after.
_SYSTEM = (
    "You tag a user's remembered statements with PERSONA slots — stable facts and stable "
    "interaction preferences about the USER.\n"
    "The input is one numbered statement per line, e.g. `0: ...`, `1: ...`.\n"
    'Return ONLY a JSON object of the form {"tags": [ ... ]}, where each element is\n'
    '  {"index": <the LINE NUMBER of the statement, an integer>, "slot": <one slot name>, '
    '"value": <short phrase>, "confidence": <number between 0.0 and 1.0>}\n'
    "The slot MUST be copied EXACTLY from this list — any other spelling is discarded:\n"
    "<SLOTS>\n"
    "Pick the closest name on the list. If none fits, emit NO tag for that statement; never "
    "invent a slot name and never use a word from the statement as a slot name.\n"
    "WORKED EXAMPLE — input:\n"
    "0: The user is a pastry chef in Lyon.\n"
    "1: The user reads about sourdough every day.\n"
    "2: The Loire is the longest river in France.\n"
    "output:\n"
    '{"tags": [{"index": 0, "slot": "occupation", "value": "pastry chef", '
    '"confidence": 0.9}, {"index": 1, "slot": "followed_topic", "value": "sourdough", '
    '"confidence": 0.8}]}\n'
    "Statement 2 was about the world, not about the user, so it produced no tag.\n"
    "'value' is a short canonical phrase, never a sentence copied from the input.\n"
    "Emit at most one tag per index per slot."
)

_KEY_TAGS = "tags"
_KEY_INDEX = "index"
_KEY_SLOT = "slot"
_KEY_VALUE = "value"
_KEY_CONFIDENCE = "confidence"


class ClassifierSlotTagger:
    """Spec line 103's ``models.classify_model`` tagger, batched.

    Routed on ``Task.CLASSIFY`` (``providers/catalog.py:42``), which ``TaskClassMapper`` resolves
    to ``ModelSettings.classify_model`` (``providers/task_map.py:36``) — the canonical field
    §6 line 238 pins, so persona still invents no ``persona_model``.

    **The prompt is MEASURED against the deployed model, not composed and hoped for.** Every
    element of it below was added because the previous version was measured to fail on the one
    model this project actually deploys — Ollama ``qwen2.5:0.5b`` on ``mu-dev-slm`` — with four
    statements at temperature 0:

    * **v1** addressed the statement by ``"i"`` and showed an envelope of ``<placeholders>``.
      Every reply put a SLOT NAME in ``"i"``; the first call disagreed with the next two.
      ``_parse_row`` yielded **0 usable tags out of 4, on every call.**
    * **v2** named the field ``"index"``, said in words that it is the line number, and carried a
      worked example that is REAL JSON. Rows became well-formed and reproducible — but the model
      answered with ``"follows"``, ``"topic"`` and ``"coffee"``, slot names lifted from the
      statement, and three rows out of four were discarded for an unknown slot. **1 usable tag.**
    * **v3, shipped:** the vocabulary is presented as a list with a ONE-CLAUSE GLOSS per slot
      (:data:`_SLOT_GLOSS`) and the model is told to pick the closest listed name or emit nothing.
      **4 rows out of 4 parse, byte-identical across four calls.**

    A subsystem whose classifier returns nothing builds no persona — the composed-and-inputless
    failure this module exists to remove, one layer further down and invisible to every unit test,
    because a unit test writes the reply itself. **The QUALITY of the assignments at 0.5B is
    poor** (the model mis-assigns slots and drifts an index), and that is a model-capability
    finding for the data reviewer, not something a prompt fixes: what is proved here is that the
    pipeline produces real, parseable, deterministic verdicts from a real model.

    The prompt's worked example is fed through this module's own parser by
    ``test_persona_reader_unit``, and the gloss table is asserted to cover ``PersonaSlot``
    exactly, so neither can drift away from the code again.

    **A malformed or partly-malformed reply drops the offending ROWS, and only those.** That is
    not a silent partial: a classifier is a per-row judgement, an unparseable row is a row with no
    verdict, and a row with no verdict is exactly "this memory evidences no slot" — the answer the
    spec says most rows have. What is NOT tolerated is a malformed ENVELOPE (not JSON, not an
    object, no ``tags`` list): that is the model failing to answer at all, and it RAISES so the
    caller's named degrade fires instead of persona quietly learning nothing.
    """

    def __init__(self, *, router: PersonaSynthesisPort, settings: PersonaSettings) -> None:
        self._router = router
        self._settings = settings

    async def tag(self, items: Sequence[MemoryItem]) -> Mapping[str, tuple[SlotTag, ...]]:
        out: dict[str, tuple[SlotTag, ...]] = {}
        for batch in _batched(items, self._settings.tagger_batch_size):
            out.update(await self._tag_batch(batch))
        return out

    async def _tag_batch(self, batch: Sequence[MemoryItem]) -> dict[str, tuple[SlotTag, ...]]:
        completion = await self._router.generate(
            Task.CLASSIFY,
            [
                Message(role=MessageRole.SYSTEM, content=_system_prompt()),
                Message(role=MessageRole.USER, content=_render_batch(batch)),
            ],
            max_tokens=self._settings.tagger_max_tokens,
            temperature=self._settings.tagger_temperature,
            response_format="json_object",
        )
        rows = _parse_envelope(completion.text)
        tagged: dict[str, list[SlotTag]] = {}
        dropped = 0
        for row in rows:
            parsed = _parse_row(row, batch)
            if parsed is None:
                dropped += 1
                continue
            item_id, tag = parsed
            bucket = tagged.setdefault(item_id, [])
            if any(existing.slot is tag.slot for existing in bucket):
                continue  # "at most one tag per index per slot" — keep the first, deterministically
            bucket.append(tag)
        if dropped:
            # Content-free: a COUNT and a batch size, never a slot value or a memory body.
            _log.warning("persona_tag_rows_dropped", dropped=dropped, batch=len(batch))
        # Every item in the batch gets an entry, INCLUDING the untagged ones: an empty tuple is a
        # real, cacheable verdict ("no slot"), and without it the reader would re-ask the model
        # about the same untagged memory on every single rebuild — the exact cost spec line 103
        # forbids, reintroduced through the back door.
        return {item.id: tuple(tagged.get(item.id, ())) for item in batch}


_SLOT_VOCABULARY = "\n".join(
    f"  {slot.value} — {gloss}" for slot, gloss in sorted(_SLOT_GLOSS.items())
)


def _system_prompt() -> str:
    """``_SYSTEM`` with the live slot vocabulary substituted.

    A ``str.replace`` on a sentinel, NOT ``str.format``: the prompt shows the model the literal
    JSON envelope it must return, braces and all, and ``format`` reads those braces as fields —
    it raised ``KeyError: '"tags"'`` on every single call. Found by the test above, not reasoned
    about, which is the only reason it is not still there.
    """
    return _SYSTEM.replace("<SLOTS>", _SLOT_VOCABULARY)


def _render_batch(batch: Sequence[MemoryItem]) -> str:
    """The batch as prompt text, index-addressed. Sorted by nothing — the caller's order IS the
    index space, so it must not be re-sorted here."""
    return "\n".join(f"{index}: {item.content}" for index, item in enumerate(batch))


def _parse_envelope(text: str) -> list[object]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("persona slot-tag reply was not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("persona slot-tag reply was not a JSON object")
    rows = payload.get(_KEY_TAGS)
    if not isinstance(rows, list):
        raise ValueError("persona slot-tag reply carried no 'tags' list")
    return rows


def _parse_row(row: object, batch: Sequence[MemoryItem]) -> tuple[str, SlotTag] | None:
    """One row -> ``(item_id, tag)``, or ``None`` when the row is unusable.

    Every rejection below is a row the model got wrong in a way that would otherwise corrupt a
    persona: an index outside the batch would attribute one memory's trait to another, an unknown
    slot name would be silently dropped by the enum anyway, and an out-of-range confidence would
    be rejected by ``SlotValue``'s own bound three layers later, at rebuild time.
    """
    if not isinstance(row, dict):
        return None
    index = row.get(_KEY_INDEX)
    slot_name = row.get(_KEY_SLOT)
    value = row.get(_KEY_VALUE)
    confidence = row.get(_KEY_CONFIDENCE)
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(batch):
        return None
    if not isinstance(slot_name, str) or not isinstance(value, str) or not value.strip():
        return None
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        return None
    if not 0.0 <= float(confidence) <= 1.0:
        return None
    try:
        slot = PersonaSlot(slot_name)
    except ValueError:
        return None
    return batch[index].id, SlotTag(slot=slot, value=value.strip(), confidence=float(confidence))


def _batched(items: Sequence[MemoryItem], size: int) -> Iterable[Sequence[MemoryItem]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ------------------------------------------------------------------------------- the reader
class PartitionPersonaEvidenceReader:
    """``PersonaEvidenceReader`` over the user's own PRIVATE partition (spec §1).

    Both methods are bounded: :meth:`evidence_for` walks at most ``limit`` rows through the
    façade's own paging cursor and stops, and :meth:`evidence_for_ids` fetches exactly the ids it
    was given, at a bounded concurrency. Neither can turn into an unbounded partition scan
    (memory-health §3.1 house rule).
    """

    def __init__(
        self,
        *,
        repo: PersonaPartitionReader,
        tagger: PersonaSlotTagger,
        settings: PersonaSettings | None = None,
    ) -> None:
        self._repo = repo
        self._tagger = tagger
        self._settings = settings or PersonaSettings()
        #: ``MemoryItem.id`` -> the classifier's verdict. See the module docstring: the TAG is
        #: cached, never the evidence row. Insertion-ordered, so eviction is FIFO and total.
        self._tags: dict[str, tuple[SlotTag, ...]] = {}

    async def evidence_for(self, ns: Namespace, *, limit: int) -> Sequence[PersonaEvidence]:
        """At most ``limit`` persona-tagged propositions from ``ns`` (spec §1)."""
        assert_private(ns, "persona.evidence_for")
        items = await self._walk(ns, limit=limit)
        return await self._evidence(items)

    async def evidence_for_ids(
        self, ns: Namespace, ids: frozenset[str]
    ) -> Sequence[PersonaEvidence]:
        """Evidence for specific ids — the §2.4 line 121 incremental path."""
        assert_private(ns, "persona.evidence_for_ids")
        items = await self._fetch(ns, ids)
        return await self._evidence(items)

    # ------------------------------------------------------------------------------ internals
    async def _walk(self, ns: Namespace, *, limit: int) -> list[MemoryItem]:
        """One bounded, cursor-paged walk over the DURABLE tiers. Stops at ``limit`` rows, at an
        exhausted partition, at an EMPTY page, or at ``max_evidence_pages`` — whichever comes
        first. All four bounds are load-bearing; the empty-page one is the subtle one.

        **Why the walk cannot simply run to ``cursor is None``, MEASURED against the real stores.**
        ``MemoryRepository.enumerate``'s own docstring pins the rule: *"``tiers`` narrows which
        tiers THIS PAGE reads — not which tiers the walk covers… The one caller that narrows is
        the degraded retry, and it narrows for a page, not for a walk."* A narrowed page carries
        the narrowed-away tier's position forward in the cursor verbatim, so with STM bound the
        continuation is **never** ``None``. The façade then hands back ONE empty "stalled" page
        (``cursor.py:MAX_STALLED_PAGES == 1``) and the NEXT call raises
        ``TierRepositoryUnavailableError`` naming the unread tier — deliberately, so a caller
        cannot mistake a partial read for a complete one. Before this bound existed, a
        four-memory partition returned ``4 rows, cursor=…{"stm": ""}``, then an empty page, then
        the raise: **every sleep-time rebuild degraded, and no persona could ever be written on a
        real container.** Composed, subscribed, driven — and permanently silent.

        **And the walk cannot drop the narrowing either**, which was the first fix and was wrong.
        ``add()`` writes STM *and* MTM, and an un-narrowed walk dedupes the two copies of one
        memory to the STM representative (``TIER_ORDER`` puts STM first), so every row came back
        ``tier == STM`` and this reader's own durable-tier filter discarded all four. **Measured:
        ``tiers=None`` → ``4 rows, tiers=['stm','stm','stm','stm']``; ``tiers={MTM, LTM}`` → the
        same four as MTM.** So ``MemoryItem.tier`` from an un-narrowed walk is a REPRESENTATIVE,
        not a durability signal, and the narrowing is what makes ``_EVIDENCE_TIERS`` mean what
        it says.

        The cost of stopping at an empty page is bounded and named: a page whose rows are ALL
        rejected by the ``state`` predicate also ends the walk early. Persona's evidence read is
        a bounded sample by construction (``max_evidence_items``), so that costs recall of
        evidence for one sleep-time tick, never correctness — and the alternative costs the whole
        subsystem.
        """
        if limit <= 0:
            return []
        collected: list[MemoryItem] = []
        cursor: str | None = None
        for _page in range(self._settings.max_evidence_pages):
            page, cursor = await self._repo.enumerate(
                ns,
                states=_EVIDENCE_STATES,
                tiers=_EVIDENCE_TIERS,
                pinned=None,
                cursor=cursor,
                limit=min(self._settings.evidence_page_size, limit - len(collected)),
            )
            collected.extend(page)
            if not page or cursor is None or len(collected) >= limit:
                break
        return collected[:limit]

    async def _fetch(self, ns: Namespace, ids: frozenset[str]) -> list[MemoryItem]:
        """By-id fetch at a bounded concurrency. Sorted so the batch — and therefore the model
        prompt and any cached completion — is deterministic for the same id set."""
        gate = asyncio.Semaphore(self._settings.evidence_fetch_concurrency)

        async def one(memory_id: str) -> MemoryItem | None:
            async with gate:
                return await self._repo.get(ns, memory_id)

        fetched = await asyncio.gather(*(one(memory_id) for memory_id in sorted(ids)))
        return [
            item
            for item in fetched
            if item is not None and item.tier in _EVIDENCE_TIERS and item.state in _EVIDENCE_STATES
        ]

    async def _evidence(self, items: Sequence[MemoryItem]) -> list[PersonaEvidence]:
        """Tag the ones this process has not seen, then join EVERY tag to its LIVE item."""
        unseen = [item for item in items if item.id not in self._tags]
        if unseen:
            self._remember(await self._tagger.tag(unseen))
        return [
            PersonaEvidence(
                slot=tag.slot, value=tag.value, item=item, tag_confidence=tag.confidence
            )
            for item in items
            for tag in self._tags.get(item.id, ())
        ]

    def _remember(self, tags: Mapping[str, tuple[SlotTag, ...]]) -> None:
        """Cache with a FIFO bound (DEV-STANDARDS rule 3: no unbounded in-memory growth).

        Eviction is lossless in the limit, exactly like ``PersonaService.note_promoted``'s queue
        drop: an evicted id is simply re-classified the next time it is walked. The cost of a
        too-small cache is money, never correctness.
        """
        self._tags.update(tags)
        overflow = len(self._tags) - self._settings.tag_cache_max_items
        for _ in range(max(0, overflow)):
            self._tags.pop(next(iter(self._tags)))
