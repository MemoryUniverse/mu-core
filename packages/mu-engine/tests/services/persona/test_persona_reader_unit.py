"""``PartitionPersonaEvidenceReader`` + ``ClassifierSlotTagger`` — the input side of persona
(``persona-design.md`` §1, §2.2 line 103).

Until this module's subject existed, ``PersonaEvidenceReader`` had NO implementation in any
``src/`` tree, so ``PersonaService.rebuild`` could only ever be handed evidence by a test double
and the whole subsystem was DEAD by definition. These tests are about the two properties that
make the real reader worth having: it is BOUNDED (it can never become a partition scan or a
per-rebuild model bill), and it is CORRECT ABOUT WHAT IT CACHES (the classifier's verdict, never
the memory's reinforcement).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pytest

from mu_contracts.domain.errors import NamespaceIsolationError
from mu_contracts.domain.model.memory import MemoryItem, Namespace, State, Tier
from mu_contracts.domain.model.persona import PersonaSlot
from mu_engine.providers.catalog import Task
from mu_engine.services.persona.reader import (
    _KEY_INDEX,
    _SLOT_GLOSS,
    ClassifierSlotTagger,
    PartitionPersonaEvidenceReader,
    SlotTag,
    _parse_envelope,
    _parse_row,
    _system_prompt,
)
from mu_engine.services.persona.service import _INCREMENTAL_TIERS
from mu_engine.services.persona.settings import PersonaSettings

from .conftest import StubRouter

pytestmark = pytest.mark.unit


class FakePartition:
    """A ``PersonaPartitionReader`` over a fixed list, recording every call so the BOUNDS are
    asserted on real arguments rather than trusted from a docstring."""

    def __init__(self, items: Sequence[MemoryItem], *, page: int = 100) -> None:
        self._items = list(items)
        self._page = page
        self.enumerate_calls: list[tuple[Namespace, frozenset[State], frozenset[Tier] | None, int]]
        self.enumerate_calls = []
        self.get_calls: list[str] = []

    async def enumerate(
        self,
        ns: Namespace,
        *,
        states: frozenset[State],
        tiers: frozenset[Tier] | None,
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        del pinned
        self.enumerate_calls.append((ns, states, tiers, limit))
        start = int(cursor or "0")
        stop = min(start + min(self._page, limit), len(self._items))
        nxt = str(stop) if stop < len(self._items) else None
        return self._items[start:stop], nxt

    async def get(self, ns: Namespace, id: str) -> MemoryItem | None:
        del ns
        self.get_calls.append(id)
        return next((item for item in self._items if item.id == id), None)


class CountingTagger:
    """Tags every item with one PREFERENCE slot and counts how many items it was ASKED about —
    the number the whole "amortised, not re-run every rebuild" argument turns on."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    async def tag(self, items: Sequence[MemoryItem]) -> Mapping[str, tuple[SlotTag, ...]]:
        self.asked.extend(item.id for item in items)
        return {
            item.id: (SlotTag(slot=PersonaSlot.PREFERENCE, value=item.content, confidence=0.8),)
            for item in items
        }


def _reader(
    items: Sequence[MemoryItem], *, tagger: CountingTagger | None = None, **kw: object
) -> tuple[PartitionPersonaEvidenceReader, FakePartition, CountingTagger]:
    partition = FakePartition(items, page=int(kw.pop("page", 100)))
    tag = tagger or CountingTagger()
    reader = PartitionPersonaEvidenceReader(
        repo=partition,
        tagger=tag,
        settings=PersonaSettings(**kw),  # type: ignore[arg-type]
    )
    return reader, partition, tag


# ------------------------------------------------------------------ the cost model (line 103)
async def test_a_memory_is_classified_once_not_once_per_rebuild(
    ns: Namespace, make_item: Callable[..., MemoryItem]
):
    """Spec line 103: *"one small call amortised over ingest, not re-run every rebuild"*.

    The reader cannot cache on the item (``MemoryItem`` is ``extra="forbid"``), so it caches
    beside it. This is the assertion that makes that substitution honest: three rebuilds over the
    same partition ask the classifier about each memory exactly once. Delete the cache and this
    fails with 9 instead of 3 — which is the bill.
    """
    items = [make_item(memory_id=f"m{i}", content=f"fact {i}") for i in range(3)]
    reader, _partition, tagger = _reader(items)

    for _ in range(3):
        assert len(await reader.evidence_for(ns, limit=10)) == 3

    assert tagger.asked == ["m0", "m1", "m2"]


async def test_an_untagged_memory_is_also_remembered_as_untagged(
    ns: Namespace, make_item: Callable[..., MemoryItem]
):
    """The back door into the same bill: most memories evidence NO slot, and a cache that only
    remembered the tagged ones would re-ask the model about every untagged row forever."""

    class SilentTagger(CountingTagger):
        async def tag(self, items: Sequence[MemoryItem]) -> Mapping[str, tuple[SlotTag, ...]]:
            self.asked.extend(item.id for item in items)
            return {item.id: () for item in items}

    tagger = SilentTagger()
    reader, _partition, _t = _reader([make_item(memory_id="m0")], tagger=tagger)

    assert await reader.evidence_for(ns, limit=10) == []
    assert await reader.evidence_for(ns, limit=10) == []
    assert tagger.asked == ["m0"]


async def test_the_cache_holds_the_tag_and_joins_it_to_the_live_item(
    ns: Namespace, make_item: Callable[..., MemoryItem]
):
    """Why the cache holds a ``SlotTag`` and not a ``PersonaEvidence``.

    ``WeightedSlotV1Aggregator`` scores reinforcement off ``mention_count``/``access_count`` and
    decay off ``last_seen``. Caching a built evidence row would freeze all three at whatever they
    were the first time the memory was classified, and §3.3's decay would quietly stop working.
    """
    reader, partition, _t = _reader([make_item(memory_id="m0", mention_count=1)])
    first = await reader.evidence_for(ns, limit=10)
    assert first[0].item.mention_count == 1

    partition._items[0] = make_item(memory_id="m0", mention_count=7, access_count=3)
    second = await reader.evidence_for(ns, limit=10)

    assert second[0].item.mention_count == 7
    assert second[0].item.access_count == 3
    assert second[0].value == first[0].value  # the VERDICT is the cached half


def test_the_cache_is_bounded(ns: Namespace, make_item: Callable[..., MemoryItem]):
    reader, _p, _t = _reader([], tag_cache_max_items=2)
    reader._remember({f"m{i}": () for i in range(5)})
    assert len(reader._tags) == 2


# ---------------------------------------------------------------------------- the read bounds
async def test_the_walk_reads_only_the_durable_tiers_and_only_active_rows(
    ns: Namespace, make_item: Callable[..., MemoryItem]
):
    """An STM row is a TTL window, not a trait; a superseded row is a fact the engine has already
    decided no longer holds. Both would corrupt a persona if they voted.

    **The narrowing is REQUIRED, not stylistic, and dropping it was measured to be wrong.**
    ``add()`` writes STM and MTM, and an un-narrowed walk dedupes the two copies of one memory to
    its STM representative, so every row comes back ``tier == STM`` and a reader that filtered on
    ``MemoryItem.tier`` afterwards would discard all of them. Measured against the real stores:
    ``tiers=None`` → ``4 rows, tiers=['stm','stm','stm','stm']``; ``tiers={MTM, LTM}`` → the same
    four as MTM. ``tier`` from an un-narrowed walk is a REPRESENTATIVE, not a durability signal.
    """
    reader, partition, _t = _reader([make_item(memory_id="m0")])
    await reader.evidence_for(ns, limit=10)

    _ns, states, tiers, _limit = partition.enumerate_calls[0]
    assert states == frozenset({State.ACTIVE})
    assert tiers == frozenset({Tier.MTM, Tier.LTM})


def test_the_readers_tier_rule_and_the_services_tier_rule_agree():
    """The same rule is stated in two modules (``reader._EVIDENCE_TIERS`` and
    ``service._INCREMENTAL_TIERS``). Two statements of one rule drift; this is the pin that makes
    them fail together instead."""
    from mu_engine.services.persona.reader import _EVIDENCE_TIERS

    assert _EVIDENCE_TIERS == _INCREMENTAL_TIERS


async def test_the_walk_stops_at_the_limit_even_with_more_pages_available(
    ns: Namespace, make_item: Callable[..., MemoryItem]
):
    items = [make_item(memory_id=f"m{i}") for i in range(50)]
    reader, partition, _t = _reader(items, page=5)

    got = await reader.evidence_for(ns, limit=12)

    assert len(got) == 12
    # The bound that matters is per CALL: the reader never asks a store for more than the caller
    # still needs, and never more than one configured page. (The first draft of this assertion
    # summed the requested limits across pages and was simply wrong about the invariant — a
    # cursor-paged walk legitimately re-asks for the shrinking remainder.)
    assert all(limit <= 12 for *_rest, limit in partition.enumerate_calls)
    assert all(
        limit <= PersonaSettings().evidence_page_size for *_rest, limit in partition.enumerate_calls
    )


async def test_an_empty_page_ends_the_walk_because_the_next_call_would_raise(
    ns: Namespace, make_item: Callable[..., MemoryItem]
):
    """The bound that keeps persona ALIVE on a real container, and it is not an optimisation.

    A tier-narrowed walk's continuation is never ``None`` while STM is bound: the façade carries
    the narrowed-away tier's position forward, hands back ONE empty stalled page
    (``services/memory/cursor.py:MAX_STALLED_PAGES == 1``), and RAISES
    ``TierRepositoryUnavailableError`` on the next call. Measured against the real stores before
    this bound existed: page 1 → ``4 rows, cursor=…{"stm": ""}``, page 2 → empty, page 3 → raise.
    Every sleep-time rebuild degraded and NO persona could ever be written.

    So the walk stops at the FIRST empty page — one call, never two. Removing the ``not page``
    term makes this test fail on the count, and makes the real subsystem fail on the store.
    """

    class NeverEnding(FakePartition):
        async def enumerate(self, ns: Namespace, **kw: object) -> tuple[list[MemoryItem], str]:
            self.enumerate_calls.append((ns, frozenset(), None, 0))
            return [], "always-more"

    partition = NeverEnding([])
    reader = PartitionPersonaEvidenceReader(
        repo=partition,
        tagger=CountingTagger(),
        settings=PersonaSettings(max_evidence_pages=4),
    )

    assert await reader.evidence_for(ns, limit=100) == []
    assert len(partition.enumerate_calls) == 1


async def test_the_page_cap_still_bounds_a_walk_that_keeps_returning_rows(
    ns: Namespace, make_item: Callable[..., MemoryItem]
):
    """``max_evidence_pages`` is the OTHER half of the bound: a façade that keeps returning a
    short, non-empty page with a continuation would page forever without it. Both terms are
    needed — the empty-page term never fires here."""

    class Endless(FakePartition):
        async def enumerate(self, ns: Namespace, **kw: object) -> tuple[list[MemoryItem], str]:
            self.enumerate_calls.append((ns, frozenset(), None, 0))
            return [make_item(memory_id=f"m{len(self.enumerate_calls)}")], "always-more"

    partition = Endless([])
    reader = PartitionPersonaEvidenceReader(
        repo=partition,
        tagger=CountingTagger(),
        settings=PersonaSettings(max_evidence_pages=4),
    )

    got = await reader.evidence_for(ns, limit=1000)

    assert len(partition.enumerate_calls) == 4
    assert len(got) == 4


async def test_both_reads_refuse_a_shared_namespace_before_touching_the_store(
    shared_ns: Namespace, make_item: Callable[..., MemoryItem]
):
    """§0 line 53: a room has no personality, so a room's partition is never READ for one."""
    reader, partition, _t = _reader([make_item(memory_id="m0")])

    with pytest.raises(NamespaceIsolationError, match="not found"):
        await reader.evidence_for(shared_ns, limit=10)
    with pytest.raises(NamespaceIsolationError, match="not found"):
        await reader.evidence_for_ids(shared_ns, frozenset({"m0"}))
    assert partition.enumerate_calls == []
    assert partition.get_calls == []


async def test_the_incremental_read_fetches_exactly_the_ids_and_drops_the_wrong_tier(
    ns: Namespace, make_item: Callable[..., MemoryItem]
):
    stm_row = make_item(memory_id="m1").model_copy(update={"tier": Tier.STM})
    reader, partition, _t = _reader([make_item(memory_id="m0"), stm_row])

    got = await reader.evidence_for_ids(ns, frozenset({"m0", "m1", "missing"}))

    assert partition.get_calls == ["m0", "m1", "missing"]  # sorted, deterministic
    assert [ev.item.id for ev in got] == ["m0"]


# ---------------------------------------------------------------------------- the classifier
def _tagger(reply: str) -> tuple[ClassifierSlotTagger, StubRouter]:
    router = StubRouter([reply])
    return ClassifierSlotTagger(router=router, settings=PersonaSettings()), router


async def test_the_tagger_routes_on_classify_and_asks_for_json(
    make_item: Callable[..., MemoryItem],
):
    """Spec line 238: persona reads ``models.classify_model`` and never invents a persona model.
    Routing by ``Task.CLASSIFY`` is what makes that structural."""
    tagger, router = _tagger('{"tags": []}')
    await tagger.tag([make_item(memory_id="m0")])

    task, _messages, max_tokens, temperature, response_format = router.calls[0]
    assert task is Task.CLASSIFY
    assert response_format == "json_object"
    assert (max_tokens, temperature) == (
        PersonaSettings().tagger_max_tokens,
        PersonaSettings().tagger_temperature,
    )


async def test_a_good_reply_becomes_tags_keyed_by_the_right_memory(
    make_item: Callable[..., MemoryItem],
):
    tagger, _router = _tagger(
        '{"tags": [{"index": 1, "slot": "expertise", "value": "rust", "confidence": 0.7}]}'
    )
    got = await tagger.tag([make_item(memory_id="m0"), make_item(memory_id="m1")])

    assert got["m0"] == ()
    assert got["m1"] == (SlotTag(slot=PersonaSlot.EXPERTISE, value="rust", confidence=0.7),)


@pytest.mark.parametrize(
    "row",
    [
        '{"index": 9, "slot": "expertise", "value": "rust", "confidence": 0.7}',  # out of batch
        '{"index": 0, "slot": "not_a_slot", "value": "x", "confidence": 0.7}',  # unknown slot
        '{"index": 0, "slot": "expertise", "value": "", "confidence": 0.7}',  # empty value
        '{"index": 0, "slot": "expertise", "value": "x", "confidence": 4}',  # out of range
        '{"index": true, "slot": "expertise", "value": "x", "confidence": 0.7}',  # bool != index
        '"not an object"',
    ],
)
async def test_an_unusable_row_is_dropped_and_only_that_row(
    row: str, make_item: Callable[..., MemoryItem]
):
    """An out-of-batch index is the dangerous one: it would attribute one memory's trait to
    another memory — a provenance corruption no later layer could detect."""
    tagger, _router = _tagger(f'{{"tags": [{row}]}}')
    got = await tagger.tag([make_item(memory_id="m0")])
    assert got == {"m0": ()}


@pytest.mark.parametrize("reply", ["not json", "[]", '{"nope": 1}'])
async def test_a_malformed_envelope_raises(reply: str, make_item: Callable[..., MemoryItem]):
    """A model that did not answer at all is not the same as a model that answered "no slots".
    The first must reach the caller's named degrade; the second must not."""
    tagger, _router = _tagger(reply)
    with pytest.raises(ValueError, match="persona slot-tag reply"):
        await tagger.tag([make_item(memory_id="m0")])


def test_the_prompt_names_the_same_index_key_the_parser_reads() -> None:
    """The defect this pins was live and MEASURED, not hypothetical.

    The first prompt addressed a statement by ``"i"`` and showed the model an envelope of
    ``<placeholders>`` instead of real JSON. Against the one model this project deploys
    (``qwen2.5:0.5b``), every reply put a SLOT NAME in ``"i"`` and ``_parse_row`` rejected 4 rows
    out of 4 — a classifier that returns nothing, and therefore a persona that is composed,
    subscribed, driven and permanently empty. That is the exact failure mode this subsystem was
    re-composed to remove, one layer further down, and no unit test could see it because every
    unit test writes the reply itself.

    The prompt and the parser are two halves of one wire contract. This asserts they agree.
    """
    assert f'"{_KEY_INDEX}"' in _system_prompt()


def test_every_persona_slot_has_a_gloss_and_the_prompt_shows_all_of_them() -> None:
    """The gloss table is part of the wire contract with the model, not documentation.

    MEASURED: given the bare slot NAMES, the deployed model answered with `"follows"`, `"topic"`
    and `"coffee"` — words lifted from the statement — and `_parse_row` discarded three rows out
    of four. With one clause of meaning per name it copied names off the list instead. So a slot
    that ships with no gloss is a slot the classifier will not reliably choose, and this is the
    pin that stops one from shipping.
    """
    assert set(_SLOT_GLOSS) == set(PersonaSlot)
    prompt = _system_prompt()
    for slot, gloss in _SLOT_GLOSS.items():
        assert f"{slot.value} — {gloss}" in prompt


def test_the_prompts_worked_example_survives_this_modules_own_parser(
    make_item: Callable[..., MemoryItem],
) -> None:
    """Stronger than the key check: the example reply we SHOW the model is parsed by the code that
    will read the model's imitation of it. A prompt that demonstrates a row the parser would drop
    teaches the model to fail."""
    prompt = _system_prompt()
    start = prompt.index('{"tags": [{')
    example = prompt[start : prompt.index("}]}", start) + len("}]}")]

    rows = _parse_envelope(example)
    batch = [make_item(memory_id="m0"), make_item(memory_id="m1")]
    parsed = [_parse_row(row, batch) for row in rows]

    assert parsed == [
        ("m0", SlotTag(slot=PersonaSlot.OCCUPATION, value="pastry chef", confidence=0.9)),
        ("m1", SlotTag(slot=PersonaSlot.FOLLOWED_TOPIC, value="sourdough", confidence=0.8)),
    ]


async def test_the_batch_bound_is_real(make_item: Callable[..., MemoryItem]):
    router = StubRouter(['{"tags": []}'] * 3)
    tagger = ClassifierSlotTagger(router=router, settings=PersonaSettings(tagger_batch_size=2))

    await tagger.tag([make_item(memory_id=f"m{i}") for i in range(5)])

    assert len(router.calls) == 3
