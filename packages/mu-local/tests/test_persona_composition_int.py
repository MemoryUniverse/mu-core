"""Persona, COMPOSED and DRIVEN and proved BY CONSEQUENCE — MVP-SPEC §5.1.9 acceptance (d)+(e).

The defect this file exists to close, quoted from the spec (`docs/specs/MVP-SPEC.md:854`):
*"`PersonaService` has zero construction sites in any `src/` tree in any of the three repos;
`git grep -i persona -- packages/mu-local/src packages/mu-engine-server/src` returns nothing."*
Reproduced at `dev/mlm-build@eaf6c00` before this wave: both greps returned nothing, and
`mu-client`'s inject assembler was budgeting **15 % of the context window** for a
`<persona>` string that could never be non-empty (`live_context.py:696,924`).

So the point of these tests is NOT that a `PersonaProfile` round-trips — the unit suite already
proves that, and proved it while the subsystem was dead. The point is the two things a dead
subsystem cannot do:

* **(d) COMPOSED + DRIVEN.** `LocalContainer` holds a real `PersonaService`, and a real
  `consolidate()` over real stores reaches it through the real `InprocBus` — a persona write
  lands on the SWEEP span, and NOT on the ingest span, which is the property the service's sync
  `note_promoted` was designed for and which nothing could observe while nothing composed it.
* **(e) CONSEQUENCE.** A persona *changes what recall returns*. Same authorized candidate set,
  different ORDER (`persona-design.md` §5.4 rule 4(c)), bounded by `affinity_weight`.

REAL, zero mocks (DEV-STANDARDS, non-negotiable): the live `mu-dev-cache`/`mu-dev-qdrant`/
`mu-dev-falkordb` containers, the real offline MiniLM embedder, the real `ModelRouter` over the
real `mu-dev-slm` Ollama endpoint, the real shipped `InMemoryPersonaRepository`. Nothing here is
patched, stubbed or monkeypatched. If a store is down the fixture RAISES (BLOCKED, never faked);
if the SLM is down the two model-dependent tests SKIP on an env probe, the house convention from
`test_local_llm_slm_int.py`.

**Why one test seeds a profile instead of asking the model for one.** The consequence assertion
has to be deterministic (DEV-STANDARDS: no flaky tests), and it must hold for EVERY persona, not
only for whichever portrait a 0.5B model happens to write today. So
`test_a_real_persona_reorders_a_real_recall_and_changes_nothing_else` writes a real
`PersonaProfile` through the real `PersonaRepository` port — a real adapter, real data, no double
— and asserts the re-ordering against a real recall over real stores.
`test_a_real_sleeptime_tick_builds_a_real_persona_from_real_memories` is the other half: it seeds
nothing and proves the model path actually produces a profile and a non-empty brief.
"""

from __future__ import annotations

import asyncio
import contextlib
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_contracts.domain.events import DegradedModeEntered, SleeptimeTick
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.persona import PersonaProfile, PersonaSlot, SlotValue
from mu_engine.providers._contracts import ModelGroupUnavailableError
from mu_engine.services.persona import (
    AFFINITY_SLOTS,
    PersonaService,
    PersonaSettings,
    PersonaShapedRankedRead,
    persona_key,
)
from mu_engine.services.persona.reader import PersonaTaggerUnusableError
from mu_engine.services.persona.shaping import _WORD
from mu_engine.services.recall.dto import RecallItemView as _EngineRecallItem
from mu_engine.services.recall.dto import RecallQuery
from mu_engine.services.recall.dto import RecallResult as _EngineRecallResult
from mu_local import LocalMemory
from mu_local.config import ModelProfileSettings, StorageSettings

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"

#: The four memories. Three of them evidence a persona slot about the USER; the fourth is a fact
#: about the world, which the classifier is told (and, measured, manages) to leave untagged.
#: The one recall text every consequence assertion uses. Fixed, so the two reads it compares
#: differ in exactly one thing: whether a persona exists.
_QUERY = "what does the user care about?"

_MEMORIES = (
    "The user is an expert in kubernetes operators and writes them for a living.",
    "The user follows quantum computing research every week.",
    "The user prefers dark roast coffee in the morning.",
    "Mount Everest is the tallest mountain on Earth.",
)


class SlmTestSettings(BaseSettings):
    """Central-config home for THIS file's SLM wiring — mirrors `test_local_llm_slm_int.py`
    verbatim (DEV-STANDARDS rule 3: never a literal at the call site)."""

    model_config = SettingsConfigDict(
        env_prefix="MU_TEST_SLM__",
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = "http://127.0.0.1:11435/v1"
    probe_url: str = "http://127.0.0.1:11435"
    probe_timeout_s: float = 2.0
    model: str = "qwen2.5:0.5b"
    max_tokens: int = 512
    temperature: float = 0.0


def _slm_reachable(cfg: SlmTestSettings) -> bool:
    try:
        with urllib.request.urlopen(cfg.probe_url, timeout=cfg.probe_timeout_s) as resp:  # noqa: S310
            return bool(200 <= int(resp.status) < 300)
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


_SLM_CFG = SlmTestSettings()
_SLM_UP = _slm_reachable(_SLM_CFG)
_SLM_REASON = (
    f"mu-dev-slm unreachable at {_SLM_CFG.probe_url} (env probe) — "
    "persona's slot tagger IS models.classify_model (persona-design.md:103), so with no model "
    "there is no persona to compose. Start it: docker compose -f docker-compose.slm.yml up -d"
)


@pytest_asyncio.fixture
async def persona_mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """A `LocalMemory` whose `LocalContainer` composes persona — i.e. one with a model profile."""
    profile = ModelProfileSettings(
        base_url=_SLM_CFG.base_url,
        model=_SLM_CFG.model,
        max_tokens=_SLM_CFG.max_tokens,
        temperature=_SLM_CFG.temperature,
    )
    memory = LocalMemory(
        StorageSettings(llm=profile),
        workspace=f"wsp{uid}",
        namespace=f"orgp{uid}",
        settings=settings,
    )
    try:
        yield memory
    finally:
        await _teardown(settings, f"p{uid}")
        await memory.aclose()


@pytest_asyncio.fixture
async def modelless_mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """The zero-model default — the ABSENCE half of the composition rule."""
    memory = LocalMemory(workspace=f"wsn{uid}", namespace=f"orgn{uid}", settings=settings)
    try:
        yield memory
    finally:
        await _teardown(settings, f"n{uid}")
        await memory.aclose()


async def _teardown(settings: Settings, uid: str) -> None:
    """Drop every qdrant collection / falkordb graph / redis key this run created (rule 14)."""
    qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        for coll in (await qdrant.get_collections()).collections:
            if uid in coll.name:
                with contextlib.suppress(Exception):
                    await qdrant.delete_collection(coll.name)
    finally:
        await qdrant.close()

    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        for g in await db.list_graphs():
            name = g.decode() if isinstance(g, bytes) else g
            if uid in name:
                with contextlib.suppress(Exception):
                    await db.select_graph(name).delete()
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()

    redis: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    try:
        keys = [k async for k in redis.scan_iter(match=f"*{uid}*".encode())]
        if keys:
            await redis.delete(*keys)
    finally:
        await redis.aclose()


def _ns(mem: LocalMemory) -> Namespace:
    """The η these tests act on — built the same way `LocalMemory` builds its own."""
    return Namespace(
        org=mem._org,
        workspace=mem._workspace,
        user=_USER,
        session=_SESSION,
        visibility=Visibility.PRIVATE,
    )


async def _write_memories(mem: LocalMemory) -> None:
    """Write the four memories, earning the STM->MTM promotion explicitly (importance 0.5 sits
    below the 0.6 gate, so the default would stay STM-only and evidence only reads MTM/LTM)."""
    for content in _MEMORIES:
        result = await mem.add(content, user=_USER, session=_SESSION, importance_score=0.9)
        assert result.promoted, "STM->MTM deterministic promotion did not fire"


# ------------------------------------------------------------------------------ (d) COMPOSED
async def test_the_local_container_composes_a_real_persona_service(
    persona_mem: LocalMemory,
) -> None:
    """MVP-SPEC §5.1.9(d), first half: `LocalContainer` EXPOSES a constructed `PersonaService`.

    Before this wave the whole file this asserts against contained the word "persona" zero times.
    """
    container = persona_mem._container
    assert container.persona is not None, "persona did not compose on a model-wired container"
    assert isinstance(container.persona.service, PersonaService)
    # …and the shaper is actually IN the recall path, not merely constructed beside it. This is
    # the line that distinguishes "built" from "driven" on the read side.
    assert isinstance(container.recall, PersonaShapedRankedRead)


async def test_a_model_less_container_leaves_persona_absent_and_recall_unwrapped(
    modelless_mem: LocalMemory,
) -> None:
    """The ABSENCE half. Persona's slot tagger IS `models.classify_model`, so a zero-model
    deployment has no evidence source; the house rule (the same one that leaves `health`/`pin`
    `None`) says stay absent rather than compose something inputless.

    This is also what keeps the recall path byte-identical for every deployment that has no
    model: no decorator, no keyed persona read, no cost."""
    container = modelless_mem._container
    assert container.persona is None
    assert not isinstance(container.recall, PersonaShapedRankedRead)


# -------------------------------------------------------------------------------- (e) DRIVEN
@pytest.mark.skipif(not _SLM_UP, reason=_SLM_REASON)
async def test_a_real_consolidate_reaches_persona_through_the_real_bus(
    persona_mem: LocalMemory,
) -> None:
    """MVP-SPEC §5.1.9(d), first half — the SPAN assertion.

    Two spans, one assertion each:

    * **ingest** — after four `add()` calls, there is STILL no persona. `MemoryPromoted` is
      published INLINE inside `remember()` by a bus that awaits its handlers, so anything the
      persona subsystem did there would run on the user's capture latency. It queues an id and
      returns; `PersonaService.note_promoted` is a plain `def` precisely so it CANNOT await a
      store or a model, and this is where that design is observed rather than described.
    * **sweep** — a real `consolidate()` publishes `ConsolidationCompleted` on the real
      `InprocBus`, and the persona sweeper's per-user counter advances. That counter is the proof
      the subscription `build_persona` installs is live on the SAME bus the user's own verbs
      publish to — the single fact that was false at HEAD, where neither composition root
      mentioned persona at all.

    `consolidate()` also runs DISTILL's own LLM extractor, which is not what this test is about;
    a shared-endpoint outage there SKIPS rather than reporting a persona failure that did not
    happen. The persona chain itself is proved end-to-end by the next test, which does not go
    through distill.
    """
    ns = _ns(persona_mem)
    container = persona_mem._container
    assert container.persona is not None
    repo = container.persona.repository

    await _write_memories(persona_mem)

    # --- the INGEST span: work was QUEUED, nothing was written -------------------------------
    assert await repo.get(ns) is None, "a persona write happened on the ingest span"
    assert (
        persona_key(ns) in container.persona.service._pending
    ), "MemoryPromoted never reached PersonaService — the bus subscription is not wired"

    # --- the SWEEP span ----------------------------------------------------------------------
    try:
        await persona_mem.consolidate(user=_USER, session=_SESSION)
    except ModelGroupUnavailableError as exc:  # DISTILL's extractor, not persona's classifier
        pytest.skip(f"the shared SLM was unavailable to DISTILL, so no sweep ran: {exc}")

    assert (
        container.persona.sweeper._ticks.get(persona_key(ns)) == 1
    ), "ConsolidationCompleted never reached the persona sweeper"


@pytest.mark.skipif(not _SLM_UP, reason=_SLM_REASON)
async def test_a_real_sleeptime_tick_builds_a_real_persona_from_real_memories(
    persona_mem: LocalMemory,
) -> None:
    """MVP-SPEC §5.1.9(d) second half and (e) — the whole persona chain, on real everything.

    The trigger is `SleeptimeTick`, published on the container's OWN `InprocBus`. That is the
    event `persona-design.md` §2.4 line 119 names, and it had **zero publishers anywhere in this
    repo** — so this is also the assertion that the designed subscription is live, not only the
    substitute one the previous test drives (`ConsolidationCompleted`, the shipped sleep-time
    signal; ARCHITECTURE-DELTAS).

    Everything downstream of that publish is real and unmocked: the real partition walk over
    Valkey/Qdrant/FalkorDB, the real `ClassifierSlotTagger` against the real SLM, the real
    aggregator, the real two-level synthesizer, the real repository.
    """
    ns = _ns(persona_mem)
    container = persona_mem._container
    assert container.persona is not None
    repo = container.persona.repository

    await _write_memories(persona_mem)
    assert await repo.get(ns) is None

    degrades = _watch_degrades(container)
    await container.bus.publish(SleeptimeTick(namespace=ns))

    assert (
        container.persona.sweeper._ticks.get(persona_key(ns)) == 1
    ), "SleeptimeTick never reached the persona sweeper — the DESIGNED trigger is not subscribed"
    # A clean sweep emits NO persona degrade. This assertion is the one that turns the old
    # "flaky" failure into a diagnosis: when the classifier returns nothing usable the sweep now
    # names it, and this line fails with the REASON in the message instead of with a bare
    # `profile is None` that could equally mean min_support, a store, or a dead subscription.
    assert not degrades, f"the sweep degraded: {[(d.mode, d.reason, d.detail) for d in degrades]}"
    profile = await repo.get(ns)
    assert profile is not None, (
        "the sweep ran but wrote no persona — either the classifier returned no usable tag on "
        "any of the four memories, or min_support was not reached"
    )
    assert profile.version == 1
    assert profile.slots, "a persona with no slots is the dead subsystem wearing a version number"
    assert profile.source_memory_count >= PersonaSettings().min_support

    # (e) A rendered `<persona>` section is NON-EMPTY. `load_brief` is the exact read the inject
    # assembler's `persona_brief` is fed from, so this is the byte the 15 % budget was reserved
    # for — measured, not assumed.
    brief = await repo.load_brief(ns)
    assert brief is not None
    body, etag = brief
    assert body.strip(), "overall_brief is empty — the <persona> section would still render blank"
    assert len(body) <= PersonaSettings().brief_char_limit
    assert etag


@pytest.mark.skipif(not _SLM_UP, reason=_SLM_REASON)
async def test_the_real_classifier_builds_a_persona_from_every_rotation_of_the_real_batch(
    persona_mem: LocalMemory,
) -> None:
    """The REAL defect behind the word "flaky", asserted against the real model.

    The sweep that "failed in the full suite and passed alone" was never about the suite. It was
    measured, on the VM, against the real `qwen2.5:0.5b`:

    * **the classifier is deterministic** — the same prompt at ``tagger_temperature`` 0.0 returned
      the byte-identical reply on 6 consecutive serial calls and on 12 concurrent ones, so
      concurrency and suite load change nothing;
    * **it is ORDER-sensitive** — over all 24 orderings of these same four memories, 3 produced
      ZERO usable verdicts: the model abandons the slot list and answers ``"mountain"``,
      ``"coffee"``, ``"kubernetes_operator"``, names lifted from the statements;
    * **the order is the store's, and it is a different one every run** — the evidence walk was
      stable 12/12 within a namespace and came back ``(1,2,0,3)``, ``(3,0,2,1)`` and ``(2,1,3,0)``
      in three consecutive fresh namespaces. Every run of this file rolls a fresh ``uid``, so
      every run drew a fresh order, and roughly one run in eight drew a losing one.

    So the test that asserts a persona gets built was a lottery, and the pipeline's answer to
    losing it was to drop all four verdicts and write nothing, at ``DEBUG``.

    This test removes the lottery instead of re-rolling it. It runs the REAL tagger over the REAL
    memories in every ROTATION of the walk order — four rotations, so whichever statement the
    store happens to put first, one of them is the world-fact-first order that was measured to
    break the model — and asserts every one of them still yields a verdict. Setting
    ``tagger_unusable_retries=0`` makes it RED.
    """
    ns = _ns(persona_mem)
    container = persona_mem._container
    assert container.persona is not None
    await _write_memories(persona_mem)

    reader = container.persona.service._evidence
    items = await reader._walk(ns, limit=PersonaSettings().max_evidence_items)
    assert len(items) == len(_MEMORIES), "the walk did not return the four memories to classify"

    for offset in range(len(items)):
        rotated = [*items[offset:], *items[:offset]]
        tags = await reader._tagger.tag(rotated)
        assert any(tags[item.id] for item in rotated), (
            f"rotation {offset} produced no usable slot tag for ANY of the four memories — a "
            "sweep on this order writes an empty persona and the 15 % inject budget is spent on "
            "nothing"
        )


async def test_the_named_persona_degrade_reaches_a_real_subscriber_on_the_container_bus(
    persona_mem: LocalMemory,
) -> None:
    """The DEGRADE contract's LAST leg, on the real container and the real bus.

    The rotation above makes the failure rare; it cannot make it impossible, because no prompt
    makes a 0.5B model correct. So the terminal state has to be honest — and "honest" ends at an
    operator's subscriber, which is the one leg no unit test can reach.

    **What this proves, exactly.** The container's OWN ``InprocBus`` — the one ``build_persona``
    subscribed the sweeper to — delivers the persona degrade to a real subscriber with the exact
    payload an operator consumes: component ``persona``, mode ``persona_no_usable_tags``, reason
    ``persona_tagger_unusable``. A rename of either constant, or a bus that stopped carrying
    ``DegradedModeEntered``, is RED here.

    **What it does NOT prove, said plainly rather than implied.** It does not drive the raise
    through ``PersonaService``, because nothing can make the real model return an unusable reply
    on demand and a stand-in tagger is a mock — barred in an integration test (DEV-STANDARDS).
    That leg (tagger raises -> service raises + audits -> sweeper emits THIS payload) is pinned
    in ``test_persona_reader_unit`` / ``test_persona_service_unit`` / ``test_persona_driver_unit``,
    where a double is sanctioned. The two halves meet on the constants asserted below.
    """
    ns = _ns(persona_mem)
    container = persona_mem._container
    assert container.persona is not None
    degrades = _watch_degrades(container)

    await container.persona.sweeper._degrade(
        ns,
        detail=PersonaTaggerUnusableError.__name__,
        mode=PersonaTaggerUnusableError.mode,
        reason=PersonaTaggerUnusableError.reason,
    )

    assert [(d.component, d.mode, d.reason.value, d.detail) for d in degrades] == [
        (
            "persona",
            "persona_no_usable_tags",
            "persona_tagger_unusable",
            "PersonaTaggerUnusableError",
        )
    ]


# ---------------------------------------------------------------------------- (e) CONSEQUENCE
async def test_a_real_persona_reorders_a_real_recall_and_changes_nothing_else(
    persona_mem: LocalMemory,
) -> None:
    """`persona-design.md` §5.4 rule 4(c), as an executable assertion: *"two callers with opposite
    personas over the same PRIVATE partition get the SAME candidate set, differing only in
    ORDER"*.

    This is the test the phrase "prove it BY CONSEQUENCE" names. It runs the SAME real recall over
    the SAME real stores twice — once with no persona, once with a real profile written through
    the real `PersonaRepository` port — and asserts the caller SEES a different answer.

    **The target hit is CHOSEN FROM THE RUN'S OWN SCORES, not hardcoded**, and that is what makes
    the assertion honest rather than lucky. §5.2 bounds the nudge: a matching hit is multiplied by
    at most `1 + affinity_weight * confidence`, so persona can only overtake a neighbour that is
    within that factor. The test picks the highest-ranked hit that a bounded nudge is ALLOWED to
    lift, gives the persona a term unique to that hit, and asserts it actually overtakes. A test
    that asserted "persona puts hit N first" unconditionally would be asserting that persona
    dominates semantic evidence — the exact thing §5.2 forbids.

    The persona is then ERASED and the ranker's own order must come back byte-for-byte: a
    subsystem that permanently perturbs a read is not a relevance prior, it is a corrupted index.
    """
    container = persona_mem._container
    assert container.persona is not None
    ns = _ns(persona_mem)
    settings = PersonaSettings()

    await _write_memories(persona_mem)

    shaped_read = container.recall
    assert isinstance(shaped_read, PersonaShapedRankedRead)
    inner = shaped_read._inner  # the SAME RecallService, undecorated
    scope = persona_mem._scope(_USER, _SESSION)
    query = RecallQuery(namespace=ns, text=_QUERY, limit=10)

    unshaped = await _eventually(lambda: inner.recall(scope, query))
    assert (
        len(unshaped.items) >= 2
    ), "the ranker returned fewer than two hits — there is nothing an ORDER assertion can say"

    confidence = 0.9
    ceiling = 1.0 + settings.affinity_weight * confidence
    target = _liftable_hit(unshaped.items, ceiling=ceiling)
    assert target is not None, (
        "no hit in this run is within the bounded-nudge factor of the one above it, so a "
        "§5.2-conformant persona could not legally re-order this result at all"
    )
    index, hit = target
    term = _term_unique_to(hit, unshaped.items)
    assert term is not None, "no affinity term is unique to the target hit"
    # The shaper's own matching rule, asserted BEFORE the consequence, so a failure below can
    # only mean "persona did not act", never "the test picked a term persona cannot see".
    matched = [i.memory_id for i in unshaped.items if term in _shaper_words(i.content)]
    assert matched == [hit.memory_id], (
        f"the chosen term matches {len(matched)} hits, not exactly the target — "
        f"scores={[round(i.fused_score, 6) for i in unshaped.items]} "
        f"floor={[i.is_floor for i in unshaped.items]}"
    )

    await container.persona.repository.upsert(
        PersonaProfile(
            namespace=ns,
            slots={
                # An AFFINITY slot (`shaping.AFFINITY_SLOTS`). `response_style` describes HOW to
                # speak, not WHAT the user cares about, and correctly moves nothing.
                PersonaSlot.EXPERTISE: SlotValue(
                    value=term,
                    confidence=confidence,
                    support_ids=(hit.memory_id,),
                    updated_at=datetime.now(UTC),
                )
            },
            overall_brief=f"Works on {term}.",
            brief_etag="e" * 8,
            version=1,
            rebuilt_at=datetime.now(UTC),
            source_memory_count=1,
        )
    )
    assert PersonaSlot.EXPERTISE in AFFINITY_SLOTS
    assert settings.affinity_min_confidence <= confidence
    assert len(term) >= settings.affinity_min_term_chars

    shaped = await shaped_read.recall(scope, query)

    # 1. The CANDIDATE SET is untouched — persona is voice and relevance, NEVER access.
    assert {i.memory_id for i in shaped.items} == {i.memory_id for i in unshaped.items}
    # 2. …and so are the ranker's own scores: persona re-orders, it never rewrites the testimony.
    by_id = {i.memory_id: i.fused_score for i in unshaped.items}
    assert all(i.fused_score == by_id[i.memory_id] for i in shaped.items)
    # 3. The ORDER changed — the thing a DEAD persona can never do.
    shaped_ids = [i.memory_id for i in shaped.items]
    unshaped_ids = [i.memory_id for i in unshaped.items]
    assert (
        shaped_ids != unshaped_ids
    ), "persona changed nothing a caller can see — that is the dead subsystem's behaviour"
    # 4. …in the specific, bounded way §5.2 licenses: the matching hit overtook its neighbour.
    neighbour = unshaped.items[index - 1].memory_id
    assert shaped_ids.index(hit.memory_id) < index, "the matching hit did not move up at all"
    assert shaped_ids.index(hit.memory_id) < shaped_ids.index(
        neighbour
    ), "the matching hit did not overtake the neighbour a bounded nudge licensed it to pass"

    # 5. And it is reversible: erase the persona, get the ranker's own order back exactly.
    assert await container.persona.repository.delete(ns) is True
    again = await shaped_read.recall(scope, query)
    assert [i.memory_id for i in again.items] == unshaped_ids


async def test_a_persona_of_only_voice_slots_changes_no_order(persona_mem: LocalMemory) -> None:
    """The other side of the same rule, and the reason `AFFINITY_SLOTS` is a frozenset rather than
    "every slot": `response_style` and `interaction_pace` say HOW to answer, not WHAT the user
    cares about. Letting them move a ranking would make persona shape RELEVANCE with VOICE.

    Mutating `AFFINITY_SLOTS` to include `PersonaSlot.RESPONSE_STYLE` turns this test RED.
    """
    container = persona_mem._container
    assert container.persona is not None
    ns = _ns(persona_mem)

    await _write_memories(persona_mem)

    shaped_read = container.recall
    assert isinstance(shaped_read, PersonaShapedRankedRead)
    scope = persona_mem._scope(_USER, _SESSION)
    query = RecallQuery(namespace=ns, text=_QUERY, limit=10)

    before = await _eventually(lambda: shaped_read.recall(scope, query))
    assert before.items

    term = _term_unique_to(before.items[-1], before.items)
    assert term is not None
    await container.persona.repository.upsert(
        PersonaProfile(
            namespace=ns,
            slots={
                PersonaSlot.RESPONSE_STYLE: SlotValue(
                    value=term,
                    confidence=1.0,
                    support_ids=(before.items[-1].memory_id,),
                    updated_at=datetime.now(UTC),
                )
            },
            overall_brief=f"Answer in the style of {term}.",
            brief_etag="f" * 8,
            version=1,
            rebuilt_at=datetime.now(UTC),
            source_memory_count=1,
        )
    )
    assert PersonaSlot.RESPONSE_STYLE not in AFFINITY_SLOTS

    after = await shaped_read.recall(scope, query)
    assert [i.memory_id for i in after.items] == [i.memory_id for i in before.items]


# ------------------------------------------------------------------------------------ helpers
def _watch_degrades(container: object) -> list[DegradedModeEntered]:
    """Subscribe a real handler to the container's REAL bus and keep what it publishes.

    Not a mock and not a patch: ``InprocBus.subscribe`` is the same public seam ``build_persona``
    itself uses, and this adds one more subscriber to the live bus. It is the only way to observe
    the degrade the way an operator's consumer would.
    """
    seen: list[DegradedModeEntered] = []

    async def record(event: DegradedModeEntered) -> None:
        if event.component == "persona":
            seen.append(event)

    container.bus.subscribe(DegradedModeEntered, record)  # type: ignore[attr-defined]
    return seen


async def _eventually(read: Callable[[], Awaitable[_EngineRecallResult]]) -> _EngineRecallResult:
    """Poll until the ranker returns hits — qdrant/falkordb apply writes asynchronously. Same
    shape as `test_local_roundtrip_int._eventually`, typed for the ENGINE result the container's
    recall returns (the facade's `RecallResult` conversion is not under test here)."""
    last = await read()
    for _ in range(40):  # ~8s ceiling
        if last.items:
            return last
        await asyncio.sleep(0.2)
        last = await read()
    return last


def _liftable_hit(
    items: Sequence[_EngineRecallItem], *, ceiling: float
) -> tuple[int, _EngineRecallItem] | None:
    """The highest-ranked hit a §5.2-BOUNDED nudge is allowed to lift past its neighbour.

    `ceiling` is `1 + affinity_weight * confidence`, the largest multiplier `PersonaAffinityShaper`
    may apply. A hit whose score times that ceiling still does not reach the hit above it CANNOT
    legally move, so choosing one would be asserting that persona dominates semantic evidence.
    """
    for index in range(1, len(items)):
        above = items[index - 1]
        hit = items[index]
        # Same BLOCK only. The ranker returns `[*protected_floor, *fused_tail]` on two different
        # score scales (`recall/ranker.py:_merge_floor`), and persona may not move a hit across
        # that boundary — so a cross-block pair is not a pair a bounded nudge can act on.
        if hit.is_floor is not above.is_floor:
            continue
        if hit.fused_score * ceiling > above.fused_score:
            return index, hit
    return None


def _term_unique_to(hit: _EngineRecallItem, items: Sequence[_EngineRecallItem]) -> str | None:
    """The longest word in `hit` that appears in NO other hit.

    Unique on purpose: `PersonaAffinityShaper` boosts every hit a term matches, so a term shared
    with a sibling would lift both and prove nothing about WHICH hit persona moved.
    """
    others = {
        word
        for other in items
        if other.memory_id != hit.memory_id
        for word in _words(other.content)
    }
    candidates = sorted(_words(hit.content) - others, key=len, reverse=True)
    return candidates[0] if candidates else None


def _words(content: str) -> set[str]:
    """The shaper's OWN tokenisation, imported rather than re-implemented.

    An earlier version of this helper split on whitespace and stripped punctuation by hand. It
    disagreed with `shaping._WORD` on real content, so the test could pick a "unique" term the
    shaper saw in two hits — which is exactly the class of bug a test must not have. The shaper
    is the authority on what a term is; this helper only chooses among its words.
    """
    return _shaper_words(content)


def _shaper_words(content: str) -> set[str]:
    """`PersonaAffinityShaper`'s exact word grain (`shaping._WORD`), case-folded."""
    return {w.casefold() for w in _WORD.findall(content)}
