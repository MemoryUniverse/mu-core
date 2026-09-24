"""S1b — read-time neighbour expansion, end to end over REAL Redis/Valkey (DEV-STANDARDS: zero
mocks). ``docs/tracking/TRACE-0923.md`` §7/§6.2/§5.1; ADR pending in ``docs/decisions/``.

Proves the two halves together, over the real store, not in isolation:

1. **Write side** (``local_memory.py::_next_turn_seq_base``) — successive ``LocalMemory.add()``
   calls on the same session assign a real, increasing ``MemoryItem.turn_seq``, round-tripped
   through Redis's generic JSON blob (``redis_mapper.py``) with zero adapter changes.
2. **Read side** (``ranker.py::_expand_neighbors``) — a query that only lexically matches an
   "anchor" turn surfaces that anchor's ±1 conversational NEIGHBOUR once
   ``MU_RECALL__NEIGHBOR_EXPAND_RADIUS`` is raised above its inert default (0), even though the
   neighbour shares NOT ONE WORD with the query and sits outside the STM recency floor's own
   10-item window — reproducing §5.1's own failure shape (the retriever finds the right
   conversational moment, in this case failing to at all under the shipped default, and finding
   it only once the neighbour is pulled in).

``MU_RECALL__STM_SCORING=lexical`` and ``MU_RECALL__FLOOR_PROTECT_LIMIT=0`` are pinned for the
SAME reason ``test_config_wiring_int.py`` pins its own knobs: this test's premise is the
neighbour-expansion mechanism specifically, and both the real MiniLM embedder's semantic fuzziness
("embed" mode might independently surface a semantically-related neighbour on its own, which would
make a positive result ambiguous) and the UNCONDITIONAL recency-floor guarantee (a separate,
already-tested mechanism, ``test_recall_ranker_unit.py``) would otherwise contaminate the isolated
claim this test makes.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Iterator

import pytest
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_contracts.contracts.recall import RecallItemView
from mu_engine.config import get_engine_settings
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_local import LocalMemory
from mu_local.config import StorageSettings

pytestmark = pytest.mark.integration

_USER = "u1"
_SESSION = "s1"
_QUERY = "What inspired the painting for the art show?"
# Shares zero content words with `_QUERY` — the point (§5.4-style construction, this repo's own
# established pattern for an isolated lexical-channel test).
_GOLD_NEIGHBOR = (
    "Caroline: I made this after visiting a support center. It gave me unity and strength."
)
_ANCHOR = "Melanie: Wow, that painting for the art show is amazing! What inspired it?"
_FILLERS = (
    "The weather today is sunny",
    "I had cereal for breakfast",
    "The train was late this morning",
    "My favorite color is blue",
    "There was traffic on the highway",
    "The store closes early on Sundays",
    "I need to buy new running shoes",
    "The meeting got rescheduled to Friday",
    "My phone battery died again",
)


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings()


@pytest.fixture(autouse=True)
def _clean_engine_settings_env() -> Iterator[None]:
    """Same leak-guard discipline as ``test_config_wiring_int.py``'s own fixture of this name —
    this module is the only OTHER one touching these three ``MU_RECALL__*`` vars, and the
    ``get_engine_settings`` cache is process-global."""
    keys = (
        "MU_RECALL__NEIGHBOR_EXPAND_RADIUS",
        "MU_RECALL__STM_SCORING",
        "MU_RECALL__FLOOR_PROTECT_LIMIT",
    )
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    get_engine_settings.cache_clear()


def _make_memory(settings: Settings) -> tuple[LocalMemory, str]:
    """Constructs ``LocalMemory`` (and its ``LocalContainer``/``ThreeChannelRecallRanker``, which
    read ``get_engine_settings()`` ONCE at construction time) fresh, so the caller's env-var
    overrides + ``get_engine_settings.cache_clear()`` — which MUST happen BEFORE this call, never
    after — are actually baked into the composed ranker this test then exercises. A module-scoped
    fixture that built ``LocalMemory`` once up front (the pre-fix shape here) would compose it
    against whatever settings were live at COLLECTION time, before any test body's env mutation
    ever ran — exactly the ordering bug ``test_config_wiring_int.py``'s own ``_seed_and_recall``
    helper avoids by constructing fresh, in the same pattern this mirrors."""
    uid = uuid.uuid4().hex[:12]
    memory = LocalMemory(
        StorageSettings(), workspace=f"ws{uid}", namespace=f"org{uid}", settings=settings
    )
    return memory, uid


async def _teardown(settings: Settings, uid: str) -> None:
    redis: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    try:
        keys = [k async for k in redis.scan_iter(match=f"*{uid}*".encode())]
        if keys:
            await redis.delete(*keys)
    finally:
        await redis.aclose()


async def _seed(mem: LocalMemory) -> None:
    """turn_seq 0 = GOLD_NEIGHBOR (oldest — falls OUTSIDE the default 10-item STM recency floor
    once the fillers land), turn_seq 1 = ANCHOR (the only item lexically matching `_QUERY`),
    turn_seq 2..10 = 9 bland fillers (push GOLD_NEIGHBOR out of the recency-10 window while
    keeping ANCHOR inside it)."""
    await mem.add(_GOLD_NEIGHBOR, user=_USER, session=_SESSION)
    await mem.add(_ANCHOR, user=_USER, session=_SESSION)
    for f in _FILLERS:
        await mem.add(f, user=_USER, session=_SESSION)


@pytest.mark.asyncio
async def test_turn_seq_is_persisted_and_increases_across_successive_add_calls(
    settings: Settings,
) -> None:
    """Reaches into the composition root directly (``LocalContainer.stm``, the SAME pattern
    ``test_config_wiring_int.py`` uses for its own internal proofs) because ``turn_seq`` is an
    engine-internal field this phase deliberately does NOT add to the wire-versioned
    ``mu_contracts`` ``MemoryResponse`` (out of this phase's lane — see the ADR); the public
    ``LocalMemory.get()`` verb therefore cannot see it. Real STM (redis) round-trip only."""
    memory, uid = _make_memory(settings)
    try:
        ns = Namespace(
            org=f"org{uid}",
            workspace=f"ws{uid}",
            user=_USER,
            session=_SESSION,
            visibility=Visibility.PRIVATE,
        )
        r0 = await memory.add(_GOLD_NEIGHBOR, user=_USER, session=_SESSION)
        r1 = await memory.add(_ANCHOR, user=_USER, session=_SESSION)

        item0 = await memory._container.stm.get(ns, r0.memory_id)
        item1 = await memory._container.stm.get(ns, r1.memory_id)
        assert item0 is not None and item1 is not None
        assert item0.turn_seq is not None and item1.turn_seq is not None
        assert item1.turn_seq == item0.turn_seq + 1, (
            f"turn_seq did not increase by one across successive add() calls on the same "
            f"session: {item0.turn_seq!r} -> {item1.turn_seq!r}"
        )
    finally:
        await _teardown(settings, uid)
        await memory.aclose()


@pytest.mark.asyncio
async def test_neighbor_expansion_is_off_by_default(settings: Settings) -> None:
    os.environ["MU_RECALL__STM_SCORING"] = "lexical"
    os.environ["MU_RECALL__FLOOR_PROTECT_LIMIT"] = "0"
    get_engine_settings.cache_clear()
    assert get_engine_settings().recall.stm_scoring == "lexical"
    assert get_engine_settings().recall.neighbor_expand_radius == 0

    memory, uid = _make_memory(settings)
    try:
        await _seed(memory)
        result = await _eventually_two(memory)

        contents = [it.content for it in result]
        assert _ANCHOR in contents, f"the anchor did not surface — fixture broken: {contents!r}"
        assert _GOLD_NEIGHBOR not in contents, (
            "the neighbour surfaced with neighbor_expand_radius at its default (0) — it must "
            "not, or the 'inert by default' guarantee (dto.py) is broken"
        )
    finally:
        await _teardown(settings, uid)
        await memory.aclose()


@pytest.mark.asyncio
async def test_neighbor_expansion_surfaces_the_reply_neighbor_when_enabled(
    settings: Settings,
) -> None:
    os.environ["MU_RECALL__STM_SCORING"] = "lexical"
    os.environ["MU_RECALL__FLOOR_PROTECT_LIMIT"] = "0"
    os.environ["MU_RECALL__NEIGHBOR_EXPAND_RADIUS"] = "1"
    get_engine_settings.cache_clear()
    assert get_engine_settings().recall.stm_scoring == "lexical"
    assert get_engine_settings().recall.neighbor_expand_radius == 1

    memory, uid = _make_memory(settings)
    try:
        await _seed(memory)
        result = await _eventually_two(memory)

        contents = [it.content for it in result]
        assert _ANCHOR in contents, f"the anchor itself did not surface: {contents!r}"
        assert _GOLD_NEIGHBOR in contents, (
            "raising MU_RECALL__NEIGHBOR_EXPAND_RADIUS did not surface the anchor's turn_seq "
            f"neighbour — the S1b read path did not reach the composed ranker: {contents!r}"
        )

        # AD-233's OWN blocker, closed end to end and guarded HERE for the first time (VERIFY
        # 2026-09-24). The pass that fixed AD-233 added `turn_seq`/`is_neighbor` to the canonical
        # `mu_contracts.contracts.recall.RecallItemView` and routed all three hand-duplicated
        # projections through one shared mapping — but every test it shipped for that fix was a
        # UNIT test of the DTO or of the mapping function in isolation. Nothing asserted the two
        # fields on what `LocalMemory.recall` actually RETURNS, which is the only surface the eval
        # harness's `getattr(item, "is_neighbor", False)` attribution (AD-228) ever reads. That is
        # precisely the gap AD-233 was: the mechanism worked, the wire dropped it, and a suite full
        # of green unit tests said nothing. Assert it on the real, composed, real-Redis result.
        by_content = {it.content: it for it in result}
        anchor_view = by_content[_ANCHOR]
        neighbor_view = by_content[_GOLD_NEIGHBOR]
        assert neighbor_view.is_neighbor is True, (
            "the expanded neighbour reached LocalMemory.recall's canonical RecallItemView with "
            "is_neighbor=False — the engine->canonical projection is dropping the flag again "
            "(AD-233), so every downstream neighbour attribution is a structural zero"
        )
        assert (
            neighbor_view.turn_seq is not None
        ), "the expanded neighbour's turn_seq did not survive the canonical projection"
        assert (
            anchor_view.is_neighbor is False
        ), "the ANCHOR was marked is_neighbor — the flag is being stamped on the wrong rows"
        assert anchor_view.turn_seq is not None, (
            "the anchor carried no turn_seq on the canonical surface, so nothing downstream can "
            "tell which conversational turn a returned row came from"
        )
        assert neighbor_view.turn_seq == anchor_view.turn_seq - 1, (
            f"the neighbour's turn_seq is not the anchor's -1: {neighbor_view.turn_seq!r} vs "
            f"{anchor_view.turn_seq!r}"
        )
    finally:
        await _teardown(settings, uid)
        await memory.aclose()


async def _eventually_two(memory: LocalMemory) -> list[RecallItemView]:
    """STM is synchronous-write (redis), so no eventual-consistency poll is strictly required —
    kept anyway (short, bounded) for parity with this suite's other real-store tests and as a
    defence against a future write path that becomes async."""
    last: list[RecallItemView] = []
    for _ in range(10):
        result = await memory.recall(
            _QUERY, user=_USER, session=_SESSION, tier=MemoryTier.STM, limit=2
        )
        last = list(result.items)
        if len(last) == 2:
            break
        await asyncio.sleep(0.2)
    return last
