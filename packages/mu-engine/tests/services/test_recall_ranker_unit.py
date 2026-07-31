"""``ThreeChannelRecallRanker`` unit tests — the data-quality assessment #1 fix.

Repro + regression test for ``docs/tracking/DATA-QUALITY-ASSESSMENT.md`` §3.1/#1: a
``recall()`` within one session used to return the WHOLE STM recency floor, unconditionally,
in insertion order, with ZERO room left for the query-relevant MTM channel — every query in a
session with >= ``limit`` STM items produced a byte-identical, query-blind list.

Pure unit test: fake in-memory STM/MTM/LTM tier repos (no containers, no network) so the ranker's
fuse/protect logic is exercised in isolation, deterministically, in milliseconds.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.model.recall import CallerIdentitySet, Vector
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored

_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)


def _item(content: str, *, tier: MemoryTier, at: datetime) -> MemoryItem:
    return MemoryItem(
        content=content,
        kind=MemoryKind.PROPOSITION,
        namespace=_NS,
        state=MemoryState.ACTIVE,
        tier=tier,
        owner_id=_NS.user,
        workspace_id=_NS.workspace,
        session_id=_NS.session,
        created_at=at,
        updated_at=at,
    )


class _FakeStm:
    """A session with N items, oldest-first insertion — ``recent()`` returns newest-first
    (mirrors ``RedisStmAdapter._recent_impl``: ZREVRANGE, constant score=1.0, is_floor=True)."""

    def __init__(self, items: list[MemoryItem]) -> None:
        self._items = items  # insertion order, oldest first

    async def put(self, item: MemoryItem) -> None:  # pragma: no cover - unused this test
        self._items.append(item)

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        return next((i for i in self._items if i.id == memory_id), None)

    async def recent(self, ns: Namespace, *, limit: int) -> list[Scored[MemoryItem]]:
        newest_first = list(reversed(self._items))[:limit]
        return [
            Scored(item=i, score=1.0, channel=RecallChannel.STM_FLOOR, rank=rank, is_floor=True)
            for rank, i in enumerate(newest_first)
        ]

    async def evict(self, ns: Namespace, memory_id: str) -> None:  # pragma: no cover
        self._items = [i for i in self._items if i.id != memory_id]


class _FakeMtm:
    """Returns a caller-supplied ranked hit list for a given query vector — a stand-in for real
    cosine search: the test controls exactly what "relevance" looks like per query."""

    def __init__(self, hits_by_query: dict[tuple[float, ...], list[MemoryItem]]) -> None:
        self._hits_by_query = hits_by_query

    async def upsert(self, item: MemoryItem) -> None:  # pragma: no cover - unused this test
        raise NotImplementedError

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: object | None = None,
    ) -> list[Scored[MemoryItem]]:
        hits = self._hits_by_query.get(tuple(query_vector), [])
        return [
            Scored(item=i, score=1.0 - 0.01 * rank, channel=RecallChannel.MTM_DENSE, rank=rank)
            for rank, i in enumerate(hits[:limit])
        ]

    async def invalidate(self, *a: object, **k: object) -> None:  # pragma: no cover
        raise NotImplementedError

    async def remove(self, ns: Namespace, memory_id: str) -> None:  # pragma: no cover
        raise NotImplementedError


class _EmptyLtm:
    """No LTM edges wired for this test — isolates the STM/MTM interaction under test."""

    async def upsert_fact(self, item: MemoryItem) -> None:  # pragma: no cover
        raise NotImplementedError

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        return []

    async def facts_at(self, *a: object, **k: object) -> list[MemoryItem]:  # pragma: no cover
        raise NotImplementedError

    async def find_conflicts(self, *a: object, **k: object) -> list[MemoryItem]:  # pragma: no cover
        raise NotImplementedError

    async def invalidate(self, *a: object, **k: object) -> None:  # pragma: no cover
        raise NotImplementedError

    async def resolve_entity(self, *a: object, **k: object) -> object:  # pragma: no cover
        raise NotImplementedError


def _build_ranker(
    *, stm_items: list[MemoryItem], mtm_hits_by_query: dict[tuple[float, ...], list[MemoryItem]]
) -> ThreeChannelRecallRanker:
    return ThreeChannelRecallRanker(
        stm=_FakeStm(stm_items),  # type: ignore[arg-type]
        mtm=_FakeMtm(mtm_hits_by_query),  # type: ignore[arg-type]
        ltm=_EmptyLtm(),  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(),
        clock=FrozenClock(datetime(2026, 7, 31, tzinfo=UTC)),
    )


async def _rank(
    ranker: ThreeChannelRecallRanker, query_vec: Vector, *, limit: int = 10
) -> list[str]:
    result = await ranker.rank(
        _NS,
        "irrelevant-this-phase",
        query_vec,
        limit=limit,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )
    return [it.memory_id for it in result.items]


@pytest.mark.asyncio
async def test_full_session_floor_no_longer_swamps_the_relevant_mtm_hit() -> None:
    """THE BUG (§3.1/#1): a session with exactly ``limit`` STM items used to consume the entire
    result budget, so an MTM hit relevant to the query NEVER made it into the top-K. After the
    fix, a query whose MTM channel ranks a specific item #1 must surface that item in the result,
    even though the session already holds 10 (== default limit) STM items."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    stm_items = [
        _item(f"session chatter #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n))
        for n in range(10)  # == RecallSettings().recency_floor_limit == default limit
    ]
    target = _item("Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(stm_items=stm_items, mtm_hits_by_query={query_vec: [target]})

    ids = await _rank(ranker, list(query_vec))

    assert target.id in ids, "query-relevant MTM hit was crowded out by the STM floor"


@pytest.mark.asyncio
async def test_nonsense_query_yields_a_different_result_than_a_targeted_query() -> None:
    """THE BUG's headline symptom: four different queries in the same session returned a
    BYTE-IDENTICAL top-10 (assessment §3.1). With the MTM channel now genuinely contributing to
    the fused result (instead of being crowded out), a query whose MTM channel has no relevant
    hit must NOT return the same list as a query whose MTM channel surfaces a specific item."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    stm_items = [
        _item(f"session chatter #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n))
        for n in range(10)
    ]
    target = _item("Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base)
    targeted_vec = (0.9, 0.1)
    nonsense_vec = (0.0, 0.0)
    ranker = _build_ranker(stm_items=stm_items, mtm_hits_by_query={targeted_vec: [target]})

    targeted_ids = await _rank(ranker, list(targeted_vec))
    nonsense_ids = await _rank(ranker, list(nonsense_vec))

    assert target.id in targeted_ids
    assert target.id not in nonsense_ids
    assert targeted_ids != nonsense_ids, "recall is still query-blind (session-dump artifact)"


@pytest.mark.asyncio
async def test_most_recent_facts_are_still_unconditionally_protected() -> None:
    """The recency-floor INTENT (never evict a just-said fact) must survive the fix: the
    ``floor_protect_limit`` most-recent STM items are still guaranteed present + flagged
    ``is_floor=True``, even when the MTM channel has an unrelated, highly-ranked hit."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    stm_items = [
        _item(f"session chatter #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n))
        for n in range(10)
    ]
    just_said = stm_items[-1]  # most recently added
    other_hit = _item("an unrelated MTM fact", tier=MemoryTier.MTM, at=base)
    query_vec = (0.5, 0.5)
    ranker = _build_ranker(stm_items=stm_items, mtm_hits_by_query={query_vec: [other_hit]})

    result = await ranker.rank(
        _NS,
        "q",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    floor_ids = {it.memory_id for it in result.items if it.is_floor}
    assert just_said.id in floor_ids, "the just-said fact is no longer protected from eviction"
    settings = RecallSettings()
    assert len(floor_ids) <= settings.floor_protect_limit, (
        "protection is no longer bounded — the floor is swamping the result again"
    )
