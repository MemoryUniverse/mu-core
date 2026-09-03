"""``ThreeChannelRecallRanker`` unit tests — the data-quality assessment #1 fix.

Repro + regression test for ``docs/tracking/DATA-QUALITY-ASSESSMENT.md`` §3.1/#1: a
``recall()`` within one session used to return the WHOLE STM recency floor, unconditionally,
in insertion order, with ZERO room left for the query-relevant MTM channel — every query in a
session with >= ``limit`` STM items produced a byte-identical, query-blind list.

Pure unit test: fake in-memory STM/MTM/LTM tier repos (no containers, no network) so the ranker's
fuse/protect logic is exercised in isolation, deterministically, in milliseconds.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.model.recall import Vector
from mu_engine.platform.clock import FrozenClock
from mu_engine.providers._contracts import RerankHit
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

    async def recent(
        self,
        ns: Namespace,
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        # AD-128: the port grew the Model-A caller set; this fake stands in for a PRIVATE-η
        # partition, where §7.4 authorizes by to_prefix() and the set is not consulted.
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

    async def traverse_entities(
        self,
        ns: Namespace,
        *,
        query: str,
        max_hops: int,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        return []  # D-4: no entity edges wired for this test — isolates STM/MTM interaction


class _FakeLtm(_EmptyLtm):
    """Returns a FIXED, query-BLIND hit list, ranked as given — a stand-in for the real
    ``graph_recall`` flat seed, which is deliberately query-blind (``subject=None``, whole-
    partition, recency-ordered; ``ranker.py`` module docstring). Unlike ``_FakeMtm`` this does
    NOT vary its answer by query vector, because the real adapter it stands in for doesn't
    either — that query-blindness is exactly the property the weight_ltm regression test below
    depends on."""

    def __init__(self, hits: list[MemoryItem]) -> None:
        self._hits = hits

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        return [
            Scored(item=i, score=1.0 - 0.01 * rank, channel=RecallChannel.LTM_GRAPH, rank=rank)
            for rank, i in enumerate(self._hits[:limit])
        ]


class _FakeEmbedder:
    """D1 test double for ``EmbeddingPort``: returns a caller-supplied vector per exact content
    string (no real model) so STM "embed" scoring can be exercised deterministically."""

    def __init__(self, vectors_by_content: dict[str, list[float]]) -> None:
        self._vectors_by_content = vectors_by_content
        self.model_name = "fake-embedder"
        self.dimension = 2

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectors_by_content.get(t, [0.0, 0.0]) for t in texts]


def _build_ranker(
    *,
    stm_items: list[MemoryItem],
    mtm_hits_by_query: dict[tuple[float, ...], list[MemoryItem]],
    settings: RecallSettings | None = None,
    embedder: _FakeEmbedder | None = None,
    ltm_hits: list[MemoryItem] | None = None,
) -> ThreeChannelRecallRanker:
    # D1 (§3.1 follow-up): these pre-existing floor/dedup tests target the §3.1/#1 fuse-swamping
    # fix and the D4 cross-tier-dedup fix, NOT the D1 relevance scorer — pin `stm_scoring="recency"`
    # (the pre-D1 behavior) so they keep testing those mechanisms in isolation. D1's OWN behavior
    # (embed/lexical scoring, floor reorder-by-relevance) gets its own tests below.
    return ThreeChannelRecallRanker(
        stm=_FakeStm(stm_items),
        mtm=_FakeMtm(mtm_hits_by_query),
        ltm=_EmptyLtm() if ltm_hits is None else _FakeLtm(ltm_hits),  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=settings or RecallSettings(stm_scoring="recency"),
        clock=FrozenClock(datetime(2026, 7, 31, tzinfo=UTC)),
        embedder=embedder,
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
async def test_fused_score_carries_the_real_rrf_value_not_the_channel_native_score() -> None:
    """D2 (STATE-AND-DEFECTS-0829.md): ``RecallItemView.fused_score`` used to be silently
    overwritten with the CHANNEL-NATIVE score (``Scored.score``) instead of the value
    ``reciprocal_rank_fusion`` actually computed for that candidate (``ranker.py``'s
    ``for scored, _score in fused_pairs`` discarded ``_score``). With STM/LTM empty, an MTM hit
    at rank 0 has channel-native score ``1.0`` (``_FakeMtm``'s scoring), but its REAL RRF score —
    its channel weighted a third of three, at rank 0, with the default ``k=60`` — is
    ``(1/3) * (1/61)`` ~= ``0.00546``: a completely different, much smaller number. If
    ``fused_score`` still equalled ``1.0`` the discard bug would be back.

    ``weight_stm=1.0``/``weight_ltm=1.0`` are both pinned explicitly (AD-204 lowered the STM
    CLASS default to ``0.1``, and the weight_ltm fix below it lowered the LTM one the same way —
    two separate fixes, for two separate defects, each argued and measured on real IR/answer-
    quality data, not on this unit test's arithmetic) so this test keeps proving D2's plumbing
    claim in isolation, at the exact equal-thirds weighting its own docstring states, regardless
    of where either default sits."""
    target = _item(
        "Ada's flight to Denver is on Thursday",
        tier=MemoryTier.MTM,
        at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=[],
        mtm_hits_by_query={query_vec: [target]},
        settings=RecallSettings(stm_scoring="recency", weight_stm=1.0, weight_ltm=1.0),
    )

    result = await ranker.rank(
        _NS,
        "irrelevant-this-phase",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    hit = next(it for it in result.items if it.memory_id == target.id)
    expected_rrf = (1.0 / 3.0) * (1.0 / (60 + 0 + 1))
    assert hit.fused_score == pytest.approx(expected_rrf)
    assert hit.fused_score != pytest.approx(
        1.0
    ), "fused_score equals the raw channel-native score again — the D2 discard bug is back"


@pytest.mark.asyncio
async def test_a_protected_floor_row_also_carries_the_rrf_value_not_its_stm_native_score() -> None:
    """D2's other half: the test above runs with ``stm_items=[]``, so it never sees the ONE path
    that re-stamps ``fused_score`` after fusion — ``_merge_floor``'s protected-floor restore.

    That restore used to swap the whole ``floor_views`` twin back in, and the twin is built by
    ``_to_view(s, "stm")`` with NO ``fused_score`` override — i.e. it carried the STM-native
    relevance score, which is exactly the number D2 exists to keep out of this field. It also put
    two incomparable scales (~1e-1 STM against ~1e-2 RRF) into one returned list. The restore now
    re-stamps the FLAG only.

    With one STM item and nothing else, the STM channel is weighted one of three and the row is at
    rank 0, so its RRF value is ``(1/3) * (1/61)``. ``_FakeStm`` scores the floor on the
    ``"recency"`` scorer, whose score is ``1.0`` for the most recent row — so if the twin were
    swapped back in, ``fused_score`` would read ``1.0`` here.

    ``weight_stm=1.0``/``weight_ltm=1.0`` pinned for the same reason as the sibling test above:
    AD-204 and the weight_ltm fix each changed a CLASS default weighting, not this D2 plumbing
    claim's equal-thirds arithmetic.
    """
    just_said = _item(
        "Ada just said the deploy passphrase is violet-anchor-77",
        tier=MemoryTier.STM,
        at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    ranker = _build_ranker(
        stm_items=[just_said],
        mtm_hits_by_query={},
        settings=RecallSettings(stm_scoring="recency", weight_stm=1.0, weight_ltm=1.0),
    )

    result = await ranker.rank(
        _NS,
        "irrelevant-this-phase",
        [0.9, 0.1],
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    hit = next(it for it in result.items if it.memory_id == just_said.id)
    assert hit.is_floor, "the protected-floor membership flag was not restored"
    assert hit.fused_score == pytest.approx((1.0 / 3.0) * (1.0 / (60 + 0 + 1)))
    assert hit.fused_score != pytest.approx(1.0), (
        "a protected floor row carries its STM-native score again — the whole floor_views twin "
        "was swapped back in and D2 is only two-thirds fixed"
    )


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
async def test_floor_no_longer_forces_a_query_blind_item_ahead_of_a_relevant_hit() -> None:
    """D3 (STATE-AND-DEFECTS-0829.md): a protected (most-recently-written) STM item used to lead
    the result UNCONDITIONALLY regardless of relevance (``_merge_floor``'s pre-fix
    ``[*floor_views, *tail]``) — measured on LoCoMo (1,531 labelled queries,
    RETRIEVAL-EVAL-0829.md §5) as ``floor_items = 1531 x 3`` exactly: the top THREE slots of
    EVERY single result, chosen without reference to the query. With the floor's own fused rank
    now deciding position (a protected member is rescued at the TAIL only if it does not earn a
    competitive rank on its own), a clearly more relevant MTM hit must rank AHEAD of the
    just-said, irrelevant chatter that used to automatically occupy rank 0 — while the SAME
    protected set stays guaranteed present (membership, not position, is what "never evicted"
    means now).

    ``weight_mtm=2.0`` is explicit here rather than relying on the class default (AD-204 later
    made that default equal this override — see the companion test right below, which proves the
    SAME claim with zero explicit weight, i.e. proves the shipped default actually clears this
    bar): pinning it keeps THIS test's claim scoped to D3's "position, not just presence" fix,
    independent of wherever the default sits."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    stm_items = [
        _item(f"session chatter #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n))
        for n in range(10)
    ]
    target = _item("Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)
    # weight_mtm > weight_stm breaks the rank-0-vs-rank-0 RRF tie (both channels' best candidate
    # lands at rank 0) cleanly in the genuinely-relevant channel's favor — a principled knob
    # already exposed on RecallSettings, not a thumb on the scale specific to this test.
    settings = RecallSettings(stm_scoring="recency", weight_mtm=2.0)
    ranker = _build_ranker(
        stm_items=stm_items, mtm_hits_by_query={query_vec: [target]}, settings=settings
    )

    result = await ranker.rank(
        _NS,
        "irrelevant-this-phase",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    ids = [it.memory_id for it in result.items]
    protected_ids = {it.id for it in stm_items[-3:]}  # floor_protect_limit default = 3, recency
    assert ids[0] == target.id, (
        "a protected-but-irrelevant STM item still leads — the pre-D3 unconditional-prepend "
        "shape is back"
    )
    assert protected_ids <= set(ids), "a protected member was evicted, not merely reordered"


@pytest.mark.asyncio
async def test_default_settings_also_rank_the_relevant_hit_ahead_of_recency_noise() -> None:
    """AD-204 (RETRIEVAL-EVAL-0829.md §5.3 / STATE-AND-DEFECTS-0829.md D3): equal (1.0/1.0/1.0)
    in-arm weights gave the STM channel's best-of-ten recency candidate the SAME RRF rank
    authority as the MTM channel's genuine best-of-the-corpus candidate — measured on LoCoMo as a
    shipped 3-channel fuse WORSE than its own MTM channel alone at every cutoff (recall@1 0.0049
    vs 0.1625). This is the sibling test above's ``weight_mtm=2.0`` override removed entirely —
    bare ``RecallSettings(stm_scoring="recency")``, the actual shipped default (AD-204 landed as
    ``weight_stm=0.1``, a 10:1 discount, not a raised ``weight_mtm``) — to prove the fix is live in
    the field default an operator gets with no config at all, not only demonstrable with a
    hand-picked override. Mutation check: reverting ``RecallSettings.weight_stm``'s default to
    ``1.0`` must fail this test (the rank-0-vs-rank-0 RRF tie would then go the other way, exactly
    as it did before AD-204)."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    stm_items = [
        _item(f"session chatter #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n))
        for n in range(10)
    ]
    target = _item("Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={query_vec: [target]},
        settings=RecallSettings(stm_scoring="recency"),  # bare default — no weight override
    )

    result = await ranker.rank(
        _NS,
        "irrelevant-this-phase",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    ids = [it.memory_id for it in result.items]
    protected_ids = {it.id for it in stm_items[-3:]}
    assert ids[0] == target.id, (
        "the SHIPPED DEFAULT still lets a protected-but-irrelevant STM item lead — AD-204's "
        "weight_mtm default was reverted or never actually reached RecallSettings()"
    )
    assert protected_ids <= set(ids), "a protected member was evicted, not merely reordered"


@pytest.mark.asyncio
async def test_default_settings_rank_the_relevant_mtm_hit_ahead_of_query_blind_ltm_noise() -> None:
    """weight_ltm fix (accuracy lane, 2026-08-31, RETRIEVAL-EVAL-0829.md §11) — the LTM sibling of
    the ``weight_stm`` test above. AD-204 left ``weight_ltm`` at 1.0 because the LTM channel had
    NEVER contributed a single item to any measured result (no eval command called
    ``LocalMemory.consolidate()``, so the graph tier was always empty). Once populated, the SAME
    rank-authority mismatch AD-204 fixed for STM reproduces for LTM: `graph_recall`'s flat seed is
    query-BLIND (whole-partition, recency-ordered, `subject=None` — module docstring), so it can
    rank a query-irrelevant candidate at RRF rank 0 with FULL weight — measured on LoCoMo as a
    full-corpus answer-quality COLLAPSE (37.2% -> 11.9%, gpt-5 judge, every category worse) the
    moment the graph tier is actually populated at the shipped `weight_ltm=1.0`.

    This reproduces the shape in miniature: an MTM channel with the genuinely relevant fact NOT at
    its own rank 0 (a realistic case — MTM's own dense search doesn't always put the right answer
    first) against an LTM channel returning ONE query-blind noise candidate at rank 0. At
    `weight_ltm=1.0` (pre-fix), `1.0/(60+0+1) = 0.01639` for the noise item beats
    `1.0/(60+3+1) = 0.015625` for the relevant-but-rank-3 MTM item — the noise item wins the top
    slot. Mutation check: reverting `RecallSettings.weight_ltm`'s default to `1.0` must fail this
    test (the noise item would then rank first, exactly as it did in the full-corpus collapse)."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    query_vec = (0.9, 0.1)
    decoys = [_item(f"unrelated MTM decoy #{n}", tier=MemoryTier.MTM, at=base) for n in range(3)]
    target = _item("Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base)
    noise = _item("Let me know if you need anything", tier=MemoryTier.LTM, at=base)
    ranker = _build_ranker(
        stm_items=[],
        mtm_hits_by_query={query_vec: [*decoys, target]},  # target arrives at MTM's OWN rank 3
        ltm_hits=[noise],  # query-blind: returned regardless of query_vec, at LTM's rank 0
        settings=RecallSettings(stm_scoring="recency"),  # bare default — no weight override
    )

    result = await ranker.rank(
        _NS,
        "irrelevant-this-phase",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    ids = [it.memory_id for it in result.items]
    assert ids.index(target.id) < ids.index(noise.id), (
        "a query-blind LTM candidate at LTM's own rank 0 still outranks a genuinely relevant MTM "
        "hit at MTM's own rank 3 — the SHIPPED DEFAULT weight_ltm is back to giving the "
        "query-blind flat graph seed full rank authority, the exact regression measured as a "
        "full-corpus answer-quality collapse (37.2% -> 11.9%)"
    )


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
    assert (
        len(floor_ids) <= settings.floor_protect_limit
    ), "protection is no longer bounded — the floor is swamping the result again"


# ---------------------------------------------------------------------------------------------
# D1 — STM relevance scoring (DATA-QUALITY-ASSESSMENT.md §3.1, floor-fix follow-up to 02fbed9).
# The tests above prove the STM channel no longer SWAMPS the fused result by count/insertion
# order; these prove it now carries a REAL per-candidate relevance signal of its own, and that
# the unconditionally-protected floor block is reorderable by that signal.
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_scoring_ranks_the_relevant_stm_item_above_recency() -> None:
    """ "embed" mode (the default): an STM item said LONG AGO but semantically close to the query
    must outrank STM items said MORE RECENTLY but irrelevant — pure recency rank could never do
    this; only a real per-candidate relevance score can."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    relevant = _item(
        "Ada's flight to Denver is on Thursday", tier=MemoryTier.STM, at=base
    )  # oldest -> recency rank LAST
    filler = [
        _item(f"session chatter #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n + 1))
        for n in range(9)
    ]
    stm_items = [relevant, *filler]  # relevant is oldest -> recency-last, embed-first
    query_vec = [0.9, 0.1]
    embedder = _FakeEmbedder(
        {
            relevant.content: [0.9, 0.1],  # near-identical to the query
            **{f.content: [0.0, 1.0] for f in filler},  # orthogonal -> irrelevant
        }
    )
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={},
        settings=RecallSettings(stm_scoring="embed"),
        embedder=embedder,
    )

    ids = await _rank(ranker, query_vec)

    assert ids.index(relevant.id) < ids.index(
        filler[0].id
    ), "embed-scored STM relevance did not outrank a merely-more-recent irrelevant item"


@pytest.mark.asyncio
async def test_embed_scoring_without_an_injected_embedder_fails_loud() -> None:
    """ "embed" is the DEFAULT ``stm_scoring`` — a ranker built with no embedder must raise, never
    silently fall back to recency ordering (DEV-STANDARDS: no silent stubs)."""
    from mu_engine.services.recall.ranker import StmScoringConfigError

    base = datetime(2026, 7, 31, tzinfo=UTC)
    stm_items = [_item("session chatter", tier=MemoryTier.STM, at=base)]
    ranker = _build_ranker(
        stm_items=stm_items, mtm_hits_by_query={}, settings=RecallSettings(), embedder=None
    )

    with pytest.raises(StmScoringConfigError):
        await _rank(ranker, [0.1, 0.1])


@pytest.mark.asyncio
async def test_lexical_scoring_ranks_token_overlap_above_recency_with_no_embedder() -> None:
    """ "lexical" mode: the minimum-viable fallback needs no embedder at all — token overlap
    against the raw query text is enough to outrank a merely-more-recent irrelevant item."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    relevant = _item("denver flight thursday", tier=MemoryTier.STM, at=base)
    filler = [
        _item(f"chatter number {n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n + 1))
        for n in range(9)
    ]
    stm_items = [relevant, *filler]
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={},
        settings=RecallSettings(stm_scoring="lexical"),
    )

    # `_rank` hardcodes a fixed query text; lexical scoring needs the REAL query text, so call
    # `rank()` directly here instead.
    result = await ranker.rank(
        _NS,
        "denver flight thursday",
        [0.0, 0.0],
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )
    ids = [it.memory_id for it in result.items]

    assert ids.index(relevant.id) < ids.index(
        filler[0].id
    ), "lexical overlap did not outrank a merely-more-recent, unrelated item"


@pytest.mark.asyncio
async def test_protected_floor_is_reordered_by_relevance_not_recency() -> None:
    """D1 (b): the PROTECTED floor block (never evicted) must still contain the same
    recency-selected members, but a just-said IRRELEVANT fact must no longer occupy rank 1
    ahead of an older, but query-relevant, protected member."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    # Two items inside the protected window (floor_protect_limit=3 default): an older-but-relevant
    # fact and a just-said-but-irrelevant one. A third filler keeps the window non-trivial.
    relevant = _item(
        "Ada's flight to Denver is on Thursday", tier=MemoryTier.STM, at=base + timedelta(minutes=8)
    )
    just_said_irrelevant = _item(
        "ok cool", tier=MemoryTier.STM, at=base + timedelta(minutes=9)
    )  # most recent -> would be recency-rank-0 pre-D1
    older_filler = [
        _item(f"chatter #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n))
        for n in range(8)
    ]
    stm_items = [*older_filler, relevant, just_said_irrelevant]
    query_vec = [0.9, 0.1]
    embedder = _FakeEmbedder(
        {
            relevant.content: [0.9, 0.1],
            just_said_irrelevant.content: [0.0, 1.0],
            **{f.content: [0.0, 1.0] for f in older_filler},
        }
    )
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={},
        settings=RecallSettings(stm_scoring="embed"),
        embedder=embedder,
    )

    result = await ranker.rank(
        _NS,
        "irrelevant",
        query_vec,
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    floor_views = [it for it in result.items if it.is_floor]
    floor_ids = {v.memory_id for v in floor_views}
    assert just_said_irrelevant.id in floor_ids, "recency-selected protection membership changed"
    assert relevant.id in floor_ids, "recency-selected protection membership changed"
    assert floor_views[0].memory_id == relevant.id, (
        "the just-said-but-irrelevant fact still leads the protected block over the "
        "older-but-relevant one — floor is not reorderable by relevance"
    )


class _RecordingLtm(_EmptyLtm):
    """Records the ``caller_identity_set`` each LTM arm is actually CALLED with.

    C2 GAP: the adapter-level regression tests
    (``tests/storage/test_falkor_traverse_authz_int.py``) call ``traverse_entities`` DIRECTLY, so
    nothing covered the CALL SITE — deleting ``caller_identity_set=caller`` from
    ``ThreeChannelRecallRanker._ltm_channel`` reopened the exact bypass with a fully green suite
    (verified by mutation). The wall is only as good as the thread that reaches it.
    """

    def __init__(self) -> None:
        self.graph_recall_caller: object = "<never called>"
        self.traverse_caller: object = "<never called>"

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        self.graph_recall_caller = caller_identity_set
        return []

    async def traverse_entities(
        self,
        ns: Namespace,
        *,
        query: str,
        max_hops: int,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        self.traverse_caller = caller_identity_set
        return []


@pytest.mark.asyncio
async def test_ranker_threads_the_caller_identity_set_to_the_traversal_arm() -> None:
    """C2 CALL-SITE REGRESSION: the multi-hop arm must receive the SAME caller identity set the
    flat ``graph_recall`` seed receives. ``traverse_entities`` DERIVES memory ids from a
    workspace-wide entity graph; with no caller set the adapter's SHARED hydration has nothing to
    match ``m.authorized_ids`` against and the ACL clause is skipped entirely."""
    ltm = _RecordingLtm()
    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([]),  # type: ignore[arg-type]
        mtm=_FakeMtm({}),  # type: ignore[arg-type]
        ltm=ltm,  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(stm_scoring="recency"),
        clock=FrozenClock(datetime(2026, 7, 31, tzinfo=UTC)),
    )
    caller = frozenset({"principal-alice"})
    assert RecallSettings().ltm_max_hops > 0, "precondition: the traversal arm is ON by default"

    await ranker.rank(
        _NS,
        "who is Bo's manager?",
        [0.0, 0.0],
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=caller,
    )

    assert ltm.traverse_caller == caller, (
        "C2 AUTHZ BYPASS at the CALL SITE: the ranker did not thread caller_identity_set into "
        "traverse_entities — the adapter's SHARED ACL clause is then skipped entirely"
    )
    assert ltm.graph_recall_caller == caller, "the flat seed's caller threading regressed"


class _ContentScoredReranker:
    """A ``RerankProviderPort`` test double keyed by document CONTENT (not position) — the
    ranker's fused pool order depends on RRF, so a content-keyed fake is robust to exactly which
    rank each candidate lands at, unlike an index-keyed fake."""

    def __init__(self, scores_by_content: dict[str, float]) -> None:
        self._scores = scores_by_content

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[RerankHit]:
        return [
            RerankHit(index=i, score=self._scores.get(doc, 0.0)) for i, doc in enumerate(documents)
        ]


@pytest.mark.asyncio
async def test_reranker_prunes_a_low_relevance_mtm_candidate_end_to_end() -> None:
    """ACCURACY-PLAN-0831.md item 6, wired end-to-end: an MTM distractor that ranked ahead of the
    genuinely relevant hit on raw RRF position is pruned once the reranker scores it low, and the
    surviving item carries its real ``rerank_score`` (previously always ``None`` — the gate had no
    caller)."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    relevant = _item("Ada's flight to Denver is Thursday", tier=MemoryTier.MTM, at=base)
    distractor = _item("chatter about the weather", tier=MemoryTier.MTM, at=base)
    query_vec = [0.5, 0.5]
    reranker = _ContentScoredReranker({relevant.content: 0.9, distractor.content: 0.1})
    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([]),  # type: ignore[arg-type]
        mtm=_FakeMtm({tuple(query_vec): [distractor, relevant]}),  # type: ignore[arg-type]
        ltm=_EmptyLtm(),  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(
            stm_scoring="recency",
            rerank_enabled=True,
            rerank_min_score=0.5,
            rerank_top_fraction=0.5,
        ),
        clock=FrozenClock(datetime(2026, 7, 31, tzinfo=UTC)),
        reranker=reranker,
    )

    result = await ranker.rank(
        _NS,
        "when is Ada's flight",
        query_vec,
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    ids = [it.memory_id for it in result.items]
    assert distractor.id not in ids, "below the adaptive cutoff (0.1 < 0.5) — must be pruned"
    assert relevant.id in ids
    relevant_view = next(it for it in result.items if it.memory_id == relevant.id)
    assert relevant_view.rerank_score == 0.9


@pytest.mark.asyncio
async def test_reranker_pruning_a_protected_floor_member_does_not_evict_it() -> None:
    """The "never evict a just-said fact" guarantee (AD-195) survives the rerank gate: a
    floor-protected STM item the reranker scores BELOW the cutoff must still appear in the final
    result — rescued by ``_merge_floor``'s pre-existing "protected id missing from `fused`"
    fallback, which needed no change for this (see ``rerank_gate.py``'s own docstring)."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    floor_item = _item("ok cool", tier=MemoryTier.STM, at=base)  # just-said, irrelevant
    query_vec = [0.5, 0.5]
    reranker = _ContentScoredReranker({floor_item.content: 0.05})  # well below min_score

    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([floor_item]),  # type: ignore[arg-type]
        mtm=_FakeMtm({}),  # type: ignore[arg-type]
        ltm=_EmptyLtm(),  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(
            stm_scoring="recency",
            floor_protect_limit=1,
            rerank_enabled=True,
            rerank_min_score=0.5,
            rerank_top_fraction=0.5,
        ),
        clock=FrozenClock(datetime(2026, 7, 31, tzinfo=UTC)),
        reranker=reranker,
    )

    result = await ranker.rank(
        _NS,
        "irrelevant query",
        query_vec,
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    ids = [it.memory_id for it in result.items]
    assert floor_item.id in ids, "protected floor member was evicted by the rerank gate"
    view = next(it for it in result.items if it.memory_id == floor_item.id)
    assert view.is_floor is True


@pytest.mark.asyncio
async def test_reranker_dark_by_default_is_byte_identical_to_no_rerank() -> None:
    """No ``reranker`` injected -> ``rerank_enabled``'s default value is irrelevant, the gate is
    dark, and every item's ``rerank_score`` stays ``None`` — the pre-existing shipped default
    (``_build_ranker`` never passes ``reranker=``, so this is also an implicit regression guard
    for every other test in this file)."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    only = _item("anything", tier=MemoryTier.MTM, at=base)
    query_vec = [0.1, 0.2]
    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([]),  # type: ignore[arg-type]
        mtm=_FakeMtm({tuple(query_vec): [only]}),  # type: ignore[arg-type]
        ltm=_EmptyLtm(),  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(stm_scoring="recency"),
        clock=FrozenClock(datetime(2026, 7, 31, tzinfo=UTC)),
    )

    result = await ranker.rank(
        _NS,
        "q",
        query_vec,
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    assert [it.memory_id for it in result.items] == [only.id]
    assert result.items[0].rerank_score is None
