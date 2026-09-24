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
from mu_engine.services.recall.dto import RecallChannels, RecallItemView, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion, reciprocal_rank_fusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker, _narrow_after_expansion
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored

_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)


def _item(
    content: str, *, tier: MemoryTier, at: datetime, turn_seq: int | None = None
) -> MemoryItem:
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
        turn_seq=turn_seq,
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

    async def demoted(
        self,
        ns: Namespace,
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[MemoryItem]]:
        """AD-250 fix (ADR 0061): this fake models an ordinary (never-demoted) session, so the
        demoted channel is always empty — `ThreeChannelRecallRanker.rank` calls this
        unconditionally now, so the fake needed it to keep working at all."""
        return []

    async def reinforce(self, ns: Namespace, memory_id: str, *, at: datetime) -> MemoryItem | None:
        """AD-250 fix (ADR 0061): a real, functioning fake of the read-stat write-back — bumps
        `access_count` in place, exactly like the shipped adapters, so a test asserting on it
        (rather than merely on call-count) can. `ThreeChannelRecallRanker.rank` calls this
        unconditionally for every STM-channel hit in its final result (`RecallSettings.
        reinforce_on_recall` defaults True), so this fake needed it to keep working at all."""
        for i, item in enumerate(self._items):
            if item.id == memory_id:
                reinforced = item.model_copy(
                    update={"access_count": item.access_count + 1, "updated_at": at}
                )
                self._items[i] = reinforced
                return reinforced
        return None


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
        seed_entity_uids: object = None,
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


# =================================================================================================
# ADR 0060 — `ltm_protect_limit`: the LTM channel was not merely discounted, it was
# STRUCTURALLY EXCLUDED. See `dto.py`'s `ltm_protect_limit` docstring and
# `docs/tracking/eval-runs/2026-09-24-ltm-channel-zero-slots.md` for the full arithmetic proof.
# =================================================================================================
def test_shipped_default_ltm_protect_limit_is_zero() -> None:
    """Literal regression guard — ADR 0060's own measurement (`dto.py`'s field docstring) is the
    reason this is 0, not 1: at 1, 383/383 real LoCoMo queries got their guaranteed graph slot
    and gold_in_context DROPPED (280->273/383). The mechanism is shipped; it is not turned on."""
    assert RecallSettings().ltm_protect_limit == 0


@pytest.mark.asyncio
async def test_default_settings_still_exclude_ltm_at_protect_limit_zero() -> None:
    """MUTATION CHECK (baseline half): the SHIPPED bare default (`ltm_protect_limit=0`)
    reproduces the PRE-FIX structural exclusion exactly — an LTM candidate that cannot win the
    RRF fuse on its own (buried under a full MTM pool) never appears in the result at all, not
    merely low-ranked."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    query_vec = (0.9, 0.1)
    decoys = [_item(f"unrelated MTM decoy #{n}", tier=MemoryTier.MTM, at=base) for n in range(20)]
    graph_fact = _item("Ada's team meets on Mondays", tier=MemoryTier.LTM, at=base)
    ranker = _build_ranker(
        stm_items=[],
        mtm_hits_by_query={query_vec: decoys},
        ltm_hits=[graph_fact],
        settings=RecallSettings(stm_scoring="recency", ltm_protect_limit=0),
    )

    result = await ranker.rank(
        _NS,
        "q",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    ids = [it.memory_id for it in result.items]
    assert graph_fact.id not in ids, (
        "ltm_protect_limit=0 must reproduce the structural exclusion this ADR fixes — a real "
        "graph fact, present and query-relevant-by-recency, still winning zero slots"
    )


@pytest.mark.asyncio
async def test_ltm_protect_limit_rescues_a_graph_fact_the_fuse_would_otherwise_exclude() -> None:
    """MUTATION CHECK (fix half): the SAME setup as the test above, but with the MECHANISM turned
    on (`ltm_protect_limit=1` — not the shipped default; see ADR 0060 / `dto.py`'s field
    docstring for why the default stays 0) — the graph fact must now appear. Reverting the merge
    in `ranker.py` turns this red while the test above stays green, isolating exactly what the
    mechanism does when an operator enables it."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    query_vec = (0.9, 0.1)
    decoys = [_item(f"unrelated MTM decoy #{n}", tier=MemoryTier.MTM, at=base) for n in range(20)]
    graph_fact = _item("Ada's team meets on Mondays", tier=MemoryTier.LTM, at=base)
    ranker = _build_ranker(
        stm_items=[],
        mtm_hits_by_query={query_vec: decoys},
        ltm_hits=[graph_fact],
        settings=RecallSettings(stm_scoring="recency", ltm_protect_limit=1),
    )

    result = await ranker.rank(
        _NS,
        "q",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    ids = [it.memory_id for it in result.items]
    assert graph_fact.id in ids, (
        "ltm_protect_limit=1 must rescue at least one graph-tier candidate instead of pinning "
        "the channel at exactly zero — an operator who turns this on must get what it promises"
    )


@pytest.mark.asyncio
async def test_ltm_protect_limit_never_outranks_a_relevant_mtm_hit() -> None:
    """The rescue is PRESENCE, not PRIORITY: reproduces
    `test_default_settings_rank_the_relevant_mtm_hit_ahead_of_query_blind_ltm_noise`'s exact
    scenario but with `ltm_protect_limit=1` (the shipped default, which that test leaves
    implicit) made explicit — the query-blind noise item is now GUARANTEED a slot, and it must
    STILL rank behind the genuinely relevant MTM hit, never ahead of it."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    query_vec = (0.9, 0.1)
    decoys = [_item(f"unrelated MTM decoy #{n}", tier=MemoryTier.MTM, at=base) for n in range(3)]
    target = _item("Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base)
    noise = _item("Let me know if you need anything", tier=MemoryTier.LTM, at=base)
    ranker = _build_ranker(
        stm_items=[],
        mtm_hits_by_query={query_vec: [*decoys, target]},
        ltm_hits=[noise],
        settings=RecallSettings(stm_scoring="recency", ltm_protect_limit=1),
    )

    result = await ranker.rank(
        _NS,
        "q",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    ids = [it.memory_id for it in result.items]
    assert noise.id in ids, "the rescue should have guaranteed the noise item a slot this time"
    assert ids.index(target.id) < ids.index(noise.id), (
        "a rescued LTM candidate must never rank AHEAD of a genuinely relevant MTM hit — presence "
        "is guaranteed, priority is not"
    )


@pytest.mark.asyncio
async def test_ltm_protect_limit_is_bounded_not_the_whole_pool() -> None:
    """A graph tier with far more qualifying candidates than `ltm_protect_limit` must still only
    ever spend AT MOST `ltm_protect_limit` of the `limit` result slots on rescued LTM items —
    bounded regardless of corpus size, the whole point of a floor rather than a raised weight."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    query_vec = (0.9, 0.1)
    decoys = [_item(f"unrelated MTM decoy #{n}", tier=MemoryTier.MTM, at=base) for n in range(20)]
    graph_facts = [_item(f"graph fact #{n}", tier=MemoryTier.LTM, at=base) for n in range(5)]
    ranker = _build_ranker(
        stm_items=[],
        mtm_hits_by_query={query_vec: decoys},
        ltm_hits=graph_facts,
        settings=RecallSettings(stm_scoring="recency", ltm_protect_limit=2),
    )

    result = await ranker.rank(
        _NS,
        "q",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    ltm_ids_in_result = {f.id for f in graph_facts} & {it.memory_id for it in result.items}
    assert (
        len(ltm_ids_in_result) == 2
    ), f"expected exactly ltm_protect_limit=2 rescued LTM items, got {len(ltm_ids_in_result)}"


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
# T1 option (c) — the STM floor guarantee made conditional on relevance (TRACE-0923.md §7/§7.1,
# `ranker.py::_protected_floor_ids`). The SHIPPED default (`floor_protect_min_relevance=0.5`,
# set on evidence — `dto.py`'s own docstring has the sweep) gates a genuinely irrelevant
# just-said fact; `-1.0` (cosine similarity's true minimum) is the value that reproduces the
# ORIGINAL, fully unconditional AD-195 guarantee the tests above pin.
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t1c_shipped_default_gates_a_genuinely_irrelevant_just_said_fact() -> None:
    """``floor_protect_min_relevance`` SHIPS at ``0.5`` (evidence-based — `dto.py`'s own sweep
    table): a just-said fact with zero lexical overlap against the query must NOT be
    unconditionally forced into the window at the shipped default — the whole point of T1c."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    stm_items = [
        _item(f"session chatter #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n))
        for n in range(10)
    ]
    just_said = stm_items[-1]
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={},
        settings=RecallSettings(stm_scoring="lexical"),  # class default floor_protect_min_relevance
    )

    result = await ranker.rank(
        _NS,
        "irrelevant this phase",
        [0.1, 0.1],
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    floor_ids = {it.memory_id for it in result.items if it.is_floor}
    assert just_said.id not in floor_ids, (
        "the shipped default (0.5) did not gate a just-said fact with ZERO query overlap from "
        "PROTECTED (is_floor) status — T1c's whole point (stop paying for an unconditional, "
        "query-blind protection guarantee) is not reaching this default"
    )


@pytest.mark.asyncio
async def test_t1c_legacy_value_reproduces_the_original_unconditional_guarantee() -> None:
    """``floor_protect_min_relevance=-1.0`` (cosine similarity's true minimum) is the documented
    escape hatch back to AD-195's ORIGINAL, fully unconditional guarantee — the exact behaviour
    the shipped 0.5 default now narrows."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    stm_items = [
        _item(f"session chatter #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n))
        for n in range(10)
    ]
    just_said = stm_items[-1]
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={},
        settings=RecallSettings(stm_scoring="lexical", floor_protect_min_relevance=-1.0),
    )

    ids = await _rank(ranker, [0.1, 0.1], limit=10)

    assert just_said.id in ids, "floor_protect_min_relevance=-1.0 must reproduce AD-195 verbatim"


@pytest.mark.asyncio
async def test_t1c_rollback_value_protects_a_genuinely_negative_embed_score() -> None:
    """Regression guard for the bug an earlier ``0.0`` default shipped with (caught live on
    ``mu-dev-vm`` by ``test_persona_composition_int.py::
    test_a_real_persona_reorders_a_real_recall_and_changes_nothing_else`` against real stores +
    the real MiniLM embedder — see ``RecallSettings.floor_protect_min_relevance``'s own docstring
    for the full account). ``stm_scoring="embed"``'s real range is cosine similarity, ``[-1.0,
    1.0]``, NOT ``[0.0, 1.0]`` — a floor candidate whose embedding points AWAY from the query
    scores genuinely negative, so ONLY ``-1.0`` (cosine's true minimum) is a genuinely inert
    rollback value. Pins that: at ``-1.0`` the negative-scoring candidate is still PROTECTED
    (``is_floor=True``), which a ``0.0`` "looks inert" value would silently break.

    VERIFY PASS 2026-09-23: this test previously asserted ``just_said.id in ids`` at the SHIPPED
    default and claimed "the shipped default must still protect it". Both halves were wrong —
    at the shipped ``0.5`` bar a ``-1.0`` cosine is DELIBERATELY not protected (that is T1c), and
    presence in ``ids`` is not protection when nothing else competes for the slot: the assertion
    still passed with ``_protected_floor_ids`` mutated to return an empty set, i.e. it could not
    fail for the reason it named."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    just_said = _item("completely unrelated aside", tier=MemoryTier.STM, at=base)
    query_vec = [1.0, 0.0]
    embedder = _FakeEmbedder({just_said.content: [-1.0, 0.0]})  # cosine == -1.0, the true minimum
    ranker = _build_ranker(
        stm_items=[just_said],
        mtm_hits_by_query={},
        settings=RecallSettings(stm_scoring="embed", floor_protect_min_relevance=-1.0),
        embedder=embedder,
    )

    result = await ranker.rank(
        _NS,
        "q",
        query_vec,
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    assert just_said.id in {it.memory_id for it in result.items if it.is_floor}, (
        "a floor candidate with a genuinely negative cosine score was not protected at the "
        "documented rollback value -1.0 — the exact class of bug an earlier 0.0 default "
        "shipped with (a bar that claims to be inert but is not)"
    )


@pytest.mark.asyncio
async def test_t1c_still_protects_a_just_said_fact_that_is_relevant() -> None:
    """ADR 0052's live-agent-session promise, pinned: *"the most recent STM fact is never evicted
    from the answer window PROVIDED it clears ``min_relevance`` against the asked query"*. A
    conditional guarantee that never fires is not a guarantee — so this constructs the case the
    ADR says is the typical one (a just-said fact that IS about the current question,
    ``stm_scoring="embed"``, the SHIPPED scoring mode and the SHIPPED bar) against a strong,
    competing MTM hit, and asserts the just-said fact is still PROTECTED (``is_floor=True``),
    not merely present."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    just_said = _item("the flight to Denver is on Thursday", tier=MemoryTier.STM, at=base)
    stale = [
        _item(f"session chatter #{n}", tier=MemoryTier.STM, at=base - timedelta(minutes=n + 1))
        for n in range(9)
    ]
    competing_mtm = _item("an unrelated but highly ranked MTM fact", tier=MemoryTier.MTM, at=base)
    query_vec = (1.0, 0.0)
    embedder = _FakeEmbedder({just_said.content: [1.0, 0.0]})  # cosine == 1.0, clears 0.5
    ranker = _build_ranker(
        stm_items=[*stale, just_said],
        mtm_hits_by_query={query_vec: [competing_mtm]},
        settings=RecallSettings(stm_scoring="embed"),  # class default bar: the SHIPPED 0.5
        embedder=embedder,
    )

    result = await ranker.rank(
        _NS,
        "when is the flight",
        list(query_vec),
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )

    assert just_said.id in {it.memory_id for it in result.items if it.is_floor}, (
        "a just-said fact that IS relevant to the query lost AD-195's protection at the shipped "
        "floor_protect_min_relevance=0.5 — T1c was supposed to narrow the guarantee to the "
        "irrelevant case, not retire it"
    )


def test_protected_floor_ids_filters_by_relevance_within_the_eligibility_window() -> None:
    """Direct unit test of ``_protected_floor_ids`` (the T1c mechanism itself), independent of
    fusion/RRF arithmetic: ``protect_n`` still bounds ELIGIBILITY (recency-selected, unchanged
    from pre-T1c) but a candidate below ``min_relevance`` is now excluded from the protected id
    set, and a candidate outside the eligibility window is excluded regardless of its score."""
    from mu_engine.services.recall.ranker import _protected_floor_ids

    base = datetime(2026, 7, 31, tzinfo=UTC)
    items = [
        _item(f"item-{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n)) for n in range(5)
    ]
    # `floor` — recency order, newest first (items[4], items[3], items[2], items[1], items[0]).
    floor = [
        Scored(item=it, score=1.0, channel=RecallChannel.STM_FLOOR, rank=rank, is_floor=True)
        for rank, it in enumerate(reversed(items))
    ]
    # `floor_scored` — SAME items, relevance scores assigned deliberately out of recency order:
    # items[4] (most recent) is IRRELEVANT (0.0); items[2] (3rd most recent, still inside the
    # protect_n=3 eligibility window) is RELEVANT (0.8); items[0]/[1] are outside the window.
    relevance = {
        items[4].id: 0.0,
        items[3].id: 0.2,
        items[2].id: 0.8,
        items[1].id: 0.9,
        items[0].id: 1.0,
    }
    floor_scored = [
        Scored(item=it, score=relevance[it.id], channel=RecallChannel.STM_FLOOR, rank=0)
        for it in items
    ]

    protected = _protected_floor_ids(
        floor=floor, floor_scored=floor_scored, protect_n=3, min_relevance=0.5
    )

    assert protected == {items[2].id}, (
        "items[4]/[3] are eligible (top-3 recency) but score below the 0.5 bar — excluded; "
        "items[2] is eligible AND clears the bar — included; items[1]/[0] score high but are "
        "OUTSIDE the protect_n=3 eligibility window — excluded regardless of relevance"
    )

    # A bar at or below every candidate's score reproduces the pre-T1c "every eligible member
    # protected" behaviour exactly. NOTE (verify pass 2026-09-23): 0.0 is NOT the shipped default
    # (0.5 is) and is NOT the documented rollback value either (-1.0 is, because
    # `stm_scoring="embed"` scores by cosine over [-1.0, 1.0]) — this line said "the shipped
    # default" and was wrong on both counts.
    assert _protected_floor_ids(
        floor=floor, floor_scored=floor_scored, protect_n=3, min_relevance=0.0
    ) == {items[4].id, items[3].id, items[2].id}


@pytest.mark.asyncio
async def test_t1c_recency_scoring_makes_the_bar_a_uniform_switch_not_a_filter() -> None:
    """Documented degenerate case (``_protected_floor_ids`` docstring): ``stm_scoring="recency"``
    gives every STM candidate the SAME constant score (1.0), so the bar can only switch
    protection uniformly on (``<= 1.0``) or off (``> 1.0``) — never filter some candidates and
    not others, because there is no per-candidate relevance signal to filter by."""

    async def _floor_ids(min_relevance: float) -> set[str]:
        base = datetime(2026, 7, 31, tzinfo=UTC)
        stm_items = [_item("chatter", tier=MemoryTier.STM, at=base)]
        ranker = _build_ranker(
            stm_items=stm_items,
            mtm_hits_by_query={},
            settings=RecallSettings(
                stm_scoring="recency", floor_protect_min_relevance=min_relevance
            ),
        )
        result = await ranker.rank(
            _NS,
            "q",
            [0.1, 0.1],
            limit=10,
            channels=RecallChannels(),
            caller_identity_set=frozenset[str](),
        )
        return {it.memory_id for it in result.items if it.is_floor}

    assert await _floor_ids(1.0) != set()
    assert await _floor_ids(1.1) == set()


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
        # T1c added a SECOND, INDEPENDENT gate on protection membership (this test's own module
        # section below) — pin `floor_protect_min_relevance=-1.0` (AD-195's original, fully
        # unconditional value) so this D1 test keeps isolating exactly what its own docstring
        # says: membership stays recency-selected, only IN-BLOCK ORDER is relevance-driven.
        settings=RecallSettings(stm_scoring="embed", floor_protect_min_relevance=-1.0),
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
        seed_entity_uids: object = None,
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


class _SeedRecordingLtm(_EmptyLtm):
    """AD-258: records the ``seed_entity_uids`` list ``_ltm_channel`` actually passed into
    ``traverse_entities`` — the call-site regression proof for the content-aware seed, the same
    role ``_RecordingLtm`` plays for ``caller_identity_set`` above."""

    def __init__(self) -> None:
        self.seed_entity_uids_seen: object = "<never called>"

    async def traverse_entities(
        self,
        ns: Namespace,
        *,
        query: str,
        max_hops: int,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        seed_entity_uids: object = None,
    ) -> list[Scored[MemoryItem]]:
        self.seed_entity_uids_seen = seed_entity_uids
        return []


@pytest.mark.asyncio
async def test_ltm_channel_seeds_traversal_from_the_mtm_channels_own_entity_uids() -> None:
    """AD-258 call-site regression: ``_ltm_channel`` must harvest ``entity_uids`` off the SAME
    MTM semantic-search task the MTM channel itself ran — no second embedding call, no query-text
    re-derivation — and pass them to ``traverse_entities`` as ``seed_entity_uids``, ordered by MTM
    rank, deduplicated, and bounded to ``settings.ltm_entity_seed_pool``. Mutation-checked: with
    ``ltm_entity_seed_pool`` reverted to not being read (or ``_resolve_seed_entity_uids`` deleted)
    this fails with ``seed_entity_uids_seen is None``."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    query_vec = [1.0, 0.0]

    def _mtm_item(name: str, entity_uids: list[str] | None) -> MemoryItem:
        item = _item(name, tier=MemoryTier.MTM, at=base)
        if entity_uids is not None:
            item.metadata = {"entity_uids": entity_uids}
        return item

    # Rank order matters: top-2 (pool=2 below) contribute uids, the 3rd (in-pool but beyond the
    # seed pool) must NOT, and the 4th carries no backfilled uids at all (never distilled) —
    # exercising "contributes nothing, never an error" in the same test.
    hits = [
        _mtm_item("Ada manages the Denver team", ["ent_ada", "ent_denver_team"]),
        _mtm_item("Ada's flight is Thursday", ["ent_ada", "ent_thursday"]),  # dup ent_ada
        _mtm_item("outside the seed pool", ["ent_outside_pool"]),
        _mtm_item("never distilled to the graph", None),
    ]
    ltm = _SeedRecordingLtm()
    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([]),  # type: ignore[arg-type]
        mtm=_FakeMtm({tuple(query_vec): hits}),  # type: ignore[arg-type]
        ltm=ltm,  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(stm_scoring="recency", ltm_entity_seed_pool=2),
        clock=FrozenClock(base),
    )

    await ranker.rank(
        _NS,
        "who does Ada manage?",
        query_vec,
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=None,
    )

    assert ltm.seed_entity_uids_seen == ["ent_ada", "ent_denver_team", "ent_thursday"], (
        "expected rank-ordered, deduplicated entity uids from the top-2 MTM hits only "
        f"(pool=2); got {ltm.seed_entity_uids_seen!r}"
    )


@pytest.mark.asyncio
async def test_ltm_entity_seed_pool_zero_reproduces_token_only_seed_exactly() -> None:
    """The shipped default (``ltm_entity_seed_pool=0``): AD-258's seed must be fully inert unless
    explicitly enabled — ``None`` reaches ``traverse_entities`` (token-match-only, byte-identical
    to pre-AD-258 behavior), never an empty list (a caller-observable difference some adapters
    could treat differently) and never anything derived from the MTM hits below despite them
    carrying real ``entity_uids``."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    query_vec = [1.0, 0.0]
    hits = [_item("Ada manages the Denver team", tier=MemoryTier.MTM, at=base)]
    hits[0].metadata = {"entity_uids": ["ent_ada", "ent_denver_team"]}
    ltm = _SeedRecordingLtm()
    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([]),  # type: ignore[arg-type]
        mtm=_FakeMtm({tuple(query_vec): hits}),  # type: ignore[arg-type]
        ltm=ltm,  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(stm_scoring="recency"),  # ltm_entity_seed_pool default (0)
        clock=FrozenClock(base),
    )
    assert RecallSettings().ltm_entity_seed_pool == 0, "precondition: shipped default is off"

    await ranker.rank(
        _NS,
        "who does Ada manage?",
        query_vec,
        limit=10,
        channels=RecallChannels(),
        caller_identity_set=None,
    )

    assert ltm.seed_entity_uids_seen is None


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


# ---------------------------------------------------------------------------------------------
# S1b — read-time neighbour expansion (TRACE-0923.md §7/§6.2/§5.1, `ranker.py::_expand_neighbors`).
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s1b_neighbor_expansion_off_by_default() -> None:
    """``neighbor_expand_radius`` defaults to 0 — a candidate outside the returned window must
    not appear just because it happens to be a turn_seq neighbour of something that made it."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    fillers = [
        _item(f"filler #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n), turn_seq=100 + n)
        for n in range(10)
    ]
    neighbor_item = _item(
        "completely unrelated old aside",
        tier=MemoryTier.STM,
        at=base - timedelta(days=1),
        turn_seq=6,
    )
    stm_items = [neighbor_item, *fillers]
    strong_mtm = _item(
        "Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base, turn_seq=5
    )
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={query_vec: [strong_mtm]},
        settings=RecallSettings(stm_scoring="recency", floor_protect_limit=0),
    )

    ids = await _rank(ranker, list(query_vec), limit=1)

    assert ids == [strong_mtm.id]
    assert neighbor_item.id not in ids


@pytest.mark.asyncio
async def test_s1b_legacy_rows_with_no_turn_seq_degrade_gracefully_when_expansion_is_on() -> None:
    """TRACE-0923.md §7's explicit S1b requirement, and ``_expand_neighbors``'s own contract:
    every row written before ``turn_seq`` existed (or by a write path that assigns none) carries
    ``turn_seq=None`` and must be SKIPPED, never read as ``0`` and never raise — with expansion
    switched ON. Added by the verify pass: no pure-unit test covered this path (mutating
    ``_expand_neighbors`` to a hard no-op left the whole unit suite green)."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    # A row that WOULD be pulled in as a neighbour if a `turn_seq=None` anchor were read as 0 —
    # `recency_floor_limit=0` keeps the STM channel from surfacing it on its own, so its presence
    # in the result can only come from expansion.
    trap = _item("a row at turn_seq 1", tier=MemoryTier.STM, at=base, turn_seq=1)
    legacy_mtm = _item("a legacy MTM fact", tier=MemoryTier.MTM, at=base)  # turn_seq is None
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=[trap],
        mtm_hits_by_query={query_vec: [legacy_mtm]},
        settings=RecallSettings(
            neighbor_expand_radius=2,
            stm_scoring="recency",
            floor_protect_limit=0,
            recency_floor_limit=0,
        ),
    )

    ids = await _rank(ranker, list(query_vec), limit=10)

    assert ids == [legacy_mtm.id], (
        "a turn_seq=None anchor was expanded anyway — it must be SKIPPED, never read as 0 "
        f"(reading it as 0 pulls in the turn_seq=1 trap row): {ids!r}"
    )


@pytest.mark.asyncio
async def test_s1b_expansion_inserts_a_neighbour_that_no_channel_ranked() -> None:
    """The POSITIVE half of S1b, deterministically and store-free. Added by the verify pass:
    before it, every pure-unit S1b test asserted the neighbour's ABSENCE, so mutating
    ``_expand_neighbors`` into an unconditional no-op left the unit suite entirely green and only
    the real-Redis integration test caught it."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    # One anchor the MTM channel ranks, and one STM row that NOTHING ranks into the window on its
    # own — it is only reachable as the anchor's turn_seq neighbour.
    neighbor_item = _item(
        "the reply that carries the answer", tier=MemoryTier.STM, at=base, turn_seq=6
    )
    anchor = _item(
        "Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base, turn_seq=5
    )
    query_vec = (0.9, 0.1)

    def _ids(radius: int) -> ThreeChannelRecallRanker:
        return _build_ranker(
            stm_items=[neighbor_item],
            mtm_hits_by_query={query_vec: [anchor]},
            settings=RecallSettings(
                neighbor_expand_radius=radius,
                stm_scoring="recency",
                floor_protect_limit=0,
                recency_floor_limit=0,  # the STM channel contributes NOTHING on its own
            ),
        )

    off = await _rank(_ids(0), list(query_vec), limit=10)
    on = await _rank(_ids(1), list(query_vec), limit=10)

    assert off == [anchor.id], f"fixture broken — the neighbour is reachable without S1b: {off!r}"
    assert neighbor_item.id in on, (
        "neighbour expansion at radius=1 did not insert the anchor's ±1 turn_seq neighbour — the "
        f"S1b read path is a no-op: {on!r}"
    )


@pytest.mark.asyncio
async def test_s1b_neighbor_never_borrows_a_higher_weighted_channels_scale() -> None:
    """Regression guard for a scoring bug caught live on `mu-dev-vm` (measured account in
    `_expand_neighbors`'s own docstring): a neighbour discovered via an MTM-sourced anchor must
    compete at the STM channel's OWN ``weight_stm``-discounted scale, never at the anchor's
    higher-weighted MTM scale — otherwise an irrelevant neighbour of a top MTM hit crowds out a
    genuinely second-best MTM candidate purely because of which channel happened to find it."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    # 10 recent, irrelevant fillers fill the default `recency_floor_limit=10` window; the
    # neighbour (turn_seq=6, oldest) is pushed OUTSIDE it — reachable only via
    # `_expand_neighbors`'s own wider session scan, never the ordinary `floor` fetch, so this
    # test exercises a GENUINE insertion, not a no-op skip of an already-present candidate.
    fillers = [
        _item(f"filler #{n}", tier=MemoryTier.STM, at=base + timedelta(minutes=n), turn_seq=100 + n)
        for n in range(10)
    ]
    neighbor_item = _item(
        "completely unrelated old aside",
        tier=MemoryTier.STM,
        at=base - timedelta(days=1),
        turn_seq=6,
    )
    stm_items = [neighbor_item, *fillers]
    strong_mtm = _item(
        "Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base, turn_seq=5
    )
    second_mtm = _item("Ada's dentist appointment is on Friday", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)

    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={query_vec: [strong_mtm, second_mtm]},
        settings=RecallSettings(
            neighbor_expand_radius=1, stm_scoring="recency", floor_protect_limit=0
        ),
    )

    ids = await _rank(ranker, list(query_vec), limit=2)

    assert ids == [strong_mtm.id, second_mtm.id], (
        "an irrelevant neighbour of the top MTM anchor crowded out the genuinely second-best "
        f"MTM candidate — the neighbour-scoring regression this test guards against: {ids!r}"
    )


@pytest.mark.asyncio
async def test_s1b_expansion_does_not_discard_the_rerank_gates_ordering() -> None:
    """VERIFY PASS 2026-09-23 — regression test for a real interaction defect between the two
    read-path stages. ``AdaptiveRerankGate.apply`` reorders the pool and records its verdict in
    ``RecallItemView.rerank_score``; it deliberately does NOT rewrite ``fused_score`` (which
    D2/``_to_view`` require to stay the RRF value). ``_expand_neighbors`` then ran
    ``expanded.sort(key=fused_score)`` over the WHOLE list, which silently reverted the pool to
    raw RRF order — so switching ``neighbor_expand_radius`` on turned the reranker off in effect.
    Both knobs are operator-flippable, so the combination is reachable in production."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    # RRF ranks `first_by_rrf` ahead of `second_by_rrf` (MTM channel order); the reranker inverts
    # that, and both clear the gate's cutoff so neither is pruned.
    first_by_rrf = _item("chatter about the weather", tier=MemoryTier.MTM, at=base, turn_seq=5)
    second_by_rrf = _item("Ada's flight to Denver is Thursday", tier=MemoryTier.MTM, at=base)
    query_vec = [0.5, 0.5]
    reranker = _ContentScoredReranker({second_by_rrf.content: 0.9, first_by_rrf.content: 0.8})

    def _ranker(radius: int) -> ThreeChannelRecallRanker:
        return ThreeChannelRecallRanker(
            stm=_FakeStm([]),  # type: ignore[arg-type]
            mtm=_FakeMtm({tuple(query_vec): [first_by_rrf, second_by_rrf]}),  # type: ignore[arg-type]
            ltm=_EmptyLtm(),  # type: ignore[arg-type]
            fusion=ReciprocalRankFusion(),
            settings=RecallSettings(
                stm_scoring="recency",
                rerank_enabled=True,
                rerank_min_score=0.5,
                rerank_top_fraction=0.5,
                neighbor_expand_radius=radius,
            ),
            clock=FrozenClock(datetime(2026, 7, 31, tzinfo=UTC)),
            reranker=reranker,
        )

    async def _ids(radius: int) -> list[str]:
        result = await _ranker(radius).rank(
            _NS,
            "when is Ada's flight",
            query_vec,
            limit=10,
            channels=RecallChannels(),
            caller_identity_set=frozenset[str](),
        )
        return [it.memory_id for it in result.items]

    off = await _ids(0)
    assert off == [
        second_by_rrf.id,
        first_by_rrf.id,
    ], f"fixture broken — rerank did not reorder: {off!r}"

    on = await _ids(1)
    assert on == off, (
        "turning neighbour expansion on silently reverted the pool to raw RRF order and threw "
        f"away the rerank gate's verdict: rerank-only {off!r} vs with-expansion {on!r}"
    )


@pytest.mark.asyncio
async def test_s1b_after_anchor_placement_puts_the_neighbour_inside_the_window() -> None:
    """AD-232 / `RecallSettings.neighbor_expand_placement`. Under the shipped ``"tail"`` policy a
    neighbour scores below every real candidate and is therefore truncated away whenever the pool
    is already ``limit``-deep — which is why S1b measured as a no-op. ``"after_anchor"`` places it
    directly behind the candidate that surfaced it, so it survives the same truncation."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    neighbor_item = _item(
        "the reply that carries the answer", tier=MemoryTier.STM, at=base, turn_seq=6
    )
    anchor = _item("Ada's flight to Denver is Thursday", tier=MemoryTier.MTM, at=base, turn_seq=5)
    filler = _item("an unrelated but well-ranked MTM row", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)

    def _ranker(placement: str) -> ThreeChannelRecallRanker:
        return _build_ranker(
            stm_items=[neighbor_item],
            mtm_hits_by_query={query_vec: [anchor, filler]},
            settings=RecallSettings(
                neighbor_expand_radius=1,
                neighbor_expand_placement=placement,  # type: ignore[arg-type]
                stm_scoring="recency",
                floor_protect_limit=0,
                recency_floor_limit=0,
            ),
        )

    tail = await _rank(_ranker("tail"), list(query_vec), limit=2)
    after = await _rank(_ranker("after_anchor"), list(query_vec), limit=2)

    assert tail == [anchor.id, filler.id], f"'tail' should truncate the neighbour away: {tail!r}"
    assert after == [
        anchor.id,
        neighbor_item.id,
    ], f"'after_anchor' did not place the neighbour behind its anchor: {after!r}"


# ---------------------------------------------------------------------------------------------
# Shape A / Shape B — TRACE-0923.md follow-up, ARCHITECTURE-DELTAS.md AD-233's own amendment:
# "the two live options are the free-riding insertion ... or a wider fetch narrowed after
# expansion. Nothing else is left." `neighbor_free_ride` / `neighbor_expand_widen`.
# ---------------------------------------------------------------------------------------------


async def _full_rank(
    ranker: ThreeChannelRecallRanker, query_vec: Vector, *, limit: int = 10
) -> list[RecallItemView]:
    result = await ranker.rank(
        _NS,
        "irrelevant-this-phase",
        query_vec,
        limit=limit,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )
    return list(result.items)


def _two_mtm_one_neighbor_fixture() -> tuple[list[MemoryItem], MemoryItem, MemoryItem, MemoryItem]:
    """One anchor (turn_seq=5) reachable via MTM, a second MTM hit nothing links to a neighbour,
    and one STM row (turn_seq=6) reachable ONLY as the anchor's turn_seq neighbour — the SAME
    shape ``test_s1b_expansion_inserts_a_neighbour_that_no_channel_ranked`` uses, with a second
    MTM candidate added so a "does this displace a real candidate" question is answerable.
    Returns ``(stm_items, anchor, second_mtm, neighbor_item)``."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    neighbor_item = _item(
        "the reply that carries the answer", tier=MemoryTier.STM, at=base, turn_seq=6
    )
    anchor = _item(
        "Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base, turn_seq=5
    )
    second_mtm = _item("Ada's dentist appointment is on Friday", tier=MemoryTier.MTM, at=base)
    return [neighbor_item], anchor, second_mtm, neighbor_item


@pytest.mark.asyncio
async def test_shipped_default_never_shows_the_neighbour_and_never_exceeds_limit() -> None:
    """Baseline pin — neither Shape A nor Shape B enabled: the costs-a-slot mechanism truncates
    the neighbour away exactly as AD-231/AD-232 measured, and the result is always exactly
    `limit` items."""
    stm_items, anchor, second_mtm, neighbor_item = _two_mtm_one_neighbor_fixture()
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={query_vec: [anchor, second_mtm]},
        settings=RecallSettings(
            neighbor_expand_radius=1,
            stm_scoring="recency",
            floor_protect_limit=0,
            recency_floor_limit=0,  # isolate: the neighbour must be reachable ONLY via expansion
        ),
    )

    items = await _full_rank(ranker, list(query_vec), limit=2)

    assert [it.memory_id for it in items] == [anchor.id, second_mtm.id]
    assert len(items) == 2
    assert neighbor_item.id not in [it.memory_id for it in items]


@pytest.mark.asyncio
async def test_neighbor_free_ride_adds_the_neighbour_without_displacing_anything() -> None:
    """Shape A. The neighbour is APPENDED past `limit` — nothing already selected is displaced,
    and the result is allowed to exceed `limit` (the ADR's own "bounded to prompt length, never
    to precision" trade)."""
    stm_items, anchor, second_mtm, neighbor_item = _two_mtm_one_neighbor_fixture()
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={query_vec: [anchor, second_mtm]},
        settings=RecallSettings(
            neighbor_expand_radius=1,
            neighbor_free_ride=True,
            stm_scoring="recency",
            floor_protect_limit=0,
            recency_floor_limit=0,
        ),
    )

    items = await _full_rank(ranker, list(query_vec), limit=2)
    ids = [it.memory_id for it in items]

    assert ids == [
        anchor.id,
        second_mtm.id,
        neighbor_item.id,
    ], f"free-riding must keep both real winners AND append the neighbour: {ids!r}"
    assert len(items) == 3, "free-riding must be allowed to exceed the caller's limit"
    neighbor_view = next(it for it in items if it.memory_id == neighbor_item.id)
    assert neighbor_view.is_neighbor is True


@pytest.mark.asyncio
async def test_neighbor_free_ride_only_attaches_to_anchors_that_actually_survived() -> None:
    """A neighbour of an anchor that did NOT make the final `limit` window must not appear either
    — free-riding rides with a WINNER, it does not independently resurrect a loser's context."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    # `winner` carries a turn_seq far from anything else in the fixture, so it IS a valid anchor
    # (the free-ride mechanism inspects every surviving item's `turn_seq`) but has no neighbour of
    # its own to find — isolating "did the LOSER's neighbour leak through" from "does a winner
    # with no real neighbour break anything".
    winner = _item(
        "Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base, turn_seq=50
    )
    loser = _item(
        "Ada's dentist appointment is on Friday", tier=MemoryTier.MTM, at=base, turn_seq=5
    )
    losers_neighbor = _item("the reply nobody should see", tier=MemoryTier.STM, at=base, turn_seq=6)
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=[losers_neighbor],
        mtm_hits_by_query={query_vec: [winner, loser]},
        settings=RecallSettings(
            neighbor_expand_radius=1,
            neighbor_free_ride=True,
            stm_scoring="recency",
            floor_protect_limit=0,
            recency_floor_limit=0,
        ),
    )

    items = await _full_rank(ranker, list(query_vec), limit=1)
    ids = [it.memory_id for it in items]

    assert ids == [winner.id], f"only the surviving winner may free-ride a neighbour: {ids!r}"


@pytest.mark.asyncio
async def test_neighbor_expand_widen_lets_the_neighbour_displace_the_weakest_real_candidate() -> (
    None
):
    """Shape B. Unlike free-riding, this DOES cost something — the result stays exactly `limit`
    items, and the weakest ordinary candidate (`second_mtm`) is displaced to make room for the
    neighbour, which is the priced trade `neighbor_expand_widen`'s own docstring names."""
    stm_items, anchor, second_mtm, neighbor_item = _two_mtm_one_neighbor_fixture()
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={query_vec: [anchor, second_mtm]},
        settings=RecallSettings(
            neighbor_expand_radius=1,
            neighbor_expand_widen=1,
            stm_scoring="recency",
            floor_protect_limit=0,
            recency_floor_limit=0,
        ),
    )

    items = await _full_rank(ranker, list(query_vec), limit=2)
    ids = [it.memory_id for it in items]

    assert ids == [
        anchor.id,
        neighbor_item.id,
    ], f"widen=1 should displace the weakest real candidate for the neighbour: {ids!r}"
    assert len(items) == 2, "unlike free-riding, Shape B must never exceed the caller's limit"
    assert second_mtm.id not in ids, "the displaced candidate is the priced cost of this shape"


@pytest.mark.asyncio
async def test_neighbor_expand_widen_zero_is_byte_identical_to_the_shipped_default() -> None:
    stm_items, anchor, second_mtm, _neighbor_item = _two_mtm_one_neighbor_fixture()
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=stm_items,
        mtm_hits_by_query={query_vec: [anchor, second_mtm]},
        settings=RecallSettings(
            neighbor_expand_radius=1,
            neighbor_expand_widen=0,
            stm_scoring="recency",
            floor_protect_limit=0,
            recency_floor_limit=0,
        ),
    )

    items = await _full_rank(ranker, list(query_vec), limit=2)

    assert [it.memory_id for it in items] == [anchor.id, second_mtm.id]


@pytest.mark.asyncio
async def test_neighbor_expand_widen_never_evicts_a_protected_floor_member() -> None:
    """The correctness property `_narrow_after_expansion` exists for: widening the working limit
    must not let a rescued, unconditionally-protected floor member (AD-195/ADR 0052) get cut back
    out by the narrow step. `recency_floor_limit`/`floor_protect_limit` are left at shipped
    defaults here specifically so a real protected member is in play."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    # A lone, very stale STM item — protected by `floor_protect_limit` regardless of relevance
    # under `stm_scoring="recency"` (D1's documented uniform on/off — see `_protected_floor_ids`'s
    # own docstring).
    protected = _item("a just-said aside", tier=MemoryTier.STM, at=base, turn_seq=1)
    # `anchor` carries a turn_seq with NO row at ±1 in the STM fixture (`protected` sits at 1,
    # far below 100) — `neighbor_expand_radius` must be > 0 for `neighbor_expand_widen` to be
    # live at all (dto.py's own docstring: "ignored when neighbor_expand_radius == 0"), but this
    # keeps the fixture from actually INSERTING a neighbour, isolating the floor-rescue
    # interaction this test is about from the neighbour-insertion mechanism the other tests cover.
    anchor = _item(
        "Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base, turn_seq=100
    )
    second_mtm = _item("Ada's dentist appointment is on Friday", tier=MemoryTier.MTM, at=base)
    third_mtm = _item("Ada's dry cleaning is ready", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(
        stm_items=[protected],
        mtm_hits_by_query={query_vec: [anchor, second_mtm, third_mtm]},
        settings=RecallSettings(
            neighbor_expand_radius=1,
            neighbor_expand_widen=1,
            stm_scoring="recency",
        ),
    )

    items = await _full_rank(ranker, list(query_vec), limit=2)
    ids = [it.memory_id for it in items]

    assert (
        protected.id in ids
    ), f"a protected floor member was evicted by the widen/narrow round-trip: {ids!r}"
    assert len(items) == 2


def _nv(mid: str, score: float, *, is_floor: bool = False, is_neighbor: bool = False):
    """A bare `RecallItemView` for the direct `_narrow_after_expansion` tests below — this helper
    exercises the pure function on an order the caller controls exactly, instead of trying to coax
    the whole composed ranker into producing one (which, under any weighting reachable from
    `RecallSettings`, keeps every STM-family row at the tail anyway — see the docstring below)."""
    return RecallItemView(
        memory_id=mid,
        content=f"content {mid}",
        content_hash=f"h-{mid}",
        tier=MemoryTier.STM if (is_floor or is_neighbor) else MemoryTier.MTM,
        channel="stm" if (is_floor or is_neighbor) else "mtm",
        namespace=_NS,
        fused_score=score,
        is_floor=is_floor,
        is_neighbor=is_neighbor,
    )


def test_narrow_after_expansion_filters_the_pool_it_never_reorders_it() -> None:
    """VERIFY 2026-09-24 — Shape B's narrow step rebuilt the pool in a NEW order.

    `_narrow_after_expansion` chose its survivors by partitioning the merged pool into three
    buckets and CONCATENATING them — `[*rest[:room], *rescued_neighbors, *protected]`. That is a
    RE-ORDER of a list every stage upstream and downstream treats as a RANKING, and it is the same
    class of defect AD-232 already found one stage earlier: `_expand_neighbors` used to end with
    `expanded.sort(...)`, silently discarding `AdaptiveRerankGate.apply`'s ordering. The rule that
    fix established is the one applied here — **decide membership, never position**.

    Two things the concatenation broke, both visible in the assertion below. `_merge_floor`'s own
    D3 contract (STATE-AND-DEFECTS-0829.md) is that "a protected member keeps whatever position
    fusion actually earned it" — the concatenation moved every protected member to the END
    regardless. And the injector's token budgeter trims from the TAIL, so a protected just-said
    fact that fusion had ranked FIRST became the first thing a tight budget dropped, behind a
    bottom-scored speculative neighbour.

    Tested directly on the pure helper rather than through `rank()`: under every weighting
    `RecallSettings` can express, an STM-family row (a floor member or a neighbour) lands at the
    tail of the fused pool anyway, so the composed ranker cannot be made to produce the input
    order that discriminates the two implementations. That is also why this reorder was never
    noticed — it is unobservable end-to-end today and would surface the moment the fuse's ordering
    changed. Mutation check: restore the concatenation and this goes red.
    """
    floor_first = _nv("F", 0.9, is_floor=True)  # fusion ranked the protected member FIRST
    strong = _nv("A", 0.8)
    weak = _nv("B", 0.7)
    neighbour = _nv("N", 0.001, is_neighbor=True)  # appended last, by construction

    out = _narrow_after_expansion(
        [floor_first, strong, weak, neighbour], limit=3, neighbor_rescue_budget=1
    )
    ids = [v.memory_id for v in out]

    assert ids == ["F", "A", "N"], (
        "the narrow step must FILTER the pool, keeping the incoming order: the protected member "
        "keeps the first position fusion gave it (D3), the weakest ordinary candidate `B` is what "
        "the rescued neighbour displaces, and the neighbour stays where it was appended. The "
        f"pre-fix concatenation returned ['A', 'N', 'F'] instead: {ids!r}"
    )


def test_narrow_after_expansion_still_keeps_every_protected_member() -> None:
    """The guarantee the rewrite must not lose: `_merge_floor`'s protected members survive the
    narrow unconditionally (AD-195/ADR 0052), even when the neighbour budget would otherwise want
    their slot."""
    out = _narrow_after_expansion(
        [
            _nv("A", 0.9),
            _nv("F1", 0.5, is_floor=True),
            _nv("F2", 0.4, is_floor=True),
            _nv("N", 0.001, is_neighbor=True),
        ],
        limit=2,
        neighbor_rescue_budget=1,
    )
    ids = [v.memory_id for v in out]

    assert ids == ["F1", "F2"], f"a protected member lost its slot to the neighbour budget: {ids!r}"


# =================================================================================================
# AD-259 — the MTM read-stat write-back (`_reinforce_mtm_hits`)
# =================================================================================================
class _ReinforcingMtm(_FakeMtm):
    """``_FakeMtm`` plus a real, functioning ``reinforce`` — the SHIPPED shape (Qdrant/Weaviate).

    Deliberately a SUBCLASS rather than an edit to ``_FakeMtm``: the base class staying WITHOUT
    ``reinforce`` is what keeps every other test in this module an ongoing regression test for
    the capability guard in ``_reinforce_mtm_hits`` (see
    ``test_a_vector_backend_without_reinforce_does_not_break_recall`` below).
    """

    def __init__(self, hits_by_query: dict[tuple[float, ...], list[MemoryItem]]) -> None:
        super().__init__(hits_by_query)
        self.reinforced: list[str] = []

    async def reinforce(self, ns: Namespace, memory_id: str, *, at: datetime) -> MemoryItem | None:
        self.reinforced.append(memory_id)
        return None


@pytest.mark.asyncio
async def test_a_recalled_mtm_hit_is_reinforced_exactly_once() -> None:
    """AD-259: the trigger that was missing entirely. ``recall-service-design.md`` §5.1 cites
    ``mtm_qdrant.py:388`` as already doing this write-back; that file contained the string
    ``access_count`` ZERO times, so a memory recalled every day demoted on the identical schedule
    as one nobody ever touched — ``access_count`` is the only salience term a recall can move.

    Proved end to end against real Qdrant in
    ``tests/lifecycle/test_lifecycle_walk_end_to_end_int.py::
    test_a_memory_the_user_keeps_recalling_is_not_demoted_on_schedule``; this unit test pins the
    ranker-side contract (once per DISTINCT surviving MTM id, never per channel candidate).

    MUTATION CHECK (run, red): remove ``self._reinforce_mtm_hits(ns, items)`` from the
    ``asyncio.gather`` in ``rank`` — ``mtm.reinforced`` stays empty.
    """
    base = datetime(2026, 7, 31, tzinfo=UTC)
    target = _item("the deploy window is Friday 16:00", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)
    mtm = _ReinforcingMtm({query_vec: [target]})
    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([]),
        mtm=mtm,  # type: ignore[arg-type]
        ltm=_EmptyLtm(),  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(stm_scoring="recency"),
        clock=FrozenClock(base),
    )

    ids = await _rank(ranker, list(query_vec))

    assert target.id in ids
    assert mtm.reinforced == [
        target.id
    ], f"expected exactly one reinforcement of the surviving MTM hit, got {mtm.reinforced}"


@pytest.mark.asyncio
async def test_reinforce_on_recall_off_reinforces_no_mtm_hit() -> None:
    """The A/B off-switch covers BOTH channels, not just STM — an operator who turns the
    write-back off must get a genuinely read-only recall."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    target = _item("the deploy window is Friday 16:00", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)
    mtm = _ReinforcingMtm({query_vec: [target]})
    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([]),
        mtm=mtm,  # type: ignore[arg-type]
        ltm=_EmptyLtm(),  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(stm_scoring="recency", reinforce_on_recall=False),
        clock=FrozenClock(base),
    )

    ids = await _rank(ranker, list(query_vec))

    assert target.id in ids
    assert mtm.reinforced == []


@pytest.mark.asyncio
async def test_a_vector_backend_without_reinforce_does_not_break_recall() -> None:
    """AD-259's blast radius, pinned. ``STORE_REGISTRY.build`` is typed ``-> Any``, and three of
    the six shipped vector backends (``PgVectorMtmAdapter``/``ChromaMtmAdapter``/
    ``FaissMtmAdapter``) implement no by-id verbs at all (``tier_capabilities.py``'s own module
    docstring names them). Adding an unconditional by-id WRITE to the READ path would therefore
    turn every recall on those deployments into an ``AttributeError`` — a hot-path crash bought
    for a best-effort stat write.

    ``_FakeMtm`` (the base class, with no ``reinforce``) stands in for exactly that backend.

    MUTATION CHECK (run, red): delete the ``if not hasattr(self._mtm, "reinforce"): return``
    guard in ``_reinforce_mtm_hits`` — this test fails with
    ``AttributeError: '_FakeMtm' object has no attribute 'reinforce'``, and so do ~40 others in
    this module.
    """
    base = datetime(2026, 7, 31, tzinfo=UTC)
    target = _item("the deploy window is Friday 16:00", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(stm_items=[], mtm_hits_by_query={query_vec: [target]})

    ids = await _rank(ranker, list(query_vec))  # must not raise

    assert target.id in ids


@pytest.mark.asyncio
async def test_a_memory_in_both_tiers_is_reinforced_in_both_stores() -> None:
    """AD-259's correction to ADR 0061, and the reason the end-to-end walk caught what the
    piece-tests could not.

    ``_channel_label`` labels the FUSED WINNER. A memory that lives in BOTH tiers — the ordinary
    case for anything recently ingested, since ``WriteStmStage`` and ``DeterministicPromoteStage``
    both write it — fuses to ONE view under ONE label. ADR 0061's ``channel == "stm"`` filter
    therefore reinforced the Valkey row and left the Qdrant point at ``access_count=0``, which is
    the row the DEMOTION gate reads (``scan_for_demotion``). MEASURED on real stores: ten genuine
    recalls, Valkey 10, Qdrant 0.

    The two copies are distinct rows with independent lifecycle gates, so a recall of that memory
    is a genuine use of both. Both legs now take every returned id; ``reinforce`` on a store that
    does not hold the id is a documented no-op.

    MUTATION CHECK (run, red): restore either leg's ``if v.channel == "..."`` filter — the store
    on the other side of the label records nothing.
    """
    base = datetime(2026, 7, 31, tzinfo=UTC)
    both = _item("the deploy window is Friday 16:00", tier=MemoryTier.MTM, at=base)
    query_vec = (0.9, 0.1)
    stm = _FakeStm([both])  # the SAME id also sits in the STM recency floor
    mtm = _ReinforcingMtm({query_vec: [both]})
    ranker = ThreeChannelRecallRanker(
        stm=stm,  # type: ignore[arg-type]
        mtm=mtm,  # type: ignore[arg-type]
        ltm=_EmptyLtm(),  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(stm_scoring="recency"),
        clock=FrozenClock(base),
    )

    ids = await _rank(ranker, list(query_vec))

    assert both.id in ids
    assert mtm.reinforced == [both.id], (
        "the MTM point was not reinforced — this is the AD-259 defect: the fused view carried the "
        "STM label, so only the Valkey row was reinforced and the Qdrant point the demotion gate "
        "reads stayed at access_count=0"
    )
    stm_row = await stm.get(_NS, both.id)
    assert stm_row is not None
    assert stm_row.access_count == 1, "the STM row was not reinforced either"


# =================================================================================================
# ADR 0062 / AD-259's named-open item — the LTM read-stat write-back (`_reinforce_ltm_hits`)
# =================================================================================================
class _ReinforcingLtm(_FakeLtm):
    """``_FakeLtm`` plus a real, functioning ``reinforce`` — the SHIPPED shape
    (``FalkorLtmAdapter``).

    Deliberately a SUBCLASS rather than an edit to ``_FakeLtm``/``_EmptyLtm``: those staying
    WITHOUT ``reinforce`` is what keeps every OTHER test in this module an ongoing regression
    test for the capability guard in ``_reinforce_ltm_hits`` (see
    ``test_a_graph_backend_without_reinforce_does_not_break_recall`` below).
    """

    def __init__(self, hits: list[MemoryItem]) -> None:
        super().__init__(hits)
        self.reinforced: list[str] = []

    async def reinforce(self, ns: Namespace, memory_id: str, *, at: datetime) -> MemoryItem | None:
        self.reinforced.append(memory_id)
        return None


@pytest.mark.asyncio
async def test_a_recalled_ltm_hit_is_reinforced_exactly_once() -> None:
    """ADR 0062 / AD-259's still-open half, closed: ``LtmTierRepository`` had no ``reinforce`` at
    all, so a COLD LTM fact's ``COLD -> ACTIVE`` reactivate-on-recall edge (spec §9,
    ``recall-service-design.md`` §5.1) had no trigger and ``RetentionService`` had nothing to
    read on its next sweep. This pins the ranker-side contract (once per DISTINCT surviving LTM
    id, never per channel candidate) — the storage-layer stat bump and the
    ``RetentionService._sweep`` reactivation this write-back feeds are proved separately, against
    real FalkorDB, in ``tests/storage/test_graph_falkor_int.py`` and
    ``tests/lifecycle/test_retention_int.py``.

    MUTATION CHECK (run, red): remove ``self._reinforce_ltm_hits(ns, items)`` from the
    ``asyncio.gather`` in ``rank`` — ``ltm.reinforced`` stays empty.
    """
    base = datetime(2026, 7, 31, tzinfo=UTC)
    target = _item("Ada uses Postgres", tier=MemoryTier.LTM, at=base)
    query_vec = (0.9, 0.1)
    ltm = _ReinforcingLtm([target])
    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([]),
        mtm=_FakeMtm({}),  # type: ignore[arg-type]
        ltm=ltm,  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(stm_scoring="recency"),
        clock=FrozenClock(base),
    )

    ids = await _rank(ranker, list(query_vec))

    assert target.id in ids
    assert ltm.reinforced == [
        target.id
    ], f"expected exactly one reinforcement of the surviving LTM hit, got {ltm.reinforced}"


@pytest.mark.asyncio
async def test_reinforce_on_recall_off_reinforces_no_ltm_hit() -> None:
    """The A/B off-switch covers ALL THREE channels, not just STM/MTM — an operator who turns the
    write-back off must get a genuinely read-only recall."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    target = _item("Ada uses Postgres", tier=MemoryTier.LTM, at=base)
    query_vec = (0.9, 0.1)
    ltm = _ReinforcingLtm([target])
    ranker = ThreeChannelRecallRanker(
        stm=_FakeStm([]),
        mtm=_FakeMtm({}),  # type: ignore[arg-type]
        ltm=ltm,  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=RecallSettings(stm_scoring="recency", reinforce_on_recall=False),
        clock=FrozenClock(base),
    )

    ids = await _rank(ranker, list(query_vec))

    assert target.id in ids
    assert ltm.reinforced == []


@pytest.mark.asyncio
async def test_a_graph_backend_without_reinforce_does_not_break_recall() -> None:
    """The LTM twin of ``test_a_vector_backend_without_reinforce_does_not_break_recall``.
    ``FalkorLtmAdapter`` is the only shipped ``GraphStorePort`` implementation today and it DOES
    implement ``reinforce`` — but the port is a ``Protocol`` (``STORE_REGISTRY.build`` is typed
    ``-> Any``, exactly like the vector role), so an alternate/future graph backend or a test
    double built against the pre-AD-259 port shape must degrade, not crash, the read path.
    ``_FakeLtm``/``_EmptyLtm`` (no ``reinforce``) stand in for exactly that.

    MUTATION CHECK (run, red): delete the ``if not hasattr(self._ltm, "reinforce"): return``
    guard in ``_reinforce_ltm_hits`` — this test fails with
    ``AttributeError: '_FakeLtm' object has no attribute 'reinforce'``, and so do every other
    test in this module that ever populates the LTM channel.
    """
    base = datetime(2026, 7, 31, tzinfo=UTC)
    target = _item("Ada uses Postgres", tier=MemoryTier.LTM, at=base)
    query_vec = (0.9, 0.1)
    ranker = _build_ranker(stm_items=[], mtm_hits_by_query={}, ltm_hits=[target])

    ids = await _rank(ranker, list(query_vec))  # must not raise

    assert target.id in ids


# =================================================================================================
# AD-273 — the per-channel RRF constant (`RecallSettings.rrf_k_ltm`, `fusion.py`'s optional `ks`).
#
# ADR 0069/AD-273 shipped this lever and used it to run six `gold_in_context` arms whose NEGATIVE
# result ("no `k_ltm` both wins slots and helps") is now the settled answer on the graph tier. That
# conclusion is only worth anything if the lever was genuinely live: a silently-ignored `ks` would
# have produced arms A-C's "0 ltm items" for a reason that has nothing to do with fusion, and the
# whole experiment would be an artifact. It shipped with ZERO tests (verified by grep: no test in
# any repo referenced `rrf_k_ltm`, and no fusion test passed `ks=`). These four are that test.
# =================================================================================================
def test_rrf_k_ltm_ships_inert() -> None:
    """Literal guard on the shipped default, the AD-273 twin of
    ``test_shipped_default_ltm_protect_limit_is_zero``: every measured non-``None`` value cost
    4-7 `gold_in_context` queries, so the mechanism ships built and turned OFF."""
    assert RecallSettings().rrf_k_ltm is None


def test_ks_none_is_byte_identical_to_the_shared_k() -> None:
    """``ks=None`` — what every call site predating AD-273 still passes, ``RecallService``'s own
    private/shared federation fuse included — must reproduce the shared-``k`` arithmetic EXACTLY,
    not approximately. Equal floats, not `pytest.approx`: the claim in `fusion.py`'s docstring is
    "BYTE-IDENTICAL", and anything weaker would let a rounding change ride in unnoticed."""
    channels = [["a", "b"], ["b", "c"]]
    weights = [1.0, 0.1]
    without = reciprocal_rank_fusion(channels, key=lambda s: s, weights=weights, k=60)
    with_none = reciprocal_rank_fusion(channels, key=lambda s: s, weights=weights, k=60, ks=None)
    explicit_shared = reciprocal_rank_fusion(
        channels, key=lambda s: s, weights=weights, k=60, ks=[60, 60]
    )
    assert without == with_none
    assert without == explicit_shared


def test_a_per_channel_k_changes_only_its_own_channels_decay() -> None:
    """The arithmetic AD-273's whole experiment rests on, asserted directly: dropping ONE
    channel's ``k`` raises that channel's rank-0 contribution by exactly ``k_shared+1`` over
    ``k_channel+1`` and leaves every OTHER channel's score untouched.

    MUTATION CHECK (run, red both ways): revert `fusion.py`'s ``channel_k = ks[idx] if ks is not
    None else k`` to ``channel_k = k`` and this test fails on the first assertion ("the per-channel
    k did nothing") — the ``ks`` parameter would be accepted, validated for length, and silently
    ignored, which is the exact shape that would have invalidated ADR 0069's six eval arms."""
    channels = [["mtm-only"], ["ltm-only"]]
    weights = [1.0, 0.1]
    shared = dict(reciprocal_rank_fusion(channels, key=lambda s: s, weights=weights, k=60))
    per_channel = dict(
        reciprocal_rank_fusion(channels, key=lambda s: s, weights=weights, k=60, ks=[60, 5])
    )

    assert (
        per_channel["ltm-only"] > shared["ltm-only"]
    ), "the per-channel k did nothing — `ks` was accepted and ignored"
    # exact, not approximate: (0.1/1.1) * 1/6  vs  (0.1/1.1) * 1/61
    assert per_channel["ltm-only"] == pytest.approx(shared["ltm-only"] * 61 / 6)
    assert (
        per_channel["mtm-only"] == shared["mtm-only"]
    ), "lowering the LTM channel's k moved a channel it has no business touching"


@pytest.mark.asyncio
async def test_rrf_k_ltm_lets_the_ltm_seed_take_a_slot_the_shared_k_denies_it() -> None:
    """END TO END through the ranker, which is where AD-273's eval arms actually ran: the SAME
    scenario as ``test_default_settings_rank_the_relevant_mtm_hit_ahead_of_query_blind_ltm_noise``
    (a query-blind LTM candidate at LTM rank 0 vs the genuinely relevant MTM hit at MTM rank 3),
    run twice — once at the shipped default and once at ``rrf_k_ltm=5``, the exact arm ADR 0069
    measured as "wins 383/383 slots".

    At the shipped weights (0.1/1.0/0.1, normalized 1/12 : 10/12 : 1/12) the arithmetic is
    ``(1/12)/(60+1) = 0.001366`` for the LTM seed against ``(10/12)/(60+3+1) = 0.013021`` for the
    relevant MTM hit — the seed loses, which is the structural exclusion four passes have
    measured. At ``rrf_k_ltm=5`` it is ``(1/12)/(5+1) = 0.013889`` and the seed WINS the top slot,
    which is why arms D/E/F cost 4-7 queries of `gold_in_context`.

    **This is the test that makes ADR 0069's negative result trustworthy**: it proves the knob was
    live, so arms A-C's "0 ltm items" is a real measurement of the arithmetic and not a
    silently-dropped parameter. MUTATION CHECK (run, red): drop `ranker.py`'s ``channel_ks`` list
    (pass no ``ks=`` to ``self._fusion.fuse``) and the second half fails — the LTM seed stays
    buried at ``rrf_k_ltm=5``, i.e. the setting the eval harness was varying would have changed
    nothing at all."""
    base = datetime(2026, 7, 31, tzinfo=UTC)
    query_vec = (0.9, 0.1)
    decoys = [_item(f"unrelated MTM decoy #{n}", tier=MemoryTier.MTM, at=base) for n in range(3)]
    target = _item("Ada's flight to Denver is on Thursday", tier=MemoryTier.MTM, at=base)
    noise = _item("Let me know if you need anything", tier=MemoryTier.LTM, at=base)

    async def _order(settings: RecallSettings) -> list[str]:
        ranker = _build_ranker(
            stm_items=[],
            mtm_hits_by_query={query_vec: [*decoys, target]},
            ltm_hits=[noise],
            settings=settings,
        )
        result = await ranker.rank(
            _NS,
            "irrelevant-this-phase",
            list(query_vec),
            limit=10,
            channels=RecallChannels(),
            caller_identity_set=frozenset[str](),
        )
        return [it.memory_id for it in result.items]

    shipped = await _order(RecallSettings(stm_scoring="recency"))
    assert shipped.index(target.id) < shipped.index(noise.id), (
        "baseline half: at the shipped default the query-blind LTM seed must stay behind the "
        "relevant MTM hit"
    )

    lowered = await _order(RecallSettings(stm_scoring="recency", rrf_k_ltm=5))
    assert lowered.index(noise.id) < lowered.index(target.id), (
        "rrf_k_ltm=5 did NOT change the ranking — the per-channel k never reached fusion, so "
        "ADR 0069's six eval arms were varying a setting with no effect"
    )
