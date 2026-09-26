"""Dynamic channel budgets — owner's second Lane-B ask ("when one path yields too little context,
take more from the others"). Repro + regression tests for ``dto.py``'s ``dynamic_channel_budget``
and ``ranker.py``'s ``_reallocate_channel_widths``/``_tail_density``/the ``rank()`` wiring.

Pure unit tests for the reallocation arithmetic (no I/O, mirrors ``width.py``'s own
``derive_recall_limit`` test shape) plus fake-store wiring tests proving: (a) the mechanism is
dark/byte-identical by default, (b) it refetches ONLY a saturated channel, at the reallocated
width, and leaves a starved channel alone, and (c) the total per-call width budget never grows
past the static baseline.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_engine.platform.clock import FrozenClock
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import (
    ThreeChannelRecallRanker,
    _reallocate_channel_widths,
    _tail_density,
)
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import RecallChannel, Scored

_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)


def _item(content: str, *, at: datetime) -> MemoryItem:
    return MemoryItem(
        content=content,
        kind=MemoryKind.PROPOSITION,
        namespace=_NS,
        state=MemoryState.ACTIVE,
        tier=MemoryTier.MTM,
        owner_id=_NS.user,
        workspace_id=_NS.workspace,
        session_id=_NS.session,
        created_at=at,
        updated_at=at,
    )


def _scored(score: float) -> Scored[MemoryItem]:
    base = datetime(2026, 9, 26, tzinfo=UTC)
    return Scored(item=_item("x", at=base), score=score, channel=RecallChannel.MTM_DENSE)


# ------------------------------------------------------------------------------------------------
# _reallocate_channel_widths — pure arithmetic
# ------------------------------------------------------------------------------------------------


def test_no_starved_channel_is_a_no_op() -> None:
    """Every channel filled its baseline exactly — nothing freed, nothing to redistribute."""
    baseline = {"stm": 10, "mtm": 20, "ltm": 20}
    yields = {"stm": 10, "mtm": 20, "ltm": 20}
    assert _reallocate_channel_widths(baseline, yields) == baseline


def test_starved_channel_frees_width_for_the_only_saturated_channel() -> None:
    """STM and LTM both returned less than their own baseline (both starved, freeing 5 and 20);
    MTM filled its pool (the only saturated channel) and absorbs the entire freed 25 — capped at
    `baseline*max_multiplier - baseline = 40` headroom, well above the 25 on offer here."""
    baseline = {"stm": 10, "mtm": 20, "ltm": 20}
    yields = {"stm": 5, "mtm": 20, "ltm": 0}
    result = _reallocate_channel_widths(baseline, yields)
    assert result["ltm"] == 20, "a starved channel is never refetched at a larger width"
    assert result["stm"] == 10, "a starved channel's own width entry is never rewritten either"
    assert result["mtm"] == 45, "the only saturated channel absorbs the whole freed 5+20"


def test_added_width_never_exceeds_what_starved_channels_actually_freed() -> None:
    """The precise conserved quantity: the WIDTH-TO-FETCH entry for a starved channel is left
    unchanged (never reduced to its yield — refetching it would just repeat the same answer), so
    the raw ``sum(result.values())`` can rise above ``sum(baseline.values())`` whenever anything
    is redistributed. What must never happen is redistributing MORE than starved channels proved
    they are not using: ``sum(added-to-saturated) <= sum(freed-by-starved)``. (The claim that
    matters for `gold_in_context` — that the total number of candidates a real refetch actually
    RETURNS never exceeds the static baseline total — is the end-to-end wiring test below,
    `test_dynamic_channel_budget_bounds_total_actual_candidates_fed_to_fusion`: a starved
    channel's real yield is always <= its baseline by definition, so accounting by ACTUAL yield
    rather than by raw width-entry gives the tighter, literally-conserved number.)"""
    cases = [
        ({"stm": 10, "mtm": 20, "ltm": 20}, {"stm": 3, "mtm": 20, "ltm": 20}),
        ({"stm": 10, "mtm": 20, "ltm": 20}, {"stm": 10, "mtm": 5, "ltm": 20}),
        ({"stm": 10, "mtm": 20, "ltm": 20}, {"stm": 0, "mtm": 0, "ltm": 0}),
        ({"stm": 10, "mtm": 20, "ltm": 20}, {"stm": 10, "mtm": 20, "ltm": 20}),
    ]
    for baseline, yields in cases:
        freed = sum(
            baseline[ch] - yields.get(ch, 0)
            for ch in baseline
            if baseline[ch] > 0 and yields.get(ch, 0) < baseline[ch]
        )
        result = _reallocate_channel_widths(baseline, yields, max_multiplier=1000.0)
        added = sum(max(0, result[ch] - baseline[ch]) for ch in baseline)
        assert added <= freed
        # A channel is never given LESS than its own baseline width entry (only ever more, or
        # unchanged) — a starved channel already asked for everything it has; asking for less
        # next time buys nothing (it is not asked again at all).
        for ch in baseline:
            assert result[ch] >= baseline[ch]


def test_freed_width_splits_by_baseline_share_between_two_saturated_channels() -> None:
    """STM starved (frees 10); MTM (baseline 20) and LTM (baseline 30) both saturated — the
    freed width splits 2:3 in their favour (LTM's bigger pool already earned more trust)."""
    baseline = {"stm": 10, "mtm": 20, "ltm": 30}
    yields = {"stm": 0, "mtm": 20, "ltm": 30}
    result = _reallocate_channel_widths(baseline, yields, max_multiplier=10.0)
    assert result["stm"] == 10
    # 10 freed, split 20:30 -> mtm gets floor(10*20/50)=4, ltm gets the deterministic remainder 6.
    assert result["mtm"] == 24
    assert result["ltm"] == 36
    assert result["mtm"] + result["ltm"] == baseline["mtm"] + baseline["ltm"] + 10


def test_max_multiplier_caps_a_single_channels_growth() -> None:
    """A tight cap (max_multiplier=1.0, i.e. no growth allowed at all) leaves every channel at
    its baseline even though width was freed — the safety valve, not the redistribution math,
    is what is under test here."""
    baseline = {"stm": 10, "mtm": 20, "ltm": 20}
    yields = {"stm": 10, "mtm": 20, "ltm": 0}
    result = _reallocate_channel_widths(baseline, yields, max_multiplier=1.0)
    assert result == baseline


def test_density_tempers_the_split_toward_the_channel_with_the_stronger_tail() -> None:
    """MTM and LTM are equally-baseline-wide (20 each) and both saturate it — by baseline share
    alone they would split freed width evenly. But LTM's own tail has collapsed (density 0.0,
    e.g. its scores fell to near-zero by the end of its returned pool) while MTM's stayed strong
    (density 1.0) — the freed width (from a starved STM channel) should favour MTM entirely."""
    baseline = {"stm": 10, "mtm": 20, "ltm": 20}
    yields = {"stm": 0, "mtm": 20, "ltm": 20}
    density = {"mtm": 1.0, "ltm": 0.0}
    result = _reallocate_channel_widths(baseline, yields, density=density, max_multiplier=10.0)
    assert result["mtm"] > baseline["mtm"], "the dense-tailed channel gets a real share"
    assert result["ltm"] == baseline["ltm"], "the collapsed-tail channel gets none of the split"


def test_disabled_channel_baseline_zero_is_never_touched() -> None:
    """A channel the caller turned off (baseline 0, e.g. `channels.ltm=False`) is neither
    starved nor saturated — this function must never hand it a nonzero width."""
    baseline = {"stm": 10, "mtm": 20, "ltm": 0}
    yields = {"stm": 3, "mtm": 20, "ltm": 0}
    result = _reallocate_channel_widths(baseline, yields)
    assert result["ltm"] == 0


def test_all_saturated_density_zero_falls_back_to_an_even_split_not_nothing() -> None:
    """If density collapses every saturated channel's weight to zero, the freed width must still
    go SOMEWHERE (an even split) rather than silently vanishing."""
    baseline = {"stm": 10, "mtm": 20, "ltm": 20}
    yields = {"stm": 0, "mtm": 20, "ltm": 20}
    density = {"mtm": 0.0, "ltm": 0.0}
    result = _reallocate_channel_widths(baseline, yields, density=density, max_multiplier=10.0)
    assert result["mtm"] > baseline["mtm"]
    assert result["ltm"] > baseline["ltm"]
    assert result["mtm"] + result["ltm"] == baseline["mtm"] + baseline["ltm"] + 10


# ------------------------------------------------------------------------------------------------
# _tail_density
# ------------------------------------------------------------------------------------------------


def test_tail_density_fewer_than_two_items_defaults_to_one() -> None:
    assert _tail_density([]) == 1.0
    assert _tail_density([_scored(0.9)]) == 1.0


def test_tail_density_nonpositive_top_defaults_to_one() -> None:
    assert _tail_density([_scored(0.0), _scored(0.0)]) == 1.0


def test_tail_density_is_the_tail_to_top_ratio_clamped() -> None:
    assert _tail_density([_scored(1.0), _scored(0.5), _scored(0.25)]) == pytest.approx(0.25)
    # A pathological "tail scores higher than top" input still clamps into [0, 1] rather than
    # returning a ratio > 1 — this function never claims a channel is MORE than fully dense.
    assert _tail_density([_scored(0.5), _scored(2.0)]) == 1.0


# ------------------------------------------------------------------------------------------------
# Wiring — ThreeChannelRecallRanker.rank() with fake stores
# ------------------------------------------------------------------------------------------------


class _RecordingMtm:
    """Returns a fixed pool of hits, best-first, truncated to whatever `limit` each call asks
    for — and records every `limit` it was called with, so a test can assert the SECOND
    (reallocated) fetch actually asked for a wider pool, not just that results changed."""

    def __init__(self, pool_size: int) -> None:
        base = datetime(2026, 9, 26, tzinfo=UTC)
        self._pool = [_item(f"mtm-{i}", at=base) for i in range(pool_size)]
        self.calls: list[int] = []

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: object | None = None,
    ) -> list[Scored[MemoryItem]]:
        self.calls.append(limit)
        return [
            Scored(item=i, score=1.0 - 0.001 * rank, channel=RecallChannel.MTM_DENSE, rank=rank)
            for rank, i in enumerate(self._pool[:limit])
        ]

    async def upsert(self, item: MemoryItem) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    async def invalidate(self, *a: object, **k: object) -> None:  # pragma: no cover
        raise NotImplementedError

    async def remove(self, ns: Namespace, memory_id: str) -> None:  # pragma: no cover
        raise NotImplementedError


class _RecordingEmptyLtm:
    """A real, structurally-complete LTM double that always returns nothing — starved by
    construction — and records every `limit` `graph_recall` is called with."""

    def __init__(self) -> None:
        self.calls: list[int] = []

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
        self.calls.append(limit)
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
        return []


class _EmptyStm:
    """No STM candidates — isolates this test to the MTM/LTM reallocation under test, exactly
    like `_EmptyLtm` isolates the STM/MTM interaction in the sibling ranker test file."""

    async def recent(
        self, ns: Namespace, *, limit: int, caller_identity_set: frozenset[str] | None = None
    ) -> list[Scored[MemoryItem]]:
        return []

    async def demoted(
        self, ns: Namespace, *, limit: int, caller_identity_set: frozenset[str] | None = None
    ) -> list[Scored[MemoryItem]]:
        return []

    async def put(self, item: MemoryItem) -> None:  # pragma: no cover
        raise NotImplementedError

    async def get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:  # pragma: no cover
        return None

    async def evict(self, ns: Namespace, memory_id: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def reinforce(
        self, ns: Namespace, memory_id: str, *, at: datetime, relevance_score: float | None = None
    ) -> MemoryItem | None:  # pragma: no cover
        return None


def _build(
    *, mtm_pool_size: int, settings: RecallSettings
) -> tuple[ThreeChannelRecallRanker, _RecordingMtm, _RecordingEmptyLtm]:
    mtm = _RecordingMtm(mtm_pool_size)
    ltm = _RecordingEmptyLtm()
    ranker = ThreeChannelRecallRanker(
        stm=_EmptyStm(),  # type: ignore[arg-type]
        mtm=mtm,  # type: ignore[arg-type]
        ltm=ltm,  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=settings,
        clock=FrozenClock(datetime(2026, 9, 26, tzinfo=UTC)),
    )
    return ranker, mtm, ltm


@pytest.mark.asyncio
async def test_dynamic_channel_budget_off_by_default_never_refetches() -> None:
    """LTM is starved (empty) and MTM is saturated (a 30-item pool against the default 20-wide
    baseline) — exactly the shape reallocation is FOR — but the shipped default
    (`dynamic_channel_budget=False`) must be byte-identical to before: one fetch per channel,
    no second round trip, no widened width."""
    settings = RecallSettings(stm_scoring="recency", channel_pool_size=20)
    ranker, mtm, ltm = _build(mtm_pool_size=30, settings=settings)

    result = await ranker.rank(
        _NS,
        "q",
        [0.1, 0.2],
        limit=10,
        channels=RecallChannels(stm=False, mtm=True, ltm=True),
        caller_identity_set=frozenset[str](),
    )

    assert mtm.calls == [20], "dark by default: exactly one fetch, at the static baseline width"
    assert ltm.calls == [20]
    assert len(result.items) == 10


@pytest.mark.asyncio
async def test_dynamic_channel_budget_on_refetches_only_the_saturated_channel_wider() -> None:
    """The mechanism itself: LTM (starved, empty) frees its whole baseline; MTM (saturated, a
    30-item real pool against a 20-wide baseline) is the only channel that can use it and is
    refetched ONCE more, at the reallocated (wider) width. LTM is never asked twice — it already
    proved it has nothing more to give."""
    settings = RecallSettings(
        stm_scoring="recency",
        channel_pool_size=20,
        dynamic_channel_budget=True,
        dynamic_channel_budget_max_multiplier=3.0,
    )
    ranker, mtm, ltm = _build(mtm_pool_size=30, settings=settings)

    result = await ranker.rank(
        _NS,
        "q",
        [0.1, 0.2],
        limit=10,
        channels=RecallChannels(stm=False, mtm=True, ltm=True),
        caller_identity_set=frozenset[str](),
    )

    assert mtm.calls == [20, 40], "MTM saturated its baseline, so it is refetched at 20+freed(20)"
    assert ltm.calls == [20], "LTM was starved (0 < 20): never refetched, it had nothing more"
    assert len(result.items) == 10


@pytest.mark.asyncio
async def test_dynamic_channel_budget_bounds_total_actual_candidates_fed_to_fusion() -> None:
    """The literal conservation guarantee: LTM is starved (0 actual candidates, always) and MTM
    has effectively unlimited real depth (a 1000-item pool) — so MTM's refetch fills its ENTIRE
    reallocated width exactly. The total number of candidates actually returned to the ranker
    (0 from LTM + MTM's full reallocated width) lands at EXACTLY `sum(baseline)` here, the tight
    upper bound: LTM contributed nothing, and every unit of its unused baseline width was handed
    to, and fully used by, the one channel that could use it — never more than the static
    baseline total, which is what keeps `gold_in_context` comparable between the static and
    dynamic arms (both this arm and a static-width baseline expose fusion to the SAME total
    candidate count; only its distribution across channels differs)."""
    settings = RecallSettings(
        stm_scoring="recency",
        channel_pool_size=20,
        dynamic_channel_budget=True,
        dynamic_channel_budget_max_multiplier=1000.0,
    )
    ranker, mtm, ltm = _build(mtm_pool_size=1000, settings=settings)

    await ranker.rank(
        _NS,
        "q",
        [0.1, 0.2],
        limit=10,
        channels=RecallChannels(stm=False, mtm=True, ltm=True),
        caller_identity_set=frozenset[str](),
    )

    assert ltm.calls == [20]
    assert mtm.calls[0] == 20
    total_baseline = 20 + 20  # mtm + ltm, the static per-call pool width for each
    last_mtm_fetch = mtm.calls[-1]
    total_actual_candidates = 0 + last_mtm_fetch  # ltm always yields 0 here
    assert total_actual_candidates == total_baseline


@pytest.mark.asyncio
async def test_dynamic_channel_budget_no_refetch_when_nothing_is_saturated() -> None:
    """Both channels starved (MTM pool smaller than its own baseline) — freed width exists but
    there is no saturated channel to receive it, so neither channel is refetched."""
    settings = RecallSettings(
        stm_scoring="recency", channel_pool_size=20, dynamic_channel_budget=True
    )
    ranker, mtm, ltm = _build(mtm_pool_size=5, settings=settings)

    await ranker.rank(
        _NS,
        "q",
        [0.1, 0.2],
        limit=10,
        channels=RecallChannels(stm=False, mtm=True, ltm=True),
        caller_identity_set=frozenset[str](),
    )

    assert mtm.calls == [20], "MTM returned only 5 of its own 20-wide pool: starved, not refetched"
    assert ltm.calls == [20]
