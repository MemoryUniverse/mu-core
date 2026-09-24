"""The ``channel_pool_size``/``limit`` pool-trap fix — ACCURACY-PLAN-0831.md §1.4.

**The bug.** ``ThreeChannelRecallRanker.rank`` used to fetch every channel at a FIXED
``settings.channel_pool_size`` (default 20), completely independent of the caller's ``limit`` — so
``dto.py``'s own documented invariant ("per-channel fetch width > limit", ADR 0010) silently
inverted at ``limit >= 20`` and a wide ``limit`` (e.g. 30 or 60, exactly what the context-budget
width derivation now asks for) sliced a candidate union that was never actually widened.

These tests spy the MTM/LTM channel calls directly (the ``limit=`` kwarg each receives IS the
resolved per-channel pool width) rather than inferring it from result contents, so they pin the
EXACT arithmetic (`max(channel_pool_size, ceil(limit * channel_pool_multiplier))`), not just "some
improvement happened."
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.model.recall import Vector
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.recall.dto import RecallChannels, RecallSettings
from mu_engine.services.recall.fusion import ReciprocalRankFusion
from mu_engine.services.recall.ranker import ThreeChannelRecallRanker
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.domain.recall import Scored

_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)


class _RecordingMtm:
    """Records every ``limit`` it is called with; returns nothing (isolates the POOL WIDTH
    question from fusion/ranking, which the sibling ``test_recall_ranker_unit.py`` already
    covers)."""

    def __init__(self) -> None:
        self.limits_seen: list[int] = []

    async def semantic(
        self,
        ns: Namespace,
        query_vector: list[float],
        *,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
        sparse_query: object | None = None,
    ) -> list[Scored[object]]:
        self.limits_seen.append(limit)
        return []


class _RecordingLtm:
    def __init__(self) -> None:
        self.graph_recall_limits_seen: list[int] = []
        self.traverse_limits_seen: list[int] = []

    async def graph_recall(
        self,
        ns: Namespace,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[object]]:
        self.graph_recall_limits_seen.append(limit)
        return []

    async def traverse_entities(
        self,
        ns: Namespace,
        *,
        query: str,
        max_hops: int,
        limit: int,
        caller_identity_set: frozenset[str] | None = None,
    ) -> list[Scored[object]]:
        self.traverse_limits_seen.append(limit)
        return []


class _EmptyStm:
    async def recent(
        self, ns: Namespace, *, limit: int, caller_identity_set: frozenset[str] | None = None
    ) -> list[Scored[object]]:
        return []

    async def demoted(
        self, ns: Namespace, *, limit: int, caller_identity_set: frozenset[str] | None = None
    ) -> list[Scored[object]]:
        """AD-250 fix (ADR 0061): `ThreeChannelRecallRanker.rank` now calls this unconditionally
        alongside `recent` — this fake needed it to keep working at all."""
        return []


def _build(
    settings: RecallSettings,
) -> tuple[ThreeChannelRecallRanker, _RecordingMtm, _RecordingLtm]:
    mtm = _RecordingMtm()
    ltm = _RecordingLtm()
    ranker = ThreeChannelRecallRanker(
        stm=_EmptyStm(),  # type: ignore[arg-type]
        mtm=mtm,  # type: ignore[arg-type]
        ltm=ltm,  # type: ignore[arg-type]
        fusion=ReciprocalRankFusion(),
        settings=settings,
        clock=FrozenClock(datetime(2026, 9, 1, tzinfo=UTC)),
    )
    return ranker, mtm, ltm


async def _rank(ranker: ThreeChannelRecallRanker, *, limit: int) -> None:
    query_vec: Vector = (0.1, 0.2)
    await ranker.rank(
        _NS,
        "q",
        query_vec,
        limit=limit,
        channels=RecallChannels(),
        caller_identity_set=frozenset[str](),
    )


@pytest.mark.asyncio
async def test_pool_scales_with_limit_past_the_configured_floor() -> None:
    """limit=30, channel_pool_size=20 (default), channel_pool_multiplier=2.0 (default):
    ceil(30 * 2.0) = 60 > the 20 floor, so every channel must be asked for 60, not 20 — pinning
    the EXACT bug this fix closes (the pre-fix ranker would have asked for 20 regardless)."""
    settings = RecallSettings()
    ranker, mtm, ltm = _build(settings)

    await _rank(ranker, limit=30)

    assert mtm.limits_seen == [60]
    assert ltm.graph_recall_limits_seen == [60]


@pytest.mark.asyncio
async def test_pool_never_drops_below_the_configured_floor_at_a_small_limit() -> None:
    """limit=5, channel_pool_size=20: ceil(5*2.0)=10 < the 20 floor, so the floor must still win —
    a narrow explicit limit must not narrow the candidate pool below what was always configured."""
    settings = RecallSettings()
    ranker, mtm, ltm = _build(settings)

    await _rank(ranker, limit=5)

    assert mtm.limits_seen == [20]
    assert ltm.graph_recall_limits_seen == [20]


@pytest.mark.asyncio
async def test_pool_multiplier_is_configurable_and_actually_used() -> None:
    """A non-default multiplier changes the resolved pool — proves the ratio is READ from
    settings, not a value baked into the ranker."""
    settings = RecallSettings(channel_pool_multiplier=3.0, channel_pool_size=1)
    ranker, mtm, ltm = _build(settings)

    await _rank(ranker, limit=10)

    assert mtm.limits_seen == [30]
    assert ltm.graph_recall_limits_seen == [30]


@pytest.mark.asyncio
async def test_pool_matches_the_measured_k_curve_ratio_at_limit_60() -> None:
    """RETRIEVAL-EVAL-0829.md §13.1's own k-curve run hand-configured ``channel_pool_size=120`` at
    ``limit=60`` — exactly a 2.0x ratio. The default ``channel_pool_multiplier=2.0`` must reproduce
    that SAME pool width automatically, with no operator override, at the same limit."""
    settings = RecallSettings()  # channel_pool_multiplier default == 2.0
    ranker, mtm, ltm = _build(settings)

    await _rank(ranker, limit=60)

    assert mtm.limits_seen == [120]
    assert ltm.graph_recall_limits_seen == [120]


@pytest.mark.asyncio
async def test_ltm_traversal_arm_also_receives_the_scaled_pool() -> None:
    """The multi-hop traversal call (``ltm_max_hops`` default 2, so the arm runs) must receive the
    SAME scaled pool as the flat graph_recall seed — not left at the old fixed width."""
    settings = RecallSettings()
    ranker, mtm, ltm = _build(settings)

    await _rank(ranker, limit=30)

    assert ltm.traverse_limits_seen == [60]
