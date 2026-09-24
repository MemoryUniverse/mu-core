"""``RecallService._protect_floor`` must not cut S1b's free-riding neighbours back off.

VERIFY 2026-09-24. ADR 0053's Shape A (``RecallSettings.neighbor_free_ride``) is built on ONE
promise, stated in that setting's own docstring: a neighbour "rides along WITH the anchor that
earned its slot instead of consuming one of its own", so ``RecallResult.items`` "can carry MORE
than ``limit`` entries" and "nothing already ranked in ever loses its slot to make room for one".
``ranker._attach_free_riding_neighbors`` keeps that promise. :func:`_protect_floor`, ONE federation
layer above it, did not: it re-sliced the pool to ``limit`` and deleted every rider.

MEASURED before the fix on real Qdrant/Valkey (``mu-dev-vm``, LoCoMo conv-30, 30 queries,
``neighbor_expand_radius=1 neighbor_free_ride=true``): the ranker emitted **527 ``is_neighbor``
rows, every one of them past ``limit``**, and ``LocalMemory.recall`` returned **300 items with
ZERO of them**. Every `neighbor_items_seen=0` ever recorded for this shape (ADR 0053's amendment,
ARCHITECTURE-DELTAS AD-236, ``docs/tracking/eval-runs/2026-09-24/README.md``) was therefore a
measurement of a mechanism that never reached the surface — the SAME class of mistake AD-233
already recorded once, for the ``is_neighbor`` field itself.

These are pure unit tests over the module-level helper: no stores, no network.
"""

from __future__ import annotations

import pytest

from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.services.recall.dto import RecallItemView
from mu_engine.services.recall.service import _protect_floor
from mu_engine.storage.domain.memory import MemoryTier

pytestmark = pytest.mark.unit

_NS = Namespace(org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE)


def _view(
    mid: str, score: float, *, is_floor: bool = False, is_neighbor: bool = False
) -> RecallItemView:
    return RecallItemView(
        memory_id=mid,
        content=f"content {mid}",
        content_hash=f"h-{mid}",
        tier=MemoryTier.MTM,
        channel="mtm",
        namespace=_NS,
        fused_score=score,
        is_floor=is_floor,
        is_neighbor=is_neighbor,
    )


def test_a_free_riding_neighbour_survives_the_federation_slice() -> None:
    """THE FIX. Revert `_protect_floor` to its `head = items[:limit]` form and this goes red:
    the two riders are cut and the result is exactly `limit` items with no `is_neighbor` row."""
    ranked = [_view(f"m{i}", 1.0 - i / 100) for i in range(3)]
    riders = [_view("n1", 0.001, is_neighbor=True), _view("n2", 0.0009, is_neighbor=True)]

    out = _protect_floor([*ranked, *riders], limit=3)
    ids = [v.memory_id for v in out]

    assert ids == ["m0", "m1", "m2", "n1", "n2"], (
        "a free-riding neighbour was cut by the federation slice — Shape A's whole premise is "
        f"that it rides PAST `limit` without displacing anything: {ids!r}"
    )
    assert sum(1 for v in out if v.is_neighbor) == 2


def test_a_rider_never_costs_a_real_candidate_its_slot() -> None:
    """The other half of the same promise: riders must not consume any of the `limit` slots the
    ranked candidates compete for. With 5 ranked candidates and limit=3, the top 3 are kept —
    exactly as they would be with no rider present — and the riders are appended after."""
    ranked = [_view(f"m{i}", 1.0 - i / 100) for i in range(5)]
    rider = _view("n1", 0.001, is_neighbor=True)

    with_rider = [v.memory_id for v in _protect_floor([*ranked, rider], limit=3)]
    without_rider = [v.memory_id for v in _protect_floor(list(ranked), limit=3)]

    assert without_rider == ["m0", "m1", "m2"]
    assert with_rider == [
        "m0",
        "m1",
        "m2",
        "n1",
    ], f"the rider displaced a real candidate instead of riding past the limit: {with_rider!r}"


def test_the_floor_rescue_still_works_when_riders_are_present() -> None:
    """D3's own rescue (a protected member fusion ranked OUTSIDE the window is re-appended at the
    END, never the front) must be computed over the RANKED pool only — counting riders into it
    would let a rider push a protected member out of the window it is guaranteed."""
    ranked = [
        _view("m0", 0.9),
        _view("m1", 0.8),
        _view("f1", 0.1, is_floor=True),  # ranked outside limit=2 -> must be rescued
    ]
    rider = _view("n1", 0.001, is_neighbor=True)

    out = [v.memory_id for v in _protect_floor([*ranked, rider], limit=2)]

    assert "f1" in out, f"the protected floor member was evicted: {out!r}"
    assert out == ["m0", "f1", "n1"], f"unexpected rescue/rider layout: {out!r}"


def test_it_is_byte_identical_at_the_shipped_default_where_nothing_is_a_neighbour() -> None:
    """INERT BY DEFAULT: `neighbor_free_ride=false` ships, so no item ever carries
    `is_neighbor=True` and this function must behave exactly as it did before the fix."""
    items = [_view("m0", 0.9), _view("m1", 0.8), _view("f1", 0.1, is_floor=True)]

    out = [v.memory_id for v in _protect_floor(items, limit=2)]

    assert out == ["m0", "f1"]
