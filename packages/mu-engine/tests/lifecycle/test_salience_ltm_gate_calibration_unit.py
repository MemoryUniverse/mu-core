"""``score_for_ltm_gate`` vs ``LifecycleSettings.promote_mtm_ltm`` — the ADR 0058 recalibration.

ADR 0058's verify pass found the periodic MTM->LTM gate arithmetically reachable (ADR 0054's F2
fix) but EMPIRICALLY DEAD: at the shipped ``promote_mtm_ltm=0.9``, the lowest importance ever
admitted by an exhaustive grid was 0.84, and only at ``access_count>=10`` — strictly above every
importance the product's own capture path writes (ingest default 0.5, ``thinking_finding_
importance`` 0.55, ``thinking_decision_importance`` 0.70; ``mu-client/src/mu_client/config.py``).
This test is the mutation check for the fix: it re-runs the SAME exhaustive grid against the
SHIPPED, UNMODIFIED ``SalienceSettings``/``LifecycleSettings`` defaults (no half-life/usage_cap
override — the exact thing FAULT-HUNT-0924 F2 named as the trap in the pre-fix test) and asserts
the real capture-path distribution is now reachable.

Pure function, no stores, no clock dependency (``score_for_ltm_gate`` drops recency — the whole
point of ADR 0054's fix), deterministic, milliseconds.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings, SalienceSettings
from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_NS = Namespace(org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE)
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

# The real capture-path distribution (cited verbatim in RetentionSettings.promote_mtm_ltm's own
# docstring): ingest default, mu-client's two thinking-turn stamps.
_INGEST_DEFAULT_IMPORTANCE = 0.5  # mu_contracts/domain/model/memory.py:277
_THINKING_FINDING_IMPORTANCE = 0.55  # mu_client/src/mu_client/config.py:185
_THINKING_DECISION_IMPORTANCE = 0.70  # mu_client/src/mu_client/config.py:184


def _item(*, importance: float, access_count: int) -> MemoryItem:
    return MemoryItem(
        content="a captured fact",
        kind=MemoryKind.PROPOSITION,
        namespace=_NS,
        owner_id="u",
        workspace_id="w",
        session_id="s",
        created_at=_EPOCH,
        importance_score=importance,
        access_count=access_count,
    )


def _min_admitted_importance(*, promote_mtm_ltm: float) -> float | None:
    """Exhaustive grid over the SHIPPED, UNMODIFIED SalienceSettings — importance 0..1 in 0.01
    steps x access_count 0..60 — mirroring FAULT-HUNT-0924 probe_salience.py's own method.
    Returns the lowest importance admitted at ANY access_count, or None if nothing is."""
    strategy = SalienceStrategy(SalienceSettings())  # shipped defaults, no override
    clock = FrozenClock(_EPOCH)  # score_for_ltm_gate drops recency — the clock is inert here
    best: float | None = None
    imp = 0
    while imp <= 100:
        importance = imp / 100
        for access_count in range(0, 61):
            item = _item(importance=importance, access_count=access_count)
            score = strategy.score_for_ltm_gate(item, clock=clock)
            if score >= promote_mtm_ltm:
                if best is None or importance < best:
                    best = importance
                break
        imp += 1
    return best


def _access_count_needed(*, importance: float, promote_mtm_ltm: float) -> int | None:
    strategy = SalienceStrategy(SalienceSettings())
    clock = FrozenClock(_EPOCH)
    for access_count in range(0, 61):
        item = _item(importance=importance, access_count=access_count)
        if strategy.score_for_ltm_gate(item, clock=clock) >= promote_mtm_ltm:
            return access_count
    return None


def test_shipped_default_is_no_longer_0_9() -> None:
    """The literal regression guard: a future edit that reverts the recalibration goes red here
    before it ever reaches the grid assertions below."""
    assert LifecycleSettings().promote_mtm_ltm == pytest.approx(0.6)


@pytest.mark.parametrize(
    "importance",
    [
        _INGEST_DEFAULT_IMPORTANCE,
        _THINKING_FINDING_IMPORTANCE,
        _THINKING_DECISION_IMPORTANCE,
    ],
)
def test_every_real_capture_path_importance_is_reachable_at_some_access_count(
    importance: float,
) -> None:
    """MUTATION CHECK: reverting ``LifecycleSettings.promote_mtm_ltm`` to the pre-fix 0.9 makes
    this go red for ALL THREE parametrizations at once — none of these three importances ever
    clears 0.9 at any access_count (ADR 0058's own exhaustive-grid finding, reproduced here)."""
    threshold = LifecycleSettings().promote_mtm_ltm
    needed = _access_count_needed(importance=importance, promote_mtm_ltm=threshold)
    assert needed is not None, (
        f"importance={importance} is UNREACHABLE at any access_count<=60 against the shipped "
        f"promote_mtm_ltm={threshold} — the gate is an off switch for this real capture-path "
        f"stamp, exactly the defect ADR 0058 named"
    )
    # Reachable through genuine re-engagement (well under usage_cap=10), never a single recall.
    assert 1 <= needed <= 9, (
        f"importance={importance} needed access_count={needed} to clear the gate — expected "
        f"single-digit re-use, not a one-time recall (0) or an unreachable/near-cap value"
    )


def test_pinned_max_importance_is_admitted_immediately() -> None:
    """An explicitly max-importance (1.0) fact — a user-pinned or model-flagged "critical" stamp
    — should not have to wait on usage at all: this is the gate's own designed ceiling case."""
    threshold = LifecycleSettings().promote_mtm_ltm
    needed = _access_count_needed(importance=1.0, promote_mtm_ltm=threshold)
    assert needed == 0


def test_pre_fix_0_9_is_still_dead_for_the_whole_capture_distribution() -> None:
    """Documents, and locks in, WHY the recalibration was necessary — the historical 0.9 value
    (not the shipped one) genuinely admits nothing the product writes, reproducing ADR 0058's
    "lowest importance ever admitted is 0.84" finding exactly."""
    lowest_admitted = _min_admitted_importance(promote_mtm_ltm=0.9)
    assert lowest_admitted == pytest.approx(0.84)
    for importance in (
        _INGEST_DEFAULT_IMPORTANCE,
        _THINKING_FINDING_IMPORTANCE,
        _THINKING_DECISION_IMPORTANCE,
    ):
        assert _access_count_needed(importance=importance, promote_mtm_ltm=0.9) is None


def test_recalibrated_threshold_still_requires_real_usage_below_the_ingest_promote_floor() -> None:
    """The recalibration is not a removal of the gate: an importance BELOW the ingest promote
    floor (0.6, ``IngestSettings.importance_promote``) must still need MORE than a token amount of
    re-use — the gate stays selective, it just stops being unreachable."""
    threshold = LifecycleSettings().promote_mtm_ltm
    needed = _access_count_needed(importance=0.3, promote_mtm_ltm=threshold)
    # Below-floor importance either never clears the gate on usage alone, or needs heavy re-use —
    # never a cheap single recall.
    assert needed is None or needed >= 5
