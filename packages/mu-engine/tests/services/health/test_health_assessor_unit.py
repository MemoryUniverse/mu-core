"""``heuristic_v1`` — the flag matrix and the Ebbinghaus PORT fidelity (memory-health §4, §10).

Pure: the assessor takes ``now`` as an argument and a pre-loaded ``ConflictEdges`` snapshot, so
there is no clock, no store and no model anywhere in this file.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timedelta

import pytest

from mu_contracts.domain.model.conflict import ConflictEdgeRow, ConflictEdges, ConflictState
from mu_contracts.domain.model.health import MemoryHealthFlag
from mu_contracts.domain.model.memory import MemoryItem, State, Tier
from mu_engine.services.health.assessor import HEURISTIC_V1, HeuristicV1Assessor, health_registry
from mu_engine.services.health.settings import HealthSettings

from .conftest import T0, salience

pytestmark = pytest.mark.unit

_NO_CONFLICTS = ConflictEdges()


@pytest.fixture
def settings() -> HealthSettings:
    return HealthSettings()


@pytest.fixture
def assessor(settings: HealthSettings) -> HeuristicV1Assessor:
    return HeuristicV1Assessor(settings)


def _edges(
    memory_id: str,
    *,
    state: ConflictState = ConflictState.DETECTED,
    pin_blocked: bool = False,
    confidence: float | None = None,
    peers: frozenset[str] = frozenset({"other"}),
) -> ConflictEdges:
    return ConflictEdges(
        rows_by_memory={
            memory_id: ConflictEdgeRow(
                memory_id=memory_id,
                peer_ids=peers,
                conflict_id="c1",
                state=state,
                pin_blocked=pin_blocked,
                detected_confidence=confidence,
            )
        }
    )


# ══════════════════════════════════════════════════════ the Ebbinghaus curve (§4, §10 item 10) ══
def test_retention_is_the_docstring_intended_curve_not_the_upstream_precedence_bug() -> None:
    """Golden value pinning DEVIATION 1 of the MemoryBank port.

    ``forget_memory.py:36``'s literal body is ``math.exp(-t / 5*S)``, which Python parses as
    ``exp(-t*S/5)`` — higher strength forgets FASTER, contradicting its own docstring at
    ``:30-34``. We adopt the docstring-intended ``exp(-Δt / S)``. At Δt = S = 5 days those two
    readings are two orders of magnitude apart, so this value cannot be produced by the buggy
    form: it is a real discriminator, not a restatement.
    """
    components = salience(strength=5.0, scored_at=T0)

    retention = components.strength_retention(T0 + timedelta(days=5), min_strength=1.0)

    assert retention == pytest.approx(math.exp(-1.0))
    assert retention != pytest.approx(math.exp(-5.0 * 5.0 / 5.0))  # the upstream literal


def test_higher_strength_decays_more_slowly() -> None:
    """The docstring property the upstream literal inverts (``forget_memory.py:30-34``)."""
    at = T0 + timedelta(days=10)
    weak = salience(strength=2.0, scored_at=T0).strength_retention(at, min_strength=1.0)
    strong = salience(strength=20.0, scored_at=T0).strength_retention(at, min_strength=1.0)
    assert strong > weak


def test_min_strength_floors_the_divisor_and_a_bad_floor_fails_loud() -> None:
    at = T0 + timedelta(days=1)
    assert salience(strength=0.0, scored_at=T0).strength_retention(
        at, min_strength=1.0
    ) == pytest.approx(math.exp(-1.0))
    with pytest.raises(ValueError, match="min_strength"):
        salience(strength=5.0, scored_at=T0).strength_retention(at, min_strength=0.0)


def test_stm_and_unscored_items_claim_no_decay(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    """The lens never invents a risk it cannot evidence."""
    at = T0 + timedelta(days=365)
    assert assessor.retention(make_item(tier=Tier.STM), now=at) == 1.0
    assert assessor.retention(make_item(sal=None), now=at) == 1.0


# ═══════════════════════════════════════════════════════════════════════ the flag matrix (§4) ══
def test_healthy_active_item_has_no_flags(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    flags = assessor.assess(make_item(), now=T0, conflict_edges=_NO_CONFLICTS)
    assert flags == frozenset()


def test_decaying_band_is_exactly_what_the_next_sweep_would_archive(
    assessor: HeuristicV1Assessor,
    settings: HealthSettings,
    make_item: Callable[..., MemoryItem],
) -> None:
    """R in [curve.demote_retention, stale_retention_band) -> DECAYING; below the lower edge the
    item is past warning (the sweep acts) and above the upper edge it is healthy."""
    item = make_item(sal=salience(strength=5.0, scored_at=T0))
    inside = T0 + timedelta(days=5.0 * -math.log(0.45))  # R == 0.45, inside [0.3, 0.6)
    below = T0 + timedelta(days=5.0 * -math.log(0.2))
    above = T0 + timedelta(days=5.0 * -math.log(0.9))

    assert settings.curve.demote_retention == 0.3
    assert settings.stale_retention_band == 0.6
    assert MemoryHealthFlag.DECAYING in assessor.assess(
        item, now=inside, conflict_edges=_NO_CONFLICTS
    )
    assert MemoryHealthFlag.DECAYING not in assessor.assess(
        item, now=below, conflict_edges=_NO_CONFLICTS
    )
    assert MemoryHealthFlag.DECAYING not in assessor.assess(
        item, now=above, conflict_edges=_NO_CONFLICTS
    )


def test_stale_requires_both_age_and_low_recency(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    old = T0 + timedelta(hours=200)  # > stale_after_h (168)
    aged_and_cold = make_item(sal=salience(recency=0.05), last_seen=T0)
    aged_but_warm = make_item(sal=salience(recency=0.9), last_seen=T0)
    fresh_and_cold = make_item(sal=salience(recency=0.05), last_seen=old)

    assert MemoryHealthFlag.STALE in assessor.assess(
        aged_and_cold, now=old, conflict_edges=_NO_CONFLICTS
    )
    assert MemoryHealthFlag.STALE not in assessor.assess(
        aged_but_warm, now=old, conflict_edges=_NO_CONFLICTS
    )
    assert MemoryHealthFlag.STALE not in assessor.assess(
        fresh_and_cold, now=old, conflict_edges=_NO_CONFLICTS
    )


def test_unscored_item_is_never_called_stale(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    """An unknown conjunct must not be read as True."""
    flags = assessor.assess(
        make_item(sal=None, last_seen=T0),
        now=T0 + timedelta(days=90),
        conflict_edges=_NO_CONFLICTS,
    )
    assert MemoryHealthFlag.STALE not in flags


def test_quarantined_and_low_confidence_both_raise_low_confidence(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    quarantined = make_item(state=State.QUARANTINED)
    assert MemoryHealthFlag.LOW_CONFIDENCE in assessor.assess(
        quarantined, now=T0, conflict_edges=_NO_CONFLICTS
    )

    active = make_item()
    assert MemoryHealthFlag.LOW_CONFIDENCE in assessor.assess(
        active, now=T0, conflict_edges=_edges("mem_1", confidence=0.2)
    )
    assert MemoryHealthFlag.LOW_CONFIDENCE not in assessor.assess(
        active, now=T0, conflict_edges=_edges("mem_1", confidence=0.9)
    )


def test_conflicting_fires_on_an_unresolved_edge_or_a_pin_block(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    item = make_item()
    assert MemoryHealthFlag.CONFLICTING in assessor.assess(
        item, now=T0, conflict_edges=_edges("mem_1", state=ConflictState.DETECTED)
    )
    assert MemoryHealthFlag.CONFLICTING in assessor.assess(
        item,
        now=T0,
        conflict_edges=_edges("mem_1", state=ConflictState.RESOLVED, pin_blocked=True),
    )
    assert MemoryHealthFlag.CONFLICTING not in assessor.assess(
        item, now=T0, conflict_edges=_edges("mem_1", state=ConflictState.RESOLVED)
    )


def test_archived_state_flags_archived(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    flags = assessor.assess(make_item(state=State.ARCHIVED), now=T0, conflict_edges=_NO_CONFLICTS)
    assert flags == frozenset({MemoryHealthFlag.ARCHIVED})


def test_pinned_marker_is_added_to_the_exit_flags_never_instead_of_them(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    """Spec §4 line 222: the exit flags are still shown, so the user sees what WOULD happen."""
    decaying_at = T0 + timedelta(days=5.0 * -math.log(0.45))
    pinned = make_item(pinned=True, sal=salience(strength=5.0, scored_at=T0))
    unpinned = make_item(pinned=False, sal=salience(strength=5.0, scored_at=T0))

    pinned_flags = assessor.assess(pinned, now=decaying_at, conflict_edges=_NO_CONFLICTS)
    unpinned_flags = assessor.assess(unpinned, now=decaying_at, conflict_edges=_NO_CONFLICTS)

    assert pinned_flags == unpinned_flags | {MemoryHealthFlag.PINNED}
    assert MemoryHealthFlag.DECAYING in pinned_flags


def test_stm_items_are_never_decaying_or_stale(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    """STM is a TTL window, not a decay curve (spec §2.1 line 80)."""
    stm = make_item(tier=Tier.STM, sal=salience(strength=0.1, recency=0.0), last_seen=T0)
    flags = assessor.assess(stm, now=T0 + timedelta(days=365), conflict_edges=_NO_CONFLICTS)
    assert MemoryHealthFlag.DECAYING not in flags
    assert MemoryHealthFlag.STALE not in flags


# ═══════════════════════════════════════════════════════════════ the registry seam (§3.3) ══
def test_default_key_resolves_and_an_unknown_key_fails_loud() -> None:
    from mu_contracts.config.settings import Settings
    from mu_contracts.domain.errors import UnknownComponentError

    built = health_registry.create(HEURISTIC_V1, Settings())
    assert built.key == HEURISTIC_V1
    with pytest.raises(UnknownComponentError):
        health_registry.create("learned_v99", Settings())


def test_assess_is_a_pure_function_of_its_arguments(
    assessor: HeuristicV1Assessor, make_item: Callable[..., MemoryItem]
) -> None:
    """Called twice with the same inputs it returns the same flags, and the item is unchanged —
    the property ``MemoryHealthService``'s CQRS read-purity rests on."""
    item = make_item()
    before = item.model_dump()
    at: datetime = T0 + timedelta(days=3)

    first = assessor.assess(item, now=at, conflict_edges=_NO_CONFLICTS)
    second = assessor.assess(item, now=at, conflict_edges=_NO_CONFLICTS)

    assert first == second
    assert item.model_dump() == before
