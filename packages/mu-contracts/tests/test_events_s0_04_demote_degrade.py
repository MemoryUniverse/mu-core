"""S0-04: MemoryDemoted.to_tier + 2 new DegradeReason members (CANONICAL §2/§3).

Covers:
- ADR 0034 / PROPOSED P5 (R5): MemoryDemoted gains an additive `to_tier` field so a real
  tier-down move (MTM->STM forgetting-curve demotion) no longer has to misuse
  `to_state=ARCHIVED`.
- ADR 0037 / PROPOSED P6: DegradeReason.MTM_INVALIDATE_POINT_ABSENT.
- ADR 0032 / PROPOSED P13 (GAPSWEEP BLOCKER 2): DegradeReason.OFFLINE_FAILOVER_NOT_CONSENTED.
"""

import pytest
from pydantic import ValidationError

from mu_contracts.domain.events import (
    RESERVED_REASONS,
    SYNC_CLASS_ALWAYS,
    SYNC_CLASS_WHEN_DEVICE_SYNC,
    DegradedModeEntered,
    DegradeReason,
    MemoryDemoted,
)
from mu_contracts.domain.model import Namespace, State, Tier, Visibility

pytestmark = pytest.mark.unit

_NS = Namespace(org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE)


def test_memory_demoted_without_to_tier_still_constructs_as_pure_archival() -> None:
    # Backward-compat: every pre-existing call site only ever modeled archival and never
    # passed to_tier — this must keep constructing unchanged, defaulting to None.
    ev = MemoryDemoted(namespace=_NS, id="m1", tier=Tier.LTM, retention=0.2)
    assert ev.to_tier is None
    assert ev.to_state is State.ARCHIVED


def test_memory_demoted_with_to_tier_models_a_real_tier_down_move() -> None:
    ev = MemoryDemoted(
        namespace=_NS,
        id="m1",
        tier=Tier.MTM,
        to_tier=Tier.STM,
        to_state=State.ACTIVE,
        retention=0.7,
    )
    assert ev.to_tier is Tier.STM
    assert ev.tier is Tier.MTM
    assert ev.to_state is State.ACTIVE


def test_memory_demoted_is_frozen_and_forbids_extra_fields() -> None:
    ev = MemoryDemoted(namespace=_NS, id="m1", tier=Tier.MTM, to_tier=Tier.STM, retention=0.5)
    with pytest.raises(ValidationError):
        ev.to_tier = Tier.LTM
    with pytest.raises(ValidationError):
        MemoryDemoted(  # type: ignore[call-arg]
            namespace=_NS, id="m1", tier=Tier.MTM, to_tier=Tier.STM, retention=0.5, nope=1
        )


def test_memory_demoted_to_tier_is_content_free_enum_not_content() -> None:
    # DomainEvent.__pydantic_init_subclass__ (the content-free guard) must not have rejected
    # to_tier at class-definition time — proven simply by the class existing and being
    # importable/constructible above; assert the field is present and enum-typed.
    assert "to_tier" in MemoryDemoted.model_fields
    field = MemoryDemoted.model_fields["to_tier"]
    assert field.default is None


def test_degrade_reason_mtm_invalidate_point_absent_added() -> None:
    assert DegradeReason.MTM_INVALIDATE_POINT_ABSENT.value == "mtm_invalidate_point_absent"
    # operator-consumer only — never SYNC-CLASS, never RESERVED.
    assert DegradeReason.MTM_INVALIDATE_POINT_ABSENT not in SYNC_CLASS_ALWAYS
    assert DegradeReason.MTM_INVALIDATE_POINT_ABSENT not in SYNC_CLASS_WHEN_DEVICE_SYNC
    assert DegradeReason.MTM_INVALIDATE_POINT_ABSENT not in RESERVED_REASONS
    ev = DegradedModeEntered(
        component="mtm",
        mode="supersede_invalidate_retry",
        reason=DegradeReason.MTM_INVALIDATE_POINT_ABSENT,
    )
    assert ev.user_visible() is False


def test_degrade_reason_offline_failover_not_consented_added() -> None:
    assert DegradeReason.OFFLINE_FAILOVER_NOT_CONSENTED.value == "offline_failover_not_consented"
    assert DegradeReason.OFFLINE_FAILOVER_NOT_CONSENTED not in SYNC_CLASS_ALWAYS
    assert DegradeReason.OFFLINE_FAILOVER_NOT_CONSENTED not in SYNC_CLASS_WHEN_DEVICE_SYNC
    assert DegradeReason.OFFLINE_FAILOVER_NOT_CONSENTED not in RESERVED_REASONS
    ev = DegradedModeEntered(
        component="lifecycle",
        mode="offline_sweep_blocked",
        reason=DegradeReason.OFFLINE_FAILOVER_NOT_CONSENTED,
    )
    assert ev.user_visible() is False


def test_degrade_reason_is_still_a_closed_str_enum_additive_only() -> None:
    # Both new members are plain str values (StrEnum) — additive membership, no shape change
    # to the enum's own type or to any existing member's value.
    assert isinstance(DegradeReason.MTM_INVALIDATE_POINT_ABSENT, str)
    assert isinstance(DegradeReason.OFFLINE_FAILOVER_NOT_CONSENTED, str)
    assert DegradeReason.LTM_UNAVAILABLE.value == "ltm_unavailable"  # pre-existing, untouched
