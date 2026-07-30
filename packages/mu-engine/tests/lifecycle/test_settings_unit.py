"""Unit tests for ``mu_engine.lifecycle.settings`` (S0-07).

Pure logic — no containers/SLM (Stage 0): asserts every default from
``memory-lifecycle-manager-spec.md`` §16 constructs verbatim, the pre-TTL-scan/pre-TTL-window
arithmetic invariant (§7b) holds for the shipped defaults, and the frozen/extra-forbid
validation contract (DEV-STANDARDS rule 3 central-config seam) is enforced.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from mu_engine.lifecycle.settings import (
    HostedMirrorConsent,
    LifecycleSettings,
    ManagerModeSettings,
    MtmWorkingGraphSettings,
    OwnershipSettings,
    RetentionSettings,
    SalienceSettings,
)


def test_lifecycle_settings_constructs_with_spec_defaults() -> None:
    """LifecycleSettings() constructs with every §16 default, verbatim."""
    settings = LifecycleSettings()

    assert settings.enabled is True
    assert settings.maintenance_interval_s == 86_400
    assert settings.full_scan_interval_s == 86_400
    assert settings.batch_size == 20
    assert settings.session_idle_s == 900
    assert settings.max_users_per_sweep == 500
    assert settings.max_items_per_user_sweep == 2_000

    # GAPSWEEP BLOCKER 3 fix — named bound, not an unnamed "bounded delta".
    assert settings.capture_ack_p99_delta_budget_ms == 50

    # MAJOR 4 fix — decoupled pre-TTL rescue cadence.
    assert settings.pre_ttl_scan_interval_s == 120

    assert settings.promote_stm_mtm == pytest.approx(0.7)
    assert settings.promote_mtm_ltm == pytest.approx(0.9)
    assert settings.promote_min_age_h == pytest.approx(24.0)
    assert settings.pre_ttl_window_s == 300

    assert settings.demote_mtm == pytest.approx(0.3)
    assert settings.demotion_enabled is True
    assert settings.ltm_demotion_enabled is False

    assert settings.quarantine_ttl_d == 7

    assert settings.use_llm_adjudicator is True
    assert settings.adjudication_budget_per_sweep == 50
    assert settings.adjudication_degrade_threshold_s == pytest.approx(30.0)

    assert settings.config_version == "v1"
    assert settings.policy_version == "v1"

    assert isinstance(settings.salience, SalienceSettings)
    assert isinstance(settings.retention, RetentionSettings)
    assert isinstance(settings.manager_mode, ManagerModeSettings)
    assert isinstance(settings.ownership, OwnershipSettings)
    assert isinstance(settings.mtm_working_graph, MtmWorkingGraphSettings)


def test_salience_settings_defaults() -> None:
    salience = SalienceSettings()
    assert salience.w_recency == pytest.approx(0.5)
    assert salience.w_usage == pytest.approx(0.2)
    assert salience.w_importance == pytest.approx(0.3)
    assert salience.recency_half_life_h == pytest.approx(24.0)
    assert salience.usage_cap == 10
    # Weights sum to 1 (§6 "rel DROPPED off the sweep").
    assert salience.w_recency + salience.w_usage + salience.w_importance == pytest.approx(1.0)


def test_retention_settings_defaults() -> None:
    retention = RetentionSettings()
    assert retention.ephemeral_grace_s == 0
    assert retention.durable_cold_after_d == 180
    assert retention.durable_cold_importance_max == pytest.approx(0.2)
    assert retention.gc_history_window_d == 365
    assert retention.reactivate_on_recall is True
    assert retention.respect_pins is True


def test_manager_mode_settings_defaults() -> None:
    manager_mode = ManagerModeSettings()
    assert manager_mode.default_mode == "managed"
    assert manager_mode.enforce_engine_side is True


def test_hosted_mirror_consent_enum_members() -> None:
    """HostedMirrorConsent StrEnum (spec §4, X4) — exactly the three pinned members."""
    assert set(HostedMirrorConsent) == {
        HostedMirrorConsent.NOT_CONSENTED,
        HostedMirrorConsent.IMPLICIT_VIA_HYBRID,
        HostedMirrorConsent.CONSENTED,
    }
    assert HostedMirrorConsent.NOT_CONSENTED.value == "not_consented"
    assert HostedMirrorConsent.IMPLICIT_VIA_HYBRID.value == "implicit_via_hybrid"
    assert HostedMirrorConsent.CONSENTED.value == "consented"


def test_ownership_settings_defaults_and_consent_gate() -> None:
    ownership = OwnershipSettings()
    assert ownership.lease_ttl_s == 900
    assert ownership.lease_heartbeat_s == 240
    assert ownership.handback_reconcile is True
    # DEFAULT = implicit_via_hybrid (spec §4 X4 consent gate).
    assert ownership.hosted_mirror_consent is HostedMirrorConsent.IMPLICIT_VIA_HYBRID

    # S4: renew well under lease_ttl_s/2 — a live holder never lapses.
    assert ownership.lease_heartbeat_s < ownership.lease_ttl_s / 2


def test_mtm_working_graph_settings_defaults() -> None:
    mtm_working_graph = MtmWorkingGraphSettings()
    assert mtm_working_graph.enabled is False
    assert mtm_working_graph.max_nodes_per_user == 5_000
    assert mtm_working_graph.expansion == "none"
    assert mtm_working_graph.hops == 1
    assert mtm_working_graph.seed_top_n == 5


def test_pre_ttl_scan_interval_bounded_by_half_pre_ttl_window() -> None:
    """§7b S-4 arithmetic invariant: pre_ttl_scan_interval_s <= pre_ttl_window_s / 2.

    Guarantees >=2 scan ticks land inside every item's pre_ttl_window_s-wide rescue window
    regardless of phase (120 <= 150 for the shipped defaults).
    """
    settings = LifecycleSettings()
    assert settings.pre_ttl_scan_interval_s <= settings.pre_ttl_window_s / 2
    assert settings.pre_ttl_scan_interval_s == 120
    assert settings.pre_ttl_window_s / 2 == pytest.approx(150.0)


@pytest.mark.parametrize(
    "model_factory",
    [
        lambda: LifecycleSettings(unknown_field="nope"),  # type: ignore[call-arg]
        lambda: SalienceSettings(unknown_field="nope"),  # type: ignore[call-arg]
        lambda: RetentionSettings(unknown_field="nope"),  # type: ignore[call-arg]
        lambda: ManagerModeSettings(unknown_field="nope"),  # type: ignore[call-arg]
        lambda: OwnershipSettings(unknown_field="nope"),  # type: ignore[call-arg]
        lambda: MtmWorkingGraphSettings(unknown_field="nope"),  # type: ignore[call-arg]
    ],
)
def test_extra_fields_forbidden(model_factory: Callable[[], object]) -> None:
    """extra='forbid' on every sub-tree — no silent-drift config surface."""
    with pytest.raises(ValidationError):
        model_factory()


def test_settings_are_frozen() -> None:
    """frozen=True — central-config is immutable once constructed (no mutation drift)."""
    settings = LifecycleSettings()
    with pytest.raises(ValidationError):
        settings.enabled = False

    salience = SalienceSettings()
    with pytest.raises(ValidationError):
        salience.w_recency = 0.9
