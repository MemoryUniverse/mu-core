"""DegradationPolicy — named degrade + two-consumer fan-out (spec §12/§14, §15.8)."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from mu_contracts.domain.errors import (
    ProviderError,
    SchemaDriftError,
    StoreUnavailableError,
    WorkflowUnavailableError,
)
from mu_contracts.domain.events import DegradedModeEntered, DegradeReason
from mu_engine.platform.degradation import (
    DegradationPolicy,
    assert_sync_status_wired,
)

pytestmark = pytest.mark.unit


class _CapturingMetrics:
    def __init__(self) -> None:
        self.incs: list[tuple[str, Mapping[str, str] | None]] = []

    def inc(self, name: str, *, labels: Mapping[str, str] | None = None, value: int = 1) -> None:
        del value
        self.incs.append((name, labels))

    def observe(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        del name, value, labels

    def gauge(self, name: str, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        del name, value, labels


class _CapturingSyncStatus:
    def __init__(self) -> None:
        self.applied: list[DegradedModeEntered] = []

    def apply(self, event: DegradedModeEntered) -> None:
        self.applied.append(event)


def test_decide_maps_ltm_to_recall_mtm_only() -> None:
    d = DegradationPolicy().decide("recall", "ltm", StoreUnavailableError())
    assert d is not None
    assert d.mode == "recall_mtm_only"
    assert d.reason is DegradeReason.LTM_UNAVAILABLE


def test_decide_maps_temporal_and_schema_drift() -> None:
    pol = DegradationPolicy()
    share = pol.decide("share", "temporal", WorkflowUnavailableError())
    assert share is not None and share.reason is DegradeReason.TEMPORAL_UNAVAILABLE
    cap = pol.decide("capture", "capture", SchemaDriftError())
    assert cap is not None and cap.mode == "halt_source"


def test_decide_returns_none_for_loud_deny_paths() -> None:
    pol = DegradationPolicy()
    # recall + mtm store down, and answer + provider error -> no degrade path -> re-raise loud.
    assert pol.decide("recall", "mtm", StoreUnavailableError()) is None
    assert pol.decide("answer", "llm", ProviderError()) is None


def test_fan_out_operator_only_for_non_sync_class() -> None:
    pol = DegradationPolicy()
    metrics = _CapturingMetrics()
    event = DegradedModeEntered(
        component="temporal", mode="inline_dispatch", reason=DegradeReason.TEMPORAL_UNAVAILABLE
    )
    pol.fan_out(event, metrics=metrics, sync_status=None)  # no sync sink needed
    assert metrics.incs == [("mu_degraded_mode_total", {"component": "temporal"})]


def test_fan_out_sync_class_reaches_both_consumers() -> None:
    pol = DegradationPolicy()
    metrics = _CapturingMetrics()
    sync = _CapturingSyncStatus()
    event = DegradedModeEntered(
        component="device_sync", mode="reconcile_stalled", reason=DegradeReason.SYNC_STALLED
    )
    pol.fan_out(event, metrics=metrics, sync_status=sync)
    assert len(metrics.incs) == 1  # operator always
    assert sync.applied == [event]  # user (SYNC-CLASS)


def test_fan_out_sync_class_without_sink_is_contract_violation() -> None:
    pol = DegradationPolicy()
    event = DegradedModeEntered(
        component="device_sync", mode="reconcile_stalled", reason=DegradeReason.SYNC_STALLED
    )
    with pytest.raises(RuntimeError, match="contract violation"):
        pol.fan_out(event, metrics=_CapturingMetrics(), sync_status=None)


def test_server_unreachable_user_visible_only_for_device_sync() -> None:
    non_sync = DegradedModeEntered(
        component="recall", mode="m", reason=DegradeReason.SERVER_UNREACHABLE
    )
    device = DegradedModeEntered(
        component="device_sync", mode="m", reason=DegradeReason.SERVER_UNREACHABLE
    )
    assert non_sync.user_visible() is False
    assert device.user_visible() is True


def test_assert_sync_status_wired_raises_when_unwired() -> None:
    with pytest.raises(RuntimeError, match="SyncStatusProjector unwired"):
        assert_sync_status_wired(None)
    assert_sync_status_wired(_CapturingSyncStatus())  # wired -> no raise
