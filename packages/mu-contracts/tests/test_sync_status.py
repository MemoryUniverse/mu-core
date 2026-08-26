"""``sync_status.py`` — the user-visible sync assurance DTOs (CANONICAL §7.15).

Phase-3 step 10 unblock: ``SyncStatusProjector`` (mu-server) is blocked on these contracts
existing in mu-core. This test proves the shape, not the (not-yet-built) projector.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.domain.events import DegradeReason
from mu_contracts.domain.model.sync_status import DeviceSyncStatus, SyncState, SyncStatusView

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _device(**overrides: object) -> DeviceSyncStatus:
    fields: dict[str, object] = {
        "device_id": "dev_abc123",
        "label": "laptop",
        "last_synced_seq": 10,
        "head_seq": 12,
        "behind_by": 2,
        "last_synced_at": _NOW,
        "state": SyncState.BEHIND,
        "last_error_kind": None,
    }
    fields.update(overrides)
    return DeviceSyncStatus(**fields)  # type: ignore[arg-type]


def test_sync_state_is_the_closed_six_member_set() -> None:
    # Pins the six literal values CANONICAL §7.15 names — a renamed or added member here is a
    # wire-breaking change to a shared read DTO, so the test fails loudly on either.
    assert {m.value for m in SyncState} == {
        "in_sync",
        "behind",
        "stalled",
        "failed",
        "offline",
        "reseeding",
    }


def test_device_sync_status_defaults_and_roundtrip() -> None:
    d = _device()
    assert d.undelivered_count == 0
    assert d.state is SyncState.BEHIND
    assert d.behind_by == 2


def test_device_sync_status_is_frozen() -> None:
    d = _device()
    with pytest.raises(ValidationError):
        d.behind_by = 99


def test_device_sync_status_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        _device(bogus_field="nope")


def test_device_sync_status_behind_by_has_no_invented_bound() -> None:
    # Fidelity fix: CANONICAL-CONTRACTS.md:725-746 types `behind_by` as a bare `int` with no
    # `ge=0` constraint. A prior version of this DTO added `Field(ge=0)`, which was validation
    # invented beyond the published contract (a consumer validating differently from CANONICAL
    # is a wire incompatibility) — removed to match CANONICAL exactly. This test pins that a
    # negative value is now accepted at the DTO layer, same as CANONICAL's own shape.
    d = _device(behind_by=-1)
    assert d.behind_by == -1


def test_device_sync_status_carries_named_degrade_reason() -> None:
    d = _device(state=SyncState.STALLED, last_error_kind=DegradeReason.SYNC_STALLED)
    assert d.last_error_kind is DegradeReason.SYNC_STALLED


def test_device_sync_status_last_synced_at_is_required() -> None:
    # Fidelity fix: CANONICAL-CONTRACTS.md:725-746 types `last_synced_at: datetime | None` with
    # no default — REQUIRED (the value may be None, but the field must be supplied). A prior
    # version defaulted it to `None`, silently making a required field optional.
    fields = {
        "device_id": "dev_abc123",
        "label": "laptop",
        "last_synced_seq": 10,
        "head_seq": 12,
        "behind_by": 2,
        "state": SyncState.BEHIND,
        "last_error_kind": None,
    }
    with pytest.raises(ValidationError):
        DeviceSyncStatus(**fields)  # type: ignore[arg-type]
    # ... but supplying it explicitly as None is fine — the field is required, not non-nullable.
    assert _device(last_synced_at=None).last_synced_at is None


def test_device_sync_status_last_error_kind_is_required() -> None:
    # Same fidelity fix as above, for `last_error_kind: DegradeReason | None` (no default).
    fields = {
        "device_id": "dev_abc123",
        "label": "laptop",
        "last_synced_seq": 10,
        "head_seq": 12,
        "behind_by": 2,
        "last_synced_at": _NOW,
        "state": SyncState.BEHIND,
    }
    with pytest.raises(ValidationError):
        DeviceSyncStatus(**fields)  # type: ignore[arg-type]


def test_sync_status_view_defaults_pending_counts_to_zero() -> None:
    # CANONICAL-CONTRACTS.md:732 (C10 merge) — both design docs (S:359-365, D:436-443) omit
    # these two fields; CANONICAL adds them. This is the fixture proving they exist AND default
    # to 0 (the soft-fail value), which a design-doc-only reading of S/D would miss entirely.
    view = SyncStatusView(
        principal_id="p1",
        head_seq=12,
        devices=(_device(),),
        fleet_state=SyncState.BEHIND,
        generated_at=_NOW,
    )
    assert view.pending_conflicts == 0
    assert view.pending_notifications == 0
    assert view.devices[0].device_id == "dev_abc123"


def test_sync_status_view_carries_explicit_pending_counts() -> None:
    view = SyncStatusView(
        principal_id="p1",
        head_seq=12,
        devices=(),
        fleet_state=SyncState.IN_SYNC,
        generated_at=_NOW,
        pending_conflicts=3,
        pending_notifications=1,
    )
    assert view.pending_conflicts == 3
    assert view.pending_notifications == 1


def test_sync_status_view_devices_is_a_tuple_not_a_list() -> None:
    # Recorded deviation from CANONICAL's literal `list[DeviceSyncStatus]` (see module
    # docstring): every `frozen=True` model directly under `domain/model/` uses `tuple` for a
    # collection field instead, so `frozen=True` actually blocks mutation of the collection, not
    # just reassignment of the attribute. NOT a package-wide invariant — `mu_contracts.contracts`
    # and `mu_contracts.ports` both have `frozen=True` models using `list`; reported as a delta,
    # not resolved here.
    view = SyncStatusView(
        principal_id="p1",
        head_seq=0,
        devices=(),
        fleet_state=SyncState.IN_SYNC,
        generated_at=_NOW,
    )
    assert isinstance(view.devices, tuple)


def test_sync_status_view_devices_is_required() -> None:
    # Fidelity fix: CANONICAL-CONTRACTS.md:725-746 types `devices: list[DeviceSyncStatus]` with
    # no default — REQUIRED. A prior version defaulted it to `()`, silently making a required
    # field optional.
    with pytest.raises(ValidationError):
        SyncStatusView(  # type: ignore[call-arg]
            principal_id="p1",
            head_seq=0,
            fleet_state=SyncState.IN_SYNC,
            generated_at=_NOW,
        )


def test_sync_status_view_is_frozen_and_forbids_extra() -> None:
    view = SyncStatusView(
        principal_id="p1",
        head_seq=0,
        devices=(),
        fleet_state=SyncState.IN_SYNC,
        generated_at=_NOW,
    )
    with pytest.raises(ValidationError):
        view.head_seq = 1
    with pytest.raises(ValidationError):
        SyncStatusView(
            principal_id="p1",
            head_seq=0,
            devices=(),
            fleet_state=SyncState.IN_SYNC,
            generated_at=_NOW,
            extra_field="nope",  # type: ignore[call-arg]
        )
