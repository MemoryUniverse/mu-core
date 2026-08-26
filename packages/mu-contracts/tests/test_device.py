"""``DeviceRecord`` — the single device DTO (sync-devices §A.2).

Narrowly scoped to the phase-3 addition: `last_synced_at`, the data source
`DeviceSyncStatus.last_synced_at` (mu_contracts.domain.model.sync_status) has no honest field to
project from without it (phase-3 spec §7).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.model.device import (
    ClientMode,
    DevicePlatform,
    DeviceRecord,
    DeviceState,
    PrivacyTier,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _record(**overrides: object) -> DeviceRecord:
    fields: dict[str, object] = {
        "device_id": "dev_abc123",
        "org_id": "org1",
        "workspace_id": "ws1",
        "principal_id": "p1",
        "public_key": "pk",
        "platform": DevicePlatform.MACOS,
        "client_mode": ClientMode.FULL_LOCAL,
        "privacy_tier": PrivacyTier.SERVER_READABLE,
        "state": DeviceState.ACTIVE,
        "enrolled_at": _NOW,
    }
    fields.update(overrides)
    return DeviceRecord(**fields)  # type: ignore[arg-type]


def test_last_synced_at_defaults_to_none() -> None:
    record = _record()
    assert record.last_synced_at is None


def test_last_synced_at_accepts_a_hub_advanced_timestamp() -> None:
    record = _record(last_synced_at=_NOW)
    assert record.last_synced_at == _NOW
