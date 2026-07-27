"""Device registry vocabulary — the ONE DeviceRecord DTO + the (client_mode, privacy_tier) axis.

Authority: sync-devices-gateway-spec.md §A.2 (ported verbatim; ``SyncMode``/``DeviceBinding``/
``DeviceEnrollment``/``public_key_ref`` are all DELETED — one DTO). The canonical model is the
orthogonal ``(client_mode, privacy_tier)`` pair; ``HostingMode`` is a DERIVED routing tag with a
documented bijection, never stored/asserted independently (CANONICAL §1 rule 6, §7.18).

``E2E`` is RESERVED (CANONICAL §7.18) — refused at enroll. ``DeviceState.REVOKED`` rows are
retained (invalidate-don't-delete). ``last_synced_seq``/``last_lamport`` are HUB-advanced only —
a device cannot lie its cursor forward past what the log delivered.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ClientMode",
    "DeviceEnrollRequest",
    "DevicePlatform",
    "DeviceRecord",
    "DeviceState",
    "HostingMode",
    "PrivacyTier",
]


class DevicePlatform(StrEnum):
    MACOS = "macos"
    LINUX = "linux"
    WINDOWS = "windows"
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"
    OTHER = "other"


class ClientMode(StrEnum):
    """Per-device topology axis. HYBRID is fleet-emergent (a full_local primary + thin
    secondaries), never a per-device value."""

    THIN = "thin"  # server space is source of truth; cache-only local
    FULL_LOCAL = "full_local"  # device has its own store; server is the merge relay


class PrivacyTier(StrEnum):
    SERVER_READABLE = "server_readable"  # SHIPPING: server may read plaintext (audited no-look)
    E2E = "e2e"  # RESERVED §7.18 — refused at enroll


class HostingMode(StrEnum):
    """DERIVED routing tag (never stored/asserted independently)."""

    LOCAL = "local"  # (full_local, server_readable) with an on-device store
    HOSTED_SERVER_READABLE = "hosted_server_readable"  # (thin, server_readable)
    HOSTED_E2E = "hosted_e2e"  # RESERVED — (thin|full_local, e2e), unreachable


class DeviceState(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    REVOKED = "revoked"  # invalidate-don't-delete: REVOKED rows retained


class DeviceRecord(BaseModel):
    """The single device DTO (sync-devices §A.2). ``device_id`` is deterministic/idempotent:
    ``"dev_" + sha256(workspace_id | principal_id | public_key)[:24]``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    device_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)  # tenancy tag == Namespace.workspace
    principal_id: str = Field(min_length=1)  # the ONE owning principal
    public_key: str = Field(min_length=1)  # base64 device public key
    platform: DevicePlatform
    label: str = Field(default="", max_length=120)  # user-chosen; never derived from content
    app_build: str = Field(default="", max_length=64)
    client_mode: ClientMode
    privacy_tier: PrivacyTier
    state: DeviceState = DeviceState.PENDING
    enrolled_at: datetime
    last_seen_at: datetime | None = None
    last_synced_seq: int = Field(default=0, ge=0)  # HUB-advanced cursor, never device-asserted
    last_lamport: int = Field(default=0, ge=0)  # §7.17-5: hub's durable VC entry
    key_epoch: int = Field(default=0, ge=0)  # RESERVED §7.18 — no shipping path writes it
    revoked_at: datetime | None = None
    revoked_by: str | None = None

    def is_syncable(self) -> bool:
        return self.state is DeviceState.ACTIVE

    @property
    def hosting_mode(self) -> HostingMode:
        if self.privacy_tier is PrivacyTier.E2E:
            return HostingMode.HOSTED_E2E  # RESERVED, unreachable
        if self.client_mode is ClientMode.FULL_LOCAL:
            return HostingMode.LOCAL
        return HostingMode.HOSTED_SERVER_READABLE


class DeviceEnrollRequest(BaseModel):
    """The enroll payload (sync-devices §A.2). The tier gate (RESERVED-refusal + fleet
    uniformity) is applied by ``DeviceService.enroll`` before any write."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    public_key: str = Field(min_length=1)
    platform: DevicePlatform
    label: str = Field(default="", max_length=120)
    app_build: str = Field(default="", max_length=64)
    client_mode: ClientMode
    privacy_tier: PrivacyTier
