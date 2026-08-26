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
    "RevokeReason",
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
    ``"dev_" + sha256(org_id | principal_id | public_key)[:24]``.

    ⚠ **The hash basis is ``org_id``, not ``workspace_id``** — ``CANONICAL-CONTRACTS.md:647``
    (*"keyed on `org` post-ADR-0026"*) and phase-3 spec D-12. This docstring said
    ``workspace_id`` until the ``org_id`` field below landed, and the two together were the
    "two carriers for one hash input" D-33 closes: the id is computed from ``org_id``, so a
    reader who transcribes the old sentence builds a DIFFERENT id for the same keypair and
    forks a phantom device on every re-enrol.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    device_id: str = Field(min_length=1)
    #: **The tenancy ROOT (ADR 0026), and the first term of the ``device_id`` hash** — added by
    #: phase-3 spec D-12/D-13 so the registry's isolation is structural rather than a filter a
    #: later caller can drop. ``DeviceRow`` (``mu-engine``'s ``schema.py``) has carried this
    #: column since ``46ae4bcc2472``; the DTO did not, so the adapter could not populate it.
    org_id: str = Field(min_length=1)
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
    #: **Phase-3 spec §7 / `DeviceSyncStatus.last_synced_at`'s data source** — added because
    #: `DeviceSyncStatus.last_synced_at` (``sync_status.py``, CANONICAL §7.15) has no honest field
    #: to project from without it: `last_seen_at` is "any contact" (heartbeat/IPC), `last_lamport`
    #: is the hub's VC entry with no timestamp, and neither answers "when did a sync last land."
    #: HUB-advanced only, same discipline as `last_synced_seq`/`last_lamport` — never
    #: device-asserted forward.
    #:
    #: ⚠ **NOT YET PERSISTED — always `None` in production today.** ``DeviceRow`` (``mu-engine``'s
    #: ``schema.py``, same row referenced above) has no `last_synced_at` column (verified: its
    #: field list runs `last_seen_at` → `last_synced_seq` directly, nothing between them) and no
    #: writer sets this field. Adding the column is a migration in the storage layer, outside this
    #: package's ownership (`mu-contracts` is DTOs only) — tracked as an open gap, not fixed here.
    #: Do not read this field as carrying real data until that column and its writer exist.
    last_synced_at: datetime | None = None
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

    #: ⚠ **Present on the PORT DTO and absent from the WIRE body — D-33's precedence rule.**
    #: ``DeviceEnrollRequest`` is constructed SERVER-SIDE by ``DeviceService.enroll`` from the
    #: verified ``AuthContext``; the HTTP body for ``POST /v1/devices/enroll`` carries no
    #: ``org_id``/``principal_id``/``workspace_id`` at all, because a caller that can name its
    #: own tenancy can name someone else's (``mu-server/CLAUDE.md`` invariant 2). This is why
    #: ``enroll`` keeps its single-argument face while D-13 puts ``org_id`` first on every other
    #: port method: one carrier per call, never both unsigned.
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    public_key: str = Field(min_length=1)
    platform: DevicePlatform
    label: str = Field(default="", max_length=120)
    app_build: str = Field(default="", max_length=64)
    client_mode: ClientMode
    privacy_tier: PrivacyTier


class RevokeReason(StrEnum):
    """The CLOSED set a device revocation may carry — phase-3 spec **D-34**.

    ``DeviceRevoked.reason`` is a required free-form ``str`` in the event catalog
    (``CANONICAL-CONTRACTS.md:375``), and the bus's content-free guard is a **field-name** check
    (``events.py``'s ``_FORBIDDEN_EVENT_FIELDS``) — so an operator- or user-supplied sentence in
    ``reason`` passes class definition and lands on the SHARED bus and in every trace of that
    event. The codebase already has the right precedent one layer over: ``MemoryItem.quarantined``
    documents *"`reason` is a named code, never memory content"*.

    So the SERVICE face takes this enum and the EVENT field stays ``str`` (carrying the member's
    value). ``StrEnum`` makes that a projection rather than a conversion, and it makes the closed
    set the only thing a caller can construct.

    ``UNSPECIFIED`` exists because ``D:203``/``S:136`` type the service's ``reason`` as
    ``str | None`` while the event's field is REQUIRED: a ``None`` reason has no defined mapping,
    so the service substitutes this constant rather than widening the event.
    """

    UNSPECIFIED = "unspecified"
    USER_REQUESTED = "user_requested"
    LOST_OR_STOLEN = "lost_or_stolen"
    DECOMMISSIONED = "decommissioned"
    POLICY = "policy"
    KEY_COMPROMISE = "key_compromise"
