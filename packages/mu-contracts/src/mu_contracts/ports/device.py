"""DeviceRegistryPort — the SINGLE device registry port (sync-devices-gateway-spec.md §A.3).

The gateway does NOT define its own. ``touch()`` is unified and carries THREE things —
content-free ``ClientMetadata`` (heartbeat) AND ``last_synced_seq`` AND ``last_lamport`` — so a
two-signature split-brain is structurally impossible. All three hot-path mutables are HUB-advanced
only (a device cannot lie its cursor/VC forward). Reads are scoped ``(workspace, principal[, id])``,
never a global ``device_id`` scan.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.model.device import DeviceEnrollRequest, DeviceRecord, DeviceState

__all__ = ["ClientMetadata", "DeviceRegistryPort"]


class ClientMetadata(BaseModel):
    """Content-free heartbeat metadata carried on ``touch()`` (gateway §7.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_build: str = ""
    platform: str = ""


@runtime_checkable
class DeviceRegistryPort(Protocol):
    async def enroll(self, req: DeviceEnrollRequest) -> DeviceRecord:  # idempotent put-if-absent
        ...

    async def resolve(self, device_id: str) -> DeviceRecord | None: ...

    async def devices_of(self, principal_id: str) -> list[DeviceRecord]: ...

    async def touch(
        self,
        device_id: str,
        *,
        meta: ClientMetadata,
        last_seen_at: datetime,
        last_synced_seq: int,
        last_lamport: int,
    ) -> None: ...

    async def revoke(self, device_id: str, *, reason: str) -> None: ...

    async def set_state(
        self, device_id: str, state: DeviceState, *, at: datetime, by: str | None = None
    ) -> DeviceRecord:  # compare-and-set (PENDING→ACTIVE)
        ...
