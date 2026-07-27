"""PrivateSyncLogPort — ONE log, ONE seq, TWO appenders (sync-devices-gateway-spec.md §B.3).

``append`` is the SOLE seq authority (hub-assigned monotonic). ``read`` is the durable backfill
from a cursor (contiguity apply-rule, CANONICAL §7.6 X16). ``floor_seq`` drives the reseed
decision when a device has fallen behind the retention window.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.device_sync import PrivateDelta

__all__ = ["PrivateSyncLogPort"]


@runtime_checkable
class PrivateSyncLogPort(Protocol):
    async def append(
        self, workspace_id: str, principal_id: str, delta: PrivateDelta
    ) -> int:  # -> seq; sole seq authority
        ...

    async def read(
        self, workspace_id: str, principal_id: str, *, after_seq: int, limit: int
    ) -> list[PrivateDelta]: ...

    async def head_seq(self, workspace_id: str, principal_id: str) -> int: ...

    async def floor_seq(
        self, workspace_id: str, principal_id: str
    ) -> int:  # retention floor → reseed decision
        ...
