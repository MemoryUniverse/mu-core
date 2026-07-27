"""Notification ports (trust-surfaces-notifications-spec.md §4).

``NotificationRepository`` is the per-principal durable inbox (contiguity backfill by ``seq``);
``NotificationPreferenceRepository`` holds per-user, per-category channel policy. ACTION_REQUIRED
inbox floor is enforced in the service, not here (a pref that disables it is clamped at write).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.notification import Notification, NotificationPreferences

__all__ = ["NotificationPreferenceRepository", "NotificationRepository"]


@runtime_checkable
class NotificationRepository(Protocol):
    async def add(self, notification: Notification) -> None:  # idempotent by notification_id
        ...

    async def list_for(
        self, workspace_id: str, principal_id: str, *, after_seq: int, limit: int
    ) -> list[Notification]: ...

    async def mark_read(self, notification_id: str) -> None: ...

    async def head_seq(self, workspace_id: str, principal_id: str) -> int: ...


@runtime_checkable
class NotificationPreferenceRepository(Protocol):
    async def get(self, workspace_id: str, principal_id: str) -> NotificationPreferences | None: ...

    async def upsert(self, prefs: NotificationPreferences) -> None: ...
