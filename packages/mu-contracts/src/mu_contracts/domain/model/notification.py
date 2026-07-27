"""Notification vocabulary — the content-free per-principal inbox record + preferences.

Authority: trust-surfaces-notifications-spec.md §4.1-§4.2. A notification is content-free by
construction: the human-readable string is rendered client-side from a template keyed by
``(kind, params)`` — never stored, never on a bus. ``params`` is validated content-free-by-type
(``dict[str, str]`` scalars; no ``body``/``text``/``content``/``message`` key — same construction
lint as the bus, CANONICAL §3.1). ``notification_id`` is idempotent/coalescing; ``seq`` is a
per-principal monotonic cursor (contiguity backfill, §4.6).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ChannelPolicy",
    "FrameRef",
    "Notification",
    "NotificationCategory",
    "NotificationKind",
    "NotificationPreferences",
    "NotificationSeverity",
    "NotificationState",
    "QuietHours",
]

# Keys that would smuggle memory content into a content-free record (CANONICAL §3.1).
_FORBIDDEN_PARAM_KEYS = frozenset({"body", "text", "content", "message", "prompt", "raw", "secret"})


class NotificationCategory(StrEnum):
    CONFLICT = "conflict"
    SYNC = "sync"
    REVOCATION = "revocation"
    SHARING = "sharing"
    MEMORY_HEALTH = "memory_health"
    SYSTEM = "system"


class NotificationKind(StrEnum):
    """Finer, closed — one per trigger (pinned like ``DegradeReason``)."""

    CONFLICT_PENDING = "conflict_pending"
    CONFLICT_RESOLVED = "conflict_resolved"
    SYNC_STALLED = "sync_stalled"
    SYNC_FAILED = "sync_failed"
    PRIMARY_SYNC_LAG = "primary_sync_lag"
    DEVICE_WIPE_PENDING = "device_wipe_pending"
    DEVICE_REVOKED = "device_revoked"
    GRANT_REVOKED = "grant_revoked"
    SHARED_WITH_YOU = "shared_with_you"
    SUBSCRIPTION_MATCHED = "subscription_matched"
    MEMORY_HEALTH_ALERT = "memory_health_alert"


class NotificationSeverity(StrEnum):
    INFO = "info"  # can be fully muted
    WARNING = "warning"  # inbox floor unless category muted
    ACTION_REQUIRED = "action_required"  # ALWAYS lands in the inbox (safety floor §4.3)


class NotificationState(StrEnum):
    UNREAD = "unread"
    READ = "read"
    DISMISSED = "dismissed"


class FrameRef(BaseModel):
    """Deep-link target of a notification (content-free id + kind)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref_kind: str = Field(min_length=1)  # conflict|device|grant|index|session
    ref_id: str = Field(min_length=1)


class Notification(BaseModel):
    """The per-principal inbox record (trust-surfaces §4.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    notification_id: str = Field(min_length=1)  # sha256(ns.to_prefix|kind|dedup_key)[:24]
    workspace_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)  # the ONE recipient; never a broadcast
    seq: int = Field(ge=0)  # per-principal monotonic cursor (contiguity backfill)
    category: NotificationCategory
    kind: NotificationKind
    severity: NotificationSeverity
    params: dict[str, str] = Field(default_factory=dict)  # CONTENT-FREE scalars only
    ref: FrameRef | None = None
    source_event: str = Field(min_length=1)  # the DomainEvent class name that produced it
    plane: str = Field(min_length=1)  # "local" | "shared"
    state: NotificationState = NotificationState.UNREAD
    created_at: datetime
    read_at: datetime | None = None

    @field_validator("params")
    @classmethod
    def _params_are_content_free(cls, params: dict[str, str]) -> dict[str, str]:
        bad = _FORBIDDEN_PARAM_KEYS & {k.lower() for k in params}
        if bad:
            raise ValueError(
                f"notification params must be content-free; forbidden keys: {sorted(bad)}"
            )
        return params


class QuietHours(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    start_hour_utc: int = Field(default=0, ge=0, le=23)
    end_hour_utc: int = Field(
        default=0, ge=0, le=23
    )  # push suppressed in-window; inbox still written


class ChannelPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    inbox_enabled: bool = True  # write the durable record (the badge/list)
    push_enabled: bool = True  # emit the personal:{pid} notification frame
    min_severity: NotificationSeverity = NotificationSeverity.INFO


class NotificationPreferences(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str = Field(min_length=1)
    principal_id: str = Field(min_length=1)
    per_category: dict[NotificationCategory, ChannelPolicy] = Field(default_factory=dict)
    default_policy: ChannelPolicy = ChannelPolicy()
    quiet_hours: QuietHours = QuietHours()
    updated_at: datetime
