"""``SyncStatusView`` — the USER-VISIBLE sync assurance surface (CANONICAL-CONTRACTS.md §7.15).

**The requirement, verbatim:** *"the user must be sure there is sync."* Every ingredient already
existed (``DeviceRecord.last_synced_seq``, ``PrivateSyncLogPort.head_seq``) and nothing composed
them; the system's entire degradation vocabulary (``DegradeReason``, "no silent fallback" as a
Layer-0 rule) was routed to ``MetricSink`` with a **"platform only"** consumer — an excellent
**operator** guarantee and a non-existent **user** one. This module pins the user half. It is
**distinct from, and additional to,** ``MetricSink`` — never a substitute either way.

Authority: ``CANONICAL-CONTRACTS.md:725-746`` (§7.15, the code block headed
``# mu_core/domain/model/sync_status.py``) is the RANK-ABOVE-ALL shape, transcribed here verbatim.
``sync-devices-gateway-spec.md`` §B2.1 (``S:346-364``) and ``device-registry-sync-design.md``
§5a.1 (``D:346-364``) carry the same ``SyncState``/``DeviceSyncStatus`` shape but a
``SyncStatusView`` that OMITS two fields CANONICAL adds — see the docstring on
``SyncStatusView.pending_conflicts`` below. CANONICAL wins; this module follows CANONICAL.

Content-free by construction (project rule 3 / CANONICAL §3): every field here is an id, a count,
an enum value, or a timestamp — never memory content. A SYNC-CLASS ``DegradeReason`` (the closed
subset in ``mu_contracts.domain.events.SYNC_CLASS_ALWAYS``) that does not reach this view is a
**contract violation** (CANONICAL §7.15).

This is a **read DTO only** — the computation (``SyncStatusProjector``) is a ``[server]``
component, built elsewhere, never in ``mu-core`` (CLAUDE.md: mu-core holds only what mu-client AND
mu-server both need on the wire).

DEV-STANDARDS deviation (recorded, mirrors ``domain/events.py``'s own recorded deviation): CANONICAL
types ``SyncStatusView.devices`` as ``list[DeviceSyncStatus]``. Every ``frozen=True`` model
directly inside this ``domain/model/`` subpackage uses ``tuple[..., ...]`` for a collection field
instead (zero ``: list[`` field annotations anywhere under ``domain/model/`` — verified by grep,
2026-08) — a ``list`` attribute stays internally mutable under Pydantic's ``frozen=True``, which
only blocks attribute *reassignment*, not in-place mutation of the collection it points at.
``tuple`` is transcribed here as ``tuple[DeviceSyncStatus, ...]`` to match that ``domain/model/``
convention; the field set and semantics are otherwise identical to CANONICAL.

⚠ **That convention does NOT hold package-wide, and this docstring previously claimed it did**
("every frozen=True model in **this package**"). It is false at that scope: ``contracts/recall.py``
(``RecallResult.items: list[RecallItemView]``), ``contracts/views.py``
(``ContextView.items: list[RecallItemView]``), and ``ports/stores.py``
(``QdrantPoint.vector: list[float]``) are all ``frozen=True`` models elsewhere in ``mu_contracts``
that use ``list``. The claim is corrected here to the scope it can actually support
(``domain/model/`` only); whether ``tuple`` over CANONICAL's ``list`` is the right call at all is
reported as a delta, not decided in this docstring.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.events import DegradeReason

__all__ = ["DeviceSyncStatus", "SyncState", "SyncStatusView"]


class SyncState(StrEnum):
    """The one-glance sync health for a device or a fleet (CANONICAL §7.15)."""

    IN_SYNC = "in_sync"  # behind_by == 0 and the reconcile tick is current
    BEHIND = "behind"  # behind_by > 0, catching up normally, tick is current
    STALLED = "stalled"  # behind_by > 0 AND the tick has not advanced in `stall_after_ticks`
    FAILED = "failed"  # a delta is dead-lettered, or key/wipe ack is overdue — needs a user action
    OFFLINE = "offline"  # no contact within `device_offline_after_s`
    RESEEDING = "reseeding"  # snapshot re-seed in progress (SYNC_BACKFILL_WINDOW_EXCEEDED)


class DeviceSyncStatus(BaseModel):
    """Per-device row on the fleet view (CANONICAL §7.15). Holds nothing a device could lie
    about: ``last_synced_seq`` is HUB-advanced (mirrors ``DeviceRecord.last_synced_seq``),
    ``head_seq`` is the log's, ``undelivered_count`` is derived from unacked/dead-lettered
    records — never device-asserted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # CANONICAL-CONTRACTS.md:725-746 types every field below (through `last_error_kind`) as a
    # bare, unbounded type with no default — i.e. REQUIRED, and with no min_length/ge/max_length
    # constraint. Transcribed verbatim: no validation invented beyond CANONICAL's shape.
    device_id: str
    label: str  # the user-chosen name — this surface is READ BY A HUMAN
    last_synced_seq: int
    head_seq: int  # the hub's current head for this principal
    behind_by: int  # head_seq - last_synced_seq — the join nothing performed
    last_synced_at: datetime | None  # REQUIRED (no default) — value itself may still be None
    state: SyncState
    # REQUIRED (no default). The NAMED reason (§10), never free text.
    last_error_kind: DegradeReason | None
    undelivered_count: int = 0  # dead-lettered/unacked deltas from THIS device


class SyncStatusView(BaseModel):
    """The per-principal fleet view (CANONICAL §7.15). Three surfaces read this SAME view —
    ``GET /v1/devices/sync/status``, the daemon's ``mu status`` IPC, and the ``memory.local.status``
    MCP tool — none computes its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # CANONICAL-CONTRACTS.md:725-746: `principal_id`, `head_seq`, `devices`, `fleet_state`,
    # `generated_at` are bare, unbounded, and REQUIRED (no default) — transcribed verbatim.
    principal_id: str
    head_seq: int
    # REQUIRED (no default). CANONICAL types this `list[DeviceSyncStatus]` — see docstring above.
    devices: tuple[DeviceSyncStatus, ...]
    fleet_state: SyncState  # worst state across devices — the one-glance answer
    generated_at: datetime
    #: **CANONICAL-CONTRACTS.md:732, the C10 merge** (trust-surfaces §CC-1 + api-mcp §11.3) — a
    #: content-free soft count. ``sync-devices-gateway-spec.md:359-365`` (S) and
    #: ``device-registry-sync-design.md:436-443`` (D) both OMIT this field; CANONICAL is the
    #: rank-above-all authority (CLAUDE.md) and ADDS it, so it is transcribed here even though
    #: neither design doc shows it. Soft-fails to 0 on a read failure — never fails the sync
    #: numbers (CANONICAL §7.15).
    pending_conflicts: int = 0
    #: See ``pending_conflicts`` — same C10 merge, same omission in S/D, same soft-fail-to-0 rule.
    pending_notifications: int = 0
