"""Private cross-device sync vocabulary — SyncOp + PrivateDelta + PrivateSnapshot.

Authority: sync-devices-gateway-spec.md §B.2 (ported verbatim), CANONICAL §7.5 X5 (the
``REINSTATE`` op + the re-derivable ``(winner_id, loser_id, valid_at, lamport,
origin_device_id)`` tuple), §7.17 (logical-clock ordering; ``lamport`` is the authority,
``occurred_at`` is advisory only).

The persisted delta is content-free on the shipping tier — the body lives in the user's
private-hosted partition and ``payload_ref`` points at it. ``seq`` is hub-assigned and is the
per-user monotonic ordering authority; ``last_synced_seq`` is advanced BY THE HUB on delivery
ack, never asserted forward by the device (CANONICAL §7.6 X16).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mu_contracts.domain.model.conflict import ResolutionOrigin

__all__ = ["PrivateDelta", "PrivateSnapshot", "SyncOp"]


class SyncOp(StrEnum):
    """The seven sync operations (CANONICAL §7.5 X5 / C7 merge). ``PIN``/``UNPIN`` propagate a
    THIN pin/unpin fleet-wide (§7.14/§7.26); ``REINSTATE`` clears a local supersede when
    re-evaluation over the merged delta set says the item is active again."""

    UPSERT = "upsert"
    SUPERSEDE = "supersede"
    TOMBSTONE = "tombstone"
    DEMOTE = "demote"
    REINSTATE = "reinstate"
    PIN = "pin"
    UNPIN = "unpin"


class PrivateDelta(BaseModel):
    """One entry in the per-user private sync log. Content-free on the shipping tier —
    ``payload_ref`` points at the body in the private-hosted partition (sync-devices §B.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int = Field(ge=0)  # hub-assigned, per-user monotonic ordering authority
    origin_device_id: str = Field(min_length=1)  # SENTINEL "server" for appender-B; DO NOT branch
    op: SyncOp
    memory_id: str = Field(min_length=1)  # tier-stable MemoryItem.id (CANONICAL §7.1)
    content_hash: str = Field(min_length=1)  # client-computed over plaintext — version/dedupe key
    tier: str = Field(min_length=1)  # Tier enum value (stm|mtm|ltm)
    valid_at: datetime  # bi-temporal validity
    valid_at_inferred: bool = False  # §7.17-1: True iff from DATE_EXTRACTION_FALLBACK
    lamport: int = Field(ge=0)  # per-device logical clock at author time (the ordering authority)
    occurred_at: datetime  # device wall-clock — ADVISORY only, NEVER the ordering authority
    provenance_id: str = Field(min_length=1)  # origin lineage (content-free id, §7.10 G4)
    winner_id: str | None = None  # REQUIRED for op in {SUPERSEDE, REINSTATE}
    loser_id: str | None = None  # REQUIRED for op in {SUPERSEDE, REINSTATE}
    # §7.17 item 4a — the two leading (dominant, non-lexicographic) terms of the total order.
    # Both MUST live on the delta itself (CANONICAL:777 "every field is on the delta itself"),
    # since `total_order_key` (mu_engine.services.conflict.order) is a pure function over two
    # PrivateDelta candidates with no store/registry access. `pinned` mirrors the winning
    # MemoryItem's pin state onto the delta that asserts/reinforces it; `resolution_origin` is
    # set to MANUAL on the delta a manual conflict resolution emits (conflict-resolution-async
    # -design.md:261: "reuse SyncOp.SUPERSEDE/REINSTATE ... plus resolution_origin='manual'").
    pinned: bool = False
    resolution_origin: ResolutionOrigin | None = None
    caused_by_seq: int | None = None  # replica-apply echo of log seq N; projector DROPS it
    payload_ref: str | None = None  # pointer into the private-hosted partition (body lives there)
    key_epoch: int = Field(default=0, ge=0)  # RESERVED §7.18
    ciphertext: bytes | None = None  # RESERVED §7.18 (E2E only)

    @model_validator(mode="after")
    def _supersede_carries_pair(self) -> PrivateDelta:
        """SUPERSEDE/REINSTATE MUST carry the full pair so any replica can re-derive the
        decision (CANONICAL §7.5 X5) — a one-way stamp on the loser is a lost-update hazard."""
        if self.op in (SyncOp.SUPERSEDE, SyncOp.REINSTATE) and (
            self.winner_id is None or self.loser_id is None
        ):
            raise ValueError(f"{self.op.value} requires both winner_id and loser_id")
        return self


class PrivateSnapshot(BaseModel):
    """Re-seed payload (sync-devices §B.9) — content-free refs into the private-hosted
    partition; bodies materialized by REST under fresh authz."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(min_length=1)
    head_seq: int = Field(ge=0)
    items: tuple[str, ...] = ()  # payload_refs
