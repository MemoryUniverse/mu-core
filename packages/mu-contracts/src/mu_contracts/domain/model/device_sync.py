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

#: ⚠ **Every string field below is bounded to the width of the column that stores it**
#: (``mu_engine.storage.relational.schema.PrivateSyncLogRow``: ``String(128)`` for every
#: content-free id, ``String(512)`` for ``payload_ref``). This is a SECURITY bound, not tidiness.
#:
#: Without it an authenticated principal can put arbitrary unbounded text in ``content_hash`` or
#: ``memory_id``; the driver raises ``StringDataRightTruncation``, which is a ``DBAPIError`` and
#: NOT an ``IntegrityError``, so ``mu-server``'s append retry does not catch it and no handler in
#: ``errors.py`` maps it — it escapes as a 500 whose exception string carries the whole bound
#: parameter tuple into the hosted plane's logs. That is a breach of the content-free discipline
#: (project rule 3 / ``mu-server/CLAUDE.md`` invariant 4) reached through a path nothing else
#: guards, and a log-flood vector besides. Bounded here, it is a 422 at the wire.
#:
#: ⚠ **These bounds match ``schema.py``, which is STRICTER than the shipped Alembic DDL.** Revision
#: ``46ae4bcc2472`` created ``private_sync_log`` with bare ``sa.String()`` (unbounded VARCHAR),
#: while ``schema.py`` — what ``Base.metadata.create_all`` builds, and what the integration suites
#: run against — declares ``String(128)``. So a migration-built deployment would have SILENTLY
#: STORED the oversized value rather than raising. Bounding the wire type closes both shapes at
#: once; reconciling the two DDLs is a separate, pre-existing drift.
_ID_MAX = 128  # every content-free id column: String(128)
_REF_MAX = 512  # payload_ref: String(512)
#: ``BigInteger`` on Postgres. asyncpg refuses an out-of-range value client-side with a
#: ``DataError`` (verified against the real database), which is again a ``DBAPIError`` → 500.
_BIGINT_MAX = 2**63 - 1
#: RESERVED §7.18 has no reader on this plane, so this is purely an ALLOCATION bound: without it
#: ``ciphertext`` is the one field left through which a bounded batch still costs unbounded memory.
#: It is not a semantic decision about the E2E tier, which stays RESERVED.
_CIPHERTEXT_MAX = 1 << 20


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

    seq: int = Field(ge=0, le=_BIGINT_MAX)  # hub-assigned, per-user monotonic ordering authority
    # SENTINEL "server" for appender-B; DO NOT branch
    origin_device_id: str = Field(min_length=1, max_length=_ID_MAX)
    op: SyncOp
    memory_id: str = Field(min_length=1, max_length=_ID_MAX)  # tier-stable MemoryItem.id (§7.1)
    # client-computed over plaintext — version/dedupe key, and trusted for DEDUPE ONLY (§7.14b)
    content_hash: str = Field(min_length=1, max_length=_ID_MAX)
    tier: str = Field(min_length=1, max_length=_ID_MAX)  # Tier enum value (stm|mtm|ltm)
    valid_at: datetime  # bi-temporal validity
    valid_at_inferred: bool = False  # §7.17-1: True iff from DATE_EXTRACTION_FALLBACK
    # per-device logical clock at author time (the ordering authority)
    lamport: int = Field(ge=0, le=_BIGINT_MAX)
    occurred_at: datetime  # device wall-clock — ADVISORY only, NEVER the ordering authority
    provenance_id: str = Field(min_length=1, max_length=_ID_MAX)  # origin lineage (§7.10 G4)
    # winner_id/loser_id are REQUIRED for op in {SUPERSEDE, REINSTATE} — see the validator below
    winner_id: str | None = Field(default=None, max_length=_ID_MAX)
    loser_id: str | None = Field(default=None, max_length=_ID_MAX)
    # §7.17 item 4a — the two leading (dominant, non-lexicographic) terms of the total order.
    # Both MUST live on the delta itself (CANONICAL:777 "every field is on the delta itself"),
    # since `total_order_key` (mu_engine.services.conflict.order) is a pure function over two
    # PrivateDelta candidates with no store/registry access. `pinned` mirrors the winning
    # MemoryItem's pin state onto the delta that asserts/reinforces it; `resolution_origin` is
    # set to MANUAL on the delta a manual conflict resolution emits (conflict-resolution-async
    # -design.md:261: "reuse SyncOp.SUPERSEDE/REINSTATE ... plus resolution_origin='manual'").
    pinned: bool = False
    resolution_origin: ResolutionOrigin | None = None
    #: The principal that resolved a conflict manually. CANONICAL:538 pins that
    #: ``SUPERSEDE``/``REINSTATE`` *"additionally carry ``resolution_origin ∈ {automatic, manual}``
    #: + ``resolved_by``"*, and phase-3 spec §3.3 schedules the field here; it landed one step late
    #: because `resolution_origin` arrived without it. It is deliberately NOT one of the seven
    #: total-order terms — `resolution_origin == "manual"` is the term, and *who* resolved it must
    #: not change *which* item wins, or two replicas with different views of the actor would
    #: diverge. It is an ATTRIBUTION field: it is what makes a sticky manual winner explicable on
    #: the surface that renders it, and it is a content-free principal id.
    resolved_by: str | None = Field(default=None, max_length=_ID_MAX)
    # replica-apply echo of log seq N; projector DROPS it. ⚠ Appender A SCRUBS it to None on
    # ingest (`SyncHubService._normalize`): only the plane's own appender B may author it.
    caused_by_seq: int | None = Field(default=None, ge=0, le=_BIGINT_MAX)
    # pointer into the private-hosted partition (the body lives there)
    payload_ref: str | None = Field(default=None, max_length=_REF_MAX)
    key_epoch: int = Field(default=0, ge=0, le=_BIGINT_MAX)  # RESERVED §7.18
    ciphertext: bytes | None = Field(default=None, max_length=_CIPHERTEXT_MAX)  # RESERVED §7.18

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

    principal_id: str = Field(min_length=1, max_length=_ID_MAX)
    head_seq: int = Field(ge=0, le=_BIGINT_MAX)
    items: tuple[str, ...] = ()  # payload_refs
