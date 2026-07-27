"""The full relational schema — SQLAlchemy 2.x typed ``DeclarativeBase`` (spec §2).

ONE schema module (spec §2 "Abstraction seam"): the SAME shape binds to Postgres (server)
and SQLite (client); adapters MUST NOT hand-roll per-dialect SQL. Content-free discipline
(spec §0 / CANONICAL §3): every column holds ids / hashes / enum values / counts /
timestamps only — NEVER raw memory text. The only text-adjacent columns are ``label``
(device name) and ``slug`` (workspace handle), both content-free.

PORT: "control/history defaults to SQLite unconditionally" from mem0
(``mem0/mem0/configs/base.py:14,43-46``); UPGRADED from mem0's raw per-driver SQL to
SQLAlchemy Core for portability (recorded deviation, storage-pluggable §2.1).

Dialect divergence (spec §2, degrade D7): Postgres partial indexes
(``WHERE revoked_at IS NULL``) + GIN have no portable equivalent — expressed via
``postgresql_where=`` / ``postgresql_using="gin"``, which SQLite/MySQL ignore
(plain composite index). This is an index-efficiency degrade, never a correctness change.

MySQL portability (owner stage 2026-07-27, adding ``mysql`` as a relational backend), confirmed
against a REAL ``mu-dev-mysql`` container (two errors, both fixed here):
1. Every ``String`` column now carries an explicit length — bare ``VARCHAR`` (no length) is
   valid DDL on Postgres/SQLite but MySQL's ``CREATE TABLE`` REJECTS it (``CompileError: VARCHAR
   requires a length on dialect mysql``). ``String(128)`` is the default for identifiers/
   enums/hashes; wider free-form fields (``object_ref``, ``public_key``, ``payload_ref``) get
   ``String(512)``; ``namespace_prefix`` gets ``String(384)`` — all sized so composite indexes/
   PKs/unique constraints stay under InnoDB's 3072-byte key limit at ``utf8mb4`` (4 bytes/char)
   (``OperationalError: Specified key was too long; max key length is 3072 bytes`` on the
   4-column ``agent_bindings`` PK and the 4-string ``ux_synclog_occurrence`` unique constraint
   before this fix). This is a portability fix, not a semantic change — Postgres/SQLite accept a
   length-bound ``VARCHAR`` identically, and no realistic id/hash/enum value in this schema
   approaches these bounds.
2. The two GIN indexes on JSON columns (``gin_prov_meta``, ``gin_conflict_members``) are
   POSTGRES-ONLY DDL, gated by ``DDL(...).execute_if(dialect="postgresql")`` at the bottom of
   this module rather than declared in ``__table_args__`` — MySQL 8 cannot index a raw JSON
   column at all (``JSON column ... cannot be used in key specification``); SQLite tolerates a
   plain index (TEXT affinity) but gets no GIN benefit either. Both dialects still get a working,
   correct table — only the JSON-search acceleration is Postgres-specific (degrade **D7**,
   ``RELATIONAL_INDEX_REDUCED`` — an efficiency loss, never a correctness loss, spec §7).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    DDL,
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB as _PG_JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = ["Base"]

# jsonb on Postgres, json elsewhere (spec §2 line 171).
JSONB = JSON().with_variant(_PG_JSONB, "postgresql")


class Base(DeclarativeBase):
    """The single declarative base binding every table (spec §2)."""


def _dt() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False)


def _dt_opt() -> Mapped[datetime | None]:
    return mapped_column(DateTime(timezone=True))


# ============================================================ §2.1 control-plane identity/authz
# Un-collapsed η (CANONICAL §1 rule 3 / ADR 0026): `org` is the tenant/billing/RESIDENCY root,
# `workspace` is a many-per-org grouping. The tenancy grain is `org`, so every tenant-scoped
# table carries an explicit `org_id` column (NOT collapsed into `workspace_id`), and residency
# keys on `orgs.residency_region` (rule 7). One org contains many workspaces.
class Org(Base):
    __tablename__ = "orgs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # tenant/billing/residency root
    slug: Mapped[str] = mapped_column(String(128), nullable=False)  # content-free handle
    residency_region: Mapped[str] = mapped_column(String(128), nullable=False, default="")  # rule 7
    created_at: Mapped[datetime] = _dt()

    __table_args__ = (Index("ux_orgs_slug", "slug", unique=True),)


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    slug: Mapped[str] = mapped_column(String(128), nullable=False)  # content-free handle
    created_at: Mapped[datetime] = _dt()

    __table_args__ = (
        Index("ux_workspaces_org_slug", "org_id", "slug", unique=True),  # slug unique WITHIN org
        Index("ix_workspaces_org", "org_id"),
    )


class Membership(Base):
    __tablename__ = "workspace_memberships"

    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(128), nullable=False)  # OWNER|ADMIN|MEMBER|GUEST
    status: Mapped[str] = mapped_column(String(128), nullable=False)  # active|suspended|removed
    expires_at: Mapped[datetime | None] = _dt_opt()
    created_at: Mapped[datetime] = _dt()

    __table_args__ = (
        PrimaryKeyConstraint("workspace_id", "principal_id"),
        Index("ix_ws_membership_org", "org_id", "principal_id"),
    )


class Principal(Base):
    __tablename__ = "principals"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # pseudonymous stable id
    kind: Mapped[str] = mapped_column(String(128), nullable=False)  # human|agent|service


class NamespaceDef(Base):
    __tablename__ = "namespaces"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    owner_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    allowed_visibilities: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class SessionDef(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    namespace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    state: Mapped[str] = mapped_column(String(128), nullable=False)  # open|closed
    expires_at: Mapped[datetime | None] = _dt_opt()
    created_at: Mapped[datetime] = _dt()


class SessionParticipant(Base):
    __tablename__ = "session_participants"

    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # a PRINCIPAL id (Model A)
    principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    joined_at: Mapped[datetime] = _dt()
    left_at: Mapped[datetime | None] = _dt_opt()  # offboarding => re-stamp trigger

    __table_args__ = (PrimaryKeyConstraint("session_id", "principal_id"),)


class AgentBinding(Base):
    __tablename__ = "agent_bindings"

    agent_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    namespace_id: Mapped[str | None] = mapped_column(String(128), default="")
    session_id: Mapped[str | None] = mapped_column(String(128), default="")
    allowed_tools: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    policy_cap: Mapped[str] = mapped_column(String(128), nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("agent_principal_id", "workspace_id", "namespace_id", "session_id"),
    )


# ============================================================ §2.2 acl_entries
class AclEntry(Base):
    __tablename__ = "acl_entries"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # ShareableRef id (by value)
    object_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    subject_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    grantee_kind: Mapped[str] = mapped_column(String(128), nullable=False)  # user|session|role
    grant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = _dt()
    revoked_at: Mapped[datetime | None] = _dt_opt()  # NULL = live

    __table_args__ = (
        Index(
            "ix_acl_object_active",
            "workspace_id",
            "object_ref",
            postgresql_where=text("revoked_at IS NULL"),
        ),
        Index(
            "ix_acl_subject_active",
            "subject_principal_id",
            postgresql_where=text("revoked_at IS NULL"),
        ),
    )


# ============================================================ §2.3 grants
class Grant(Base):
    __tablename__ = "grants"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    object_ref: Mapped[str] = mapped_column(String(512), nullable=False)  # SHARED objects only
    grantor: Mapped[str] = mapped_column(String(128), nullable=False)
    grantee_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    grantee_kind: Mapped[str] = mapped_column(String(128), nullable=False)  # user|session|role
    direction: Mapped[str] = mapped_column(String(128), nullable=False)
    parent_grant_id: Mapped[str | None] = mapped_column(String(128))  # revoke cascade edge
    created_at: Mapped[datetime] = _dt()
    revoked_at: Mapped[datetime | None] = _dt_opt()

    __table_args__ = (
        Index(
            "ix_grants_grantee_active",
            "grantee_principal_id",
            postgresql_where=text("revoked_at IS NULL"),
        ),
        Index("ix_grants_parent", "parent_grant_id", postgresql_where=text("revoked_at IS NULL")),
        Index(
            "ix_grants_object",
            "workspace_id",
            "object_ref",
            postgresql_where=text("revoked_at IS NULL"),
        ),
    )


# ===================================== §2.4 memory_provenance (the MemoryItem mirror)
class MemoryProvenance(Base):
    __tablename__ = "memory_provenance"

    memory_id: Mapped[str] = mapped_column(String(128), primary_key=True)  # tier-stable id
    # tenancy/billing root (ADR 0026)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    namespace_prefix: Mapped[str] = mapped_column(String(384), nullable=False)  # to_prefix()
    owner_id: Mapped[str] = mapped_column(String(128), nullable=False)
    visibility: Mapped[str] = mapped_column(String(128), nullable=False)  # private|shared
    kind: Mapped[str] = mapped_column(String(128), nullable=False)  # proposition|reference
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)  # version/dedupe key
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    tier: Mapped[str] = mapped_column(String(128), nullable=False)  # stm|mtm|ltm
    state: Mapped[str] = mapped_column(String(128), nullable=False)  # active|archived|...
    artifact_ref: Mapped[str | None] = mapped_column(String(128))  # first-class (CANONICAL §7.1)
    provenance_id: Mapped[str] = mapped_column(String(128), nullable=False)  # required non-empty
    superseded_by: Mapped[str | None] = mapped_column(String(128))
    synced_at: Mapped[datetime] = _dt()
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        # idempotent sync dedupe at the ratified (org, workspace) grain (ADR 0026; Cognee unique
        # key → (org_id, workspace_id, name)); content_hash is the version key within that grain.
        Index("ux_prov_chash", "org_id", "workspace_id", "content_hash", unique=True),
        Index("ix_prov_owner", "org_id", "workspace_id", "owner_id"),
        Index(
            "ix_prov_artifact",
            "workspace_id",
            "artifact_ref",
            postgresql_where=text("artifact_ref IS NOT NULL"),
        ),
        # NOTE: the `meta` GIN index is Postgres-only DDL, registered via `event.listen(...)`
        # near the bottom of this module (module docstring point 2) — NOT declared here, because
        # MySQL cannot index a raw JSON column at all (unlike the `postgresql_where=` partial
        # indexes above, which degrade gracefully to a plain index on every other dialect).
    )


# ============================== §2.5 fact_provenance (content-free LTM-fact MIRROR)
class FactProvenance(Base):
    __tablename__ = "fact_provenance"

    memory_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    namespace_prefix: Mapped[str] = mapped_column(String(384), nullable=False)
    subject_entity_uid: Mapped[str] = mapped_column(String(128), nullable=False)  # NOT surface text
    predicate: Mapped[str] = mapped_column(String(128), nullable=False)  # controlled vocab token
    object_entity_uid: Mapped[str | None] = mapped_column(String(128))
    object_value_hash: Mapped[str | None] = mapped_column(String(128))  # sha256 of a literal value
    polarity: Mapped[str] = mapped_column(String(128), nullable=False)  # positive|negative
    valid_at: Mapped[datetime] = _dt()
    invalid_at: Mapped[datetime | None] = _dt_opt()  # set on supersede
    valid_at_inferred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    state: Mapped[str] = mapped_column(String(128), nullable=False)  # active|superseded
    recorded_at: Mapped[datetime] = _dt()

    __table_args__ = (
        Index("ix_fact_conflict", "namespace_prefix", "subject_entity_uid", "predicate"),
        Index(
            "ix_fact_active",
            "namespace_prefix",
            "state",
            postgresql_where=text("state = 'active'"),
        ),
    )


# ============================================================ §2.6 provenance_ledger
class ProvenanceLedgerRow(Base):
    __tablename__ = "provenance_ledger"

    stream_id: Mapped[str] = mapped_column(String(128), nullable=False)  # provenance_id
    version: Mapped[int] = mapped_column(Integer, nullable=False)  # monotonic within stream
    action: Mapped[str] = mapped_column(
        String(128), nullable=False
    )  # ORIGIN|DERIVED|SUPERSEDED|COMPOSED
    memory_id: Mapped[str] = mapped_column(String(128), nullable=False)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    at: Mapped[datetime] = _dt()
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        PrimaryKeyConstraint("stream_id", "version"),
        Index("ix_prov_ledger_memory", "org_id", "workspace_id", "memory_id"),
    )


# ============================================================ §2.7 metering
class UsageEventRow(Base):
    __tablename__ = "usage_event"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)  # per-workspace monotonic
    occurred_at: Mapped[datetime] = _dt()
    # billing/tenancy root (ADR 0026)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    to_prefix: Mapped[str] = mapped_column(String(128), nullable=False)
    visibility: Mapped[str] = mapped_column(String(128), nullable=False)
    product: Mapped[str] = mapped_column(String(128), nullable=False)
    deployment_mode: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension: Mapped[str] = mapped_column(String(128), nullable=False)  # MeterDimension
    quantity: Mapped[int] = mapped_column(BigInteger, nullable=False)
    unit: Mapped[str] = mapped_column(String(128), nullable=False)
    op: Mapped[str | None] = mapped_column(String(128))
    tier: Mapped[str | None] = mapped_column(String(128))
    source: Mapped[str | None] = mapped_column(String(128))
    model: Mapped[str | None] = mapped_column(String(128))
    role: Mapped[str | None] = mapped_column(String(128))
    engine: Mapped[str | None] = mapped_column(String(128))
    device_id: Mapped[str | None] = mapped_column(String(128))
    tokens_in: Mapped[int | None] = mapped_column(BigInteger)
    tokens_out: Mapped[int | None] = mapped_column(BigInteger)
    cached_input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    reasoning_tokens: Mapped[int | None] = mapped_column(BigInteger)
    prev_hash: Mapped[str | None] = mapped_column(String(128))  # tamper chain
    row_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    __table_args__ = (
        Index("ix_usage_ws_seq", "workspace_id", "seq"),
        Index("ix_usage_ws_occurred", "workspace_id", "occurred_at"),
        Index("ix_usage_org_occurred", "org_id", "occurred_at"),  # billing rollup at org grain
    )


class UsageOutboxRow(Base):
    __tablename__ = "usage_outbox"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)  # UsageEvent
    enqueued_at: Mapped[datetime] = _dt()
    drained_at: Mapped[datetime | None] = _dt_opt()


class UsageRollupRow(Base):
    __tablename__ = "usage_rollup"

    bucket_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    # billing/tenancy root (ADR 0026)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dimension: Mapped[str] = mapped_column(String(128), nullable=False)
    hour_bucket: Mapped[datetime] = _dt()
    quantity: Mapped[int] = mapped_column(BigInteger, nullable=False)
    qualifiers: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    source_high_water_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)


# ============================================================ §2.8 devices + private sync-log
class DeviceRow(Base):
    __tablename__ = "devices"

    device_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # the ONE owning principal
    principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    public_key: Mapped[str] = mapped_column(String(512), nullable=False)
    platform: Mapped[str] = mapped_column(String(128), nullable=False)
    label: Mapped[str] = mapped_column(
        String(255), nullable=False, default=""
    )  # user-chosen, content-free
    app_build: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    client_mode: Mapped[str] = mapped_column(String(128), nullable=False)  # thin|full_local
    privacy_tier: Mapped[str] = mapped_column(String(128), nullable=False)  # server_readable
    state: Mapped[str] = mapped_column(String(128), nullable=False, default="pending")
    enrolled_at: Mapped[datetime] = _dt()
    last_seen_at: Mapped[datetime | None] = _dt_opt()
    last_synced_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    last_lamport: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    revoked_at: Mapped[datetime | None] = _dt_opt()
    revoked_by: Mapped[str | None] = mapped_column(String(128))

    __table_args__ = (Index("ix_devices_principal", "org_id", "principal_id"),)


class PrivateSyncLogRow(Base):
    __tablename__ = "private_sync_log"

    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # sync-log key root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    principal_id: Mapped[str] = mapped_column(String(128), nullable=False)  # the per-user stream
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)  # hub-assigned ordering authority
    origin_device_id: Mapped[str] = mapped_column(String(128), nullable=False)
    op: Mapped[str] = mapped_column(String(128), nullable=False)  # upsert|supersede|tombstone|...
    memory_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    tier: Mapped[str] = mapped_column(String(128), nullable=False)
    valid_at: Mapped[datetime] = _dt()
    valid_at_inferred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    lamport: Mapped[int] = mapped_column(BigInteger, nullable=False)
    occurred_at: Mapped[datetime] = _dt()  # ADVISORY, never ordering authority
    provenance_id: Mapped[str] = mapped_column(String(128), nullable=False)
    winner_id: Mapped[str | None] = mapped_column(String(128))
    loser_id: Mapped[str | None] = mapped_column(String(128))
    payload_ref: Mapped[str | None] = mapped_column(String(512))

    __table_args__ = (
        # ADR 0026: the sync-log stream key is (org_id, principal_id); seq is the hub-assigned
        # per-stream ordering authority within it.
        PrimaryKeyConstraint("org_id", "principal_id", "seq"),
        UniqueConstraint(
            "org_id",
            "principal_id",
            "content_hash",
            "origin_device_id",
            "lamport",
            name="ux_synclog_occurrence",
        ),
        Index("ix_synclog_stream_seq", "org_id", "principal_id", "seq"),
    )


# ============================================================ §2.9 conflict_records + audit_log
class ConflictRecordRow(Base):
    __tablename__ = "conflict_records"

    conflict_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    namespace_prefix: Mapped[str] = mapped_column(String(384), nullable=False)
    member_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)  # memory ids
    predicate_key: Mapped[str] = mapped_column(String(128), nullable=False)
    method: Mapped[str] = mapped_column(String(128), nullable=False)
    detected_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    proposed_winner_id: Mapped[str | None] = mapped_column(String(128))
    state: Mapped[str] = mapped_column(String(128), nullable=False)  # ConflictState
    resolution_kind: Mapped[str | None] = mapped_column(String(128))
    resolved_winner_id: Mapped[str | None] = mapped_column(String(128))
    # auto|manual|system_degraded
    resolution_origin: Mapped[str | None] = mapped_column(String(128))
    resolved_by: Mapped[str | None] = mapped_column(String(128))
    policy_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    detected_at: Mapped[datetime] = _dt()
    resolved_at: Mapped[datetime | None] = _dt_opt()

    __table_args__ = (
        Index("ix_conflict_pending", "namespace_prefix", "state"),
        # NOTE: the `member_ids` GIN index is Postgres-only DDL — see `gin_prov_meta` note above
        # and the `event.listen(...)` registration near the bottom of this module.
    )


class AuditLogRow(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = _dt()
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    target_id: Mapped[str | None] = mapped_column(String(128))  # = MemoryItem.id BY VALUE, no FK
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_audit_ws_ts", "workspace_id", "ts"),
        Index("ix_audit_org_ts", "org_id", "ts"),
    )


# ============================================================ §2.10 trust_ledger_entries
class TrustLedgerEntryRow(Base):
    __tablename__ = "trust_ledger_entries"

    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)  # per-workspace monotonic
    action: Mapped[str] = mapped_column(String(128), nullable=False)  # TrustLedgerAction
    actor_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    subject_principal_id: Mapped[str | None] = mapped_column(String(128))
    object_ref: Mapped[str | None] = mapped_column(String(512))
    at: Mapped[datetime] = _dt()
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    prev_hash: Mapped[str | None] = mapped_column(String(128))
    # SHA-256 linked-hash chain
    entry_hash: Mapped[str] = mapped_column(String(128), nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("workspace_id", "seq"),
        Index("ix_trust_object", "workspace_id", "object_ref"),
        Index("ix_trust_principal", "workspace_id", "subject_principal_id"),
    )


# ============================================================ Postgres-only GIN DDL (docstring 2)
# MySQL 8 rejects a plain index on a raw JSON column outright (`JSON column ... cannot be used
# in key specification`) and SQLite gets no GIN benefit from one — so these two indexes are NOT
# portable `Index(...)` entries in `__table_args__` (unlike the `postgresql_where=` partial
# indexes above, which degrade gracefully everywhere). `execute_if(dialect="postgresql")` makes
# `create_all` skip them entirely on every other dialect (degrade D7, never a correctness loss).
def _postgres_only_ddl(statement: str) -> DDL:
    """SQLAlchemy's ``DDL.__init__`` ships no parameter annotations despite ``py.typed`` — one
    named ``no-untyped-call`` suppression here rather than one per call site (same unstubbed-
    boundary shape as the other suppressions in this package: pgvector_mtm.py, falkor_ltm.py)."""
    return DDL(statement)  # type: ignore[no-untyped-call]


event.listen(
    MemoryProvenance.__table__,
    "after_create",
    _postgres_only_ddl(
        "CREATE INDEX gin_prov_meta ON memory_provenance USING GIN (meta)"
    ).execute_if(dialect="postgresql"),
)
event.listen(
    ConflictRecordRow.__table__,
    "after_create",
    _postgres_only_ddl(
        "CREATE INDEX gin_conflict_members ON conflict_records USING GIN (member_ids)"
    ).execute_if(dialect="postgresql"),
)
