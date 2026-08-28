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
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT as _MYSQL_MEDIUMTEXT
from sqlalchemy.dialects.postgresql import JSONB as _PG_JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from mu_contracts.domain.model.room import MAX_DEDUPE_KEY_CHARS

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


# Extended for the principal registry (mu-server-phase3-devices-sync-spec.md §4b.2/A). Every new
# column is additive on a table with zero rows today (verified: no `Principal(` call site in any
# repo) — §4b.2/D records that NOT NULL-with-no-server-default is safe ONLY because of that, and the
# same migration's docstring repeats the warning for the next editor.
class Principal(Base):
    __tablename__ = "principals"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # pseudonymous stable id
    kind: Mapped[str] = mapped_column(String(128), nullable=False)  # human|agent|service
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (§4b.2/A)
    # operator-supplied free text — NOT content-free by construction; never an event field, span
    # attribute or log field (§4b.2/A, §4b.8/6; the `DomainEvent` field-name guard cannot see it).
    display_name: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(
        String(128), nullable=False, default="active"
    )  # active|suspended|removed (MembershipStatus) — resolution refuses non-active (§4b.3 step 6)
    created_at: Mapped[datetime] = _dt()  # transaction time (bi-temporal half)
    updated_at: Mapped[datetime] = _dt()  # advanced on every mutation
    disabled_at: Mapped[datetime | None] = _dt_opt()  # world time of suspension/removal
    # principal id, or "bootstrap" for the seeded root
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)  # bootstrap|admin|self_enroll

    __table_args__ = (
        # every legitimate LISTING is org-scoped; the resolver's PK lookup is the only un-scoped
        # read and it immediately cross-checks the credential's org_id (§4b.2/A). Deliberately no
        # UNIQUE(org_id, id): `id` is already globally unique as the PK, and a composite unique
        # would imply an id may repeat across orgs, which it may not (§4b.2/A).
        Index("ix_principals_org", "org_id"),
    )


# NEW (mu-server-phase3-devices-sync-spec.md §4b.2/B). SPEC DECISION D-43 — named
# `principal_credentials`, not `api_keys`: the field set is adopted verbatim from
# `auth-key-issuance-spec.md`'s `ApiKeyRecord` (minus the two fields §4b.1 cuts), but the first row
# this table ever holds is the existing deployment token, not an API key, and `kind` discriminates.
class PrincipalCredential(Base):
    __tablename__ = "principal_credentials"

    # public lookup handle embedded in the presented token — PK because verification must be one
    # O(1) indexed read, never a scan over hashes (§4b.2/B, D-51). Not secret.
    key_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    kind: Mapped[str] = mapped_column(String(128), nullable=False)  # api_key|deployment_token
    # denormalized onto the credential so resolution performs no join (§4b.2/B) -> AuthContext.org
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # -> AuthContext.workspace
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # the principal this credential authenticates AS. Integrity is enforced by the sole writer
    # (PrincipalService), never a database ForeignKey — D-45: the same schema.py binds to SQLite on
    # the client where FK enforcement is off by default, and a FK would make ORM insert order
    # load-bearing across two adapters in two repos. This is a decision, not an omission.
    principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # HMAC-SHA256(pepper, secret), hex
    hashed_secret: Mapped[str] = mapped_column(String(128), nullable=False)
    # a slice for kind='api_key' (non-secret by construction, D-51's arithmetic); the literal
    # constant "mu_bootstrap" for kind='deployment_token', NEVER a slice (D-52 — the asymmetry is
    # the whole point: D-50's bootstrap credential has no key_id segment, so a slice of it is 12
    # characters of a live secret).
    display_prefix: Mapped[str] = mapped_column(String(128), nullable=False)
    # operator-chosen, content-free
    label: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    created_at: Mapped[datetime] = _dt()
    # best-effort, written OUTSIDE the request's critical section, at most once per
    # auth.last_used_resolution_s (D-46) — never a per-request UPDATE.
    last_used_at: Mapped[datetime | None] = _dt_opt()
    # NULL = non-expiring; also the rotation-grace carrier
    expires_at: Mapped[datetime | None] = _dt_opt()
    revoked_at: Mapped[datetime | None] = _dt_opt()  # invalidate-don't-delete
    revoked_by: Mapped[str | None] = mapped_column(String(128))
    rotated_from_key_id: Mapped[str | None] = mapped_column(String(128))  # rotation lineage
    # principal id, or "bootstrap" for the seeded root
    issued_by: Mapped[str] = mapped_column(String(128), nullable=False)

    __table_args__ = (
        # the only listing query is "the credentials of this principal"; org_id-first means it
        # cannot be answered cross-tenant even by a caller who gets the argument order wrong.
        Index("ix_principal_credentials_principal", "org_id", "principal_id"),
        # D-44: two rows sharing a hash is either a CSPRNG failure or a reused pepper across
        # environments — either is a security incident, and this makes it fail loud at issue
        # instead of silently authenticating two principals with one string. Deliberately no
        # unique on (org_id, principal_id): a principal legitimately holds several credentials at
        # once (that overlap IS rotation).
        UniqueConstraint("hashed_secret", name="ux_principal_credentials_secret"),
    )


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
    # read|write|delete|share — WHAT this row grants (AD-67, governance-transfer-core-spec:201).
    # Without it the standing ACL grid cannot answer "does this subject hold READ on this object"
    # by row lookup, and the read path would have to fold the grant chain — destroying the one
    # property that justifies this table existing beside `grants` (spec:204).
    permission: Mapped[str] = mapped_column(String(128), nullable=False)
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
    """THE append-only provenance ledger — one table, two planes (AD-106).

    ⚠ **This shape is a UNION, and the AD-106 half that produced it is deliberately visible.**
    Until revision ``a71f3c9de205`` there were TWO provenance tables for one concept: this one
    (the memory-lineage ledger, `storage-schema-rowmapper-spec.md` §2.6) and
    ``transfer_provenance`` (``mu-server/src/mu_server/transfer/store_pg.py``), stood up because
    the shipped ``ProvenanceAction`` had four members and the transfer FSM needed eight, so half
    the governance plane's rows could not be written here at all (AD-62). AD-62 landed the eight,
    which removed the reason the fork existed. ``governance-transfer-core-spec.md`` §4 and this
    spec's §2.6 both describe ONE append-only ledger per object stream, so the fork is a
    divergence, not a design.

    The columns below the lineage block are what a transfer event needs and a memory-lineage
    append does not, which is why every one of them is NULLABLE. Two mappings are worth stating
    because a reader looking for the transfer plane's own column names will not find them:
    ``transfer_provenance.actor_principal_id`` IS :attr:`actor_id` and
    ``transfer_provenance.occurred_at`` IS :attr:`at` — the same fact under this table's name.
    Duplicating them would have produced two "who acted" columns that can disagree.

    **What identifies the subject of a row.** A memory-lineage append sets :attr:`memory_id`; a
    transfer event sets ``(object_type, object_id)``, whose object may be a ``ContextIndex``, a
    ``ContextPacket`` or a grant — none of which is a memory. That is why :attr:`memory_id`
    became nullable here. The invariant "exactly one of them identifies the subject" is enforced
    at the pydantic boundary, not by a DDL ``CHECK``, for the same reason :attr:`action` carries
    no ``CHECK`` (see below): a constraint here turns every vocabulary addition into a migration
    on a table whose values are already validated before they arrive.

    **``position`` is Postgres-only, and that is a stated scope limit rather than a degrade.** It
    is the org-scoped total order the ledger scan (``read_all(org_id, from_position)``) walks, and
    it is DB-assigned from a sequence — the shape ``transfer_provenance`` already had, kept so the
    governance adapter's INSERT does not have to change when it re-points here. MySQL cannot give
    a non-key column an ``AUTO_INCREMENT``, so on any dialect but Postgres the column exists and
    stays NULL. The lineage half of this table is fully portable exactly as before; the governance
    plane is Postgres-only by deployment (``mu_server.control_plane.migrate`` builds its Alembic
    config from ``PostgresSettings``), so nothing that can run today is left without an order.
    Unlike the D7 index degrade this is not "the same answer, slower" — a NULL ``position`` has no
    scan order — which is why it is written as a scope limit and not filed as a degrade.
    """

    __tablename__ = "provenance_ledger"

    # ---- lineage core (spec §2.6, unchanged in meaning) ------------------------------------
    #: 512 rather than 128: the transfer plane's stream ids are ``object_ref_key``-derived
    #: composites, and its own table sized this column at 512 for that reason.
    stream_id: Mapped[str] = mapped_column(String(512), nullable=False)  # provenance_id
    #: BigInteger, not Integer: an append-only stream has no reason to stop at 2**31.
    version: Mapped[int] = mapped_column(BigInteger, nullable=False)  # monotonic within stream
    # A `ProvenanceAction` VALUE (mu_contracts.domain.model.governance): origin|composed|shared|
    # pulled|reshared|accepted|revoked|superseded, plus the deprecated `derived` (AD-62). This is
    # a plain VARCHAR with no CHECK constraint and no database ENUM type — deliberately, and it is
    # why extending the enum from four members to eight needed **no migration at all**: the value
    # set is enforced by pydantic at the boundary, not by the DDL, so widening it produces zero
    # schema delta. Keep it that way; a CHECK here would make every vocabulary addition a
    # migration on a table whose values are already validated before they arrive.
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    #: NULL on a governance event, whose subject is ``(object_type, object_id)`` — see the class
    #: docstring. Never NULL on a memory-lineage append.
    memory_id: Mapped[str | None] = mapped_column(String(128))
    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    at: Mapped[datetime] = _dt()
    meta: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # ---- append identity + ledger order ------------------------------------------------------
    #: The caller-supplied idempotency key. UNIQUE at the DATABASE, because an append-only ledger
    #: whose duplicate-suppression lives only in application code is one retry away from a
    #: double-counted history.
    event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    #: Org-scoped total order for the ledger scan. Postgres-only — see the class docstring.
    position: Mapped[int | None] = mapped_column(BigInteger)

    # ---- governance/transfer event fields (NULL on a memory-lineage append) ------------------
    object_type: Mapped[str | None] = mapped_column(String(32))  # ShareableType value
    object_id: Mapped[str | None] = mapped_column(String(128))
    object_content_hash: Mapped[str | None] = mapped_column(String(128))
    #: 384 to match this schema's own ``namespace_prefix`` width, which is strictly wider than the
    #: 256 ``transfer_provenance`` used — so no existing governance row can fail to fit.
    origin_namespace_id: Mapped[str | None] = mapped_column(String(384))
    grantor_principal_id: Mapped[str | None] = mapped_column(String(128))
    grantee_kind: Mapped[str | None] = mapped_column(String(32))  # PrincipalRefKind value
    grantee_id: Mapped[str | None] = mapped_column(String(128))
    grant_id: Mapped[str | None] = mapped_column(String(128))
    packet_id: Mapped[str | None] = mapped_column(String(128))
    source_refs: Mapped[list[Any] | None] = mapped_column(JSONB)
    content_hash: Mapped[str | None] = mapped_column(String(128))
    cascade_root_grant_id: Mapped[str | None] = mapped_column(String(128))

    __table_args__ = (
        # ``org_id`` joined the PK in ``a71f3c9de205``. Without it two orgs whose streams share an
        # id collide on one history — the tenancy defect ``transfer_provenance`` had already
        # avoided by keying ``(org_id, stream_id, version)``, and which this table would have
        # inherited the moment a second plane wrote to it.
        PrimaryKeyConstraint("org_id", "stream_id", "version"),
        UniqueConstraint("event_id", name="ux_prov_ledger_event"),
        Index("ix_prov_ledger_memory", "org_id", "workspace_id", "memory_id"),
        Index("ix_prov_ledger_position", "org_id", "position"),
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
    # §7.17 item 4a — the two leading (dominant) total-order terms, additive (CANONICAL:777
    # "every field is on the delta itself"; ADR 0046/CANONICAL §7.17 item 4a). Mirrors
    # PrivateDelta.pinned / PrivateDelta.resolution_origin (mu_contracts.domain.model.device_sync).
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # auto|manual|system_degraded
    resolution_origin: Mapped[str | None] = mapped_column(String(128))
    # The remaining three columns build step 2c named and revision 1ab7f7175baa did not carry.
    # Each is additive and nullable-safe; none changes an existing constraint.
    #
    # `resolved_by` — CANONICAL:538's attribution half of a manual resolution, mirroring
    # PrivateDelta.resolved_by. Without the column a manually-resolved SUPERSEDE round-trips
    # through the log having forgotten who resolved it.
    resolved_by: Mapped[str | None] = mapped_column(String(128))
    # `caused_by_seq` — the loop-suppression marker, already a shipped field on the WIRE delta
    # (`PrivateDelta.caused_by_seq`, "replica-apply echo of log seq N; projector DROPS it"). With
    # no column the field silently vanishes on the way through the log, so a replica-apply echo
    # reads back indistinguishable from an original write — which is the one distinction appender
    # B's loop suppression is built on.
    caused_by_seq: Mapped[int | None] = mapped_column(BigInteger)
    # `appended_at` — the HUB's receive time, server-set on every append and never client-supplied.
    # It is the retention window's only possible basis: `SyncSettings.private_sync_log_retention_s`
    # is a duration, and with no timestamp on the row a pruner (O-32) cannot be written at all.
    # No reader today, by construction — that is O-32, and this closes one of its two blockers.
    #
    # ⚠ The ``server_default`` mirrors revision ``9c41d0b7ae52``, which needed one to add a NOT
    # NULL column to a possibly-populated table. Declared here too so the ORM model and the
    # migrated database do not DRIFT: without it the next ``alembic revision --autogenerate``
    # compares model-without-default against database-with-default and proposes DROPPING the
    # default, which would make the very next additive revision unsafe on a live table. The
    # adapter still sets the value explicitly on every insert, so this is a migration device and
    # never a second writer.
    appended_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )

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


# ============================================================ §2.x ROOMS — the S3 room runtime
# ``rooms-sessions-subscriptions-spec.md`` §2.2/§3.2, ``ARCHITECTURE-DELTAS.md`` AD-28 items (1)
# and (2)/(3). Three tables, added together by revision ``b3d47c9a1e02``, because they are one
# aggregate: ``room_session`` + ``room_participant`` are the roster half (``SessionRepository``)
# and ``room_log`` is the append-only ordering authority (``RoomLogRepository``).
#
# **Why NEW tables rather than widening the shipped ``sessions``/``session_participants``.** AD-28
# item (3) is phrased as missing COLUMNS on those two tables, and that phrasing is not what was
# built — deliberately. ``sessions`` is the control-plane home of ``SessionDefinition``, whose
# ``SessionState`` has exactly ``{OPEN, CLOSED}``; the room's ``RoomState`` also has ``PAUSED``,
# and ``Session.to_definition()`` projects ``PAUSED`` *down* to ``OPEN`` on purpose (a store
# outage is not an end-of-session for authorization or billing). Carrying two different state
# axes on one row would make that projection lossy in the store, and would widen a shipped table
# that other components already read. The room tables are a different concept on a different
# axis, so they get their own rows. This is also the shape the only existing implementation
# assumes — ``mu_server.rooms.session_pg.PostgresSessionRepository`` and its ``REQUIRED_DDL``
# target ``room_session``/``room_participant`` verbatim, and ``mu_server.rooms.room_log_pg``
# targets ``room_log`` — so the migration and the adapters cannot silently disagree.
#
# **Tenancy** (CANONICAL §1 rule 5): every primary key here BEGINS at ``(org_id, workspace_id)``.
# A room id is never a key on its own, so a cross-tenant read is not a filter bug away.
#
# **Content-free** (CANONICAL §3.1): ``room_log.body`` is the ONE column in this schema that holds
# message text, and it is legitimate — ``RoomMessage`` is the STORED object and the discipline
# binds the bus/logs/traces/metering, not the store. Everything else here is ids, hashes, enum
# values, counts and timestamps. ``room_participant.display_name`` is principal-supplied identity,
# not memory content.
class RoomSessionRow(Base):
    """The room aggregate's durable row — ``Session`` minus the roster (spec:79-82)."""

    __tablename__ = "room_session"

    org_id: Mapped[str] = mapped_column(String(128), nullable=False)  # tenancy root (ADR 0026)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    id: Mapped[str] = mapped_column(String(128), nullable=False)  # == room_id == session_id
    namespace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    visibility: Mapped[str] = mapped_column(String(32), nullable=False)  # always 'shared'
    state: Mapped[str] = mapped_column(String(32), nullable=False)  # open|paused|closed
    floor_policy: Mapped[str] = mapped_column(String(32), nullable=False)
    #: A HINT (spec:82) — ``room_log`` is the ordering authority. Stored so a room reloaded after a
    #: crash does not have to scan the log to answer "roughly where were we".
    last_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: The bus-announce cursor: ``seq > announced_through_seq`` is a durable statement that the
    #: ``RoomMessagePosted`` announce is still OWED, which is what makes a crash between append and
    #: publish recoverable instead of a permanent interior hole in the bus stream.
    announced_through_seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: Optimistic-concurrency guard for spec:132's whole-aggregate save. ``UPDATE ... WHERE
    #: version = :expected``; zero rows updated is a concurrent commit, not a missing room.
    version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = _dt()

    __table_args__ = (PrimaryKeyConstraint("org_id", "workspace_id", "id", name="pk_room_session"),)


class RoomParticipantRow(Base):
    """One roster row (spec:74). Rewritten wholesale inside the aggregate's save transaction."""

    __tablename__ = "room_participant"

    org_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    room_id: Mapped[str] = mapped_column(String(128), nullable=False)
    principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(200))
    joined_at: Mapped[datetime] = _dt()
    #: Set = the offboarding re-stamp trigger (CANONICAL §7.4/M15). The row is KEPT so the
    #: provenance of every message the principal authored survives the departure.
    left_at: Mapped[datetime | None] = _dt_opt()
    presence: Mapped[str] = mapped_column(String(32), nullable=False)
    owner_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    #: Authorizes a BOUND_AGENT (``scope.py:186`` ``AgentBinding``); ``None`` for everyone else.
    binding_id: Mapped[str | None] = mapped_column(String(128))
    #: ``frozenset[str]``, JSON-encoded. TEXT rather than JSON because MySQL 8 cannot index or
    #: default a raw JSON column and nothing here queries INTO the value — it is read back whole.
    capabilities: Mapped[str] = mapped_column(Text, nullable=False)
    #: **Roster ORDER, and it is load-bearing.** ``Session.join`` appends, and the bootstrap rule
    #: that admits the first member reads "a room with NO participant rows"; without an explicit
    #: ordinal the roster comes back in whatever order the planner chose and the aggregate is
    #: reconstructed differently on every load.
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint(
            "org_id", "workspace_id", "room_id", "principal_id", name="pk_room_participant"
        ),
        ForeignKeyConstraint(
            ["org_id", "workspace_id", "room_id"],
            ["room_session.org_id", "room_session.workspace_id", "room_session.id"],
            name="fk_room_participant_session",
            ondelete="CASCADE",
        ),
    )


class RoomLogRow(Base):
    """The append-only per-room event store — **and the sole ``seq`` ordering authority**.

    ``seq`` is not a counter: a client applies a frame only if ``seq == last_applied + 1``
    (CONTIGUITY, CANONICAL:566), so the sequence must be gap-free and duplicate-free per room. The
    primary key is what makes "duplicate-free" a property of the DATABASE rather than of the
    appender's care, and ``ux_roomlog_dedupe`` is what makes an idempotent replay resolvable in
    one statement (``ON CONFLICT ON CONSTRAINT ux_roomlog_dedupe DO NOTHING``).
    """

    __tablename__ = "room_log"

    org_id: Mapped[str] = mapped_column(String(128), nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    room_id: Mapped[str] = mapped_column(String(128), nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: spec:76's ``msg_<uuid>`` surrogate. **NULLABLE, and the reason is worth stating:** the
    #: shipped appender (``mu_server.rooms.room_log_pg._APPEND_SQL``) is a fixed thirteen-column
    #: INSERT that does not mention this column and lives in a repo this lane may not edit, so a
    #: ``NOT NULL`` here would make every existing append fail at the database. There is no
    #: portable server default either — ``gen_random_uuid()`` is Postgres-only and this schema
    #: also binds SQLite and MySQL. NULL therefore means "written by an appender that predates
    #: the column", and the durable identity remains the primary key below.
    id: Mapped[str | None] = mapped_column(String(160))
    author_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    author_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    #: Width is :data:`mu_contracts.domain.model.room.MAX_DEDUPE_KEY_CHARS`, IMPORTED rather than
    #: retyped: the REST edge bounds the ``Idempotency-Key`` header to the same number, and when
    #: the two drifted an over-long header reached this column, Postgres answered ``22001``, and a
    #: pure client-input error was re-labelled a store outage that PAUSED a room.
    dedupe_key: Mapped[str] = mapped_column(String(MAX_DEDUPE_KEY_CHARS), nullable=False)
    #: **``Addressing.to_principal_ids`` — AD-28 item (1), the whole point of the delta.** Before
    #: this column a DIRECTED post had nowhere durable to record who it was for: the recipients
    #: were accepted by the service and then lost. JSON-encoded list of principal ids (ids, not
    #: content). NULLABLE with NULL ≡ ``()`` ≡ BROADCAST — which is a real domain value, not a
    #: missing-data sentinel (spec:78: "empty = broadcast"), so nullability costs no information
    #: here. It also has to be nullable: the thirteen-column appender above cannot write it, and a
    #: TEXT column cannot carry a literal DEFAULT on MySQL.
    to_principal_ids: Mapped[str | None] = mapped_column(Text)
    #: ``Addressing.reply_to_seq`` — the thread pointer, and one of spec:77's four dedupe inputs.
    #: NULL = not a reply.
    reply_to_seq: Mapped[int | None] = mapped_column(BigInteger)
    #: The one column in this schema that holds message text, and legitimately so (CANONICAL §3.1
    #: binds the bus, not the store). ``MEDIUMTEXT`` on MySQL because :data:`MAX_BODY_CHARS` is
    #: 32 000 CHARACTERS and MySQL's ``TEXT`` holds 65 535 BYTES — a body of multi-byte text would
    #: be silently truncated at the ~21 800-character mark on that dialect.
    body: Mapped[str] = mapped_column(
        Text().with_variant(_MYSQL_MEDIUMTEXT, "mysql"), nullable=False
    )
    #: When the author posted it (``RoomMessage.posted_at``). NOT ``created_at``: ``room_session``
    #: one table up already means "when the room was opened" by that name — see the name-drift
    #: ruling in ``mu_contracts/domain/model/room.py``'s module docstring.
    posted_at: Mapped[datetime] = _dt()
    #: When the LOG committed it. Distinct from ``posted_at`` on purpose: the gap between them is
    #: the append latency, and only the second one is monotonic with ``seq``.
    appended_at: Mapped[datetime] = _dt()

    __table_args__ = (
        PrimaryKeyConstraint("org_id", "workspace_id", "room_id", "seq", name="pk_room_log"),
        # Named LITERALLY because `ON CONFLICT ON CONSTRAINT ux_roomlog_dedupe` names it: a
        # rename here turns the idempotent-replay branch into an unhandled IntegrityError on the
        # first client retry in production. The backfill read (`after_seq` + ORDER BY seq) is
        # served by the primary key's leading columns and needs no index of its own.
        UniqueConstraint(
            "org_id", "workspace_id", "room_id", "dedupe_key", name="ux_roomlog_dedupe"
        ),
        # The surrogate id is an IDENTITY, so it is unique per room where it is present at all.
        # Multi-NULL is permitted by every dialect here, which is what lets the pre-column rows
        # above coexist with written ones.
        UniqueConstraint("org_id", "workspace_id", "room_id", "id", name="ux_roomlog_message_id"),
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
