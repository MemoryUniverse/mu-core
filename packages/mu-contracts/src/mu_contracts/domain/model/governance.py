"""Governance vocabulary — ShareableRef, Grant (the moat), RBAC, ComposedContext.

Authority: governance-transfer-core-spec.md §1-§6 (ported verbatim in shape; RBAC ported
from Cognee ``permission_types.py:1`` / ``Role.py:7-27`` / ``UserRole.py:6-13``). All models
are pydantic v2, frozen value objects. NO field ever holds raw memory content (CANONICAL §3):
ids, hashes, refs, enums, counts, timestamps, principal ids only.

``ComposedContext`` (the third governed object, §6) is co-located here because it is expressed
in terms of ``ShareableRef``. Its BODY is a versioned ``ContextArtifact`` (``body_ref``); the
snapshot never changes — only its freshness marker (CANONICAL §7.10 G8).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AclEntry",
    "ComposedContext",
    "Direction",
    "Grant",
    "GrantState",
    "Permission",
    "PrincipalRef",
    "PrincipalRefKind",
    "ProvenanceAction",
    "Revocation",
    "Role",
    "RoleMembership",
    "ShareableRef",
    "ShareableType",
    "TransferClass",
    "TransferState",
]


class ShareableType(StrEnum):
    """The three governed object types."""

    MEMORY = "memory"  # a MemoryItem / MemoryNode (proposition or reference)
    ARTIFACT = "artifact"  # a ContextArtifact (transcript/file/code/url/image)
    COMPOSED = "composed"  # a ComposedContext (generated bundle, §6)


class Permission(StrEnum):
    """Ported verbatim: cognee ``permission_types.py:1``."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"


class Direction(StrEnum):
    """Grant/transfer direction across the two planes."""

    PUBLISH = "publish"  # local -> shared
    PULL = "pull"  # shared -> local (one-shot)
    SUBSCRIBE = "subscribe"  # shared -> local (standing)


class PrincipalRefKind(StrEnum):
    """The asymmetric target kind of a grant (exploded to member principals at accept time)."""

    PRINCIPAL = "principal"  # a single human/agent/service
    ROLE = "role"  # a named workspace/org role (all members)
    SESSION = "session"  # every participant of a shared session/room


class GrantState(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"  # severed; provenance retained (invalidate-don't-delete)
    EXPIRED = "expired"
    SUPERSEDED = "superseded"  # replaced by a re-issued grant (idempotent content change)


class TransferClass(StrEnum):
    """Boundary class of a transfer (BoundaryGuard invariant)."""

    PRIVATE = "private"  # NEVER transits the shared plane
    WORKSPACE_SHARED = "workspace_shared"  # visible within the origin workspace
    ORG_SHARED = "org_shared"  # across workspaces of one org (needs cross_workspace_sharing)
    PUBLIC = "public"  # org-external (needs cross_org_sharing + federation link)


class TransferState(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    DELIVERED = "delivered"
    ACCEPTED = "accepted"  # terminal-success; grant+provenance+stamp happen here
    REVOKED = "revoked"  # terminal
    EXPIRED = "expired"  # terminal


class ProvenanceAction(StrEnum):
    """Append-only lineage actions on the provenance ledger (storage-schema §2.6)."""

    ORIGIN = "origin"
    DERIVED = "derived"
    SUPERSEDED = "superseded"
    COMPOSED = "composed"


class ShareableRef(BaseModel):
    """Stable, content-addressed handle to any of the three governed objects. Content-free —
    crosses the bus freely (CANONICAL §3). ``org_id`` added by the un-collapse (§0.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    object_type: ShareableType
    object_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)  # ties a grant to an EXACT version
    org_id: str = Field(min_length=1)  # tenant/billing/residency root
    workspace_id: str = Field(min_length=1)
    origin_namespace_id: str = Field(min_length=1)  # the .session / logical origin partition

    def stream_id(self) -> str:
        """ProvenanceLedger stream key (storage-schema §2.6; includes org_id post-un-collapse)."""
        return f"prov:{self.org_id}:{self.object_type.value}:{self.object_id}"

    def canonical(self) -> str:
        """Deterministic digest input (trust-ledger §4.1)."""
        return "|".join(
            (
                self.object_type.value,
                self.object_id,
                self.content_hash,
                self.org_id,
                self.workspace_id,
                self.origin_namespace_id,
            )
        )


class PrincipalRef(BaseModel):
    """Asymmetric target of a grant: a principal, a role, or a whole session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: PrincipalRefKind
    id: str = Field(min_length=1)  # principal_id | role_id | session_id
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)


class Revocation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    revoked_by: str = Field(min_length=1)
    revoked_at: datetime
    reason: str | None = Field(default=None, max_length=500)  # named reason; NEVER content
    cascade_root_grant_id: str = Field(min_length=1)


class Grant(BaseModel):
    """The asymmetric, revocable, chainable grant aggregate root (governance-transfer §2).

    Idempotency: ``id = grant_<sha256(object_ref.content_hash | grantor | grantee | packet_id)>``;
    ``GrantRepository.add`` is put-if-absent. A ``Grant`` + its derivation subtree is the unit the
    revoke cascade traverses; the aggregate holds only ``parent_grant_id``/``depth``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    object_ref: ShareableRef
    grantor_principal_id: str = Field(min_length=1)
    grantee: PrincipalRef  # PRINCIPAL | ROLE | SESSION
    direction: Direction
    permissions: frozenset[Permission] = Field(min_length=1)
    policy_id: str = Field(min_length=1)  # the TransferPolicy this grant bound to
    packet_id: str | None = None  # ContextPacket that carried it (transport)
    parent_grant_id: str | None = None  # set on re-share; None for a root grant
    depth: int = Field(default=0, ge=0)  # re-share depth (0 = original)
    state: GrantState = GrantState.ACTIVE
    revocation: Revocation | None = None
    issued_at: datetime
    expires_at: datetime | None = None
    provenance_id: str = Field(min_length=1)  # first ledger event id for this grant

    def is_terminal(self) -> bool:
        return self.state in (GrantState.REVOKED, GrantState.EXPIRED)

    def can(self, permission: Permission, *, at: datetime | None = None) -> bool:
        return self.is_active(at=at) and permission in self.permissions

    def is_active(self, *, at: datetime | None = None) -> bool:
        if self.state is not GrantState.ACTIVE:
            return False
        if at is not None and self.expires_at is not None and self.expires_at <= at:
            return False
        return True


class Role(BaseModel):
    """Named permission-bearing principal, scoped to an org, optionally a workspace.
    Ported: cognee ``Role.py:7-27`` (unique ``(org_id, workspace_id, name)``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)  # was Cognee tenant_id (§0.1)
    workspace_id: str | None = None  # None = org-wide role; set = workspace-scoped
    name: str = Field(min_length=1)


class RoleMembership(BaseModel):
    """Ported: cognee ``UserRole.py:6-13`` (user-to-role join)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(min_length=1)
    role_id: str = Field(min_length=1)
    granted_at: datetime


class AclEntry(BaseModel):
    """The materialized non-session ACL row (CANONICAL §8-M9). Materialized to a concrete
    exploded-PRINCIPAL row on accept so revoke/offboarding have rows to mark and reads never
    fold the grant chain. ``object_ref`` references a ``ShareableRef`` id by value (no FK)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    object_ref: str = Field(min_length=1)  # ShareableRef id (by value, storage-models §3.4)
    subject_principal_id: str = Field(min_length=1)  # exploded PRINCIPAL id (never a role/session)
    grantee_kind: PrincipalRefKind  # user|session|role (exploded to members at accept)
    grant_id: str = Field(min_length=1)
    created_at: datetime
    revoked_at: datetime | None = None  # NULL = live


class ComposedContext(BaseModel):
    """An immutable generated bundle over sources (governance-transfer §6). Body is a versioned
    ``ContextArtifact`` (``body_ref``); the snapshot never changes — only its freshness marker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    namespace_id: str = Field(min_length=1)
    source_memory_ids: tuple[str, ...] = ()
    source_artifact_ids: tuple[str, ...] = ()
    intent: str = Field(min_length=1)  # short label, NOT the generated body
    body_ref: str = Field(min_length=1)  # by-id handle to the versioned ContextArtifact
    content_hash: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)
    created_at: datetime

    def as_ref(self) -> ShareableRef:
        return ShareableRef(
            object_type=ShareableType.COMPOSED,
            object_id=self.id,
            content_hash=self.content_hash,
            org_id=self.org_id,
            workspace_id=self.workspace_id,
            origin_namespace_id=self.namespace_id,
        )
