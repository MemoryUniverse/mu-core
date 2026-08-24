"""ClientScope / Principal / AgentBinding / membership split — the acting identity.

Authority: platform-layer0-spec.md §3 (the un-collapse adds ``org_id`` + splits membership
into ``OrgMembership`` + ``WorkspaceMembership`` + carries the never-None acting agent).
CANONICAL §1 rule 2 (agent is NOT an η field; it lives here) and rule 8.

The active request identity is resolved by the transport BEFORE any repository call and is
immutable per request. ``ClientScope.namespace()`` emits the un-collapsed η
(``memory.Namespace``). The acting-agent identity lives here, never on η.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.memory import Namespace, Visibility

__all__ = [
    "AgentBinding",
    "AgentKind",
    "ClientScope",
    "MembershipStatus",
    "OrgMembership",
    "Principal",
    "PrincipalKind",
    "PrincipalRecord",
    "PrincipalSource",
    "ResolvedPrincipal",
    "WorkspaceMembership",
]


class PrincipalKind(StrEnum):
    """A principal is a human, an agent, or a service (agents are principals, never seats)."""

    HUMAN = "human"
    AGENT = "agent"
    SERVICE = "service"


class AgentKind(StrEnum):
    """The acting-agent classification. ``HUMAN_PROXY`` is a human acting directly:
    ``agent_principal_id == principal_id`` — the SAME code path as an agent write
    (platform-layer0 §3; agent-subagent-identity-design §5.2)."""

    HUMAN_PROXY = "human_proxy"
    SHARED_AGENT = "shared_agent"
    BOUND_AGENT = "bound_agent"
    SUBAGENT = "subagent"


class MembershipStatus(StrEnum):
    """Invalidate-don't-delete: a removed membership is marked, not dropped (offboarding
    re-stamp trigger, CANONICAL §7.4/M15)."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    REMOVED = "removed"


class Principal(BaseModel):
    """A human OR an agent OR a service — all principals. ``id`` goes into η.user and into
    ``authorized_ids`` (platform-layer0 §3; ported spine mma/hackathon scope.py:52)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)  # principal id (human user id, or agent_principal_id)
    org_id: str = Field(min_length=1)  # tenant/billing root (un-collapse)
    kind: PrincipalKind = PrincipalKind.HUMAN
    display_name: str | None = Field(default=None, max_length=200)


class PrincipalSource(StrEnum):
    """How a principal row came to exist — a closed, content-free set
    (mu-server-phase3-devices-sync-spec.md §4b.2/A). It is the field that lets an operator tell a
    SEEDED identity from a minted one without reading ``created_by``."""

    BOOTSTRAP = "bootstrap"
    ADMIN = "admin"
    SELF_ENROLL = "self_enroll"


class PrincipalRecord(BaseModel):
    """The registry ROW as the resolver reads it back — ``principals``
    (mu-server-phase3-devices-sync-spec.md §4b.2/A).

    Here rather than in the commercial plane by **ADR 0047**: a type that only *names* a concept is
    vocabulary, and vocabulary lives in open ``mu-core`` whichever plane's work motivates it. The
    *resolver* (``PostgresPrincipalRegistry``) stays in ``mu-server`` — that is O-44's closure, and
    the split is the point: the shape of the record is a contract, reading it is behaviour.

    ⚠ **``PrincipalRecord`` is deliberately NOT ``Principal``, and the overlap is not a bug.**
    §4b.2/A: *"a DTO is the wire vocabulary, a row is the record"*. They agree on four fields;
    ``created_at``/``updated_at``/``disabled_at``/``created_by``/``source`` belong to the record
    ONLY and must never be added to ``Principal``.

    ⚠ ``display_name`` is the one field here that is **not content-free by construction** — it is
    operator-supplied free text. It is therefore never an event field, never a span attribute and
    never a log field (§4b.2/A, §4b.8/6: the ``DomainEvent`` field-name guard cannot see this class
    of leak, so the discipline is on the reader, not on a guard).

    ``status`` and ``disabled_at`` are a **bi-temporal pair**: ``status`` carries the transaction-
    time fact, ``disabled_at`` the world time of the suspension/removal. Either alone is a
    single-axis row. Rows are retained on removal (invalidate-don't-delete) because the id is
    stamped into ``authorized_ids`` and into ``private_sync_log.principal_id`` rows that must stay
    attributable for audit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)  # pseudonymous stable id; goes into η.user and authorized_ids
    org_id: str = Field(min_length=1)  # tenancy root — cross-checked against the credential's
    kind: PrincipalKind = PrincipalKind.HUMAN
    display_name: str | None = Field(default=None, max_length=200)  # ⚠ free text; never emitted
    status: MembershipStatus = MembershipStatus.ACTIVE
    created_at: datetime  # transaction time
    updated_at: datetime  # advanced on every mutation
    disabled_at: datetime | None = None  # world time of the suspension/removal
    created_by: str = Field(min_length=1)  # provenance: a principal id, or the literal "bootstrap"
    source: PrincipalSource

    def is_resolvable(self) -> bool:
        """The principal-side half of §4b.3 step 6. A suspended/removed principal is REFUSED, and
        the refusal is a status check rather than a missing row — which is why every caller must
        collapse it into the same non-enumerating failure as "no such row" (§4b.3, harm 4)."""
        return self.status is MembershipStatus.ACTIVE


class ResolvedPrincipal(BaseModel):
    """What a **verified** credential resolves to — the return of
    ``PrincipalRegistryPort.resolve_credential`` (mu-server-phase3-devices-sync-spec.md §4b.3
    step 7, §4b.4). Vocabulary, so it lives here (ADR 0047); the resolver does not.

    These three fields are exactly the coordinates the edge needs downstream and no more:
    ``principal_id`` becomes ``AuthContext.principal_id``, ``org_id`` becomes ``AuthContext.org``,
    ``workspace_id`` becomes ``AuthContext.workspace`` — each **replacing** a process-wide settings
    default that a multi-tenant plane must never read for a request's tenancy (mu-server invariant
    2: a caller that can name its own org can name someone else's, and a plane that reads one
    default for every caller routes every tenant to the first tenant's home region).

    ⚠ It carries **no credential material** — no secret, no ``hashed_secret``, no ``key_id``. A
    resolved identity travels through the admission path; the thing that proved it does not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)  # -> AuthContext.org
    workspace_id: str = Field(min_length=1)  # -> AuthContext.workspace


class OrgMembership(BaseModel):
    """Billing/tenant-root membership (the un-collapse split; platform-layer0 §3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    role: str = Field(default="member", min_length=1)
    status: MembershipStatus = MembershipStatus.ACTIVE
    expires_at: datetime | None = None

    def is_active(self, *, at: datetime | None = None) -> bool:
        now = at or datetime.now(UTC)
        expiry = self.expires_at
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return self.status is MembershipStatus.ACTIVE and (expiry is None or expiry > now)


class WorkspaceMembership(BaseModel):
    """Per-workspace membership (the un-collapse split; realizes ``workspace_memberships``,
    storage-schema §2.1; ported mma/hackathon scope.py:60)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    role: str = Field(default="member", min_length=1)
    status: MembershipStatus = MembershipStatus.ACTIVE
    expires_at: datetime | None = None

    def is_active(self, *, at: datetime | None = None) -> bool:
        now = at or datetime.now(UTC)
        expiry = self.expires_at
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        return self.status is MembershipStatus.ACTIVE and (expiry is None or expiry > now)


class AgentBinding(BaseModel):
    """A bound agent's capability cap (tool + scope). ``assert_allows`` gates every
    bound-agent action, making "bound agents act under the member's identity, see only the
    room's shared context" enforceable (platform-layer0 §3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    binding_id: str = Field(min_length=1)
    agent_principal_id: str = Field(min_length=1)
    owner_principal_id: str = Field(min_length=1)  # parent agent, or the root human
    allowed_tools: frozenset[str] = frozenset()

    def assert_allows(self, tool: str, target: Namespace) -> None:
        """Raise if ``tool`` is outside the cap or ``target`` is outside the binding's scope.
        Concrete policy is applied by the governance layer; the cap check is structural here."""
        del target  # scope check is layered in governance; the tool cap is enforced here
        if self.allowed_tools and tool not in self.allowed_tools:
            from mu_contracts.domain.errors import AuthorizationError

            raise AuthorizationError(f"tool not permitted by binding: {tool!r}")


class ClientScope(BaseModel):
    """The active ``(principal, org, workspace, session)`` + the acting agent. Resolved by the
    transport before any repository call; immutable per request (platform-layer0 §3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = Field(min_length=1)  # the OWNER principal (human, or the agent's owner)
    org_id: str = Field(min_length=1)  # η.org
    workspace_id: str = Field(min_length=1)  # η.workspace
    session_id: str = Field(min_length=1)  # η.session
    # the ACTING agent; == principal_id for HUMAN_PROXY (NEVER None).
    agent_principal_id: str = Field(min_length=1)
    agent_kind: AgentKind = AgentKind.HUMAN_PROXY
    agent_path: str = ""  # resolved agent path (cheap; cached by agent_principal_id)

    def namespace(self, visibility: Visibility) -> Namespace:
        """Derive the storage η. SHARED is session-scoped (user='*'), not principal-scoped."""
        if visibility is Visibility.SHARED:
            return Namespace.shared(
                org=self.org_id, workspace=self.workspace_id, session=self.session_id
            )
        return Namespace(
            org=self.org_id,
            workspace=self.workspace_id,
            user=self.agent_principal_id,
            session=self.session_id,
            visibility=visibility,
        )

    def assert_authorized(self, target: Namespace, operation: str) -> None:
        """Reject a cross-org / cross-workspace target — the belt to ``TenancyGuard``'s suspenders
        (platform-layer0 §3/§5). A cross-session target is rejected too UNLESS ``target`` is
        PRIVATE: for PRIVATE, ``session`` is a recall FILTER + provenance stamp, never an
        isolation boundary (ADR 0030 "keep-and-scope" — the federate-live, session_scope=None
        default deliberately surfaces the SAME user's OTHER sessions); the same-user invariant for
        PRIVATE is independently enforced by ``DefaultTenancyGuard.assert_scope``'s user-slot check
        that runs right after this. A SHARED (room) target's session stays a hard wall — rooms are
        real walls, ADR 0030 "Alternatives and tradeoffs" — so cross-session SHARED is still
        rejected. Raises ``NamespaceIsolationError`` with a non-enumerating message (never echo the
        requested id)."""
        del operation
        cross_session = target.session != self.session_id and target.visibility is not (
            Visibility.PRIVATE
        )
        if target.org != self.org_id or target.workspace != self.workspace_id or cross_session:
            from mu_contracts.domain.errors import NamespaceIsolationError

            raise NamespaceIsolationError("not found")
