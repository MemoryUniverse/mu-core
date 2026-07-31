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
