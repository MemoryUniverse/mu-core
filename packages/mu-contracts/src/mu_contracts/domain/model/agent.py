"""Agent / subagent identity — the FIRST real realization of the identity model.

Authority: ``docs/superpowers/design/agent-subagent-identity-design.md`` (§1 the identity model,
§2 visibility) + ``docs/tracking/AGENT-INTEGRATION-AUDIT-AND-PLAN.md`` §6 (subagents). Until this
module every piece of the model was DEAD: ``AgentKind.SUBAGENT`` (``scope.py``) and the
``AgentEnrolled``/``SubagentSpawned`` DTOs (``events.py``) existed but had ZERO call sites.

Phase 1.5 lands the SMALLEST REAL partition that satisfies the three requirements jointly:

  (1) a subagent's memory is stored in a DISTINCT physical ``to_prefix()`` partition — an
      agent-scoped SESSION — not merely a ``[subagent:...]`` text prefix;
  (2) the OWNER (parent) session recalls across its subagents' memories via the EXISTING
      federate-live recall (``qdrant_mtm._user_prefix`` truncates the session segment, so the
      user-prefix arm surfaces every session under the same ``η.user``);
  (3) cross-USER isolation is preserved for free — the user-prefix still carries the owner's
      principal id, so a different human's partition never matches.

FEDERATION CHOICE (deliberate, stated). The design's strict model puts ``η.user =
agent_principal_id`` (a subagent's OWN partition, isolated from the parent, readable only via an
``AgentVisibilityGrant``). That model makes requirement (2) — parent-reads-subagent — need a NEW
granted recall arm (``AgentVisibilityResolver``), which is not yet built. The task's federation
requirement is explicitly the LIGHTER, session-grounded one ("subagent→parent visibility within the
same user+session … ground it in the existing recall federation … do NOT break cross-USER
isolation"). Those two are only jointly satisfiable if ``η.user`` stays the OWNER and the agent
partitions by SESSION. So Phase 1.5 keeps ``η.user = owner`` and derives an agent-scoped SESSION
from ``agent_principal_id``. The acting-agent identity still rides on ``ClientScope``
(``agent_principal_id`` + ``AgentKind.SUBAGENT``) — CANONICAL §1 rule 2, never a sixth η axis — so a
future upgrade to the strict own-partition model (own ⊕ granted arm, registry, enrollment events)
is additive, not a rewrite. That heavier machinery (``AgentRegistryPort``/``AgentRegistryService``,
``AgentVisibilityGrant``, the lifecycle events) is DEFERRED and flagged, never faked here.

``agent_principal_id`` is deterministic + idempotent (design §1.1): the same
``(workspace, owner, parent, framework, external_key)`` always mints the same id, so no registry
service is required for v1 — re-resolving a subagent yields the identical partition.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.scope import AgentKind, ClientScope

__all__ = [
    "AgentFramework",
    "AgentIdentity",
    "mint_agent_principal_id",
    "resolve_subagent_identity",
    "subagent_partition_session",
    "subagent_write_namespace",
]

# The agent-scoped session segment delimiter. Only chars OUTSIDE ``_FORBIDDEN_NS_CHARS``
# (``/:|\\ \t\n\x00``, memory.py) may appear in a namespace component — ``.`` is legal, ``:`` is
# NOT (which is exactly why the agent partition can NOT be a ``[subagent:...]``-style value and must
# be a real, separator-safe session).
_SUBAGENT_SESSION_INFIX = ".sub."


class AgentFramework(StrEnum):
    """The agentic framework an agent principal is hosted in (design §1.1). ``CLAUDE_CODE`` is the
    only one with a live capture path this phase; the rest are named so the closed enum needs no
    breaking change when their parsers land."""

    CLAUDE_CODE = "claude_code"
    CLAUDE_DESKTOP = "claude_desktop"
    CLAUDE_WEB = "claude_web"
    GPT = "gpt"
    LANGGRAPH = "langgraph"
    OTHER = "other"


class AgentIdentity(BaseModel):
    """A resolved agent/subagent principal (design §1.1, the v1 subset actually consumed this
    phase). Frozen value object, minted deterministically — NOT a registry row (the
    ``AgentRegistryPort``/``AgentRegistryService`` write-side is deferred; §6D). ``kind`` is
    ``AgentKind.SUBAGENT`` for every identity this module mints today."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_principal_id: str = Field(min_length=1)  # goes onto ClientScope.agent_principal_id
    workspace_id: str = Field(min_length=1)
    owner_principal_id: str = Field(min_length=1)  # the human/service at the top of the tree
    parent_agent_id: str | None = None  # the SPAWNING agent principal; None ⇒ top-level
    agent_path: str = Field(min_length=1)  # materialized path (O(1) ancestry / subtree prefix)
    kind: AgentKind = AgentKind.SUBAGENT
    framework: AgentFramework = AgentFramework.CLAUDE_CODE
    external_key: str = Field(min_length=1)  # framework-native idempotent enroll key


def mint_agent_principal_id(
    *,
    workspace_id: str,
    owner_principal_id: str,
    parent_agent_id: str | None,
    framework: AgentFramework,
    external_key: str,
) -> str:
    """Deterministic + idempotent principal id (design §1.1):
    ``"agt_" + sha256(workspace | owner | parent | framework | external_key)[:24]``.

    Including ``parent_agent_id`` means two subagents sharing a LOCAL name under different
    supervisors get DISTINCT ids — no collision, no cross-read. Re-minting the same tuple is a
    no-op (identical id ⇒ identical partition), which is why no registry write is needed for v1."""
    raw = "|".join(
        (
            workspace_id,
            owner_principal_id,
            parent_agent_id or "",
            framework.value,
            external_key,
        )
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"agt_{digest}"


def resolve_subagent_identity(
    *,
    workspace_id: str,
    owner_principal_id: str,
    parent_session_id: str,
    agent_type: str,
    framework: AgentFramework = AgentFramework.CLAUDE_CODE,
    parent_agent_id: str | None = None,
) -> AgentIdentity:
    """Resolve (mint, deterministically) the identity of a Claude Code subagent from its capture
    attribution. ``external_key`` is the design's Claude Code key — the subagent NAME (Task tool
    ``subagent_type``) joined with the parent session id (design §3 table) — so the same subagent in
    the same session always resolves to the same partition, while the same-named subagent in a
    DIFFERENT session gets a distinct one."""
    external_key = f"{agent_type}@{parent_session_id}"
    agent_principal_id = mint_agent_principal_id(
        workspace_id=workspace_id,
        owner_principal_id=owner_principal_id,
        parent_agent_id=parent_agent_id,
        framework=framework,
        external_key=external_key,
    )
    agent_path = (
        f"{parent_agent_id}/{agent_principal_id}" if parent_agent_id else f"/{agent_principal_id}"
    )
    return AgentIdentity(
        agent_principal_id=agent_principal_id,
        workspace_id=workspace_id,
        owner_principal_id=owner_principal_id,
        parent_agent_id=parent_agent_id,
        agent_path=agent_path,
        kind=AgentKind.SUBAGENT,
        framework=framework,
        external_key=external_key,
    )


def subagent_partition_session(owner_session: str, agent_principal_id: str) -> str:
    """The agent-scoped SESSION a subagent's memory is partitioned under (the federation choice,
    module docstring). ``{owner_session}.sub.{agent_principal_id}`` — a DISTINCT, separator-safe
    ``η.session`` value that (a) makes ``to_prefix()`` a distinct physical partition and (b) keeps
    the SAME leading user-prefix as the owner, so ``qdrant_mtm._user_prefix`` federation surfaces it
    to the owner's recall while the owner's principal id in that prefix keeps other users out."""
    return f"{owner_session}{_SUBAGENT_SESSION_INFIX}{agent_principal_id}"


def subagent_write_namespace(scope: ClientScope) -> Namespace:
    """Derive the η a SUBAGENT-kind ``ClientScope`` writes/reads under — the ONE place the Phase 1.5
    federation choice is applied, driven by the acting-agent fields the identity model already
    carries on ``ClientScope``. Load-bearing use of ``AgentKind.SUBAGENT`` +
    ``agent_principal_id`` (their first real call site):

    * ``η.user`` = ``scope.principal_id`` (the OWNER) — owner-federation + cross-user isolation.
    * ``η.session`` = :func:`subagent_partition_session` of the owner's session and the acting
      agent principal — the DISTINCT agent partition.

    A non-subagent scope is a programming error here (the caller gates on kind first)."""
    if scope.agent_kind is not AgentKind.SUBAGENT:
        raise ValueError(
            "subagent_write_namespace requires a SUBAGENT-kind ClientScope; "
            f"got agent_kind={scope.agent_kind.value!r}"
        )
    return Namespace(
        org=scope.org_id,
        workspace=scope.workspace_id,
        user=scope.principal_id,
        session=subagent_partition_session(scope.session_id, scope.agent_principal_id),
        visibility=Visibility.PRIVATE,
    )
