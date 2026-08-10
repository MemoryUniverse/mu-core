"""Agent / subagent identity — Phase 1.5 (mu_contracts.domain.model.agent).

Pure-unit proofs (no stores) for the identity model's first real realization: deterministic id
mint, subagent isolation-by-construction, and the FEDERATION CHOICE encoded in
``subagent_write_namespace`` — owner-scoped ``η.user`` (federation + cross-user isolation) with an
agent-scoped ``η.session`` (the distinct partition).
"""

from __future__ import annotations

import pytest

from mu_contracts.domain.model.agent import (
    AgentFramework,
    mint_agent_principal_id,
    resolve_subagent_identity,
    subagent_partition_session,
    subagent_write_namespace,
)
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.scope import AgentKind, ClientScope

pytestmark = pytest.mark.unit


def _mint(**over: str | None) -> str:
    base: dict[str, str | None] = {
        "workspace_id": "ws1",
        "owner_principal_id": "alice",
        "parent_agent_id": None,
        "external_key": "researcher@s1",
    }
    base.update(over)
    return mint_agent_principal_id(
        workspace_id=str(base["workspace_id"]),
        owner_principal_id=str(base["owner_principal_id"]),
        parent_agent_id=base["parent_agent_id"],  # type: ignore[arg-type]
        framework=AgentFramework.CLAUDE_CODE,
        external_key=str(base["external_key"]),
    )


def test_mint_is_deterministic_and_idempotent() -> None:
    assert _mint() == _mint()
    assert _mint().startswith("agt_")
    assert len(_mint()) == len("agt_") + 24


def test_mint_distinguishes_every_input_axis() -> None:
    base = _mint()
    assert _mint(workspace_id="ws2") != base
    assert _mint(owner_principal_id="bob") != base
    assert _mint(parent_agent_id="agt_parent") != base
    assert _mint(external_key="reviewer@s1") != base  # different subagent name
    assert _mint(external_key="researcher@s2") != base  # same name, different session


def test_two_subagents_same_local_name_different_supervisor_are_distinct() -> None:
    """design §1.1: including parent_agent_id means two 'researcher' subagents under different
    supervisors get DISTINCT principal ids — no collision, no cross-read."""
    a = _mint(parent_agent_id="agt_sup_a")
    b = _mint(parent_agent_id="agt_sup_b")
    assert a != b


def test_resolve_subagent_identity_shape() -> None:
    ident = resolve_subagent_identity(
        workspace_id="ws1",
        owner_principal_id="alice",
        parent_session_id="s1",
        agent_type="researcher",
    )
    assert ident.kind is AgentKind.SUBAGENT
    assert ident.framework is AgentFramework.CLAUDE_CODE
    assert ident.external_key == "researcher@s1"
    assert ident.owner_principal_id == "alice"
    assert ident.agent_principal_id.startswith("agt_")
    assert ident.agent_path == f"/{ident.agent_principal_id}"
    # idempotent: re-resolving the same attribution yields the SAME partition principal.
    again = resolve_subagent_identity(
        workspace_id="ws1",
        owner_principal_id="alice",
        parent_session_id="s1",
        agent_type="researcher",
    )
    assert again.agent_principal_id == ident.agent_principal_id


def test_partition_session_is_separator_safe_and_namespace_valid() -> None:
    """The agent-scoped session must be a LEGAL η component (no ``:`` — which is exactly why a
    ``[subagent:...]`` value cannot be a partition); Namespace construction is the real gate."""
    apid = _mint()
    sess = subagent_partition_session("owner-session", apid)
    assert sess == f"owner-session.sub.{apid}"
    # constructing a Namespace with it must NOT raise (proves separator-safety at the real gate).
    ns = Namespace(
        org="o", workspace="w", user="alice", session=sess, visibility=Visibility.PRIVATE
    )
    assert ns.session == sess


def _subagent_scope(owner: str, owner_session: str, agent_principal_id: str) -> ClientScope:
    return ClientScope(
        principal_id=owner,
        org_id="acme",
        workspace_id="proj",
        session_id=owner_session,
        agent_principal_id=agent_principal_id,
        agent_kind=AgentKind.SUBAGENT,
        agent_path=f"/{agent_principal_id}",
    )


def test_write_namespace_keeps_owner_user_and_agent_session() -> None:
    """The federation choice: η.user stays the OWNER (federation + isolation), η.session is the
    distinct agent partition."""
    apid = _mint()
    scope = _subagent_scope("alice", "sess-1", apid)
    ns = subagent_write_namespace(scope)
    assert ns.user == "alice"  # OWNER, not the agent principal
    assert ns.visibility is Visibility.PRIVATE
    assert ns.session == subagent_partition_session("sess-1", apid)
    assert ns.session != "sess-1"  # distinct from the owner session → distinct partition


def test_write_namespace_federates_to_owner_but_isolates_across_users() -> None:
    """The subagent partition shares the owner's user-prefix (federate-live surfaces it to the
    owner) while a DIFFERENT owner's partition never shares that prefix (cross-user isolation).
    Mirrors qdrant_mtm._user_prefix: the leading segments up to (and including) the user slot."""
    apid = _mint()

    def user_prefix(ns: Namespace) -> str:
        # to_prefix() is mu/{org}/{ws}/{vis}/{user}/{session}; the federation grain drops session.
        return "/".join(ns.to_prefix().split("/")[:-1])

    owner_a = subagent_write_namespace(_subagent_scope("alice", "sess-1", apid))
    owner_a_plain = Namespace(
        org="acme", workspace="proj", user="alice", session="sess-1", visibility=Visibility.PRIVATE
    )
    owner_b = subagent_write_namespace(_subagent_scope("bob", "sess-1", apid))

    # subagent under alice shares alice's OWN user-prefix → owner recall federates it.
    assert user_prefix(owner_a) == user_prefix(owner_a_plain)
    # a different human never shares the prefix → cross-user isolation preserved.
    assert user_prefix(owner_a) != user_prefix(owner_b)


def test_write_namespace_rejects_non_subagent_scope() -> None:
    human = ClientScope(
        principal_id="alice",
        org_id="acme",
        workspace_id="proj",
        session_id="s1",
        agent_principal_id="alice",  # HUMAN_PROXY
    )
    with pytest.raises(ValueError, match="SUBAGENT"):
        subagent_write_namespace(human)
