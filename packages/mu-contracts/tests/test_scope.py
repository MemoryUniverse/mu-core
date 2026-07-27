"""ClientScope invariants — acting agent + namespace isolation (platform-layer0 §3/§15.3)."""

import pytest

from mu_contracts.domain.errors import NamespaceIsolationError
from mu_contracts.domain.model import AgentKind, ClientScope, Visibility

pytestmark = pytest.mark.unit


def _scope() -> ClientScope:
    return ClientScope(
        principal_id="alice",
        org_id="acme",
        workspace_id="proj",
        session_id="s1",
        agent_principal_id="alice",  # HUMAN_PROXY: acting == owner
    )


def test_agent_principal_id_never_none_defaults_human_proxy() -> None:
    s = _scope()
    assert s.agent_principal_id == s.principal_id
    assert s.agent_kind is AgentKind.HUMAN_PROXY


def test_namespace_private_uses_agent_principal_as_user() -> None:
    ns = _scope().namespace(Visibility.PRIVATE)
    assert ns.user == "alice"
    assert ns.to_prefix() == "mu/acme/proj/private/alice/s1"


def test_namespace_shared_zeroes_user() -> None:
    ns = _scope().namespace(Visibility.SHARED)
    assert ns.user == "*"
    assert ns.visibility is Visibility.SHARED


def test_assert_authorized_rejects_cross_tenant() -> None:
    s = _scope()
    foreign = s.namespace(Visibility.PRIVATE).model_copy(update={"org": "evilcorp"})
    with pytest.raises(NamespaceIsolationError):
        s.assert_authorized(foreign, "recall")


def test_assert_authorized_allows_own_namespace() -> None:
    s = _scope()
    s.assert_authorized(s.namespace(Visibility.PRIVATE), "recall")  # no raise
