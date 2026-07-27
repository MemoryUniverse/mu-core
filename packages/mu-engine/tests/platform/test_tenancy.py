"""TenancyGuard persistence-level isolation (platform-layer0-spec §5, §15.10)."""

from __future__ import annotations

import pytest

from mu_contracts.domain.errors import NamespaceIsolationError
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.platform.tenancy import DefaultTenancyGuard

pytestmark = pytest.mark.unit


def _scope() -> ClientScope:
    return ClientScope(
        principal_id="p1",
        org_id="orgA",
        workspace_id="wsA",
        session_id="s1",
        agent_principal_id="p1",
    )


def test_private_own_namespace_passes() -> None:
    guard = DefaultTenancyGuard()
    ns = _scope().namespace(Visibility.PRIVATE)
    guard.assert_scope(_scope(), ns, "read")  # no raise


def test_shared_namespace_passes() -> None:
    guard = DefaultTenancyGuard()
    ns = _scope().namespace(Visibility.SHARED)
    assert ns.user == "*"
    guard.assert_scope(_scope(), ns, "read")


def test_cross_org_denied() -> None:
    guard = DefaultTenancyGuard()
    other = Namespace(
        org="orgB", workspace="wsA", user="p1", session="s1", visibility=Visibility.PRIVATE
    )
    with pytest.raises(NamespaceIsolationError):
        guard.assert_scope(_scope(), other, "read")


def test_cross_workspace_denied() -> None:
    guard = DefaultTenancyGuard()
    other = Namespace(
        org="orgA", workspace="wsZ", user="p1", session="s1", visibility=Visibility.PRIVATE
    )
    with pytest.raises(NamespaceIsolationError):
        guard.assert_scope(_scope(), other, "read")


def test_private_other_users_slot_denied() -> None:
    guard = DefaultTenancyGuard()
    # same org/ws/session but a DIFFERENT user slot than the acting agent -> isolation breach.
    other = Namespace(
        org="orgA",
        workspace="wsA",
        user="someone_else",
        session="s1",
        visibility=Visibility.PRIVATE,
    )
    with pytest.raises(NamespaceIsolationError):
        guard.assert_scope(_scope(), other, "read")


def test_denial_message_is_non_enumerating() -> None:
    guard = DefaultTenancyGuard()
    other = Namespace(
        org="orgB", workspace="wsA", user="p1", session="s1", visibility=Visibility.PRIVATE
    )
    with pytest.raises(NamespaceIsolationError) as ei:
        guard.assert_scope(_scope(), other, "read")
    assert "orgB" not in str(ei.value)  # never echoes the requested target
