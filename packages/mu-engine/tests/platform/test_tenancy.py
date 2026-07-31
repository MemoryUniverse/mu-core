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


def test_private_cross_session_same_user_allowed() -> None:
    """ADR 0030: a PRIVATE η in a DIFFERENT session of the SAME (org, workspace, user) must
    federate, not raise — the fix for the cross-session-federation bug. This is the belt-and-
    suspenders layer ``RecallService.recall`` (service.py:134) re-checks every surviving item
    against; it must let a genuinely cross-session, same-user hit through."""
    guard = DefaultTenancyGuard()
    other_session = Namespace(
        org="orgA", workspace="wsA", user="p1", session="s-other", visibility=Visibility.PRIVATE
    )
    guard.assert_scope(_scope(), other_session, "read")  # no raise


def test_private_cross_session_different_user_denied() -> None:
    """The security half: relaxing the session check for PRIVATE must NOT weaken cross-user
    isolation. Same org/workspace, a DIFFERENT session AND a DIFFERENT user slot — still denied,
    because the user-slot check (step 2) is untouched by the session relaxation."""
    guard = DefaultTenancyGuard()
    other_user_other_session = Namespace(
        org="orgA",
        workspace="wsA",
        user="someone_else",
        session="s-other",
        visibility=Visibility.PRIVATE,
    )
    with pytest.raises(NamespaceIsolationError):
        guard.assert_scope(_scope(), other_user_other_session, "read")


def test_shared_cross_session_denied() -> None:
    """Rooms are real walls (ADR 0030): a SHARED η in a different session must stay denied even
    though PRIVATE now federates across sessions."""
    guard = DefaultTenancyGuard()
    other_room = Namespace(
        org="orgA", workspace="wsA", user="*", session="room-other", visibility=Visibility.SHARED
    )
    with pytest.raises(NamespaceIsolationError):
        guard.assert_scope(_scope(), other_room, "read")


def test_denial_message_is_non_enumerating() -> None:
    guard = DefaultTenancyGuard()
    other = Namespace(
        org="orgB", workspace="wsA", user="p1", session="s1", visibility=Visibility.PRIVATE
    )
    with pytest.raises(NamespaceIsolationError) as ei:
        guard.assert_scope(_scope(), other, "read")
    assert "orgB" not in str(ei.value)  # never echoes the requested target
