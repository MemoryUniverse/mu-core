"""Governance model — Grant lifecycle + ShareableRef + ComposedContext (governance §1-§6)."""

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.model import (
    ComposedContext,
    Direction,
    Grant,
    GrantState,
    Permission,
    PrincipalRef,
    PrincipalRefKind,
    ShareableRef,
    ShareableType,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_LATER = datetime(2026, 6, 1, tzinfo=UTC)


def _ref() -> ShareableRef:
    return ShareableRef(
        object_type=ShareableType.MEMORY,
        object_id="m1",
        content_hash="h1",
        org_id="acme",
        workspace_id="proj",
        origin_namespace_id="ns1",
    )


def test_shareable_ref_deterministic_digests() -> None:
    r = _ref()
    assert r.stream_id() == "prov:acme:memory:m1"
    assert r.canonical() == "memory|m1|h1|acme|proj|ns1"


def _grant(**over: object) -> Grant:
    base: dict[str, object] = {
        "id": "grant_1",
        "object_ref": _ref(),
        "grantor_principal_id": "alice",
        "grantee": PrincipalRef(
            kind=PrincipalRefKind.PRINCIPAL, id="bob", org_id="acme", workspace_id="proj"
        ),
        "direction": Direction.PUBLISH,
        "permissions": frozenset({Permission.READ}),
        "policy_id": "pol1",
        "issued_at": _NOW,
        "provenance_id": "prov1",
    }
    base.update(over)
    return Grant(**base)  # type: ignore[arg-type]


def test_grant_active_and_permission_check() -> None:
    g = _grant()
    assert g.is_active(at=_NOW) is True
    assert g.can(Permission.READ, at=_NOW) is True
    assert g.can(Permission.WRITE, at=_NOW) is False


def test_grant_expiry_and_terminal() -> None:
    g = _grant(expires_at=_NOW)
    assert g.is_active(at=_LATER) is False  # expired by time
    revoked = _grant(state=GrantState.REVOKED)
    assert revoked.is_terminal() is True
    assert revoked.is_active(at=_NOW) is False


def test_permissions_min_length_enforced() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _grant(permissions=frozenset())


def test_composed_context_as_ref() -> None:
    cc = ComposedContext(
        id="cc1",
        org_id="acme",
        workspace_id="proj",
        namespace_id="ns1",
        source_memory_ids=("m1", "m2"),
        intent="team brief",
        body_ref="artifact_9",
        content_hash="hc",
        provenance_id="p",
        created_at=_NOW,
    )
    ref = cc.as_ref()
    assert ref.object_type is ShareableType.COMPOSED
    assert ref.object_id == "cc1"
    assert ref.origin_namespace_id == "ns1"
