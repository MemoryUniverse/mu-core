"""Unit tests for the plane-gate validator (sdk-engine-server-design.md §2.5; build-plan Stage B
task B0). Pure logic, no store — everything here is ``pytest.mark.unit``."""

from __future__ import annotations

import pytest

from mu_contracts.domain.errors import PlaneFieldRejectedError
from mu_contracts.validation.plane_gate import (
    PRIVATE_PLANE_FIELDS,
    SHARED_PLANE_FIELDS,
    validate_plane_fields,
)

pytestmark = pytest.mark.unit


def test_field_groups_match_design_2_5_add_signature() -> None:
    """design §2.5's canonical ``add`` superset signature is the authority for field names."""
    assert PRIVATE_PLANE_FIELDS == {"user", "session", "agent"}
    assert SHARED_PLANE_FIELDS == {"visibility", "subject", "predicate", "object"}
    assert PRIVATE_PLANE_FIELDS.isdisjoint(SHARED_PLANE_FIELDS)


def test_shared_field_rejected_when_no_shared_plane_configured() -> None:
    """design §2.5 worked example: 'passing visibility= while mode="embedded" and shared=None
    raises'."""
    with pytest.raises(PlaneFieldRejectedError) as exc_info:
        validate_plane_fields(
            {"visibility": "shared"},
            private_configured=True,
            shared_configured=False,
        )
    err = exc_info.value
    assert err.field == "visibility"
    assert err.plane == "shared"


def test_each_shared_field_is_gated() -> None:
    for name, value in (
        ("visibility", "shared"),
        ("subject", "alice"),
        ("predicate", "likes"),
        ("object", "coffee"),
    ):
        with pytest.raises(PlaneFieldRejectedError):
            validate_plane_fields({name: value}, private_configured=True, shared_configured=False)


def test_private_field_rejected_under_shared_only_configuration() -> None:
    """The mirror direction: LocalMemory-shaped fields on a remote/shared-only client."""
    with pytest.raises(PlaneFieldRejectedError) as exc_info:
        validate_plane_fields(
            {"user": "ada"},
            private_configured=False,
            shared_configured=True,
        )
    err = exc_info.value
    assert err.field == "user"
    assert err.plane == "private"


def test_each_private_field_is_gated() -> None:
    for name, value in (("user", "ada"), ("session", "s1"), ("agent", "agent-1")):
        with pytest.raises(PlaneFieldRejectedError):
            validate_plane_fields({name: value}, private_configured=False, shared_configured=True)


def test_private_only_configuration_accepts_private_fields() -> None:
    """LocalMemory today: private_configured=True, shared_configured=False (mu-local/
    local_memory.py:12-13 — 'NO SHARED partition ... those are mu-server concepts')."""
    validate_plane_fields(
        {"user": "ada", "session": "s1", "agent": None},
        private_configured=True,
        shared_configured=False,
    )  # no raise


def test_dual_plane_configuration_accepts_both_groups() -> None:
    """design §4's dual-plane worked example: mode="local_server" + a populated shared=."""
    validate_plane_fields(
        {"user": "ada", "visibility": "shared"},
        private_configured=True,
        shared_configured=True,
    )  # no raise


def test_none_values_are_treated_as_not_supplied() -> None:
    """Every plane-gated kwarg defaults to None on both canonical signatures — a caller that
    never touched the kwarg (still at its None default) must not be rejected."""
    validate_plane_fields(
        {"user": None, "visibility": None, "subject": None},
        private_configured=True,
        shared_configured=False,
    )  # no raise — nothing was actually supplied


def test_unplane_gated_fields_always_pass_through() -> None:
    """'already-common across both surfaces today, unchanged' fields (tier/metadata/…) are never
    plane-gated, regardless of which plane(s) are configured."""
    validate_plane_fields(
        {"tier": "stm", "metadata": {"k": "v"}, "importance_score": 0.5},
        private_configured=False,
        shared_configured=False,
    )  # no raise


def test_shared_only_client_rejects_private_field_even_when_absent_from_kwargs_is_fine() -> None:
    """Sanity: an empty/omitted mapping never raises, regardless of plane configuration."""
    validate_plane_fields({}, private_configured=False, shared_configured=False)  # no raise


def test_error_message_names_field_and_plane() -> None:
    with pytest.raises(PlaneFieldRejectedError, match="'visibility'.*'shared'"):
        validate_plane_fields(
            {"visibility": "shared"}, private_configured=True, shared_configured=False
        )
