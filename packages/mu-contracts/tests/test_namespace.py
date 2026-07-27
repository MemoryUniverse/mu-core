"""Namespace (η) invariants — the tenancy partition key (CANONICAL §1, platform-layer0 §2).

Acceptance obligations (platform-layer0 §15.2): exactly 5 fields; separator-injection rejected
on every component; ``to_prefix()`` = ``mu/{org}/{workspace}/{visibility}/{user_slot}/{session}``
(6 segments); ``shared()`` zeroes ``user`` to ``"*"``; ``parts()`` ↔ ``Namespace(*parts)``
round-trips; byte-stable and region-agnostic.
"""

import pytest
from pydantic import ValidationError

from mu_contracts.domain.model import Namespace, Visibility

pytestmark = pytest.mark.unit


def _private() -> Namespace:
    return Namespace(
        org="acme", workspace="proj", user="alice", session="s1", visibility=Visibility.PRIVATE
    )


def test_exactly_five_fields_in_order() -> None:
    assert Namespace.PARTS == ("org", "workspace", "user", "session", "visibility")
    assert _private().parts() == ("acme", "proj", "alice", "s1", "private")


def test_to_prefix_byte_shape_private() -> None:
    ns = _private()
    prefix = ns.to_prefix()
    assert prefix == "mu/acme/proj/private/alice/s1"
    assert prefix.split("/") == ["mu", "acme", "proj", "private", "alice", "s1"]  # 6 segments


def test_to_prefix_shared_zeroes_user_slot() -> None:
    ns = Namespace.shared(org="acme", workspace="proj", session="room1")
    assert ns.user == "*"
    assert ns.visibility is Visibility.SHARED
    assert ns.to_prefix() == "mu/acme/proj/shared/*/room1"
    assert len(ns.to_prefix().split("/")) == 6  # byte-shape identical to private


@pytest.mark.parametrize("bad", ["a/b", "a:b", "a|b", "a\\b", "a b", "a\tb", "a\nb", "a\x00b", ""])
@pytest.mark.parametrize("field", ["org", "workspace", "user", "session"])
def test_separator_injection_rejected_on_every_component(field: str, bad: str) -> None:
    base = {"org": "o", "workspace": "w", "user": "u", "session": "s", "visibility": "private"}
    base[field] = bad
    with pytest.raises(ValidationError):
        Namespace(**base)  # type: ignore[arg-type]


def test_parts_round_trip() -> None:
    ns = _private()
    assert Namespace.from_parts(ns.parts()) == ns


def test_shared_round_trip() -> None:
    ns = Namespace.shared(org="o", workspace="w", session="s")
    assert Namespace.from_parts(ns.parts()) == ns


def test_namespace_is_frozen_and_forbids_extra() -> None:
    ns = _private()
    with pytest.raises(ValidationError):
        ns.org = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Namespace(
            org="o",
            workspace="w",
            user="u",
            session="s",
            visibility=Visibility.PRIVATE,
            region="eu",  # type: ignore[call-arg]
        )
