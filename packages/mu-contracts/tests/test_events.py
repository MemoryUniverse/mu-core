"""Event catalog — content-free guard + DegradedModeEntered (CANONICAL §2/§3)."""

import pytest
from pydantic import ValidationError

from mu_contracts.domain.events import (
    RESERVED_REASONS,
    SYNC_CLASS_ALWAYS,
    DegradedModeEntered,
    DegradeReason,
    DomainEvent,
    MemoryCaptured,
    RoomMessagePosted,
)
from mu_contracts.domain.model import Namespace, Tier, Visibility

pytestmark = pytest.mark.unit


def test_content_bearing_field_name_is_rejected_at_class_definition() -> None:
    with pytest.raises(TypeError, match="content-free"):

        class Leaky(DomainEvent):  # pragma: no cover - definition must raise
            body: str


def test_content_hash_is_allowed() -> None:
    # RoomMessagePosted carries content_hash (a locator, not the body) and must construct.
    ev = RoomMessagePosted(
        room_id="r1",
        seq=1,
        author_principal_id="alice",
        author_kind="human",  # type: ignore[arg-type]
        kind="utterance",  # type: ignore[arg-type]
        correlation_id="c1",
        content_hash="deadbeef",
    )
    assert ev.content_hash == "deadbeef"


def test_events_are_frozen_and_forbid_extra() -> None:
    ns = Namespace(org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE)
    ev = MemoryCaptured(namespace=ns, ids=["m1"])
    assert ev.tier is Tier.STM
    with pytest.raises(ValidationError):
        ev.tier = Tier.MTM
    with pytest.raises(ValidationError):
        MemoryCaptured(namespace=ns, ids=["m1"], nope=1)  # type: ignore[call-arg]


def test_degraded_mode_four_fields_and_user_visibility() -> None:
    stalled = DegradedModeEntered(
        component="device_sync", mode="reconcile_stalled", reason=DegradeReason.SYNC_STALLED
    )
    assert stalled.user_visible() is True  # SYNC-CLASS
    op_only = DegradedModeEntered(
        component="ltm", mode="recall_mtm_only", reason=DegradeReason.LTM_UNAVAILABLE
    )
    assert op_only.user_visible() is False  # operator-only


def test_server_unreachable_is_sync_class_only_under_device_sync() -> None:
    on_device = DegradedModeEntered(
        component="device_sync", mode="spooled", reason=DegradeReason.SERVER_UNREACHABLE
    )
    elsewhere = DegradedModeEntered(
        component="gateway", mode="spooled", reason=DegradeReason.SERVER_UNREACHABLE
    )
    assert on_device.user_visible() is True
    assert elsewhere.user_visible() is False


def test_sync_class_and_reserved_sets_are_disjoint() -> None:
    assert SYNC_CLASS_ALWAYS.isdisjoint(RESERVED_REASONS)
    assert DegradeReason.SYNC_STALLED in SYNC_CLASS_ALWAYS
    assert DegradeReason.E2E_NO_SERVER_EMBED in RESERVED_REASONS
