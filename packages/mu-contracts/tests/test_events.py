"""Event catalog — content-free guard + DegradedModeEntered (CANONICAL §2/§3)."""

import pytest
from pydantic import ValidationError

from mu_contracts.domain.events import (
    RESERVED_REASONS,
    SYNC_CLASS_ALWAYS,
    ComposedContextStale,
    DegradedModeEntered,
    DegradeReason,
    DomainEvent,
    MemoryCaptured,
    RevokeCascadeCompleted,
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


# ==================================================================================================
# AD-42 — the cascade receipt's counts have a home on the event
# ==================================================================================================
def test_revoke_cascade_completed_carries_the_two_counts_the_receipt_reads() -> None:
    """``governance-transfer-core-spec.md:762`` and ``design-governance-transfer.puml:659`` both
    emit ``(root, revoked_count, ack_pending_count, cache_entries_purged)``, and
    ``trust-ledger-spec.md:366`` requires ``{"revoked","ack_pending","cache_purged"}`` to be
    DERIVABLE from this event. With the fields absent the producer routed them onto the ledger row
    instead, so the receipt's ``PARTIAL``/``SETTLED`` honesty depended on a side channel."""
    ev = RevokeCascadeCompleted(
        root_grant_id="g1", revoked_count=7, ack_pending_count=1, cache_entries_purged=12
    )
    assert (ev.ack_pending_count, ev.cache_entries_purged) == (1, 12)


def test_revoke_cascade_completed_counts_default_to_zero_for_existing_producers() -> None:
    """Zero is the honest reading of what a producer that measures neither reports — and the
    default is what lets every shipped call site keep constructing unchanged."""
    ev = RevokeCascadeCompleted(root_grant_id="g1", revoked_count=1)
    assert (ev.ack_pending_count, ev.cache_entries_purged) == (0, 0)


def test_revoke_cascade_completed_refuses_a_negative_count() -> None:
    """A negative purge count is not a measurement; it is a bug that would flow straight into a
    signed receipt."""
    with pytest.raises(ValidationError):
        RevokeCascadeCompleted(root_grant_id="g1", revoked_count=1, cache_entries_purged=-1)


# ==================================================================================================
# AD-69 — ComposedContextStale (CANONICAL §7.10 G8; spec:476, listed at :875)
# ==================================================================================================
def test_composed_context_stale_carries_a_count_and_no_sources() -> None:
    """Observational by contract: the snapshot stays immutable and no grant is auto-severed, so
    the event carries the freshness COUNT and no state. A count is content-free (CANONICAL §3)."""
    ev = ComposedContextStale(composed_id="cmp1", sources_superseded_count=2)
    assert ev.composed_id == "cmp1"
    assert ev.sources_superseded_count == 2


def test_composed_context_stale_refuses_a_zero_count() -> None:
    """ "Stale because zero sources went stale" is not a signal, it is noise on the bus."""
    with pytest.raises(ValidationError):
        ComposedContextStale(composed_id="cmp1", sources_superseded_count=0)
