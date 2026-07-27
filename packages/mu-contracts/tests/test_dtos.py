"""The referenced DTOs + persona/notification invariants (storage-schema §1, persona §3.1,
trust-surfaces §4.1)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.domain.model import (
    ConflictEdgeRow,
    ConflictEdges,
    ConflictState,
    EntityCandidate,
    EntityResolution,
    MemoryItem,
    MemoryKind,
    Namespace,
    Notification,
    NotificationCategory,
    NotificationKind,
    NotificationSeverity,
    PersonaProfile,
    RecallChannel,
    Scored,
    SparseQuery,
    Usage,
    Validity,
    Visibility,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_sparse_query_length_invariant() -> None:
    SparseQuery(indices=(1, 2), values=(0.5, 0.9), encoder="bm25")  # ok
    with pytest.raises(ValidationError):
        SparseQuery(indices=(1, 2, 3), values=(0.5,), encoder="bm25")


def test_scored_is_generic_over_memory_item() -> None:
    item = MemoryItem(
        id="m1",
        namespace=Namespace(
            org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
        ),
        kind=MemoryKind.PROPOSITION,
        content="c",
        validity=Validity(valid_at=_NOW, recorded_at=_NOW),
        last_seen=_NOW,
        provenance_id="p",
    )
    scored: Scored[MemoryItem] = Scored(item=item, score=0.8, channel=RecallChannel.MTM_DENSE)
    assert scored.item.id == "m1"
    assert scored.is_floor is False


def test_usage_to_meter_fields() -> None:
    u = Usage(prompt_tokens=10, completion_tokens=5, cached_input_tokens=3, reasoning_tokens=2)
    assert u.to_meter_fields() == {
        "tokens_in": 10,
        "tokens_out": 5,
        "cached_input_tokens": 3,
        "reasoning_tokens": 2,
    }


def test_entity_resolution_shape() -> None:
    res = EntityResolution(
        canonical_name="ada lovelace",
        entity_uid=None,
        candidates=(EntityCandidate(entity_uid="e1", canonical_name="Ada", similarity=0.7),),
    )
    assert res.entity_uid is None  # ambiguous → caller tie-break over candidates
    assert res.candidates[0].similarity == 0.7


def test_conflict_edges_predicates() -> None:
    edges = ConflictEdges(
        rows_by_memory={
            "m1": ConflictEdgeRow(
                memory_id="m1",
                peer_ids=frozenset({"m2"}),
                conflict_id="c1",
                state=ConflictState.DETECTED,
                pin_blocked=True,
            ),
            "m3": ConflictEdgeRow(
                memory_id="m3",
                peer_ids=frozenset({"m4"}),
                conflict_id="c2",
                state=ConflictState.RESOLVED,
            ),
        }
    )
    assert edges.unresolved_for("m1") is True
    assert edges.pin_blocked_for("m1") is True
    assert edges.unresolved_for("m3") is False  # RESOLVED
    assert edges.unresolved_for("absent") is False


def test_notification_params_must_be_content_free() -> None:
    with pytest.raises(ValidationError):
        Notification(
            notification_id="n1",
            workspace_id="w",
            principal_id="alice",
            seq=1,
            category=NotificationCategory.SYNC,
            kind=NotificationKind.SYNC_STALLED,
            severity=NotificationSeverity.WARNING,
            params={"body": "your laptop last synced 3 days ago"},  # forbidden key
            source_event="DegradedModeEntered",
            plane="local",
            created_at=_NOW,
        )


def test_persona_profile_is_private_only() -> None:
    shared = Namespace.shared(org="o", workspace="w", session="s")
    with pytest.raises(ValidationError):
        PersonaProfile(
            namespace=shared,
            overall_brief="brief",
            brief_etag="etag",
            version=1,
            rebuilt_at=_NOW,
            source_memory_count=3,
        )
