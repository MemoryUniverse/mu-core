"""Unit tests for the re-homed canonical per-verb return DTO family
(sdk-engine-server-design.md §2.5 "Return DTOs"; ``SDK-BUILD-DECISIONS.md`` Decision B; build-plan
Stage B task B0). Pure pydantic construction/validation, no store — all ``pytest.mark.unit``."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.contracts import (
    ConsolidateView,
    ContextView,
    MemoryResponse,
    MemoryWriteResult,
    RecallChannels,
    RecallItemView,
    RecallResult,
)
from mu_contracts.domain.model.memory import Namespace, Tier, Visibility

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_NS = Namespace(org="o", workspace="w", user="ada", session="s1", visibility=Visibility.PRIVATE)


def test_memory_response_field_set_is_unchanged_from_the_frozen_wire_schema() -> None:
    """Byte-parity check: the field NAMES on the re-homed MemoryResponse are the exact set
    documented at mu_sdk.models.memory.MemoryResponse (read verbatim before the move) — this is
    the frozen HTTP wire schema and must not gain/lose/rename a field."""
    expected = {
        "id",
        "content",
        "content_type",
        "tier",
        "state",
        "importance_score",
        "access_count",
        "created_at",
        "updated_at",
        "namespace",
        "metadata",
        "source",
        "speaker_kind",
        "speaker_id",
        "source_id",
        "session_id",
        "turn_id",
        "entity_id",
        "asserted_state",
        "gds_pagerank",
        "subject",
        "predicate",
        "object",
        "object_kind",
        "object_type",
        "object_value",
        "polarity",
        "predicate_cardinality",
        "valid_at",
        "invalid_at",
        "expires_at",
        "last_seen",
        "mention_count",
        "relevance_score",
        "parent_ids",
        "child_ids",
        "content_hash",
    }
    assert set(MemoryResponse.model_fields) == expected
    assert MemoryResponse.model_config.get("extra") == "forbid"


def test_memory_response_minimal_construction() -> None:
    resp = MemoryResponse(
        id="m1",
        content="hello",
        content_type="text",
        tier="stm",
        state="active",
        importance_score=0.5,
        access_count=0,
        created_at=_NOW,
        updated_at=_NOW,
        namespace="mu/o/w/private/ada/s1",
    )
    assert resp.content_hash == ""
    assert resp.parent_ids == []


def test_memory_write_result_requires_the_must_add_namespace_field() -> None:
    """Decision B MUST-ADD: namespace is required (no default) on the canonical add-receipt."""
    with pytest.raises(ValidationError):
        MemoryWriteResult(  # type: ignore[call-arg]
            memory_id="m1", content_hash="h1", promoted=True, tiers_written=("mtm",)
        )


def test_memory_write_result_events_emitted_defaults_empty() -> None:
    receipt = MemoryWriteResult(
        memory_id="m1",
        content_hash="h1",
        promoted=True,
        tiers_written=("mtm",),
        namespace="mu/o/w/private/ada/s1",
    )
    assert receipt.events_emitted == ()
    assert receipt.model_config.get("frozen") is True


def test_consolidate_view_requires_the_must_add_noop_field() -> None:
    """Decision B MUST-ADD: noop (DistillReport already exposes it; the pre-Decision-B embedded
    shape silently dropped it)."""
    with pytest.raises(ValidationError):
        ConsolidateView(facts_extracted=3, added=1, superseded=1)  # type: ignore[call-arg]
    view = ConsolidateView(facts_extracted=3, added=1, superseded=1, noop=1)
    assert view.noop == 1


def test_context_view_items_use_the_canonical_recall_item_view() -> None:
    """Decision B cross-cutting: ContextView.items and RecallResult.items share ONE hit-item
    type (RecallItemView), not two near-duplicate shapes."""
    item = RecallItemView(
        memory_id="m1", content="hi", tier=Tier.STM, channel="stm", fused_score=1.0
    )
    ctx = ContextView(text="- hi", items=[item], degraded=None)
    assert ctx.items[0] is item
    recall = RecallResult(
        namespace=_NS,
        items=[item],
        channels_run=RecallChannels(),
        degraded=None,
        generated_at=_NOW,
    )
    assert type(recall.items[0]) is type(ctx.items[0])


def test_recall_item_view_has_no_engine_internal_federate_dedup_fields() -> None:
    """Decision B recommendation, applied: the surface item drops content_hash + per-item
    namespace (mu_engine.services.recall.dto.RecallItemView carries those; this canonical
    surface shape deliberately does not)."""
    assert "content_hash" not in RecallItemView.model_fields
    assert "namespace" not in RecallItemView.model_fields


def test_recall_result_memory_ids_projection() -> None:
    item = RecallItemView(
        memory_id="m1", content="hi", tier=Tier.MTM, channel="mtm", fused_score=0.9
    )
    result = RecallResult(
        namespace=_NS,
        items=[item],
        channels_run=RecallChannels(),
        generated_at=_NOW,
    )
    assert result.memory_ids == ["m1"]
