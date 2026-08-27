"""The health/pin DTOs against CANONICAL §7.26 (memory-health §2.2 / §2.3).

The one assertion that matters here is a NEGATIVE one: ``MemoryHealthView`` is *"a content-free
read projection (counts/enums/timestamps)"* (CANONICAL §7.26), so no field on it may carry memory
text. Spec §2.2 line 110 declares ``MemoryHealthEntry.preview: str | None`` — a bounded content
snippet — and CANONICAL wins on conflict (spec §0 line 11). This test pins the resolution so a
later "the spec says preview" edit cannot quietly re-open it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.domain.model.health import (
    AT_RISK_FLAGS,
    MemoryHealthEntry,
    MemoryHealthFlag,
    MemoryHealthSummary,
    MemoryHealthView,
)
from mu_contracts.domain.model.memory import Namespace, State, Tier, Visibility
from mu_contracts.domain.model.pin import PinRequest, PinResult

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)

#: Field names that would make the projection content-bearing.
_FORBIDDEN = {"preview", "content", "text", "body", "snippet", "message", "summary_text"}


def _entry(**over: object) -> MemoryHealthEntry:
    base: dict[str, object] = {
        "memory_id": "mem_1",
        "tier": Tier.MTM,
        "state": State.ACTIVE,
        "flags": frozenset({MemoryHealthFlag.STALE}),
        "retention": 0.4,
        "last_seen": _T0,
        "pinned": False,
    }
    base.update(over)
    return MemoryHealthEntry(**base)  # type: ignore[arg-type]


def test_no_health_dto_carries_a_content_bearing_field() -> None:
    for model in (MemoryHealthEntry, MemoryHealthSummary, MemoryHealthView):
        assert _FORBIDDEN & set(model.model_fields) == set(), model.__name__


def test_preview_is_rejected_outright() -> None:
    """``extra='forbid'`` means a caller cannot smuggle the dropped field back in."""
    with pytest.raises(ValidationError):
        _entry(preview="the user prefers dark mode")


def test_entries_are_frozen_and_immutable() -> None:
    entry = _entry()
    with pytest.raises(ValidationError):
        entry.pinned = True  # type: ignore[misc]


def test_retention_is_bounded_to_the_curves_range() -> None:
    with pytest.raises(ValidationError):
        _entry(retention=1.5)


def test_pinned_and_archived_are_not_at_risk_flags() -> None:
    """They are status markers; the at-risk-only default must not surface an item for them."""
    assert MemoryHealthFlag.PINNED not in AT_RISK_FLAGS
    assert MemoryHealthFlag.ARCHIVED not in AT_RISK_FLAGS
    assert AT_RISK_FLAGS == frozenset(
        {
            MemoryHealthFlag.STALE,
            MemoryHealthFlag.LOW_CONFIDENCE,
            MemoryHealthFlag.CONFLICTING,
            MemoryHealthFlag.DECAYING,
        }
    )


def test_view_defaults_to_a_complete_empty_page() -> None:
    ns = Namespace(org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE)
    view = MemoryHealthView(
        namespace=ns,
        summary=MemoryHealthSummary(total=0, pinned_count=0),
        generated_at=_T0,
    )
    assert view.entries == ()
    assert view.partial is False
    assert view.next_cursor is None


def test_pin_request_bounds_the_reason_and_forbids_extras() -> None:
    assert PinRequest(memory_id="m").reason is None
    with pytest.raises(ValidationError):
        PinRequest(memory_id="m", reason="x" * 5000)
    with pytest.raises(ValidationError):
        PinRequest(memory_id="m", content="secret")  # type: ignore[call-arg]


def test_pin_result_is_frozen() -> None:
    result = PinResult(memory_id="m", pinned=True, pinned_at=_T0, version=1)
    with pytest.raises(ValidationError):
        result.pinned = False  # type: ignore[misc]
