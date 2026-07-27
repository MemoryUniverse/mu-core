"""MemoryItem aggregate — id-stability + pure lifecycle transitions (memory-layer §1, §7.1)."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.domain.model import (
    MemoryItem,
    MemoryKind,
    Namespace,
    State,
    Tier,
    Validity,
    Visibility,
)

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _item() -> MemoryItem:
    return MemoryItem(
        id="mem_1",
        namespace=Namespace(
            org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
        ),
        kind=MemoryKind.PROPOSITION,
        content="the user prefers dark mode",
        validity=Validity(valid_at=_NOW, recorded_at=_NOW),
        last_seen=_NOW,
        provenance_id="prov_1",
    )


def test_provenance_id_required_non_empty() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(
            id="mem_1",
            namespace=_item().namespace,
            kind=MemoryKind.PROPOSITION,
            content="c",
            validity=Validity(valid_at=_NOW, recorded_at=_NOW),
            last_seen=_NOW,
            provenance_id="",  # min_length=1 — a null/empty provenance is a bug
        )


def test_id_carried_unchanged_across_tiers() -> None:
    item = _item()
    promoted = item.with_tier(Tier.MTM, at=_NOW)
    assert promoted.id == item.id  # §7.1 — id minted once, never re-minted on promote
    assert promoted.tier is Tier.MTM


def test_can_promote_to_adjacency_only() -> None:
    item = _item()
    assert item.can_promote_to(Tier.MTM) is True
    assert item.can_promote_to(Tier.LTM) is False  # not adjacent from STM
    assert item.with_tier(Tier.MTM, at=_NOW).can_promote_to(Tier.LTM) is True


def test_superseded_by_closes_bitemporal_window() -> None:
    later = datetime(2026, 2, 1, tzinfo=UTC)
    loser = _item().superseded_by("mem_2", at=later)
    assert loser.state is State.SUPERSEDED
    assert loser.validity.invalid_at == later
    assert loser.validity.valid_at == _NOW  # world-start unchanged


def test_reinforced_bumps_counts() -> None:
    item = _item()
    r = item.reinforced(at=_NOW, importance=0.9)
    assert r.access_count == item.access_count + 1
    assert r.mention_count == item.mention_count + 1
    assert r.importance == 0.9


def test_archived_and_quarantined_transitions() -> None:
    assert _item().archived(at=_NOW).state is State.ARCHIVED
    assert _item().quarantined(at=_NOW, reason="low_confidence").state is State.QUARANTINED


def test_importance_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(
            id="x",
            namespace=_item().namespace,
            kind=MemoryKind.PROPOSITION,
            content="c",
            validity=Validity(valid_at=_NOW, recorded_at=_NOW),
            last_seen=_NOW,
            provenance_id="p",
            importance=1.5,
        )
