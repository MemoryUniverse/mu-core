"""Unit tests for the S0-06 retention additions to ``MemoryItem`` (ADR 0035).

Covers: additive/backward-compatible defaults for ``retention_class``/``cold``,
``MemoryState.EXPIRED`` membership, absence of any ``valid_until`` field (F3), and a
``to_dict()``/``from_dict()`` round-trip extended to the two new fields — following the
existing round-trip pattern in ``tests/storage/test_mappers_unit.py``.
"""

from __future__ import annotations

import pytest

from mu_engine.storage.domain.memory import (
    MemoryItem,
    MemoryKind,
    MemoryState,
    Polarity,
    RetentionClass,
)
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit

_NS = Namespace(org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE)


def _item(**overrides: object) -> MemoryItem:
    defaults: dict[str, object] = {
        "content": "Ada uses Postgres",
        "kind": MemoryKind.PROPOSITION,
        "namespace": _NS,
        "owner_id": "u",
        "workspace_id": "w",
        "session_id": "s",
        "subject": "Ada",
        "predicate": "uses",
        "object": "Postgres",
        "polarity": Polarity.POSITIVE,
    }
    defaults.update(overrides)
    return MemoryItem(**defaults)  # type: ignore[arg-type]


def test_retention_class_and_cold_have_backward_compatible_defaults() -> None:
    """Existing callers (distill.py/qdrant_mtm.py/falkor_ltm.py) construct MemoryItem
    without naming the new fields — they must still get sane, valid defaults."""
    item = _item()
    assert isinstance(item.retention_class, RetentionClass)
    assert item.cold is False


def test_retention_class_enum_has_exactly_three_members() -> None:
    assert {m.value for m in RetentionClass} == {"permanent", "durable", "ephemeral"}


def test_retention_class_is_settable_to_each_member() -> None:
    for rc in (RetentionClass.PERMANENT, RetentionClass.DURABLE, RetentionClass.EPHEMERAL):
        item = _item(retention_class=rc)
        assert item.retention_class is rc


def test_cold_flag_is_settable() -> None:
    item = _item(cold=True)
    assert item.cold is True


def test_memory_state_gains_expired_alongside_existing_members() -> None:
    assert {m.value for m in MemoryState} == {
        "active",
        "archived",
        "superseded",
        "quarantined",
        "deleted",
        "expired",
    }
    assert MemoryState.EXPIRED.value == "expired"


def test_no_valid_until_field_exists_on_memory_item() -> None:
    """F3 (ADR 0035): EPHEMERAL end-of-validity reuses the EXISTING ``invalid_at``
    field — no second field is introduced."""
    assert "valid_until" not in MemoryItem.model_fields


def test_to_dict_from_dict_roundtrip_covers_retention_fields() -> None:
    """Extends the existing to_dict()/from_dict() round-trip pattern (test_mappers_unit.py)
    to the two new fields; byte-stable across the wire-form boundary (spec §6 invariant 7)."""
    item = _item(retention_class=RetentionClass.PERMANENT, cold=True)
    restored = MemoryItem.from_dict(item.to_dict())
    assert restored == item
    assert restored.retention_class is RetentionClass.PERMANENT
    assert restored.cold is True


def test_to_dict_from_dict_roundtrip_default_retention_fields() -> None:
    item = _item()
    restored = MemoryItem.from_dict(item.to_dict())
    assert restored == item
    assert restored.retention_class is item.retention_class
    assert restored.cold is item.cold


def test_to_dict_serializes_retention_class_as_plain_string() -> None:
    item = _item(retention_class=RetentionClass.EPHEMERAL)
    data = item.to_dict()
    assert data["retention_class"] == "ephemeral"
    assert data["cold"] is False
