"""AD-308 follow-up (team-lead review, 2026-09-26): ``_build_memory_item`` — the ONE mint point
for a captured ``MemoryItem`` (CANONICAL §7.1) — must stamp ``valid_at`` from the raw capture's
OWN text when a temporal clause resolves, so a plain verbatim STM/MTM item (never LTM-promoted)
still carries a usable date. Before this fix, `valid_at` was unconditionally `None` for every
capture: date resolution only ever ran during MTM->LTM DISTILL (`pipelines/distill.py`), which
AD-310/AD-311 measured never wins a recall slot (0 of 3,000 items) — so the extracted signal never
reached an answer for essentially all of this engine's actual traffic.

Pure unit: `_build_memory_item` is a plain function of `(IngestActivity, at)` -> `MemoryItem`, no
store, no container (mirrors `test_ingest_turn_seq_unit.py`'s own "pure unit, no containers"
convention for this same module).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_engine.pipelines.concrete.ingest import IngestActivity, _build_memory_item

_NS = Namespace(org="o", workspace="w", user="u1", session="s1", visibility=Visibility.PRIVATE)
_AT = datetime(2026, 7, 29, 15, 30, 0, tzinfo=UTC)


def _activity(text: str) -> IngestActivity:
    return IngestActivity(namespace=_NS, host="test-host", session_offset="off-1", text=text)


def test_a_resolvable_relative_clause_stamps_valid_at() -> None:
    item = _build_memory_item(
        _activity("I went to a LGBTQ support group yesterday and it was so powerful"), at=_AT
    )
    assert item.valid_at == _AT - timedelta(days=1)


def test_a_resolvable_absolute_date_stamps_valid_at() -> None:
    item = _build_memory_item(_activity("Ada moved to Berlin in 2021"), at=_AT)
    assert item.valid_at == datetime(2021, 1, 1, tzinfo=UTC)


def test_no_resolvable_clause_leaves_valid_at_none_not_a_guess() -> None:
    """The capture-time resolver must NEVER fall back to `at`/`created_at` the way DISTILL's own
    LOUD fallback does for a PROMOTED fact — that would stamp today's wall-clock date onto the
    ~80%+ of captures that state no date at all, which is a WRONG date, not an honest unknown."""
    item = _build_memory_item(_activity("Ada uses Postgres for the new service"), at=_AT)
    assert item.valid_at is None


def test_content_is_never_altered_by_the_temporal_scan() -> None:
    """Unlike `decompose_to_spo` (which strips the temporal clause into the object cleanup), the
    capture-time path must leave `content` byte-identical — this is a verbatim STM record, not an
    extracted fact, and `content_hash` identity depends on it."""
    text = "I went to a LGBTQ support group yesterday and it was so powerful"
    item = _build_memory_item(_activity(text), at=_AT)
    assert item.content == text


def test_created_at_is_still_the_real_capture_instant_not_the_resolved_date() -> None:
    """`valid_at` (world-time) and `created_at` (transaction time) are bi-temporally distinct —
    resolving a WORLD-time date must never overwrite the actual capture instant."""
    item = _build_memory_item(_activity("Ada moved to Berlin in 2021"), at=_AT)
    assert item.created_at == _AT
    assert item.valid_at != item.created_at


def test_first_sentence_with_a_resolvable_clause_wins_over_a_later_one() -> None:
    item = _build_memory_item(
        _activity("Ada moved to Berlin in 2021. She also visited Paris in 2019."), at=_AT
    )
    assert item.valid_at == datetime(2021, 1, 1, tzinfo=UTC)
