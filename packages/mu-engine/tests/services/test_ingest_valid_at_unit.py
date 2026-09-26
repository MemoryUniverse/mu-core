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


def _activity(text: str, *, occurred_at: datetime | None = None) -> IngestActivity:
    return IngestActivity(
        namespace=_NS,
        host="test-host",
        session_offset="off-1",
        text=text,
        occurred_at=occurred_at,
    )


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


# ---------------------------------------------------------------------------------------------
# AD-312: `occurred_at` — the asserted WORLD-TIME anchor a caller who knows it (a backdated
# import, a benchmark harness replaying dated content) can supply. `None` (every test above) is
# byte-identical to pre-AD-312 behaviour; these tests pin the NEW behaviour specifically.
# ---------------------------------------------------------------------------------------------

_STORY_DATE = datetime(2023, 5, 7, tzinfo=UTC)  # a plausible LoCoMo-era date, NOT `_AT`


def test_relative_clause_anchors_on_occurred_at_not_the_real_capture_instant() -> None:
    """THE defect this field exists to fix: `_AT` is 2026, `_STORY_DATE` is 2023. Without
    `occurred_at`, "yesterday" would resolve to `_AT - 1 day` (2026) — wrong for a message
    actually said on `_STORY_DATE`."""
    item = _build_memory_item(
        _activity("I went to a support group yesterday", occurred_at=_STORY_DATE), at=_AT
    )
    assert item.valid_at == _STORY_DATE - timedelta(days=1)
    assert item.valid_at is not None
    assert item.valid_at.year == 2023


def test_no_in_text_clause_falls_back_to_occurred_at_itself() -> None:
    """No relative/absolute clause in the text at all — `occurred_at` alone is still the best
    available world-time signal (mem0's own harness stamps every memory with the session's real
    date the same way, `add.py:83`) rather than leaving `valid_at` `None`."""
    item = _build_memory_item(
        _activity("Ada uses Postgres for the new service", occurred_at=_STORY_DATE), at=_AT
    )
    assert item.valid_at == _STORY_DATE


def test_an_explicit_absolute_date_in_text_still_wins_over_occurred_at() -> None:
    """A more SPECIFIC in-text assertion ("in 2021") outranks the coarser session-level
    `occurred_at` — the item is about a fact from 2021, not about `occurred_at`'s own date."""
    item = _build_memory_item(
        _activity("Ada moved to Berlin in 2021", occurred_at=_STORY_DATE), at=_AT
    )
    assert item.valid_at == datetime(2021, 1, 1, tzinfo=UTC)


def test_occurred_at_absent_is_byte_identical_to_pre_ad312_behaviour() -> None:
    with_none = _build_memory_item(_activity("Ada uses Postgres", occurred_at=None), at=_AT)
    without_param = _build_memory_item(
        IngestActivity(
            namespace=_NS, host="test-host", session_offset="off-1", text="Ada uses Postgres"
        ),
        at=_AT,
    )
    assert with_none.valid_at is None
    assert without_param.valid_at is None
