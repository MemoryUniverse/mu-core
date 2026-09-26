"""AD-308: RELATIVE temporal clause resolution ("yesterday", "two months ago", "last Tuesday")
anchored on the per-item capture instant ``now`` — pure logic, no container, no model (mirrors
``test_extract_heuristic_unit.py``'s own fixed-``NOW`` convention; DEV-STANDARDS: no wall-clock
without a seed).

Before AD-308, ``decompose_to_spo`` only recovered an ABSOLUTE date/year behind an explicit
preposition (`_TEMPORAL_TAIL`); every relative expression fell through to
``valid_at=None``/``valid_at_inferred=True`` and the pipeline's LOUD ``recorded_at`` fallback
(``distill.py:626-630``). These tests pin the new behaviour against an exact resolved instant (not
just "is not None"), so a wrong unit-day constant or a wrong weekday-shift direction fails them.

Sentences deliberately use only the pre-existing COPULA pattern ("SUBJECT is/was [not] OBJECT") —
the one pattern every one of these relative clauses can be appended to without also exercising a
different decomposition rule, so a failure here isolates the temporal resolver, not the parser.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_engine.services.extract import HeuristicSpoExtractor
from mu_engine.storage.domain.memory import Polarity

pytestmark = pytest.mark.unit

# A KNOWN weekday, pinned deliberately (not `datetime.now()`) so weekday-relative assertions are
# exact and reviewable without a calendar lookup: 2026-07-29 is a Wednesday.
NOW = datetime(2026, 7, 29, 15, 30, 0, tzinfo=UTC)
assert NOW.weekday() == 2  # Wednesday — guards the fixture itself against a silent date typo


async def _extract(text: str):
    return await HeuristicSpoExtractor().extract(text, now=NOW)


async def test_yesterday() -> None:
    facts = await _extract("Ada was in Berlin yesterday")
    f = facts[0]
    assert (f.subject, f.predicate, f.object) == ("Ada", "is", "Berlin")
    assert f.valid_at == NOW - timedelta(days=1)
    assert f.valid_at_inferred is False


async def test_today() -> None:
    facts = await _extract("Ada is busy today")
    assert facts[0].valid_at == NOW


async def test_tomorrow_with_negation() -> None:
    facts = await _extract("Ada is not available tomorrow")
    f = facts[0]
    assert f.polarity is Polarity.NEGATIVE
    assert f.object == "available"
    assert f.valid_at == NOW + timedelta(days=1)


async def test_n_days_ago() -> None:
    facts = await _extract("Ada was in Berlin 3 days ago")
    assert facts[0].valid_at == NOW - timedelta(days=3)


async def test_n_months_ago() -> None:
    facts = await _extract("Ada was in Berlin two months ago")
    assert facts[0].valid_at == NOW - timedelta(days=60)


async def test_a_year_ago_indefinite_article() -> None:
    facts = await _extract("Ada was busy a year ago")
    assert facts[0].valid_at == NOW - timedelta(days=365)


async def test_about_n_weeks_ago() -> None:
    facts = await _extract("Ada was in London about 6 weeks ago")
    assert facts[0].valid_at == NOW - timedelta(days=42)


async def test_last_weekday_same_weekday_as_now_shifts_a_full_week() -> None:
    # NOW is a Wednesday; "last Wednesday" must NOT resolve to NOW itself.
    facts = await _extract("Ada was busy last Wednesday")
    expected = NOW.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=7)
    assert facts[0].valid_at == expected


async def test_last_weekday_before_now() -> None:
    # NOW is Wednesday 2026-07-29; the most recent Monday before it is 2026-07-27.
    facts = await _extract("Ada was in Paris last Monday")
    assert facts[0].valid_at == datetime(2026, 7, 27, tzinfo=UTC)


async def test_next_weekday_after_now() -> None:
    # NOW is Wednesday 2026-07-29; the next Friday is 2026-07-31.
    facts = await _extract("Ada is busy next Friday")
    assert facts[0].valid_at == datetime(2026, 7, 31, tzinfo=UTC)


async def test_last_week() -> None:
    facts = await _extract("Ada was busy last week")
    assert facts[0].valid_at == NOW - timedelta(days=7)


async def test_next_month() -> None:
    facts = await _extract("Ada is busy next month")
    assert facts[0].valid_at == NOW + timedelta(days=30)


async def test_last_year() -> None:
    facts = await _extract("Ada was busy last year")
    assert facts[0].valid_at == NOW - timedelta(days=365)


async def test_relative_clause_never_leaks_into_the_object() -> None:
    facts = await _extract("Ada was in Berlin yesterday")
    assert facts[0].object == "Berlin"
    assert "yesterday" not in facts[0].object


async def test_absolute_date_still_takes_priority_and_is_unaffected() -> None:
    # Absolute-date path (`_TEMPORAL_TAIL`) is untouched by AD-308's relative resolver — this is
    # a non-regression pin on the pre-existing behaviour, not a new one.
    facts = await _extract("Ada moved to Berlin in 2021")
    assert facts[0].valid_at == datetime(2021, 1, 1, tzinfo=UTC)


async def test_unrecognised_relative_phrase_stays_inferred() -> None:
    # "a while ago" is deliberately OUTSIDE the closed vocabulary — never a silent guess.
    facts = await _extract("Ada was busy a while ago")
    assert facts[0].valid_at is None
    assert facts[0].valid_at_inferred is True


async def test_relative_ago_wrong_unit_multiplier_is_caught() -> None:
    """Mutation check: swapping `_RELATIVE_UNIT_DAYS["month"]` for a wrong constant (e.g. 31)
    must flip this assertion, proving the test is pinned to the real value (30), not a tautology.
    """
    facts = await _extract("Ada was in Berlin two months ago")
    assert facts[0].valid_at == NOW - timedelta(days=60)
    assert facts[0].valid_at != NOW - timedelta(days=62)
