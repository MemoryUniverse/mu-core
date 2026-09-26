"""``extract_valid_at`` — the CAPTURE-time date signal (AD-308 follow-up, team-lead review
2026-09-26), pure unit, no SPO decomposition involved. Complements
``test_ingest_valid_at_unit.py`` (which pins the wiring into ``_build_memory_item``) by pinning
the function's own sentence-iteration and "never guess" behaviour in isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_engine.services.extract import extract_valid_at

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 29, 15, 30, 0, tzinfo=UTC)


def test_relative_clause_resolves() -> None:
    assert extract_valid_at("I adopted a dog yesterday", now=NOW) == NOW - timedelta(days=1)


def test_absolute_clause_resolves() -> None:
    assert extract_valid_at("Ada moved to Berlin in 2021", now=NOW) == datetime(
        2021, 1, 1, tzinfo=UTC
    )


def test_no_temporal_clause_returns_none() -> None:
    assert extract_valid_at("Ada uses Postgres for the new service", now=NOW) is None


def test_multi_sentence_text_uses_the_first_resolvable_clause() -> None:
    text = "Ada uses Postgres. She moved to Berlin in 2021. She visited Paris in 2019."
    assert extract_valid_at(text, now=NOW) == datetime(2021, 1, 1, tzinfo=UTC)


def test_empty_text_returns_none() -> None:
    assert extract_valid_at("", now=NOW) is None


def test_only_whitespace_and_punctuation_returns_none() -> None:
    assert extract_valid_at("... !? ", now=NOW) is None
