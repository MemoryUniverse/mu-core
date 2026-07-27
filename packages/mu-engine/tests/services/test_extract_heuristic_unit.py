"""Unit tests for the deterministic heuristic SPO extractor (MVP default).

Pure logic, no container, no model — the extractor is offline and deterministic
(DEV-STANDARDS: unit tests may run pure logic; no wall-clock without a seed — the fixed
``NOW`` is injected).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_engine.services.extract import HeuristicSpoExtractor, decompose_to_spo
from mu_engine.storage.domain.memory import Polarity

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 27, 12, 0, 0, tzinfo=UTC)


async def _extract(text: str):
    return await HeuristicSpoExtractor().extract(text, now=NOW)


async def test_copula_is() -> None:
    facts = await _extract("John is an engineer")
    assert len(facts) == 1
    f = facts[0]
    assert (f.subject, f.predicate, f.object) == ("John", "is", "engineer")
    assert f.polarity is Polarity.POSITIVE


async def test_copula_negation_flips_polarity() -> None:
    facts = await _extract("Ada is not happy")
    assert len(facts) == 1
    assert facts[0].predicate == "is"
    assert facts[0].object == "happy"
    assert facts[0].polarity is Polarity.NEGATIVE


async def test_prepositional_verb_predicate_is_verb_prep() -> None:
    facts = await _extract("Ada lives in Paris")
    assert (facts[0].subject, facts[0].predicate, facts[0].object) == ("Ada", "lives_in", "Paris")


async def test_transitive_verb() -> None:
    facts = await _extract("Ada uses Postgres")
    assert (facts[0].subject, facts[0].predicate, facts[0].object) == ("Ada", "uses", "Postgres")


async def test_possessive_attribute() -> None:
    facts = await _extract("Ada's favorite color is blue")
    assert (facts[0].subject, facts[0].predicate, facts[0].object) == (
        "Ada",
        "favorite_color",
        "blue",
    )


async def test_trailing_date_sets_valid_at_not_inferred() -> None:
    facts = await _extract("Ada moved to Berlin in 2021")
    f = facts[0]
    assert (f.subject, f.predicate, f.object) == ("Ada", "moved_to", "Berlin")
    assert f.valid_at == datetime(2021, 1, 1, tzinfo=UTC)
    assert f.valid_at_inferred is False


async def test_iso_date() -> None:
    facts = await _extract("Ada works at Globex since 2022-03-01")
    f = facts[0]
    assert f.predicate == "works_at"
    assert f.object == "Globex"
    assert f.valid_at == datetime(2022, 3, 1, tzinfo=UTC)
    assert f.valid_at_inferred is False


async def test_no_date_marks_inferred() -> None:
    facts = await _extract("Ada uses Postgres")
    assert facts[0].valid_at is None
    assert facts[0].valid_at_inferred is True


async def test_chitchat_and_noise_dropped() -> None:
    # "Hi." — below the min-token floor; expletive-subject sentence — no durable identity.
    assert await _extract("Hi.") == []
    assert await _extract("There are branches in trees") == []


async def test_multiple_sentences_split() -> None:
    facts = await _extract("Ada lives in Paris. Ada uses Postgres.")
    preds = sorted(f.predicate for f in facts)
    assert preds == ["lives_in", "uses"]


async def test_source_span_points_into_text() -> None:
    text = "Ada lives in Paris. Ada uses Postgres."
    facts = decompose_to_spo(text, now=NOW)
    for f in facts:
        assert f.source_span is not None
        start, end = f.source_span
        assert text[start:end].strip().startswith("Ada")


async def test_determinism_same_input_same_output() -> None:
    a = await _extract("Ada lives in Paris. Ada works at Acme.")
    b = await _extract("Ada lives in Paris. Ada works at Acme.")
    assert [f.model_dump() for f in a] == [f.model_dump() for f in b]
