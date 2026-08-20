"""Unit tests for the deterministic heuristic SPO extractor (MVP default).

Pure logic, no container, no model — the extractor is offline and deterministic
(DEV-STANDARDS: unit tests may run pure logic; no wall-clock without a seed — the fixed
``NOW`` is injected).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_engine.services.extract import ExtractionSettings, HeuristicSpoExtractor, decompose_to_spo
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


async def test_settings_min_tokens_is_actually_consumed_by_the_extractor() -> None:
    """CONFIG-AND-DATA-FIX-PLAN.md §1.2 C2: ``ExtractionSettings.min_tokens`` MUST reach
    ``HeuristicSpoExtractor`` via its ``settings=`` constructor arg — the exact param the
    composition roots now thread as ``HeuristicSpoExtractor(settings=engine_settings.extraction)``
    instead of the previous bare ``HeuristicSpoExtractor()`` (which always fell back to
    ``ExtractionSettings()``'s bare ``min_tokens=3``, unreachable from the environment).

    ``"Ada is happy"`` is a 3-token sentence with a real copula pattern — at the DEFAULT
    ``min_tokens=3`` it extracts; raising the floor to ``min_tokens=4`` via an explicit
    ``settings=`` must now drop it as chit-chat noise, proving the constructor arg — not just the
    class default — decides.
    """
    text = "Ada is happy"
    default_facts = await HeuristicSpoExtractor().extract(text, now=NOW)
    assert len(default_facts) == 1  # min_tokens=3 default: 3-token sentence clears the floor

    raised_floor = HeuristicSpoExtractor(settings=ExtractionSettings(min_tokens=4))
    raised_facts = await raised_floor.extract(text, now=NOW)
    assert raised_facts == [], (
        "min_tokens=4 (passed via settings=) did not reach the extractor — the 3-token sentence "
        f"still extracted: {raised_facts!r}"
    )


# --------------------------------------------------------------------------- D-7 canonicalization
# REGRESSION: `_canonicalize_predicate` existed and was documented as the D-7 predicate-
# normalization seam, but the prepositional-verb and transitive-verb branches minted their
# predicate DIRECTLY and never called it — so the canonical map was dead on exactly the branches
# that produce `lives_in`/`resides_in`/`works_at`/`worked_at`. Those are the tense/synonym variants
# `DistillSettings.functional_predicates` already enumerates as ONE relation, so leaving them
# un-unified meant two surface forms of the same fact never collided on (subject, predicate) in
# `find_conflicts` and BOTH stayed active — the "MU keeps stale truth" symptom.


@pytest.mark.parametrize(
    ("text", "expected_predicate", "expected_object"),
    [
        ("Ada lives in Paris", "lives_in", "Paris"),
        ("Ada resides in Madrid", "lives_in", "Madrid"),  # synonym -> same canonical predicate
        ("Ada lived in Rome", "lives_in", "Rome"),  # past tense -> same canonical predicate
        ("Ada works at Acme", "works_at", "Acme"),
        ("Ada worked at Globex", "works_at", "Globex"),  # past tense -> same canonical predicate
    ],
)
async def test_tense_and_synonym_variants_canonicalize_to_one_predicate(
    text: str, expected_predicate: str, expected_object: str
) -> None:
    facts = await _extract(text)
    assert len(facts) == 1
    assert (facts[0].predicate, facts[0].object) == (expected_predicate, expected_object)


async def test_ambiguous_semantic_verbs_are_not_canonicalized_by_default() -> None:
    """`moved_to` is deliberately NOT folded into `lives_in` by default: "moved to" is ambiguous
    across domains ("Ada moved to Berlin" = residence, "the standup moved to 11am" = schedule), so
    defaulting it would manufacture FALSE supersessions — strictly worse for a memory system than
    missing one. It stays a distinct predicate unless a deployment opts in explicitly.
    """
    facts = await _extract("Ada moved to Berlin")
    assert len(facts) == 1
    assert facts[0].predicate == "moved_to"


async def test_operator_supplied_canonical_map_reaches_the_verb_branches() -> None:
    """The documented opt-in (`MU_EXTRACTION__CANONICAL_PREDICATE_MAP`) must actually reach the
    prepositional-verb branch — this is what makes the deliberate default above safe to keep.
    """
    settings = ExtractionSettings(canonical_predicate_map={"moved_to": "lives_in"})
    facts = decompose_to_spo("Ada moved to Berlin", now=NOW, settings=settings)
    assert len(facts) == 1
    assert (facts[0].predicate, facts[0].object) == ("lives_in", "Berlin")


# ------------------------------------------------------------------------------- the NOISE GATE
# An agent transcript is mostly INSTRUCTIONS and QUESTIONS, not assertions. Live-reproduced on the
# real hook path: the deterministic decomposer stored "Do not use any tools." as the fact
# ('Do', 'use', 'any tools') — the exact INVERSE of what the sentence said, because negation
# stripping had already removed the "not" — and turned "What is the passphrase?" into a fact whose
# subject was "What". A memory system that confidently asserts "Do use any tools" back to an agent
# is worse than one that remembers nothing.


@pytest.mark.parametrize(
    "instruction",
    [
        "Do not use any tools.",
        "Acknowledge in one short sentence.",
        "Answer with just the passphrase, or say UNKNOWN.",
        "Reply with only the memory_id.",
        "Please summarize the file.",
        "Run this exact command and report ONLY its output lines.",
        "Use the memory-universe MCP tool recall.",
        "Remember to check the logs.",
        "Explain the tradeoff in two sentences.",
    ],
)
async def test_instructions_are_never_stored_as_facts(instruction: str) -> None:
    assert await _extract(instruction) == [], (
        "an imperative was stored as an assertion — a fact headed by an imperative auxiliary is a "
        "parse failure, not a memory"
    )


@pytest.mark.parametrize(
    "question",
    [
        "What is the deploy passphrase?",
        "Who is Ada's manager?",
        "When is the standup?",
        "Where does Ada live?",
        "Why did the build fail?",
        "Which model should we use?",
    ],
)
async def test_questions_are_never_stored_as_facts(question: str) -> None:
    """A question ASSERTS nothing. Storing one produces a 'fact' whose subject is an interrogative
    pronoun, which can then be recalled and injected as though it were knowledge."""
    assert await _extract(question) == []


async def test_a_question_does_not_swallow_the_clause_after_it() -> None:
    """`?` is a sentence terminator like `.`. Before it was one, the object of a question ran
    straight past the question mark and absorbed the following instruction — live: the object
    became 'deploy passphrase for the ZEPHYR release? Answer with just the passphrase, or say
    UNKNOWN'."""
    facts = await _extract(
        "What is the deploy passphrase for the ZEPHYR release? "
        "Answer with just the passphrase, or say UNKNOWN. Do not use any tools."
    )
    assert facts == []


async def test_the_real_fact_in_an_instruction_wrapped_prompt_still_survives() -> None:
    """The gate must be a SCALPEL, not a mute button: the assertion inside a prompt that also
    carries instructions is exactly the thing worth remembering."""
    facts = await _extract(
        "The deploy passphrase for the ZEPHYR release is 'violet-anchor-77'. "
        "Acknowledge in one short sentence. Do not use any tools."
    )
    assert len(facts) == 1
    assert facts[0].object == "'violet-anchor-77'"
    assert "passphrase" in facts[0].subject


async def test_a_discourse_prefix_is_stripped_so_the_same_fact_collides() -> None:
    """The other half of the duplicate problem. An assistant's "Noted — the passphrase is X" used
    to yield the subject "Noted — the passphrase", which shares no identity with the user's own
    "the passphrase" — so the SAME fact stored twice and never superseded."""
    assistant = await _extract("Noted — the ZEPHYR deploy passphrase is 'violet-anchor-77'.")
    user = await _extract("the ZEPHYR deploy passphrase is 'violet-anchor-77'.")

    assert len(assistant) == 1 and len(user) == 1
    assert assistant[0].subject == user[0].subject, (
        "the same fact stated conversationally does not collide with its plain form — this is why "
        "the demo stored the passphrase five times"
    )


async def test_a_subject_merely_starting_with_a_marker_word_is_untouched() -> None:
    """Only a marker followed by a separator is a discourse prefix. 'Note taking' is a subject."""
    facts = await _extract("Note taking is hard")
    assert len(facts) == 1
    assert facts[0].subject == "Note taking"
