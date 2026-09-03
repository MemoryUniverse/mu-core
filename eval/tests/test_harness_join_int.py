"""The harness's own instrument check, on the REAL stores. ZERO mocks (DEV-STANDARDS).

Every number this harness reports depends on ONE join: a recalled body must resolve back to the
LoCoMo turn id it was ingested from. If that join is wrong, every recall@k is wrong in a way that
looks like a ranking result — the worst possible failure for an evaluation tool, because it is
indistinguishable from the thing being measured.

So this test proves the join end to end against live Valkey/Qdrant/FalkorDB: ingest known turns
through the product's own ``LocalMemory.add``, recall one of them by its own words, and assert the
returned body resolves to the right ``dia_id``. It asserts nothing about ranking QUALITY — that is
what the harness measures, not what it asserts.

If a container is down this RAISES (BLOCKED, never faked), matching
``packages/mu-local/tests/test_local_roundtrip_int.py``.
"""

from __future__ import annotations

import uuid

import pytest
from mu_eval.corpus import ingest_conversation, local_memory_for
from mu_eval.locomo import Conversation, LabelledQuery, Turn
from mu_eval.runner import _resolve_ranked_ids

pytestmark = pytest.mark.integration

_TURNS = (
    ("D1:1", "Caroline", "Hey Mel, I finally adopted a rescue greyhound named Pepper."),
    ("D1:2", "Melanie", "That is wonderful, how old is Pepper?"),
    ("D1:3", "Caroline", "She is four, and she sleeps about nineteen hours a day."),
    ("D2:1", "Melanie", "I signed up for the Rotterdam half marathon in October."),
    ("D2:2", "Caroline", "My deploy passphrase for the ZEPHYR release is violet-anchor-77."),
)


def _conversation() -> Conversation:
    return Conversation(
        sample_id=f"harness{uuid.uuid4().hex[:6]}",
        speaker_a="Caroline",
        speaker_b="Melanie",
        turns=tuple(
            Turn(
                dia_id=dia_id,
                session_index=int(dia_id[1]),
                session_date="7 May 2023",
                speaker=speaker,
                text=text,
            )
            for dia_id, speaker, text in _TURNS
        ),
        queries=(
            LabelledQuery(
                query_id="q0",
                question="What is the deploy passphrase for the ZEPHYR release?",
                answer="violet-anchor-77",
                evidence=("D2:2",),
                category=4,
            ),
        ),
    )


async def test_recalled_body_resolves_to_its_source_turn_id() -> None:
    conversation = _conversation()
    gold = {"D2:2"}
    async with local_memory_for(conversation, run_id=f"j{uuid.uuid4().hex[:6]}") as memory:
        index, report = await ingest_conversation(
            memory, conversation, user="evaluser", session="s", importance=0.9
        )
        assert report.turns_written == len(_TURNS)
        assert report.promoted == len(_TURNS), (
            "importance 0.9 must clear IngestSettings.importance_promote (0.6) — if this fails the "
            "corpus never reached the vector tier and every downstream number is about the ingest "
            "gate, not about ranking"
        )

        result = await memory.recall(
            conversation.queries[0].question, user="evaluser", session="s", limit=10
        )
        assert result.items, "recall returned nothing from a live, populated partition"

        ranked = _resolve_ranked_ids(result.items, index, gold)
        assert len(ranked) == len(result.items), "every returned item must occupy a rank position"
        assert "D2:2" in ranked, (
            "the gold turn was returned by recall but did not resolve back to its dia_id — the "
            "harness join is broken, which would silently turn every recall@k into a wrong number"
        )
        # An item the harness never wrote must not be credited to some arbitrary turn.
        assert all(
            r.startswith("__not_in_corpus__") or r in {d for d, _, _ in _TURNS} for r in ranked
        )


async def test_a_body_the_harness_never_wrote_is_never_credited() -> None:
    """The join must be exact, not fuzzy — a near-miss body is a MISS, not a hit.

    Without this, a distilled LTM fact ("Caroline owns Pepper") derived from a gold turn could be
    silently credited as if the gold turn itself had been retrieved, inflating recall@k for a
    system that never returned the labelled evidence.
    """
    conversation = _conversation()
    async with local_memory_for(conversation, run_id=f"j{uuid.uuid4().hex[:6]}") as memory:
        index, _ = await ingest_conversation(
            memory, conversation, user="evaluser", session="s", importance=0.9
        )
        exact = "Caroline: She is four, and she sleeps about nineteen hours a day."
        assert index.resolve(exact) == ["D1:3"]
        assert index.resolve("Caroline owns a greyhound named Pepper") == []


async def test_consolidate_flag_actually_populates_the_ltm_graph_tier() -> None:
    """Mutation check for the ``consolidate=True`` wiring (RETRIEVAL-EVAL-0829.md follow-up,
    2026-08-31): every baseline/answer-quality run to date called ``LocalMemory.add`` only, so
    ``LocalMemory.consolidate()`` (MTM->LTM DISTILL) never ran and the graph tier was
    UNCONDITIONALLY EMPTY for the whole measurement — the D6 multi-hop traversal arm
    (``ranker.py::_ltm_channel``) had nothing to traverse in every prior run. This pins the fix:
    with ``consolidate=True``, (a) the ingest report says so and reports a non-zero fact count,
    and (b) an ``LTM``-tier-only recall — which returns NOTHING unless the graph tier actually has
    rows — comes back non-empty. Reverting ``ingest_conversation``'s consolidate call (or leaving
    the flag unwired) fails this at (a) or (b) respectively.
    """
    from mu_engine.storage.domain.memory import MemoryTier

    conversation = _conversation()
    async with local_memory_for(conversation, run_id=f"j{uuid.uuid4().hex[:6]}") as memory:
        _, report = await ingest_conversation(
            memory, conversation, user="evaluser", session="s", importance=0.9, consolidate=True
        )
        assert report.consolidated is True
        assert report.facts_extracted > 0, (
            "consolidate=True ran but extracted zero facts from a 5-turn conversation with a "
            "clear SPO fact in it ('the deploy passphrase ... is violet-anchor-77') — the "
            "distill call is wired but not doing real work"
        )

        ltm_only = await memory.recall(
            "deploy passphrase", user="evaluser", session="s", limit=10, tier=MemoryTier.LTM
        )
        assert ltm_only.items, (
            "an LTM-tier-only recall returned nothing after consolidate() ran — the graph tier "
            "is still empty from this reader's point of view even though distill reported facts "
            "extracted, which is exactly the gap that left the multi-hop traversal arm untested "
            "in every prior baseline/answer-quality run"
        )


async def test_without_the_flag_the_ltm_tier_stays_empty_the_prior_behaviour() -> None:
    """The control for the mutation check above: ``consolidate`` defaults to ``False`` and MUST
    reproduce the harness's prior behaviour exactly (every run before 2026-08-31) — an LTM-only
    recall after a plain ``ingest_conversation`` call returns nothing, because nothing ever wrote
    the graph tier. If this goes green with items present, ``consolidate`` stopped defaulting to
    off, and every non-consolidate run this repo has on record silently changed meaning.
    """
    from mu_engine.storage.domain.memory import MemoryTier

    conversation = _conversation()
    async with local_memory_for(conversation, run_id=f"j{uuid.uuid4().hex[:6]}") as memory:
        _, report = await ingest_conversation(
            memory, conversation, user="evaluser", session="s", importance=0.9
        )
        assert report.consolidated is False
        assert report.facts_extracted == 0

        ltm_only = await memory.recall(
            "deploy passphrase", user="evaluser", session="s", limit=10, tier=MemoryTier.LTM
        )
        assert not ltm_only.items, (
            "the LTM tier had items with no consolidate() call anywhere in this run — "
            "consolidate's default flipped, or something else now writes the graph tier at add() "
            "time; either way this is no longer the harness's documented prior behaviour"
        )
