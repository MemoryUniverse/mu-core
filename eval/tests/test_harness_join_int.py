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
