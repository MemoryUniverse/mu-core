"""Unit tier for the mem0 head-to-head harness — pure, no stores, no `mem0ai`, no network.

`mem0_h2h.mem0_arm` imports `mem0` LAZILY (inside `build_memory`) exactly so this file can import
and test the two pieces that decide whether the head-to-head is fair at all:

  * `speaker_batches` — the reproduction of mem0's own LoCoMo batching. If this drifts from
    `evaluation/src/memzero/add.py`, the mem0 arm stops being "mem0 as they benchmark it".
  * the gold-turn provenance the arm's `gold_in_context` is joined on.

Both were wrong once in this lane's own first draft, in ways only a run caught (`Memory.search`
takes `top_k`, not `limit`, so every k in the first sweep silently ran at the default 20 —
`mem0/memory/main.py:1379-1391`). These tests are the cheap version of catching that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from mem0_h2h.mem0_arm import Meter, speaker_batches  # noqa: E402
from mu_eval.locomo import Conversation, LabelledQuery, Turn  # noqa: E402


def _conversation() -> Conversation:
    turns = [
        Turn(
            dia_id=f"D{session}:{index}",
            session_index=session,
            session_date=f"1:00 pm on {session} May, 2023",
            speaker="Caroline" if index % 2 == 1 else "Melanie",
            text=f"utterance s{session} t{index}",
        )
        for session in (1, 2)
        for index in range(1, 6)
    ]
    return Conversation(
        sample_id="conv-test",
        speaker_a="Caroline",
        speaker_b="Melanie",
        turns=tuple(turns),
    )


def test_batches_never_straddle_a_session() -> None:
    """mem0's `process_conversation` slices EACH SESSION's own message list (`add.py:88-118`).

    Batching across the session boundary would put two different `timestamp`s into one `add()`
    call, and a memory can only carry one — so the later session's facts would be stamped with the
    earlier session's date. That is a silent temporal corruption of the arm under test.
    """
    for batch in speaker_batches(_conversation(), for_speaker="Caroline", batch_size=4):
        sessions = {int(dia.split(":")[0][1:]) for dia in batch["dia_ids"]}
        assert len(sessions) == 1, f"batch straddles sessions {sessions}: {batch['dia_ids']}"
        assert batch["session_index"] in sessions


def test_every_turn_is_ingested_exactly_once_per_speaker() -> None:
    """No turn dropped, none duplicated — the corpus both arms see must be the same corpus."""
    conversation = _conversation()
    for batch_size in (1, 2, 3, 10):
        seen = [
            dia
            for batch in speaker_batches(
                conversation, for_speaker="Caroline", batch_size=batch_size
            )
            for dia in batch["dia_ids"]
        ]
        assert seen == [turn.dia_id for turn in conversation.turns], batch_size


def test_roles_invert_between_the_two_speaker_passes() -> None:
    """`add.py:104-111` builds `messages` and `messages_reverse`: the SAME turn is `user` in the
    pass that ingests its own speaker and `assistant` in the other. mem0 extracts preferentially
    from `user` turns, so collapsing the two passes into one role assignment would quietly change
    what their extractor is even shown."""
    conversation = _conversation()
    a = speaker_batches(conversation, for_speaker="Caroline", batch_size=1)
    b = speaker_batches(conversation, for_speaker="Melanie", batch_size=1)
    roles_a = [batch["messages"][0]["role"] for batch in a]
    roles_b = [batch["messages"][0]["role"] for batch in b]
    assert roles_a == [("user" if r == "assistant" else "assistant") for r in roles_b]
    assert set(roles_a) == {"user", "assistant"}


def test_speaker_name_is_prefixed_onto_every_message() -> None:
    """`add.py:105`: `f"{speaker}: {chat['text']}"`. LoCoMo questions name the speakers, so an
    unprefixed message makes half the corpus unattributable — for BOTH arms equally, which is
    exactly why neither arm may quietly do it differently."""
    for batch in speaker_batches(_conversation(), for_speaker="Caroline", batch_size=2):
        for message in batch["messages"]:
            assert message["content"].split(":", 1)[0] in {"Caroline", "Melanie"}


class _Response:
    def __init__(self, prompt: int, completion: int) -> None:
        self.model = "gpt-5-2025-08-07"
        self.usage = type(
            "U",
            (),
            {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "completion_tokens_details": type("D", (), {"reasoning_tokens": 0})(),
            },
        )()


def test_meter_aborts_the_run_when_measured_spend_crosses_the_cap() -> None:
    """The cap is the only thing standing between a mis-projected run and the owner's card.

    It must fire on MEASURED usage (the response's own block), not on a projection — this run's
    own projection was 6.5% low, and an earlier hand projection in this lane was out by 2x.
    """
    meter = Meter(cap_usd=0.01)
    meter.observe(_Response(prompt=1_000, completion=100))  # $0.00225 — under
    assert meter.cost_usd == pytest.approx(0.00225)
    with pytest.raises(RuntimeError, match="crossed the"):
        meter.observe(_Response(prompt=1_000, completion=1_000))  # cumulative $0.01450 — over
    assert meter.served_models == {"gpt-5-2025-08-07"}


class _RecordingMemory:
    """Stands in for `mem0.Memory`, recording exactly what the arm asked it for."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, query: str, **kwargs: object) -> dict:
        # The real `Memory.search` is keyword-only and does NOT accept `limit`; anything the arm
        # passes under the wrong name would land in ITS `**kwargs` and be dropped in silence.
        # Reproduce that exactly, so a wrong kwarg here fails loudly instead of measuring nothing.
        assert "limit" not in kwargs, "mem0.Memory.search takes top_k, not limit"
        self.calls.append({"query": query, **kwargs})
        return {"results": []}


def test_score_asks_mem0_for_the_width_it_was_given() -> None:
    """THE bug this lane shipped and then caught by running: `Memory.search(..., limit=k)`.

    `mem0/memory/main.py:1379-1391` is keyword-only with `top_k: int = 20`; `limit` is swallowed
    by `**kwargs`. The first k-sweep therefore ran EVERY point at top_k=20 and returned a flat
    `items=40.0` from k=1 to k=20 — a comparison that looked finished and measured one width.
    """
    from mem0_h2h.mem0_arm import score

    memory = _RecordingMemory()
    conversation = _conversation().model_copy(
        update={
            "queries": (
                LabelledQuery(
                    query_id="q1",
                    question="when?",
                    answer="2023",
                    evidence=("D1:1",),
                    category=2,
                ),
            )
        }
    )
    result = score(memory, conversation, memory_dia_ids={}, per_speaker_k=3)

    assert result["queries_scored"] == 1
    assert len(memory.calls) == 2, "both speakers' partitions must be searched (add.py:112-121)"
    assert [call["top_k"] for call in memory.calls] == [3, 3]
