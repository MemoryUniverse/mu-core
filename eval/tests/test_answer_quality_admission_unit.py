"""``_eligible_queries`` — T4 (``TRACE-0923.md`` §7), the ``answer_quality.py`` half of the same
defect ``test_runner_admission_unit.py`` covers for ``runner.py``.

Before the fix, ``_eligible_queries`` read ``if query.is_adversarial or not query.evidence or not
query.answer: skipped_adversarial += 1`` — a non-adversarial row with no evidence (or no answer
text) was counted as adversarial, with no way to tell the two reasons apart in
``AnswerQualityReport``. Pure/no stores/no LLM: ``_eligible_queries`` only reads the in-memory
``Conversation``.
"""

from __future__ import annotations

from mu_eval.answer_quality import _eligible_queries
from mu_eval.locomo import ADVERSARIAL_CATEGORY, Conversation, LabelledQuery, Turn

_TURNS = (
    Turn(dia_id="D1:1", session_index=1, session_date="7 May 2023", speaker="A", text="hi"),
    Turn(dia_id="D1:2", session_index=1, session_date="7 May 2023", speaker="B", text="hey"),
)


def _conv(queries: tuple[LabelledQuery, ...]) -> Conversation:
    return Conversation(
        sample_id="conv-99", speaker_a="A", speaker_b="B", turns=_TURNS, queries=queries
    )


def test_adversarial_row_counted_as_adversarial_only() -> None:
    q = LabelledQuery(
        query_id="q0", question="?", answer="", evidence=(), category=ADVERSARIAL_CATEGORY
    )
    eligible, skipped_adversarial, skipped_no_evidence, skipped_no_gold = _eligible_queries(
        _conv((q,)), None
    )
    assert (len(eligible), skipped_adversarial, skipped_no_evidence, skipped_no_gold) == (
        0,
        1,
        0,
        0,
    )


def test_non_adversarial_empty_evidence_counted_as_no_evidence_not_adversarial() -> None:
    """The exact defect: category 3, not adversarial, empty evidence."""
    q = LabelledQuery(query_id="q0", question="?", answer="something", evidence=(), category=3)
    eligible, skipped_adversarial, skipped_no_evidence, skipped_no_gold = _eligible_queries(
        _conv((q,)), None
    )
    assert (len(eligible), skipped_adversarial, skipped_no_evidence, skipped_no_gold) == (
        0,
        0,
        1,
        0,
    )


def test_evidence_naming_only_unknown_turns_is_no_gold() -> None:
    q = LabelledQuery(
        query_id="q0", question="?", answer="something", evidence=("D9:99",), category=4
    )
    eligible, skipped_adversarial, skipped_no_evidence, skipped_no_gold = _eligible_queries(
        _conv((q,)), None
    )
    assert (len(eligible), skipped_adversarial, skipped_no_evidence, skipped_no_gold) == (
        0,
        0,
        0,
        1,
    )


def test_eligible_row_is_scored() -> None:
    q = LabelledQuery(
        query_id="q0", question="?", answer="something", evidence=("D1:1",), category=4
    )
    eligible, skipped_adversarial, skipped_no_evidence, skipped_no_gold = _eligible_queries(
        _conv((q,)), None
    )
    assert (len(eligible), skipped_adversarial, skipped_no_evidence, skipped_no_gold) == (
        1,
        0,
        0,
        0,
    )
