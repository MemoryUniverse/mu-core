"""``classify_query_admission`` — T4 (``TRACE-0923.md`` §7): a non-adversarial row with EMPTY
``evidence`` must be counted as ``"no_evidence"``, never folded into ``"adversarial"``.

Before the fix, ``runner.py:246`` read ``if query.is_adversarial or not query.evidence:
skipped_adversarial += 1`` — a category-3 (open domain) row shipping ``evidence: []`` (real rows:
conv-26 qa[30]/qa[46]) walked the SAME branch as a true category-5 adversarial row, so a run's own
printed ``skipped_adversarial`` overcounted by exactly the number of such rows (49 reported vs 47
actually adversarial on conv-26) with no separate counter to show where the extra two came from.
Pure/no stores: this is the classification decision only, not the recall round trip.
"""

from __future__ import annotations

from mu_eval.locomo import ADVERSARIAL_CATEGORY, LabelledQuery
from mu_eval.runner import classify_query_admission

_KNOWN = {"D1:1", "D1:2", "D2:1"}


def _query(*, evidence: tuple[str, ...], category: int) -> LabelledQuery:
    return LabelledQuery(
        query_id="conv-99::q0",
        question="q",
        answer="a",
        evidence=evidence,
        category=category,
    )


def test_adversarial_row_is_classified_adversarial() -> None:
    q = _query(evidence=(), category=ADVERSARIAL_CATEGORY)
    assert classify_query_admission(q, _KNOWN) == "adversarial"


def test_non_adversarial_row_with_no_evidence_is_no_evidence_not_adversarial() -> None:
    """The exact defect: category 3 (open domain), NOT adversarial, empty evidence."""
    q = _query(evidence=(), category=3)
    assert classify_query_admission(q, _KNOWN) == "no_evidence"


def test_evidence_naming_only_unknown_turns_is_no_gold_in_corpus() -> None:
    q = _query(evidence=("D9:99",), category=4)
    assert classify_query_admission(q, _KNOWN) == "no_gold_in_corpus"


def test_evidence_naming_a_known_turn_is_eligible() -> None:
    q = _query(evidence=("D1:1",), category=4)
    assert classify_query_admission(q, _KNOWN) is None


def test_partial_evidence_overlap_is_still_eligible() -> None:
    """One gold turn known, one not — still eligible (scored against the known subset), distinct
    from the all-unknown case above."""
    q = _query(evidence=("D1:1", "D9:99"), category=4)
    assert classify_query_admission(q, _KNOWN) is None
