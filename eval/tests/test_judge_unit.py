"""Unit tests for the compact, budget-fit judge prompt (Azure Ministral-3B: 400 tokens total,
prompt + completion). Each assertion is checked against a MEASURED failure mode, not a hope:

  * the truncation caps are pointless if a long field still comes out long — tested by feeding
    fields well past every cap and checking the ENCODED result, not just character count (a
    truncated string can still be long in characters if the fallback ratio is wrong).
  * the whole point of this module is fitting a hard ceiling — tested end-to-end with the
    dataset's own measured worst case (question/gold-answer token lengths from
    ``token_budget.count_tokens``'s docstring), not a hand-picked short example that would pass
    by accident.

No stores, no network required beyond the tokenizer's local HF cache (already warm on this
machine) — this file needs neither the VM nor an Azure credential.
"""

from __future__ import annotations

import pytest
from mu_eval.judge import (
    COMPACT_JUDGE_SYSTEM_PROMPT,
    MINISTRAL_GOLD_ANSWER_MAX_TOKENS,
    MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS,
    MINISTRAL_QUESTION_MAX_TOKENS,
    MINISTRAL_RESPONSE_MAX_TOKENS,
    compact_judge_prompt,
    estimated_prompt_tokens,
)
from mu_eval.token_budget import count_tokens, truncate_to_tokens

pytestmark = pytest.mark.unit

# The 400-token ceiling is prompt + completion TOGETHER (measured from live Azure response
# headers — see openai_chat.py docstring). Budgeting to 300 for the prompt leaves >90 tokens of
# headroom for the ~6-token completion cap, which is the margin this whole module exists to buy.
_PROMPT_BUDGET = 300


def test_truncate_to_tokens_actually_shortens_a_long_field() -> None:
    long_text = "word " * 500  # ~2500 chars, certainly over any of the per-field caps
    truncated = truncate_to_tokens(long_text, MINISTRAL_RESPONSE_MAX_TOKENS)
    assert truncated != long_text
    assert count_tokens(truncated) <= MINISTRAL_RESPONSE_MAX_TOKENS


def test_truncate_to_tokens_is_a_noop_under_the_cap() -> None:
    short = "7 May 2023"
    assert truncate_to_tokens(short, MINISTRAL_GOLD_ANSWER_MAX_TOKENS) == short


def test_compact_prompt_fits_budget_on_a_short_realistic_row() -> None:
    question = "When did Caroline go to the LGBTQ support group?"
    gold = "7 May 2023"
    user = compact_judge_prompt(question=question, gold_answer=gold, response=gold)
    total = estimated_prompt_tokens(COMPACT_JUDGE_SYSTEM_PROMPT, user)
    assert total <= _PROMPT_BUDGET


def test_compact_prompt_fits_budget_even_at_every_fields_worst_case() -> None:
    # Dataset-measured worst case (mu_eval.locomo, locomo10.json, all 1540 scoreable rows):
    # question tokens max 33, gold-answer tokens max 70. Generated answers are unbounded, so this
    # exercises a response several times past its own cap too.
    question = "word " * 40  # well past the measured max-33-token question
    gold = "word " * 80  # well past the measured max-70-token gold answer
    response = "word " * 200  # a generated answer that ignored the "5-6 words" instruction
    user = compact_judge_prompt(question=question, gold_answer=gold, response=response)
    total = estimated_prompt_tokens(COMPACT_JUDGE_SYSTEM_PROMPT, user)
    assert total <= _PROMPT_BUDGET
    # And the total leaves room for the capped completion under the real 400 ceiling.
    assert total + MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS < 400


def test_compact_prompt_per_field_caps_are_enforced_independently() -> None:
    question = "q " * 100
    gold = "g " * 100
    response = "r " * 100
    user = compact_judge_prompt(question=question, gold_answer=gold, response=response)
    # Each field's contribution to the user turn cannot exceed its own cap (plus the fixed label
    # text "Question: "/"Gold: "/"Generated: " and newlines, counted here via the whole-string
    # budget rather than re-deriving field boundaries).
    assert count_tokens(user) <= (
        MINISTRAL_QUESTION_MAX_TOKENS
        + MINISTRAL_GOLD_ANSWER_MAX_TOKENS
        + MINISTRAL_RESPONSE_MAX_TOKENS
        + 10  # generous slack for "Question: \nGold: \nGenerated: " label/newline tokens
    )


def test_completion_cap_covers_a_bare_correct_or_wrong() -> None:
    # Measured on the Ministral-family tokenizer: "CORRECT" encodes to 3 tokens, " WRONG" to 2.
    # The cap must clear the larger of the two with margin for a stray leading space/punctuation.
    assert count_tokens("CORRECT") <= MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS
    assert count_tokens("WRONG") <= MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS
