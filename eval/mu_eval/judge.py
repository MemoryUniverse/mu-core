# ruff: noqa: E501, RUF001, RUF002, RUF003
# ^ DELIBERATE, and the whole point of this module. The two prompt strings below are VERBATIM
#   ports of the official LoCoMo answering/grading prompts (MemOS prompts.py:1-38 and
#   locomo_eval.py:51-81) — including their over-100-column lines and their typographic
#   apostrophes (’CORRECT’). Reflowing a line or normalising a quote character EDITS THE METRIC:
#   the prompt IS the scorer here, so lint may not touch it. Scoped to this file only.
"""The OFFICIAL LoCoMo answer methodology, ported verbatim — answer prompt + LLM judge.

CODE-ADOPTION (CLAUDE.md rule 1: work from the ACTUAL cloned source, cite ``file:line``, never
guess). Both strings below are copied CHARACTER-FOR-CHARACTER from the cloned reference
checkout; nothing was paraphrased, shortened, or "improved":

  * ``ANSWER_PROMPT`` — port of ``ANSWER_PROMPT_MEM0`` at
    ``/home/user/D/abstract_project/mma/other_repos/MemOS/evaluation/scripts/locomo/prompts.py:1-38``
    (the mem0-style answering prompt that every LoCoMo comparison in that harness uses).
  * ``JUDGE_SYSTEM_PROMPT`` / ``judge_prompt()`` — port of ``locomo_grader``'s ``system_prompt``
    and ``accuracy_prompt`` at
    ``/home/user/D/abstract_project/mma/other_repos/MemOS/evaluation/scripts/locomo/locomo_eval.py:50-81``.
    Note the typographic apostrophes (``’CORRECT’``) in the original — preserved, because
    changing a judge prompt changes the metric.

The official grader parses ``json.loads(content)["label"]`` and calls the answer CORRECT iff the
label case-folds to ``"correct"`` (``locomo_eval.py:82-95``). ``parse_judgement`` reproduces that
exactly, plus ONE documented deviation, argued below.

DEVIATION (recorded, not hidden — CODE-ADOPTION rule 4). The official grader runs
``gpt-4o-mini`` with structured JSON output. This project has no frontier-model key wired into
the evaluation path; the only LLM on the VM is ``qwen2.5:0.5b`` (an OpenAI-compatible endpoint at
127.0.0.1:11435). A 0.5B model is NOT a credible substitute for gpt-4o-mini as a judge, so:

  1. ``parse_judgement`` tolerates a bare ``CORRECT``/``WRONG`` outside JSON, because a small
     model frequently ignores the JSON instruction. This LOOSENS parsing, never grading.
  2. ``judge_control_set`` exists so the judge is VALIDATED BEFORE its numbers are believed: it
     grades gold-answer-vs-itself (must be CORRECT) and gold-vs-a-mismatched-answer (must be
     WRONG). A judge that fails its own control set produces a number this harness REFUSES to
     report as the headline. That is the difference between an unusable judge and an unnoticed
     one.

The retrieval metrics in ``metrics.py`` need no judge at all, which is why they — not the judge —
carry the baseline in a run where no adequate judge model is available.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from mu_eval.token_budget import count_tokens, truncate_to_tokens

__all__ = [
    "ANSWER_PROMPT",
    "COMPACT_JUDGE_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS",
    "ChatPort",
    "ControlSetResult",
    "answer_prompt",
    "compact_judge_prompt",
    "estimated_prompt_tokens",
    "judge_control_set",
    "judge_prompt",
    "parse_judgement",
]

# --- VERBATIM PORT: MemOS evaluation/scripts/locomo/prompts.py:1-38 (ANSWER_PROMPT_MEM0) --------
ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    {context}

    Question: {question}

    Answer:
    """

# --- VERBATIM PORT: MemOS evaluation/scripts/locomo/locomo_eval.py:51-53 ------------------------
JUDGE_SYSTEM_PROMPT = """
        You are an expert grader that determines if answers to questions match a gold standard answer
        """


def answer_prompt(*, context: str, question: str) -> str:
    """``ANSWER_PROMPT_MEM0.format(context=..., question=...)`` — the official call shape."""
    return ANSWER_PROMPT.format(context=context, question=question)


def judge_prompt(*, question: str, gold_answer: str, response: str) -> str:
    """VERBATIM port of ``locomo_grader``'s ``accuracy_prompt`` (locomo_eval.py:55-81)."""
    return f"""
    Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a ’gold’ (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

    Now it’s time for the real question:
    Question: {question}
    Gold answer: {gold_answer}
    Generated answer: {response}

    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

    Just return the label CORRECT or WRONG in a json format with the key as "label".
    """


# --- COMPACT VARIANT: fits Azure Foundry Ministral-3B's 400-token-total ceiling --------------
#
# NOT a verbatim port, and deliberately not a paraphrase of one either — a DIFFERENT prompt,
# argued here, because the verbatim ``JUDGE_SYSTEM_PROMPT``/``judge_prompt()`` above measure at
# ~442 tokens for even a short row (Ministral-8B-Instruct-2410 tokenizer, chat-template applied —
# see ``token_budget.py`` for why that tokenizer stands in for the un-published 3B one), which
# already exceeds the 400-token ceiling before a single completion token is spent: every call
# 429s on size alone, independent of the 1-req/60s quota. CLAUDE.md rule 12 applies to a prompt
# the way it applies to code: when reality (a hard 400-token wall) contradicts what exists (a
# 428-442 token prompt), the prompt changes, recorded here rather than silently drifting.
#
# What was CUT and why it does not weaken the rubric:
#   * the worked example ("Do you remember what I got... / A shell necklace") — the generosity
#     rule it illustrates is stated directly below instead of shown; a 3B instruct model follows
#     an explicit rule at least as well as it infers one from a single example, and the example
#     cost ~45 tokens for zero additional decision content.
#   * the chain-of-thought instruction ("first, a one-sentence explanation, then the verdict") —
#     the whole point of a fixed-budget judge is a capped completion (see
#     ``MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS``); asking for prose before the label spends
#     completion tokens the ceiling does not have, and the harness never reads the explanation.
#   * the JSON-output instruction — ``parse_judgement`` already accepts a bare label (documented
#     deviation above, extended to this variant); JSON syntax has zero decision content and costs
#     both prompt tokens (the instruction) and completion tokens (the wrapper punctuation).
# What was KEPT, unweakened, because these are the actual test:
#   * "same topic/fact counts CORRECT even if longer or reworded" — the rule that makes the judge
#     generous the way the official rubric is generous, not stricter than it.
#   * the time-question carve-out ("same date/period counts CORRECT even if the format differs")
#     — this is the rule the LoCoMo temporal-category questions specifically exercise.
#   * an explicit WRONG condition ("different topic or fact") — the rule the negative controls in
#     ``judge_control_set`` exist to verify the judge actually enforces, not just default-accepts.
COMPACT_JUDGE_SYSTEM_PROMPT = (
    "Grade a GENERATED answer against a GOLD answer for the same QUESTION.\n"
    "CORRECT: same topic/fact as gold, even if worded differently or longer. Time answers: "
    'same date/period counts CORRECT even if the format differs (e.g. "7 May" = "May 7th").\n'
    "WRONG: different topic or fact than gold.\n"
    "Reply with exactly one word, CORRECT or WRONG. Nothing else."
)

# Per-field truncation caps (tokens, Ministral-family tokenizer). Calibrated against this
# project's own LoCoMo corpus (1540 scoreable rows, measured with ``token_budget.count_tokens``):
# question tokens max 33 / p95 21, gold-answer tokens max 70 / p95 18 — so 40 covers the dataset's
# question distribution outright and the gold-answer distribution save one 70-token outlier, which
# a 40-token prefix still carries the leading (topic-bearing) clause of. ``response`` is the one
# field NOT bounded by the dataset (it is whatever the system under test produced, and the answer
# prompt's own "less than 5-6 words" instruction is not enforced), so it gets a taller 80-token
# cap: enough headroom that truncation is the rare path, not the common one.
MINISTRAL_QUESTION_MAX_TOKENS = 40
MINISTRAL_GOLD_ANSWER_MAX_TOKENS = 40
MINISTRAL_RESPONSE_MAX_TOKENS = 80
# 1-3 tokens covers a bare "CORRECT"/"WRONG" (measured: "CORRECT"=3, " WRONG"=2 on the Ministral-
# family tokenizer); a couple of tokens of margin for a leading space or stray punctuation without
# opening the door to the paragraph-length completions this budget cannot afford.
MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS = 6


def estimated_prompt_tokens(system: str, user: str) -> int:
    """Cheap pre-flight estimate (see ``token_budget``) of ``system + user`` token cost, BEFORE
    the provider's own chat template/special tokens. Used only to decide whether to even attempt
    a call; the authoritative count is always the live response's ``usage.prompt_tokens``."""
    return count_tokens(system) + count_tokens(user)


def compact_judge_prompt(*, question: str, gold_answer: str, response: str) -> str:
    """Compact, truncated user turn for :data:`COMPACT_JUDGE_SYSTEM_PROMPT`. Truncation keeps the
    PREFIX of each field (see ``token_budget.truncate_to_tokens``) — the rubric only needs the
    topic/fact, which both gold answers and on-topic generated answers state up front."""
    q = truncate_to_tokens(question, MINISTRAL_QUESTION_MAX_TOKENS)
    g = truncate_to_tokens(gold_answer, MINISTRAL_GOLD_ANSWER_MAX_TOKENS)
    r = truncate_to_tokens(response, MINISTRAL_RESPONSE_MAX_TOKENS)
    return f"Question: {q}\nGold: {g}\nGenerated: {r}"


_BARE_LABEL = re.compile(r"\b(CORRECT|WRONG)\b", re.IGNORECASE)


def parse_judgement(content: str) -> bool | None:
    """Official parse (``json.loads(...)["label"] .strip().lower() == "correct"``), plus the ONE
    documented deviation: a bare ``CORRECT``/``WRONG`` outside JSON is accepted.

    Returns ``None`` when NO label can be read at all — an unparseable judgement is reported as
    unparseable, never silently graded WRONG (that would make a broken judge look like a broken
    memory system, which is exactly the confusion this whole harness exists to end).
    """
    text = content.strip()
    try:
        payload: Any = json.loads(text)
        if isinstance(payload, dict) and "label" in payload:
            return str(payload["label"]).strip().lower() == "correct"
    except (json.JSONDecodeError, TypeError):
        pass
    matches = _BARE_LABEL.findall(text)
    if not matches:
        return None
    # "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation
    # script" — the official prompt's own words. If the model did both anyway, the judgement is
    # unusable; say so rather than picking one.
    labels = {m.upper() for m in matches}
    if labels == {"CORRECT", "WRONG"}:
        return None
    return labels == {"CORRECT"}


class ChatPort(Protocol):
    """Narrow chat seam so the judge can be pointed at any OpenAI-compatible endpoint."""

    async def complete(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str: ...


class ControlSetResult(BaseModel):
    """Did the judge pass its own sanity gate? See the module docstring, deviation (2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    positives: int
    positives_correct: int
    negatives: int
    negatives_wrong: int
    unparseable: int

    @property
    def usable(self) -> bool:
        """A judge is usable only if it agrees with itself on gold answers AND rejects mismatched
        ones. The bar is deliberately generous (>=0.9 both ways) — a judge that cannot clear a
        control set this easy cannot grade a real system's answers."""
        if self.positives == 0 or self.negatives == 0:
            return False
        return (
            self.positives_correct / self.positives >= 0.9
            and self.negatives_wrong / self.negatives >= 0.9
        )


async def judge_control_set(
    chat: ChatPort,
    *,
    questions: Sequence[str],
    gold_answers: Sequence[str],
    system_prompt: str = JUDGE_SYSTEM_PROMPT,
    build_prompt: Callable[..., str] = judge_prompt,
    max_tokens: int | None = None,
) -> ControlSetResult:
    """Grade the judge before believing it.

    POSITIVE controls: the gold answer is handed back as the generated answer — the official
    rubric ("as long as it touches on the same topic as the gold answer") makes CORRECT the only
    defensible label. NEGATIVE controls: each question is paired with a DIFFERENT row's gold
    answer (rotate by one), which the rubric must call WRONG.

    ``system_prompt``/``build_prompt``/``max_tokens`` default to the verbatim MemOS-ported prompt
    (unbounded completion) — the original qwen2.5:0.5b lineage. Pass
    ``system_prompt=COMPACT_JUDGE_SYSTEM_PROMPT``, ``build_prompt=compact_judge_prompt``,
    ``max_tokens=MINISTRAL_JUDGE_MAX_COMPLETION_TOKENS`` for the token-budget-fit Ministral-3B
    variant (``build_prompt`` must accept ``question``/``gold_answer``/``response`` keywords,
    matching :func:`judge_prompt`'s and :func:`compact_judge_prompt`'s shared shape).
    """
    if len(questions) != len(gold_answers):
        raise ValueError("questions and gold_answers must be the same length")
    n = len(questions)
    if n < 2:
        raise ValueError("need at least 2 rows to build a negative control")

    positives_correct = 0
    negatives_wrong = 0
    unparseable = 0

    for i in range(n):
        verdict = parse_judgement(
            await chat.complete(
                system=system_prompt,
                user=build_prompt(
                    question=questions[i], gold_answer=gold_answers[i], response=gold_answers[i]
                ),
                max_tokens=max_tokens,
            )
        )
        if verdict is None:
            unparseable += 1
        elif verdict:
            positives_correct += 1

    for i in range(n):
        mismatched = gold_answers[(i + 1) % n]
        verdict = parse_judgement(
            await chat.complete(
                system=system_prompt,
                user=build_prompt(
                    question=questions[i], gold_answer=gold_answers[i], response=mismatched
                ),
                max_tokens=max_tokens,
            )
        )
        if verdict is None:
            unparseable += 1
        elif not verdict:
            negatives_wrong += 1

    return ControlSetResult(
        positives=n,
        positives_correct=positives_correct,
        negatives=n,
        negatives_wrong=negatives_wrong,
        unparseable=unparseable,
    )
