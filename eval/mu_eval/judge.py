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
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

__all__ = [
    "ANSWER_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "ChatPort",
    "ControlSetResult",
    "answer_prompt",
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

    async def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str: ...


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
) -> ControlSetResult:
    """Grade the judge before believing it.

    POSITIVE controls: the gold answer is handed back as the generated answer — the official
    rubric ("as long as it touches on the same topic as the gold answer") makes CORRECT the only
    defensible label. NEGATIVE controls: each question is paired with a DIFFERENT row's gold
    answer (rotate by one), which the rubric must call WRONG.
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
                system=JUDGE_SYSTEM_PROMPT,
                user=judge_prompt(
                    question=questions[i], gold_answer=gold_answers[i], response=gold_answers[i]
                ),
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
                system=JUDGE_SYSTEM_PROMPT,
                user=judge_prompt(
                    question=questions[i], gold_answer=gold_answers[i], response=mismatched
                ),
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
