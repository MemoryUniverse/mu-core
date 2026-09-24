"""The answer-quality run: real retrieval -> real answer generation -> the OFFICIAL LoCoMo judge.

Neither ``runner.py`` (ranking metrics only — did the gold TURN appear in the ranked list) nor
``judge.py``'s ``judge_control_set`` (validates the judge on synthetic gold-vs-gold /
gold-vs-mismatched pairs, never a real generated answer) assembles the thing every comparable
system (mem0, MemOS, Zep) reports as its headline: retrieve real context through the SHIPPED
recall path, generate an answer with the OFFICIAL mem0-ported ``ANSWER_PROMPT``
(``judge.py:ANSWER_PROMPT``/``answer_prompt``), and grade that generated answer against the
dataset's own gold answer with the OFFICIAL LoCoMo judge rubric (``judge.py``). This module is
that assembly — the one number this project has never been able to report because no judge had
cleared its own control set until D4 closed (``STATE-AND-DEFECTS-0829.md`` D4).

Reuses, deliberately, rather than re-implementing:
  * ``corpus.ingest_conversation`` / ``corpus.local_memory_for`` — the SAME real-store ingest and
    teardown ``run_baseline`` uses (CLAUDE.md rule 14: own every partition this creates).
  * ``runner._await_index`` — the same bounded poll that waits for the vector tier to actually be
    queryable before the first question fires (Qdrant upserts are async).
  * ``judge.ANSWER_PROMPT``/``answer_prompt`` and ``judge.COMPACT_JUDGE_SYSTEM_PROMPT``/
    ``compact_judge_prompt``/``parse_judgement`` — the verbatim-ported official prompts, unchanged.
  * ``openai_chat.OpenAICompatChat`` — the same hand-rolled client the judge-control runs used,
    with the same gpt-5-specific fixes (``completion_tokens_field``, ``omit_temperature``).

CONTEXT FORMAT re-attaches the LoCoMo session date mem0's ``f"{created_at}: {memory}"`` carries
natively. MU's ``RecallItemView`` (``mu_contracts.contracts.recall``) has no per-item timestamp on
the surface a caller reads, so the date cannot come from the recalled item itself — it is rejoined
by the HARNESS, from its own corpus, the same way ``runner.py`` joins a recalled body back to its
gold turn id (normalized-body lookup, ``TurnIndex.resolve``). Measured why this matters, not
assumed: a first pilot run with no date attached produced answers like ``"Yesterday."``/``"Last
year"`` for temporal questions — the ANSWER_PROMPT's own instruction 6 ("convert relative time
references to specific dates ... based on the memory timestamp") is unanswerable with no timestamp
in the context, which would have depressed the temporal-reasoning category for a reason that is a
harness gap, not a memory-quality one. Each context line is therefore
``f"{session_date}: {speaker}: {text}"`` — the speaker prefix from ingest (``locomo.Turn.
ingest_text``) plus the harness-rejoined date. An item whose body was not written by this harness
(no normalized-body match — an LTM-distilled paraphrase, chiefly) is left undated rather than
guessed.

CONCURRENCY. gpt-5 on the ``libo-ai`` resource allows 500 requests / 50,000 tokens per 60s
(``STATE-AND-DEFECTS-0829.md`` D4 update) — a different order of magnitude from Ministral-3B's
1 request/60s, so the strict single-flight ``PacedOpenAICompatChat`` built for that deployment
would leave nearly all of this quota idle (each call has ~1-3s of model latency; sequential
dispatch could not reach even the *token* ceiling, let alone the request one). A bounded
``asyncio.Semaphore`` lets enough calls run concurrently to approach the real ceiling, with a
single reactive retry on 429 (same backoff shape as ``PacedOpenAICompatChat``, just without the
proactive inter-request floor that deployment does not need).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mu_eval.corpus import local_memory_for
from mu_eval.judge import (
    ANSWER_PROMPT,  # noqa: F401  (re-exported for callers that want the exact string)
    COMPACT_JUDGE_SYSTEM_PROMPT,
    answer_prompt,
    compact_judge_prompt,
    parse_judgement,
)
from mu_eval.locomo import CATEGORY_NAMES, Conversation, LabelledQuery
from mu_eval.openai_chat import CompletionResult, OpenAICompatChat, RateLimitError
from mu_eval.repeats import RepeatSummary, summarize_repeats
from mu_eval.runner import gold_ids_present
from mu_eval.usage import CallUsage

__all__ = [
    "ANSWER_SYSTEM_PROMPT",
    "AnswerQualityReport",
    "CategoryStats",
    "QueryResult",
    "RepeatedAnswerQualityReport",
    "compute_pairwise_noise",
    "eligible_query_count",
    "run_answer_quality",
    "run_answer_quality_repeated",
]

# mem0's own reference harness (other_repos/mem0/evaluation/src/mem0/search.py — not ported
# verbatim like ANSWER_PROMPT because it is a one-line role description, not a scored rubric)
# pairs ANSWER_PROMPT with a plain instruction-following system turn. Kept short and generic on
# purpose: the grading rubric is what decides correctness, not the system prompt's wording.
ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided memories."
)


class QueryResult(BaseModel):
    """One graded row — kept per-row so a misgrade can be read directly (D4's own diagnostic
    practice: 'read the actual negatives it gets wrong', not just the aggregate)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    category: int
    question: str
    gold_answer: str
    generated_answer: str
    context_items: int
    verdict: bool | None  # True=CORRECT, False=WRONG, None=unparseable judge output
    # PER-ROW ATTRIBUTION (HOW-THEY-MEASURE-0901.md F1 / item 1): did ANY gold evidence turn
    # actually appear in the context handed to the answering model, per `runner.gold_ids_present`
    # (the SAME join `runner.py` uses to score ranking, so this and the ranking metrics can never
    # quietly disagree about what counts as "retrieved"). This is what turns "we are retrieval-
    # capped" from an inference drawn by dividing two aggregates into a per-row measurement: a
    # WRONG verdict with `gold_in_context=True` is a GENERATION failure (the model had the answer
    # and got it wrong anyway); `gold_in_context=False` is a RETRIEVAL failure (the model never
    # had a chance). See `CategoryStats.wrong_retrieved`/`wrong_not_retrieved` for the aggregate.
    gold_in_context: bool

    # PER-ROW COST (CLAUDE.md eval lane: "per-row usage too, not only totals, so an expensive
    # category can be found"). One `CallUsage` per LLM call this row made — `None` only when the
    # row predates this fix (an older artifact) or the call itself never returned a usage block
    # (`usage.parse_call_usage` docstring). Two separate fields, not one summed total, because the
    # answer and judge calls are usually different models at different prices.
    answer_usage: CallUsage | None = None
    judge_usage: CallUsage | None = None

    # NOISE-FLOOR / CHARACTERISATION FIELDS (TRUSTWORTHY-MEASUREMENT-0924). Defaults keep an
    # older artifact loading unchanged (same discipline `gold_in_context` and `*_usage` already
    # established for this model).
    #
    # The RAW judge completion, kept regardless of whether it parsed — this is what let this pass
    # characterise "unparseable" from actual outputs instead of the bare counter: before this
    # field existed, an unparseable row's own artifact carried no way to tell "empty because the
    # budget ran out", "a refusal", or "both CORRECT and WRONG present" apart after the fact
    # without re-running the whole row. `None` only for a row written before this fix.
    judge_raw: str | None = None
    # Did `_complete_with_retry`'s budget-exhaustion policy have to intervene on this row's answer
    # / judge call? Recorded even when the row ended up correctly parsed anyway (the retry
    # SUCCEEDED) — so a report can show how often the safety net fired, not only its effect on the
    # unparseable count. See `_complete_with_retry`/`_is_budget_exhausted` for the exact policy.
    answer_budget_retried: bool = False
    judge_budget_retried: bool = False


class CategoryStats(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    correct: int
    wrong: int
    unparseable: int
    # Attribution breakdown (item 1) — all three derived from `QueryResult.gold_in_context`,
    # always populated (never `None`/omitted) so a reader is never left to assume a report
    # predates this fix.
    gold_retrieved: int = 0  # rows (any verdict) whose gold evidence was in the context
    wrong_retrieved: int = 0  # verdict=WRONG AND gold was retrieved -> a GENERATION failure
    wrong_not_retrieved: int = 0  # verdict=WRONG AND gold was NOT retrieved -> a RETRIEVAL failure

    # How often the budget-exhaustion retry (`_complete_with_retry`) fired on this bucket's rows,
    # and how many of the STILL-unparseable rows show the exact budget-exhaustion wire signature
    # even after that retry (`finish_reason == "length"` and blank content on the LAST attempt) —
    # i.e. genuinely exhausted the RETRIED budget too, as opposed to unparseable for some other
    # reason (a non-empty judge output that names neither/both labels). Both default 0 so an
    # artifact from before this fix still loads.
    answer_budget_retried: int = 0
    judge_budget_retried: int = 0
    unparseable_budget_exhausted: int = 0  # subset of `unparseable`, see docstring above
    unparseable_other: int = 0  # `unparseable` minus `unparseable_budget_exhausted`

    @property
    def scoreable(self) -> int:
        """``correct + wrong + unparseable`` — every row that reached the judge and got SOME
        verdict back, parseable or not. The honest denominator for `accuracy_all_scoreable`."""
        return self.correct + self.wrong + self.unparseable

    @property
    def accuracy(self) -> float:
        """PARSEABLE-SUBSET accuracy: correct / (correct + wrong) — unparseable EXCLUDED from the
        denominator, same discipline as ``ControlSetResult``: an unparseable judgement is a
        judge-infra failure, never silently counted as a wrong answer (that would blame the
        memory system for the judge's parsing miss). Report this ALONGSIDE
        `accuracy_all_scoreable`, always — never let a reader see only one (item 3 / F3: two arms
        of the same reported delta were once silently computed over different row sets, 1462 vs
        ~1520, because only this narrower denominator was ever printed)."""
        scored = self.correct + self.wrong
        return self.correct / scored if scored else 0.0

    @property
    def accuracy_all_scoreable(self) -> float:
        """ALL-SCOREABLE accuracy: correct / (correct + wrong + unparseable) — the OPPOSITE
        editorial choice from `accuracy`: an unparseable judgement counts AGAINST the system here.
        Neither denominator is "the" right one; the fix (item 3) is reporting both, always, with
        the skip counts that produced the gap (`unparseable` here; `AnswerQualityReport.
        skipped_adversarial`/`skipped_no_gold_in_corpus` upstream of it) broken out by reason
        rather than folded silently into one number."""
        scored = self.scoreable
        return self.correct / scored if scored else 0.0


class AnswerQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset: str
    samples: int
    answer_model: str
    judge_model: str
    recall_limit: int
    importance: float
    queries_scored: int
    skipped_adversarial: int
    # T4 (`TRACE-0923.md` §7): see the identical field on `runner.ArmReport` — a
    # non-adversarial row with empty `evidence` is a distinct reason from "adversarial", counted
    # separately rather than folded into `skipped_adversarial`. Default 0 so an artifact written
    # before this fix still loads through `AnswerQualityReport.model_validate_json`.
    skipped_no_evidence: int = 0
    skipped_no_gold_in_corpus: int
    overall: CategoryStats
    by_category: dict[str, CategoryStats]
    rows: list[QueryResult] = Field(default_factory=list)
    # The item-2 provenance block (`provenance.build_provenance`), which
    # `__main__._cmd_answer_quality` merges into `--out` as
    # `report.model_dump(...) | {"provenance": ...}`.
    #
    # DECLARED HERE, not left as a bare merged key, because this model is ALSO the reader:
    # `__main__._cmd_compare` re-opens a written artifact with
    # `AnswerQualityReport.model_validate_json`, and `extra="forbid"` above means an undeclared
    # key is a hard ValidationError. Measured, not theorised: with `provenance` merged but
    # undeclared, `mu_eval compare` failed on EVERY artifact the harness writes — item 4's whole
    # command was unusable against item 2's own output. The unit tests missed it because they
    # construct `AnswerQualityReport` in memory and never round-trip through the JSON file, so
    # the two halves were never exercised together (see `test_compare_roundtrip_unit.py`, which
    # now closes exactly that gap).
    #
    # `dict[str, Any]` rather than a pydantic model on purpose: `build_provenance` returns a
    # plain dict whose shape deliberately grows (a new store knob appears in
    # `recall_settings` the moment the engine gains one), and a strict nested model here would
    # turn "the engine gained a setting" into "every older artifact fails to load".
    provenance: dict[str, Any] | None = None

    # Same merge-at-write-time pattern as `provenance` (`__main__._cmd_answer_quality`: `report.
    # model_dump(...) | {"provenance": ..., "usage": ...}`), and declared here for the identical
    # reason `provenance` is: `extra="forbid"` above means an undeclared merged key is a hard
    # ValidationError the moment `mu_eval compare` re-opens the artifact
    # (`usage.RunUsage.model_dump(mode="json")` — see `usage.build_run_usage`).
    usage: dict[str, Any] | None = None


# BUDGET-EXHAUSTION RETRY POLICY (TRUSTWORTHY-MEASUREMENT-0924: "characterise it, then handle it
# deterministically"). Read the real failing rows first (`STATE-AND-DEFECTS-0829.md` D4 update,
# item 3; `ARCHITECTURE-DELTAS.md` AD-232's paired 150-row run): every unparseable row inspected
# so far shares ONE exact wire signature — `finish_reason: "length"` with `content: ""` and
# `reasoning_tokens` at or near the requested cap (e.g. `reasoning_tokens: 300` on a 300-token
# budget). gpt-5 spent the ENTIRE completion budget on hidden reasoning tokens before emitting a
# single visible token. That is not a refusal (`finish_reason` would be `"content_filter"`), not a
# parse that is too strict (there is no content to parse — `parse_judgement`'s bare-label regex
# and JSON path both correctly return `None` on an empty string), and not "something else" — it is
# budget exhaustion, characterised from the actual wire response, not inferred from the unparseable
# counter. A genuinely malformed-but-NON-EMPTY completion (a real format miss, or the "both CORRECT
# and WRONG present" case `parse_judgement` already refuses) is a DIFFERENT failure that a bigger
# budget cannot fix, so it is deliberately NOT retried here — retrying on content alone would
# silently paper over an actual parser gap instead of surfacing it.
#
# The fix is ONE deterministic, bounded retry, doubling the completion budget, only on that exact
# signature. `BUDGET_RETRY_MULTIPLIER=2` and `BUDGET_RETRY_MAX_TOKENS` are stated here, not
# invented at the call site, so the policy is the same for every caller of `_complete_with_retry`
# (answer and judge calls alike) and is auditable in one place.
BUDGET_RETRY_MULTIPLIER = 2
# Hard ceiling on the RETRIED budget (never on the caller's original ask) — bounds the worst-case
# extra cost of the safety net to at most this many completion tokens on any one call, instead of
# an unbounded doubling if a caller ever passes a very large `max_tokens`.
BUDGET_RETRY_MAX_TOKENS = 3000
# MEASURED, not guessed (TRUSTWORTHY-MEASUREMENT-0924's own real run, gpt-5, conv-26, 300 judge
# rows): a SINGLE doubling (300 -> 600) was NOT always enough. Every one of the 11 unparseable
# rows across both repeats showed the identical wire signature (`content=""`,
# `finish_reason="length"`, `reasoning_tokens=600`) at the FIRST retry's 600-token budget too —
# i.e. this project's own real evidence, not a hypothetical, shows one retry level is
# insufficient for a real (if small, ~2-5%) share of rows. `BUDGET_RETRY_MAX_ATTEMPTS` bounds how
# many ADDITIONAL doublings `_complete_with_retry` will make beyond the first attempt (2 => at
# most 300 -> 600 -> 1200 for the judge's default budget) before accepting an exhausted result and
# reporting it honestly rather than looping further. Kept well under `BUDGET_RETRY_MAX_TOKENS` so
# the ceiling, not this count, is normally what stops the loop for a caller with a larger
# `max_tokens`.
BUDGET_RETRY_MAX_ATTEMPTS = 2


def _is_budget_exhausted(result: CompletionResult) -> bool:
    """The exact, wire-verified signature of hidden-reasoning budget exhaustion — see the policy
    note above. Both conditions are required: `finish_reason` alone is not enough (a model can
    legitimately finish at the cap with SOME visible content already written), and blank content
    alone is not enough (a `"stop"`/`"content_filter"` finish with empty content is a different,
    NOT budget-shaped, failure that this policy must not paper over)."""
    return result.finish_reason == "length" and result.content.strip() == ""


async def _complete_with_retry(
    chat: OpenAICompatChat, *, system: str, user: str, max_tokens: int
) -> CompletionResult:
    """ONE reactive retry on 429 (same backoff contract as ``PacedOpenAICompatChat``: sleep the
    server's own ``Retry-After``/reset header, never retry instantly, minus the proactive
    per-request floor gpt-5's quota does not need), PLUS a separate, deterministic, BOUNDED-LOOP
    retry on hidden-reasoning budget exhaustion: up to ``BUDGET_RETRY_MAX_ATTEMPTS`` further
    attempts, each doubling the previous attempt's budget (``BUDGET_RETRY_MULTIPLIER``), capped at
    ``BUDGET_RETRY_MAX_TOKENS`` — see the policy note above those constants, including the REAL
    measured evidence for why a single retry level was not enough to make this loop, not a single
    `if`. Concurrency is gated by the CALLER (the whole per-query pipeline, recall included — see
    ``run_answer_quality``'s ``sem``), not here.

    Returns ``CompletionResult`` (content + usage IN-BAND), not a bare string: this coroutine runs
    concurrently, up to ``concurrency`` at a time, against ONE shared chat client per role — see
    ``CompletionResult``'s own docstring for why reading ``chat.last_usage`` after the fact would
    be a race under that concurrency, and why returning it bundled with the content is not.

    The RETURNED ``CompletionResult`` always reflects the LAST attempt made (the final retried one,
    when any budget retry fired) — never a silently discarded earlier attempt — and
    ``result.budget_retried`` records whether AT LEAST one retry happened, so a caller can count
    how often the safety net was needed rather than only seeing its effect on the parse. A row that
    is STILL exhausted after every attempt this policy allows is returned as-is (still exhausted,
    `budget_retried=True`) — reported honestly as budget-shaped rather than looped forever.
    """

    async def _attempt(tokens: int) -> CompletionResult:
        try:
            return await chat.complete_with_usage(system=system, user=user, max_tokens=tokens)
        except RateLimitError as exc:
            await asyncio.sleep(exc.retry_after_s if exc.retry_after_s is not None else 5.0)
            return await chat.complete_with_usage(system=system, user=user, max_tokens=tokens)

    result = await _attempt(max_tokens)
    tokens = max_tokens
    retried = False
    for _ in range(BUDGET_RETRY_MAX_ATTEMPTS):
        if not _is_budget_exhausted(result):
            break
        next_tokens = min(tokens * BUDGET_RETRY_MULTIPLIER, BUDGET_RETRY_MAX_TOKENS)
        if next_tokens <= tokens:
            # Already at (or the multiplier rounds to) the ceiling — a further identical-or-
            # smaller call would not be a genuine retry, so stop rather than spend another call
            # that cannot possibly change the outcome.
            break
        result = await _attempt(next_tokens)
        tokens = next_tokens
        retried = True
    return result.model_copy(update={"budget_retried": True}) if retried else result


def _eligible_queries(
    conversation: Conversation, max_queries: int | None
) -> tuple[list[LabelledQuery], int, int, int]:
    """Same admission rule as ``runner.run_baseline``: drop adversarial/no-answer rows and rows
    whose gold evidence names no turn this harness actually ingested. Counted, never silently
    dropped."""
    known = {t.dia_id for t in conversation.turns}
    queries = list(conversation.queries)
    if max_queries is not None:
        queries = queries[:max_queries]
    eligible: list[LabelledQuery] = []
    skipped_adversarial = 0
    skipped_no_evidence = 0
    skipped_no_gold = 0
    for query in queries:
        if query.is_adversarial:
            skipped_adversarial += 1
            continue
        # T4: distinct from "adversarial" — a non-adversarial row with no evidence, or no gold
        # answer text, cannot be scored but was never a category-5 row (TRACE-0923.md §7).
        if not query.evidence or not query.answer:
            skipped_no_evidence += 1
            continue
        if not any(e in known for e in query.evidence):
            skipped_no_gold += 1
            continue
        eligible.append(query)
    return eligible, skipped_adversarial, skipped_no_evidence, skipped_no_gold


def eligible_query_count(
    conversations: Sequence[Conversation], max_queries_per_sample: int | None = None
) -> int:
    """Total rows a ``run_answer_quality`` call over ``conversations`` will actually score — the
    SAME admission rule ``_eligible_queries`` applies per-conversation, summed. Public (not
    ``_``-prefixed like its per-conversation sibling) because the CLI needs it BEFORE a run starts,
    to print a projected cost (``usage.project_cost``) from the real row count rather than a guess
    — the whole point being that a run can be abandoned before it is paid for, not after."""
    return sum(len(_eligible_queries(c, max_queries_per_sample)[0]) for c in conversations)


def _is_budget_exhausted_row(row: QueryResult) -> bool:
    """Same wire signature `_is_budget_exhausted` checks live (blank content), applied
    RETROSPECTIVELY to a row's stored `judge_raw` — the honest split between "still unparseable
    because the (possibly retried) budget was exhausted" and "unparseable for some other reason"
    (non-empty content that names neither/both labels). `judge_raw is None` (an artifact written
    before this fix carries no raw text to check) is conservatively NOT counted as budget-shaped —
    an older artifact's unparseable rows fall entirely into `unparseable_other` rather than a
    guess."""
    return row.verdict is None and row.judge_raw is not None and row.judge_raw.strip() == ""


def _stats(rows: Sequence[QueryResult]) -> CategoryStats:
    correct = sum(1 for r in rows if r.verdict is True)
    wrong = sum(1 for r in rows if r.verdict is False)
    unparseable = sum(1 for r in rows if r.verdict is None)
    gold_retrieved = sum(1 for r in rows if r.gold_in_context)
    wrong_retrieved = sum(1 for r in rows if r.verdict is False and r.gold_in_context)
    wrong_not_retrieved = sum(1 for r in rows if r.verdict is False and not r.gold_in_context)
    answer_budget_retried = sum(1 for r in rows if r.answer_budget_retried)
    judge_budget_retried = sum(1 for r in rows if r.judge_budget_retried)
    unparseable_budget_exhausted = sum(1 for r in rows if _is_budget_exhausted_row(r))
    return CategoryStats(
        n=len(rows),
        correct=correct,
        wrong=wrong,
        unparseable=unparseable,
        gold_retrieved=gold_retrieved,
        wrong_retrieved=wrong_retrieved,
        wrong_not_retrieved=wrong_not_retrieved,
        answer_budget_retried=answer_budget_retried,
        judge_budget_retried=judge_budget_retried,
        unparseable_budget_exhausted=unparseable_budget_exhausted,
        unparseable_other=unparseable - unparseable_budget_exhausted,
    )


async def run_answer_quality(
    *,
    conversations: Sequence[Conversation],
    run_id: str,
    answer_chat: OpenAICompatChat,
    judge_chat: OpenAICompatChat,
    answer_model: str,
    judge_model: str,
    recall_limit: int = 10,
    importance: float = 0.9,
    max_queries_per_sample: int | None = None,
    answer_max_tokens: int = 400,
    judge_max_tokens: int = 300,
    concurrency: int = 12,
    keep_rows: bool = True,
    progress: Any = None,
    consolidate: bool = False,
    settings: object | None = None,
) -> AnswerQualityReport:
    """Ingest every conversation into a real, isolated partition (same discipline as
    ``run_baseline``), then for each eligible query: recall real context through the shipped
    3-channel fuse, generate an answer with the official prompt, grade it with the official
    judge. ``progress``, if given, is called as ``progress(done, total)`` after each graded row —
    used to print a heartbeat on a run long enough that silence would look like a hang.

    ``settings`` (NEW — HOW-THEY-MEASURE-0901.md F2): threaded straight through to
    ``local_memory_for``, matching ``run_baseline``'s own signature (which already took it). Prior
    to this fix, this function ALWAYS called ``local_memory_for(conversation, run_id=run_id)``
    with no ``settings=`` at all, so the storage ``Settings`` behind every answer-quality run was
    whatever ``mu_local.composition.get_settings()`` resolved from ambient environment on the
    machine that happened to invoke it — never recorded, never overridable from this call. Passing
    it through does not, by itself, close the SEPARATE finding that the ranker's own
    ``RecallSettings`` has no override seam on ``LocalMemory`` at all (``provenance.
    recall_settings_snapshot``'s own docstring) — that is a ``mu-local`` composition-root change,
    outside this file's ownership — but it does close the part that was fixable here: the storage
    config this harness's OWN corpus/teardown code depends on (``corpus.local_memory_for``,
    ``corpus._teardown``) is no longer silently ambient.
    """
    from mu_eval.runner import _await_index  # lazy: keeps this module importable store-free

    user = "evaluser"
    session = "evalsession"
    sem = asyncio.Semaphore(concurrency)
    skipped_adversarial = 0
    skipped_no_evidence = 0
    skipped_no_gold = 0
    rows: list[QueryResult] = []
    total_eligible = eligible_query_count(conversations, max_queries_per_sample)
    done = 0

    for conversation in conversations:
        async with local_memory_for(conversation, run_id=run_id, settings=settings) as opaque:
            memory: Any = opaque
            from mu_eval.corpus import ingest_conversation

            index, _ingest_report = await ingest_conversation(
                memory,
                conversation,
                user=user,
                session=session,
                importance=importance,
                consolidate=consolidate,
            )
            if conversation.turns:
                await _await_index(
                    memory, conversation.turns[0].text[:120], user=user, session=session
                )

            eligible, sk_adv, sk_no_ev, sk_gold = _eligible_queries(
                conversation, max_queries_per_sample
            )
            skipped_adversarial += sk_adv
            skipped_no_evidence += sk_no_ev
            skipped_no_gold += sk_gold
            turn_by_dia_id = conversation.turn_by_dia_id()
            known_dia_ids = set(turn_by_dia_id)

            def _context_line(
                content: str, *, index: Any = index, turn_by_dia_id: Any = turn_by_dia_id
            ) -> str:
                """Re-attach the LoCoMo session date the harness's own corpus knows but the
                recalled item does not carry (module docstring: ``RecallItemView`` has no
                per-item timestamp). Joins by normalized body — the same join ``runner.py``
                uses to score ranking — so the ANSWER_PROMPT's own instruction to convert
                relative time references ("last year") into concrete ones is actually
                answerable, rather than silently unanswerable for every temporal-reasoning row."""
                dia_ids = index.resolve(content)
                if dia_ids and dia_ids[0] in turn_by_dia_id:
                    return f"{turn_by_dia_id[dia_ids[0]].session_date}: {content}"
                return content

            async def _one(
                query: LabelledQuery,
                *,
                memory: Any = memory,
                index: Any = index,
                known_dia_ids: Any = known_dia_ids,
            ) -> QueryResult:
                nonlocal done
                # Same admission rule `runner.run_baseline` uses for its own gold set: evidence
                # ids that name a turn this harness never ingested (image-only turns carry no
                # body) cannot be "retrieved" by definition, so they are excluded here too —
                # otherwise a category-5-shaped labelling gap would show up as a manufactured
                # retrieval failure instead of the labelling gap it actually is.
                gold = {e for e in query.evidence if e in known_dia_ids}
                # The WHOLE pipeline is gated by `sem`, not just the two LLM calls: `recall()` is
                # a real multi-store round trip (Valkey + Qdrant + FalkorDB), and firing all of a
                # conversation's ~150-200 queries at it via a bare `asyncio.gather` (no bound)
                # MEASURED a `TimeoutError` inside the recall verb's own per-call timeout under
                # concurrency=20 with recall left unbounded — a self-inflicted load spike on the
                # shared VM stores, not a system defect. Bounding recall by the same semaphore
                # that paces the LLM calls keeps concurrency at a level the stores demonstrated
                # they handle (the pilot runs above, concurrency<=5, never hit this).
                async with sem:
                    result = await memory.recall(
                        query.question, user=user, session=session, limit=recall_limit
                    )
                    context = (
                        "\n".join(f"- {_context_line(item.content)}" for item in result.items)
                        or "(no memories retrieved)"
                    )
                    answer_result = await _complete_with_retry(
                        answer_chat,
                        system=ANSWER_SYSTEM_PROMPT,
                        user=answer_prompt(context=context, question=query.question),
                        max_tokens=answer_max_tokens,
                    )
                    generated = answer_result.content
                    judge_result = await _complete_with_retry(
                        judge_chat,
                        system=COMPACT_JUDGE_SYSTEM_PROMPT,
                        user=compact_judge_prompt(
                            question=query.question, gold_answer=query.answer, response=generated
                        ),
                        max_tokens=judge_max_tokens,
                    )
                    verdict_raw = judge_result.content
                row = QueryResult(
                    query_id=query.query_id,
                    category=query.category,
                    question=query.question,
                    gold_answer=query.answer,
                    generated_answer=generated,
                    context_items=len(result.items),
                    verdict=parse_judgement(verdict_raw),
                    gold_in_context=gold_ids_present(result.items, index, gold),
                    answer_usage=answer_result.usage,
                    judge_usage=judge_result.usage,
                    judge_raw=verdict_raw,
                    answer_budget_retried=answer_result.budget_retried,
                    judge_budget_retried=judge_result.budget_retried,
                )
                done += 1
                if progress is not None:
                    progress(done, total_eligible)
                return row

            rows.extend(await asyncio.gather(*[_one(q) for q in eligible]))

    overall = _stats(rows)
    by_category: dict[str, CategoryStats] = {}
    for cat, name in CATEGORY_NAMES.items():
        subset = [r for r in rows if r.category == cat]
        if subset:
            by_category[f"{cat}:{name}"] = _stats(subset)

    return AnswerQualityReport(
        run_id=run_id,
        dataset="locomo10",
        samples=len(conversations),
        answer_model=answer_model,
        judge_model=judge_model,
        recall_limit=recall_limit,
        importance=importance,
        queries_scored=len(rows),
        skipped_adversarial=skipped_adversarial,
        skipped_no_evidence=skipped_no_evidence,
        skipped_no_gold_in_corpus=skipped_no_gold,
        overall=overall,
        by_category=by_category,
        rows=rows if keep_rows else [],
    )


class RepeatedAnswerQualityReport(BaseModel):
    """N independent full runs of ``run_answer_quality`` + the mean/spread over each headline
    number (item 5). Each run is a genuinely separate ingest (and, with ``consolidate=True``, a
    genuinely separate LLM-driven LTM derivation) under its own run id — see
    ``repeats.run_n_times``'s own docstring for why a shallower repeat would not have caught the
    defect this exists for."""

    model_config = ConfigDict(extra="forbid")

    runs: list[AnswerQualityReport]
    overall_accuracy: RepeatSummary  # parseable-subset accuracy, one value per run
    overall_accuracy_all_scoreable: RepeatSummary
    by_category_accuracy: dict[str, RepeatSummary]
    # THE NOISE FLOOR (TRUSTWORTHY-MEASUREMENT-0924), measured directly rather than assumed: a
    # paired `compare.compare_runs` between each pair of CONSECUTIVE repeats of this SAME,
    # unchanged config. Unlike `overall_accuracy.spread` (which only says how far the two AGGREGATE
    # numbers moved), this names WHICH rows moved and in which direction — including the
    # `became_unparseable`/`recovered_from_unparseable` churn that a bare accuracy spread cannot
    # see at all (two runs can tie on accuracy while churning a dozen rows' parseability in
    # opposite directions). Empty when `num_runs < 2` or any run was written with `keep_rows=False`
    # (a comparison needs per-row verdicts — see `compare.compare_runs`'s own guard).
    #
    # `dict[str, Any]` rather than `compare.CompareResult` directly, same reason and same
    # precedent as `AnswerQualityReport.provenance`/`.usage` above (`compare.py` imports FROM this
    # module, so importing its model back here would be circular) — each entry is
    # `CompareResult.model_dump(mode="json")`, so a caller who wants the typed object back can
    # `CompareResult.model_validate(entry)`.
    pairwise_noise: list[dict[str, Any]] = Field(default_factory=list)


async def run_answer_quality_repeated(
    *,
    conversations: Sequence[Conversation],
    run_id_prefix: str,
    num_runs: int,
    **kwargs: Any,
) -> RepeatedAnswerQualityReport:
    """``run_answer_quality`` repeated ``num_runs`` times (``repeats.run_n_times``), summarized.

    ``num_runs=1`` (the default everywhere this is wired from the CLI) is intentionally NOT
    special-cased into a bare ``run_answer_quality`` call — it goes through the exact same
    ``run_n_times``/``summarize_repeats`` path as any other N, so a 1-run "spread" of 0.0 is
    computed the same way a real spread is, never a separately-coded shortcut that could drift
    from it. Callers that want the pre-existing, unwrapped single-run return shape should keep
    calling ``run_answer_quality`` directly — this function's own return type is intentionally
    different (``RepeatedAnswerQualityReport``, not ``AnswerQualityReport``), so a caller cannot
    mistake one for the other.
    """
    from mu_eval.repeats import run_n_times

    async def _once(run_id: str) -> AnswerQualityReport:
        return await run_answer_quality(conversations=conversations, run_id=run_id, **kwargs)

    reports = await run_n_times(num_runs=num_runs, run_id_prefix=run_id_prefix, run_once=_once)
    run_ids = [r.run_id for r in reports]

    overall_accuracy = summarize_repeats(
        metric_name="overall.accuracy",
        values=[r.overall.accuracy for r in reports],
        run_ids=run_ids,
    )
    overall_accuracy_all_scoreable = summarize_repeats(
        metric_name="overall.accuracy_all_scoreable",
        values=[r.overall.accuracy_all_scoreable for r in reports],
        run_ids=run_ids,
    )

    # Only categories present in EVERY run are summarized (a category can be legitimately absent
    # from a run's `by_category` — see the `if subset:` guard above); a category present in only
    # some runs would make "spread" compare different-N samples under one label, which is exactly
    # the "silently different row sets" failure item 3 exists to end, one level up.
    common_categories = (
        set.intersection(*(set(r.by_category) for r in reports)) if reports else set()
    )
    by_category_accuracy: dict[str, RepeatSummary] = {}
    for name in sorted(common_categories):
        by_category_accuracy[name] = summarize_repeats(
            metric_name=f"by_category[{name}].accuracy",
            values=[r.by_category[name].accuracy for r in reports],
            run_ids=run_ids,
        )

    pairwise_noise = compute_pairwise_noise(reports)

    return RepeatedAnswerQualityReport(
        runs=reports,
        overall_accuracy=overall_accuracy,
        overall_accuracy_all_scoreable=overall_accuracy_all_scoreable,
        by_category_accuracy=by_category_accuracy,
        pairwise_noise=pairwise_noise,
    )


def compute_pairwise_noise(reports: Sequence[AnswerQualityReport]) -> list[dict[str, Any]]:
    """`compare.compare_runs` between each pair of CONSECUTIVE repeats — see
    `RepeatedAnswerQualityReport.pairwise_noise`'s own docstring for why this, not just the
    accuracy spread, is the honest noise-floor measurement. Lazy import: `compare.py` imports
    `AnswerQualityReport`/`QueryResult` FROM this module, so a module-level import here would be
    circular (the same reason `run_answer_quality` lazy-imports `runner._await_index`)."""
    if len(reports) < 2:
        return []
    import itertools

    from mu_eval.compare import compare_runs

    results: list[dict[str, Any]] = []
    for a, b in itertools.pairwise(reports):
        if not a.rows or not b.rows:
            # `compare_runs` requires per-row data (`keep_rows=True`, the default) to pair on —
            # silently skip rather than raise, so an aggregate-only repeated run still returns its
            # accuracy spread even though it cannot also return row-level churn.
            continue
        results.append(compare_runs(a, b).model_dump(mode="json"))
    return results
