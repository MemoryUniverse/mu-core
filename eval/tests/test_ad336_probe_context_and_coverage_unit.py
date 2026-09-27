"""Unit tier for the two AD-336 fixes in `eval/mem0_h2h/ours_arm.py`.

Both exist because a lane published a "next step" that could not have run.

1. `LlmExtractProbe`'s wrapper signature had gone stale against `FactExtractorPort.extract`.
   AD-335 widened the port with `context`, and `DistillPipeline._collect_facts` passes it on
   EVERY call (the extractor, not the pipeline, decides whether to act on it). The probe's
   wrapper took only `(self_, text, *, now)`, so `H2H_LLM_EXTRACT=1` — the flag behind AD-332's
   and AD-333's arms E and F, and behind the aggregate re-measurement ADR 0108 names as AD-335's
   own next step — raised `TypeError: _counted() got an unexpected keyword argument 'context'` on
   the first distilled window. Nothing in the eval suite could see it, because the probe was
   only ever exercised against the OLD signature.

2. The per-row export record now carries `gold_word_coverage` beside `gold_in_context`. AD-334
   shipped that metric into `mu_eval/`, but every headline retrieval number in SCORECARD.md comes
   from this harness, which did not compute it — so the metric built precisely to be invariant to
   write-time rewriting was absent from the arms that do write-time rewriting.

Pure: a fake `LLMProviderPort`, zero stores, zero network, zero model.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

_MEM0_H2H_DIR = Path(__file__).resolve().parent.parent / "mem0_h2h"
if str(_MEM0_H2H_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_MEM0_H2H_DIR.parent))

from mem0_h2h.ours_arm import LlmExtractProbe, build_context_row  # noqa: E402

from mu_engine.providers._contracts import Completion, Message, Usage  # noqa: E402
from mu_engine.services.extract import (  # noqa: E402
    ExtractionSettings,
    FactExtractorPort,
    LlmFactExtractor,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)


class _RecordingProvider:
    """Returns canned texts in order and RECORDS every user message it was sent, so a test can
    assert what the extractor actually put in front of the model rather than inferring it."""

    def __init__(self, *texts: str) -> None:
        self._texts = list(texts)
        self.user_messages: list[str] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: str | None = None,
    ) -> Completion:
        self.user_messages.append(next(m.content for m in reversed(messages)))
        text = self._texts.pop(0) if self._texts else "{}"
        return Completion(
            text=text,
            model_group=model,
            model_id="fake:test",
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


# ---------------------------------------------------------------------------------------------
# 1. The probe must mirror the CURRENT port signature.
# ---------------------------------------------------------------------------------------------


async def test_probe_accepts_the_context_kwarg_the_port_now_declares() -> None:
    """MUTATION-CHECKED: narrow `_counted` back to `(self_, text, *, now)` and this raises
    `TypeError: ... unexpected keyword argument 'context'` — which is exactly what every
    `H2H_LLM_EXTRACT=1` arm did on the AD-335 tree, at the first distilled window.

    `FactExtractorPort.extract`'s own signature is asserted here too, so this test fails loudly
    if the port ever drops `context` again rather than silently becoming vacuous.
    """
    assert "context" in FactExtractorPort.extract.__annotations__

    provider = _RecordingProvider('{"facts": ["Ada lives in Paris"]}')
    extractor = LlmFactExtractor(provider, model_group="hard_extract")

    probe = LlmExtractProbe()
    async with probe.watch():
        facts = await extractor.extract("Ada lives in Paris.", now=NOW, context="Ada is French.")

    assert len(facts) == 1
    assert probe.calls == 1


async def test_probe_forwards_the_context_value_not_merely_tolerates_the_kwarg() -> None:
    """A wrapper that accepted `**kwargs` and then dropped them would pass the test above while
    silently disabling the very mechanism the arm is measuring — the AD-328 failure shape ("a
    rerank arm silently degraded to a broken fallback and the numbers came back BETTER"), and the
    reason this second test exists instead of trusting the signature.

    Asserted on what the MODEL was shown: with reference resolution enabled and a referent that is
    a verbatim substring of the context, `LlmFactExtractor` appends `[Sweden]` to the text it hands
    the fact call. If `context` never arrived, resolution is skipped and the bracket is absent.
    """
    provider = _RecordingProvider(
        '{"referent": "Sweden"}', '{"facts": ["Caroline moved from Sweden"]}'
    )
    extractor = LlmFactExtractor(
        provider,
        model_group="hard_extract",
        settings=ExtractionSettings(reference_resolution_enabled=True),
    )

    probe = LlmExtractProbe()
    async with probe.watch():
        await extractor.extract(
            "I have missed the food since I moved from my home country.",
            now=NOW,
            context="My home country, Sweden, has the best cinnamon buns.",
        )

    assert any("[Sweden]" in m for m in provider.user_messages), provider.user_messages
    assert probe.calls == 1


async def test_probe_still_counts_calls_made_without_a_context() -> None:
    """The heuristic-extractor and no-neighbour paths pass no `context` at all; widening the
    wrapper must not have made `context` mandatory."""
    provider = _RecordingProvider('{"facts": ["X uses Y"]}', '{"facts": ["X uses Y"]}')
    extractor = LlmFactExtractor(provider, model_group="hard_extract")

    probe = LlmExtractProbe()
    async with probe.watch():
        await extractor.extract("one.", now=NOW)
        await extractor.extract("two.", now=NOW, context=None)

    assert probe.calls == 2


# ---------------------------------------------------------------------------------------------
# 2. The export row carries BOTH retrieval metrics.
# ---------------------------------------------------------------------------------------------


def _row(**over: object) -> dict:
    kwargs: dict = {
        "query_id": "conv-26::q11",
        "question": "Where did Caroline move from?",
        "gold_answer": "Sweden",
        "category": 4,
        "gold": {"D1:7"},
        "lines": ["- 7 May 2023: Caroline: I moved from my home country."],
        "product_lines": ["- Caroline: I moved from my home country."],
        "items": 1,
        "product_dated": 0,
        "gold_in_context": True,
    }
    kwargs.update(over)
    return build_context_row(**kwargs)  # type: ignore[arg-type]


def test_export_row_carries_gold_word_coverage_beside_gold_in_context() -> None:
    """MUTATION-CHECKED: drop `gold_word_coverage` from `build_context_row`'s returned dict and
    this fails. It is the field whose absence meant AD-332's 4 %-on-a-populated-store arm had to
    be re-diagnosed by hand a day later instead of read off the artifact."""
    row = _row()
    assert "gold_in_context" in row
    assert "gold_word_coverage" in row
    # "Sweden" is nowhere in this context -> the gold answer's own words did not make it.
    assert row["gold_word_coverage"] == 0.0
    # ...and the turn-id join says the gold TURN was retrieved. The two metrics disagreeing on
    # this row is the whole point of reporting both.
    assert row["gold_in_context"] is True


def test_coverage_is_one_when_the_gold_answer_words_are_present() -> None:
    row = _row(product_lines=["- Caroline: I moved from Sweden in 2009."])
    assert row["gold_word_coverage"] == 1.0


def test_coverage_is_computed_over_context_product_not_the_harness_rejoin() -> None:
    """MUTATION-CHECKED: score `lines` instead of `product_lines` and this fails.

    It has to be `context_product`: that is the string the answering model and the judge are given
    (`answer_h2h.py`), while `context` is the harness-side date rejoin that exists nowhere in the
    shipped product (see `sweep_one_arm`'s own comment). Scoring the rejoin would credit the
    engine for words only the benchmark harness could supply.
    """
    row = _row(
        lines=["- 7 May 2023: Caroline: I moved from Sweden."],
        product_lines=["- Caroline: I moved from my home country."],
    )
    assert row["gold_word_coverage"] == 0.0


def test_row_renders_the_no_memories_sentinel_for_both_context_fields() -> None:
    row = _row(lines=[], product_lines=[], items=0, gold_in_context=False)
    assert row["context"] == "(no memories retrieved)"
    assert row["context_product"] == "(no memories retrieved)"


def test_gold_is_exported_sorted_so_the_artifact_diffs_cleanly() -> None:
    row = _row(gold={"D5:2", "D1:7", "D3:1"})
    assert row["gold"] == ["D1:7", "D3:1", "D5:2"]


# ---------------------------------------------------------------------------------------------
# 3. The probe must count EVERY model call one `extract` made, not just the last.
# ---------------------------------------------------------------------------------------------


async def test_probe_sums_usage_across_a_two_call_extract() -> None:
    """MUTATION-CHECKED: replace the accumulating list with a holder that keeps only the most
    recent completion and this fails — 20/4, not 30/6.

    This is the defect the AD-336 measurement arms surfaced in their own artifact: with reference
    resolution ON, `extract` makes a reference call AND a fact call, and the reported
    `prompt_tokens` rose by only 524 across 419 turns. Tokens are a billable dimension
    (CLAUDE.md rule 5), so this under-count is a real measurement defect even on a free local model.
    """
    provider = _RecordingProvider(
        '{"referent": "Sweden"}', '{"facts": ["Caroline moved from Sweden"]}'
    )
    provider_usages = [(10, 2), (20, 4)]

    async def _complete(messages, **kw):  # type: ignore[no-untyped-def]
        prompt, completion = provider_usages[len(provider.user_messages)]
        provider.user_messages.append(next(m.content for m in reversed(messages)))
        text = provider._texts.pop(0) if provider._texts else "{}"
        return Completion(
            text=text,
            model_group=kw["model"],
            model_id="fake:test",
            usage=Usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            ),
        )

    provider.complete = _complete  # type: ignore[assignment,method-assign]
    extractor = LlmFactExtractor(
        provider,
        model_group="hard_extract",
        settings=ExtractionSettings(reference_resolution_enabled=True),
    )

    probe = LlmExtractProbe()
    async with probe.watch():
        await extractor.extract("...my home country.", now=NOW, context="My home country, Sweden.")

    assert probe.calls == 1, "one extract() invocation"
    assert probe.provider_calls == 2, "two MODEL calls inside it"
    assert probe.prompt_tokens == 30
    assert probe.completion_tokens == 6


async def test_provider_calls_equals_calls_for_a_single_call_extract() -> None:
    """With reference resolution off (the shipped default) the two counters must agree — otherwise
    every historical `calls=419` artifact would become ambiguous."""
    provider = _RecordingProvider('{"facts": ["X uses Y"]}')
    extractor = LlmFactExtractor(provider, model_group="hard_extract")
    probe = LlmExtractProbe()
    async with probe.watch():
        await extractor.extract("one.", now=NOW, context="a neighbouring turn.")
    assert (probe.calls, probe.provider_calls) == (1, 1)
