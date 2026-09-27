"""Unit tier for `eval/mem0_h2h/ours_arm.py`'s `LlmExtractProbe` (AD-332).

Pure — a fake `LLMProviderPort` stands in for the model router (same convention as
`packages/mu-engine/tests/services/test_extract_llm_unit.py::_FakeProvider`), zero stores, zero
network. This is the ASSERT-DON'T-INFER instrument AD-328's own register row demands ("a rerank arm
silently degraded to a broken fallback and the numbers came back BETTER") — if this probe ever
mis-counts, every `H2H_LLM_EXTRACT=1` number this project reports is unverifiable, so the probe
itself gets a test, not just the arm it watches.
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

from mem0_h2h.ours_arm import LlmExtractProbe  # noqa: E402

from mu_engine.providers._contracts import Completion, Message, Usage  # noqa: E402
from mu_engine.services.extract import LlmFactExtractor  # noqa: E402

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 27, 12, 0, 0, tzinfo=UTC)


class _FakeProvider:
    """A fake `LLMProviderPort` returning canned mem0 JSON + a real `Usage` block."""

    def __init__(self, text: str, *, prompt_tokens: int, completion_tokens: int) -> None:
        self._text = text
        self._usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: str | None = None,
    ) -> Completion:
        self.calls += 1
        return Completion(
            text=self._text, model_group=model, model_id="fake:test", usage=self._usage
        )


async def test_probe_counts_one_call_and_its_real_usage() -> None:
    provider = _FakeProvider(
        '{"facts": ["Ada lives in Paris"]}', prompt_tokens=123, completion_tokens=7
    )
    extractor = LlmFactExtractor(provider, model_group="hard_extract")

    probe = LlmExtractProbe()
    async with probe.watch():
        facts = await extractor.extract("Ada lives in Paris.", now=NOW)

    assert len(facts) == 1
    assert probe.calls == 1
    assert probe.prompt_tokens == 123
    assert probe.completion_tokens == 7


async def test_probe_accumulates_over_multiple_extract_calls() -> None:
    provider = _FakeProvider('{"facts": ["X uses Y"]}', prompt_tokens=10, completion_tokens=2)
    extractor = LlmFactExtractor(provider, model_group="hard_extract")

    probe = LlmExtractProbe()
    async with probe.watch():
        await extractor.extract("one.", now=NOW)
        await extractor.extract("two.", now=NOW)
        await extractor.extract("three.", now=NOW)

    assert probe.calls == 3
    assert probe.prompt_tokens == 30
    assert probe.completion_tokens == 6


async def test_probe_restores_the_original_method_on_exit() -> None:
    """MUTATION-STYLE: if `watch()` failed to restore `LlmFactExtractor.extract`, a call made
    AFTER the `async with` block would still increment the (now out-of-scope) probe's counters,
    silently leaking instrumentation into an unrelated arm of the same process — exactly the
    class of "it ran, but not the way you think" defect AD-328 found in the rerank path. Asserted
    directly on the class object, not merely on behaviour, so the assertion cannot pass by
    accident if some other call also happens to leave `facts` non-empty.
    """
    original = LlmFactExtractor.extract
    provider = _FakeProvider('{"facts": ["A B C"]}', prompt_tokens=1, completion_tokens=1)
    extractor = LlmFactExtractor(provider, model_group="hard_extract")

    probe = LlmExtractProbe()
    async with probe.watch():
        await extractor.extract("inside.", now=NOW)

    assert LlmFactExtractor.extract is original
    await extractor.extract("outside.", now=NOW)
    assert probe.calls == 1  # the second call, after the context exited, was NOT counted


async def test_probe_reports_zero_when_extractor_never_invoked() -> None:
    """The guard `sweep_one_arm` builds on: `H2H_LLM_EXTRACT=1` with a probe that stayed at
    zero calls must be distinguishable from "it ran" — this is the raw signal that guard reads.
    """
    probe = LlmExtractProbe()
    async with probe.watch():
        pass
    assert probe.calls == 0
