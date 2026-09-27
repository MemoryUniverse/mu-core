"""Unit tests for AD-335 write-time reference resolution (`LlmFactExtractor._resolve_reference`).

DEV-STANDARDS permits mocks ONLY in pure unit tests of isolated logic: a fake ``LLMProviderPort``
stands in for the model router, sequenced so the resolution call and the mem0 fact-retrieval call
each get their own canned response. The mechanism itself — whether a 0.5B local SLM can actually
resolve these references, and how often it hallucinates or misses — was measured against the REAL
model (not mocked) and is written up at
``docs/tracking/eval-runs/2026-09-27-ad335-write-time-reference-resolution/README.md``; these
tests lock the WIRING (the guard, the fail-safe fallback, the default-off posture), which the real
model's unreliability is exactly why the guard exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from mu_engine.providers._contracts import Completion, Message
from mu_engine.services.extract import (
    ExtractionSettings,
    HeuristicSpoExtractor,
    LlmFactExtractor,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 27, 12, 0, 0, tzinfo=UTC)

_ON = ExtractionSettings(reference_resolution_enabled=True)


class _SequencedFakeProvider:
    """A fake ``LLMProviderPort`` returning ONE canned response per call, in order — needed here
    because reference resolution (when enabled) issues a SEPARATE call before the mem0
    fact-retrieval call, and the two need independent canned answers."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: str | None = None,
    ) -> Completion:
        self.calls.append(
            {
                "model": model,
                "response_format": response_format,
                "system": messages[0].content,
                "user": messages[-1].content,
            }
        )
        text = self._texts[len(self.calls) - 1]
        return Completion(text=text, model_group=model, model_id="fake:test")


# ------------------------------------------------------------------------------- default posture


async def test_reference_resolution_disabled_by_default() -> None:
    """`ExtractionSettings()` default is `reference_resolution_enabled=False`: even WITH context,
    only the ONE mem0 fact-retrieval call is issued — no resolution call, byte-identical to
    pre-AD-335 behaviour. MUTATION CHECK: flip the default to `True` and this fails (2 calls)."""
    provider = _SequencedFakeProvider(['{"facts": ["Ada lives in Paris"]}'])
    extractor = LlmFactExtractor(provider, model_group="hard_extract")

    await extractor.extract("Ada lives in Paris.", now=NOW, context="Ada moved to France.")

    assert len(provider.calls) == 1, (
        "reference resolution ran despite the default-off setting -- "
        f"{len(provider.calls)} calls issued"
    )


async def test_reference_resolution_skips_call_when_no_context() -> None:
    """Enabled, but `context=None` (e.g. the item is the first/only one in the distill window):
    no resolution call — there is nothing to resolve against."""
    provider = _SequencedFakeProvider(['{"facts": []}'])
    extractor = LlmFactExtractor(provider, model_group="hard_extract", settings=_ON)

    await extractor.extract("Hi.", now=NOW, context=None)

    assert len(provider.calls) == 1


# --------------------------------------------------------------------------------- happy path


async def test_reference_resolution_augments_the_text_sent_to_fact_retrieval() -> None:
    """A referent that IS a verbatim substring of `context` gets appended in brackets, and THAT
    augmented text — not the original — is what the mem0 fact-retrieval call sees."""
    provider = _SequencedFakeProvider(
        [
            '{"referent": "Sweden"}',
            '{"facts": ["Caroline moved from Sweden"]}',
        ]
    )
    extractor = LlmFactExtractor(provider, model_group="hard_extract", settings=_ON)

    facts = await extractor.extract(
        "I don't talk to my family much since I moved from my home country.",
        now=NOW,
        context="Oh, my home country, Sweden, is very cold this time of year.",
    )

    assert len(provider.calls) == 2
    fact_retrieval_user_msg = provider.calls[1]["user"]
    assert fact_retrieval_user_msg == (
        "I don't talk to my family much since I moved from my home country. [Sweden]"
    )
    assert any(f.object == "Sweden" for f in facts)


# ------------------------------------------------------------------------------- the guard


async def test_reference_resolution_rejects_a_referent_not_verbatim_in_context() -> None:
    """REGRESSION target for AD-335's own measured finding: the real 0.5B SLM sometimes echoes a
    phrase from TARGET_TURN itself (not CONTEXT_TURNS) as if it were the referent. Rejected here
    by the ALREADY-IN-TARGET no-op check (the echoed phrase is, by construction, already in the
    text) — belt-and-braces with the separate verbatim-in-context guard the sibling
    hallucination test below isolates specifically."""
    provider = _SequencedFakeProvider(
        [
            '{"referent": "the black and white design"}',  # echoed from TARGET, not in context
            '{"facts": []}',
        ]
    )
    extractor = LlmFactExtractor(provider, model_group="hard_extract", settings=_ON)

    await extractor.extract(
        "The black and white design is beautiful. Did you make it?",
        now=NOW,
        context="Yeah, I made this bowl in my class.",
    )

    fact_retrieval_user_msg = provider.calls[1]["user"]
    assert fact_retrieval_user_msg == "The black and white design is beautiful. Did you make it?"


async def test_reference_resolution_rejects_a_hallucinated_referent_on_a_no_reference_turn() -> (
    None
):
    """REGRESSION target for AD-335's own measured finding: on a turn with NO vague reference at
    all, the real 0.5B SLM still sometimes returns a referent-shaped answer (a false positive).
    The referent here ("a work meeting") is present in NEITHER `context` NOR the target text
    itself, so this isolates the verbatim-in-CONTEXT guard specifically — unlike the sibling test
    above, where the echoed referent happens to also be caught by the separate
    already-in-target no-op check. MUTATION CHECK: delete the
    `referent.casefold() not in bounded_context.casefold()` guard and only THIS test goes RED
    (confirmed live: with only that guard removed, the sibling test above still passes on the
    already-in-target check alone, so it alone would not have caught the regression)."""
    provider = _SequencedFakeProvider(
        [
            '{"referent": "a work meeting"}',  # not present anywhere in context OR target
            '{"facts": ["Ada has a meeting at 3pm tomorrow"]}',
        ]
    )
    extractor = LlmFactExtractor(provider, model_group="hard_extract", settings=_ON)

    await extractor.extract(
        "I have a meeting at 3pm tomorrow.",
        now=NOW,
        context="The weather is nice today.",
    )

    assert provider.calls[1]["user"] == "I have a meeting at 3pm tomorrow."


async def test_reference_resolution_noop_when_referent_already_in_target() -> None:
    """A referent that IS verbatim in `context` but is ALSO already stated in the text itself is
    not re-appended — nothing to add."""
    provider = _SequencedFakeProvider(
        [
            '{"referent": "Sweden"}',
            '{"facts": ["Caroline is from Sweden"]}',
        ]
    )
    extractor = LlmFactExtractor(provider, model_group="hard_extract", settings=_ON)

    await extractor.extract(
        "Sweden is where I grew up.",
        now=NOW,
        context="Oh, Sweden, that is very cold this time of year.",
    )

    assert provider.calls[1]["user"] == "Sweden is where I grew up."


# --------------------------------------------------------------------------------- fail-safe


async def test_reference_resolution_empty_referent_is_a_noop() -> None:
    provider = _SequencedFakeProvider(['{"referent": ""}', '{"facts": []}'])
    extractor = LlmFactExtractor(provider, model_group="hard_extract", settings=_ON)

    await extractor.extract("Hi there.", now=NOW, context="Some prior turn.")

    assert provider.calls[1]["user"] == "Hi there."


async def test_reference_resolution_malformed_json_falls_back_to_original_text() -> None:
    """The resolution call's response fails to parse: fail-safe, never crash, original text used
    unchanged — same posture as `_parse_mem0_facts` on malformed JSON."""
    provider = _SequencedFakeProvider(["not json at all", '{"facts": []}'])
    extractor = LlmFactExtractor(provider, model_group="hard_extract", settings=_ON)

    await extractor.extract("Hi there.", now=NOW, context="Some prior turn.")

    assert provider.calls[1]["user"] == "Hi there."


async def test_reference_resolution_context_is_bounded() -> None:
    """`reference_resolution_max_context_chars` bounds the prompt-size/cost of the context sent
    to the resolution call — never the whole, unbounded neighbour text."""
    settings = ExtractionSettings(
        reference_resolution_enabled=True, reference_resolution_max_context_chars=10
    )
    provider = _SequencedFakeProvider(['{"referent": ""}', '{"facts": []}'])
    extractor = LlmFactExtractor(provider, model_group="hard_extract", settings=settings)

    long_context = "x" * 1000
    await extractor.extract("Hi.", now=NOW, context=long_context)

    resolution_user_msg = str(provider.calls[0]["user"])
    assert "x" * 1000 not in resolution_user_msg
    assert "x" * 10 in resolution_user_msg


# ---------------------------------------------------------------- heuristic extractor conformance


async def test_heuristic_extractor_accepts_and_ignores_context() -> None:
    """`FactExtractorPort.extract` grew a `context` kwarg for AD-335; the deterministic extractor
    has no model call to feed it into and must accept + ignore it without error (Protocol
    conformance, `build_extractor(use_llm=False)`'s runtime type)."""
    extractor = HeuristicSpoExtractor()
    facts = await extractor.extract("Ada lives in Paris.", now=NOW, context="some neighbour turn")
    assert facts[0].object == "Paris"
