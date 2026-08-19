"""Unit tests for the mem0 LLM fact-extractor WIRING (LLM-real deferred until Azure).

DEV-STANDARDS permits mocks ONLY in pure unit tests of isolated logic: here a fake
``LLMProviderPort`` stands in for the model router so we verify the wiring — the mem0
FACT_RETRIEVAL call is issued with the configured model-group + ``json_object`` response
format, and the returned ``{"facts": [...]}`` strings are structured into ``ExtractedFact``s via
the SAME deterministic decomposer the heuristic uses. The real Azure-backed extraction is
deferred (the box cannot reach Azure); this locks the seam so that path drops in unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from mu_engine.providers._contracts import Completion, Message
from mu_engine.services.extract import (
    ExtractionSettings,
    LlmFactExtractor,
    build_extractor,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 27, 12, 0, 0, tzinfo=UTC)


class _FakeProvider:
    """A fake ``LLMProviderPort`` capturing the call and returning canned mem0 JSON."""

    def __init__(self, text: str) -> None:
        self._text = text
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
        return Completion(text=self._text, model_group=model, model_id="fake:test")


async def test_llm_extractor_structures_returned_facts() -> None:
    provider = _FakeProvider('{"facts": ["Ada lives in Paris", "Ada uses Postgres"]}')
    extractor = LlmFactExtractor(provider, model_group="hard_extract")
    facts = await extractor.extract("...conversation window...", now=NOW)

    preds = sorted(f.predicate for f in facts)
    assert preds == ["lives_in", "uses"]
    # the mem0 Call A was issued against the configured group with json_object format.
    assert provider.calls[0]["model"] == "hard_extract"
    assert provider.calls[0]["response_format"] == "json_object"


async def test_llm_extractor_tolerates_code_fenced_json() -> None:
    provider = _FakeProvider('```json\n{"facts": ["Ada works at Acme"]}\n```')
    extractor = LlmFactExtractor(provider, model_group="hard_extract")
    facts = await extractor.extract("x", now=NOW)
    assert facts[0].predicate == "works_at"
    assert facts[0].object == "Acme"


async def test_llm_extractor_empty_facts_for_chitchat() -> None:
    provider = _FakeProvider('{"facts": []}')
    extractor = LlmFactExtractor(provider, model_group="hard_extract")
    assert await extractor.extract("Hi.", now=NOW) == []


async def test_llm_extractor_bad_json_is_empty_not_crash() -> None:
    provider = _FakeProvider("not json at all")
    extractor = LlmFactExtractor(provider, model_group="hard_extract")
    assert await extractor.extract("x", now=NOW) == []


def test_build_extractor_defaults_to_heuristic() -> None:
    ex = build_extractor(use_llm=False)
    assert ex.name == "heuristic_spo_v1"


def test_build_extractor_llm_requires_wiring() -> None:
    with pytest.raises(ValueError, match="no llm_provider"):
        build_extractor(use_llm=True)


def test_build_extractor_llm_wired() -> None:
    provider = _FakeProvider("{}")
    ex = build_extractor(use_llm=True, llm_provider=provider, model_group="hard_extract")
    assert ex.name == "llm_mem0_v1"


# ------------------------------------------------------------------ LLM degrade floor (union)
# REGRESSION, live-reproduced on the real 0.5B dev SLM: the LLM path returned FEWER propositions
# than the pure-function deterministic decomposer would over the SAME text, so a functional
# contradiction never collided and `superseded` came back 0 where the heuristic gave 1 — the stale
# fact stayed active. A small local model silently dropping below the deterministic floor is a
# strict regression, and nothing guarded it (unlike the adjudicator, which already degrades to
# `_heuristic_only_verdict`). The union can only ADD propositions literally readable from the
# source text, so it can never invent content.


async def test_llm_extractor_unions_the_deterministic_floor_when_the_model_underextracts() -> None:
    """The model returns only ONE of the two facts stated in the text; the floor supplies
    the other."""
    provider = _FakeProvider('{"facts": ["Ada lives in Paris"]}')
    extractor = LlmFactExtractor(provider, model_group="hard_extract")

    facts = await extractor.extract("Ada lives in Paris. Ada works at Acme.", now=NOW)

    by_pred = {f.predicate: f.object for f in facts}
    assert by_pred["lives_in"] == "Paris"
    assert by_pred["works_at"] == "Acme", (
        "the deterministic floor did not backfill the fact the model dropped — a weak model can "
        "still silently lose a stated fact, which is what broke supersession on the real SLM"
    )


async def test_llm_extractor_union_does_not_duplicate_a_fact_both_paths_found() -> None:
    """Identity is case-folded S/P/O + polarity, so the SAME fact found by BOTH the model and the
    floor is stored once — the union must not double-write."""
    provider = _FakeProvider('{"facts": ["ada LIVES IN paris"]}')
    extractor = LlmFactExtractor(provider, model_group="hard_extract")

    facts = await extractor.extract("Ada lives in Paris.", now=NOW)

    assert len(facts) == 1, (
        f"expected one deduped fact, got {[(f.subject, f.object) for f in facts]}"
    )


async def test_llm_extractor_floor_union_can_be_disabled() -> None:
    """The floor is a documented, named setting — a deployment that genuinely wants raw LLM-only
    output can turn it off and get the exact pre-fix behaviour back."""
    provider = _FakeProvider('{"facts": ["Ada lives in Paris"]}')
    extractor = LlmFactExtractor(
        provider,
        model_group="hard_extract",
        settings=ExtractionSettings(llm_union_with_heuristic_floor=False),
    )

    facts = await extractor.extract("Ada lives in Paris. Ada works at Acme.", now=NOW)

    assert [f.predicate for f in facts] == ["lives_in"]
