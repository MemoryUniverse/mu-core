"""S2 write-time enrichment extraction — the ONE LLM pass behind ``EnrichmentExtractorPort``
(ADR-0055; ``engine-core-spec.md`` §6.4 stage 4; ``docs/tracking/ARCHITECTURE-DELTAS.md`` AD-241).

PORT of A-mem's ``MemoryNote.analyze_content`` (``other_repos/A-mem/memory_layer.py:308-400``):
one structured-JSON prompt that returns ``{"keywords": [...], "context": "...", "tags": [...]}``
over the memory's content — keywords/context/tags, verbatim shape, not invented. The owner's
brief names this explicitly ("A-mem indexes the content plus an LLM context sentence, keywords
and tags") as the pattern S2 reimplements, as against mem0's ADD/UPDATE/DELETE diff loop (already
ported, for a different question, in ``pipelines/distill.py``).

**One deliberate, cited deviation from the source.** A-mem's own ``analyze_content`` swallows a
JSON-parse failure and returns ``{"keywords": [], "context": "General", "tags": []}``
(``memory_layer.py:386-400``) — a SILENT empty-but-successful result indistinguishable from a
real (if thin) analysis. DEV-STANDARDS forbids exactly that shape ("every failure path is a named
DegradeReason or a typed raise — never a silent wrong answer"). :class:`LlmEnrichmentExtractor`
therefore RAISES :class:`EnrichmentParseError` on a malformed/empty response instead; the caller
(``pipelines.enrichment_worker.EnrichmentWorker``) is the one place that decides what a failed
attempt means for the job (retry with backoff, eventually dead-letter) — never this seam, and
never a fake-successful empty payload landing on a real memory.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.enrichment import EnrichmentPayload
from mu_contracts.ports.time import Clock
from mu_engine.providers._contracts import LLMProviderPort, Message, MessageRole

__all__ = [
    "EnrichmentParseError",
    "EnrichmentSettings",
    "LlmEnrichmentExtractor",
]


class EnrichmentParseError(Exception):
    """The model response was not the required ``{"keywords", "context", "tags"}`` JSON object.
    Never swallowed here (see module docstring) — the worker's retry/backoff/dead-letter policy
    is the only place this outcome is absorbed."""


class EnrichmentSettings(BaseModel, frozen=True):
    """LLM enrichment-call knobs (DEV-STANDARDS rule 3 — central config, never inlined at the
    call site; same tracked-seam pattern as ``ExtractionSettings``/``DistillSettings``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = 512
    temperature: float = 0.0
    # Cost control (the owner's $20 gpt-5 budget note): truncate before the call rather than pay
    # for — and blow the token-per-minute Azure cap on — a pathologically long captured turn. A
    # truncated call still enriches the salient head of the content; it never blocks or drops the
    # raw memory, which keeps its FULL untruncated text regardless (`content` is untouched).
    max_content_chars: int = Field(default=4000, ge=1)
    min_keywords: int = Field(default=3, ge=0)
    min_tags: int = Field(default=3, ge=0)


_AMEM_ANALYZE_PROMPT = (
    "Generate a structured analysis of the following content by:\n"
    "1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)\n"
    "2. Extracting core themes and contextual elements\n"
    "3. Creating relevant categorical tags\n\n"
    "Format the response as a JSON object:\n"
    "{\n"
    '  "keywords": [\n'
    "    // several specific, distinct keywords that capture key concepts and terminology\n"
    "    // Order from most to least important\n"
    "    // Don't include keywords that are the name of the speaker or time\n"
    "    // At least three keywords, but don't be too redundant.\n"
    "  ],\n"
    '  "context":\n'
    "    // one sentence summarizing:\n"
    "    // - Main topic/domain\n"
    "    // - Key arguments/points\n"
    "    // - Intended audience/purpose\n"
    "  ,\n"
    '  "tags": [\n'
    "    // several broad categories/themes for classification\n"
    "    // Include domain, format, and type tags\n"
    "    // At least three tags, but don't be too redundant.\n"
    "  ]\n"
    "}\n\n"
    "Return ONLY the JSON object. No prose outside it.\n\n"
    "Content for analysis:\n"
)


def _parse_amem_analysis(raw: str) -> tuple[list[str], str, list[str]]:
    """Parse A-mem's ``{"keywords": [...], "context": "...", "tags": [...]}`` JSON, tolerant of
    a ```json code fence (the SAME strip A-mem's own ``re.sub(r'^```json\\s*|\\s*```$', ...)``
    performs at ``memory_layer.py:378``). Raises :class:`EnrichmentParseError` rather than A-mem's
    own silent empty-dict fallback — see module docstring."""
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    if not cleaned:
        raise EnrichmentParseError("empty model response")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise EnrichmentParseError(f"not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EnrichmentParseError("response is not a JSON object")
    keywords, context, tags = payload.get("keywords"), payload.get("context"), payload.get("tags")
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise EnrichmentParseError("'keywords' is not a list of strings")
    if not isinstance(context, str) or not context.strip():
        raise EnrichmentParseError("'context' is not a non-empty string")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        raise EnrichmentParseError("'tags' is not a list of strings")
    return (
        [k.strip() for k in keywords if k.strip()],
        context.strip(),
        [t.strip() for t in tags if t.strip()],
    )


@runtime_checkable
class _Extractor(Protocol):
    async def enrich(self, content: str) -> EnrichmentPayload: ...


class LlmEnrichmentExtractor:
    """``EnrichmentExtractorPort`` — PORT of A-mem's ``analyze_content`` (see module docstring),
    behind the canonical ``LLMProviderPort`` (the SAME seam ``LlmFactExtractor`` uses, model-layer
    §2.2), so the call goes through the router's retry/degrade/metering — never a bespoke client.
    """

    name = "llm_amem_v1"

    def __init__(
        self,
        provider: LLMProviderPort,
        *,
        model_group: str,
        clock: Clock,
        settings: EnrichmentSettings | None = None,
    ) -> None:
        self._provider = provider
        self._model_group = model_group
        self._clock = clock
        self._settings = settings or EnrichmentSettings()

    async def enrich(self, content: str) -> EnrichmentPayload:
        s = self._settings
        truncated = content[: s.max_content_chars]
        completion = await self._provider.complete(
            [Message(role=MessageRole.USER, content=_AMEM_ANALYZE_PROMPT + truncated)],
            model=self._model_group,
            max_tokens=s.max_tokens,
            temperature=s.temperature,
            response_format="json_object",
        )
        keywords, context, tags = _parse_amem_analysis(completion.text)
        return EnrichmentPayload(
            keywords=tuple(keywords),
            context=context,
            tags=tuple(tags),
            model=completion.model_id or self._model_group,
            enriched_at=self._enriched_at(),
        )

    def _enriched_at(self) -> datetime:
        return self._clock.now()
