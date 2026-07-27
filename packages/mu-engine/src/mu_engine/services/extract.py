"""Fact extraction — the S3 MTM->LTM "what is a fact" seam (data-extraction-methodology §3.2).

Two interchangeable implementations behind ONE ``FactExtractorPort`` (DEV-STANDARDS rule 6,
strategy pattern):

* :class:`HeuristicSpoExtractor` — the **MVP default**: a deterministic, no-LLM SPO/triple
  extractor over English declarative text. It is REAL (works offline, no model call) so the
  whole MTM->LTM path is integration-testable now against a live FalkorDB. It is honestly a
  heuristic (documented pattern set + limits), not a stub.
* :class:`LlmFactExtractor` — the quality path: PORT of the mem0 fact-extraction call
  (``other_repos/mem0/mem0/memory/main.py:432-456`` + ``FACT_RETRIEVAL_PROMPT`` at
  ``other_repos/mem0/mem0/configs/prompts.py:14``). The LLM returns the *salient atomic
  fact strings* (mem0's "Call A"); the SPO structuring reuses the SAME deterministic
  decomposer so the two extractors emit identical ``ExtractedFact`` shapes. Wired behind the
  canonical ``LLMProviderPort`` (model router). LLM-**real** integration is DEFERRED until
  Azure reachability (box can't reach azure); the wiring is unit-tested with a fake provider.

The mem0 ADD/UPDATE/DELETE(->SUPERSEDE)/NOOP *diff loop* itself lives in the DISTILL pipeline
(``pipelines/distill.py``) — extraction answers "what facts", the diff loop answers "vs what
is already in LTM".
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from mu_engine.providers._contracts import LLMProviderPort, Message, MessageRole
from mu_engine.storage.domain.memory import FactObjectKind, Polarity

__all__ = [
    "ExtractedFact",
    "FactExtractorPort",
    "HeuristicSpoExtractor",
    "LlmFactExtractor",
    "decompose_to_spo",
]


class ExtractedFact(BaseModel):
    """One atomic SPO proposition distilled from source text (methodology §3.2 a/b).

    ``valid_at`` carries the world-time truth start when a date was recovered from the text;
    ``valid_at_inferred=True`` (the default) marks that no date was found and the DISTILL
    pipeline must fall back to ``recorded_at`` LOUDLY (CANONICAL §7.17 / DegradeReason
    ``DATE_EXTRACTION_FALLBACK``). ``source_span`` is the OPTIONAL LangExtract char-offset
    provenance anchor (methodology §2.6, validate-before-locking) into the source ``text``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str
    predicate: str
    object: str
    polarity: Polarity = Polarity.POSITIVE
    object_kind: FactObjectKind = FactObjectKind.LITERAL
    content: str
    valid_at: datetime | None = None
    valid_at_inferred: bool = True
    source_span: tuple[int, int] | None = None


@runtime_checkable
class FactExtractorPort(Protocol):
    """The swappable extractor seam (methodology §3.2; strategy pattern, registry-selectable).

    ``name`` is the strategy key used for content-free observability + settings selection.
    """

    name: str

    async def extract(self, text: str, *, now: datetime) -> list[ExtractedFact]: ...


# ---------------------------------------------------------------------------------------------
# Deterministic SPO decomposition — shared by BOTH extractors (DRY, DEV-STANDARDS rule 6).
# ---------------------------------------------------------------------------------------------

# A trailing temporal clause -> world-time valid_at (Graphiti valid_at, edge_operations.py:205).
# ISO `YYYY-MM-DD` or a bare year; kept deliberately narrow (deterministic, no NLU).
_TEMPORAL_TAIL = re.compile(
    r"\s+(?:in|on|since|from|as of)\s+" r"((?:19|20)\d{2}(?:-\d{2}-\d{2})?)\.?\s*$",
    re.IGNORECASE,
)

# Copulas -> predicate "is"; negation markers flip polarity (mem0 stores the negative too).
_COPULAS = ("is not", "are not", "was not", "were not", "isn't", "aren't", "wasn't", "weren't")
_COPULAS_POS = ("is", "are", "was", "were")

# Prepositional verbs: predicate = "<verb>_<prep>" (e.g. lives_in, works_at). Single-cardinality
# ones (lives_in/works_at/...) are configured functional in DistillSettings, not here.
_PREP_VERBS = frozenset(
    {
        "lives",
        "live",
        "lived",
        "resides",
        "reside",
        "resided",
        "stays",
        "stay",
        "works",
        "work",
        "worked",
        "based",
        "located",
        "moved",
        "relocated",
    }
)
_PREPS = frozenset({"in", "at", "to", "for", "on", "with", "from"})

# Transitive verbs: predicate = the verb itself (e.g. uses, prefers, owns).
_TRANS_VERBS = frozenset(
    {
        "uses",
        "use",
        "used",
        "likes",
        "like",
        "liked",
        "prefers",
        "prefer",
        "preferred",
        "owns",
        "own",
        "owned",
        "has",
        "have",
        "had",
        "knows",
        "know",
        "speaks",
        "speak",
        "drives",
        "drive",
        "plays",
        "play",
        "loves",
        "love",
        "wants",
        "want",
        "needs",
        "need",
        "enjoys",
        "enjoy",
        "supports",
        "support",
    }
)

# Pronominal / expletive subjects carry no durable identity -> dropped (noise, methodology §1.2
# "chit-chat" residue; mem0 FACT_RETRIEVAL drops "There are branches in trees").
_NOISE_SUBJECTS = frozenset(
    {"there", "it", "this", "that", "here", "they", "we", "you", "i", "he", "she"}
)
_ARTICLES = frozenset({"a", "an", "the"})
_MIN_TOKENS = 3


def _parse_temporal(raw: str) -> datetime:
    if len(raw) == 4:  # bare year -> Jan 1 UTC (deterministic floor)
        return datetime(int(raw), 1, 1, tzinfo=UTC)
    return datetime.fromisoformat(raw).replace(tzinfo=UTC)


def _strip_article(tokens: list[str]) -> str:
    while tokens and tokens[0].lower() in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens).strip()


def _find_word(lowered: str, phrase: str) -> int | None:
    """Index of ``phrase`` in ``lowered`` on whole-word boundaries, else ``None``."""
    m = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", lowered)
    return m.start() if m else None


def _emit(
    subject: str,
    predicate: str,
    obj: str,
    *,
    polarity: Polarity,
    content: str,
    valid_at: datetime | None,
    span: tuple[int, int] | None,
) -> ExtractedFact | None:
    subject = subject.strip()
    obj = obj.strip()
    predicate = predicate.strip().lower().replace(" ", "_")
    if not subject or not predicate or not obj:
        return None
    if subject.lower() in _NOISE_SUBJECTS:
        return None
    return ExtractedFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        polarity=polarity,
        content=content,
        valid_at=valid_at,
        valid_at_inferred=valid_at is None,
        source_span=span,
    )


def _decompose_sentence(sentence: str, *, base_offset: int, full_text: str) -> ExtractedFact | None:
    original = sentence.strip()
    if not original:
        return None

    # (1) pull a trailing temporal clause off the tail before matching (keeps date out of object).
    valid_at: datetime | None = None
    m = _TEMPORAL_TAIL.search(original)
    body = original
    if m:
        try:
            valid_at = _parse_temporal(m.group(1))
            body = original[: m.start()].strip()
        except ValueError:
            valid_at = None  # unparseable -> leave for the pipeline's LOUD recorded_at fallback

    # span into the ORIGINAL source text (LangExtract char-offset provenance pattern, §2.6).
    start = full_text.find(original, base_offset)
    span = (start, start + len(original)) if start >= 0 else None

    lowered = body.lower()

    # (2) possessive attribute: "Ada's favorite color is blue".
    poss = re.match(r"^(.+?)'s\s+(.+?)\s+(?:is|are|was|were)\s+(.+)$", body, re.IGNORECASE)
    if poss:
        return _emit(
            poss.group(1),
            poss.group(2),
            _strip_article(poss.group(3).split()),
            polarity=Polarity.POSITIVE,
            content=original,
            valid_at=valid_at,
            span=span,
        )

    # (3) copula (negation-aware): predicate "is". Negatives (multi-word) are tried first so the
    # positive-stem loop never mis-fires on a negated clause.
    for cop, polarity in ((c, Polarity.NEGATIVE) for c in _COPULAS):
        idx = _find_word(lowered, cop)
        if idx is not None:
            subj = body[:idx].strip()
            obj = _strip_article(body[idx + len(cop) :].split())
            return _emit(
                subj,
                "is",
                obj,
                polarity=polarity,
                content=original,
                valid_at=valid_at,
                span=span,
            )
    for cop in _COPULAS_POS:
        idx = _find_word(lowered, cop)
        if idx is not None:
            subj = body[:idx].strip()
            obj = _strip_article(body[idx + len(cop) :].split())
            return _emit(
                subj,
                "is",
                obj,
                polarity=Polarity.POSITIVE,
                content=original,
                valid_at=valid_at,
                span=span,
            )

    # (4) verb-based patterns.
    tokens = body.split()
    negated = any(t.lower() in {"not", "never", "no"} or t.lower().endswith("n't") for t in tokens)
    polarity = Polarity.NEGATIVE if negated else Polarity.POSITIVE
    clean = [t for t in tokens if t.lower() not in {"not", "never", "no"}]
    lowclean = [t.lower().strip(".,;:") for t in clean]

    for i, tok in enumerate(lowclean):
        if tok in _PREP_VERBS and i + 1 < len(lowclean) and lowclean[i + 1] in _PREPS:
            subj = " ".join(clean[:i])
            predicate = f"{tok}_{lowclean[i + 1]}"
            obj = _strip_article(clean[i + 2 :])
            return _emit(
                subj,
                predicate,
                obj,
                polarity=polarity,
                content=original,
                valid_at=valid_at,
                span=span,
            )
    for i, tok in enumerate(lowclean):
        if tok in _TRANS_VERBS and 0 < i < len(clean) - 1:
            subj = " ".join(clean[:i])
            obj = _strip_article(clean[i + 1 :])
            return _emit(
                subj,
                tok,
                obj,
                polarity=polarity,
                content=original,
                valid_at=valid_at,
                span=span,
            )
    return None


def decompose_to_spo(text: str, *, now: datetime) -> list[ExtractedFact]:
    """Deterministically split ``text`` into atomic SPO ``ExtractedFact``s (no model call).

    Sentence-splits on ``.;\\n``, drops chit-chat/noise (too-short, expletive subject, no
    recognised verb), and matches an ordered, documented pattern set:
    possessive-attribute → copula(±negation) → prepositional-verb → transitive-verb. Anything
    that matches no pattern is dropped (honest: this is a heuristic, not a parser). ``now`` is
    accepted for signature symmetry with the port; dates come only from the text itself.
    """
    facts: list[ExtractedFact] = []
    offset = 0
    for raw in re.split(r"[.;\n]", text):
        chunk = raw.strip()
        if chunk:
            if len(chunk.split()) >= _MIN_TOKENS:
                fact = _decompose_sentence(chunk, base_offset=offset, full_text=text)
                if fact is not None:
                    facts.append(fact)
        offset += len(raw) + 1
    return facts


# ---------------------------------------------------------------------------------------------
# MVP default — deterministic, offline, REAL.
# ---------------------------------------------------------------------------------------------
class HeuristicSpoExtractor:
    """Deterministic SPO extractor implementing :class:`FactExtractorPort` (MVP default).

    No model call, no network — the MTM->LTM pipeline is real-integration-testable with this
    alone. Delegates to :func:`decompose_to_spo`.
    """

    name = "heuristic_spo_v1"

    async def extract(self, text: str, *, now: datetime) -> list[ExtractedFact]:
        return decompose_to_spo(text, now=now)


# ---------------------------------------------------------------------------------------------
# Quality path — mem0 fact-extraction call (PORT); LLM-real DEFERRED until Azure reachability.
# ---------------------------------------------------------------------------------------------

# PORT of mem0 FACT_RETRIEVAL_PROMPT (other_repos/mem0/mem0/configs/prompts.py:14). Condensed
# faithfully (recorded deviation, CODE-ADOPTION rule 4): same instruction — return a JSON object
# with a "facts" key holding a list of atomic, self-contained proposition strings; return an
# EMPTY list for chit-chat (mem0's few-shot: "Hi." -> {"facts": []}).
_MEM0_FACT_SYSTEM = (
    "You are a Personal Information Organizer that extracts relevant, atomic facts, "
    "user memories, and preferences from a conversation. Extract personal preferences, "
    "important personal details, plans/intentions, and professional details. Return ONLY a "
    'JSON object of the form {"facts": ["...", "..."]} where each string is a single, '
    "self-contained fact. Return an empty list for greetings or contentless small-talk. "
    "Do not include any prose outside the JSON."
)


class LlmFactExtractor:
    """mem0 fact-extraction call behind the canonical ``LLMProviderPort`` (methodology §2.3).

    Runs mem0's "Call A" (the *what-is-salient* pass) with ``models.hard_extract_model``, then
    reuses the deterministic :func:`decompose_to_spo` to structure each returned fact string
    into an ``ExtractedFact`` — so an LLM-extracted fact and a heuristic-extracted fact are the
    SAME shape downstream. LLM-real integration is DEFERRED (Azure unreachable on this box);
    the wiring is unit-tested with a fake ``LLMProviderPort``.
    """

    name = "llm_mem0_v1"

    def __init__(
        self,
        provider: LLMProviderPort,
        *,
        model_group: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        self._provider = provider
        self._model_group = model_group
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def extract(self, text: str, *, now: datetime) -> list[ExtractedFact]:
        completion = await self._provider.complete(
            [
                Message(role=MessageRole.SYSTEM, content=_MEM0_FACT_SYSTEM),
                Message(role=MessageRole.USER, content=text),
            ],
            model=self._model_group,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            response_format="json_object",
        )
        fact_strings = _parse_mem0_facts(completion.text)
        out: list[ExtractedFact] = []
        for fs in fact_strings:
            out.extend(decompose_to_spo(fs, now=now))
        return out


def _parse_mem0_facts(raw: str) -> list[str]:
    """Parse mem0's ``{"facts": [...]}`` JSON, tolerant of code fences (main.py:441-455)."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", cleaned).strip()
    if not cleaned:
        return []
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    facts = payload.get("facts", []) if isinstance(payload, dict) else []
    return [f for f in facts if isinstance(f, str) and f.strip()]


def build_extractor(
    *,
    use_llm: bool,
    llm_provider: LLMProviderPort | None = None,
    model_group: str | None = None,
) -> FactExtractorPort:
    """Resolve the configured extractor (DEV-STANDARDS rule 6 strategy selection; fail-loud).

    ``use_llm=False`` (MVP default) → :class:`HeuristicSpoExtractor`. ``use_llm=True`` requires
    a wired ``LLMProviderPort`` + ``model_group`` (``models.hard_extract_model``) → the mem0
    LLM extractor; a missing provider is a wiring bug and raises (never a silent downgrade).
    """
    if not use_llm:
        return HeuristicSpoExtractor()
    if llm_provider is None or model_group is None:
        raise ValueError(
            "LLM extractor selected (use_llm=True) but no llm_provider/model_group wired"
        )
    return LlmFactExtractor(llm_provider, model_group=model_group)
