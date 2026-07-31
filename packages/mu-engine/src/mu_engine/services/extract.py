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

from pydantic import BaseModel, ConfigDict, Field

from mu_engine.providers._contracts import LLMProviderPort, Message, MessageRole
from mu_engine.storage.domain.memory import FactObjectKind, Polarity

__all__ = [
    "ExtractedFact",
    "ExtractionSettings",
    "FactExtractorPort",
    "HeuristicSpoExtractor",
    "LlmFactExtractor",
    "decompose_to_spo",
]


class ExtractionSettings(BaseModel):
    """LLM fact-extraction knobs (data-extraction-methodology §2.3).

    Central-config home (DEV-STANDARDS rule 3): ``max_tokens``/``temperature`` are NOT inlined in
    the ``complete()`` call path — they flow from here (the sanctioned defaults, same tracked-seam
    pattern as ``DistillSettings``/``RecallSettings``: wired from ``settings.model.extraction`` when
    that subtree lands, taken explicitly until then, no re-shape). ``temperature=0`` keeps the
    salient-fact pass deterministic (mem0's Call-A determinism, ``main.py:432-456``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = 1024
    temperature: float = 0.0
    # Chit-chat/noise floor for the deterministic decomposer (methodology §1.2): a chunk with
    # fewer than this many whitespace-split tokens is dropped before pattern-matching — never a
    # bare literal at the call site (DEV-STANDARDS rule 3).
    min_tokens: int = 3
    # CONFIG-AND-DATA-FIX-PLAN.md §1.1 Group A: the map-reduce chunk-sizing ratio
    # (``providers/chunking.py::LongTextChunker._window_texts``'s word-budget fallback,
    # ``approx = budget_tokens * chunk_token_ratio``, an approx-tokens-per-word factor for the
    # tokenizer-unavailable path) used to be a bare literal (``* 3 // 4``) inline in that method —
    # promoted here so ``MU_EXTRACTION__CHUNK_TOKEN_RATIO`` reaches it via ``EngineSettings``
    # (``build_model_router(..., chunk_token_ratio=...)`` threads it into ``LongTextChunker``).
    chunk_token_ratio: float = Field(default=0.75, gt=0.0, le=1.0)

    # ---------------------------------------------------------------------------------------
    # D5-quick (CONFIG-AND-DATA-FIX-PLAN.md PART 2 §D5; ARCHITECTURE-CONFORMANCE.md D-6/D-7).
    #
    # The closed, hardcoded verb sets at ``_COPULAS``/``_PREP_VERBS``/``_TRANS_VERBS`` (below)
    # dropped every change/motion sentence outright — "Ada moved her flight to the Denver
    # offsite to Thursday" matched no copula, no listed prep-verb, no listed trans-verb and
    # returned ZERO facts, so the v2/current fact never reached LTM (PIPELINE-TRACE-DIAGNOSIS.md
    # §1). These five fields extend that closed vocabulary WITHOUT touching the hardcoded module
    # constants themselves (D-6: env-overridable via ``EngineSettings`` ->
    # ``MU_EXTRACTION__PREP_VERBS_EXTRA=...`` etc., ``__`` reaches a frozenset/dict field the
    # same way ``functional_predicates`` already does on ``DistillSettings``).
    # ---------------------------------------------------------------------------------------

    # Extends ``_PREP_VERBS`` (verb immediately followed by a preposition: predicate
    # ``<verb>_<prep>``, e.g. "reports to" -> ``reports_to``).
    prep_verbs_extra: frozenset[str] = frozenset({"changed", "switched", "rescheduled", "reports"})
    # Extends ``_TRANS_VERBS`` (predicate = the verb itself). ``expires``/``departs`` may be the
    # LAST token in the sentence once a trailing temporal clause is stripped into ``valid_at``
    # ("Ada's passport expires in 2028." -> body "...expires", no textual object left) — the
    # decomposer falls back to the recovered date as the object rather than dropping the fact.
    trans_verbs_extra: frozenset[str] = frozenset(
        {"manages", "manage", "managed", "departs", "leaves", "expires"}
    )
    # Feeds the NEW "value-change clause" decomposition step (``_match_change_clause``): the
    # "SUBJECT VERB [POSSESSIVE-PRONOUN] NOUN-PHRASE to NEWVAL" / "SUBJECT's NOUN-PHRASE [AUX]
    # VERB to NEWVAL" shapes neither ``_PREP_VERBS`` nor ``_TRANS_VERBS`` can parse (a direct
    # object, or a genitive noun-phrase subject, sits between the verb and the trailing
    # "to <value>" clause). ``upgraded``/``extended`` are NOT in the task's literal verb list —
    # added to cover this repo's own gold-set mangled sentences verbatim (``bo_laptop_v2``
    # "upgraded ... to a MacBook Pro 16-inch", ``ada_deadline_v2`` "got extended to October
    # 24th"; ``demos/pipeline_trace/gold_items.py``), documented here rather than silently.
    change_value_verbs: frozenset[str] = frozenset(
        {
            "moved",
            "changed",
            "updated",
            "rescheduled",
            "switched",
            "renamed",
            "relocated",
            "upgraded",
            "extended",
        }
    )
    # "X became Y" is a state-change COPULA, not a value-change clause (no trailing "to Y") —
    # aliases into the SAME predicate ("is") the plain copula step already emits, so a
    # "became"-sentence collides with an "is"-sentence about the same subject for free.
    copula_change_verbs: frozenset[str] = frozenset({"became", "becomes"})
    # Filler words stripped from the sentence BODY (whole-word, case-insensitive) before any
    # pattern is tried — "now" (task list: marks present-tense currency, e.g. "...is now
    # Thursday") and "named" (copula-adjacent filler, "her sister is named Mira" -> object
    # cleanup item 2) never reach subject/predicate/object.
    filler_words: frozenset[str] = frozenset({"now", "named"})
    # D-7 predicate normalization: literal computed surface-predicate -> canonical override,
    # checked BEFORE the generic head-initial-preposition rule in ``_canonicalize_predicate``. A
    # small, NAMED, extensible exception table (methodology's "at least a stable canonical
    # predicate" floor) — NOT meant to be exhaustive; ops extend via
    # ``MU_EXTRACTION__CANONICAL_PREDICATE_MAP='{"...": "..."}'``.
    canonical_predicate_map: dict[str, str] = Field(
        default_factory=lambda: {"hotel_booking": "hotel", "sister": "has_sibling"}
    )


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

# D5-quick (grammatical function words — NOT a vocabulary/domain knob, so these stay code
# constants same as ``_ARTICLES``/``_PREPS``, unlike the settings-driven verb sets above).
# Auxiliary run stripped off the tail of a value-change clause's pre-verb phrase ("...standup
# **was** moved to 9:30am", "...deadline **got** extended to October 24th").
_AUX_WORDS = frozenset({"was", "is", "are", "were", "has", "have", "had", "got", "been"})
# Possessive pronoun stripped off the head of the noun-phrase between a change-verb and its
# trailing "to <newval>" clause ("moved **her** flight to ... to Thursday").
_POSSESSIVE_PRONOUNS = frozenset({"his", "her", "their", "its", "my", "your", "our"})
# Leading prepositions stripped from an extracted OBJECT (methodology object-cleanup item 2):
# "on Tuesday" -> "Tuesday", "in Room A" -> "Room A". Deliberately broader than ``_PREPS`` (adds
# "into"/"as") since an object never legitimately starts with one of these as live content.
_OBJECT_LEADING_PREPS = frozenset(
    {"in", "on", "at", "for", "with", "from", "of", "into", "to", "as"}
)
# Constructor/param DEFAULT only (DEV-STANDARDS rule 3: no hardcoded constant lives in the
# decomposer LOGIC) — mirrors ``ExtractionSettings.min_tokens``; the live value is DI-threaded
# by :class:`HeuristicSpoExtractor` (a bare :func:`decompose_to_spo` call, e.g. in a unit test,
# still gets a sane, named default).
_DEFAULT_MIN_TOKENS = 3


def _parse_temporal(raw: str) -> datetime:
    if len(raw) == 4:  # bare year -> Jan 1 UTC (deterministic floor)
        return datetime(int(raw), 1, 1, tzinfo=UTC)
    return datetime.fromisoformat(raw).replace(tzinfo=UTC)


def _clean_object(tokens: list[str], *, settings: ExtractionSettings) -> str:
    """D5-quick object cleanup (methodology item 2): iteratively strip a leading article,
    filler word (``settings.filler_words`` — e.g. a leftover "named"), preposition
    (``on Tuesday`` -> ``Tuesday``, ``in Room A`` -> ``Room A``), or change-verb-phrase
    (``moved to Room B`` -> ``Room B``, a leftover un-decomposed clause the plain copula step
    can still hand this function) off an extracted object. Loops to a fixed point so a chain
    ("the on Tuesday" — pathological, but cheap to cover) fully resolves.
    """
    changed = True
    while tokens and changed:
        changed = False
        head = tokens[0].lower().strip(".,;:")
        if head in _ARTICLES or head in settings.filler_words or head in _OBJECT_LEADING_PREPS:
            tokens = tokens[1:]
            changed = True
            continue
        if (
            head in settings.change_value_verbs
            and len(tokens) > 1
            and tokens[1].lower().strip(".,;:") in _PREPS
        ):
            tokens = tokens[2:]
            changed = True
    return " ".join(tokens).strip()


def _strip_filler_words(body: str, filler_words: frozenset[str]) -> str:
    """Removes every whole-word occurrence of a filler word (case-insensitive) from ``body``
    BEFORE any pattern is tried, so it never leaks into subject/predicate/object (D5-quick item
    2: "her sister is **named** Mira" -> "her sister is Mira"). No-op for an empty vocab."""
    if not filler_words:
        return body
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(w) for w in sorted(filler_words)) + r")(?!\w)",
        re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", pattern.sub("", body)).strip()


def _canonicalize_predicate(phrase: str, *, settings: ExtractionSettings) -> str:
    """D-7 predicate normalization: a raw, space-joined predicate PHRASE (e.g. from the
    possessive-attribute pattern's noun-phrase group, or a value-change clause's genitive/
    direct-object split) -> a canonical snake_case predicate, so two surface forms of the SAME
    underlying relation collide on ``predicate`` (the unblocker for D2 supersession).

    Order: (1) a leading article is dropped: ``"flight to the Denver offsite"`` never lets one
    survive mid-phrase past step (3) anyway, but a leading one ("the Q3 planning meeting") would
    otherwise poison the snake_case key. (2) the FULL computed snake_case is checked against
    ``settings.canonical_predicate_map`` first (a named override, e.g. ``hotel_booking`` ->
    ``hotel``, ``sister`` -> ``has_sibling``). (3) failing that, a head-initial-preposition rule:
    a phrase containing a preposition ("flight **to** the Denver offsite", "hotel **in**
    Tokyo") is a prepositional-attachment noun phrase in English — everything BEFORE the first
    preposition is the head noun and becomes the canonical predicate ("flight", "hotel"), with
    the map consulted once more on that shorter head. (4) no preposition -> the phrase is a
    (possibly compound) noun with no attachment to strip; its own snake_case IS the canonical
    predicate (methodology's "at least a stable canonical predicate" floor — not claimed to be
    semantically ideal for every compound, just STABLE across the same phrase's v1/v2 forms).
    """
    words = [w for w in phrase.split() if w]
    while words and words[0].lower() in _ARTICLES:
        words = words[1:]
    if not words:
        return ""
    snake = "_".join(w.lower() for w in words)
    if snake in settings.canonical_predicate_map:
        return settings.canonical_predicate_map[snake]
    for j, w in enumerate(words):
        if w.lower() in _PREPS and j > 0:
            head = "_".join(x.lower() for x in words[:j])
            return settings.canonical_predicate_map.get(head, head)
    return snake


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


def _match_change_clause(
    body: str,
    *,
    original: str,
    valid_at: datetime | None,
    span: tuple[int, int] | None,
    settings: ExtractionSettings,
) -> ExtractedFact | None:
    """D5-quick value-change clause (CONFIG-AND-DATA-FIX-PLAN.md PART 2 §D5; the D-6 unblocker).

    Recognises a change-of-value sentence built around one of ``settings.change_value_verbs``
    with a trailing "to <newval>" clause — the shape none of the pre-existing patterns parse,
    because either a direct-object noun phrase (Pattern A) or a genitive noun-phrase subject
    (Pattern B) sits BETWEEN the verb and the value:

    * Pattern A — "SUBJECT VERB [POSSESSIVE-PRONOUN] NOUN-PHRASE to NEWVAL"
      ("Ada **moved** her **flight to the Denver offsite** to Thursday"): the pre-verb phrase IS
      the bare subject; the predicate comes from the mid noun-phrase (possessive pronoun
      stripped), canonicalized (D-7) so it collides with the SAME noun-phrase's
      possessive-attribute predicate from a v1 sentence ("Ada's flight to the Denver offsite is
      on Tuesday" -> predicate ``flight``, same canonicalization).
    * Pattern B — "SUBJECT's NOUN-PHRASE [AUX] VERB to NEWVAL"
      ("Ada's project deadline **got extended** to October 24th", "Bo's team standup **moved**
      to 9:30am"): the pre-verb phrase splits on the genitive exactly like the possessive-
      attribute pattern, so subject/predicate collide with THAT pattern's v1 fact.
    * A determiner-led noun phrase with neither a genitive nor a mid ("The Q3 planning meeting
      **was moved** to Room B") gets the copula-default predicate ``is`` — the same predicate
      the plain copula step already assigns to "The Q3 planning meeting **is** in Room A",
      so the two collide too.

    A bare "SUBJECT VERB to NEWVAL" with NEITHER a genitive NOR a mid NOR a leading determiner
    ("Ada moved to Berlin") is deliberately left unmatched (returns ``None``) — that shape is
    ALREADY handled correctly by the existing prepositional-verb pattern (predicate
    ``moved_to``); this function must never regress a working pattern to claim a new one.
    """
    tokens = body.split()
    if not tokens:
        return None
    negated = any(t.lower() in {"not", "never", "no"} or t.lower().endswith("n't") for t in tokens)
    polarity = Polarity.NEGATIVE if negated else Polarity.POSITIVE
    clean = [t for t in tokens if t.lower() not in {"not", "never", "no"}]
    lowclean = [t.lower().strip(".,;:") for t in clean]

    verb_idx = next((i for i, t in enumerate(lowclean) if t in settings.change_value_verbs), None)
    if verb_idx is None:
        return None

    post = clean[verb_idx + 1 :]
    lowpost = [t.strip(".,;:") for t in lowclean[verb_idx + 1 :]]
    # rightmost "to"/"into" with something after it -> the LAST value clause wins ("her flight to
    # the Denver offsite **to Thursday**" — the inner "to" belongs to the noun phrase, not here).
    to_idx = next(
        (
            j
            for j in range(len(lowpost) - 1, -1, -1)
            if lowpost[j] in ("to", "into") and j + 1 < len(lowpost)
        ),
        None,
    )
    if to_idx is None:
        return None

    mid = post[:to_idx]
    newval_tokens = post[to_idx + 1 :]

    pre = clean[:verb_idx]
    while pre and pre[-1].lower().strip(".,;:") in _AUX_WORDS:
        pre = pre[:-1]
    pre_text = " ".join(pre)
    genitive = re.match(r"^(.+?)'s\s+(.+)$", pre_text) if pre_text else None

    if mid:
        subject = pre_text
        mid_words = [t for t in mid if t.lower() not in _POSSESSIVE_PRONOUNS]
        predicate_phrase = " ".join(mid_words)
        predicate = _canonicalize_predicate(predicate_phrase, settings=settings)
    elif genitive:
        subject = genitive.group(1)
        predicate = _canonicalize_predicate(genitive.group(2), settings=settings)
    elif pre and pre[0].lower() in _ARTICLES:
        subject = pre_text
        predicate = "is"
    else:
        return None  # ambiguous bare "SUBJECT VERB to NEWVAL" -> defer to the prep-verb pattern

    if not subject.strip() or not predicate or not newval_tokens:
        return None
    obj = _clean_object(newval_tokens, settings=settings)
    return _emit(
        subject, predicate, obj, polarity=polarity, content=original, valid_at=valid_at, span=span
    )


def _decompose_sentence(
    sentence: str, *, base_offset: int, full_text: str, settings: ExtractionSettings
) -> ExtractedFact | None:
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

    # (1.5) D5-quick: strip filler words (e.g. "named") from the body BEFORE any pattern is
    # tried, so "Ada's sister is named Mira" reads as "Ada's sister is Mira" everywhere below.
    body = _strip_filler_words(body, settings.filler_words)

    # span into the ORIGINAL source text (LangExtract char-offset provenance pattern, §2.6).
    start = full_text.find(original, base_offset)
    span = (start, start + len(original)) if start >= 0 else None

    lowered = body.lower()

    # (1.7) D5-quick value-change clause — tried BEFORE possessive-attribute/copula so an
    # aux+change-verb ("was moved", "got extended") is never mistaken for a plain copula, and a
    # direct-object noun phrase between the verb and a trailing "to <newval>" is captured as the
    # predicate instead of being swallowed into a dirty object.
    change_fact = _match_change_clause(
        body, original=original, valid_at=valid_at, span=span, settings=settings
    )
    if change_fact is not None:
        return change_fact

    # (2) possessive attribute: "Ada's favorite color is blue".
    poss = re.match(r"^(.+?)'s\s+(.+?)\s+(?:is|are|was|were)\s+(.+)$", body, re.IGNORECASE)
    if poss:
        return _emit(
            poss.group(1),
            _canonicalize_predicate(poss.group(2), settings=settings),
            _clean_object(poss.group(3).split(), settings=settings),
            polarity=Polarity.POSITIVE,
            content=original,
            valid_at=valid_at,
            span=span,
        )

    # (3) copula (negation-aware): predicate "is". Negatives (multi-word) are tried first so the
    # positive-stem loop never mis-fires on a negated clause. "became"/"becomes"
    # (``copula_change_verbs``) alias to the SAME "is" predicate (a state-change copula).
    for cop, polarity in ((c, Polarity.NEGATIVE) for c in _COPULAS):
        idx = _find_word(lowered, cop)
        if idx is not None:
            subj = body[:idx].strip()
            obj = _clean_object(body[idx + len(cop) :].split(), settings=settings)
            return _emit(
                subj,
                "is",
                obj,
                polarity=polarity,
                content=original,
                valid_at=valid_at,
                span=span,
            )
    for cop in (*_COPULAS_POS, *settings.copula_change_verbs):
        idx = _find_word(lowered, cop)
        if idx is not None:
            subj = body[:idx].strip()
            obj = _clean_object(body[idx + len(cop) :].split(), settings=settings)
            return _emit(
                subj,
                "is",
                obj,
                polarity=Polarity.POSITIVE,
                content=original,
                valid_at=valid_at,
                span=span,
            )

    # (4) verb-based patterns. Vocab = the closed module constant UNION the settings-driven
    # extension (D-6: env-overridable without touching this code, ``MU_EXTRACTION__*_EXTRA``).
    prep_verbs = _PREP_VERBS | settings.prep_verbs_extra
    trans_verbs = _TRANS_VERBS | settings.trans_verbs_extra
    tokens = body.split()
    negated = any(t.lower() in {"not", "never", "no"} or t.lower().endswith("n't") for t in tokens)
    polarity = Polarity.NEGATIVE if negated else Polarity.POSITIVE
    clean = [t for t in tokens if t.lower() not in {"not", "never", "no"}]
    lowclean = [t.lower().strip(".,;:") for t in clean]

    for i, tok in enumerate(lowclean):
        if tok in prep_verbs and i + 1 < len(lowclean) and lowclean[i + 1] in _PREPS:
            subj = " ".join(clean[:i])
            predicate = f"{tok}_{lowclean[i + 1]}"
            obj = _clean_object(clean[i + 2 :], settings=settings)
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
        if tok in trans_verbs and 0 < i <= len(clean) - 1:
            # the object is normally everything after the verb; if the verb is the LAST token
            # (its object was consumed by the temporal-tail strip in step (1), e.g. "Ada's
            # passport expires in 2028" -> body "...expires") fall back to the recovered date
            # rather than dropping a fact that WAS there (D5-quick vocab-add coverage for
            # ``expires``/``departs``); no date recovered -> keep scanning for another verb.
            if i + 1 < len(clean):
                obj = _clean_object(clean[i + 1 :], settings=settings)
            elif valid_at is not None:
                obj = valid_at.date().isoformat()
            else:
                continue
            subj = " ".join(clean[:i])
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


def decompose_to_spo(
    text: str,
    *,
    now: datetime,
    min_tokens: int = _DEFAULT_MIN_TOKENS,
    settings: ExtractionSettings | None = None,
) -> list[ExtractedFact]:
    """Deterministically split ``text`` into atomic SPO ``ExtractedFact``s (no model call).

    Sentence-splits on ``.;\\n``, drops chit-chat/noise (too-short, expletive subject, no
    recognised verb), and matches an ordered, documented pattern set: value-change clause (D5-
    quick) → possessive-attribute → copula(±negation, incl. "became"/"becomes") →
    prepositional-verb → transitive-verb. Anything that matches no pattern is dropped (honest:
    this is a heuristic, not a parser). ``now`` is accepted for signature symmetry with the
    port; dates come only from the text itself. ``min_tokens``/``settings`` are DI-threaded from
    :class:`ExtractionSettings` by :class:`HeuristicSpoExtractor`/:class:`LlmFactExtractor`
    (never a bare literal here); a bare call (e.g. from a unit test) gets
    ``ExtractionSettings()``'s sane defaults.
    """
    settings = settings or ExtractionSettings()
    facts: list[ExtractedFact] = []
    offset = 0
    for raw in re.split(r"[.;\n]", text):
        chunk = raw.strip()
        if chunk:
            if len(chunk.split()) >= min_tokens:
                fact = _decompose_sentence(
                    chunk, base_offset=offset, full_text=text, settings=settings
                )
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

    def __init__(self, *, settings: ExtractionSettings | None = None) -> None:
        # min_tokens flows from central config, never inlined here (rule 3).
        self._settings = settings or ExtractionSettings()

    async def extract(self, text: str, *, now: datetime) -> list[ExtractedFact]:
        return decompose_to_spo(
            text, now=now, min_tokens=self._settings.min_tokens, settings=self._settings
        )


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
        settings: ExtractionSettings | None = None,
    ) -> None:
        self._provider = provider
        self._model_group = model_group
        # max_tokens/temperature flow from central config, never inlined here (rule 3).
        self._settings = settings or ExtractionSettings()

    async def extract(self, text: str, *, now: datetime) -> list[ExtractedFact]:
        completion = await self._provider.complete(
            [
                Message(role=MessageRole.SYSTEM, content=_MEM0_FACT_SYSTEM),
                Message(role=MessageRole.USER, content=text),
            ],
            model=self._model_group,
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
            response_format="json_object",
        )
        fact_strings = _parse_mem0_facts(completion.text)
        out: list[ExtractedFact] = []
        for fs in fact_strings:
            out.extend(
                decompose_to_spo(
                    fs, now=now, min_tokens=self._settings.min_tokens, settings=self._settings
                )
            )
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
    settings: ExtractionSettings | None = None,
) -> FactExtractorPort:
    """Resolve the configured extractor (DEV-STANDARDS rule 6 strategy selection; fail-loud).

    ``use_llm=False`` (MVP default) → :class:`HeuristicSpoExtractor`. ``use_llm=True`` requires
    a wired ``LLMProviderPort`` + ``model_group`` (``models.hard_extract_model``) → the mem0
    LLM extractor; a missing provider is a wiring bug and raises (never a silent downgrade).
    ``settings`` threads the central-config ``max_tokens``/``temperature`` (DEV-STANDARDS rule 3 —
    never hardcoded in the extractor's ``complete()`` call).
    """
    if not use_llm:
        return HeuristicSpoExtractor()
    if llm_provider is None or model_group is None:
        raise ValueError(
            "LLM extractor selected (use_llm=True) but no llm_provider/model_group wired"
        )
    return LlmFactExtractor(llm_provider, model_group=model_group, settings=settings)
