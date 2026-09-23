"""The baseline retrieval run: real LoCoMo data → real stores → ranked recall → IR metrics.

One conversation = one isolated η partition (own org/workspace), ingested through
``LocalMemory.add`` and queried through ``LocalMemory.recall`` — the same two verbs the product
ships. Nothing is mocked and no store is stubbed; if a store is down the run RAISES.

The run also records the SCORE PROVENANCE of every returned item (``fused_score`` distribution,
per-channel counts, ``is_floor`` counts). That is not decoration: the only recall transcript in
this repo prints ``0.0000`` for every hit, and a scorer that returns zero for everything is either
a display bug or a ranking bug. The distribution recorded here is what tells the two apart.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mu_eval.corpus import IngestReport, TurnIndex, ingest_conversation, local_memory_for
from mu_eval.locomo import CATEGORY_NAMES, Conversation, LabelledQuery
from mu_eval.metrics import QueryScores, aggregate, score_query

__all__ = [
    "NEIGHBOR_EXPANSION_MARKER_ATTR",
    "ArmReport",
    "GoldContextHit",
    "RunReport",
    "ScoreProvenance",
    "classify_query_admission",
    "gold_context_attribution",
    "gold_ids_present",
    "run_baseline",
]

# Coordination point for the neighbour-expansion arm (S1b, `TRACE-0923.md` §7 S1b; recorded as
# AD-228 in `ARCHITECTURE-DELTAS.md`). This harness's own file ownership is `eval/` only
# (CLAUDE.md rule 12 / this lane's brief) — it cannot add a field to `mu_contracts.contracts.
# recall.RecallItemView` (that lives in `services/recall/`'s lane), so it names, here, the ONE
# attribute the READ side has committed to looking for on a recalled item, mirroring the shape
# `RecallItemView.is_floor` already uses for the same kind of "how did this item get here"
# provenance. `getattr(item, NEIGHBOR_EXPANSION_MARKER_ATTR, False)` below means: a surface that
# does not (yet) carry the field is read as "nothing came from expansion" — never a crash, and
# never mistaken for "expansion produced nothing" once the field actually exists but reports
# `False` for a real reason.
NEIGHBOR_EXPANSION_MARKER_ATTR = "is_neighbor"


class ScoreProvenance(BaseModel):
    """What the ranker actually put in ``RecallItemView.fused_score``, over the whole run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: int
    zero_scores: int
    distinct_scores: int
    min_score: float
    median_score: float
    max_score: float
    by_channel: dict[str, int]
    floor_items: int
    degraded_results: int


class ArmReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    queries_scored: int
    queries_skipped_adversarial: int
    # T4 (`TRACE-0923.md` §7): non-adversarial rows with empty `evidence` (LoCoMo category 3
    # rows carrying `evidence: []`) used to be folded into `queries_skipped_adversarial` — they
    # are neither adversarial nor "gold named a turn we never ingested" (`queries_skipped_
    # no_gold_in_corpus`), they are a distinct third reason with no counter of its own before this
    # fix. Default 0 so an older artifact re-read with `AnswerQualityReport.model_validate_json`
    # (`extra="forbid"`) still loads.
    queries_skipped_no_evidence: int = 0
    queries_skipped_no_gold_in_corpus: int
    ks: tuple[int, ...]
    overall: dict[str, dict[int, float]]
    by_category: dict[str, dict[str, dict[int, float]]]
    provenance: ScoreProvenance | None = None

    # The FREE, no-LLM `gold_in_context` metric (TRACE-0923.md "COST DISCIPLINE": needs no answer
    # model and no judge, so use it — with repeats, paired — for every arm). Distinct from
    # `recall_at_k`: `gold_in_context` is the boolean "was ANY gold turn anywhere in what this
    # query's recall actually returned", the same join `gold_ids_present` already gives
    # `answer_quality.py` per-row, now also aggregated here so a retrieval-only run reports it
    # without needing the paid answer-quality path at all. Default 0 so an artifact written before
    # this fix still loads (`extra="forbid"` above).
    gold_in_context: int = 0
    # Of `gold_in_context` above, how many resolved ONLY through an item the recall surface marked
    # `is_neighbor` (S1b) — i.e. no primary-channel item alone would have counted as a hit for
    # that query. This is what makes the neighbour-expansion arm's OWN contribution visible: an
    # expansion that only ever piggybacks on a hit a primary channel already found would leave
    # `gold_in_context` unchanged (correctly — nothing new was retrieved) and this counter at 0;
    # a nonzero value here is retrieval `gold_in_context` could not have reached without
    # expansion. Unconditionally 0 until the recall surface actually sets `is_neighbor` on an
    # item — an honest "not yet measurable", never a fabricated contribution.
    gold_in_context_via_neighbor_only: int = 0
    # Total items across all scored queries that the recall surface marked `is_neighbor`,
    # regardless of whether they were a gold hit — lets a reader see the expansion arm is even
    # switched on (nonzero) before asking whether it moved `gold_in_context` at all.
    neighbor_items_seen: int = 0


class RunReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset: str
    samples: int
    corpus_turns: int
    importance: float
    recall_limit: int
    ingest: list[IngestReport] = Field(default_factory=list)
    arms: list[ArmReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def classify_query_admission(query: LabelledQuery, known: set[str]) -> str | None:
    """Pure three-way admission classification for ONE query against the ingested turn-id set.

    Split out of ``run_baseline``\'s loop (T4, ``TRACE-0923.md`` \u00a77) so the skip/no-skip
    decision is unit-testable without a real store stack: before this fix, a non-adversarial row
    with EMPTY ``evidence`` (LoCoMo category 3 rows such as conv-26 qa[30]/qa[46], which carry
    ``evidence: []``) was folded into the SAME counter as truly adversarial rows
    (``query.is_adversarial or not query.evidence`` at the old ``runner.py:246``), so a run's own
    printed ``skipped_adversarial`` overcounted (49 reported vs 47 actually adversarial on
    conv-26) and undercounted nothing distinct for "had no evidence at all".

    Returns one of ``"adversarial"``, ``"no_evidence"``, ``"no_gold_in_corpus"``, or ``None``
    (the query is eligible to be scored).
    """
    if query.is_adversarial:
        return "adversarial"
    if not query.evidence:
        return "no_evidence"
    if not any(e in known for e in query.evidence):
        return "no_gold_in_corpus"
    return None


def _resolve_ranked_ids(items: Sequence[Any], index: TurnIndex, gold: set[str]) -> list[str]:
    """Map each recalled item back to ONE dataset turn id, best-effort in the system's favour.

    A body may correspond to several turns (identical short lines). If any of them is gold, that
    is the id credited — anything else would score a miss for a system that returned a textually
    identical, equally valid turn. Items whose body is not in the corpus index (nothing this
    harness wrote — e.g. an LTM-distilled derived fact) are kept in the ranking as a non-gold
    placeholder so they still consume their rank position, which is the honest accounting: they
    occupied a result slot the gold turn did not get.
    """
    ranked: list[str] = []
    for position, item in enumerate(items):
        candidates = index.resolve(item.content)
        hit = next((c for c in candidates if c in gold), None)
        if hit is not None:
            ranked.append(hit)
        elif candidates:
            ranked.append(candidates[0])
        else:
            ranked.append(f"__not_in_corpus__#{position}")
    return ranked


class GoldContextHit(BaseModel):
    """Whether gold appears anywhere in a recall's ``items``, broken out by HOW it got there.

    The join itself (``index.resolve(item.content)`` against ``gold``) is content-based and
    therefore already channel-agnostic: an item the neighbour-expansion arm (S1b) adds to
    ``result.items`` is resolved exactly the same way a primary-channel item is, AS LONG AS it
    carries the neighbour's own real body text — which is the only sane way to implement S1b
    (``TurnIndex`` joins ANY item's body back to its source turn, tier- and channel-agnostic by
    design; see its own docstring). So ``present`` below needed no new logic to count an expanded
    neighbour's hit. What ``present`` alone CANNOT show is whether the arm did anything: an
    expansion that only ever re-surfaces a turn a primary channel already ranked in would leave
    `present` looking identical to a run with the arm off. `via_neighbor_only` is the number that
    would move even when `present` does not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    present: bool
    via_neighbor_only: bool
    neighbor_items: int


def gold_context_attribution(
    items: Sequence[Any], index: TurnIndex, gold: set[str]
) -> GoldContextHit:
    """Per-query gold-in-context detail: presence, plus the neighbour-expansion attribution.

    ``via_neighbor_only`` is true iff gold is present AND every item that resolved to a gold turn
    id carries ``NEIGHBOR_EXPANSION_MARKER_ATTR`` (read via ``getattr(..., False)``, so an item
    from a surface that has not adopted the marker — every surface in this repo today — is always
    read as "not expansion", never crashes, and never inflates this count before the field is
    real). ``neighbor_items`` counts every marked item seen, hit or not, so a reader can tell "the
    arm produced nothing" (0) apart from "the arm produced items but none were gold" (nonzero here,
    zero above).
    """
    resolved_hit_is_neighbor: list[bool] = []
    neighbor_items = 0
    for item in items:
        if getattr(item, NEIGHBOR_EXPANSION_MARKER_ATTR, False):
            neighbor_items += 1
        if any(candidate in gold for candidate in index.resolve(item.content)):
            resolved_hit_is_neighbor.append(
                bool(getattr(item, NEIGHBOR_EXPANSION_MARKER_ATTR, False))
            )
    present = bool(resolved_hit_is_neighbor)
    return GoldContextHit(
        present=present,
        via_neighbor_only=present and all(resolved_hit_is_neighbor),
        neighbor_items=neighbor_items,
    )


def gold_ids_present(items: Sequence[Any], index: TurnIndex, gold: set[str]) -> bool:
    """True iff ANY item's body resolves (via ``TurnIndex``) to a turn id in ``gold``.

    This is per-row RETRIEVAL ATTRIBUTION (HOW-THEY-MEASURE-0901.md F1 / ACCURACY-MEASUREMENT-
    TRUSTWORTHINESS item 1): the single most useful thing the answer-quality harness did not
    record before this — whether a wrong judged answer had its gold evidence in front of the
    model at all. Uses the SAME resolution ``_resolve_ranked_ids`` uses (candidates =
    ``index.resolve(item.content)``), so "was gold retrieved" and "what rank was gold retrieved
    at" are answered by literally the same join, never two joins that could quietly disagree.

    Thin wrapper over ``gold_context_attribution`` (kept as its own function, unchanged signature
    and return type, so every existing caller — ``answer_quality.py``'s per-row
    ``QueryResult.gold_in_context`` — is untouched) so the two never drift apart into two joins
    that could quietly disagree with each other.
    """
    return gold_context_attribution(items, index, gold).present


class _ProvenanceAccumulator:
    def __init__(self) -> None:
        self.scores: list[float] = []
        self.by_channel: dict[str, int] = {}
        self.floor_items = 0
        self.degraded = 0

    def observe(self, result: Any) -> None:
        if result.degraded is not None:
            self.degraded += 1
        for item in result.items:
            self.scores.append(float(item.fused_score))
            key = f"{item.tier}/{item.channel}"
            self.by_channel[key] = self.by_channel.get(key, 0) + 1
            if item.is_floor:
                self.floor_items += 1

    def finish(self) -> ScoreProvenance:
        scores = self.scores or [0.0]
        return ScoreProvenance(
            items=len(self.scores),
            zero_scores=sum(1 for s in self.scores if s == 0.0),
            distinct_scores=len(set(self.scores)),
            min_score=min(scores),
            median_score=statistics.median(scores),
            max_score=max(scores),
            by_channel=dict(sorted(self.by_channel.items())),
            floor_items=self.floor_items,
            degraded_results=self.degraded,
        )


async def _await_index(memory: Any, probe: str, *, user: str, session: str) -> bool:
    """Wait until the VECTOR tier can actually serve the corpus, before the first query runs.

    Qdrant applies upserts asynchronously (its real consistency model, not a masked bug), so a
    query issued the instant ingest returns can miss the tail of the corpus and score a miss that
    belongs to the store's write path, not to the ranker.

    The probe is narrowed to ``tier=MTM`` DELIBERATELY. A full-channel probe is useless here: the
    STM recency floor returns items unconditionally, so ``result.items`` is non-empty on the first
    poll no matter what Qdrant has indexed, and the wait silently becomes a no-op — the exact
    shape of "a check that cannot fail". Narrowing to the dense channel makes the poll depend on
    the thing it is waiting for. Same bounded-poll pattern as
    ``packages/mu-local/tests/test_local_roundtrip_int.py::_eventually``; a genuine empty still
    fails after the ceiling rather than hanging.
    """
    from mu_engine.storage.domain.memory import MemoryTier

    for _ in range(50):  # ~10 s ceiling
        result = await memory.recall(
            probe, user=user, session=session, limit=5, tier=MemoryTier.MTM
        )
        if result.items:
            return True
        await asyncio.sleep(0.2)
    return False


async def run_baseline(
    *,
    conversations: Sequence[Conversation],
    run_id: str,
    ks: Sequence[int] = (1, 3, 5, 10),
    recall_limit: int | None = None,
    importance: float = 0.9,
    max_queries_per_sample: int | None = None,
    settings: object | None = None,
    dataset_label: str = "locomo10",
    tier: str | None = None,
    consolidate: bool = False,
) -> RunReport:
    ks = tuple(sorted(ks))
    limit = recall_limit or max(ks)
    # A single-channel run is a DIAGNOSTIC, never the headline: `tier` narrows recall to one of
    # the three channels (`LocalMemory.recall(tier=...)` -> `_channels_for_tier`), which is how
    # the dense arm's own ranking quality is separated from the fused result's.
    tier_enum = None
    if tier is not None:
        from mu_engine.storage.domain.memory import MemoryTier

        tier_enum = MemoryTier(tier)
    user = "evaluser"
    session = "evalsession"

    rows: list[QueryScores] = []
    skipped_adversarial = 0
    skipped_no_evidence = 0
    skipped_no_gold = 0
    ingest_reports: list[IngestReport] = []
    provenance = _ProvenanceAccumulator()
    notes: list[str] = []
    corpus_turns = 0
    gold_in_context = 0
    gold_in_context_via_neighbor_only = 0
    neighbor_items_seen = 0

    for conversation in conversations:
        async with local_memory_for(conversation, run_id=run_id, settings=settings) as opaque:
            # `local_memory_for` yields `object` on purpose (it imports `LocalMemory` lazily so the
            # pure modules stay store-free); the verbs are exercised through a local `Any` rather
            # than by widening that contract.
            memory: Any = opaque
            index, report = await ingest_conversation(
                memory,
                conversation,
                user=user,
                session=session,
                importance=importance,
                consolidate=consolidate,
            )
            ingest_reports.append(report)
            corpus_turns += report.turns_written
            if report.consolidated:
                notes.append(
                    f"{conversation.sample_id}: consolidated "
                    f"facts_extracted={report.facts_extracted} added={report.ltm_added} "
                    f"superseded={report.ltm_superseded} noop={report.ltm_noop} "
                    f"({report.consolidate_seconds:.1f}s)"
                )
            if conversation.turns:
                visible = await _await_index(
                    memory, conversation.turns[0].text[:120], user=user, session=session
                )
                if not visible:
                    notes.append(f"{conversation.sample_id}: corpus never became recallable")

            queries: list[LabelledQuery] = list(conversation.queries)
            if max_queries_per_sample is not None:
                queries = queries[:max_queries_per_sample]

            known = {t.dia_id for t in conversation.turns}
            for query in queries:
                reason = classify_query_admission(query, known)
                if reason == "adversarial":
                    skipped_adversarial += 1
                    continue
                if reason == "no_evidence":
                    skipped_no_evidence += 1
                    continue
                if reason == "no_gold_in_corpus":
                    # Evidence naming a turn this harness never ingested (image-only turns carry
                    # no body). Not a retrieval failure — an unretrievable label. Counted, never
                    # scored as a zero.
                    skipped_no_gold += 1
                    continue
                gold = {e for e in query.evidence if e in known}
                result = await memory.recall(
                    query.question, user=user, session=session, limit=limit, tier=tier_enum
                )
                provenance.observe(result)
                attribution = gold_context_attribution(result.items, index, gold)
                if attribution.present:
                    gold_in_context += 1
                    if attribution.via_neighbor_only:
                        gold_in_context_via_neighbor_only += 1
                neighbor_items_seen += attribution.neighbor_items
                ranked = _resolve_ranked_ids(result.items, index, gold)
                rows.append(
                    score_query(
                        query_id=query.query_id,
                        category=query.category,
                        retrieved=ranked,
                        gold=sorted(gold),
                        ks=ks,
                    )
                )

    by_category: dict[str, dict[str, dict[int, float]]] = {}
    for category, name in CATEGORY_NAMES.items():
        subset = [r for r in rows if r.category == category]
        if subset:
            by_category[f"{category}:{name}"] = aggregate(subset, ks)

    return RunReport(
        run_id=run_id,
        dataset=dataset_label,
        samples=len(conversations),
        corpus_turns=corpus_turns,
        importance=importance,
        recall_limit=limit,
        ingest=ingest_reports,
        arms=[
            ArmReport(
                label=(
                    f"single channel: {tier}"
                    if tier
                    else "federated (STM floor + MTM dense + LTM graph)"
                ),
                queries_scored=len(rows),
                queries_skipped_adversarial=skipped_adversarial,
                queries_skipped_no_evidence=skipped_no_evidence,
                queries_skipped_no_gold_in_corpus=skipped_no_gold,
                ks=ks,
                overall=aggregate(rows, ks),
                by_category=by_category,
                provenance=provenance.finish(),
                gold_in_context=gold_in_context,
                gold_in_context_via_neighbor_only=gold_in_context_via_neighbor_only,
                neighbor_items_seen=neighbor_items_seen,
            )
        ],
        notes=notes,
    )
