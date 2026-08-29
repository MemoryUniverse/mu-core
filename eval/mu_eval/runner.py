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

__all__ = ["ArmReport", "RunReport", "ScoreProvenance", "run_baseline"]


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
    queries_skipped_no_gold_in_corpus: int
    ks: tuple[int, ...]
    overall: dict[str, dict[int, float]]
    by_category: dict[str, dict[str, dict[int, float]]]
    provenance: ScoreProvenance | None = None


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
    skipped_no_gold = 0
    ingest_reports: list[IngestReport] = []
    provenance = _ProvenanceAccumulator()
    notes: list[str] = []
    corpus_turns = 0

    for conversation in conversations:
        async with local_memory_for(conversation, run_id=run_id, settings=settings) as opaque:
            # `local_memory_for` yields `object` on purpose (it imports `LocalMemory` lazily so the
            # pure modules stay store-free); the verbs are exercised through a local `Any` rather
            # than by widening that contract.
            memory: Any = opaque
            index, report = await ingest_conversation(
                memory, conversation, user=user, session=session, importance=importance
            )
            ingest_reports.append(report)
            corpus_turns += report.turns_written
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
                if query.is_adversarial or not query.evidence:
                    skipped_adversarial += 1
                    continue
                gold = {e for e in query.evidence if e in known}
                if not gold:
                    # Evidence naming a turn this harness never ingested (image-only turns carry
                    # no body). Not a retrieval failure — an unretrievable label. Counted, never
                    # scored as a zero.
                    skipped_no_gold += 1
                    continue
                result = await memory.recall(
                    query.question, user=user, session=session, limit=limit, tier=tier_enum
                )
                provenance.observe(result)
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
                queries_skipped_no_gold_in_corpus=skipped_no_gold,
                ks=ks,
                overall=aggregate(rows, ks),
                by_category=by_category,
                provenance=provenance.finish(),
            )
        ],
        notes=notes,
    )
