"""DISTILL MTM->LTM — REAL mu-dev-falkordb, ZERO mocks (DEV-STANDARDS non-negotiable).

The acceptance test for this slice: a **heuristic** (no-LLM) DISTILL run writes bi-temporal facts
into the live LTM graph, and a contradicting later fact **supersedes-by-invalidation** — the loser
drops from present-tense recall / ``facts_at(now)`` yet SURVIVES in ``facts_at(t_old)`` (Graphiti
invalidate-don't-delete, edge_operations.py:406-441). Every write goes through the real
``GraphStorePort`` (FalkorLtmAdapter) — no store client, no mock, no fake.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient

from mu_engine.pipelines.distill import DistillActionKind, DistillPipeline
from mu_engine.services.extract import HeuristicSpoExtractor
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryTier, Polarity
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id

pytestmark = pytest.mark.integration


class _FixedClock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at


def _pipeline(
    ltm: FalkorLtmAdapter,
    *,
    now: datetime,
    mtm: QdrantMtmAdapter | None = None,
    stm: RedisStmAdapter | None = None,
) -> DistillPipeline:
    # MVP default: the deterministic heuristic extractor, no LLM (real-integration-testable now).
    # ``mtm`` wires the cross-store supersede so an MTM-resident loser also drops from MTM recall.
    # ``stm`` wires the THIRD arm (§7.2 step 5, Redis bullet) so the loser also leaves the recency
    # window — without it the STM floor re-surfaces the superseded fact as a top recall hit.
    return DistillPipeline(
        ltm=ltm, extractor=HeuristicSpoExtractor(), clock=_FixedClock(now), mtm=mtm, stm=stm
    )


async def _mtm_point_state(
    client: AsyncQdrantClient, ns: Namespace, memory_id: str, *, dim: int
) -> str | None:
    """The ``state`` payload of the MTM point that BACKS ``memory_id`` (keyed by the ingest id)."""
    points = await client.retrieve(
        collection_name=collection_name(ns, dim),
        ids=[point_id(memory_id)],
        with_payload=True,
    )
    if not points:
        return None
    payload = points[0].payload or {}
    return payload.get("state")


async def test_heuristic_distill_writes_bitemporal_facts(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t1 = datetime(2020, 1, 1, tzinfo=UTC)
    window = [
        make_item(
            ns, "Ada lives in Paris", subject="Ada", predicate="lives_in", obj="Paris", valid_at=t1
        ),
        make_item(
            ns, "Ada uses Postgres", subject="Ada", predicate="uses", obj="Postgres", valid_at=t1
        ),
    ]

    report = await _pipeline(ltm, now=t1).distill(ns, window)

    assert report.facts_extracted == 2
    assert report.added == 2
    # the facts are now live in the LTM graph with their bi-temporal valid_at stamped.
    hits = {
        h.item.subject + "/" + (h.item.predicate or "")
        for h in await ltm.graph_recall(ns, subject="Ada", limit=10)
    }
    assert hits == {"Ada/lives_in", "Ada/uses"}
    at_facts = await ltm.facts_at(ns, t1, subject="Ada")
    assert {f.object for f in at_facts} == {"Paris", "Postgres"}
    assert all(f.valid_at == t1 for f in at_facts)


async def test_distill_supersede_is_invalidate_dont_delete(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    loser = make_item(
        ns, "Ada works at Acme", subject="Ada", predicate="works_at", obj="Acme", valid_at=t_old
    )
    await _pipeline(ltm, now=t_old).distill(ns, [loser])

    winner = make_item(
        ns, "Ada works at Globex", subject="Ada", predicate="works_at", obj="Globex", valid_at=t_now
    )
    report = await _pipeline(ltm, now=t_now).distill(ns, [winner])

    # the functional-predicate conflict superseded the loser (one loser invalidated).
    assert report.superseded == 1
    action = report.actions[0]
    assert action.kind is DistillActionKind.SUPERSEDE
    assert action.loser_ids == (loser.id,)

    # present-tense recall drops the superseded loser, keeps the winner.
    recall_objs = {h.item.object for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert recall_objs == {"Globex"}

    # facts_at(now) excludes the loser ...
    now_objs = {f.object for f in await ltm.facts_at(ns, t_now, subject="Ada")}
    assert now_objs == {"Globex"}
    # ... but the loser SURVIVES in history at t_old (invalidate-don't-delete).
    hist = await ltm.facts_at(ns, t_old, subject="Ada")
    assert {f.object for f in hist} == {"Acme"}
    assert hist[0].id == loser.id


async def test_functional_supersession_invalidates_loser_across_ltm_and_mtm(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """BLOCKER #1/#2 regression — the fact-node-id vs ingest-id linkage in the cross-store MTM
    invalidate. A functional supersession must invalidate the loser across BOTH LTM (falkordb,
    invalidate-don't-delete) AND MTM (qdrant, state->superseded) WITHOUT crashing.

    The loser is an UNSTRUCTURED, MTM-resident source memory: it lives in Qdrant keyed by its
    INGEST id, and its LTM fact is EXTRACTED with a fresh node id whose ``metadata['derived_from']``
    points back at that ingest id. Passing the fact-node id to the MTM invalidate 404s (the bug);
    the fix resolves the ingest id so the RIGHT Qdrant point flips.
    """
    dim = mtm._dim  # the real adapter's collection dim, for the point-state read
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    # (1) loser: an MTM-resident raw memory, distilled into an LTM fact (derived_from == its id).
    # `created_at=t_old` (Rank 5 same-batch chronology fix): this is UNSTRUCTURED content, so the
    # extracted fact's `valid_at` now anchors on `loser_src.created_at` (real source-message
    # timestamp), not the raw item's own `valid_at` field (never read by `_fact_to_item`) nor the
    # pipeline clock — pin it explicitly so `facts_at(ns, t_old, ...)` below still finds it.
    loser_src = make_item(
        ns, "Ada works at Acme", valid_at=t_old, tier=MemoryTier.MTM, created_at=t_old
    )
    await mtm.upsert(loser_src)
    await _pipeline(ltm, now=t_old, mtm=mtm).distill(ns, [loser_src])
    assert await _mtm_point_state(qdrant_client, ns, loser_src.id, dim=dim) == "active"

    # (2) a later contradicting raw memory supersedes it (functional predicate, different object).
    winner_src = make_item(
        ns, "Ada works at Globex", valid_at=t_now, tier=MemoryTier.MTM, created_at=t_now
    )
    await mtm.upsert(winner_src)
    report = await _pipeline(ltm, now=t_now, mtm=mtm).distill(ns, [winner_src])

    # supersession fired and did NOT crash on the cross-store MTM invalidate (the id-linkage bug).
    assert report.superseded == 1
    assert report.actions[0].kind is DistillActionKind.SUPERSEDE

    # (3) LTM: the loser is gone from active recall / facts_at(now), kept in history at t_old.
    recall_objs = {h.item.object for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert recall_objs == {"Globex"}
    assert {f.object for f in await ltm.facts_at(ns, t_now, subject="Ada")} == {"Globex"}
    assert {f.object for f in await ltm.facts_at(ns, t_old, subject="Ada")} == {"Acme"}

    # (4) MTM: the loser's SOURCE point (keyed by the INGEST id, resolved from derived_from) is now
    #     state='superseded' so the active-only recall filter drops it; the winner stays active.
    assert await _mtm_point_state(qdrant_client, ns, loser_src.id, dim=dim) == "superseded"
    assert await _mtm_point_state(qdrant_client, ns, winner_src.id, dim=dim) == "active"


async def test_distill_assertion_recency_decides_winner_not_valid_at(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """BUG1 FIX regression (data-quality re-assessment §2 "SUPERSESSION WINNER INVERTED",
    2026-07-31): supersedes the OLD ``test_distill_new_edge_self_expires_when_candidate_more_
    recent`` test, which asserted the OPPOSITE (pre-fix) semantics — that a LATE-ASSERTED fact
    loses to an earlier-asserted one whenever its bi-temporal ``valid_at`` reads "older". That was
    exactly the defect the real-path re-assessment caught: the real 0.5B-SLM extractor garbles/
    mis-parses ``valid_at`` (an event date parsed from free text) often enough that trusting it to
    pick the supersession winner regularly inverts the outcome — the CURRENT fact ends up
    ``superseded`` and the STALE one stays ``active``. The fix (``distill.py::_asserted_later``):
    the fact whose SOURCE STM message was captured (``created_at``) LATER always wins, REGARDLESS
    of what its ``valid_at`` says. Reproduced here with ``valid_at`` deliberately reversed
    relative to assertion order — the first-asserted fact carries the LATER world-time ``valid_at``
    (2030); the second-asserted fact carries the EARLIER one (2000) — proving the decision no
    longer keys on ``valid_at`` at all.
    """
    ns = make_ns()
    t_first_assert = datetime(2019, 1, 1, tzinfo=UTC)  # asserted (created) FIRST
    t_second_assert = datetime(2024, 1, 1, tzinfo=UTC)  # asserted (created) SECOND

    first = make_item(
        ns,
        "Ada works at Globex",
        subject="Ada",
        predicate="works_at",
        obj="Globex",
        valid_at=datetime(2030, 1, 1, tzinfo=UTC),  # world-time LATER — deliberately reversed
        created_at=t_first_assert,
    )
    await _pipeline(ltm, now=t_first_assert).distill(ns, [first])

    second = make_item(
        ns,
        "Ada works at Acme",
        subject="Ada",
        predicate="works_at",
        obj="Acme",
        valid_at=datetime(2000, 1, 1, tzinfo=UTC),  # world-time EARLIER, but asserted LATER
        created_at=t_second_assert,
    )
    report = await _pipeline(ltm, now=t_second_assert).distill(ns, [second])

    # the SECOND-asserted fact wins (assertion recency) even though its own valid_at is earlier.
    assert report.actions[0].kind is DistillActionKind.SUPERSEDE
    assert report.actions[0].loser_ids == (first.id,)
    recall_objs = {h.item.object for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert recall_objs == {"Acme"}


async def test_distill_noop_reinforces_identical_fact(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t1 = datetime(2020, 1, 1, tzinfo=UTC)
    item = make_item(
        ns, "Ada lives in Paris", subject="Ada", predicate="lives_in", obj="Paris", valid_at=t1
    )
    await _pipeline(ltm, now=t1).distill(ns, [item])

    # a DISTINCT capture (new id) asserting the identical triple -> NOOP (reinforce, not dup).
    again = make_item(
        ns, "Ada lives in Paris", subject="Ada", predicate="lives_in", obj="Paris", valid_at=t1
    )
    assert again.id != item.id
    report = await _pipeline(ltm, now=t1).distill(ns, [again])

    assert report.actions[0].kind is DistillActionKind.NOOP
    # still exactly one active fact (the original, reinforced) — the duplicate did not land.
    hits = await ltm.graph_recall(ns, subject="Ada", limit=10)
    assert len(hits) == 1
    assert hits[0].item.id == item.id
    assert hits[0].item.access_count == 1  # reinforced once


async def test_distill_non_functional_predicate_coexists(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t1 = datetime(2020, 1, 1, tzinfo=UTC)
    await _pipeline(ltm, now=t1).distill(
        ns,
        [make_item(ns, "Ada likes tea", subject="Ada", predicate="likes", obj="tea", valid_at=t1)],
    )
    report = await _pipeline(ltm, now=t1).distill(
        ns,
        [
            make_item(
                ns, "Ada likes coffee", subject="Ada", predicate="likes", obj="coffee", valid_at=t1
            )
        ],
    )
    # "likes" is not single-cardinality — both preferences stay active (COEXIST, not supersede).
    assert report.actions[0].kind is DistillActionKind.COEXIST
    objs = {h.item.object for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert objs == {"tea", "coffee"}


async def test_distill_extracts_spo_from_raw_text(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    # An UNSTRUCTURED MTM item (no subject/predicate/object) is run through the heuristic extractor.
    ns = make_ns()
    t1 = datetime(2020, 1, 1, tzinfo=UTC)
    raw = make_item(ns, "Ada uses Postgres. Ada works at Acme.", valid_at=t1)

    report = await _pipeline(ltm, now=t1).distill(ns, [raw])

    assert report.facts_extracted == 2
    triples = {
        (h.item.subject, h.item.predicate, h.item.object)
        for h in await ltm.graph_recall(ns, subject="Ada", limit=10)
    }
    assert triples == {("Ada", "uses", "Postgres"), ("Ada", "works_at", "Acme")}


async def test_distill_change_verb_sentence_now_reaches_ltm(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """D5-quick unblocker, end-to-end: an UNSTRUCTURED change-of-value sentence that used to
    extract to ZERO facts (PIPELINE-TRACE-DIAGNOSIS.md §1, `ada_flight_v2`) now reaches the real
    LTM graph through the full DISTILL pass (extractor -> reconcile -> resolve -> write)."""
    ns = make_ns()
    t1 = datetime(2024, 1, 1, tzinfo=UTC)
    raw = make_item(ns, "Ada moved her flight to the Denver offsite to Thursday.", valid_at=t1)

    report = await _pipeline(ltm, now=t1).distill(ns, [raw])

    assert report.facts_extracted == 1  # BEFORE the D5-quick fix this was 0.
    assert report.added == 1
    hits = await ltm.graph_recall(ns, subject="Ada", limit=10)
    assert {(h.item.predicate, h.item.object) for h in hits} == {("flight", "Thursday")}


async def test_distill_change_verb_v2_supersedes_v1_gold_flight_pair(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The exact `ada_flight_v1`/`ada_flight_v2` gold pair (`demos/pipeline_trace/gold_items.py`)
    run end-to-end, raw-text in, through the extractor AND the reconcile/resolve stages: v1 and
    v2 now COLLIDE on (subject, predicate) (D-7 canonicalization) and the newly-canonical
    ``flight`` predicate is in ``functional_predicates`` (single-cardinality), so v2 genuinely
    SUPERSEDES v1 in the live graph — the exact "current fact absent" defect
    (DATA-QUALITY-ASSESSMENT.md §3.2) closed."""
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    # `created_at=t_old`/`t_now` (Rank 5 same-batch chronology fix): UNSTRUCTURED content, so
    # each extracted fact's `valid_at` now anchors on the source message's OWN `created_at`
    # (real STM timestamp), not the pipeline clock — pin it explicitly so `facts_at(ns, t_old,
    # ...)` below still finds v1's fact exactly at `t_old`.
    v1 = make_item(
        ns, "Ada's flight to the Denver offsite is on Tuesday.", valid_at=t_old, created_at=t_old
    )
    report_v1 = await _pipeline(ltm, now=t_old).distill(ns, [v1])
    assert report_v1.facts_extracted == 1
    assert report_v1.added == 1

    v2 = make_item(
        ns,
        "Ada moved her flight to the Denver offsite to Thursday.",
        valid_at=t_now,
        created_at=t_now,
    )
    report_v2 = await _pipeline(ltm, now=t_now).distill(ns, [v2])

    assert report_v2.facts_extracted == 1
    assert report_v2.superseded == 1  # v2 genuinely superseded v1, not a silent-drop or a dup.
    assert report_v2.actions[0].kind is DistillActionKind.SUPERSEDE

    # present-tense recall now holds Thursday (the "current fact absent" bug is fixed) ...
    now_recall = {h.item.object for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert now_recall == {"Thursday"}
    # ... and Tuesday survives in HISTORY (invalidate-don't-delete, never a hard drop).
    hist = await ltm.facts_at(ns, t_old, subject="Ada")
    assert {f.object for f in hist} == {"Tuesday"}


async def test_distill_negation_supersedes_opposite_polarity(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    # A same-triple opposite-polarity assertion is a direct contradiction (PolarityCardinality).
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)
    await _pipeline(ltm, now=t_old).distill(
        ns,
        [
            make_item(
                ns,
                "Ada likes tea",
                subject="Ada",
                predicate="likes",
                obj="tea",
                valid_at=t_old,
                polarity=Polarity.POSITIVE,
            )
        ],
    )
    report = await _pipeline(ltm, now=t_now).distill(
        ns,
        [
            make_item(
                ns,
                "Ada does not like tea",
                subject="Ada",
                predicate="likes",
                obj="tea",
                valid_at=t_now,
                polarity=Polarity.NEGATIVE,
            )
        ],
    )
    assert report.actions[0].kind is DistillActionKind.SUPERSEDE
    live = await ltm.graph_recall(ns, subject="Ada", limit=10)
    assert len(live) == 1
    assert live[0].item.polarity is Polarity.NEGATIVE
    # the positive assertion survives historically (never deleted, just invalidated).
    hist = await ltm.facts_at(ns, t_old, subject="Ada")
    assert any(f.polarity is Polarity.POSITIVE for f in hist)


async def test_find_conflicts_collides_across_subject_casing(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """DEFECT-1 regression (real-path verify gate, live-reproduced on the REAL SLM path): the SAME
    entity extracted with a DIFFERENT surface casing across two turns ("ada"/tuesday vs
    "Ada"/Thursday, the gold flight pair) must still collide on `(subject, predicate)` — an
    exact-string `m.subject = $subject` silently drops the collision and the v2 fact lands as a
    brand-new (subject, predicate) pair (ADD) instead of superseding v1, the exact defect."""
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)
    v1 = make_item(
        ns, "ada flight tuesday", subject="ada", predicate="flight", obj="tuesday", valid_at=t_old
    )
    v2 = make_item(
        ns, "Ada flight Thursday", subject="Ada", predicate="flight", obj="Thursday", valid_at=t_now
    )
    await _pipeline(ltm, now=t_old).distill(ns, [v1])
    report = await _pipeline(ltm, now=t_now).distill(ns, [v2])

    # v2 genuinely SUPERSEDES v1 despite the casing mismatch — not a silent ADD-as-new-pair.
    assert report.superseded == 1
    assert report.actions[0].kind is DistillActionKind.SUPERSEDE

    live = {h.item.object for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert live == {"Thursday"}
    hist = {f.object for f in await ltm.facts_at(ns, t_old, subject="ada")}
    assert hist == {"tuesday"}  # the lower-cased v1 survives historically, never dropped.


async def test_distill_noop_reinforce_of_stale_message_does_not_undo_same_batch_supersede(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """DEFECT-2 regression (real-path verify gate, live-reproduced twice on the real
    ``LocalMemory.consolidate()`` path — deterministic + real-SLM): ``stm.recent()`` naturally
    returns BOTH the old raw message and its new contradicting replacement in the SAME window once
    the old message hasn't expired from STM yet. Reproduced here directly against
    ``DistillPipeline`` (no STM/LocalMemory dependency needed — the defect lives entirely in
    ``_resolve``'s NOOP branch): tick 1 consolidates ONLY the old raw text (writes it active); tick
    2 consolidates a window containing the NEW contradicting raw text FIRST and the (still
    un-expired) OLD raw text AGAIN SECOND — the exact shape/order that exposed the bug: the new
    message's SUPERSEDE action writes first, then the old message's own NOOP-reinforce action (a
    RECONCILE-time snapshot fixed before either write in this batch happened) used to blindly
    ``upsert_fact`` that stale snapshot back, resetting ``state``/``invalid_at`` to active/'' and
    silently undoing the just-applied supersession — both versions ending up active (the exact
    semantic-shadowing defect). After the fix, exactly one version is active regardless of order."""
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)
    old_text = "Ada works at Acme"
    new_text = "Ada works at Globex"

    await _pipeline(ltm, now=t_old).distill(ns, [make_item(ns, old_text, valid_at=t_old)])

    # tick 2's window: NEW first, stale OLD (re-extracted -> a FRESH LTM node id, unlike a
    # structured item) second — the order that exposed the pre-fix resurrection.
    report = await _pipeline(ltm, now=t_now).distill(
        ns,
        [
            make_item(ns, new_text, valid_at=t_now),
            make_item(ns, old_text, valid_at=t_old),
        ],
    )

    assert DistillActionKind.SUPERSEDE in {a.kind for a in report.actions}
    assert DistillActionKind.NOOP in {a.kind for a in report.actions}

    # present-tense: EXACTLY one version active — the resurrection defect is fixed.
    live = await ltm.graph_recall(ns, subject="Ada", limit=10)
    assert {h.item.object for h in live} == {"Globex"}
    assert len(live) == 1


# --------------------------------------------------------------------------------------------
# Rank 5 (data-quality) SAME-BATCH CHRONOLOGY: root-cause was `_collect_facts` stamping ONE
# shared window `now` as every extracted fact's `valid_at` fallback — two same-(subject,
# predicate) facts from two DIFFERENT source messages landing in the SAME `distill()` window
# collided on an IDENTICAL `valid_at`, a tie `_heuristic_only_verdict`'s bare `>` comparison
# silently broke toward whichever fact happened to be resolved LAST (window order), not
# whichever was genuinely more recent. Fixed: each fact now anchors on ITS OWN source
# message's real `item.created_at` (mirrors the real STM `add()` timestamp — the same value
# STM's own recency ZSET already scores by, `redis_stm.py::_put_impl`). The three pairs below
# are the exact D5-quick gold pairs (`test_extract_change_verbs_unit.py`), each genuinely
# single-cardinality (all three predicates are in `DistillSettings.functional_predicates`).
# --------------------------------------------------------------------------------------------

_GOLD_PAIRS = (
    # (subject, predicate, v1_text, v1_object, v2_text, v2_object)
    # NOTE: "Ada" is the subject of BOTH the flight and hotel pairs (deliberately, matching the
    # real gold set) — `predicate` is carried alongside so assertions below can filter on
    # (subject, predicate) and never conflate the two same-subject facts with each other.
    (
        "Ada",
        "flight",
        "Ada's flight to the Denver offsite is on Tuesday.",
        "Tuesday",
        "Ada moved her flight to the Denver offsite to Thursday.",
        "Thursday",
    ),
    (
        "The Q3 planning meeting",
        "is",
        "The Q3 planning meeting is in Room A.",
        "Room A",
        "The Q3 planning meeting was moved to Room B.",
        "Room B",
    ),
    (
        "Ada",
        "hotel",
        "Ada's hotel in Tokyo is the Park Hyatt.",
        "Park Hyatt",
        "Ada's hotel booking changed to the Aman Tokyo.",
        "Aman Tokyo",
    ),
)


async def test_distill_same_batch_newer_wins_all_three_gold_pairs_recency_first_order(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """SAME-BATCH, recency-first window order (the real ``stm.recent()`` shape, ``zrevrange`` ->
    newest member first): a SINGLE ``distill()`` call sees BOTH v1 and v2 of all three gold
    pairs at once, v2 (the genuinely newer source message, later ``created_at``) listed FIRST in
    the window. Before the fix this order made v2 resolve FIRST (a clean ADD, no live candidate
    yet) then v1 resolve SECOND against v2's now-active node with a TIED ``valid_at`` — the tie
    defaulted to SUPERSEDE in v1's (the currently-resolving incoming fact's) favour, i.e.
    BACKWARDS. After the fix v1's later resolve correctly SELF_EXPIREs against the genuinely
    more-recent v2 candidate instead."""
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    window = []
    for _subject, _predicate, v1_text, _v1_obj, v2_text, _v2_obj in _GOLD_PAIRS:
        window.append(make_item(ns, v2_text, created_at=t_now))  # newer FIRST (recency-first)
        window.append(make_item(ns, v1_text, created_at=t_old))  # older SECOND

    report = await _pipeline(ltm, now=t_now).distill(ns, window)

    assert report.facts_extracted == 6
    assert report.superseded == 3

    for subject, predicate, _v1_text, v1_obj, _v2_text, v2_obj in _GOLD_PAIRS:
        live = {
            h.item.object
            for h in await ltm.graph_recall(ns, subject=subject, predicate=predicate, limit=10)
        }
        assert live == {v2_obj}, f"{subject}/{predicate}: expected active={v2_obj!r}, got {live!r}"
        hist = {
            f.object
            for f in await ltm.facts_at(ns, t_old, subject=subject)
            if f.predicate == predicate
        }
        assert hist == {v1_obj}, f"{subject}/{predicate}: expected history={v1_obj!r}, got {hist!r}"


async def test_distill_same_batch_newer_wins_all_three_gold_pairs_reverse_order(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """Same SAME-BATCH setup, window order REVERSED (v1 listed first, v2 second) — proves the
    fix is direction-independent (the task's "regardless of resolve order" requirement), not an
    accidental side effect of one particular iteration order. Now v1 resolves first (clean ADD)
    and v2 resolves second, correctly SUPERSEDING v1 (v2's own `valid_at` is genuinely later)."""
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    window = []
    for _subject, _predicate, v1_text, _v1_obj, v2_text, _v2_obj in _GOLD_PAIRS:
        window.append(make_item(ns, v1_text, created_at=t_old))  # older FIRST (reversed order)
        window.append(make_item(ns, v2_text, created_at=t_now))  # newer SECOND

    report = await _pipeline(ltm, now=t_now).distill(ns, window)

    assert report.facts_extracted == 6
    assert report.superseded == 3

    for subject, predicate, _v1_text, v1_obj, _v2_text, v2_obj in _GOLD_PAIRS:
        live = {
            h.item.object
            for h in await ltm.graph_recall(ns, subject=subject, predicate=predicate, limit=10)
        }
        assert live == {v2_obj}, f"{subject}/{predicate}: expected active={v2_obj!r}, got {live!r}"
        hist = {
            f.object
            for f in await ltm.facts_at(ns, t_old, subject=subject)
            if f.predicate == predicate
        }
        assert hist == {v1_obj}, f"{subject}/{predicate}: expected history={v1_obj!r}, got {hist!r}"


async def test_supersession_evicts_loser_from_the_stm_recency_window(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    stm: RedisStmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """REGRESSION (third arm of memory-layer-design.md §7.2 step 5, the Redis/STM bullet).

    Live-reproduced before the fix: a functional supersession flipped the loser in LTM
    (``state=superseded`` + ``SUPERSEDED_BY``) AND in MTM (``state=superseded``), but left the raw
    source message sitting in the STM recency window. Because the STM floor force-includes the
    most-recent window entries regardless of relevance, recall then returned the SUPERSEDED fact
    as its TOP hit while the graph correctly knew it was dead — breaking the design's "absent from
    EVERY hot read" half of invalidate-don't-delete.

    The loser must leave the STM window; it must REMAIN in LTM history (never deleted).
    """
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    loser_src = make_item(
        ns, "Ada works at Acme", valid_at=t_old, tier=MemoryTier.STM, created_at=t_old
    )
    await stm.put(loser_src)
    await mtm.upsert(loser_src)
    await _pipeline(ltm, now=t_old, mtm=mtm, stm=stm).distill(ns, [loser_src])
    assert await stm.get(ns, loser_src.id) is not None, "precondition: loser is STM-resident"

    winner_src = make_item(
        ns, "Ada works at Globex", valid_at=t_now, tier=MemoryTier.STM, created_at=t_now
    )
    await stm.put(winner_src)
    await mtm.upsert(winner_src)
    report = await _pipeline(ltm, now=t_now, mtm=mtm, stm=stm).distill(ns, [winner_src])

    assert report.superseded == 1
    assert report.actions[0].kind is DistillActionKind.SUPERSEDE

    # THE FIX: the superseded loser is out of the hot STM window...
    assert (
        await stm.get(ns, loser_src.id) is None
    ), "superseded loser still resident in the STM recency window — the floor will re-surface it"
    recent_ids = {scored.item.id for scored in await stm.recent(ns, limit=50)}
    assert loser_src.id not in recent_ids
    assert winner_src.id in recent_ids, "the winner must stay in the window"

    # ...while history is fully retained in LTM (invalidate-don't-delete, never a destructive fix).
    assert {f.object for f in await ltm.facts_at(ns, t_old, subject="Ada")} == {"Acme"}
    assert {f.object for f in await ltm.facts_at(ns, t_now, subject="Ada")} == {"Globex"}


async def test_self_expire_also_invalidates_across_mtm_and_stm(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    stm: RedisStmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """REGRESSION (SELF_EXPIRE symmetry). SELF_EXPIRE is a supersession whose loser is the INCOMING
    fact. The pre-fix branch wrote ONLY the graph arm — unlike the SUPERSEDE branch, which always
    wrote all three — so a self-expiring fact stayed ``state=active`` on its MTM point and live in
    the STM window, and kept being fused back into recall. Both arms must now fire here too.
    """
    dim = mtm._dim
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    # The AUTHORITATIVE fact is asserted FIRST and is the more recently ASSERTED one...
    newer = make_item(
        ns, "Ada works at Globex", valid_at=t_now, tier=MemoryTier.STM, created_at=t_now
    )
    await stm.put(newer)
    await mtm.upsert(newer)
    await _pipeline(ltm, now=t_now, mtm=mtm, stm=stm).distill(ns, [newer])

    # ...so this LATER-distilled but EARLIER-asserted fact must SELF-EXPIRE.
    older = make_item(
        ns, "Ada works at Acme", valid_at=t_old, tier=MemoryTier.STM, created_at=t_old
    )
    await stm.put(older)
    await mtm.upsert(older)
    report = await _pipeline(ltm, now=t_now, mtm=mtm, stm=stm).distill(ns, [older])

    assert report.actions[0].kind is DistillActionKind.SELF_EXPIRE

    # present-tense truth is the authoritative fact only
    assert {f.object for f in await ltm.facts_at(ns, t_now, subject="Ada")} == {"Globex"}
    # THE FIX: the self-expired incoming fact drops out of MTM and the STM window too.
    assert await _mtm_point_state(qdrant_client, ns, older.id, dim=dim) == "superseded"
    assert (
        await stm.get(ns, older.id) is None
    ), "self-expired fact still resident in the STM recency window"
