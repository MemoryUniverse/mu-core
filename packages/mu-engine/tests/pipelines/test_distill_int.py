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
    ltm: FalkorLtmAdapter, *, now: datetime, mtm: QdrantMtmAdapter | None = None
) -> DistillPipeline:
    # MVP default: the deterministic heuristic extractor, no LLM (real-integration-testable now).
    # ``mtm`` wires the cross-store supersede so an MTM-resident loser also drops from MTM recall.
    return DistillPipeline(
        ltm=ltm, extractor=HeuristicSpoExtractor(), clock=_FixedClock(now), mtm=mtm
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
    loser_src = make_item(ns, "Ada works at Acme", valid_at=t_old, tier=MemoryTier.MTM)
    await mtm.upsert(loser_src)
    await _pipeline(ltm, now=t_old, mtm=mtm).distill(ns, [loser_src])
    assert await _mtm_point_state(qdrant_client, ns, loser_src.id, dim=dim) == "active"

    # (2) a later contradicting raw memory supersedes it (functional predicate, different object).
    winner_src = make_item(ns, "Ada works at Globex", valid_at=t_now, tier=MemoryTier.MTM)
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


async def test_distill_new_edge_self_expires_when_candidate_more_recent(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    # Graphiti new-edge self-expiry (edge_operations.py:622-639): a LATE-arriving OLDER fact loses.
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_new = datetime(2024, 1, 1, tzinfo=UTC)

    recent = make_item(
        ns, "Ada works at Globex", subject="Ada", predicate="works_at", obj="Globex", valid_at=t_new
    )
    await _pipeline(ltm, now=t_new).distill(ns, [recent])

    stale = make_item(
        ns, "Ada works at Acme", subject="Ada", predicate="works_at", obj="Acme", valid_at=t_old
    )
    report = await _pipeline(ltm, now=t_new).distill(ns, [stale])

    assert report.actions[0].kind is DistillActionKind.SELF_EXPIRE
    # the recent fact stays the live answer; the stale incoming fact is stored invalidated.
    recall_objs = {h.item.object for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert recall_objs == {"Globex"}


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
