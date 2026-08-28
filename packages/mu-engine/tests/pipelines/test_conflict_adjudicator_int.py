"""S3-01 acceptance — ``ConflictAdjudicator`` (LLM-judged supersession) + the MTM cross-store
degrade wrap. REAL mu-dev-falkordb (LTM) + mu-dev-qdrant (MTM) + the REAL local Ollama SLM
(``mu-dev-slm``, ``qwen2.5:0.5b``, port 11435) as the ``adjudicate_model`` — ZERO mocks
(DEV-STANDARDS non-negotiable).

Extends (does not reinvent) ``test_distill_llm_slm_int.py``'s proven real-SLM wiring shape —
``SlmTestSettings``/``_build_slm_catalog``/``_slm_reachable`` are imported from that module
verbatim, only the ``ModelSettings.adjudicate_model`` field matters here (``Task
.CONFLICT_ADJUDICATION`` routes to it — S0-05, ``tests/providers/test_conflict_adjudication_task
.py``). The DEV-run note authorizing this ("the SAME docker SLM shape ... this is the
adjudicate_model for LLM-judged supersession") is honored: ``models.adjudicate_model` is set to
the local SLM's model-group for this test file ONLY, same test-scoped override pattern
``test_distill_llm_slm_int.py`` already uses for ``answer_model``/``hard_extract_model``/etc. —
it never touches a production default (canonical prod default excludes the local SLM from
``Task.CONFLICT_ADJUDICATION``, ADR 0037 — that HARD-tier-only exclusion is a PRODUCTION catalog
policy, not a constraint on what a test may point the ONE test model-group at).

Acceptance items closed here (memory-lifecycle-manager-spec.md §8/§13.2, AC-3.1/AC-3.2, §14):
  1. reconcile/resolve stage split — `_reconcile` alone never calls the model, never writes.
  2. AC-3.1 — the same delta-set replayed under two `FrozenClock` instants (same total order)
     produces the identical winner both times.
  3. A genuine semantic contradiction (non-functional predicate, different objects) that
     `PolarityCardinalityHeuristic` alone would call COEXIST is correctly adjudicated SUPERSEDE
     by the REAL SLM path.
  4. AC-3.2 — `adjudication_budget_per_sweep=2` presented with 3 contradicting candidates for ONE
     winner: the LLM adjudicates the first 2, degrades the 3rd, emitting
     `DegradedModeEntered(component="conflict", reason=CONFLICT_ADJUDICATION_DEGRADED)` for the
     3rd ONLY.
  5. LLM-down (a real, unreachable local endpoint) -> the SAME `CONFLICT_ADJUDICATION_DEGRADED`
     reason + the deterministic heuristic decides.
  6. A `valid_at` TIE even the heuristic can't decide -> `ConflictRecord(MANUAL_PENDING)` parked
     via a real `InMemoryConflictRecordRepository`; NEITHER side is fabricated as a winner.
  7. `MTM_INVALIDATE_POINT_ABSENT` — a real Qdrant 404 (write-after-read visibility lag) degrades
     named instead of an uncaught error; the LTM side is already SUPERSEDED regardless; a bounded
     retry on the next `distill()` tick for the namespace flips the MTM point once visible.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from mu_contracts.domain.events import DegradedModeEntered, DegradeReason, DomainEvent
from mu_engine.lifecycle.conflict import (
    ConflictAdjudicator,
    ConflictAdjudicatorSettings,
    ConflictResolutionPolicy,
    InMemoryConflictRecordRepository,
    ResolutionMode,
)
from mu_engine.pipelines.distill import DistillActionKind, DistillPipeline
from mu_engine.platform.clock import FrozenClock
from mu_engine.providers.catalog import ModelDeployment, ModelKind, ProviderKind, ProviderRecord
from mu_engine.providers.model_router import ModelRouter, build_model_router
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings, default_local_catalog
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id

# RELATIVE, on purpose. This line used to read
# ``from tests.pipelines.test_distill_llm_slm_int import ...`` and it broke the ENTIRE suite
# whenever the repo root was on ``sys.path`` (e.g. ``python -m pytest``): pytest then names
# every module ``packages.mu-engine.tests...``, ``tests`` is not a top-level module, and
# collection is Interrupted with ZERO tests run. See ``_slm_support`` for the measurement.
from ._slm_support import _SLM_CFG, _SLM_UP, _build_slm_catalog

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _SLM_UP,
        reason=(
            f"mu-dev-slm unreachable at {_SLM_CFG.probe_url} (env probe) — "
            "start it: docker compose -f docker-compose.slm.yml up -d"
        ),
    ),
]


class _RecordingBus:
    """A REAL, minimal ``ConflictEventPublisher``/``EventPublisher`` collaborator — collects every
    published event for in-test assertions (the same "legitimate in-process instrumentation, not
    a mock of the system under test" idiom as ``InProcessWriterLease``/``InMemoryStageLedger``: the
    events it collects are genuinely produced by real ``ConflictAdjudicator``/``DistillPipeline``
    code paths, nothing about the production logic is faked)."""

    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.events.append(event)

    def degrades(self, *, component: str, reason: DegradeReason) -> list[DegradedModeEntered]:
        return [
            e
            for e in self.events
            if isinstance(e, DegradedModeEntered)
            and e.component == component
            and e.reason is reason
        ]


def _router(models: ModelSettings, catalog: ModelCatalogSettings) -> ModelRouter:
    return build_model_router(models=models, catalog=catalog)


def _unreachable_slm_catalog() -> tuple[ModelSettings, ModelCatalogSettings]:
    """A REAL ``ModelRouter`` wired at a real but genuinely unreachable local port (nothing
    listens on it) — a real network failure, not a faked/mocked provider (mirrors how
    ``_build_slm_catalog`` wires the REAL reachable one; only the port differs)."""
    provider = ProviderRecord(
        key="slm_test_unreachable",
        kind=ProviderKind.LOCAL_HTTP,
        litellm_provider="openai",
        api_base="http://127.0.0.1:19999/v1",  # nothing listens here
        is_local=True,
    )
    deployment = ModelDeployment(
        model_group="slm-test-down",
        provider_key="slm_test_unreachable",
        model_id=f"openai/{_SLM_CFG.model_id}",
        kind=ModelKind.LLM,
        extra_params={"api_key": "sk-mu-local-slm-placeholder"},
    )
    base = default_local_catalog()
    catalog = base.model_copy(update={"providers": [provider], "deployments": [deployment]})
    models = ModelSettings(
        provider="slm_test_unreachable",
        answer_model="slm-test-down",
        adjudicate_model="slm-test-down",
        hard_extract_model="slm-test-down",
        routine_extract_model="slm-test-down",
        summarize_model="slm-test-down",
        classify_model="slm-test-down",
        rerank_model="slm-test-down",
    )
    return models, catalog


async def _mtm_point_state(
    client: AsyncQdrantClient, ns: Namespace, memory_id: str, *, dim: int
) -> str | None:
    points = await client.retrieve(
        collection_name=collection_name(ns, dim), ids=[point_id(memory_id)], with_payload=True
    )
    if not points:
        return None
    return (points[0].payload or {}).get("state")


# ------------------------------------------------------------------------------------- fixtures
@pytest.fixture
def slm_router() -> ModelRouter:
    models, catalog = _build_slm_catalog(_SLM_CFG)
    return _router(models, catalog)


@pytest.fixture
def down_router() -> ModelRouter:
    models, catalog = _unreachable_slm_catalog()
    return _router(models, catalog)


# ==========================================================================================
# 1. Stage split — reconcile alone never calls the model / never writes.
# ==========================================================================================
async def test_reconcile_never_writes_or_touches_the_adjudicator(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t1 = datetime(2020, 1, 1, tzinfo=UTC)
    existing = make_item(
        ns, "Ada works at Acme", subject="Ada", predicate="works_at", obj="Acme", valid_at=t1
    )
    pipeline = DistillPipeline(ltm=ltm, clock=FrozenClock(t1))
    await pipeline.distill(ns, [existing])

    t2 = datetime(2024, 1, 1, tzinfo=UTC)
    incoming = make_item(
        ns, "Ada works at Globex", subject="Ada", predicate="works_at", obj="Globex", valid_at=t2
    )
    outcome = await pipeline._reconcile(ns, incoming)  # ReconcileConflictsStage-equivalent

    # detection-only: the candidate residue is correctly gathered ...
    assert [c.object for c in outcome.candidates] == ["Acme"]
    assert outcome.identical is None
    # ... but NOTHING was written — LTM still shows only the original fact.
    live = await ltm.graph_recall(ns, subject="Ada", limit=10)
    assert {h.item.object for h in live} == {"Acme"}


# ==========================================================================================
# 2. AC-3.1 — idempotent winner across two different FrozenClock instants (same total order).
# ==========================================================================================
async def test_ac_3_1_same_delta_set_two_clocks_same_winner(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    async def _run(now_instant: datetime, *, user: str) -> DistillActionKind:
        ns = make_ns(user=user)
        loser = make_item(
            ns, "Ada works at Acme", subject="Ada", predicate="works_at", obj="Acme", valid_at=t_old
        )
        winner = make_item(
            ns,
            "Ada works at Globex",
            subject="Ada",
            predicate="works_at",
            obj="Globex",
            valid_at=t_now,
        )
        pipeline = DistillPipeline(ltm=ltm, clock=FrozenClock(now_instant))
        await pipeline.distill(ns, [loser])
        report = await pipeline.distill(ns, [winner])
        assert report.actions[0].loser_ids == (loser.id,)
        return report.actions[0].kind

    # Two wildly different "current tick" instants — the decision must depend ONLY on the facts'
    # own valid_at, never on the pipeline's injected Clock.
    kind_a = await _run(datetime(2024, 6, 1, tzinfo=UTC), user="clock-a")
    kind_b = await _run(datetime(2030, 1, 1, tzinfo=UTC), user="clock-b")
    assert kind_a is kind_b is DistillActionKind.SUPERSEDE


# ==========================================================================================
# 3. Real SLM catches a genuine semantic contradiction the heuristic alone would miss.
# ==========================================================================================
async def test_llm_adjudicator_catches_contradiction_heuristic_misses(
    ltm: FalkorLtmAdapter,
    slm_router: ModelRouter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    # "favorite_language" is NOT in DistillSettings.functional_predicates, and the objects differ
    # -> PolarityCardinalityHeuristic alone calls this COEXIST (a real, documented blind spot).
    existing = make_item(
        ns,
        "Ada's favorite programming language is Python",
        subject="Ada",
        predicate="favorite_language",
        obj="Python",
        valid_at=t_old,
    )
    incoming = make_item(
        ns,
        "Ada's favorite programming language is now Rust",
        subject="Ada",
        predicate="favorite_language",
        obj="Rust",
        valid_at=t_now,
    )

    bus = _RecordingBus()
    adjudicator = ConflictAdjudicator(router=slm_router, bus=bus)
    pipeline = DistillPipeline(ltm=ltm, clock=FrozenClock(t_now), adjudicator=adjudicator)
    await pipeline.distill(ns, [existing])
    report = await pipeline.distill(ns, [incoming])

    action = report.actions[0]
    print(f"\n[LLM-ADJUDICATE] kind={action.kind} loser_ids={action.loser_ids}")  # noqa: T201
    assert (
        action.kind is DistillActionKind.SUPERSEDE
    ), "the REAL SLM failed to catch the semantic contradiction the heuristic alone would miss"
    assert action.loser_ids == (existing.id,)
    # no budget/LLM-down degrade fired — this was a genuine, successful LLM adjudication.
    assert not bus.degrades(
        component="conflict", reason=DegradeReason.CONFLICT_ADJUDICATION_DEGRADED
    )


# ==========================================================================================
# 4. AC-3.2 — budget caps the LLM spend; only the (N+1)th candidate degrades.
# ==========================================================================================
async def test_ac_3_2_budget_degrades_only_the_nth_plus_one_candidate(
    ltm: FalkorLtmAdapter,
    slm_router: ModelRouter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t0 = datetime(2018, 1, 1, tzinfo=UTC)

    # Three PRE-EXISTING same-(subject,predicate) facts -> one winner's residue has exactly 3
    # candidates (a functional predicate, so the heuristic ALSO agrees each is a contradiction —
    # this test is about the BUDGET mechanics, not about disagreeing with the heuristic).
    existing_ids: list[str] = []
    for i, obj in enumerate(["Acme", "Initech", "Globex"]):
        item = make_item(
            ns,
            f"Ada works at {obj}",
            subject="Ada",
            predicate="works_at",
            obj=obj,
            valid_at=t0,
            memory_id=f"cand-{i}-{ns.user}",
        )
        await ltm.upsert_fact(item)
        existing_ids.append(item.id)

    bus = _RecordingBus()
    settings = ConflictAdjudicatorSettings(adjudication_budget_per_sweep=2)
    adjudicator = ConflictAdjudicator(router=slm_router, settings=settings, bus=bus)
    pipeline = DistillPipeline(
        ltm=ltm, clock=FrozenClock(datetime(2024, 1, 1, tzinfo=UTC)), adjudicator=adjudicator
    )

    winner = make_item(
        ns,
        "Ada works at Umbrella",
        subject="Ada",
        predicate="works_at",
        obj="Umbrella",
        valid_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    report = await pipeline.distill(ns, [winner])

    degraded = bus.degrades(
        component="conflict", reason=DegradeReason.CONFLICT_ADJUDICATION_DEGRADED
    )
    print(f"\n[AC-3.2] degrade_events={len(degraded)} action={report.actions[0]}")  # noqa: T201
    assert len(degraded) == 1, "exactly the (budget+1)th candidate must degrade — no more, no less"
    # every candidate (LLM-adjudicated or budget-degraded-to-heuristic) agreed SUPERSEDE here (a
    # functional predicate) -> all 3 pre-existing facts are invalidated regardless of path.
    assert set(report.actions[0].loser_ids) == set(existing_ids)


# ==========================================================================================
# 5. LLM down (real unreachable endpoint) -> named degrade + heuristic decides.
# ==========================================================================================
async def test_llm_down_degrades_to_heuristic_named_reason(
    ltm: FalkorLtmAdapter,
    down_router: ModelRouter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)
    existing = make_item(
        ns, "Ada works at Acme", subject="Ada", predicate="works_at", obj="Acme", valid_at=t_old
    )
    incoming = make_item(
        ns, "Ada works at Globex", subject="Ada", predicate="works_at", obj="Globex", valid_at=t_now
    )

    bus = _RecordingBus()
    adjudicator = ConflictAdjudicator(router=down_router, bus=bus)
    pipeline = DistillPipeline(ltm=ltm, clock=FrozenClock(t_now), adjudicator=adjudicator)
    await pipeline.distill(ns, [existing])
    report = await pipeline.distill(ns, [incoming])

    assert bus.degrades(component="conflict", reason=DegradeReason.CONFLICT_ADJUDICATION_DEGRADED)
    action = report.actions[0]
    # the deterministic heuristic floor still correctly decides (functional predicate, recency).
    assert action.kind is DistillActionKind.SUPERSEDE
    assert action.loser_ids == (existing.id,)


# ==========================================================================================
# 6. A validity TIE even the heuristic can't decide -> MANUAL_PENDING, never a fabricated winner.
# ==========================================================================================
async def test_validity_tie_parks_manual_pending_never_fabricates(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t_same = datetime(2022, 6, 15, tzinfo=UTC)
    # BUG1 FIX (data-quality re-assessment §2): the tie this test exercises is now an ASSERTION-
    # recency tie (`created_at`), never a `valid_at` tie — `distill.py::_asserted_later` (via
    # `_heuristic_verdict`, `conflict.py`) is the ONE predicate that decides direction throughout
    # the pipeline post-fix, so an identical `created_at` (pinned explicitly here — real wall-clock
    # `make_item()` calls would never tie at datetime-microsecond resolution) is the genuine
    # "even the heuristic can't decide" case. `valid_at` is still set (bi-temporal world-time) but
    # deliberately NOT tied — proving the tie-detection no longer keys on it at all.
    existing = make_item(
        ns,
        "Ada works at Acme",
        subject="Ada",
        predicate="works_at",
        obj="Acme",
        valid_at=t_same,
        created_at=t_same,
    )
    incoming = make_item(
        ns,
        "Ada works at Globex",
        subject="Ada",
        predicate="works_at",
        obj="Globex",
        valid_at=datetime(2030, 1, 1, tzinfo=UTC),  # deliberately NOT tied — must not matter
        created_at=t_same,  # EXACT assertion-time tie with `existing`
    )

    records = InMemoryConflictRecordRepository()
    bus = _RecordingBus()
    # No router at all -> straight to the strict (tie-is-PENDING) heuristic floor.
    adjudicator = ConflictAdjudicator(bus=bus, conflict_records=records)
    pipeline = DistillPipeline(ltm=ltm, clock=FrozenClock(t_same), adjudicator=adjudicator)
    await pipeline.distill(ns, [existing])
    report = await pipeline.distill(ns, [incoming])

    action = report.actions[0]
    assert action.loser_ids == (), "a tie must never fabricate a winner (no invalidate applied)"
    assert action.kind is DistillActionKind.COEXIST

    pending = await records.pending(ns)
    assert len(pending) == 1
    record = pending[0]
    assert record.proposed_winner_id is None  # genuinely undecided — not even a "proposed" guess
    assert set(record.member_ids) == {existing.id, incoming.id}

    # BOTH facts remain active in LTM — neither side was silently chosen.
    live_objs = {h.item.object for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert live_objs == {"Acme", "Globex"}


# ==========================================================================================
# 6b. MANUAL policy — CANONICAL §7.20: parks EVERY non-COEXIST verdict, never auto-applies, even
#     when the (heuristic-floor) decision itself is perfectly clear-cut.
# ==========================================================================================
async def test_manual_policy_parks_even_a_clear_cut_verdict(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)
    existing = make_item(
        ns, "Ada works at Acme", subject="Ada", predicate="works_at", obj="Acme", valid_at=t_old
    )
    incoming = make_item(
        ns, "Ada works at Globex", subject="Ada", predicate="works_at", obj="Globex", valid_at=t_now
    )

    records = InMemoryConflictRecordRepository()
    policy = ConflictResolutionPolicy(mode=ResolutionMode.MANUAL)
    adjudicator = ConflictAdjudicator(policy=policy, conflict_records=records)
    pipeline = DistillPipeline(ltm=ltm, clock=FrozenClock(t_now), adjudicator=adjudicator)
    await pipeline.distill(ns, [existing])
    report = await pipeline.distill(ns, [incoming])

    # the heuristic floor is unambiguous here (functional predicate, `incoming` strictly newer) —
    # MANUAL policy still refuses to auto-apply it.
    assert report.actions[0].loser_ids == ()
    pending = await records.pending(ns)
    assert len(pending) == 1
    assert pending[0].proposed_winner_id == incoming.id  # the verdict WAS decided (SUPERSEDE) ...
    # ... it is simply withheld from auto-apply, per MANUAL policy — both stay live either way.
    live_objs = {h.item.object for h in await ltm.graph_recall(ns, subject="Ada", limit=10)}
    assert live_objs == {"Acme", "Globex"}


# ==========================================================================================
# 7. MTM_INVALIDATE_POINT_ABSENT — real Qdrant 404 degrades named + retries next tick.
# ==========================================================================================
async def test_mtm_invalidate_point_absent_degrades_and_retries_next_tick(
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    dim = mtm._dim
    ns = make_ns()
    t_old = datetime(2019, 1, 1, tzinfo=UTC)
    t_now = datetime(2024, 1, 1, tzinfo=UTC)

    # The loser is distilled into LTM but its MTM (Qdrant) point is deliberately NEVER upserted —
    # simulates a genuine write-after-read visibility lag: the point doesn't exist yet. A
    # DIFFERENT placeholder point IS upserted first so the collection itself exists (an
    # entirely-missing COLLECTION short-circuits `invalidate` as a silent no-op — `qdrant_mtm.py
    # :_invalidate_impl`'s `collection_exists` guard — which would not exercise the 404 path at
    # all; the real bug is a missing POINT inside an existing collection).
    placeholder = make_item(ns, "placeholder", subject="Placeholder", predicate="is", obj="x")
    await mtm.upsert(placeholder)

    loser_src = make_item(
        ns,
        "Ada works at Acme",
        subject="Ada",
        predicate="works_at",
        obj="Acme",
        valid_at=t_old,
    )

    bus = _RecordingBus()
    pipeline = DistillPipeline(ltm=ltm, clock=FrozenClock(t_old), mtm=mtm, bus=bus)
    await pipeline.distill(ns, [loser_src])

    winner_src = make_item(
        ns, "Ada works at Globex", subject="Ada", predicate="works_at", obj="Globex", valid_at=t_now
    )
    pipeline2 = DistillPipeline(ltm=ltm, clock=FrozenClock(t_now), mtm=mtm, bus=bus)
    report = await pipeline2.distill(ns, [winner_src])

    assert report.actions[0].kind is DistillActionKind.SUPERSEDE
    degraded = bus.degrades(component="mtm", reason=DegradeReason.MTM_INVALIDATE_POINT_ABSENT)
    assert len(degraded) == 1, "the absent MTM point must degrade named, not raise/silently vanish"

    # LTM (source of truth) is ALREADY superseded regardless of MTM visibility.
    now_objs = {f.object for f in await ltm.facts_at(ns, t_now, subject="Ada")}
    assert now_objs == {"Globex"}

    # The point becomes visible (the real-world race resolves) -> upsert it now, exactly like the
    # promotion pipeline would once the write finally lands.
    await qdrant_client.upsert(
        collection_name=collection_name(ns, dim),
        points=[
            PointStruct(
                id=point_id(loser_src.id),
                vector=[0.0] * dim,
                # `namespace` is not decoration: every point the promotion pipeline writes carries
                # it (`QdrantMapper.to_store`), and the by-id payload write is namespace-scoped in
                # the adapter (C3) — a hand-rolled point without it is not a point that could
                # exist, and the retry below would correctly refuse to patch it.
                payload={"state": "active", "namespace": ns.to_prefix()},
            )
        ],
    )
    assert await _mtm_point_state(qdrant_client, ns, loser_src.id, dim=dim) == "active"

    # The NEXT sweep tick for this namespace drains the bounded retry queue.
    await pipeline2.distill(ns, [])
    assert await _mtm_point_state(qdrant_client, ns, loser_src.id, dim=dim) == "superseded"
