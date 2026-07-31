"""DISTILL — the MTM->LTM consolidation pipeline (engine-core-spec §6.4 DISTILL, §7;
data-extraction-methodology §3).

Turns promoted MTM atomic facts into bi-temporal LTM knowledge-graph facts and applies
**invalidate-don't-delete** supersession. It composes:

* extraction (``services/extract.py``) — the deterministic heuristic (MVP default) or the LLM
  extractor, behind ``FactExtractorPort``;
* the **mem0 ADD/UPDATE/DELETE(->SUPERSEDE)/NOOP diff loop** — PORT of
  ``other_repos/mem0/mem0/memory/main.py:463-590`` (candidate gather → per-fact decision →
  action dispatch), with the recorded deviation ``DELETE→SUPERSEDE`` (never hard-delete,
  engine-core §7.2);
* **Graphiti bi-temporal invalidation** — PORT of the pure interval logic in
  ``other_repos/graphiti/graphiti_core/utils/maintenance/edge_operations.py:406-441`` (a loser
  edge gets ``invalid_at = winner.valid_at`` and is RETAINED — timestamps set, never removed)
  plus the new-edge self-expiry at ``edge_operations.py:622-639`` (if a candidate is more recent
  than the incoming fact, the INCOMING fact self-expires instead of superseding).

The contradiction test is cheapest-first (methodology §3.2 c): the ``PolarityCardinalityHeuristic``
— an identical triple with opposite polarity is a direct contradiction; a single-cardinality
(functional) predicate with a different object is a functional supersession (PORT
``shared/stores/graph_falkor.py:505`` semantics) — is now the **candidate gate** feeding
``mu_engine.lifecycle.conflict.ConflictAdjudicator`` (S3-01, ADR 0037, spec §8): when an
adjudicator is wired the LLM renders the real verdict over every same-subject/predicate residue
candidate (catching a genuine semantic contradiction the heuristic alone would miss), with the
heuristic demoted to the degrade floor (LLM disabled/down/budget-exhausted). With NO adjudicator
wired, the ORIGINAL heuristic-only decision applies verbatim (100% backward compatible). See
``lifecycle/conflict.py``'s module docstring for the full placement/degrade/policy contract and
CANONICAL §7.20's ``ReconcileConflictsStage``/``ResolveConflictStage`` split, mirrored here as
``_reconcile``/``_resolve``. All writes go through the ``GraphStorePort`` (LTM / FalkorDB) — never
a store client directly (repository pattern); the one exception is the narrow, NAMED translation
of the cross-store MTM invalidate's Qdrant 404 (write-after-read visibility lag) into
``DegradedModeEntered(reason=MTM_INVALIDATE_POINT_ABSENT)`` (spec §13.2 fix #2) — see
``_invalidate_mtm_guarded``. Runs off the request path under a single writer lease (LOCAL
``InlineRunner`` equivalent = the in-process lease here; Redis SETNX lease deferred to the SHARED
plane).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict
from qdrant_client.http.exceptions import UnexpectedResponse

from mu_contracts.domain.events import (
    ConsolidationCompleted,
    DegradedModeEntered,
    DegradeReason,
    DomainEvent,
    FactsExtracted,
    MemoryPromoted,
    MemorySuperseded,
)
from mu_contracts.domain.model.memory import Tier
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    TraceScope,
)
from mu_engine.services.extract import ExtractedFact, FactExtractorPort, HeuristicSpoExtractor
from mu_engine.storage.domain.memory import (
    MemoryItem,
    MemorySource,
    MemoryState,
    MemoryTier,
)
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.ports import GraphStorePort, MtmTierRepository

if TYPE_CHECKING:
    # `mu_engine.lifecycle` package's own `__init__.py` eagerly imports `promotion.py`/`manager.py`,
    # which import `DistillPipeline` FROM THIS MODULE (mu_engine/lifecycle/__init__.py's docstring:
    # "Stage-3 slices (conflict.py) ... join this re-export as their own stage merges" — but the
    # PACKAGE __init__ already unconditionally imports promotion.py/manager.py today). A top-level
    # `from mu_engine.lifecycle.conflict import ...` HERE would therefore force-run
    # `lifecycle/__init__.py` mid-way through this module's own top-level execution, which in turn
    # needs `DistillPipeline` (defined at the BOTTOM of this file) — a genuine, unavoidable-by-
    # reordering circular import. Fixed the standard way: annotations only need this under
    # `TYPE_CHECKING` (this file has `from __future__ import annotations`, so no annotation ever
    # evaluates these names at runtime); every actual runtime use (the `AdjudicationKind` enum
    # comparisons + the `AdjudicationVerdict`/no-adjudicator constructor call) does its own late,
    # function-body-local import instead (`_resolve`/`_heuristic_only_verdict` below) — by the time
    # either is ever CALLED, both packages have long finished initializing.
    from mu_engine.lifecycle.conflict import (
        AdjudicationBudget,
        AdjudicationVerdict,
        ConflictAdjudicator,
    )

__all__ = [
    "DistillAction",
    "DistillActionKind",
    "DistillPipeline",
    "DistillReport",
    "DistillSettings",
    "EventPublisher",
    "InProcessWriterLease",
    "WriterLeasePort",
]

_log = structlog.get_logger("mu_engine.pipelines.distill")

_OP = "distill.consolidate"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"


# --------------------------------------------------------------------------------------- settings
class DistillSettings(BaseModel):
    """DISTILL/conflict knobs (engine-core-spec §12 ``DistillSettings``/``ConflictSettings``).

    Declared here as a tracked seam (same pattern as ``providers/settings.py``): the central
    ``Settings`` tree does not yet carry the engine subtree, so the pipeline takes this
    explicitly and the composition root wires ``settings.distill`` when it lands — no re-shape.
    Defaults are the sanctioned central-config home (DEV-STANDARDS rule 3): no threshold or
    predicate set is hardcoded in a code path.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_k: int = 5  # mem0 top-5 candidate gather (main.py:481)
    supersede_confidence: float = 0.8  # C>=0.8 -> SUPERSEDE (Φ(C,T), engine-core §7.4)
    refine_confidence: float = 0.5
    use_llm_extractor: bool = False  # MVP default = deterministic heuristic
    # Sibling of `use_llm_extractor` (S3-01, spec §8, ADR 0037): the composition root reads this
    # to decide whether to build a real `ConflictAdjudicator` (mirrors `use_llm_extractor` ->
    # `build_extractor`'s own precedent — NOT read internally by `DistillPipeline`, which instead
    # gates purely on whether an `adjudicator` instance was actually injected). Default True on a
    # capable model; False -> heuristic-only (`build_conflict_adjudicator(use_llm=False)` -> None).
    use_llm_adjudicator: bool = True

    # Single-cardinality predicates: a *different* object supersedes (functional relation).
    # Non-functional predicates (e.g. "likes") let multiple objects coexist (ADD, not supersede).
    functional_predicates: frozenset[str] = frozenset(
        {
            "is",
            "lives_in",
            "live_in",
            "lived_in",
            "resides_in",
            "works_at",
            "work_at",
            "worked_at",
            "based_in",
            "located_in",
            "moved_to",
            "relocated_to",
            "reports_to",
            # D5-quick (predicate/functional-predicate side ONLY — reconcile/supersede ROUTING
            # is untouched, Rank 3 D2's territory): the extractor's new D-7 canonical predicates
            # (``services/extract.py::ExtractionSettings.canonical_predicate_map`` +
            # ``_canonicalize_predicate``'s head-initial-preposition rule) collapse each of these
            # sentences' v1/v2 surface predicates onto ONE stable string per subject — and every
            # one names a single-cardinality real-world relation (one current flight day, one
            # current project deadline, one laptop, one current hotel, one team standup time),
            # so a different object for the SAME (subject, predicate) is a genuine supersession,
            # not a coexisting second value. Without this entry the now-colliding v1/v2 pair
            # would fall through ``_contradicts`` to COEXIST (both facts staying active) even
            # though they now correctly reach the SAME reconcile candidate set.
            "flight",
            "project_deadline",
            "laptop",
            "hotel",
            "team_standup",
        }
    )


# --------------------------------------------------------------------------------------- ports
@runtime_checkable
class EventPublisher(Protocol):
    """Narrow content-free publish seam (subset of ``EventBusPort``); optional injection."""

    async def publish(self, event: DomainEvent) -> None: ...


@runtime_checkable
class WriterLeasePort(Protocol):
    """The one supersession writer lease (engine-core §7.5). Acquire-or-defer, plane-qualified."""

    def acquire(self, ns: Namespace) -> contextlib.AbstractAsyncContextManager[None]: ...


class InProcessWriterLease:
    """In-process per-``to_prefix()`` exclusive lease — the LOCAL ``InlineRunner`` single-writer.

    Real for the LOCAL plane (one daemon process): an ``asyncio.Lock`` per namespace prefix
    serialises concurrent reconciles so a race yields one deterministic winner. The SHARED-plane
    Redis SETNX lease (``distill-writer-lease:{plane}:{ns.to_prefix()}``) is a drop-in adapter
    behind this same port (deferred — not needed for LOCAL correctness).
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def acquire(self, ns: Namespace) -> AsyncIterator[None]:
        prefix = ns.to_prefix()
        async with self._guard:
            lock = self._locks.setdefault(prefix, asyncio.Lock())
        async with lock:
            yield


# --------------------------------------------------------------------------------------- DTOs
class DistillActionKind(StrEnum):
    """The mem0 event vocabulary, mapped onto our substrate (DELETE→SUPERSEDE, engine-core §7.2)."""

    ADD = "add"  # mem0 ADD — new subject/predicate, no live candidate
    NOOP = "noop"  # mem0 NONE — identical active fact (reinforce)
    SUPERSEDE = "supersede"  # mem0 DELETE→SUPERSEDE — loser invalidated, retained
    SELF_EXPIRE = "self_expire"  # Graphiti new-edge self-expiry (edge_operations.py:622-639)
    COEXIST = "coexist"  # non-functional predicate, different object — both stay active


class DistillAction(BaseModel):
    """One reconcile outcome (in-process return value; content is legal off the bus)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DistillActionKind
    winner_id: str
    loser_ids: tuple[str, ...] = ()
    subject: str
    predicate: str
    object: str
    valid_at: datetime
    valid_at_inferred: bool
    reason: str


class DistillReport(BaseModel):
    """Aggregate DISTILL outcome for one window (content-free counts + per-action detail)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    facts_extracted: int
    actions: tuple[DistillAction, ...] = ()

    @property
    def added(self) -> int:
        return sum(a.kind is DistillActionKind.ADD for a in self.actions)

    @property
    def superseded(self) -> int:
        return sum(len(a.loser_ids) for a in self.actions)

    @property
    def noop(self) -> int:
        return sum(a.kind is DistillActionKind.NOOP for a in self.actions)


class _ReconcileOutcome(BaseModel):
    """``ReconcileConflictsStage``-equivalent return value (CANONICAL §7.20) — pure detection: the
    gathered same-subject/predicate candidate residue + the identical-match, if any. Carries NO
    verdict and triggers NO write; ``DistillPipeline._resolve`` is the only consumer."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    winner: MemoryItem
    identical: MemoryItem | None
    candidates: tuple[MemoryItem, ...]


class _PendingMtmInvalidate(BaseModel):
    """One MTM cross-store invalidate deferred by ``_invalidate_mtm_guarded`` (spec §13.2 fix #2)
    — a not-yet-visible Qdrant point at supersede time. Retried ONCE at the top of the next
    ``distill()`` call for the same namespace (``_retry_pending_mtm_invalidates``)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    ns: Namespace
    loser_point_id: str
    winner_point_id: str
    at: datetime
    reason: str


def _heuristic_only_verdict(
    winner: MemoryItem, candidate: MemoryItem, contradicts: bool
) -> AdjudicationVerdict:
    """The degrade floor used when NO ``ConflictAdjudicator`` was ever wired into this pipeline —
    reproduces the ORIGINAL pre-ADR-0037 heuristic-only decision VERBATIM (a ``valid_at`` TIE goes
    to the incoming winner, via the same bare ``>`` comparison the original ``more_recent`` list
    comprehension used) so every pre-existing heuristic-only test/caller is unaffected. Deferred,
    function-body-local import of ``mu_engine.lifecycle.conflict`` (see the ``TYPE_CHECKING`` note
    at this module's top) — this function is never called during either module's own import, only
    from within an actual ``distill()`` invocation, so the cycle that a top-level import would hit
    cannot occur here."""
    from mu_engine.lifecycle.conflict import AdjudicationKind, AdjudicationVerdict

    if not contradicts:
        return AdjudicationVerdict(
            kind=AdjudicationKind.COEXIST,
            apply=True,
            used_llm=False,
            confidence=1.0,
            reason="no_adjudicator_heuristic_coexist",
        )
    if _valid_at(candidate) > _valid_at(winner):
        return AdjudicationVerdict(
            kind=AdjudicationKind.SELF_EXPIRE,
            apply=True,
            used_llm=False,
            confidence=1.0,
            reason="no_adjudicator_heuristic_self_expire",
        )
    return AdjudicationVerdict(
        kind=AdjudicationKind.SUPERSEDE,
        apply=True,
        used_llm=False,
        confidence=1.0,
        reason="no_adjudicator_heuristic_supersede",
    )


# --------------------------------------------------------------------------------------- pipeline
class DistillPipeline:
    """MTM->LTM consolidation. Fully async, cancellation-safe, repository-only writes."""

    def __init__(
        self,
        *,
        ltm: GraphStorePort,
        extractor: FactExtractorPort | None = None,
        clock: Clock | None = None,
        settings: DistillSettings | None = None,
        mtm: MtmTierRepository | None = None,
        bus: EventPublisher | None = None,
        lease: WriterLeasePort | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
        adjudicator: ConflictAdjudicator | None = None,
    ) -> None:
        self._ltm = ltm
        self._settings = settings or DistillSettings()
        self._extractor = extractor or HeuristicSpoExtractor()
        self._clock: Clock = clock or SystemClock()
        self._mtm = mtm  # cross-store invalidate for an un-promoted MTM-resident loser (§7.5)
        self._bus = bus
        self._lease: WriterLeasePort = lease or InProcessWriterLease()
        # Central observability (DEV-STANDARDS rule 4): span + latency/error metrics + content-free
        # audit on the consolidation op. Sinks default no-op so the pipeline is testable unwired.
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()
        # S3-01 (spec §8, ADR 0037): optional LLM-judged supersession. `None` (the default) is
        # 100% pre-ADR-0037 backward compatible — `_resolve`'s no-adjudicator degrade floor
        # (`_heuristic_only_verdict`) reproduces the ORIGINAL heuristic-only decision verbatim.
        self._adjudicator: ConflictAdjudicator | None = adjudicator
        # Bounded MTM-invalidate retry queue (spec §13.2 fix #2): a write-after-read Qdrant 404 at
        # supersede time is queued here and retried ONCE at the top of the NEXT `distill()` call
        # for the SAME namespace — never a hot retry loop (`_invalidate_mtm_guarded`,
        # `_retry_pending_mtm_invalidates`).
        self._pending_mtm_retries: list[_PendingMtmInvalidate] = []

    async def distill(self, ns: Namespace, window: Sequence[MemoryItem]) -> DistillReport:
        """Consolidate a window of promoted MTM facts into LTM (the S3 pass).

        Structured items (subject/predicate/object already set by S2) promote with their
        tier-stable id preserved (CANONICAL §7.1 id-stability). Unstructured items are run
        through the extractor, minting new proposition ids with provenance back to the source.
        Runs under the exclusive writer lease so a concurrent reconcile can't double-write. Wrapped
        in central observability (DEV-STANDARDS rule 4): a content-free span, latency (always) +
        error (on failure) metrics, and a content-free audit row on success. ``CancelledError``
        propagates and is never counted as a failure.
        """
        started = time.perf_counter()
        with self._tracer.span(_OP, attributes={"ns": ns.to_prefix()}):
            try:
                report = await self._distill(ns, window)
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._metrics.inc(_ERROR_METRIC, labels={"operation": _OP})
                raise
            finally:
                self._metrics.observe(
                    _LATENCY_METRIC, time.perf_counter() - started, labels={"operation": _OP}
                )
        self._audit.record(
            TraceScope(correlation_id=ns.to_prefix()),
            operation=_OP,
            outcome="ok",
            tier="ltm",
            visibility=ns.visibility.value,
            counts={"facts": report.facts_extracted, "superseded": report.superseded},
        )
        return report

    async def _distill(self, ns: Namespace, window: Sequence[MemoryItem]) -> DistillReport:
        async with self._lease.acquire(ns):
            candidates = await self._collect_facts(ns, window)
            if self._bus is not None:
                await self._bus.publish(FactsExtracted(namespace=ns, count=len(candidates)))

            # Bounded retry (spec §13.2 fix #2) for an MTM invalidate degraded on a PRIOR sweep
            # tick for this namespace — attempted ONCE per tick, never a hot retry loop.
            await self._retry_pending_mtm_invalidates(ns)

            # S3-01 (spec §8 S1 / AC-3.2): one fresh per-sweep-tick budget, or None when no
            # adjudicator is wired (the no-adjudicator degrade floor never touches a budget).
            budget: AdjudicationBudget | None = (
                self._adjudicator.new_budget() if self._adjudicator is not None else None
            )

            # ReconcileConflictsStage-equivalent (CANONICAL §7.20): detection ONLY, for the WHOLE
            # window, before any resolve call is even issued — reconcile is demonstrably never
            # blocked waiting on an adjudication (see `_reconcile`/`_resolve` docstrings).
            reconciled = [await self._reconcile(ns, winner) for winner in candidates]

            # ResolveConflictStage-equivalent: the only stage that may await the LLM / write.
            actions = [await self._resolve(ns, outcome, budget) for outcome in reconciled]
            report = DistillReport(facts_extracted=len(candidates), actions=tuple(actions))

            if self._bus is not None:
                await self._bus.publish(
                    ConsolidationCompleted(
                        namespace=ns, facts_n=report.facts_extracted, superseded_n=report.superseded
                    )
                )
            _log.info(
                "distill_complete",
                ns=ns.to_prefix(),
                facts=report.facts_extracted,
                added=report.added,
                superseded=report.superseded,
                noop=report.noop,
            )
            return report

    # ---- extraction -------------------------------------------------------------------------
    async def _collect_facts(self, ns: Namespace, window: Sequence[MemoryItem]) -> list[MemoryItem]:
        """Build the candidate LTM winner items from the window (structured pass + extractor).

        SAME-BATCH CHRONOLOGY FIX (Rank 5): ``now`` used to be the ONE value handed to every
        item in the window as its ``valid_at`` fallback (below, and in ``_fact_to_item``) — two
        same-(subject,predicate) facts extracted from two DIFFERENT source messages in the SAME
        ``distill()`` window landed with an IDENTICAL ``valid_at``, a tie that
        ``_heuristic_only_verdict``'s (and ``ConflictAdjudicator``'s own degrade-floor) bare
        ``>`` comparison silently broke toward whichever fact happened to be resolved LAST
        (``_distill``'s window-order-preserving ``actions = [... for outcome in reconciled]``)
        — NOT whichever was actually more recent (the backwards-supersession defect this fixes).
        ``now`` is kept ONLY for ``_promote_structured``'s ``updated_at`` bookkeeping stamp (an
        administrative "when this promotion write happened" field, not the world-time
        ``valid_at`` that drives supersession ordering) and is no longer threaded into either
        fallback-``valid_at`` computation: each extracted fact instead anchors on ITS OWN source
        message's real ``item.created_at`` (see ``_fact_to_item``) — the same real add()-time
        instant STM's own recency ``ZSET`` already scores by
        (``redis_stm.py::_put_impl``: ``zadd(recency, {item.id: item.created_at.timestamp()})``),
        so no new timestamp source needed, just no longer discarding the one that already exists
        per item in favour of one shared instant for the whole window.
        """
        now = self._clock.now()
        out: list[MemoryItem] = []
        for item in window:
            if item.subject and item.predicate and item.object:
                out.append(self._promote_structured(item, now))
            else:
                # `now=item.created_at`: the extractor's `now` param is accepted only "for
                # signature symmetry with the port" (services/extract.py's `decompose_to_spo`
                # docstring — "dates come only from the text itself", never read in the body), so
                # threading this message's OWN real timestamp here instead of the shared window
                # `now` is a no-op for today's deterministic heuristic but is the honest per-item
                # reference instant for any future extractor that DOES resolve relative dates.
                facts = await self._extractor.extract(item.content, now=item.created_at)
                out.extend(self._fact_to_item(ns, item, f) for f in facts)
        return out

    @staticmethod
    def _promote_structured(item: MemoryItem, now: datetime) -> MemoryItem:
        # id-stability: the tier-stable id carries STM->MTM->LTM (CANONICAL §7.1) — reuse it.
        winner = item.model_copy(deep=True)
        winner.tier = MemoryTier.LTM
        winner.state = MemoryState.ACTIVE
        winner.updated_at = now
        if winner.valid_at is None:
            # LOUD recorded_at fallback (CANONICAL §7.17 / DATE_EXTRACTION_FALLBACK): a null
            # world-time defaults to transaction time and is flagged inferred in metadata.
            winner.valid_at = winner.created_at or now
            winner.metadata = {**winner.metadata, "valid_at_inferred": True}
        return winner

    @staticmethod
    def _fact_to_item(ns: Namespace, source: MemoryItem, fact: ExtractedFact) -> MemoryItem:
        # SAME-BATCH CHRONOLOGY FIX (Rank 5, see `_collect_facts`'s docstring): fall back to
        # THIS message's own real STM timestamp (`source.created_at`) rather than the shared
        # per-distill()-call clock `now` a prior revision passed in here. Two same-(subject,
        # predicate) facts extracted from two DIFFERENT source messages in ONE window now get
        # two genuinely distinct `valid_at` stamps (mirroring their real STM insert order)
        # instead of colliding on one shared instant — `fact.valid_at_inferred` (set True by the
        # extractor whenever `fact.valid_at is None`, `services/extract.py:383`) already flags
        # this LOUDLY below, unaffected by this change.
        valid_at = fact.valid_at if fact.valid_at is not None else source.created_at
        meta: dict[str, Any] = {"derived_from": source.id, "extractor_span": fact.source_span}
        if fact.valid_at_inferred:
            meta["valid_at_inferred"] = True
        return MemoryItem(
            content=fact.content,
            namespace=ns,
            owner_id=source.owner_id,
            workspace_id=source.workspace_id,
            session_id=source.session_id,
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            object_kind=fact.object_kind,
            polarity=fact.polarity,
            tier=MemoryTier.LTM,
            state=MemoryState.ACTIVE,
            source=MemorySource.INFERRED,
            valid_at=valid_at,
            importance_score=source.importance_score,
            metadata=meta,
        )

    # ---- reconcile (detect only) + resolve (adjudicate + apply) -----------------------------
    # CANONICAL §7.20 stage split: `_reconcile` == ReconcileConflictsStage (no model call, no
    # write — structurally incapable of either, it never touches `self._adjudicator`).
    # `_resolve` == ResolveConflictStage (the only place that may await the LLM / write). See
    # `lifecycle/conflict.py`'s module docstring for the full contract.
    async def _reconcile(self, ns: Namespace, winner: MemoryItem) -> _ReconcileOutcome:
        """ReconcileConflictsStage-equivalent: the mem0 candidate gather (main.py:463-488, active
        same-(subject,predicate) LTM facts) + the identical-active-fact check. Pure detection —
        no model call, no write, ever."""
        subject = winner.subject or ""
        predicate = winner.predicate or ""
        obj = winner.object or ""
        raw_candidates = await self._ltm.find_conflicts(ns, subject, predicate)
        candidates = [c for c in raw_candidates if c.id != winner.id][: self._settings.candidate_k]

        # mem0 NONE / graph_falkor.py:113 ON MATCH — identical active fact => NOOP (reinforce).
        identical = next(
            (c for c in candidates if (c.object or "") == obj and c.polarity == winner.polarity),
            None,
        )
        return _ReconcileOutcome(winner=winner, identical=identical, candidates=tuple(candidates))

    async def _resolve(
        self, ns: Namespace, outcome: _ReconcileOutcome, budget: AdjudicationBudget | None
    ) -> DistillAction:
        """ResolveConflictStage-equivalent: adjudicates every residue candidate (the heuristic
        candidate GATE's full survivor set, spec §8 — not just the ones the heuristic itself would
        flag) then applies the mem0 diff loop + Graphiti bi-temporal write for the aggregate
        outcome. The ONLY method in this class that may await `ConflictAdjudicator.adjudicate`."""
        from mu_engine.lifecycle.conflict import AdjudicationKind  # see TYPE_CHECKING note (top)

        winner = outcome.winner
        if outcome.identical is not None:
            # DEFECT-2 FIX (real-path verify gate, live-reproduced twice: deterministic + real-SLM
            # public path): `outcome.identical` is a `_reconcile`-time SNAPSHOT taken for the WHOLE
            # window BEFORE any `_resolve` call writes (module docstring / `_distill` — reconcile
            # is "detection ONLY ... before any resolve call is even issued"). When the SAME
            # `stm.recent()` window carries both the OLD raw message and its NEW contradicting
            # replacement, that OLD message's own reconcile outcome resolves to this
            # NOOP-reinforce branch holding a STALE pre-write copy of the very node the NEW
            # message's SUPERSEDE action invalidates earlier in this SAME batch (`_distill`'s
            # `actions = [... for outcome in reconciled]` runs every `_resolve` in window order,
            # sequentially, over a snapshot list built entirely up-front). Blindly
            # `model_copy`-ing that stale snapshot and `upsert_fact`-ing it back (the pre-fix
            # behaviour) re-SET `state`/`invalid_at` to the snapshot's active/'' values, silently
            # UNDOING the same-batch supersession — the exact semantic-shadowing defect (both
            # versions end up `state=active` in the graph). Fix: re-read the node's CURRENT
            # present-tense reality from the store immediately before writing, never trust the
            # reconcile-time snapshot for a write. If a same-batch action already superseded it,
            # `_current_node` returns `None` (the bi-temporal `facts_at` filter now excludes it)
            # and the reinforce is skipped outright — reinforcing a dead node is meaningless, and
            # skipping (rather than writing) is the only way that can never re-activate it.
            current = await self._current_node(ns, outcome.identical, at=self._clock.now())
            if current is None:
                return self._action(
                    DistillActionKind.NOOP,
                    outcome.identical,
                    (),
                    "identical_superseded_same_batch_skip_reinforce",
                )
            reinforced = current.model_copy(deep=True)
            reinforced.access_count += 1
            reinforced.updated_at = self._clock.now()
            await self._ltm.upsert_fact(reinforced)
            return self._action(DistillActionKind.NOOP, reinforced, (), "identical_active_fact")

        # DEFECT (ada_room real-path verify, live-reproduced): `outcome.candidates` is the
        # reconcile-time snapshot (see `_live_residue`'s docstring) — re-read live, right before
        # this decision, so a same-batch sibling written earlier in THIS resolve loop is seen.
        residue = await self._live_residue(ns, winner)
        if not residue:
            await self._ltm.upsert_fact(winner)
            await self._maybe_promote_event(ns, winner)
            return self._action(DistillActionKind.ADD, winner, (), "new_subject_predicate")

        # Written EARLY (mark_conflict wiring fix, live-reproduced): a withheld (PENDING/MANUAL)
        # verdict below tags `winner`-vs-`candidate` CONFLICTS_WITH mid-loop, before either the
        # COEXIST or SUPERSEDE branch would otherwise have upserted `winner` — `mark_conflict`'s
        # own MATCH would silently no-op (find nothing, MERGE nothing) against a `winner` node
        # that doesn't exist in the graph yet. Idempotent MERGE, so the SELF_EXPIRE branch's own
        # later re-upsert (with `state=SUPERSEDED`) safely overwrites this exact same node.
        await self._ltm.upsert_fact(winner)

        # Per-candidate verdicts. Sequential within ONE winner (AC-3.2's budget/order guarantee
        # applies per sweep-tick-per-winner residue, exactly the "N+1 candidates" shape it names).
        self_expire: list[tuple[MemoryItem, AdjudicationVerdict]] = []
        supersede: list[tuple[MemoryItem, AdjudicationVerdict]] = []
        for candidate in residue:
            heuristic_flag = self._contradicts(winner, candidate)
            if self._adjudicator is not None and budget is not None:
                verdict = await self._adjudicator.adjudicate(
                    ns=ns,
                    winner=winner,
                    candidate=candidate,
                    heuristic_contradicts=heuristic_flag,
                    budget=budget,
                )
            else:
                verdict = _heuristic_only_verdict(winner, candidate, heuristic_flag)
            if not verdict.apply:
                if verdict.kind is not AdjudicationKind.COEXIST:
                    # A genuine contradiction that could NOT be auto-applied — a PENDING
                    # bi-temporal tie (spec §8 "never fabricate") or a MANUAL-policy-withheld
                    # SUPERSEDE/SELF_EXPIRE (CANONICAL §7.20). The adjudicator already parks a
                    # `ConflictRecord` in its own side-channel inbox (`conflict.py::_park`); tag
                    # the pair CONFLICTS_WITH in the GRAPH itself too — both facts stay active
                    # (never a fabricated winner) but a direct GRAPH.QUERY / a future
                    # `build_context` reader can now render "conflicting values noted" instead of
                    # two bare, silently-unrelated active facts.
                    await self._ltm.mark_conflict(ns, winner.id, candidate.id, at=self._clock.now())
                continue  # parked (MANUAL_PENDING) or otherwise withheld — never fabricate
            if verdict.kind is AdjudicationKind.SELF_EXPIRE:
                self_expire.append((candidate, verdict))
            elif verdict.kind is AdjudicationKind.SUPERSEDE:
                supersede.append((candidate, verdict))
            # COEXIST -> no action; both stay active.

        # Graphiti bi-temporal interval logic (edge_operations.py:406-441/622-639): the incoming
        # fact self-expires if ANY candidate the adjudicator judged more authoritative exists.
        if self_expire:
            newest, _ = max(self_expire, key=lambda cv: _valid_at(cv[0]))
            winner.state = MemoryState.SUPERSEDED
            winner.invalid_at = _valid_at(newest)
            await self._ltm.upsert_fact(winner)
            await self._ltm.invalidate(
                ns, winner.id, newest.id, at=_valid_at(newest), reason="superseded_by_more_recent"
            )
            return self._action(
                DistillActionKind.SELF_EXPIRE, newest, (winner.id,), "incoming_older_than_candidate"
            )

        if not supersede:
            # every residue candidate resolved COEXIST (or was parked) -> COEXIST, both stay live.
            # `winner` was already upserted above (before the per-candidate loop).
            await self._maybe_promote_event(ns, winner)
            return self._action(DistillActionKind.COEXIST, winner, (), "non_functional_coexist")

        # SUPERSEDE: incoming wins, every adjudicated loser is invalidated-not-deleted.
        # `winner` was already upserted above (before the per-candidate loop).
        await self._maybe_promote_event(ns, winner)
        loser_ids: list[str] = []
        for loser, _ in supersede:
            at = _valid_at(winner)
            await self._ltm.invalidate(
                ns, loser.id, winner.id, at=at, reason="functional_supersede"
            )
            # cross-store: an un-promoted MTM-resident loser must also drop from MTM recall (§7.5).
            # The MTM point is keyed by the INGEST id (CANONICAL §7.1 id-stability), NOT the LTM
            # fact-node id: a structured fact keeps its ingest id STM->MTM->LTM, but an EXTRACTED
            # fact mints a fresh LTM node id and records its source ingest id in
            # metadata["derived_from"]. Passing the fact-node id here targets a non-existent Qdrant
            # point (404, the id-linkage crash); resolve the ingest id so the RIGHT MTM point flips.
            if self._mtm is not None:
                await self._invalidate_mtm_guarded(ns, loser=loser, winner=winner, at=at)
            loser_ids.append(loser.id)
            if self._bus is not None:
                await self._bus.publish(
                    MemorySuperseded(
                        namespace=ns, loser_id=loser.id, winner_id=winner.id, valid_at=at
                    )
                )
        return self._action(
            DistillActionKind.SUPERSEDE, winner, tuple(loser_ids), "functional_or_polarity_conflict"
        )

    async def _live_residue(self, ns: Namespace, winner: MemoryItem) -> tuple[MemoryItem, ...]:
        """Live re-read of ``find_conflicts`` immediately before ``_resolve`` uses it (same
        DEFECT-2 precedent as ``_current_node``, below): ``outcome.candidates`` is a
        ``_reconcile``-time SNAPSHOT taken for the WHOLE window before ANY ``_resolve`` call
        writes (module docstring / ``_distill``: reconcile runs for every winner up front, THEN
        resolve runs). Two facts sharing (subject, predicate) that land in the SAME distill()
        window — e.g. the real-path defect this fixes: "The Q3 planning meeting is in Room A"
        and "...was moved to Room B" extracted from the SAME session's SAME consolidate() call —
        never see each other in that snapshot: neither has been upserted to the store yet at the
        moment either's OWN ``_reconcile`` ran, so both silently ADD as two permanently-active
        "simultaneous truths" (ada_room real-path verify finding). Re-reading here, right before
        the write decision, picks up any same-batch sibling THIS SAME ``_resolve`` loop already
        wrote earlier in the batch (`_distill`'s `actions = [... for outcome in reconciled]` runs
        every `_resolve` sequentially, in window order) exactly as a genuinely pre-existing store
        candidate would be seen — no restructuring of the reconcile/resolve split needed."""
        raw = await self._ltm.find_conflicts(ns, winner.subject or "", winner.predicate or "")
        return tuple(c for c in raw if c.id != winner.id)[: self._settings.candidate_k]

    async def _current_node(
        self, ns: Namespace, stale: MemoryItem, *, at: datetime
    ) -> MemoryItem | None:
        """Fresh present-tense re-read of ``stale.id`` (DEFECT-2 FIX helper) — the ONLY safeguard
        between a reconcile-time snapshot and a write in `_resolve`'s NOOP branch. Uses
        `GraphStorePort.facts_at` (bi-temporal `valid_at <= at AND (invalid_at = '' OR
        invalid_at > at)`, no port-interface change needed) filtered by the node's OWN stored
        `subject` string — never the winner's freshly-extracted subject, so this re-read carries
        NO casing-collision risk of its own (DEFECT-1's concern), it is comparing the node against
        itself. Returns `None` when `stale.id` no longer satisfies the present-tense filter — i.e.
        an earlier action in THIS SAME `_distill` batch already invalidated it (or it lapsed on its
        own bi-temporal window) — the caller's cue to skip the reinforce rather than resurrect it.
        """
        fresh = await self._ltm.facts_at(ns, at, subject=stale.subject)
        return next((f for f in fresh if f.id == stale.id), None)

    # ---- MTM cross-store invalidate — NAMED degrade wrap (spec §13.2 fix #2) -----------------
    async def _invalidate_mtm_guarded(
        self, ns: Namespace, *, loser: MemoryItem, winner: MemoryItem, at: datetime
    ) -> None:
        """Wraps ``self._mtm.invalidate`` (the cross-store supersede write): a not-yet-visible
        Qdrant point (write-after-read visibility lag — the demo-found 404) degrades to a NAMED
        ``DegradedModeEntered(component="mtm", reason=MTM_INVALIDATE_POINT_ABSENT)`` instead of an
        uncaught 404/500, plus a bounded retry queued for the next sweep tick. The graph-side
        SUPERSEDED write (source of truth) has ALWAYS already succeeded by the time this runs —
        both call sites in `_resolve` invalidate LTM before this — regardless of MTM visibility.
        """
        if self._mtm is None:  # guarded at both call sites; loud, not an assert (bandit S101)
            raise RuntimeError("_invalidate_mtm_guarded called with no MtmTierRepository wired")
        loser_point = self._mtm_point_id(loser)
        winner_point = self._mtm_point_id(winner)
        reason = "functional_supersede"
        try:
            await self._mtm.invalidate(ns, loser_point, winner_point, at=at, reason=reason)
        except UnexpectedResponse as exc:
            self._pending_mtm_retries.append(
                _PendingMtmInvalidate(
                    ns=ns,
                    loser_point_id=loser_point,
                    winner_point_id=winner_point,
                    at=at,
                    reason=reason,
                )
            )
            _log.warning(
                "mtm_invalidate_point_absent",
                ns=ns.to_prefix(),
                status_code=exc.status_code,
            )
            if self._bus is not None:
                await self._bus.publish(
                    DegradedModeEntered(
                        component="mtm",
                        mode="mtm_invalidate_deferred",
                        reason=DegradeReason.MTM_INVALIDATE_POINT_ABSENT,
                        detail=f"status={exc.status_code}",
                    )
                )

    async def _retry_pending_mtm_invalidates(self, ns: Namespace) -> None:
        """Drains this namespace's queued ``_PendingMtmInvalidate`` rows, ONCE, at the top of a
        `distill()` call — a still-absent point is re-queued for the tick after that; never a hot
        retry loop. A different namespace's pending rows are left untouched (this pipeline may be
        shared across namespaces; each namespace only drains its own backlog on its own tick)."""
        if not self._pending_mtm_retries or self._mtm is None:
            return
        prefix = ns.to_prefix()
        remaining: list[_PendingMtmInvalidate] = []
        for pending in self._pending_mtm_retries:
            if pending.ns.to_prefix() != prefix:
                remaining.append(pending)
                continue
            try:
                await self._mtm.invalidate(
                    pending.ns,
                    pending.loser_point_id,
                    pending.winner_point_id,
                    at=pending.at,
                    reason=pending.reason,
                )
            except UnexpectedResponse:
                remaining.append(pending)  # still absent — retried again next tick
        self._pending_mtm_retries = remaining

    def _contradicts(self, winner: MemoryItem, candidate: MemoryItem) -> bool:
        """PolarityCardinalityHeuristic (graph_falkor.py:505): opposite-polarity identical triple
        OR a functional predicate with a different object."""
        same_object = (candidate.object or "") == (winner.object or "")
        if same_object and candidate.polarity != winner.polarity:
            return True  # direct contradiction (opposite polarity, same triple)
        predicate = (winner.predicate or "").lower()
        if not same_object and predicate in self._settings.functional_predicates:
            return True  # functional supersession (single-cardinality, different object)
        return False

    @staticmethod
    def _mtm_point_id(item: MemoryItem) -> str:
        """The MTM point (ingest) id a fact traces back to (CANONICAL §7.1 id-stability).

        A STRUCTURED fact promoted STM->MTM->LTM keeps its tier-stable id, so the LTM node id IS
        the MTM point id. An EXTRACTED fact mints a fresh LTM node id and records its originating
        ingest id in ``metadata['derived_from']`` — THAT is the id of the MTM-resident source
        point (the fact-node id has no MTM point and 404s the cross-store invalidate). The MTM
        supersede must key the loser by this ingest id, never the fact-node id (§7.5).
        """
        derived = item.metadata.get("derived_from")
        return str(derived) if derived else item.id

    async def _maybe_promote_event(self, ns: Namespace, item: MemoryItem) -> None:
        if self._bus is not None:
            await self._bus.publish(
                MemoryPromoted(
                    namespace=ns, id=item.id, frm=Tier.MTM, to=Tier.LTM, reason="distill"
                )
            )

    @staticmethod
    def _action(
        kind: DistillActionKind, winner: MemoryItem, losers: tuple[str, ...], reason: str
    ) -> DistillAction:
        inferred = bool(winner.metadata.get("valid_at_inferred", False))
        return DistillAction(
            kind=kind,
            winner_id=winner.id,
            loser_ids=losers,
            subject=winner.subject or "",
            predicate=winner.predicate or "",
            object=winner.object or "",
            valid_at=_valid_at(winner),
            valid_at_inferred=inferred,
            reason=reason,
        )


def _valid_at(item: MemoryItem) -> datetime:
    """A fact's world-time validity start; never ``None`` here (filled at promotion)."""
    if item.valid_at is None:  # defensive — promotion always sets it
        return item.created_at
    return item.valid_at
