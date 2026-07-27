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

The contradiction test is cheapest-first + no-LLM (methodology §3.2 c): the
``PolarityCardinalityHeuristic`` — an identical triple with opposite polarity is a direct
contradiction; a single-cardinality (functional) predicate with a different object is a
functional supersession (PORT ``shared/stores/graph_falkor.py:505`` semantics). All writes go
through the ``GraphStorePort`` (LTM / FalkorDB) — never a store client directly (repository
pattern). Runs off the request path under a single writer lease (LOCAL ``InlineRunner``
equivalent = the in-process lease here; Redis SETNX lease deferred to the SHARED plane).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.events import (
    ConsolidationCompleted,
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

            actions: list[DistillAction] = []
            for winner in candidates:
                action = await self._reconcile_and_apply(ns, winner)
                actions.append(action)
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
        """Build the candidate LTM winner items from the window (structured pass + extractor)."""
        now = self._clock.now()
        out: list[MemoryItem] = []
        for item in window:
            if item.subject and item.predicate and item.object:
                out.append(self._promote_structured(item, now))
            else:
                facts = await self._extractor.extract(item.content, now=now)
                out.extend(self._fact_to_item(ns, item, f, now) for f in facts)
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
    def _fact_to_item(
        ns: Namespace, source: MemoryItem, fact: ExtractedFact, now: datetime
    ) -> MemoryItem:
        valid_at = fact.valid_at if fact.valid_at is not None else now
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

    # ---- reconcile + apply (mem0 diff loop + Graphiti bi-temporal) --------------------------
    async def _reconcile_and_apply(self, ns: Namespace, winner: MemoryItem) -> DistillAction:
        subject = winner.subject or ""
        predicate = winner.predicate or ""
        obj = winner.object or ""
        # mem0 candidate gather (main.py:463-488): active same-(subject,predicate) LTM facts.
        raw_candidates = await self._ltm.find_conflicts(ns, subject, predicate)
        candidates = [c for c in raw_candidates if c.id != winner.id][: self._settings.candidate_k]

        # mem0 NONE / graph_falkor.py:113 ON MATCH — identical active fact => NOOP (reinforce).
        identical = next(
            (c for c in candidates if (c.object or "") == obj and c.polarity == winner.polarity),
            None,
        )
        if identical is not None:
            reinforced = identical.model_copy(deep=True)
            reinforced.access_count += 1
            reinforced.updated_at = self._clock.now()
            await self._ltm.upsert_fact(reinforced)
            return self._action(DistillActionKind.NOOP, reinforced, (), "identical_active_fact")

        # contradiction detection, no-LLM (PolarityCardinalityHeuristic, methodology §3.2 c).
        losers = [c for c in candidates if self._contradicts(winner, c)]
        if not losers:
            # ADD (no candidate at all) or COEXIST (non-functional predicate, different object).
            await self._ltm.upsert_fact(winner)
            await self._maybe_promote_event(ns, winner)
            kind = DistillActionKind.ADD if not candidates else DistillActionKind.COEXIST
            reason = "new_subject_predicate" if not candidates else "non_functional_coexist"
            return self._action(kind, winner, (), reason)

        # Graphiti bi-temporal interval logic (edge_operations.py:406-441 / 622-639):
        # the incoming fact self-expires if ANY contradicting candidate is strictly more recent.
        more_recent = [c for c in losers if _valid_at(c) > _valid_at(winner)]
        if more_recent:
            newest = max(more_recent, key=_valid_at)
            winner.state = MemoryState.SUPERSEDED
            winner.invalid_at = _valid_at(newest)
            await self._ltm.upsert_fact(winner)
            await self._ltm.invalidate(
                ns, winner.id, newest.id, at=_valid_at(newest), reason="superseded_by_more_recent"
            )
            return self._action(
                DistillActionKind.SELF_EXPIRE, newest, (winner.id,), "incoming_older_than_candidate"
            )

        # SUPERSEDE: incoming wins, every contradicting loser is invalidated-not-deleted.
        await self._ltm.upsert_fact(winner)
        await self._maybe_promote_event(ns, winner)
        loser_ids: list[str] = []
        for loser in losers:
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
                await self._mtm.invalidate(
                    ns,
                    self._mtm_point_id(loser),
                    self._mtm_point_id(winner),
                    at=at,
                    reason="functional_supersede",
                )
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
