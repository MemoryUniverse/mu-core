"""``PersonaService`` — the SLEEP-TIME persona orchestrator (``persona-design.md`` §6 line 219).

Authority: spec §2.4 (lines 117-122), §3.3 (lines 163-169), §4 (lines 173-179), §6 (line 219).

**§0's negative half is this class's real contract.** Persona is voice + relevance, never access
(§0 line 54, §5.4), and it is sleep-time, never the hot path (§2.4 line 122). Concretely, and
structurally rather than by promise:

* This service has **no read method**. Its entire public surface is :meth:`rebuild`,
  :meth:`refresh`, :meth:`note_promoted` and :meth:`forget` — four WRITE paths. The query-time
  persona read is ``PersonaRepository.load_brief(ns)``, a load by key on the contracts port, which
  takes no query, no candidate set and no ``authorized_ids`` and therefore cannot enter a recall
  ``query_filter`` (§5.4 rule 2).
* Persona **never resolves into ``authorized_ids``** (§5.4 rule 1): this module does not import,
  construct or read a ``CallerIdentitySet``, an authorized-id resolver or any recall type.
* **The incremental path cannot execute on the caller's stack.** This is the correction of a real
  defect. Spec line 121 lets Stage 1 run incrementally on ``MemoryPromoted(to=MTM/LTM)``, and the
  first build of this subsystem implemented that as an ``async def on_promoted`` that awaited a
  repo read, an evidence read, a repo write and a bus publish. ``MemoryPromoted``'s only live
  publisher is the ingest CAPTURE path (``pipelines/concrete/ingest.py:414`` inside
  ``IngestService._run_stage``, ``services/ingest.py:275-276``) and ``InprocBus.publish`` awaits
  every handler INLINE (``platform/adapters/bus_inproc.py:59-60``, exceptions propagating to the
  publisher), so subscribing that method would have put four persona awaits — and any model call
  hiding behind ``PersonaEvidenceReader`` — inside the user's ``remember()`` call, and would have
  let a routine ``PersonaVersionConflictError`` fail their capture. A static import scanner cannot
  see a bus subscription, so no test could have caught it.

  The shape that fixes it structurally: :meth:`note_promoted` is a **plain ``def``** that records
  an id in a bounded in-memory set and does nothing else. A sync method cannot ``await`` a store,
  a bus or a model — the same argument ``TraitAggregator`` makes for being sync
  (``aggregator.py:70-82``), applied to the seam that actually carries the model. It is total: it
  has no collaborator call to fail in and no exception path to propagate into ``remember()``. The
  work it defers is done by :meth:`refresh`, on the sleep-time side, beside :meth:`rebuild`.
* :meth:`refresh` and :meth:`rebuild` **are the only methods that touch the evidence reader**, and
  both are sleep-time. That is what makes spec line 121's "no LLM" real: the property is not
  "``on_promoted`` holds no synthesizer reference" (a model can also arrive through
  ``PersonaEvidenceReader``, which spec line 103 says is classifier-backed) but "the reader is
  never reachable from a caller-facing stack at all".

**Cadence — a recorded BLOCKER.** Spec line 119 triggers the rebuild from ``SleeptimeTick``
inside a Temporal ``SleeptimeWorkflow``. ``SleeptimeTick`` exists as a DTO
(``mu_contracts/domain/events.py:391``) and **has no publisher and no subscriber anywhere** in
mu-core, mu-client or mu-server, and no such workflow exists;
``MemoryLifecycleManager.on_bus_event`` (``lifecycle/manager.py:512``) filters to
``MemoryCaptured | MemoryPromoted`` only, and ``lifecycle/`` is outside this lane's file
ownership. So :meth:`rebuild`/:meth:`refresh` are built as the callables the maintenance owner
invokes — exactly how ``RetentionService.sweep`` is invoked from
``MemoryLifecycleManager.sweep_namespace_now`` (``manager.py:610-611``) — and
:meth:`PersonaSettings.due_at_tick` carries the letta cadence rule for them. The trigger wiring is
reported, not faked.

**Evidence — the other recorded blocker.** See :mod:`mu_engine.services.persona.evidence`: the
capture-time slot tag has nowhere to live on ``MemoryItem``, so no production
``PersonaEvidenceReader`` exists yet. Everything above that seam is built and tested.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any

import structlog

from mu_contracts.domain.errors import NamespaceIsolationError
from mu_contracts.domain.events import (
    DegradedModeEntered,
    DegradeReason,
    MemoryPromoted,
    PersonaUpdated,
)
from mu_contracts.domain.model.memory import Namespace, Tier, Visibility
from mu_contracts.domain.model.persona import PersonaProfile, PersonaSlot, SlotValue
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.persona import PersonaRepository
from mu_contracts.ports.security import TenancyGuard
from mu_contracts.ports.time import Clock
from mu_engine.pipelines.distill import EventPublisher
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import (
    NoopAuditLog,
    NoopMetricSink,
    NoopTracer,
    SafeTraceFields,
    TraceScope,
    sanitize_label_value,
    sanitize_labels,
)
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.services.persona.aggregator import TraitAggregator, slots_changed
from mu_engine.services.persona.evidence import PersonaEvidence, PersonaEvidenceReader
from mu_engine.services.persona.settings import PersonaSettings
from mu_engine.services.persona.store import assert_private, persona_key
from mu_engine.services.persona.synthesizer import PortraitSynthesizer

__all__ = ["PersonaService"]

_log = structlog.get_logger("mu_engine.services.persona")

_OP_REBUILD = "persona.rebuild"
_OP_REFRESH = "persona.refresh"
_OP_FORGET = "persona.forget"
_LATENCY_METRIC = "mu_operation_latency_seconds"
_ERROR_METRIC = "mu_operation_errors_total"

#: The tiers whose promotion feeds the §2.4 line 121 incremental slot upsert. STM is excluded
#: because an STM row is a TTL window, not a durable trait — a persona built from it would learn
#: and forget on the STM clock rather than on §3.3's decay curve.
_INCREMENTAL_TIERS: frozenset[Tier] = frozenset({Tier.MTM, Tier.LTM})

#: The version a ``PersonaUpdated`` carries when the record was ERASED (:meth:`forget`). The store
#: reserves ``0`` for "never built" (``store.py:99-100``), so it is the one value that cannot
#: collide with a live profile and reads unambiguously as "load_brief will now return nothing".
_ERASED_VERSION = 0

_DEGRADE_COMPONENT = "persona"
#: The ONE named degraded path: Stage 1 landed, Stage 2 did not (spec §2.2 succeeds without any
#: model at all), so the profile carries structured slots and the PREVIOUS brief.
_DEGRADE_MODE = "persona_slots_only"

#: RECORDED GAP: CANONICAL §2's ``DegradeReason`` union has no persona member, and this lane does
#: not own ``domain/events.py``. ``LLM_UNAVAILABLE_HEURISTIC`` is the closest existing named
#: reason and is already used for exactly this shape by the conflict adjudicator
#: (``lifecycle/conflict.py:434``: no model configured -> deterministic result). A
#: ``PERSONA_SYNTHESIS_DEGRADED`` member should be added — flagged, not invented here.
_DEGRADE_REASON = DegradeReason.LLM_UNAVAILABLE_HEURISTIC


class PersonaService:
    """Sleep-time rebuild + deferred incremental slot upsert (spec §6 line 219)."""

    def __init__(
        self,
        *,
        repo: PersonaRepository,
        evidence: PersonaEvidenceReader,
        aggregator: TraitAggregator,
        synthesizer: PortraitSynthesizer | None = None,
        settings: PersonaSettings | None = None,
        clock: Clock | None = None,
        scope_guard: TenancyGuard | None = None,
        bus: EventPublisher | None = None,
        tracer: Tracer | None = None,
        metrics: MetricSink | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        """``synthesizer`` is OPTIONAL on purpose: Stage 1 is a complete, useful persona on its
        own (spec §2.2 needs no model), so a FULL-LOCAL deployment with no model wired still gets
        structured slots plus a NAMED degrade — never a silent half-result and never a hard
        refusal that would leave the local plane with no persona at all."""
        self._repo = repo
        self._evidence = evidence
        self._aggregator = aggregator
        self._synthesizer = synthesizer
        self._settings = settings or PersonaSettings()
        self._clock: Clock = clock or SystemClock()
        self._scope_guard: TenancyGuard = scope_guard or DefaultTenancyGuard()
        self._bus = bus
        self._tracer: Tracer = tracer or NoopTracer()
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._audit: AuditLog = audit or NoopAuditLog()
        #: persona key -> promoted memory ids awaiting the next :meth:`refresh`. Bounded on both
        #: axes by ``PersonaSettings`` — see :meth:`note_promoted`.
        self._pending: dict[str, set[str]] = {}
        #: Ids dropped because a bound was reached, reported as a COUNT at drain time.
        self._dropped: dict[str, int] = {}

    # ----------------------------------------------------------------------------- stage 1+2
    async def rebuild(self, scope: ClientScope, ns: Namespace) -> PersonaProfile | None:
        """Aggregate -> synthesize -> versioned upsert -> emit ``PersonaUpdated`` (spec line 219).

        Returns ``None`` — never a half-built profile — when persona is disabled, or when no
        profile exists yet and the partition has fewer than ``PersonaSettings.min_support``
        persona-tagged memories (§3.3 line 165's create-on-first-tick guard against a portrait
        synthesised from one utterance).

        **Signature delta (recorded).** Spec line 219 writes ``rebuild(ns) -> PersonaProfile``.
        ``scope`` is added as the first parameter because ``TenancyGuard.assert_scope(scope, ns,
        operation)`` needs it (the identical delta ``MemoryHealthService.assess`` records), and the
        return is widened to ``| None`` because "not enough support yet" is a normal sleep-time
        outcome on a young partition, not an error.
        """
        self._scope_guard.assert_scope(scope, ns, _OP_REBUILD)
        profile, evidence_count, changed = await self._timed(
            _OP_REBUILD, ns, lambda: self._rebuild(ns)
        )
        # A full rebuild reads the WHOLE evidence set, so anything :meth:`note_promoted` had
        # queued for this user is already folded in; keeping it would only re-do the work.
        self._drain_pending(ns)
        self._record(_OP_REBUILD, ns, profile, evidence_count=evidence_count, changed=changed)
        return profile

    async def _rebuild(self, ns: Namespace) -> tuple[PersonaProfile | None, int, int]:
        if not self._settings.enabled:
            return None, 0, 0
        assert_private(ns, _OP_REBUILD)
        now = self._clock.now()

        evidence = self._own_partition_only(
            ns, await self._evidence.evidence_for(ns, limit=self._settings.max_evidence_items)
        )
        existing = await self._repo.get(ns)
        source_count = _distinct_items(evidence)
        if existing is None and source_count < self._settings.min_support:
            _log.debug("persona_below_min_support", ns=persona_key(ns), support=source_count)
            return None, source_count, 0

        slots = self._aggregator.aggregate(evidence, now=now)
        if not slots and existing is None:
            # Every candidate decayed (or none was tagged) and there is nothing to supersede.
            # An empty portrait is not a portrait — do not mint version 1 for it.
            return None, source_count, 0

        brief = await self._brief(ns, slots, existing)
        # The AUTHORIZED ns — never a namespace read off returned data. `evidence[0].item.
        # namespace` would let one mis-partitioned row redirect the whole write into another
        # tenant's key (this repo has shipped two partition-key defects already: 0545119,
        # 1dd023c; CANONICAL §1 rule 5 / CLAUDE.md rule 4).
        profile = PersonaProfile(
            namespace=ns,
            slots=slots,
            overall_brief=brief,
            brief_etag=_etag(brief),
            version=1 if existing is None else existing.version + 1,
            rebuilt_at=now,
            source_memory_count=source_count,
        )
        await self._repo.upsert(profile)
        changed = slots_changed({} if existing is None else existing.slots, slots)
        await self._publish(profile, changed)
        return profile, source_count, changed

    async def _brief(
        self,
        ns: Namespace,
        slots: Mapping[PersonaSlot, SlotValue],
        existing: PersonaProfile | None,
    ) -> str:
        """Stage 2, or the ONE named degraded path (spec §2.3; DEV-STANDARDS rule 8).

        Never falls back to a weaker model and never guesses a portrait: on any failure the
        structured slots stand and the PREVIOUS brief is carried forward unchanged
        (supersession-not-deletion, §3.3 line 168). A first build with no model yields an empty
        brief — the slots are still worth persisting, and an empty brief is honestly empty rather
        than fabricated.

        OWNER CALL, FLAGGED: the spec states neither policy for a missing model. The house has two
        precedents — refuse loudly (``LlmNotConfiguredError``, ``surface/facade.py:350``) or
        degrade with a named reason (``lifecycle/conflict.py:434-461``). Degrading is chosen
        because refusing would mean NO persona at all on a zero-API-key FULL-LOCAL install, which
        the boundary rule forbids.
        """
        carried = "" if existing is None else existing.overall_brief
        if self._synthesizer is None:
            await self._degrade(ns, detail="synthesizer_absent")
            return carried
        try:
            overall, _strategy = await self._synthesizer.synthesize(slots)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # model unavailable, malformed reply, cap violation — no guess
            await self._degrade(ns, detail=type(exc).__name__)
            return carried
        return overall

    # ------------------------------------------------------ stage 1, incremental (§2.4 line 121)
    def note_promoted(self, ev: MemoryPromoted) -> bool:
        """Queue a promoted memory for the next :meth:`refresh`. Returns whether it was queued.

        **SYNCHRONOUS, and that is the whole design.** ``MemoryPromoted`` is published INLINE on
        the ingest capture path through a bus that awaits every handler and propagates their
        exceptions to the publisher, so anything reachable from this method runs inside the user's
        ``remember()`` call. A plain ``def`` cannot ``await`` a store, a bus or a model, so no
        amount of later editing can put persona I/O — or the classifier hiding behind
        ``PersonaEvidenceReader`` (spec line 103) — on that stack without changing this signature
        and failing ``test_persona_boundary_unit``. It is the same argument ``TraitAggregator``
        makes for being sync (``aggregator.py:70-82``).

        **Total.** It calls no collaborator: no repo, no bus, no clock, no metric sink, no tracer,
        no guard. There is nothing here that can raise into a user's capture. (No
        ``assert_scope``, deliberately: this is an ENGINE-INTERNAL event the engine itself just
        published for a namespace it already authorized, not a caller request, and there is no
        caller ``ClientScope`` on a bus handler to check. The authorization that matters happens
        on :meth:`refresh`, which is the method that actually reads and writes a partition.)

        Ignores an event this subsystem has no business in — a promotion into STM (a TTL window,
        not a durable trait) or a SHARED partition (§0 line 53: a room has no personality).

        Bounded on both axes from ``PersonaSettings`` (DEV-STANDARDS rule 3: no unbounded
        in-memory growth). Dropping on overflow is LOSSLESS in the limit: :meth:`rebuild` reads
        the whole evidence set, so a dropped id is picked up at the next full sleep-time build;
        the drop is still counted and surfaces in the next :meth:`refresh` audit row.
        """
        if not self._settings.enabled or ev.to not in _INCREMENTAL_TIERS:
            return False
        ns = ev.namespace
        if ns.visibility is not Visibility.PRIVATE:
            return False
        key = persona_key(ns)
        queue = self._pending.get(key)
        if queue is None:
            if len(self._pending) >= self._settings.max_pending_keys:
                self._dropped[key] = self._dropped.get(key, 0) + 1
                return False
            queue = self._pending[key] = set()
        if ev.id not in queue and len(queue) >= self._settings.max_pending_ids:
            self._dropped[key] = self._dropped.get(key, 0) + 1
            return False
        queue.add(ev.id)
        return True

    async def refresh(self, scope: ClientScope, ns: Namespace) -> PersonaProfile | None:
        """Fold the memories :meth:`note_promoted` queued into the stored slots — **no LLM**
        (spec §2.4 line 121: *"a cheap slot-upsert, no LLM"*).

        Sleep-time, beside :meth:`rebuild`: this is where the deferred incremental work actually
        runs, so a slow evidence read or a ``PersonaVersionConflictError`` costs a maintenance
        tick, never a user's capture.

        Does nothing when no profile exists yet: the FIRST build is :meth:`rebuild`'s job, gated
        by ``min_support`` (§3.3 line 165), and an incremental upsert must never sneak past it.
        The queue is drained whether or not the upsert lands — a lost id is picked up by the next
        full :meth:`rebuild`, which reads the whole evidence set anyway.
        """
        self._scope_guard.assert_scope(scope, ns, _OP_REFRESH)
        profile, evidence_count, changed = await self._timed(
            _OP_REFRESH, ns, lambda: self._refresh(ns)
        )
        if changed or evidence_count:
            self._record(_OP_REFRESH, ns, profile, evidence_count=evidence_count, changed=changed)
        return profile

    async def _refresh(self, ns: Namespace) -> tuple[PersonaProfile | None, int, int]:
        if not self._settings.enabled:
            return None, 0, 0
        assert_private(ns, _OP_REFRESH)
        ids = self._drain_pending(ns)
        if not ids:
            return None, 0, 0
        existing = await self._repo.get(ns)
        if existing is None:
            return None, 0, 0
        evidence = self._own_partition_only(ns, await self._evidence.evidence_for_ids(ns, ids))
        if not evidence:
            return existing, 0, 0

        # The SAME deterministic aggregator as the full rebuild — one scoring rule, not a second
        # incremental one that could disagree with it (DEV-STANDARDS rule 6, DRY).
        candidates = self._aggregator.aggregate(evidence, now=self._clock.now())
        merged = _merge_slots(existing.slots, candidates)
        changed = slots_changed(existing.slots, merged)
        count = _distinct_items(evidence)
        if not changed:
            return existing, count, 0

        profile = existing.model_copy(
            update={
                "slots": merged,
                "version": existing.version + 1,
                # `overall_brief`/`brief_etag` are UNTOUCHED: re-writing the portrait is Stage 2's
                # job and Stage 2 needs a model (§2.3). `source_memory_count` describes the last
                # full BUILD (§3.1) — an incremental upsert is not a build.
            }
        )
        await self._repo.upsert(profile)
        await self._publish(profile, changed)
        return profile, count, changed

    # ------------------------------------------------------ right-to-be-forgotten (line 169)
    async def forget(self, scope: ClientScope, ns: Namespace) -> bool:
        """Erase this user's persona record. ``True`` if one existed (spec line 169).

        Spec line 169 promises persona "has no independent durability beyond its key" — but the
        shipped store is a standalone keyed record, not a view over the memory partition, so
        without this verb a user could delete 100% of their memories and the engine would keep,
        and keep serving, an LLM paragraph describing their personality: :meth:`rebuild` on an
        empty evidence set carries the PREVIOUS brief forward unchanged (``_brief``'s
        supersession rule), which is right for a failed synthesis and wrong for an erasure.
        Governance calls this when it drops the user's PRIVATE partition.

        Also clears anything :meth:`note_promoted` had queued for the user, so a pending id
        cannot resurrect a slot after the erase.
        """
        self._scope_guard.assert_scope(scope, ns, _OP_FORGET)
        erased, _count, _changed = await self._timed(_OP_FORGET, ns, lambda: self._forget(ns))
        self._record(_OP_FORGET, ns, None, evidence_count=0, changed=0, erased=erased)
        return erased

    async def _forget(self, ns: Namespace) -> tuple[bool, int, int]:
        assert_private(ns, _OP_FORGET)
        self._drain_pending(ns)
        erased = await self._repo.delete(ns)
        if erased and self._bus is not None:
            await self._bus.publish(
                PersonaUpdated(
                    namespace=ns,
                    version=_ERASED_VERSION,
                    slots_changed=0,
                    brief_etag="",
                )
            )
        return erased, 0, 0

    # ---------------------------------------------------------------------------- side seams
    def _drain_pending(self, ns: Namespace) -> frozenset[str]:
        return frozenset(self._pending.pop(persona_key(ns), set()))

    async def _timed[T](
        self, operation: str, ns: Namespace, work: Callable[[], Coroutine[Any, Any, T]]
    ) -> T:
        """Span + latency + error metric around one operation. One implementation, so a new verb
        cannot ship with half the observability of the others (DEV-STANDARDS rule 6).

        Takes a FACTORY, not a coroutine: the span attributes are derived from ``ns``, which can
        itself be refused (a SHARED namespace), and an eagerly-built coroutine would then be
        garbage-collected un-awaited.
        """
        started = time.perf_counter()
        with self._tracer.span(operation, attributes=sanitize_labels({"ns": persona_key(ns)})):
            try:
                return await work()
            except asyncio.CancelledError:
                raise
            except BaseException:
                self._metrics.inc(_ERROR_METRIC, labels={"operation": operation})
                raise
            finally:
                self._metrics.observe(
                    _LATENCY_METRIC,
                    time.perf_counter() - started,
                    labels={"operation": operation},
                )

    def _own_partition_only(
        self, ns: Namespace, evidence: Sequence[PersonaEvidence]
    ) -> Sequence[PersonaEvidence]:
        """Refuse evidence from outside the AUTHORIZED user's own partition (§1 line 68).

        Spec line 68 says persona "reads only the user's own PRIVATE ``to_prefix()`` key-space —
        enforced by ``TenancyGuard.assert_scope`` on every read". ``PersonaEvidenceReader`` is a
        port: its docstring can *oblige* an adapter to scope its read, but an obligation on an
        unbuilt port is not enforcement, and the cost of a reader that gets it wrong is that
        another user's memory CONTENT becomes this user's slot values and their memory IDS become
        this user's persisted ``support_ids``. ``ports/device.py:14-16`` already rejected exactly
        this ("re-invitable by the next caller") — so the check lives here, above every adapter.

        Compared on the SESSION-SPANNING :func:`~mu_engine.services.persona.store.persona_key`,
        not on ``to_prefix()``: a user's own memory from an earlier session is legitimate persona
        evidence (ADR 0030 — a PRIVATE session is a provenance stamp, not an isolation boundary),
        and that is the same grain the profile itself is keyed on.

        RAISES rather than filters (DEV-STANDARDS rule 8, never a silent filter): a reader
        returning a foreign row is a defect in that adapter, and silently dropping the rows would
        leave it undetected while the persona quietly built from a shrinking, arbitrary subset.
        Non-enumerating, like every other tenancy refusal here.
        """
        expected = persona_key(ns)
        for ev in evidence:
            item_ns = ev.item.namespace
            if item_ns.visibility is not Visibility.PRIVATE or persona_key(item_ns) != expected:
                # Content-free: two namespace prefixes and an operation name, never a slot value.
                _log.error(
                    "persona_evidence_cross_partition",
                    ns=expected,
                    offending_ns=item_ns.to_prefix(),
                )
                raise NamespaceIsolationError("not found")
        return evidence

    async def _publish(self, profile: PersonaProfile, changed: int) -> None:
        """``PersonaUpdated`` — strictly content-free (spec §4 line 179): namespace, version, a
        COUNT of changed slots, and the brief's etag. No slot value, no brief text. A consumer
        that needs the brief calls ``PersonaRepository.load_brief(ns)`` by key."""
        if self._bus is None:
            return
        await self._bus.publish(
            PersonaUpdated(
                namespace=profile.namespace,
                version=profile.version,
                slots_changed=changed,
                brief_etag=profile.brief_etag,
            )
        )

    async def _degrade(self, ns: Namespace, *, detail: str) -> None:
        _log.warning(
            "persona_synthesis_degraded",
            ns=persona_key(ns),
            reason=_DEGRADE_REASON,
            detail=detail,
        )
        if self._bus is None:
            return
        await self._bus.publish(
            DegradedModeEntered(
                component=_DEGRADE_COMPONENT,
                mode=_DEGRADE_MODE,
                reason=_DEGRADE_REASON,
                detail=detail,
            )
        )

    def _record(
        self,
        operation: str,
        ns: Namespace,
        profile: PersonaProfile | None,
        *,
        evidence_count: int,
        changed: int,
        erased: bool | None = None,
    ) -> None:
        """COUNTS ONLY, and content-free BY CONSTRUCTION rather than by promise.

        Persona is inferred from a user's private data, so it is treated as at least as sensitive
        as memory content (CLAUDE.md rule 3): a slot value or brief must never reach an audit row,
        a span attribute or a metric label. Every field routed here goes through the engine's own
        content-free guards — :class:`SafeTraceFields` for the counts and
        :func:`sanitize_label_value` for the scalars — which REJECT free text (``observability.py``
        ``_SAFE_VALUE``: no whitespace, bounded length). A future edit that put a brief or a slot
        value in this call raises at the boundary instead of quietly shipping it to the sink.
        """
        fields = SafeTraceFields(
            counts={
                "evidence": evidence_count,
                "slots": 0 if profile is None else len(profile.slots),
                "slots_changed": changed,
                "version": 0 if profile is None else profile.version,
                "dropped": self._dropped.pop(persona_key(ns), 0),
            }
        )
        outcome = (
            ("erased" if erased else "absent")
            if erased is not None
            else ("ok" if profile is not None else "skipped")
        )
        self._audit.record(
            TraceScope(correlation_id=persona_key(ns)),
            operation=sanitize_label_value(operation),
            outcome=sanitize_label_value(outcome),
            visibility=sanitize_label_value(ns.visibility.value),
            counts=fields.counts,
        )


def _distinct_items(evidence: Sequence[PersonaEvidence]) -> int:
    """``PersonaProfile.source_memory_count`` — MEMORIES, not evidence rows. One memory can
    evidence several slots, and counting rows would inflate the §3.3 ``min_support`` gate into
    passing on a single utterance that happened to mention three things."""
    return len({ev.item.id for ev in evidence})


def _merge_slots(
    existing: Mapping[PersonaSlot, SlotValue], candidates: Mapping[PersonaSlot, SlotValue]
) -> dict[PersonaSlot, SlotValue]:
    """§3.3 line 168 supersession: *"a changed preference supersedes the old slot value
    (higher-confidence, newer ``SlotValue`` wins)"*.

    Deliberately does NOT drop existing slots the candidate set is silent about: the incremental
    path sees only the queued memories, so absence there is no evidence of decay. Dropping a
    decayed slot is the full rebuild's job, where the whole evidence set is in hand (§3.3 line
    167).
    """
    merged = dict(existing)
    for slot, candidate in candidates.items():
        current = merged.get(slot)
        if current is None or (candidate.confidence, candidate.updated_at) > (
            current.confidence,
            current.updated_at,
        ):
            merged[slot] = candidate
    return merged


def _etag(brief: str) -> str:
    """``sha256(overall_brief)`` — the inject skip-if-unchanged tag (spec §3.1 line 136)."""
    return hashlib.sha256(brief.encode("utf-8")).hexdigest()
