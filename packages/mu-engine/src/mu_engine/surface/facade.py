"""``SurfaceFacade`` — the ONE canonical verb surface over the embedded LOCAL composition root
(sdk-engine-server-design.md §2.3/§2.5, build-queue §13 item 1).

Placement: **mu-engine** (design §8 diagram, §13 item 1) — so both ``mu-local`` (embedded,
``mu_local.local_memory.LocalMemory``) and Stage-C's ``mu-engine-server`` can import ONE facade
class without either depending on the other's package. This module therefore must NOT import
``mu_local`` (import-linter ``core-layers``/``mu-local-layers`` contracts, ``.importlinter:13-27``
— ``mu_engine`` sits BELOW ``mu_local`` in the layer stack and may only import ``mu_contracts``).

WHY this wraps ``LocalContainer`` (not ``LocalMemory``) even though ``mu_local.local_memory.
LocalMemory`` already implements this exact verb surface (``add``/``recall``/``get``/
``consolidate``/``context``/``ask``, ``mu-local/local_memory.py:74-373``): ``LocalMemory`` is a
``mu_local``-owned class this package cannot import. Rather than leave a dangling dependency, this
module RE-DERIVES the identical orchestration ``LocalMemory`` already performs (namespace/scope
construction -> one application-service call -> DTO projection) directly over the composition
root's already-``mu_engine``-native services (``IngestService``/``RecallService``/
``DistillPipeline``/``StmTierRepository``/``ManagerModeGate``, all defined in THIS package). The
composition root (:class:`LocalContainerLike`) is taken as a constructor argument — dependency
injected — never built here: ``mu-local`` builds the real ``mu_local.composition.LocalContainer``
and hands it to this facade (or to ``mu-engine-server``'s own composition, Stage C). A real
``LocalContainer`` instance satisfies :class:`LocalContainerLike` structurally with ZERO adapter
code (its ``ingest``/``distill``/``stm``/``recall``/``mode_gate``/``llm`` attributes are already
exactly these types — ``mu-local/composition.py:230-378``).

DTOs (build-plan §4 C2 item (a) — the A→B→C handoff, superseding plan ruling 2's original
"PROVISIONAL embedded DTOs" call): build-queue §13 item 3 / the A4 decision task landed as
``SDK-BUILD-DECISIONS.md`` Decision B while this facade still returned the raw engine-native
shapes it started with (``IngestResult``, engine ``RecallResult``, ``MemoryItem``,
``DistillReport``) — the ONE canonical DTO per verb it names lives in ``mu_contracts.contracts.*``
(``views.MemoryWriteResult``/``views.ConsolidateView``/``views.ContextView``,
``recall.RecallResult``, ``memory.MemoryResponse``), the SAME shapes Stage B's B1 already re-
pointed ``mu_local.local_memory.LocalMemory`` at (``mu-local/local_memory.py``, module docstring
"UNIFIED VERB SURFACE"). This module now performs the identical engine-native -> canonical mapping
B1 performs, independently re-derived field-for-field against B1's own call sites (never importing
``mu_local.local_memory`` itself — the import boundary above still holds; only the RESULT shapes,
themselves ``mu_contracts``-homed, are shared) — the "MIRROR how B1 does this" instruction, applied
package-locally so ``mu-engine-server``'s HTTP layer (Stage C, C2) can serialize ONE wire DTO
family regardless of whether the caller went through ``LocalMemory`` or this facade. ``add``/
``recall``/``get``/``consolidate`` all changed their RETURN TYPE only — their accepted keyword
arguments are unchanged from Stage A (no shared-plane ``visibility``/``subject``/``predicate``/
``object`` plane-gating superset was added here; that is a distinct, not-yet-scoped signature
question, §2.5's "Which classes change — both" paragraph, separate from this DTO re-annotation).

``build_context`` is likewise no longer a named 501: the underlying op (recall + deterministic
render) was always trivial to re-derive (its own prior docstring said so) — only its return
shape (``mu_local.views.ContextView``) was blocked on Decision B. Now that ``ContextView`` lives in
``mu_contracts.contracts.views``, this module builds it directly, PORTing
``mu_local.local_memory._render_context``'s one-bullet-per-hit assembly (``mu-local/
local_memory.py:503-509``) since that helper is private to a package this module cannot import.

``promote``/``demote``/``update``/``delete`` are now REAL (build-queue §13 item 5 — the honest
501s are retired): each is a THIN targeted verb over the already-real lifecycle + bi-temporal
machinery, never a second copy of it. ``promote`` reuses ``PromotionService``'s own copy-on-write
STM->MTM shape / ``DistillPipeline.distill`` MTM->LTM leg; ``demote`` reuses
``DemotionService._demote_one``'s write-ahead-then-remove sequence; ``update`` = SUPERSEDE via
``MtmTierRepository``/``GraphStorePort.invalidate`` (the same invalidate the conflict path uses);
``delete`` = invalidate-don't-delete via the new ``expire`` soft-delete primitive (state=expired +
invalid_at, kept in bi-temporal history — never a hard ``DETACH DELETE`` of active data). They
LOCATE the target item via the new point-get primitives ``StmTierRepository.get`` (pre-existing) /
``MtmTierRepository.get`` / ``GraphStorePort.get_fact``, and guard a nonexistent id with
:class:`MemoryNotFoundError` (HTTP 404) and an invalid ``to_tier`` with ``ValueError`` (HTTP 400)
— honest errors, never a silent no-op / fake success.

``share`` (build-queue §13 item 3) still has no engine-side verb to delegate to — it still raises
:class:`SurfaceVerbNotImplementedError`, imported from its canonical home
``mu_contracts.domain.errors`` (CO-3), which Stage C's HTTP layer maps to HTTP 501. This is a
deliberate, NAMED gap (DEV-STANDARDS: never a silent no-op / fake success), not an oversight.
"""

from __future__ import annotations

import uuid
from typing import Any, NoReturn, Protocol

from mu_contracts.contracts.memory import MemoryResponse
from mu_contracts.contracts.recall import RecallChannels as CanonicalRecallChannels
from mu_contracts.contracts.recall import RecallItemView as CanonicalRecallItemView
from mu_contracts.contracts.recall import RecallResult as CanonicalRecallResult
from mu_contracts.contracts.views import (
    ConsolidateView,
    ContextView,
    MemoryVerbResult,
    MemoryWriteResult,
)
from mu_contracts.domain.errors import (
    LlmNotConfiguredError,
    MemoryNotFoundError,
    SurfaceVerbNotImplementedError,
)
from mu_contracts.domain.events import MemoryDemoted, MemoryPromoted
from mu_contracts.domain.model.memory import State as CanonicalState
from mu_contracts.domain.model.memory import Tier as CanonicalTier
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.mode_gate import ManagerModeGate
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.distill import (
    DistillActionKind,
    DistillPipeline,
    DistillReport,
    EventPublisher,
)
from mu_engine.platform.clock import SystemClock
from mu_engine.providers._contracts import Message, MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.providers.model_router import ModelRouter
from mu_engine.services.ingest import IngestResult, IngestService
from mu_engine.services.recall.dto import RecallChannels as _EngineRecallChannels
from mu_engine.services.recall.dto import RecallQuery
from mu_engine.services.recall.dto import RecallResult as _EngineRecallResult
from mu_engine.services.recall.service import RecallService
from mu_engine.storage.domain.memory import MemoryItem, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.ports import GraphStorePort, MtmTierRepository, StmTierRepository

__all__ = [
    "LlmNotConfiguredError",
    "LocalContainerLike",
    "MemoryNotFoundError",
    "SurfaceFacade",
    "SurfaceVerbNotImplementedError",
]

_DEFAULT_USER = "default"
_DEFAULT_SESSION = "default"
# PORT of the reference SLM integration test's ANSWER-task system prompt (parity with mu-local's
# copy, mu-local/local_memory.py:71 / mu-engine/tests/pipelines/test_distill_llm_slm_int.py:393-
# 396) — a named module constant, never an inline literal at the call site (DEV-STANDARDS rule 3).
_ASK_SYSTEM_PROMPT = "Answer the question using ONLY the given facts. Be concise."
# ``add()``'s "caller expressed no opinion" fallback — mirrors ``LocalMemory``'s identical module
# constant (``mu-local/local_memory.py``, REMEDIATION Rank 2 / A6 fix): READ off
# ``IngestActivity.importance``'s own field default rather than a second hardcoded ``0.5`` literal
# (DEV-STANDARDS rule 3), and a plain ``float`` so mypy-strict can check the ``importance=`` kwarg
# without an ``**dict`` unpack (which mypy cannot verify against a ``BaseModel``'s field types).
_DEFAULT_IMPORTANCE: float = IngestActivity.model_fields["importance"].default


class LocalContainerLike(Protocol):
    """The structural shape :class:`SurfaceFacade` needs from its injected composition root.

    Deliberately NOT ``mu_local.composition.LocalContainer`` (this package cannot import
    ``mu_local`` — module docstring / design §8 boundary rule). Every attribute named below is
    already exactly this type on the real ``LocalContainer`` (``mu-local/composition.py:230-378``:
    ``self.ingest = IngestService(...)``, ``self.distill = DistillPipeline(...)``, ``self.stm =
    STORE_REGISTRY.build("kv", ...)`` typed ``StmTierRepository``, ``self.recall =
    RecallService(...)``, ``self.mode_gate = ManagerModeGate(...)``, ``self.llm: ModelRouter |
    None``) — so a real ``LocalContainer`` instance satisfies this Protocol with ZERO adapter code,
    and ``mu-engine-server``'s own composition root (Stage C) can satisfy it just as directly.
    """

    ingest: IngestService
    distill: DistillPipeline
    stm: StmTierRepository
    # ``mtm``/``ltm`` are needed by the TARGETED lifecycle verbs (``promote`` MTM->LTM / ``demote``
    # MTM->STM / ``update`` / ``delete``) to LOCATE a memory in its current tier and run the REAL
    # tier op on that one item — the sweep-oriented ``PromotionService``/``DemotionService`` take a
    # caller-supplied window and never resolve a bare id. Already exactly these types on the real
    # ``LocalContainer`` (``mu-local/composition.py``: ``self.mtm = STORE_REGISTRY.build("vector",
    # ...)`` typed ``MtmTierRepository``, ``self.ltm = STORE_REGISTRY.build("graph", ...)`` typed
    # ``GraphStorePort``) — ZERO adapter code.
    mtm: MtmTierRepository
    ltm: GraphStorePort
    recall: RecallService
    mode_gate: ManagerModeGate
    llm: ModelRouter | None

    # The SAME real ``InprocBus`` ``ingest``/``distill`` publish onto (``LocalContainer.bus`` /
    # ``EngineContainer.bus`` — a read-only PROPERTY returning ``EventBusPort``, structurally an
    # ``EventPublisher``). Declared here as a read-only ``@property`` (not a settable attribute) so
    # a container exposing ``bus`` as a property satisfies this Protocol (a plain attribute
    # declaration would demand a settable, invariantly-typed field the property cannot match). The
    # targeted promote/demote verbs publish ``MemoryPromoted``/``MemoryDemoted`` onto it exactly as
    # ``PromotionService``/``DemotionService`` do for the automatic paths. ``None`` (heuristic/
    # unwired) simply skips emission — never a hard failure.
    @property
    def bus(self) -> EventPublisher | None: ...


class SurfaceFacade:
    """One canonical verb surface (design §2.5) over an injected :class:`LocalContainerLike`.

    ``workspace``/``namespace`` fix the η.workspace/η.org slots at construction — identical
    discipline to ``LocalMemory`` (single-tenant by construction, ``mu-local/local_memory.py:74-
    90``); each call's ``user``/``session`` populate the remaining η slots. This facade builds NO
    stores/services of its own — every verb delegates to exactly one method on the injected
    container, never a second, independently-constructed instance (DEV-STANDARDS rule 9).
    """

    def __init__(
        self,
        container: LocalContainerLike,
        *,
        workspace: str = "local",
        namespace: str = "default",
        clock: Clock | None = None,
    ) -> None:
        self._container = container
        self._workspace = workspace
        self._org = namespace
        # UTC clock for the ``at=`` bi-temporal stamps ``update``/``delete`` write (supersede /
        # soft-delete ``invalid_at``). Defaults to the SAME ``SystemClock`` the composition root
        # threads into every service (``mu-local/composition.py``) — never a bare ``datetime.now``
        # call at a verb site (MAJOR-3: bi-temporal stamps must be UTC-correct).
        self._clock: Clock = clock or SystemClock()

    # ------------------------------------------------------------------------------------- write
    async def add(
        self,
        content: str | dict[str, Any] | list[dict[str, Any]],
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        # COMMON field (design §2.5 superset, canonical wire ``AddRequest.importance_score``,
        # ``mu_contracts/contracts/requests.py:195``) — never plane-gated. Mirrors
        # ``LocalMemory.add``'s own ``importance_score`` param (``mu-local/local_memory.py``,
        # REMEDIATION Rank 2 / conformance A6 fix) field-for-field: ``None`` (the default) omits
        # ``importance=`` from the constructed ``IngestActivity`` so its own field default (0.5)
        # applies; a real value threads through to ``DeterministicPromoteStage``'s
        # ``importance >= IngestSettings.importance_promote`` gate
        # (``mu-engine/pipelines/concrete/ingest.py:230``).
        importance_score: float | None = None,
    ) -> MemoryWriteResult:
        """Ingest one activity (STM durable -> deterministic STM->MTM promote, GATED on importance
        — REMEDIATION Rank 2 / conformance A6 fix). Mirrors ``LocalMemory.add`` (``mu-local/
        local_memory.py``) field-for-field, returning the canonical
        :class:`~mu_contracts.contracts.views.MemoryWriteResult` receipt (module docstring,
        Decision B) — ``namespace``/``events_emitted`` populated from the resolved η and the
        engine's own :class:`IngestResult` respectively, zero extra I/O.

        ``promote`` is NEVER hardcoded ``True`` here (the A6 defect this fix removes: it
        short-circuited ``DeterministicPromoteStage``'s own importance/mention gate, so EVERY add
        promoted regardless of salience). ``IngestActivity(promote=False, importance=<threaded
        importance_score>)`` lets the deterministic stage decide — ``explicit promote OR
        importance>=threshold`` — matching ``LocalMemory.add``'s identical fix exactly. No
        caller-facing "force promote" override exists on this verb today (the canonical
        ``AddRequest`` names no such field)."""
        ns = self._ns(user, session)
        # ``None`` means "caller expressed no opinion" -> falls to IngestActivity's own field
        # default (module constant above) — mirrors ``LocalMemory.add``'s identical fix.
        importance = importance_score if importance_score is not None else _DEFAULT_IMPORTANCE
        last: IngestResult | None = None
        for message in _normalize_messages(content):
            activity = IngestActivity(
                namespace=ns,
                host=_host(),
                session_offset=_fresh_offset(),
                kind="user_message",
                text=message["content"],
                importance=importance,
            )
            last = await self._container.ingest.remember(activity)
        if last is None:  # empty message list — fail loud, never a silent no-op
            raise ValueError("add() received no content to remember")
        return MemoryWriteResult(
            memory_id=last.memory_id,
            content_hash=last.content_hash,
            promoted=last.promoted,
            tiers_written=last.tiers_written,
            namespace=ns.to_prefix(),
            events_emitted=last.events_emitted,
        )

    async def consolidate(
        self,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = 50,
    ) -> ConsolidateView:
        """MTM->LTM consolidation (DISTILL). Mirrors ``LocalMemory.consolidate`` (``mu-local/
        local_memory.py:176-215``) — the SAME engine-side ``ManagerModeGate.assert_manual_allowed``
        pre-check (ADR 0031), so a MANAGED namespace refuses this manual verb loud here exactly as
        it does through ``LocalMemory``, never bypassable via this second surface. Returns the
        canonical :class:`~mu_contracts.contracts.views.ConsolidateView` receipt (Decision B) —
        ``noop`` counts :class:`DistillActionKind.NOOP` actions exactly as B1 does, never silently
        dropped."""
        ns = self._ns(user, session)
        self._container.mode_gate.assert_manual_allowed(ns, "consolidate")
        recent = await self._container.stm.recent(ns, limit=limit)
        window = [scored.item for scored in recent]
        report: DistillReport = await self._container.distill.distill(ns, window)
        return ConsolidateView(
            facts_extracted=report.facts_extracted,
            added=report.added,
            superseded=report.superseded,
            noop=sum(a.kind is DistillActionKind.NOOP for a in report.actions),
        )

    # -------------------------------------------------------------------------------------- read
    async def recall(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        tier: MemoryTier | None = None,
        limit: int = 10,
    ) -> CanonicalRecallResult:
        """Federate-live RANKED recall. Mirrors ``LocalMemory.recall`` (``mu-local/
        local_memory.py:219-256``), returning the canonical
        :class:`~mu_contracts.contracts.recall.RecallResult` (Decision B) — the un-collapsed
        engine result, mapped field-for-field via :func:`_to_canonical_recall_result` (mirrors
        ``mu_local.local_memory._to_recall_result``)."""
        ns = self._ns(user, session)
        scope = self._scope(user, session)
        q = RecallQuery(namespace=ns, text=query, limit=limit, channels=_channels_for_tier(tier))
        result = await self._container.recall.recall(scope, q)
        return _to_canonical_recall_result(result)

    async def get(
        self,
        memory_id: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryResponse | None:
        """Point-get one memory by id from the caller's STM partition (``None`` if absent).
        Mirrors ``LocalMemory.get`` (``mu-local/local_memory.py:277-297``) — same phase-0 STM-only
        narrowing (MTM/LTM adapters expose no point-get yet). Returns the canonical, FROZEN-wire-
        schema :class:`~mu_contracts.contracts.memory.MemoryResponse` (Decision B) via
        :func:`_to_memory_response` — never the engine-native :class:`MemoryItem` domain object."""
        ns = self._ns(user, session)
        item = await self._container.stm.get(ns, memory_id)
        if item is None:
            return None
        return _to_memory_response(item)

    async def ask(
        self,
        question: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = 10,
    ) -> str:
        """Synthesise an answer over recalled context via the configured LLM's ANSWER task.
        Mirrors ``LocalMemory.ask`` (``mu-local/local_memory.py:238-264``) — heuristic mode
        (``container.llm is None``) refuses loudly via this module's own
        :class:`LlmNotConfiguredError` (module docstring)."""
        if self._container.llm is None:
            raise LlmNotConfiguredError(
                "ask() requires a configured LLM; the injected container is in heuristic mode "
                "(llm=None). Use recall()/get() for retrieval without synthesis."
            )
        result = await self.recall(question, user=user, session=session, limit=limit)
        facts_text = _render_facts(result)
        messages = [
            Message(role=MessageRole.SYSTEM, content=_ASK_SYSTEM_PROMPT),
            Message(
                role=MessageRole.USER,
                content=f"Facts:\n{facts_text}\n\nQuestion: {question}",
            ),
        ]
        completion = await self._container.llm.generate(Task.ANSWER, messages)
        return completion.text

    async def build_context(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = 10,
        max_chars: int | None = None,
    ) -> ContextView:
        """Private-plane context-window verb (design §2.5 REVIEW-2 FIX 1) — the portable
        ``MemoryClient`` twin of ``LocalMemory.context`` (``mu-local/local_memory.py:329-350``).

        Wired to the real op now (build-plan §4 C2 item (a) — no longer 501): recall, then
        deterministic assembly via :func:`_render_context`, a PORT of ``mu_local.local_memory.
        _render_context`` (``mu-local/local_memory.py:503-509`` — a private, underscore-prefixed
        helper in a package this module cannot import, so duplicated rather than imported, exactly
        as :func:`_normalize_messages`/:func:`_channels_for_tier` already are below). Returns the
        canonical :class:`~mu_contracts.contracts.views.ContextView`, now that it is homed in
        ``mu_contracts`` (Decision B) rather than ``mu_local.views`` — the blocker this verb's prior
        501 named is resolved."""
        result = await self.recall(query, user=user, session=session, limit=limit)
        text = _render_context(result.items, max_chars=max_chars)
        degraded = result.degraded.value if result.degraded is not None else None
        return ContextView(text=text, items=result.items, degraded=degraded)

    async def promote(
        self,
        memory_id: str,
        *,
        to_tier: str,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryVerbResult:
        """TARGETED single-memory promotion — run the REAL promotion path on ONE resident item,
        moving it up one tier (``to_tier="mtm"``: STM->MTM, or ``to_tier="ltm"``: MTM->LTM). No new
        memory logic: this LOCATES the item in its source tier and reuses the exact ops the
        automatic :class:`~mu_engine.lifecycle.promotion.PromotionService` uses — never a second
        copy of promotion logic (DEV-STANDARDS rule 6).

        * **STM->MTM** (``to_tier="mtm"``): ``stm.get`` -> copy-on-write to MTM -> ``mtm.upsert`` —
          byte-for-byte the shape ``PromotionService._promote_to_mtm`` (``promotion.py:414-432``) /
          ``DeterministicPromoteStage`` (``ingest.py:227-259``) use: ``item.model_copy(deep=True)``
          with ``tier=MTM``, ``state=ACTIVE``, refreshed ``updated_at``, then ``mtm.upsert``. The
          STM copy is left in place (its Redis TTL is its own eviction), exactly as those paths do.
        * **MTM->LTM** (``to_tier="ltm"``): ``mtm.get`` -> hand the ONE item to
          ``DistillPipeline.distill`` (the SAME real MTM->LTM leg
          ``PromotionService.sweep_mtm_to_ltm``
          delegates to, ``promotion.py:227-263``) — DISTILL extracts + writes the LTM fact(s) and
          emits its own ``MemoryPromoted(frm=MTM,to=LTM)``, so this method does not double-emit for
          that leg.

        A nonexistent id (absent from the source tier) raises :class:`MemoryNotFoundError` (HTTP
        404); an invalid ``to_tier`` raises ``ValueError`` (HTTP 400) — honest errors, never a
        silent no-op (the deliberate 501 this method used to be is now a REAL op, build-queue §13
        item 5)."""
        ns = self._ns(user, session)
        target = _parse_tier(to_tier)
        if target is MemoryTier.MTM:
            item = await self._container.stm.get(ns, memory_id)
            if item is None:
                raise MemoryNotFoundError(memory_id, reason="not resident in STM (source tier)")
            now = self._clock.now()
            promoted = item.model_copy(deep=True)
            promoted.tier = MemoryTier.MTM
            promoted.state = MemoryState.ACTIVE
            promoted.updated_at = now
            await self._container.mtm.upsert(promoted)
            events = await self._publish(
                MemoryPromoted(
                    namespace=ns,
                    id=memory_id,
                    frm=CanonicalTier.STM,
                    to=CanonicalTier.MTM,
                    reason="targeted_promote",
                )
            )
            return MemoryVerbResult(
                memory_id=memory_id,
                verb="promote",
                from_tier=MemoryTier.STM.value,
                to_tier=MemoryTier.MTM.value,
                tiers_affected=(MemoryTier.MTM.value,),
                events_emitted=events,
            )
        if target is MemoryTier.LTM:
            item = await self._container.mtm.get(ns, memory_id)
            if item is None:
                raise MemoryNotFoundError(memory_id, reason="not resident in MTM (source tier)")
            # Real MTM->LTM promote = hand this ONE item to DISTILL (the exact leg
            # PromotionService.sweep_mtm_to_ltm delegates to). DISTILL emits its own MemoryPromoted.
            await self._container.distill.distill(ns, [item])
            return MemoryVerbResult(
                memory_id=memory_id,
                verb="promote",
                from_tier=MemoryTier.MTM.value,
                to_tier=MemoryTier.LTM.value,
                tiers_affected=(MemoryTier.LTM.value,),
            )
        raise ValueError(
            f"promote: invalid to_tier {to_tier!r} — promotion moves STM->MTM (to_tier='mtm') or "
            "MTM->LTM (to_tier='ltm'); 'stm' is not a promotion target (use demote for MTM->STM)"
        )

    async def demote(
        self,
        memory_id: str,
        *,
        to_tier: str = "stm",
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryVerbResult:
        """TARGETED single-memory demotion — run the REAL MTM->STM forgetting-curve tier-down on ONE
        resident MTM item (``to_tier="stm"``). Reuses the exact write-ahead-then-commit-delete
        sequencing ``DemotionService._demote_one`` uses (``demotion.py:233-276``, "dup > loss"):
        ``mtm.get`` the item -> write the STM-tier copy FIRST (``stm.put``, id-stable,
        ``created_at`` preserved) -> THEN remove the MTM point (``mtm.remove``). Unlike the
        automatic ``DemotionService`` this is user-forced, so it does NOT re-gate on
        ``SalienceStrategy`` (the caller asked to demote THIS item); the store sequence is same.

        A nonexistent id (absent from MTM) raises :class:`MemoryNotFoundError` (404); an invalid
        ``to_tier`` raises ``ValueError`` (400)."""
        ns = self._ns(user, session)
        target = _parse_tier(to_tier)
        if target is not MemoryTier.STM:
            raise ValueError(
                f"demote: invalid to_tier {to_tier!r} — demotion moves MTM->STM (to_tier='stm') "
                "only; use promote for upward tier moves"
            )
        item = await self._container.mtm.get(ns, memory_id)
        if item is None:
            raise MemoryNotFoundError(memory_id, reason="not resident in MTM (source tier)")
        now = self._clock.now()
        stm_copy = item.model_copy(deep=True)
        stm_copy.tier = MemoryTier.STM
        stm_copy.state = MemoryState.ACTIVE
        stm_copy.updated_at = now
        # Step 1 — write-ahead: STM gets the copy BEFORE MTM loses its point (dup > loss).
        await self._container.stm.put(stm_copy)
        # Step 2 — commit: remove the MTM point (plain deletion — a tier-down move has no "winner").
        await self._container.mtm.remove(ns, memory_id)
        events = await self._publish(
            MemoryDemoted(
                namespace=ns,
                id=memory_id,
                tier=CanonicalTier.MTM,
                to_tier=CanonicalTier.STM,
                to_state=CanonicalState.ACTIVE,  # a real tier-down move, NOT archival (AC-1.3)
                retention=0.0,  # user-forced, not salience-gated — no S(m) computed here
            )
        )
        return MemoryVerbResult(
            memory_id=memory_id,
            verb="demote",
            from_tier=MemoryTier.MTM.value,
            to_tier=MemoryTier.STM.value,
            tiers_affected=(MemoryTier.STM.value, MemoryTier.MTM.value),
            events_emitted=events,
        )

    async def update(
        self,
        memory_id: str,
        new_content: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryVerbResult:
        """TARGETED update = SUPERSEDE (invalidate-don't-delete): create the NEW version and mark
        the OLD one superseded by it, over the SAME bi-temporal supersession machinery an automatic
        conflict-driven update uses (``DistillPipeline._resolve`` -> ``mtm``/``ltm.invalidate``).
        No new memory logic:

        1. LOCATE the old memory across the tiers it lives in (``stm.get`` / ``mtm.get`` /
           ``ltm.get_fact``) — a nonexistent id raises :class:`MemoryNotFoundError` (404).
        2. INGEST ``new_content`` via the REAL :meth:`add` path -> a fresh ``memory_id`` (STM
           durable, importance-gated STM->MTM), the NEW version.
        3. SUPERSEDE the old: ``mtm.invalidate`` / ``ltm.invalidate`` (``loser=old``,
           ``winner=new``, ``at=now``, ``reason="update"``) stamp the old ``state=superseded`` +
           ``invalid_at`` + a ``SUPERSEDED_BY``/``superseded_by`` link to the new id — the SAME
           invalidate the conflict path uses. The old STM copy (ephemeral, no bi-temporal
           substrate) is evicted so active recall returns the NEW version.

        Returns the NEW memory (``memory_id`` = the new id, ``superseded_id`` = the old id) —
        "return the new memory"."""
        ns = self._ns(user, session)
        now = self._clock.now()
        old_stm = await self._container.stm.get(ns, memory_id)
        old_mtm = await self._container.mtm.get(ns, memory_id)
        old_ltm = await self._container.ltm.get_fact(ns, memory_id)
        if old_stm is None and old_mtm is None and old_ltm is None:
            raise MemoryNotFoundError(memory_id)
        new_receipt = await self.add(new_content, user=user, session=session)
        new_id = new_receipt.memory_id
        affected: list[str] = []
        if old_mtm is not None:
            await self._container.mtm.invalidate(
                ns, memory_id, new_id, at=now, reason="update"
            )
            affected.append(MemoryTier.MTM.value)
        if old_ltm is not None:
            await self._container.ltm.invalidate(
                ns, memory_id, new_id, at=now, reason="update"
            )
            affected.append(MemoryTier.LTM.value)
        if old_stm is not None:
            await self._container.stm.evict(ns, memory_id)
            affected.append(MemoryTier.STM.value)
        return MemoryVerbResult(
            memory_id=new_id,
            verb="update",
            superseded_id=memory_id,
            tiers_affected=tuple(affected),
        )

    async def delete(
        self,
        memory_id: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryVerbResult:
        """TARGETED delete = INVALIDATE-DON'T-DELETE (soft delete): stop the memory appearing in
        active recall while KEEPING it in bi-temporal history — never a hard ``DETACH DELETE`` of
        active data. LOCATES the memory across the tiers it lives in and, per tier:

        * **MTM/LTM** (bi-temporal substrate): ``mtm.expire`` / ``ltm.expire`` flip ``state`` to
          ``expired`` + stamp ``invalid_at=now`` — the mandatory ``state='active'`` recall filter
          drops it, the point/node STAYS (history intact, ``invalid_at`` set), no ``superseded_by``
          fabricated (a plain delete has no winner). This reuses the SAME invalidate-don't-delete
          substrate the supersession path uses, minus the winner.
        * **STM** (ephemeral, TTL — NO bi-temporal substrate): ``stm.evict`` — there is no history
          layer in STM to preserve (STM's own history is what promotion carries up to MTM/LTM), so
          the honest "stop appearing in active recall" for an STM-only row is eviction.

        A nonexistent id (absent from every tier) raises :class:`MemoryNotFoundError` (404)."""
        ns = self._ns(user, session)
        now = self._clock.now()
        affected: list[str] = []
        stm_item = await self._container.stm.get(ns, memory_id)
        if stm_item is not None:
            await self._container.stm.evict(ns, memory_id)
            affected.append(MemoryTier.STM.value)
        mtm_item = await self._container.mtm.get(ns, memory_id)
        if mtm_item is not None:
            await self._container.mtm.expire(ns, memory_id, at=now)
            affected.append(MemoryTier.MTM.value)
        ltm_item = await self._container.ltm.get_fact(ns, memory_id)
        if ltm_item is not None:
            await self._container.ltm.expire(ns, memory_id, at=now)
            affected.append(MemoryTier.LTM.value)
        if not affected:
            raise MemoryNotFoundError(memory_id)
        return MemoryVerbResult(
            memory_id=memory_id,
            verb="delete",
            tiers_affected=tuple(affected),
            invalidated=True,
        )

    async def _publish(self, event: MemoryPromoted | MemoryDemoted) -> tuple[str, ...]:
        """Publish ONE lifecycle event onto the container's real bus (the SAME ``InprocBus``
        ``PromotionService``/``DemotionService`` publish onto) and name it in the verb receipt's
        ``events_emitted``. A no-op (empty tuple) when no bus is wired (heuristic/embedded) — never
        a hard failure, mirroring the services' own ``if self._bus is not None`` guard."""
        if self._container.bus is None:
            return ()
        await self._container.bus.publish(event)
        return (type(event).__name__,)

    async def share(
        self,
        memory_id: str,
        *,
        visibility: Visibility,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> NoReturn:
        """The private->shared crossing verb (design §2.5 REVIEW-2 FIX 4, build-queue §13 item 3)
        — absent from BOTH surfaces today (provisional name/signature per §2.5); TO BUILD."""
        del memory_id, visibility, user, session
        raise SurfaceVerbNotImplementedError(
            "share",
            reason="crossing verb absent from both surfaces (build-queue §13 item 3) — TO BUILD",
        )

    # ----------------------------------------------------------------------------------- η helpers
    def _ns(self, user: str, session: str | None) -> Namespace:
        """PORT of ``LocalMemory._ns`` (``mu-local/local_memory.py:305-312``) — identical fields,
        so a memory this facade writes and one ``LocalMemory`` writes for the same
        ``(workspace, namespace, user, session)`` land in the SAME η partition."""
        return Namespace(
            org=self._org,
            workspace=self._workspace,
            user=user,
            session=session or _DEFAULT_SESSION,
            visibility=Visibility.PRIVATE,
        )

    def _scope(self, user: str, session: str | None) -> ClientScope:
        """PORT of ``LocalMemory._scope`` (``mu-local/local_memory.py:314-321``) — ``agent_kind``
        is left at its ``ClientScope`` default (``HUMAN_PROXY``), exactly as ``LocalMemory._scope``
        does not set it either."""
        return ClientScope(
            principal_id=user,
            org_id=self._org,
            workspace_id=self._workspace,
            session_id=session or _DEFAULT_SESSION,
            agent_principal_id=user,
        )


# ------------------------------------------------------------------------------------ pure helpers
_TIER_BY_NAME: dict[str, MemoryTier] = {tier.value: tier for tier in MemoryTier}


def _parse_tier(to_tier: str) -> MemoryTier:
    """Map a wire ``to_tier`` string (``"stm"``/``"mtm"``/``"ltm"``) to :class:`MemoryTier`,
    failing LOUD on an unknown value (``ValueError`` -> HTTP 400) — never a silent default or
    no-op. Mirrors ``mu_client.mcp.tools.resolve_tier``'s allowlist discipline; a ``MemoryTier``
    passed straight through (defensive) is accepted too."""
    key = to_tier.value if isinstance(to_tier, MemoryTier) else str(to_tier).strip().lower()
    try:
        return _TIER_BY_NAME[key]
    except KeyError:
        raise ValueError(
            f"unknown to_tier {to_tier!r}; expected one of {sorted(_TIER_BY_NAME)}"
        ) from None


def _normalize_messages(
    content: str | dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """PORT of ``LocalMemory``'s private ``_normalize_messages`` (``mu-local/local_memory.py:325-
    335``, itself a PORT of mem0 ``client/main.py:153-160``) — duplicated (not imported: it is an
    unexported, underscore-prefixed helper in a package this module cannot import) so ``add()``'s
    input handling stays byte-identical across both surfaces."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return content
    raise ValueError(f"content must be str, dict, or list[dict], got {type(content).__name__}")


def _channels_for_tier(tier: MemoryTier | None) -> _EngineRecallChannels:
    """PORT of ``LocalMemory``'s private ``_channels_for_tier`` (``mu-local/local_memory.py:456-
    463``) — ``tier`` narrows recall to one channel; ``None`` runs all three."""
    if tier is None:
        return _EngineRecallChannels()
    return _EngineRecallChannels(
        stm=tier is MemoryTier.STM,
        mtm=tier is MemoryTier.MTM,
        ltm=tier is MemoryTier.LTM,
    )


def _to_canonical_recall_result(result: _EngineRecallResult) -> CanonicalRecallResult:
    """Map the engine-native :class:`~mu_engine.services.recall.dto.RecallResult` onto the
    canonical :class:`~mu_contracts.contracts.recall.RecallResult` (Decision B) — a PORT of
    ``mu_local.local_memory._to_recall_result`` (``mu-local/local_memory.py:468-497``), re-derived
    here since that helper is private to a package this module cannot import. ``namespace`` and
    ``DegradeReason`` are the SAME type on both sides (``mu_engine.storage.domain.namespace`` /
    ``mu_engine.services.recall.dto`` re-export ``mu_contracts``'s own classes) so they pass
    straight through; only the per-item ``Tier``/``RecallChannels`` shapes need an explicit re-wrap
    since the engine's internal item additionally carries a federate-dedup ``content_hash``/
    per-item ``namespace`` the canonical surface item deliberately drops."""
    return CanonicalRecallResult(
        namespace=result.namespace,
        items=[
            CanonicalRecallItemView(
                memory_id=item.memory_id,
                content=item.content,
                tier=CanonicalTier(item.tier.value),
                channel=item.channel,
                fused_score=item.fused_score,
                rerank_score=item.rerank_score,
                is_floor=item.is_floor,
                artifact_ref=item.artifact_ref,
            )
            for item in result.items
        ],
        channels_run=CanonicalRecallChannels(
            stm=result.channels_run.stm,
            mtm=result.channels_run.mtm,
            ltm=result.channels_run.ltm,
        ),
        degraded=result.degraded,
        generated_at=result.generated_at,
    )


def _to_memory_response(item: MemoryItem) -> MemoryResponse:
    """Map the engine-native :class:`~mu_engine.storage.domain.memory.MemoryItem` domain object
    onto the canonical, FROZEN-wire-schema :class:`~mu_contracts.contracts.memory.MemoryResponse`
    (Decision B) — the ``get`` return DTO this facade now serves (module docstring).
    ``content_type`` has no correlate on ``MemoryItem`` (a storage/domain record, not an
    ingest-time content-type tag); every memory this facade's ``add()`` writes originates as
    plain chat/activity text, so
    ``"text"`` is the honest default here, never a guess at a richer type the item does not carry.
    Every other field below is a direct 1:1 read off ``MemoryItem`` (module docstring's own field
    list, ``mu_engine/storage/domain/memory.py:150-198``) — no field is fabricated."""
    return MemoryResponse(
        id=item.id,
        content=item.content,
        content_type="text",
        tier=item.tier.value,
        state=item.state.value,
        importance_score=item.importance_score,
        access_count=item.access_count,
        created_at=item.created_at,
        updated_at=item.updated_at,
        namespace=item.namespace.to_prefix(),
        metadata={k: str(v) for k, v in item.metadata.items()},
        source=item.source.value,
        session_id=item.session_id,
        subject=item.subject,
        predicate=item.predicate,
        object=item.object,
        object_kind=item.object_kind.value if item.object_kind is not None else None,
        polarity=item.polarity.value,
        valid_at=item.valid_at,
        invalid_at=item.invalid_at,
        relevance_score=item.relevance_score,
        content_hash=item.content_hash,
    )


def _render_context(items: list[CanonicalRecallItemView], *, max_chars: int | None) -> str:
    """Deterministic context assembly (no LLM) — PORT of ``mu_local.local_memory._render_context``
    (``mu-local/local_memory.py:503-509``): one bullet per hit, truncated to ``max_chars``.
    Duplicated (not imported: private, underscore-prefixed, in a package this module cannot
    import), used by both :meth:`SurfaceFacade.build_context` and :func:`_render_facts` below."""
    lines = [f"- {item.content}" for item in items]
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text


def _render_facts(result: CanonicalRecallResult) -> str:
    """Deterministic context assembly for ``ask()``'s prompt only (no LLM synthesis) — reuses
    :func:`_render_context` with no ``max_chars`` ceiling, matching this facade's prior behaviour
    (the PORTed ``mu_local`` helper it mirrors is likewise unbounded on the ``ask`` path)."""
    return _render_context(result.items, max_chars=None)


def _host() -> str:
    """The ``IngestActivity.host`` stamp for a SurfaceFacade-originated write — distinct from
    mu-local's own ``_HOST = "mu-local"`` (``mu-local/local_memory.py:67``) so a captured activity's
    provenance names the surface that actually wrote it, never a borrowed literal."""
    return "mu-engine-surface-facade"


def _fresh_offset() -> str:
    """A fresh, unique ``session_offset`` per ``add()`` message — PORT of ``LocalMemory.add``'s
    ``uuid.uuid4().hex`` call (``mu-local/local_memory.py:122``): unique => never a pure M12
    replay."""
    return uuid.uuid4().hex
