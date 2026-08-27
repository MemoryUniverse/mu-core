"""``LocalMemory`` — the embedded, in-process, daemonless engine facade (mu-local-and-sdk §2.1).

PORT of ``mem0.Memory`` (``other_repos/mem0/mem0/memory/main.py:172-204`` + ``from_config``
``:235-238``): build the engine graph from config in a composition root, then expose a small verb
surface. Re-expressed over MU ports — every verb delegates to exactly ONE ``mu-engine`` application
service through the :class:`~mu_local.composition.LocalContainer` (the same discipline as the
server-side ``SurfaceFacade``), so the identical behaviour is reachable in-process (LOCAL) or over
the wire (SERVER) with no second implementation.

Single-tenant by construction (spec §3.2): ``workspace`` + ``namespace`` are fixed at construction;
each call's ``user``/``session`` populate the remaining η slots; every persisted key is a
``Namespace(...).to_prefix()`` with ``visibility=PRIVATE``. NO SHARED partition, NO grant, NO
``authorized_ids`` stamp is ever constructed here — those are ``mu-server`` concepts (spec T10).
The facade NEVER imports ``mu-server`` (CI: mu-local-no-server / import-linter mu-local-layers).

Async-first (DEV-STANDARDS rule 1: fully async, cancellation-correct): the verbs are coroutines
over the async engine, and ``async with`` releases every store connection. The sync twin the spec
names (``LocalMemory`` sync / ``AsyncLocalMemory``) wraps this via ``asyncio.run`` in a later phase
— a tracked, explicit gap, not a silent stub.

LLM WIRING (closed 2026-07-27): ``add``/``recall``/``search``/``context``/``consolidate`` are
always real (heuristic extraction + deterministic assembly work with or without an LLM). ``ask``
still refuses loudly with :class:`~mu_local.errors.LlmNotConfiguredError` (spec §7, T7) in
heuristic mode (no ``storage.llm`` configured) — mu-local NEVER fabricates an empty/degraded
synthesis. When a :class:`~mu_local.config.ModelProfileSettings` IS configured, ``ask`` recalls
context (the same deterministic assembly :meth:`context` uses) and synthesises a REAL answer via
the composition root's :class:`~mu_engine.providers.model_router.ModelRouter` ANSWER task.

UNIFIED VERB SURFACE (build-plan Stage B, task B1; sdk-engine-server-design.md §2.5;
``SDK-BUILD-DECISIONS.md`` Decision B):

- ``add``/``recall`` now accept the canonical superset's SHARED-plane fields (``visibility``/
  ``subject``/``predicate``/``object``), plane-gated through
  :func:`mu_contracts.validation.validate_plane_fields`. ``LocalMemory`` is private-plane-only BY
  CONSTRUCTION (module docstring above) — it always calls the validator with
  ``shared_configured=False``, so any of those fields supplied as non-``None`` is REJECTED with
  :class:`~mu_contracts.domain.errors.PlaneFieldRejectedError`, never silently accepted/dropped.
  This is the intended, permanent behaviour for the embedded surface, not a temporary gap.
- ``add`` returns the canonical :class:`~mu_contracts.contracts.views.MemoryWriteResult` (was
  ``mu_local.views.MemoryWriteResult``) — MUST-ADD fields ``namespace``/``events_emitted`` are
  populated from the resolved η and ``IngestResult.events_emitted`` respectively, zero extra I/O.
- ``recall`` returns the canonical :class:`~mu_contracts.contracts.recall.RecallResult` (was the
  lossy ``mu_local.views.MemoryListView`` via the now-deleted ``_to_list_view``) — the un-collapsed
  engine result, carrying ``namespace``/``channels_run``/``generated_at`` that the old view threw
  away. ``context`` merely follows ``recall``'s new item type since it assembles its window from
  ``recall``'s own output.
- ``get`` returns the canonical :class:`~mu_contracts.contracts.memory.MemoryResponse` (carryover
  CO-1; was the retired ``mu_local.views.MemoryRecordView``, a thin ranked-hit shape) — parity with
  the public ``MemoryClient.get`` (Decision B winner: a point-get is a full row, not a hit). Mirrors
  :func:`~mu_engine.surface.facade._to_memory_response` (``mu_engine/surface/facade.py:486-519``),
  the same engine-native :class:`~mu_engine.storage.domain.memory.MemoryItem` -> ``MemoryResponse``
  mapping ``SurfaceFacade.get`` already performs, re-derived here as :func:`_to_memory_response`
  below since ``mu_engine.surface.facade`` keeps that helper module-private.
- ``promote``/``demote`` are NEW — both delegate to the injected
  :class:`~mu_engine.surface.facade.SurfaceFacade` (Stage A), which raises the named
  :class:`~mu_engine.surface.facade.SurfaceVerbNotImplementedError` for both verbs today
  (build-queue §13 item 5) — an honest, named 501, never a silent no-op. This is the one place
  this facade "delegates through SurfaceFacade" (plan §3 B1 item (f)): every other verb keeps
  calling the composition root directly, since re-deriving them through the facade would change
  observable behaviour (e.g. the facade's own ``add`` stamps a different ``IngestActivity.host``,
  ``mu_engine/surface/facade.py:387-391``) — "no engine algorithm change" per the plan, not "no
  behaviour change," but this module honours the stronger bar anyway outside promote/demote.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from mu_contracts.contracts.defaults import (
    DEFAULT_CONSOLIDATE_LIMIT,
    DEFAULT_RECALL_LIMIT,
)
from mu_contracts.contracts.memory import MemoryResponse
from mu_contracts.contracts.recall import RecallChannels, RecallItemView, RecallResult
from mu_contracts.contracts.views import (
    ConsolidateView,
    ContextView,
    MemoryVerbResult,
    MemoryWriteResult,
)
from mu_contracts.domain.model.agent import (
    AgentIdentity,
    resolve_subagent_identity,
    subagent_write_namespace,
)
from mu_contracts.domain.model.memory import Tier
from mu_contracts.domain.model.scope import AgentKind, ClientScope
from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.lifecycle_lease import LifecycleLeasePort
from mu_contracts.ports.lifecycle_workflow import LifecycleWorkflowRunnerPort
from mu_contracts.ports.time import Clock
from mu_contracts.validation import validate_plane_fields
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.distill import DistillActionKind
from mu_engine.providers._contracts import Message, MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.services.ingest import IngestResult
from mu_engine.services.recall.dto import RecallChannels as _EngineRecallChannels
from mu_engine.services.recall.dto import RecallQuery
from mu_engine.services.recall.dto import RecallResult as _EngineRecallResult
from mu_engine.storage.domain.memory import MemoryItem, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.surface.facade import SurfaceFacade
from mu_local.composition import LocalContainer
from mu_local.config import StorageSettings
from mu_local.errors import LlmNotConfiguredError

if TYPE_CHECKING:  # pragma: no cover — typing only, avoids a hard import-time cycle.
    from mu_engine.lifecycle.manager import MemoryLifecycleManager, WarmRecallCacheServicePort
    from mu_engine.services.health.service import MemoryHealthService
    from mu_engine.services.pin.service import PinService

__all__ = ["LocalMemory"]

_DEFAULT_USER = "default"
_DEFAULT_SESSION = "default"
_HOST = "mu-local"
# PORT of the reference SLM integration test's ANSWER-task system prompt
# (mu-engine/tests/pipelines/test_distill_llm_slm_int.py:393-396) — a named module constant, never
# an inline literal at the call site (DEV-STANDARDS rule 3).
_ASK_SYSTEM_PROMPT = "Answer the question using ONLY the given facts. Be concise."
# ``add()``'s "caller expressed no opinion" fallback — READ off ``IngestActivity.importance``'s own
# field default rather than a second hardcoded ``0.5`` literal here (DEV-STANDARDS rule 3; also
# keeps this module in lockstep if that default ever changes). A plain ``float`` (not ``Any``) so
# mypy-strict can verify the ``importance=`` kwarg below without an ``**dict`` unpack (which mypy
# cannot type-check against a ``BaseModel``'s heterogeneous field types).
_DEFAULT_IMPORTANCE: float = IngestActivity.model_fields["importance"].default


class LocalMemory:
    """Embedded FULL-LOCAL memory entrypoint (async). Port of ``mem0.Memory`` over MU ports."""

    def __init__(
        self,
        storage: StorageSettings | None = None,
        *,
        workspace: str = "local",
        namespace: str = "default",
        settings: Any | None = None,
        lifecycle: LifecycleSettings | None = None,
    ) -> None:
        self._workspace = workspace
        self._org = namespace  # the η.org slot (spec §3.2: namespace fixes org)
        self._container = LocalContainer(
            storage or StorageSettings(), settings=settings, lifecycle=lifecycle
        )
        # Stage A's unified-surface facade, injected with THIS instance's own composition root
        # (never a second, independently-constructed container — DEV-STANDARDS rule 9). Used only
        # where delegating changes no observable behaviour today: promote/demote (module
        # docstring "UNIFIED VERB SURFACE").
        self._facade = SurfaceFacade(self._container, workspace=workspace, namespace=namespace)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LocalMemory:
        """Dict-driven construction (parity with ``mem0.Memory.from_config`` main.py:235-238):
        validate into :class:`StorageSettings`, then delegate to ``__init__``."""
        raw_storage = config.get("storage", {})
        storage = StorageSettings.model_validate(raw_storage)
        return cls(
            storage,
            workspace=str(config.get("workspace", "local")),
            namespace=str(config.get("namespace", "default")),
        )

    # ----------------------------------------------------------------------------- write
    async def add(
        self,
        content: str | dict[str, Any] | list[dict[str, Any]],
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        # COMMON field (design §2.5 superset, canonical wire ``AddRequest.importance_score``,
        # ``mu_contracts/contracts/requests.py:195``) — never plane-gated. ``None`` (the default)
        # means "caller expressed no opinion": omitted from the constructed ``IngestActivity`` so
        # its own field default (``0.5``) applies, unchanged from this verb's pre-fix behaviour. A
        # real value threads straight through to the ONE gate that decides STM->MTM promotion —
        # ``DeterministicPromoteStage``'s ``importance >= IngestSettings.importance_promote`` check
        # (``mu-engine/pipelines/concrete/ingest.py:230``) — never hardcoded to force a promote.
        importance_score: float | None = None,
        # Phase 1.5 subagent partition (AGENT-INTEGRATION-AUDIT-AND-PLAN.md §6; agent.py). When a
        # capture is attributed to a Claude Code SUBAGENT, ``agent_type`` is its name (Task tool
        # ``subagent_type``). Threading it here resolves a STABLE, deterministic
        # ``agent_principal_id`` (:func:`resolve_subagent_identity`) and writes the memory into an
        # AGENT-SCOPED SESSION under the SAME ``η.user`` (the owner) — a DISTINCT ``to_prefix()``
        # partition the owner's federate-live recall still surfaces, cross-user still isolated
        # (agent.py federation choice). ``None`` (every human/top-level caller) ⇒ the pre-existing
        # single-partition behaviour, byte-for-byte. Mutually exclusive with ``agent`` (a
        # pre-resolved identity); pass at most one.
        agent_type: str | None = None,
        agent: AgentIdentity | None = None,
        # SHARED-plane fields of the canonical superset signature (design §2.5) — accepted so the
        # signature matches the unified surface, always REJECTED (never silently dropped) because
        # LocalMemory is private-plane-only by construction (module docstring, plane_gate.py's own
        # docstring). Supplying a non-None value here always raises PlaneFieldRejectedError.
        visibility: Visibility | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,  # matches the frozen wire field name, mu_sdk parity
    ) -> MemoryWriteResult:
        """Ingest one activity (STM durable -> deterministic STM->MTM promote, GATED on importance
        — REMEDIATION Rank 2 / conformance A6 fix). ``content`` accepts ``str | dict | list[dict]``
        and normalises to messages (mem0 ``client/main.py:153-160``). Each message becomes one
        INGEST activity keyed by a fresh source offset (a genuine new occurrence, CANONICAL
        §8-M12); MTM promotion is idempotent on ``content_hash``.

        ``promote`` is NEVER hardcoded ``True`` here (that was the A6 defect: it short-circuited
        ``DeterministicPromoteStage``'s own importance/mention gate, so EVERY add promoted
        regardless of salience, defeating the tier design). This verb constructs
        ``IngestActivity(promote=False, importance=<threaded importance_score>)`` and lets the
        deterministic stage make the tier call — ``explicit promote OR importance>=threshold`` —
        exactly as engine-core-spec §12's gate is designed to. There is no caller-facing "force
        promote" override on this verb today (the canonical ``AddRequest`` names no such field);
        ``importance_score`` is the one lever a caller has, matching the wire contract's own field
        set.

        Returns the canonical :class:`~mu_contracts.contracts.views.MemoryWriteResult` receipt
        (Decision B) — ``namespace``/``events_emitted`` are populated from the resolved η and the
        engine's own :class:`~mu_engine.services.ingest.IngestResult`, zero extra I/O."""
        validate_plane_fields(
            {
                "visibility": visibility,
                "subject": subject,
                "predicate": predicate,
                "object": object,
            },
            private_configured=True,
            shared_configured=False,
        )
        ns = self._subagent_ns(user, session, agent_type=agent_type, agent=agent)
        # ``None`` means "caller expressed no opinion" -> falls to IngestActivity's own field
        # default (module constant above), never a second hardcoded literal duplicating it here.
        importance = importance_score if importance_score is not None else _DEFAULT_IMPORTANCE
        last: IngestResult | None = None
        for message in _normalize_messages(content):
            activity = IngestActivity(
                namespace=ns,
                host=_HOST,
                session_offset=uuid.uuid4().hex,  # unique ⇒ never a pure M12 replay
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
        limit: int = DEFAULT_CONSOLIDATE_LIMIT,
    ) -> ConsolidateView:
        """MTM->LTM consolidation (DISTILL): extract bi-temporal SPO facts from the recent STM
        window and write them into the LTM graph, applying invalidate-don't-delete supersession.
        Heuristic (no-LLM) extractor — real now. Daemonless: the caller drives the sweep.

        ADR 0031 / spec §3: a MANAGED namespace refuses this manual verb loud
        (:class:`~mu_engine.lifecycle.mode_gate.ManagerOwnsLifecycleError`, mapped to HTTP 409 by
        whatever surface calls this) — the engine-side :class:`ManagerModeGate` is consulted
        BEFORE delegating to ``distill`` so the caller can never self-authorize. MANUAL/HYBRID
        pass through unchanged (this is the ONE change ADR 0031 grows on this verb; no other verb
        is touched)."""
        ns = self._ns(user, session)
        self._container.mode_gate.assert_manual_allowed(ns, "consolidate")
        recent = await self._container.stm.recent(ns, limit=limit)
        window = [scored.item for scored in recent]
        report = await self._container.distill.distill(ns, window)
        return ConsolidateView(
            facts_extracted=report.facts_extracted,
            added=report.added,
            superseded=report.superseded,
            # MUST-ADD (Decision B) — a NOOP action (identical active fact, mem0 NONE-reinforce,
            # `mu_engine/pipelines/distill.py:207`) was silently dropped by the pre-Decision-B
            # embedded `mu_local.views.ConsolidateView`; count it the same way `added`/
            # `superseded` already count their own action kind (`distill.py:237-243`).
            noop=sum(a.kind is DistillActionKind.NOOP for a in report.actions),
        )

    # ----------------------------------------------------------------------------- read
    async def recall(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        tier: MemoryTier | None = None,
        limit: int = DEFAULT_RECALL_LIMIT,
        # SHARED-plane fields — same discipline as :meth:`add`; always rejected when non-None
        # (LocalMemory is private-plane-only by construction).
        visibility: Visibility | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object: str | None = None,  # matches the frozen wire field name, mu_sdk parity
    ) -> RecallResult:
        """Federate-live RANKED recall over the private-own partition (STM floor ⊕ MTM dense ⊕ LTM
        graph), fused once. LOCAL mode has no shared arm (spec §3.2). ``tier`` narrows to one
        channel; ``None`` runs all three.

        Returns the canonical :class:`~mu_contracts.contracts.recall.RecallResult` (Decision B) —
        the un-collapsed engine result (``namespace``/``channels_run``/``generated_at``, richer
        per-hit scores), replacing the lossy ``mu_local.views.MemoryListView`` this verb used to
        collapse into via the now-deleted ``_to_list_view``."""
        validate_plane_fields(
            {
                "visibility": visibility,
                "subject": subject,
                "predicate": predicate,
                "object": object,
            },
            private_configured=True,
            shared_configured=False,
        )
        ns = self._ns(user, session)
        scope = self._scope(user, session)
        q = RecallQuery(namespace=ns, text=query, limit=limit, channels=_channels_for_tier(tier))
        result = await self._container.recall.recall(scope, q)
        return _to_recall_result(result)

    async def search(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        tier: MemoryTier | None = None,
        limit: int = DEFAULT_RECALL_LIMIT,
    ) -> RecallResult:
        """mem0 muscle-memory alias for :meth:`recall` (spec verb-alias policy, §CC-6): ``recall``
        is the canonical read verb; ``search`` is a documented alias, one single behaviour."""
        return await self.recall(query, user=user, session=session, tier=tier, limit=limit)

    async def get(
        self,
        memory_id: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryResponse | None:
        """Point-get one memory by id from the caller's STM partition (``None`` if absent). NOTE:
        the phase-0 MTM/LTM adapters expose no point-get, so this reads the STM tier only — a
        tracked, explicit narrowing (the ``user``/``session`` args name the η partition, since a
        bare id cannot be located across prefix-partitioned stores).

        Returns the canonical :class:`~mu_contracts.contracts.memory.MemoryResponse` (carryover
        CO-1) — get-parity with the public ``MemoryClient.get`` (Decision B winner: a point-get is
        a full row, not a ranked hit). Was the retired ``mu_local.views.MemoryRecordView`` (module
        docstring "UNIFIED VERB SURFACE"), a thinner 6-field shape this verb collapsed the STM
        ``MemoryItem`` into; :func:`_to_memory_response` below now hydrates every
        ``MemoryResponse`` field the STM row actually carries instead."""
        ns = self._ns(user, session)
        item = await self._container.stm.get(ns, memory_id)
        if item is None:
            return None
        return _to_memory_response(item)

    async def context(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = DEFAULT_RECALL_LIMIT,
        max_chars: int | None = None,
    ) -> ContextView:
        """Assemble a context window from recalled hits by DETERMINISTIC concatenation (no LLM
        synthesis — spec §7 INJECT render, Azure PARKED). The heuristic assembly folds behind an
        injected renderer once synthesis lands, without changing recall's core.

        Returns the canonical :class:`~mu_contracts.contracts.views.ContextView` (Decision B) —
        ``items`` is now the single canonical hit-item type
        (:class:`~mu_contracts.contracts.recall.RecallItemView`), following :meth:`recall`'s own
        migration; ``degraded`` stays the plain NAMED-degrade ``str | None`` the embedded shape
        always used, derived from :class:`RecallResult`'s typed ``DegradeReason`` exactly as the
        deleted ``_to_list_view`` used to (``ContextView``'s own docstring)."""
        listing = await self.recall(query, user=user, session=session, limit=limit)
        text = _render_context(listing.items, max_chars=max_chars)
        degraded = listing.degraded.value if listing.degraded is not None else None
        return ContextView(text=text, items=listing.items, degraded=degraded)

    async def ask(
        self,
        question: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = DEFAULT_RECALL_LIMIT,
    ) -> str:
        """Synthesise an answer over recalled context via the configured LLM's ANSWER task.
        Heuristic mode (``llm=None``, the default) refuses loudly — mu-local NEVER returns an
        empty/degraded synthesis (spec §7, T7). Use :meth:`recall`/:meth:`context` for retrieval
        without synthesis."""
        if self._container.llm is None:
            raise LlmNotConfiguredError(
                "ask() requires a configured LLM; LocalMemory is in heuristic mode (llm=None). "
                "Use recall()/context() for retrieval without synthesis."
            )
        assembled = await self.context(question, user=user, session=session, limit=limit)
        messages = [
            Message(role=MessageRole.SYSTEM, content=_ASK_SYSTEM_PROMPT),
            Message(
                role=MessageRole.USER,
                content=f"Facts:\n{assembled.text}\n\nQuestion: {question}",
            ),
        ]
        completion = await self._container.llm.generate(Task.ANSWER, messages)
        return completion.text

    # ------------------------------------------------------ targeted lifecycle verbs (§13 item 5)
    async def promote(
        self,
        memory_id: str,
        *,
        to_tier: str,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryVerbResult:
        """``promote`` — TARGETED single-memory promotion (STM->MTM with ``to_tier="mtm"``,
        MTM->LTM with ``to_tier="ltm"``). Delegates to the injected
        :class:`~mu_engine.surface.facade.SurfaceFacade.promote`, which runs the REAL promotion
        path on that one resident item (``PromotionService`` copy-on-write / ``DistillPipeline``
        leg) — a nonexistent id raises
        :class:`~mu_contracts.domain.errors.MemoryNotFoundError`, an invalid ``to_tier`` raises
        ``ValueError``. No longer a 501 (build-queue §13 item 5 landed)."""
        return await self._facade.promote(memory_id, to_tier=to_tier, user=user, session=session)

    async def demote(
        self,
        memory_id: str,
        *,
        to_tier: str = "stm",
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryVerbResult:
        """``demote`` — TARGETED single-memory MTM->STM tier-down (``to_tier="stm"``). Delegates to
        :meth:`~mu_engine.surface.facade.SurfaceFacade.demote`, which reuses
        ``DemotionService._demote_one``'s write-ahead-then-remove sequence on that one item."""
        return await self._facade.demote(memory_id, to_tier=to_tier, user=user, session=session)

    async def update(
        self,
        memory_id: str,
        new_content: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryVerbResult:
        """``update`` — SUPERSEDE the old memory with a new version (invalidate-don't-delete).
        Delegates to :meth:`~mu_engine.surface.facade.SurfaceFacade.update`, which INGESTs the new
        content and marks the old ``superseded_by`` the new via the SAME
        ``MtmTierRepository``/``GraphStorePort.invalidate`` the conflict path uses. Returns the NEW
        memory (``memory_id`` = new id, ``superseded_id`` = old id)."""
        return await self._facade.update(memory_id, new_content, user=user, session=session)

    async def delete(
        self,
        memory_id: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryVerbResult:
        """``delete`` — soft-delete (invalidate-don't-delete): stop the memory appearing in active
        recall while KEEPING it in bi-temporal history. Delegates to
        :meth:`~mu_engine.surface.facade.SurfaceFacade.delete` (MTM/LTM ``expire`` = state=expired +
        invalid_at; STM ``evict`` — no bi-temporal substrate there). Never a hard delete of active
        data."""
        return await self._facade.delete(memory_id, user=user, session=session)

    # ----------------------------------------------------------------------------- lifecycle
    @property
    def bus(self) -> EventBusPort:
        """The REAL ``InprocBus`` this facade's own ``ingest``/``distill`` publish onto
        (integrate-phase accessor — ``mu_local.composition.LocalContainer.bus``). A caller that
        needs to observe THIS instance's real event stream (e.g. mu-client's daemon
        ``MaintenanceLoop``) subscribes here, never to a second, independently-constructed bus."""
        return self._container.bus

    @property
    def health(self) -> MemoryHealthService | None:
        """THIS instance's memory-health lens (``LocalContainer.health``), or ``None`` when the
        bound vector backend has no partition-walk primitive.

        The accessor exists so a HOST can reach the service the composition root already built.
        Without it mu-client held a ``LocalMemory`` and could not get at the container's
        ``health``/``pin`` at all, so its ``/health`` route answered ``health_service_not_wired``
        even on a binding that could serve — the surfaces were inert one layer above a working
        engine. ``| None`` rather than raise-if-absent: absence is a REAL binding state
        (pgvector/chroma/faiss expose no walk primitive), and every surface already has a named,
        content-free degrade for it."""
        return self._container.health

    @property
    def pin(self) -> PinService | None:
        """THIS instance's pin service (``LocalContainer.pin``), or ``None`` when a bound tier
        cannot apply the id-stable pin upsert or the partition cannot be walked (the pin-explosion
        bound is counted with one bounded ``enumerate`` page). Same accessor rationale as
        :attr:`health`."""
        return self._container.pin

    def build_lifecycle_manager(
        self,
        *,
        lease: LifecycleLeasePort | None = None,
        runner: LifecycleWorkflowRunnerPort | None = None,
        clock: Clock | None = None,
        warm_cache: WarmRecallCacheServicePort | None = None,
    ) -> MemoryLifecycleManager:
        """Passthrough to :meth:`mu_local.composition.LocalContainer.build_lifecycle_manager` —
        constructs a :class:`~mu_engine.lifecycle.manager.MemoryLifecycleManager` wired against
        THIS instance's own stores/distill/bus (never a second, independently-constructed set).
        ``lease``/``runner``/``clock``/``warm_cache`` are optional passthrough kwargs so a caller
        with real cross-process adapters (mu-client's ``SqliteWalLeaseAdapter``/``SqliteWalRunner``,
        S1-06) or a real warm-cache service (mu-client's ``RecallInjectBridge``, S3-02) can thread
        them through this SAME composition root."""
        return self._container.build_lifecycle_manager(
            lease=lease, runner=runner, clock=clock, warm_cache=warm_cache
        )

    async def aclose(self) -> None:
        """Release every store connection the container opened."""
        await self._container.close()

    async def __aenter__(self) -> LocalMemory:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ----------------------------------------------------------------------------- η helpers
    def _ns(self, user: str, session: str | None) -> Namespace:
        return Namespace(
            org=self._org,
            workspace=self._workspace,
            user=user,
            session=session or _DEFAULT_SESSION,
            visibility=Visibility.PRIVATE,
        )

    def _subagent_ns(
        self,
        user: str,
        session: str | None,
        *,
        agent_type: str | None,
        agent: AgentIdentity | None,
    ) -> Namespace:
        """The write η — Phase 1.5 (agent.py). No subagent attribution ⇒ the plain per-caller
        partition (:meth:`_ns`), unchanged for every human/top-level write. A SUBAGENT attribution
        (``agent_type`` to resolve here, or a pre-resolved ``agent``) builds a real
        ``ClientScope`` with ``agent_kind=AgentKind.SUBAGENT`` — the identity model's first live
        call site — and routes it through :func:`subagent_write_namespace`, which keeps ``η.user``
        the owner (federation + isolation) and derives an AGENT-SCOPED SESSION (the distinct
        partition)."""
        if agent_type is None and agent is None:
            return self._ns(user, session)
        if agent_type is not None and agent is not None:
            raise ValueError("pass at most one of agent_type / agent")
        identity = agent or resolve_subagent_identity(
            workspace_id=self._workspace,
            owner_principal_id=user,
            parent_session_id=session or _DEFAULT_SESSION,
            agent_type=agent_type or "",
        )
        scope = ClientScope(
            principal_id=user,  # the OWNER — η.user stays the owner (federation + isolation)
            org_id=self._org,
            workspace_id=self._workspace,
            session_id=session or _DEFAULT_SESSION,
            agent_principal_id=identity.agent_principal_id,
            agent_kind=AgentKind.SUBAGENT,
            agent_path=identity.agent_path,
        )
        return subagent_write_namespace(scope)

    def _scope(self, user: str, session: str | None) -> ClientScope:
        return ClientScope(
            principal_id=user,
            org_id=self._org,
            workspace_id=self._workspace,
            session_id=session or _DEFAULT_SESSION,
            agent_principal_id=user,
        )


# ------------------------------------------------------------------------------------ pure helpers
def _normalize_messages(
    content: str | dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise ``add`` input to messages (PORT of mem0 ``client/main.py:153-160``)."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return content
    raise ValueError(f"content must be str, dict, or list[dict], got {type(content).__name__}")


def _channels_for_tier(tier: MemoryTier | None) -> _EngineRecallChannels:
    """A ``tier`` narrows recall to one channel; ``None`` runs all three (STM/MTM/LTM)."""
    if tier is None:
        return _EngineRecallChannels()
    return _EngineRecallChannels(
        stm=tier is MemoryTier.STM,
        mtm=tier is MemoryTier.MTM,
        ltm=tier is MemoryTier.LTM,
    )


def _to_recall_result(result: _EngineRecallResult) -> RecallResult:
    """Map the engine-native :class:`~mu_engine.services.recall.dto.RecallResult` onto the
    canonical :class:`~mu_contracts.contracts.recall.RecallResult` (Decision B) — REPLACES the
    deleted ``_to_list_view``/``mu_local.views.MemoryListView`` collapse. ``namespace`` and
    ``DegradeReason`` are the SAME type on both sides (``mu_engine.storage.domain.namespace`` /
    ``mu_engine.services.recall.dto`` re-export ``mu_contracts``'s own classes, module docstrings)
    so they pass straight through; only the per-item ``Tier``/``RecallChannels`` shapes need an
    explicit re-wrap since the engine's internal item additionally carries a federate-dedup
    ``content_hash``/per-item ``namespace`` this surface deliberately drops (Decision B cross-
    cutting section, ``mu_contracts.contracts.recall`` module docstring)."""
    return RecallResult(
        namespace=result.namespace,
        items=[
            RecallItemView(
                memory_id=item.memory_id,
                content=item.content,
                tier=Tier(item.tier.value),
                channel=item.channel,
                fused_score=item.fused_score,
                rerank_score=item.rerank_score,
                is_floor=item.is_floor,
                artifact_ref=item.artifact_ref,
            )
            for item in result.items
        ],
        channels_run=RecallChannels(
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
    (Decision B; carryover CO-1) — :meth:`LocalMemory.get`'s new return DTO, replacing the retired
    ``mu_local.views.MemoryRecordView``. Mirrors
    :func:`~mu_engine.surface.facade._to_memory_response` (``mu_engine/surface/facade.py:486-519``)
    field-for-field, re-derived here rather than imported since that helper is private to a module
    this one only needs the identical mapping from, not a hard dependency on.

    ``content_type`` has no correlate on ``MemoryItem`` (a storage/domain record, not an ingest-
    time content-type tag); every memory ``add()`` writes originates as plain chat/activity text,
    so ``"text"`` is the honest default, never a guess at a richer type the item does not carry
    (same rationale as the facade's own mapping).

    UNAVAILABLE from an STM point-get, left at ``MemoryResponse``'s own field default (documented,
    never faked with a guessed value) because ``MemoryItem`` (``mu_engine/storage/domain/
    memory.py:145-201``) carries no correlate for them today: ``speaker_kind``, ``speaker_id``,
    ``source_id``, ``turn_id``, ``entity_id``, ``asserted_state``, ``gds_pagerank`` (an LTM-graph-
    only aggregate), ``object_type``, ``object_value``, ``predicate_cardinality``, ``expires_at``,
    ``last_seen``, ``mention_count`` (an LTM-graph-only aggregate), ``parent_ids``/``child_ids``
    (LTM-graph-only lineage). Every other field below is a direct 1:1 read off ``MemoryItem``."""
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


def _render_context(items: Sequence[RecallItemView], *, max_chars: int | None) -> str:
    """Deterministic context assembly: one bullet per hit, truncated to ``max_chars`` (no LLM)."""
    lines = [f"- {item.content}" for item in items]
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text
