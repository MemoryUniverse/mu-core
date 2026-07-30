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
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.lifecycle_lease import LifecycleLeasePort
from mu_contracts.ports.lifecycle_workflow import LifecycleWorkflowRunnerPort
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.providers._contracts import Message, MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.services.ingest import IngestResult
from mu_engine.services.recall.dto import RecallChannels, RecallQuery, RecallResult
from mu_engine.storage.domain.memory import MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_local.composition import LocalContainer
from mu_local.config import StorageSettings
from mu_local.errors import LlmNotConfiguredError
from mu_local.views import (
    ConsolidateView,
    ContextView,
    MemoryListView,
    MemoryRecordView,
    MemoryWriteResult,
)

if TYPE_CHECKING:  # pragma: no cover — typing only, avoids a hard import-time cycle.
    from mu_engine.lifecycle.manager import MemoryLifecycleManager

__all__ = ["LocalMemory"]

_DEFAULT_USER = "default"
_DEFAULT_SESSION = "default"
_HOST = "mu-local"
# PORT of the reference SLM integration test's ANSWER-task system prompt
# (mu-engine/tests/pipelines/test_distill_llm_slm_int.py:393-396) — a named module constant, never
# an inline literal at the call site (DEV-STANDARDS rule 3).
_ASK_SYSTEM_PROMPT = "Answer the question using ONLY the given facts. Be concise."


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
    ) -> MemoryWriteResult:
        """Ingest one activity (STM durable -> deterministic STM->MTM promote). ``content`` accepts
        ``str | dict | list[dict]`` and normalises to messages (mem0 ``client/main.py:153-160``).
        Each message becomes one INGEST activity keyed by a fresh source offset (a genuine new
        occurrence, CANONICAL §8-M12); MTM promotion is idempotent on ``content_hash``."""
        ns = self._ns(user, session)
        last: IngestResult | None = None
        for message in _normalize_messages(content):
            activity = IngestActivity(
                namespace=ns,
                host=_HOST,
                session_offset=uuid.uuid4().hex,  # unique ⇒ never a pure M12 replay
                kind="user_message",
                text=message["content"],
                promote=True,  # an explicit LOCAL add is a salient assertion → STM->MTM (mem0
                #                 add() always persists to the vector store, main.py:281)
            )
            last = await self._container.ingest.remember(activity)
        if last is None:  # empty message list — fail loud, never a silent no-op
            raise ValueError("add() received no content to remember")
        return MemoryWriteResult(
            memory_id=last.memory_id,
            content_hash=last.content_hash,
            promoted=last.promoted,
            tiers_written=last.tiers_written,
        )

    async def consolidate(
        self,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = 50,
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
        )

    # ----------------------------------------------------------------------------- read
    async def recall(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        tier: MemoryTier | None = None,
        limit: int = 10,
    ) -> MemoryListView:
        """Federate-live RANKED recall over the private-own partition (STM floor ⊕ MTM dense ⊕ LTM
        graph), fused once. LOCAL mode has no shared arm (spec §3.2). ``tier`` narrows to one
        channel; ``None`` runs all three."""
        ns = self._ns(user, session)
        scope = self._scope(user, session)
        q = RecallQuery(namespace=ns, text=query, limit=limit, channels=_channels_for_tier(tier))
        result = await self._container.recall.recall(scope, q)
        return _to_list_view(result)

    async def search(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        tier: MemoryTier | None = None,
        limit: int = 10,
    ) -> MemoryListView:
        """mem0 muscle-memory alias for :meth:`recall` (spec verb-alias policy, §CC-6): ``recall``
        is the canonical read verb; ``search`` is a documented alias, one single behaviour."""
        return await self.recall(query, user=user, session=session, tier=tier, limit=limit)

    async def get(
        self,
        memory_id: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryRecordView | None:
        """Point-get one memory by id from the caller's STM partition (``None`` if absent). NOTE:
        the phase-0 MTM/LTM adapters expose no point-get, so this reads the STM tier only — a
        tracked, explicit narrowing (the ``user``/``session`` args name the η partition, since a
        bare id cannot be located across prefix-partitioned stores)."""
        ns = self._ns(user, session)
        item = await self._container.stm.get(ns, memory_id)
        if item is None:
            return None
        return MemoryRecordView(
            memory_id=item.id,
            content=item.content,
            tier=item.tier.value,
            channel="stm",
            score=0.0,
            is_floor=True,
        )

    async def context(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = 10,
        max_chars: int | None = None,
    ) -> ContextView:
        """Assemble a context window from recalled hits by DETERMINISTIC concatenation (no LLM
        synthesis — spec §7 INJECT render, Azure PARKED). The heuristic assembly folds behind an
        injected renderer once synthesis lands, without changing recall's core."""
        listing = await self.recall(query, user=user, session=session, limit=limit)
        text = _render_context(listing.items, max_chars=max_chars)
        return ContextView(text=text, items=listing.items, degraded=listing.degraded)

    async def ask(
        self,
        question: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = 10,
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

    # ----------------------------------------------------------------------------- lifecycle
    @property
    def bus(self) -> EventBusPort:
        """The REAL ``InprocBus`` this facade's own ``ingest``/``distill`` publish onto
        (integrate-phase accessor — ``mu_local.composition.LocalContainer.bus``). A caller that
        needs to observe THIS instance's real event stream (e.g. mu-client's daemon
        ``MaintenanceLoop``) subscribes here, never to a second, independently-constructed bus."""
        return self._container.bus

    def build_lifecycle_manager(
        self,
        *,
        lease: LifecycleLeasePort | None = None,
        runner: LifecycleWorkflowRunnerPort | None = None,
        clock: Clock | None = None,
    ) -> MemoryLifecycleManager:
        """Passthrough to :meth:`mu_local.composition.LocalContainer.build_lifecycle_manager` —
        constructs a :class:`~mu_engine.lifecycle.manager.MemoryLifecycleManager` wired against
        THIS instance's own stores/distill/bus (never a second, independently-constructed set).
        ``lease``/``runner``/``clock`` are optional passthrough kwargs so a caller with real
        cross-process adapters (mu-client's ``SqliteWalLeaseAdapter``/``SqliteWalRunner``, S1-06)
        can thread them through this SAME composition root."""
        return self._container.build_lifecycle_manager(lease=lease, runner=runner, clock=clock)

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


def _channels_for_tier(tier: MemoryTier | None) -> RecallChannels:
    """A ``tier`` narrows recall to one channel; ``None`` runs all three (STM/MTM/LTM)."""
    if tier is None:
        return RecallChannels()
    return RecallChannels(
        stm=tier is MemoryTier.STM,
        mtm=tier is MemoryTier.MTM,
        ltm=tier is MemoryTier.LTM,
    )


def _to_list_view(result: RecallResult) -> MemoryListView:
    return MemoryListView(
        items=[
            MemoryRecordView(
                memory_id=item.memory_id,
                content=item.content,
                tier=item.tier.value,
                channel=item.channel,
                score=item.fused_score,
                is_floor=item.is_floor,
            )
            for item in result.items
        ],
        degraded=result.degraded.value if result.degraded is not None else None,
    )


def _render_context(items: Sequence[MemoryRecordView], *, max_chars: int | None) -> str:
    """Deterministic context assembly: one bullet per hit, truncated to ``max_chars`` (no LLM)."""
    lines = [f"- {item.content}" for item in items]
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars]
    return text
