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

DTOs (plan ruling 2 — "PROVISIONAL embedded DTOs, do not invent a new DTO now"): every verb here
returns the RAW ``mu_engine``-native result type the underlying application service already
produces (``IngestResult``, ``RecallResult``, ``MemoryItem``, ``DistillReport``) — never the
``mu_local.views`` projections (``MemoryWriteResult``/``MemoryListView``/``MemoryRecordView``/
``ConsolidateView``) ``LocalMemory`` wraps those in, since ``mu_local.views`` lives in a package
this module cannot import either. Which shape wins as the ONE canonical per-verb DTO (today's
engine-native shape, today's mu-local ``views`` shape, or a new one) is explicitly NOT decided
here — that is build-queue §13 item 3 / the parallel A4 decision task. Passing the engine-native
result straight through (no re-wrapping) is the smallest-footprint choice that prejudges nothing.

``promote``/``demote`` (build-queue §13 item 5) and the private-plane ``build_context``/shared-
plane ``share`` twins (§13 item 3, §2.5) have no engine-side implementation to delegate to yet (the
DTO/signature each would return is itself item 3's open decision) — they raise
:class:`SurfaceVerbNotImplementedError`, a locally-defined ``MemoryUniverseError`` subclass Stage B
(item 5) maps to HTTP 501. This is a deliberate, NAMED gap (DEV-STANDARDS: never a silent no-op /
fake success), not an oversight.
"""

from __future__ import annotations

import uuid
from typing import Any, NoReturn, Protocol

from mu_contracts.domain.errors import MemoryUniverseError
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.lifecycle.mode_gate import ManagerModeGate
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.distill import DistillPipeline, DistillReport
from mu_engine.providers._contracts import Message, MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.providers.model_router import ModelRouter
from mu_engine.services.ingest import IngestResult, IngestService
from mu_engine.services.recall.dto import RecallChannels, RecallQuery, RecallResult
from mu_engine.services.recall.service import RecallService
from mu_engine.storage.domain.memory import MemoryItem, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.ports import StmTierRepository

__all__ = [
    "LlmNotConfiguredError",
    "LocalContainerLike",
    "SurfaceFacade",
    "SurfaceVerbNotImplementedError",
]

_DEFAULT_USER = "default"
_DEFAULT_SESSION = "default"
# PORT of the reference SLM integration test's ANSWER-task system prompt (parity with mu-local's
# copy, mu-local/local_memory.py:71 / mu-engine/tests/pipelines/test_distill_llm_slm_int.py:393-
# 396) — a named module constant, never an inline literal at the call site (DEV-STANDARDS rule 3).
_ASK_SYSTEM_PROMPT = "Answer the question using ONLY the given facts. Be concise."


class SurfaceVerbNotImplementedError(MemoryUniverseError):
    """A canonical verb has no engine-side implementation to delegate to yet.

    Covers ``promote``/``demote`` (build-queue §13 item 5 — neither surface implements these
    today) and the ``build_context``/``share`` twins (§13 item 3 — their return-DTO shape is
    itself an open decision, §2.5). NAMED so Stage B (item 5) has ONE importable exception family
    to map onto HTTP 501 — never a bare ``NotImplementedError`` (which is a builtin, not part of
    the ``MemoryUniverseError`` wire-error hierarchy, CANONICAL §6) and never a silent no-op.
    """

    def __init__(self, verb: str, *, reason: str) -> None:
        self.verb = verb
        super().__init__(f"SurfaceFacade.{verb}() is not implemented: {reason}")


class LlmNotConfiguredError(MemoryUniverseError):
    """``ask()`` was called while the injected container carries no LLM (heuristic mode).

    A deliberate, NAMED duplicate of ``mu_local.errors.LlmNotConfiguredError`` (same refusal,
    ``mu-local/local_memory.py:238-254``) — duplicated rather than imported because this package
    may not import ``mu_local`` (module docstring). ``SurfaceFacade`` NEVER fabricates an
    empty/degraded synthesis (spec §7, T7), matching mu-local's discipline exactly. TO RECONCILE:
    build-queue §13 item 3 (or a dedicated follow-up) should fold these two classes into one home
    once mu-contracts is confirmed as the right place for it (mirrors ``BackendUnavailableError``,
    which mu-local already re-exports from mu-contracts rather than redefining, ``mu-local/
    errors.py:12``) — flagged here, not fixed, since this task owns only ``mu_engine/surface/``.
    """


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
    recall: RecallService
    mode_gate: ManagerModeGate
    llm: ModelRouter | None


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
    ) -> None:
        self._container = container
        self._workspace = workspace
        self._org = namespace

    # ------------------------------------------------------------------------------------- write
    async def add(
        self,
        content: str | dict[str, Any] | list[dict[str, Any]],
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> IngestResult:
        """Ingest one activity (STM durable -> deterministic STM->MTM promote). Mirrors
        ``LocalMemory.add`` (``mu-local/local_memory.py:105-136``) field-for-field, returning the
        engine-native :class:`IngestResult` unwrapped (module docstring, DTO ruling 2)."""
        ns = self._ns(user, session)
        last: IngestResult | None = None
        for message in _normalize_messages(content):
            activity = IngestActivity(
                namespace=ns,
                host=_host(),
                session_offset=_fresh_offset(),
                kind="user_message",
                text=message["content"],
                promote=True,
            )
            last = await self._container.ingest.remember(activity)
        if last is None:  # empty message list — fail loud, never a silent no-op
            raise ValueError("add() received no content to remember")
        return last

    async def consolidate(
        self,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = 50,
    ) -> DistillReport:
        """MTM->LTM consolidation (DISTILL). Mirrors ``LocalMemory.consolidate`` (``mu-local/
        local_memory.py:138-164``) — the SAME engine-side ``ManagerModeGate.assert_manual_allowed``
        pre-check (ADR 0031), so a MANAGED namespace refuses this manual verb loud here exactly as
        it does through ``LocalMemory``, never bypassable via this second surface."""
        ns = self._ns(user, session)
        self._container.mode_gate.assert_manual_allowed(ns, "consolidate")
        recent = await self._container.stm.recent(ns, limit=limit)
        window = [scored.item for scored in recent]
        return await self._container.distill.distill(ns, window)

    # -------------------------------------------------------------------------------------- read
    async def recall(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        tier: MemoryTier | None = None,
        limit: int = 10,
    ) -> RecallResult:
        """Federate-live RANKED recall. Mirrors ``LocalMemory.recall`` (``mu-local/
        local_memory.py:167-183``), returning the engine-native :class:`RecallResult` unwrapped."""
        ns = self._ns(user, session)
        scope = self._scope(user, session)
        q = RecallQuery(namespace=ns, text=query, limit=limit, channels=_channels_for_tier(tier))
        return await self._container.recall.recall(scope, q)

    async def get(
        self,
        memory_id: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> MemoryItem | None:
        """Point-get one memory by id from the caller's STM partition (``None`` if absent).
        Mirrors ``LocalMemory.get`` (``mu-local/local_memory.py:198-220``) — same phase-0 STM-only
        narrowing (MTM/LTM adapters expose no point-get yet)."""
        ns = self._ns(user, session)
        return await self._container.stm.get(ns, memory_id)

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

    # -------------------------------------------------------------------- TO BUILD (§13 item 3)
    async def build_context(
        self,
        query: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
        limit: int = 10,
        max_chars: int | None = None,
    ) -> NoReturn:
        """Private-plane context-window verb (design §2.5 REVIEW-2 FIX 1) — the portable
        ``MemoryClient`` twin of ``LocalMemory.context`` (``mu-local/local_memory.py:222-236``).

        The underlying op (recall + deterministic render) is trivial to re-derive, but its RETURN
        SHAPE is ``mu_local.views.ContextView`` — a ``mu_local``-owned type this module cannot
        import, and precisely the DTO question build-queue §13 item 3 has not yet settled (module
        docstring). Raising here rather than inventing an ad hoc shape now avoids prejudging that
        decision; do NOT read this as "the op doesn't exist" — it does, on ``LocalMemory``, today.
        """
        del query, user, session, limit, max_chars  # unused: signature parity pending item 3
        raise SurfaceVerbNotImplementedError(
            "build_context",
            reason=(
                "LocalMemory.context(...) exists (mu-local/local_memory.py:222) but its "
                "ContextView return shape is mu_local-owned and the canonical wire-twin DTO is "
                "undecided (build-queue §13 item 3) — TO BUILD once that decision lands"
            ),
        )

    async def promote(
        self,
        memory_id: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> NoReturn:
        """Neither ``LocalMemory`` nor ``MemoryClient`` implements ``promote`` today (design §9
        table, build-queue §13 item 5) — a named, honest 501, never a silent no-op."""
        del memory_id, user, session
        raise SurfaceVerbNotImplementedError(
            "promote", reason="no engine verb yet (build-queue §13 item 5) — maps to HTTP 501"
        )

    async def demote(
        self,
        memory_id: str,
        *,
        user: str = _DEFAULT_USER,
        session: str | None = None,
    ) -> NoReturn:
        """Neither surface implements ``demote`` today (design §9 table, build-queue §13 item 5)."""
        del memory_id, user, session
        raise SurfaceVerbNotImplementedError(
            "demote", reason="no engine verb yet (build-queue §13 item 5) — maps to HTTP 501"
        )

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


def _channels_for_tier(tier: MemoryTier | None) -> RecallChannels:
    """PORT of ``LocalMemory``'s private ``_channels_for_tier`` (``mu-local/local_memory.py:338-
    346``) — ``tier`` narrows recall to one channel; ``None`` runs all three."""
    if tier is None:
        return RecallChannels()
    return RecallChannels(
        stm=tier is MemoryTier.STM,
        mtm=tier is MemoryTier.MTM,
        ltm=tier is MemoryTier.LTM,
    )


def _render_facts(result: RecallResult) -> str:
    """Deterministic context assembly for ``ask()``'s prompt only (no LLM synthesis) — PORT of
    ``LocalMemory``'s private ``_render_context`` (``mu-local/local_memory.py:366-372``), collapsed
    to the one caller this facade has (``ask()``); NOT exposed as ``build_context``'s return value
    (that verb's shape is undecided, see its docstring above)."""
    return "\n".join(f"- {item.content}" for item in result.items)


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
