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

``promote``/``demote``/``share`` (build-queue §13 items 5/3) still have no engine-side verb to
delegate to — they still raise :class:`SurfaceVerbNotImplementedError`, a locally-defined
``MemoryUniverseError`` subclass Stage C's HTTP layer maps to HTTP 501. This is a deliberate, NAMED
gap (DEV-STANDARDS: never a silent no-op / fake success), not an oversight.
"""

from __future__ import annotations

import uuid
from typing import Any, NoReturn, Protocol

from mu_contracts.contracts.memory import MemoryResponse
from mu_contracts.contracts.recall import RecallChannels as CanonicalRecallChannels
from mu_contracts.contracts.recall import RecallItemView as CanonicalRecallItemView
from mu_contracts.contracts.recall import RecallResult as CanonicalRecallResult
from mu_contracts.contracts.views import ConsolidateView, ContextView, MemoryWriteResult
from mu_contracts.domain.errors import MemoryUniverseError
from mu_contracts.domain.model.memory import Tier as CanonicalTier
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.lifecycle.mode_gate import ManagerModeGate
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.distill import DistillActionKind, DistillPipeline, DistillReport
from mu_engine.providers._contracts import Message, MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.providers.model_router import ModelRouter
from mu_engine.services.ingest import IngestResult, IngestService
from mu_engine.services.recall.dto import RecallChannels as _EngineRecallChannels
from mu_engine.services.recall.dto import RecallQuery
from mu_engine.services.recall.dto import RecallResult as _EngineRecallResult
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
# ``add()``'s "caller expressed no opinion" fallback — mirrors ``LocalMemory``'s identical module
# constant (``mu-local/local_memory.py``, REMEDIATION Rank 2 / A6 fix): READ off
# ``IngestActivity.importance``'s own field default rather than a second hardcoded ``0.5`` literal
# (DEV-STANDARDS rule 3), and a plain ``float`` so mypy-strict can check the ``importance=`` kwarg
# without an ``**dict`` unpack (which mypy cannot verify against a ``BaseModel``'s field types).
_DEFAULT_IMPORTANCE: float = IngestActivity.model_fields["importance"].default


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
