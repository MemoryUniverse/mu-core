"""``LocalContainer`` — the ONE composition root for the embedded LOCAL engine (spec §2.2).

PORT of ``mem0.Memory.__init__`` (``other_repos/mem0/mem0/memory/main.py:172-204``): build every
store + the embedder + the LLM from config in one place, then hand the wired graph to the facade.
Re-expressed over MU ports + the fail-loud registry, with mem0's ``graph optional`` branch
(``main.py:199-204``) REMOVED — graph is MANDATORY (CANONICAL storage; spec §2.2/§3.1). This is
the SOLE owner of the composition + the one default set (APPLY-PLAN B-4); ``mu-client``'s
``LocalContainer`` and the daemon ``EngineHost`` WRAP this, add no engine logic.

Backend SELECTION + mandatory-role validation reuse the phase-0 engine registry
(``mu_engine.storage.registry`` / ``factories.STORE_REGISTRY``). EVERY role — relational, kv,
vector, graph — is built THROUGH ``STORE_REGISTRY.build(role, backend, **cfg)`` (storage-rework
code review fix, 2026-07-27): the composition root never hand-instantiates a vendor client or
imports one directly; selecting a different backend is a ``StorageSettings`` config change, ZERO
engine change (the exact per-role EXTENSION SEAM documented in ``mu_engine.storage.factories``).
The container still owns deterministic async cleanup: it reaches into the returned adapter for
its underlying vendor client (:meth:`_resolve_closer` — the same reviewed pattern the relational
control-plane used before this fix, now shared by every role) so ``close()`` releases every
connection LIFO, no client leak, cancellation-safe. STM binds the kv-role client (redis/valkey by
default); the pipeline ledger is IN-PROCESS
(:class:`~mu_engine.pipelines.ledger.InMemoryStageLedger`, step (2)) — a cross-process durable
ledger is a daemon (``mu-client``) concern, not this daemonless single-session facade.

Central observability is wired here (DEV-STANDARDS rule 4): the composition root builds the three
content-free sinks (:class:`Tracer` / :class:`MetricSink` / :class:`AuditLog`) from
:class:`~mu_local.config.ObservabilitySettings` and threads them into every service, so ingest /
recall / distill emit spans, latency/error metrics and content-free audit rows on their meaningful
ops (no more silently-no-op sinks).

LLM WIRING (closed 2026-07-27, was the ``self.llm: object | None = None`` hardcoded seam): ``llm``
defaults to ``None`` ⇒ heuristic mode — UNCHANGED, backward compatible (ingest promotion and the
DISTILL extractor stay deterministic; ``ask``/adjudication still refuse loudly via
:class:`~mu_local.errors.LlmNotConfiguredError`). When ``storage.llm`` carries a
:class:`~mu_local.config.ModelProfileSettings`, this root builds a REAL
:class:`~mu_engine.providers.model_router.ModelRouter` — the SAME LOCAL_HTTP/OpenAI-compatible
catalog shape the reference integration test proved against the real docker SLM
(``mu-engine/tests/pipelines/test_distill_llm_slm_int.py``: ``SlmTestSettings`` + ``build_model_
router`` + ``LlmFactExtractor`` dropping in unchanged against ``ModelRouter`` because it satisfies
``LLMProviderPort`` structurally) — and threads it two places: (a) ``LlmFactExtractor(router, ...)``
replaces ``HeuristicSpoExtractor`` for the DISTILL pipeline's SPO extraction; (b) the router itself
is exposed as ``self.llm`` so ``LocalMemory.ask()`` can call ``router.generate(Task.ANSWER, ...)``.
No cloud reachability is assumed — building a ``ModelRouter`` for a LOCAL_HTTP deployment opens no
in-process weights (unlike the L5 warm-local singleton the embedder uses), so this stays a cheap,
synchronous construction here, same discipline as the embedder build above.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from mu_contracts.config import Settings, get_settings
from mu_contracts.domain.errors import MemoryUniverseError
from mu_contracts.domain.model.memory import Namespace
from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.lifecycle_lease import LifecycleLeasePort
from mu_contracts.ports.lifecycle_workflow import LifecycleWorkflowRunnerPort
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.conflict import (
    ConflictAdjudicator,
    InMemoryConflictRecordRepository,
    build_conflict_adjudicator,
)
from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate
from mu_engine.lifecycle.promotion import PromotionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings, ManagerModeSettings
from mu_engine.pipelines.distill import DistillPipeline, DistillSettings
from mu_engine.pipelines.ledger import InMemoryStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import build_audit, build_metrics, build_tracer
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.providers.catalog import ModelDeployment, ModelKind, ProviderKind, ProviderRecord
from mu_engine.providers.embedding import SentenceTransformerEmbedder, build_embedder
from mu_engine.providers.model_router import ModelRouter, build_model_router
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings, default_local_catalog
from mu_engine.services.extract import (
    ExtractionSettings,
    FactExtractorPort,
    HeuristicSpoExtractor,
    LlmFactExtractor,
)
from mu_engine.services.ingest import IngestService
from mu_engine.services.recall import (
    PrincipalAuthorizedIdsResolver,
    RecallAuthorizationFilter,
    RecallService,
    RecallSettings,
    ReciprocalRankFusion,
    ThreeChannelRecallRanker,
)
from mu_engine.storage.domain.namespace import Visibility
from mu_engine.storage.factories import STORE_REGISTRY
from mu_engine.storage.ports import GraphStorePort, MtmTierRepository, StmTierRepository
from mu_engine.storage.registry import assert_mandatory_roles
from mu_local.config import (
    BackendChoice,
    ModelProfileSettings,
    ObservabilitySettings,
    StorageSettings,
)
from mu_local.errors import BackendUnavailableError
from mu_local.shared_null import LocalNullSharedRecall

if TYPE_CHECKING:  # pragma: no cover — S1-03 sibling (mu_engine.lifecycle.manager), typing only.
    from mu_engine.lifecycle.manager import MemoryLifecycleManager, WarmRecallCacheServicePort

__all__ = ["LifecycleManagerUnavailableError", "LocalContainer"]


class LifecycleManagerUnavailableError(MemoryUniverseError):
    """Raised by :meth:`LocalContainer.build_lifecycle_manager` when the Stage-1 sibling module
    ``mu_engine.lifecycle.manager`` (S1-03, ``MemoryLifecycleManager`` itself) has not landed yet
    in this tree. A NAMED, fail-loud composition-root gap (DEV-STANDARDS: real deps or BLOCKED,
    never a silently-returned ``None``/stub) — that module is built in parallel with this task and
    this factory is coded against its CANONICAL spec'd constructor shape
    (``memory-lifecycle-manager-spec.md`` §17) ahead of its landing."""


class _WorkspaceDefaultModeResolver:
    """The ONE :class:`ModePolicyResolver` mu-local composes today (spec §17a).

    Resolves ONLY the workspace-default tier of the **memory ▷ namespace ▷ workspace-default**
    order ADR 0031/spec §3 name: mu-local has no per-namespace/per-memory manager-mode override
    store yet (a tracked, named narrowing — every namespace in one embedded process resolves to
    the SAME configured default). Widening to the full order is a resolver-only change; it never
    touches :class:`ManagerModeGate` itself, which stays a pure decision function over whatever
    mode this resolver returns.
    """

    def __init__(self, settings: ManagerModeSettings) -> None:
        self._mode = ManagerMode(settings.default_mode)

    def resolve(self, ns: Namespace) -> ManagerMode:
        return self._mode


# The backends the registry ships, per role (mu_engine.storage.factories) — selecting anything
# else is a NAMED fail-loud refusal (spec §7), never a silent fallback to a different backend.
# Widened 2026-07-27 (storage-rework review FIX-1): every registry-shipped backend is reachable
# by config now, not just the phase-0 default — adding one more here is the ONLY change needed
# when the registry grows a new (role, backend) factory (mem0-pattern multi-backend, owner stage
# 2026-07-27).
_SUPPORTED_KV = frozenset({"redis", "valkey", "memory", "memcached"})
_SUPPORTED_VECTOR = frozenset({"qdrant", "pgvector", "chroma", "faiss"})
_SUPPORTED_GRAPH = frozenset({"falkordb"})
_SUPPORTED_RELATIONAL = frozenset({"sqlite", "postgres", "mysql"})
_SUPPORTED_EMBEDDING = frozenset({"minilm_local"})


def _build_llm_catalog(profile: ModelProfileSettings) -> tuple[ModelSettings, ModelCatalogSettings]:
    """Layer ONE ``ProviderKind.LOCAL_HTTP`` deployment onto the real ``default_local_catalog()``
    base (untouched offline MiniLM embedder) — the SAME shape the reference SLM integration test
    builds (``mu-engine/tests/pipelines/test_distill_llm_slm_int.py:163-196``,
    ``_build_slm_catalog``). Every ``ModelSettings`` task field points at ``profile.model_group`` so
    the ONE deployment satisfies the registry's per-task validation (``registry.py:79-88``)."""
    provider = ProviderRecord(
        key=profile.provider_key,
        kind=ProviderKind.LOCAL_HTTP,
        litellm_provider=profile.provider,
        api_base=profile.base_url,
        is_local=True,
    )
    deployment = ModelDeployment(
        model_group=profile.model_group,
        provider_key=profile.provider_key,
        model_id=f"{profile.provider}/{profile.model}",
        kind=ModelKind.LLM,
        extra_params={"api_key": profile.api_key},  # passthrough seam (catalog.py:94), not a secret
    )
    base = default_local_catalog()
    catalog = base.model_copy(update={"providers": [provider], "deployments": [deployment]})
    models = ModelSettings(
        provider=profile.provider_key,
        answer_model=profile.model_group,
        adjudicate_model=profile.model_group,
        hard_extract_model=profile.model_group,
        routine_extract_model=profile.model_group,
        summarize_model=profile.model_group,
        classify_model=profile.model_group,
        rerank_model=profile.model_group,
        max_output_tokens=profile.max_tokens,
        temperature=profile.temperature,
    )
    return models, catalog


class LocalContainer:
    """The ONE place adapters bind for the embedded LOCAL engine (spec §2.2)."""

    def __init__(
        self,
        storage: StorageSettings,
        *,
        settings: Settings | None = None,
        observability: ObservabilitySettings | None = None,
        lifecycle: LifecycleSettings | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._closers: list[Callable[[], Awaitable[None]]] = []
        self._obs: ObservabilitySettings = observability or ObservabilitySettings()

        # (0) fail-loud mandatory-role + graph-not-folded validation (reuse the engine registry).
        assert_mandatory_roles(
            {
                "relational": storage.relational.backend,
                "vector": storage.vector.backend,
                "graph": storage.graph.backend,
            },
            plane=Visibility.PRIVATE,
        )
        self._assert_supported(storage)

        # (1) the REAL local embedder (offline MiniLM); dimension is read FROM the live model,
        #     never from config (storage-indexing-design §239-240) — the vector store binds to it.
        self.embedder: SentenceTransformerEmbedder = self._build_embedder(storage.embedding)

        # (2) STM (kv role: redis by default) — built through the STORE_REGISTRY seam, exactly
        #     like every other role (spec §4.2). The durable KV write is the facade's durability
        #     floor; the pipeline ledger is IN-PROCESS (InMemoryStageLedger) — a cross-PROCESS
        #     durable ledger is a daemon concern (mu-client), not this daemonless single-session
        #     facade. An in-process ledger also keeps promote idempotency scoped to THIS instance,
        #     so a content_hash (which the engine keys content-only, not per-η) never collides
        #     across tenants/runs.
        self.stm: StmTierRepository = STORE_REGISTRY.build(
            "kv", storage.kv.backend, **self._kv_cfg(storage.kv)
        )
        self._register_closer(self.stm, "_redis", "_mc")
        self._ledger = InMemoryStageLedger()

        # (3) MTM (vector role: qdrant by default) — dim from the LIVE embedder, built through
        #     the same STORE_REGISTRY seam.
        self.mtm: MtmTierRepository = STORE_REGISTRY.build(
            "vector",
            storage.vector.backend,
            dim=self.embedder.dimension,
            **self._vector_cfg(storage.vector),
        )
        self._register_closer(self.mtm, "_qdrant")

        # (4) LTM (graph role: falkordb, the MANDATORY graph engine) — same seam.
        self.ltm: GraphStorePort = STORE_REGISTRY.build(
            "graph", storage.graph.backend, **self._graph_cfg(storage.graph)
        )
        self._register_closer(self.ltm, "_db.connection")

        # (5) control-plane (relational) — built through the STORE_REGISTRY seam (reuse). Off the
        #     ingest/recall critical path this slice; disposed on close.
        self.control = STORE_REGISTRY.build(
            "relational", storage.relational.backend, **self._relational_cfg(storage.relational)
        )
        self._register_closer(self.control, "_engine")

        # (6) LLM: None (default) ⇒ heuristic mode, unchanged; configured ⇒ a REAL ModelRouter
        #     (the reference SLM-integration catalog shape) + the LlmFactExtractor it feeds DISTILL.
        self.llm: ModelRouter | None = None
        self._extractor: FactExtractorPort = HeuristicSpoExtractor()
        if storage.llm is not None:
            self.llm = self._build_llm_router(storage.llm)
            self._extractor = LlmFactExtractor(
                self.llm,
                model_group=storage.llm.model_group,
                settings=ExtractionSettings(
                    max_tokens=storage.llm.max_tokens, temperature=storage.llm.temperature
                ),
            )

        # (7) platform singletons + the three content-free observability sinks (DEV-STANDARDS
        #     rule 4) — built once here from ObservabilitySettings and threaded into every service.
        self._clock = SystemClock()
        self._bus = InprocBus()
        self.tracer: Tracer = build_tracer(enabled=self._obs.otel_enabled, service_name="mu-local")
        self.metrics: MetricSink = build_metrics(enabled=self._obs.metrics_enabled)
        self.audit: AuditLog = build_audit(enabled=self._obs.audit_enabled)

        # (7b) lifecycle central-config + the engine-side manager-mode gate (ADR 0031; spec §3;
        #      S0-03/S0-07). ``LocalMemory.consolidate()`` calls ``self.mode_gate.assert_manual_
        #      allowed(ns, "consolidate")`` before delegating — this container is the ONE place
        #      that builds it (DEV-STANDARDS rule 9), never hand-wired at the facade. Composed
        #      against the SAME ``ModePolicyResolver`` seam (memory ▷ namespace ▷ workspace-default
        #      resolution order, spec §17a) mu-local resolves today — see
        #      ``_WorkspaceDefaultModeResolver`` above for the tracked per-namespace/per-memory
        #      override narrowing.
        #
        #      Default narrowed to MANUAL here (not ``ManagerModeSettings()``'s bare MANAGED
        #      default, spec §16): mu-local ships NO automatic sweep in this Stage — there is no
        #      ``MaintenanceLoop``/daemon (that is mu-client's S1-07) — so a bare MANAGED default
        #      would refuse every manual ``consolidate()`` with nothing running the auto sweep in
        #      its place, silently breaking "daemonless: the caller drives the sweep" (module
        #      docstring). A caller opts into MANAGED/HYBRID explicitly via
        #      ``LocalContainer(..., lifecycle=LifecycleSettings(manager_mode=ManagerModeSettings(
        #      default_mode="managed")))`` — this default is a mu-local composition-root choice,
        #      not a re-derivation of the canonical settings default (that default is untouched).
        self.lifecycle_settings: LifecycleSettings = lifecycle or LifecycleSettings(
            manager_mode=ManagerModeSettings(default_mode=ManagerMode.MANUAL.value)
        )
        self.mode_gate: ManagerModeGate = ManagerModeGate(
            self.lifecycle_settings.manager_mode,
            _WorkspaceDefaultModeResolver(self.lifecycle_settings.manager_mode),
        )

        # (7c) S3-01 (spec §8, ADR 0037): the LLM-judged conflict adjudicator, gated the SAME way
        #      the DISTILL extractor above is (``storage.llm is not None``) — a configured model
        #      profile gets a REAL ``ConflictAdjudicator`` over the SAME ``ModelRouter``
        #      (``self.llm``, routes ``Task.CONFLICT_ADJUDICATION`` -> ``models.adjudicate_model``,
        #      ADR 0037); heuristic mode (``llm=None``) builds none, and ``DistillPipeline``'s own
        #      no-adjudicator degrade floor (``_heuristic_only_verdict``) applies verbatim — 100%
        #      pre-ADR-0037 backward compatible. ``DistillSettings.use_llm_adjudicator`` (default
        #      True) is the composition-root-read gate the settings' own docstring calls for
        #      (mirrors ``use_llm_extractor`` -> ``build_extractor`` precedent) — never read
        #      internally by ``DistillPipeline`` itself. ``InMemoryConflictRecordRepository`` is
        #      the sanctioned LOCAL-plane conflict-inbox default (a real in-process adapter, not a
        #      mock) for a PENDING/MANUAL-parked verdict.
        self._distill_settings = DistillSettings()
        self.conflict_adjudicator: ConflictAdjudicator | None = None
        if self.llm is not None and self._distill_settings.use_llm_adjudicator:
            self.conflict_adjudicator = build_conflict_adjudicator(
                use_llm=True,
                router=self.llm,
                clock=self._clock,
                bus=self._bus,
                conflict_records=InMemoryConflictRecordRepository(),
            )

        # (8) application services — each facade verb delegates to exactly one of these; each gets
        #     the wired sinks so its meaningful op emits spans/metrics/audit (never silently no-op).
        self.ingest = IngestService(
            stm=self.stm,
            mtm=self.mtm,
            embedder=self.embedder,
            bus=self._bus,
            ledger=self._ledger,
            clock=self._clock,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
        )
        self.distill = DistillPipeline(
            ltm=self.ltm,
            extractor=self._extractor,
            clock=self._clock,
            settings=self._distill_settings,
            mtm=self.mtm,
            bus=self._bus,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
            adjudicator=self.conflict_adjudicator,
        )
        recall_settings = RecallSettings()
        fusion = ReciprocalRankFusion()
        ranker = ThreeChannelRecallRanker(
            stm=self.stm,
            mtm=self.mtm,
            ltm=self.ltm,
            fusion=fusion,
            settings=recall_settings,
            clock=self._clock,
        )
        authz = RecallAuthorizationFilter(
            tenancy=DefaultTenancyGuard(), authorized_ids=PrincipalAuthorizedIdsResolver()
        )
        self.recall = RecallService(
            embedder=self.embedder,
            private_ranker=ranker,
            shared_recall=LocalNullSharedRecall(clock=self._clock),
            authz=authz,
            fusion=fusion,
            settings=recall_settings,
            clock=self._clock,
            metrics=self.metrics,
            tracer=self.tracer,
        )

    async def close(self) -> None:
        """Release every store connection this container opened (LIFO), best-effort per client."""
        for closer in reversed(self._closers):
            with contextlib.suppress(Exception):  # teardown must not mask the primary error
                await closer()
        self._closers.clear()

    @property
    def bus(self) -> EventBusPort:
        """The SAME real ``InprocBus`` instance threaded into ``IngestService``/``DistillPipeline``
        above (integrate-phase accessor, Stage-1 wiring) — a caller that needs to subscribe to
        THIS container's real event stream (e.g. mu-client's daemon ``MaintenanceLoop``, S1-07)
        must observe the identical bus captured memories publish onto, never a second,
        independently-constructed ``InprocBus`` that never receives a real event (DEV-STANDARDS
        rule 9: one composition root, one bus per plane)."""
        return self._bus

    # ---------------------------------------------------------------------- MLM composition (S1-05)
    def build_lifecycle_manager(
        self,
        *,
        lease: LifecycleLeasePort | None = None,
        runner: LifecycleWorkflowRunnerPort | None = None,
        clock: Clock | None = None,
        warm_cache: WarmRecallCacheServicePort | None = None,
    ) -> MemoryLifecycleManager:
        """Construct a :class:`~mu_engine.lifecycle.manager.MemoryLifecycleManager` wired against
        the SAME ``stm``/``mtm``/``ltm``/``distill``/``bus``/``clock`` instances this container
        already built (ADR 0029 mu-local half; ADR 0031 wiring half) — never a second,
        independently-constructed set (DEV-STANDARDS rule 9: ``LocalContainer`` stays the ONE
        place adapters bind). In particular ``distill=self.distill`` is the identical
        :class:`~mu_engine.pipelines.distill.DistillPipeline` object
        :meth:`mu_local.local_memory.LocalMemory.consolidate` calls — an ``id()`` identity, not a
        second pipeline pointed at the same stores.

        ``PromotionService``/``DemotionService`` (S1-01/S1-02) have landed and are imported
        eagerly (module scope, above); ``MemoryLifecycleManager`` itself (S1-03) is the ONE
        sibling still mid-flight in parallel with this task — this factory is coded against its
        CANONICAL spec'd constructor shape (``memory-lifecycle-manager-spec.md`` §17) ahead of its
        landing (dev-context: code against the spec'd interface, not a guess). Its import is
        deferred to call time (never at module scope) so ``mu_local.composition`` stays importable
        for every OTHER consumer of ``LocalContainer`` regardless of its merge state; still
        missing, it fails LOUD here with :class:`LifecycleManagerUnavailableError`, never a
        silent stub.

        WIRING NOTE (Stage-3 integrate, closed): ``conflict=self.conflict_adjudicator`` threads the
        SAME ``ConflictAdjudicator`` (or ``None`` in heuristic mode) this container already built
        for ``self.distill`` — never a second instance. ``RetentionService`` (Stage-2, S2-01)
        remains out of scope for this daemonless embedded facade — ``retention`` stays ``None``
        (no-op) here; a future task that wants it wired threads it the same way. ``warm_cache`` is
        an optional passthrough kwarg (mirrors ``lease``/``runner``/``clock`` below) so mu-client's
        daemon can thread its real ``RecallInjectBridge`` (S3-02, the ``WarmRecallCacheServicePort``
        implementation) through this SAME composition root; mu-local's own daemonless callers omit
        it and get the ``None`` no-op default (``ready_context`` stays the honest ``wired=False``
        stub). Concrete ``LifecycleLeasePort``/``LifecycleWorkflowRunnerPort`` adapters (S1-06,
        mu-client-owned: ``SqliteWalLeaseAdapter``/``SqliteWalRunner``) are ACCEPTED as optional
        passthrough kwargs (``lease``/``runner``/``clock``) so mu-client's daemon (the ONE caller
        with a real cross-process runner) can thread its own durable adapters through this SAME
        composition root instead of re-deriving ``salience``/``promotion``/``demotion`` a second
        time — mu-local's own daemonless callers simply omit them and get the in-process defaults
        (``MemoryLifecycleManager``'s own ``_InProcessLifecycleLease``/``_InlineLifecycleRunner``).
        """
        try:
            from mu_engine.lifecycle.manager import MemoryLifecycleManager
        except ImportError as exc:
            raise LifecycleManagerUnavailableError(
                "MemoryLifecycleManager composition requires mu_engine.lifecycle.manager "
                "(S1-03) — not yet landed in this tree"
            ) from exc

        salience = SalienceStrategy(self.lifecycle_settings.salience)
        promotion = PromotionService(
            mtm=self.mtm,
            distill=self.distill,  # SAME object LocalMemory.consolidate() delegates to
            salience=salience,
            stm=self.stm,
            settings=self.lifecycle_settings,
            clock=self._clock,
            bus=self._bus,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
        )
        demotion = DemotionService(
            stm=self.stm,
            mtm_remove=self.mtm,  # CF-2: the real MtmTierRepository.remove — no local shim needed
            salience=salience,
            settings=self.lifecycle_settings,
            clock=self._clock,
            bus=self._bus,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
        )
        return MemoryLifecycleManager(
            salience=salience,
            promotion=promotion,
            demotion=demotion,
            distill=self.distill,  # SAME object LocalMemory.consolidate() delegates to
            conflict=self.conflict_adjudicator,  # SAME instance self.distill was built with
            mode_gate=self.mode_gate,
            bus=self._bus,
            settings=self.lifecycle_settings,
            lease=lease,
            runner=runner,
            clock=clock or self._clock,
            warm_cache=warm_cache,
        )

    # ---------------------------------------------------------------- backend guards + resolution
    @staticmethod
    def _assert_supported(storage: StorageSettings) -> None:
        """NAMED fail-loud refusal of a backend the STORE_REGISTRY does not ship for this role
        (spec §7). Widened 2026-07-27 (storage-rework review FIX-1) to the full registry surface
        per role — a rejection here now means the backend key itself is unregistered (e.g. a
        future graph driver beyond ``falkordb``), never that multi-backend support is missing.
        """
        checks = (
            ("kv", storage.kv.backend, _SUPPORTED_KV),
            ("vector", storage.vector.backend, _SUPPORTED_VECTOR),
            ("graph", storage.graph.backend, _SUPPORTED_GRAPH),
            ("relational", storage.relational.backend, _SUPPORTED_RELATIONAL),
            ("embedding", storage.embedding.backend, _SUPPORTED_EMBEDDING),
        )
        for role, backend, supported in checks:
            if backend not in supported:
                raise BackendUnavailableError(
                    f"role {role!r} backend {backend!r} is not registered in the STORE_REGISTRY "
                    f"(available: {sorted(supported)}); a not-yet-built driver (e.g. a graph "
                    "backend beyond falkordb — see the EXTENSION SEAM in "
                    "mu_engine.storage.factories) is a tracked gap — no silent fallback"
                )

    def _build_embedder(self, choice: BackendChoice) -> SentenceTransformerEmbedder:
        embedder = build_embedder(choice.backend, default_local_catalog())
        if not isinstance(embedder, SentenceTransformerEmbedder):  # fail-loud, never a silent None
            raise BackendUnavailableError(
                f"embedding backend {choice.backend!r} did not resolve to a local embedder"
            )
        return embedder

    @staticmethod
    def _build_llm_router(profile: ModelProfileSettings) -> ModelRouter:
        """Build the REAL ``ModelRouter`` for a configured LLM profile via the ONE model-layer
        composition entry point (``build_model_router``, model-router.py:302)."""
        models, catalog = _build_llm_catalog(profile)
        return build_model_router(models=models, catalog=catalog)

    def _kv_cfg(self, choice: BackendChoice) -> dict[str, Any]:
        """``STORE_REGISTRY.build("kv", ...)`` kwargs (spec §3.2). Only the ``redis`` factory has
        no internal Settings fallback for ``url`` (``mu_engine.storage.factories._build_redis``);
        every other kv backend (``valkey``/``memory``/``memcached``) resolves its own knobs from
        the central Settings tree when ``cfg`` doesn't supply them, so an empty override dict
        preserves that factory-owned default (DEV-STANDARDS rule 3)."""
        if choice.backend == "redis":
            return {"url": self._redis_url(choice)}
        return dict(choice.config)

    def _vector_cfg(self, choice: BackendChoice) -> dict[str, Any]:
        """``STORE_REGISTRY.build("vector", ...)`` kwargs (spec §3.3), ``dim`` supplied by the
        caller from the LIVE embedder. Only the ``qdrant`` factory has no internal Settings
        fallback for ``url``; ``pgvector``/``chroma``/``faiss`` fall back to their own Settings
        subtree exactly like ``valkey``/``memcached`` above."""
        if choice.backend == "qdrant":
            return {"url": self._qdrant_url(choice)}
        return dict(choice.config)

    def _graph_cfg(self, choice: BackendChoice) -> dict[str, Any]:
        """``STORE_REGISTRY.build("graph", ...)`` kwargs — the ``falkordb`` factory requires
        ``host``/``port`` explicitly (no internal Settings fallback), so this always resolves
        them from ``choice.config`` or the central ``GraphDBSettings``."""
        host, port = self._falkor_endpoint(choice)
        return {"host": host, "port": port}

    def _redis_url(self, choice: BackendChoice) -> str:
        url = choice.config.get("url")
        return str(url) if url else self._settings.storage.cache.url

    def _qdrant_url(self, choice: BackendChoice) -> str:
        url = choice.config.get("url")
        return str(url) if url else self._settings.storage.vector.url

    def _falkor_endpoint(self, choice: BackendChoice) -> tuple[str, int]:
        host = choice.config.get("host") or self._settings.storage.graph.host
        port = choice.config.get("port") or self._settings.storage.graph.port
        return str(host), int(port)

    def _relational_cfg(self, choice: BackendChoice) -> dict[str, str]:
        if "dsn" in choice.config:
            return {"dsn": str(choice.config["dsn"])}
        if choice.backend == "postgres":
            return {"dsn": self._settings.storage.postgres.dsn}
        if choice.backend == "mysql":
            return {"dsn": self._settings.storage.mysql.dsn}
        return {}  # sqlite factory default = in-memory

    def _register_closer(self, adapter: object, *client_paths: str) -> None:
        """Append ``adapter``'s underlying vendor-client closer to the LIFO teardown list
        (``self._closers``), if it has one.

        Registry factories return adapters, not raw clients (spec §4.2) — the container still
        owns deterministic async cleanup (DEV-STANDARDS resource management: no client leaks,
        cancellation-safe). This generalizes the ``getattr(self.control, "_engine", None)``
        pattern the relational control-plane already used (the reviewed "done right" wiring,
        code review FIX-1) to every role: it prefers an adapter that manages its own client
        lifecycle (e.g. ``PgVectorMtmAdapter.close`` for its lazily-created pool), then falls
        back to the first ``client_paths`` dotted attribute chain that resolves to a live vendor
        client (e.g. ``"_db.connection"`` unwraps the FalkorDB async client's own connection).
        Embedded backends with no network client (``memory``/``chroma``/``faiss``) simply yield
        nothing to close — never an error.
        """
        closer = self._resolve_closer(adapter, *client_paths)
        if closer is not None:
            self._closers.append(closer)

    @staticmethod
    def _resolve_closer(
        adapter: object, *client_paths: str
    ) -> Callable[[], Awaitable[None]] | None:
        """Best-effort lookup of an ``aclose``/``close``/``dispose`` callable — first on
        ``adapter`` itself, then on whichever ``client_paths`` chain resolves to a live object."""
        candidates: list[object] = [adapter]
        for path in client_paths:
            obj: Any = adapter
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                candidates.append(obj)
        for candidate in candidates:
            for method_name in ("aclose", "close", "dispose"):
                method = getattr(candidate, method_name, None)
                if method is not None:
                    return method  # type: ignore[no-any-return]
        return None
