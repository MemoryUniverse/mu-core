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

MODEL-LAYER WIRING (2026-08-28, ENG-115a / gate G6). ``self.model_router`` is now built on EVERY
container, from the SHIPPED multi-provider catalog
(:func:`mu_engine.providers.plane.build_plane_router`): anthropic · openai · azure · deepseek ·
moonshot · a keyless local chat endpoint · a local embed+rerank endpoint, narrowed by a credential
probe over the secret seam so only the providers whose keys actually resolve stay in. Before this,
``ModelCatalogSettings()`` was empty and this root only ever called ``default_local_catalog()`` —
the plane shipped ONE embedder and ZERO LLM deployments, and building a router from the wired
defaults raised ``RegistryError: task 'answer' maps to model-group 'gpt-5-chat' which has no
deployment``. ``self.llm`` (and therefore ``LocalMemory``'s LLM-dependent verbs) is unchanged:
still ``None`` unless ``storage.llm`` is configured.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

import structlog

from mu_contracts.config import Settings, get_settings
from mu_contracts.domain.errors import MemoryUniverseError
from mu_contracts.domain.model.memory import Namespace, Tier
from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.governance import ConflictRecordRepository
from mu_contracts.ports.lifecycle_lease import LifecycleLeasePort
from mu_contracts.ports.lifecycle_workflow import LifecycleWorkflowRunnerPort
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.config import EngineSettings, get_engine_settings
from mu_engine.lifecycle.conflict import (
    ConflictAdjudicator,
    InMemoryConflictRecordRepository,
    build_conflict_adjudicator,
    conflict_adjudicator_settings_from_lifecycle,
)
from mu_engine.lifecycle.counts import TierCountCache, TierCountSettings
from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate
from mu_engine.lifecycle.promotion import PromotionService
from mu_engine.lifecycle.retention import RetentionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import LifecycleSettings, ManagerModeSettings
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.pipelines.ledger import InMemoryStageLedger, StageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import build_audit, build_metrics, build_tracer
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.providers.catalog import ModelDeployment, ModelKind, ProviderKind, ProviderRecord
from mu_engine.providers.embedding import build_embedder
from mu_engine.providers.model_router import ModelRouter, build_model_router
from mu_engine.providers.plane import (
    PlaneModelLayer,
    build_plane_router,
    build_plane_secret_resolver,
    resolve_plane_model_layer,
)
from mu_engine.providers.secrets import SecretSeamResolver
from mu_engine.providers.settings import (
    CatalogSource,
    ModelCatalogSettings,
    ModelSettings,
    default_local_catalog,
)
from mu_engine.providers.sparse_encoder import build_sparse_encoder
from mu_engine.services.extract import (
    FactExtractorPort,
    HeuristicSpoExtractor,
    LlmFactExtractor,
)
from mu_engine.services.health.assessor import HeuristicV1Assessor
from mu_engine.services.health.conflict_edges import PendingConflictEdgeReader
from mu_engine.services.health.service import MemoryHealthService
from mu_engine.services.health.settings import HealthSettings
from mu_engine.services.ingest import IngestService
from mu_engine.services.memory.repository import TieredMemoryRepository
from mu_engine.services.memory.router import TierLeg, TierRouter
from mu_engine.services.persona import PersonaWiring, build_persona
from mu_engine.services.pin.service import PinService
from mu_engine.services.pin.settings import PinSettings
from mu_engine.services.recall import (
    PrincipalAuthorizedIdsResolver,
    RecallAuthorizationFilter,
    RecallService,
    ReciprocalRankFusion,
    ThreeChannelRecallRanker,
)
from mu_engine.storage.domain.namespace import Visibility
from mu_engine.storage.factories import STORE_REGISTRY
from mu_engine.storage.ports import (
    ContextRepository,
    GraphStorePort,
    MtmTierRepository,
    StmTierRepository,
)
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

_log = structlog.get_logger("mu_local.composition")


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
# "minilm_vm_http" must match `ModelCatalogSettings.http_embed_backend_key`'s default (settings.py)
# — the key `default_local_catalog` registers the HTTP embed config under when
# `http_embed_api_base` is set. Both are literals (mirrors every other row here); a rename of one
# without the other is a `RegistryError`/`EmbedderConfigError` at composition, not a silent gap.
_SUPPORTED_EMBEDDING = frozenset({"minilm_local", "minilm_vm_http"})


def _profile_rows(
    profile: ModelProfileSettings,
) -> tuple[ProviderRecord, ModelDeployment]:
    """The ONE ``ProviderKind.LOCAL_HTTP`` row a configured SLM profile contributes.

    ENG-118: the key does NOT travel in ``ModelDeployment.extra_params`` any more (that field is
    reserved for ``api_version``, model-layer-spec §4). The provider carries a
    ``credential_ref`` — a NAME — and the composition root's :class:`SecretSeamResolver` holds the
    value, so ``registry.compile_model_list`` resolves it through the one sanctioned seam
    (``registry.py:123-124``). A profile with no key (the default) declares no ``credential_ref``
    at all, which is what keeps the local endpoint credential-free on the wire.
    """
    provider = ProviderRecord(
        key=profile.provider_key,
        kind=ProviderKind.LOCAL_HTTP,
        litellm_provider=profile.provider,
        api_base=profile.base_url,
        credential_ref=profile.credential_ref if profile.api_key is not None else None,
        is_local=True,
    )
    deployment = ModelDeployment(
        model_group=profile.model_group,
        provider_key=profile.provider_key,
        model_id=f"{profile.provider}/{profile.model}",
        kind=ModelKind.LLM,
    )
    return provider, deployment


def _profile_models(profile: ModelProfileSettings, models: ModelSettings) -> ModelSettings:
    """Pin every LLM task at the profile's ONE model-group (unchanged behaviour).

    A configured profile is the operator saying *"use exactly this deployment"*, so it outranks
    the shipped catalog's per-task groups. ``embed_backend``/``embed_model`` are untouched — the
    ``EmbeddingPort`` seam is a different mechanism (§6-P5) — and ``max_output_tokens``/
    ``temperature`` still come from the WIRED ``EngineSettings.model`` (C1), not from the
    profile's own per-call extraction params.
    """
    return models.model_copy(
        update={
            "provider": profile.provider_key,
            "answer_model": profile.model_group,
            "adjudicate_model": profile.model_group,
            "hard_extract_model": profile.model_group,
            "routine_extract_model": profile.model_group,
            "summarize_model": profile.model_group,
            "classify_model": profile.model_group,
            "rerank_model": profile.model_group,
        }
    )


def _build_llm_catalog(profile: ModelProfileSettings) -> tuple[ModelSettings, ModelCatalogSettings]:
    """The catalog a CONFIGURED SLM profile resolves to — the profile's own single deployment.

    Kept as the profile path's shape (``CatalogSource.EMPTY`` + the profile row overlaid), because
    a profile means *"pin every task to this deployment"*: layering the shipped multi-provider
    table underneath it would add groups no task can reach. The plane's DEFAULT (no profile) is
    the opposite and goes through :func:`~mu_engine.providers.plane.resolve_plane_model_layer`
    with ``CatalogSource.SHIPPED``.

    CONFIG-AND-DATA-FIX-PLAN.md §1.2 C1/C2 are preserved: ``ModelSettings`` is still derived from
    the WIRED ``get_engine_settings().model`` and the catalog base is still the WIRED
    ``get_engine_settings().model_catalog``, so ``MU_MODEL__*`` / ``MU_MODEL_CATALOG__*`` reach it.
    """
    engine_settings = get_engine_settings()
    layer = _resolve_profile_layer(profile, engine_settings.model, engine_settings.model_catalog)
    return layer.models, layer.catalog


def _resolve_profile_layer(
    profile: ModelProfileSettings,
    models: ModelSettings,
    catalog: ModelCatalogSettings,
) -> PlaneModelLayer:
    """Shared by :func:`_build_llm_catalog` and ``LocalContainer._build_plane_router``."""
    provider, deployment = _profile_rows(profile)
    return resolve_plane_model_layer(
        models=_profile_models(profile, models),
        catalog=catalog.model_copy(update={"source": CatalogSource.EMPTY}),
        resolver=_profile_resolver(profile, catalog),
        overlay_providers=[provider],
        overlay_deployments=[deployment],
    )


def _profile_resolver(
    profile: ModelProfileSettings, catalog: ModelCatalogSettings
) -> SecretSeamResolver:
    """The secret seam for a profile build: the plane's own seam plus, when the profile carries a
    key VALUE (an endpoint that checks one), that value registered under its NAME."""
    overrides = {profile.credential_ref: profile.api_key} if profile.api_key is not None else None
    return build_plane_secret_resolver(catalog, overrides=overrides)


class LocalContainer:
    """The ONE place adapters bind for the embedded LOCAL engine (spec §2.2)."""

    def __init__(
        self,
        storage: StorageSettings,
        *,
        settings: Settings | None = None,
        engine_settings: EngineSettings | None = None,
        observability: ObservabilitySettings | None = None,
        lifecycle: LifecycleSettings | None = None,
        stage_ledger: StageLedger | None = None,
        conflict_records: ConflictRecordRepository | None = None,
        tier_counts: TierCountSettings | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        # CONFIG-AND-DATA-FIX-PLAN.md §1.2 C1: the ONE wired ``EngineSettings`` (C0) read here —
        # every intelligence-knob subtree below (`recall`, `distill`, `ingest`, `lifecycle`,
        # `model`) is pulled FROM this instance instead of being constructed bare, so
        # `MU_RECALL__…`/`MU_LIFECYCLE__…`/`MU_INGEST__…`/`MU_DISTILL__…`/`MU_MODEL__…` env
        # overrides actually reach the composed pipelines. Injectable (mirrors `settings` above)
        # so a caller/test can pass an already-constructed instance without touching os.environ.
        self._engine_settings: EngineSettings = engine_settings or get_engine_settings()
        self._closers: list[Callable[[], Awaitable[None]]] = []
        self._obs: ObservabilitySettings = observability or ObservabilitySettings()
        # Injection seam (design §13 item 6a): a durable caller (e.g. mu-engine-server, Stage C)
        # threads its own RedisStageLedger / durable ConflictRecordRepository through here; every
        # daemonless/embedded caller (mu-local's own callers, all tests) omits both and gets the
        # SAME in-process defaults constructed below (:2 and :7c) — byte-identical embedded
        # behavior, unchanged (AG-1).
        self._injected_stage_ledger: StageLedger | None = stage_ledger
        self._injected_conflict_records: ConflictRecordRepository | None = conflict_records

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

        # (1) the REAL embedder — in-process MiniLM by default, or the HTTP/VM backend when
        #     `storage.embedding.backend` selects it; dimension is read FROM the resolved adapter,
        #     never from config (storage-indexing-design §239-240) — the vector store binds to it.
        self.embedder: EmbeddingPort = self._build_embedder(storage.embedding)
        # Best-effort close of whatever network client the resolved embedder owns (e.g.
        # `HttpEmbedder`'s httpx client) — a no-op for the in-process singleton, which has none
        # (mirrors the `_register_closer` docstring's own "embedded backends ... yield nothing to
        # close" contract, generalized here from storage adapters to the embedder).
        self._register_closer(self.embedder)

        # (2) STM (kv role: redis by default) — built through the STORE_REGISTRY seam, exactly
        #     like every other role (spec §4.2). The durable KV write is the facade's durability
        #     floor; the pipeline ledger defaults to IN-PROCESS (InMemoryStageLedger) — a
        #     cross-PROCESS durable ledger (RedisStageLedger, mu-engine/pipelines/ledger.py:117)
        #     is a caller-supplied concern via the ``stage_ledger`` ctor param (design §13 item
        #     6a): mu-engine-server (Stage C) injects its own; every daemonless/embedded caller
        #     here gets this in-process default, unchanged (AG-1).
        #
        #     TRUTH (repaired 2026-07-31, design §13 item 6d — REVIEW-3 C2; was falsely "never
        #     collides across tenants/runs"): the promote idempotency key the ledger stores is
        #     keyed off ``content_hash`` alone (content-only, no η/namespace component,
        #     ``mu_engine.storage.domain.memory.compute_content_hash``). That IS collision-safe
        #     ACROSS separate ``LocalContainer``/process instances (each has its own ledger
        #     keyspace — an in-process ``InMemoryStageLedger`` or a distinct Redis prefix). It was
        #     NOT safe cross-USER within ONE multi-tenant instance: two different namespaces
        #     writing identical content would collide on the same bare content_hash key and the
        #     second promote would be silently skipped as a ledger hit. This is now fixed at the
        #     source by A3 (design §13 item 6d): ``DeterministicPromoteStage.idempotency_key``
        #     (``mu-engine/pipelines/concrete/ingest.py``) namespace-scopes the key so distinct
        #     namespaces never share a ledger entry, regardless of which ``StageLedger``
        #     implementation is injected here.
        self.stm: StmTierRepository = STORE_REGISTRY.build(
            "kv", storage.kv.backend, **self._kv_cfg(storage.kv)
        )
        self._register_closer(self.stm, "_redis", "_mc")
        self._ledger: StageLedger = self._injected_stage_ledger or InMemoryStageLedger()

        # (3) MTM (vector role: qdrant by default) — dim from the LIVE embedder, built through
        #     the same STORE_REGISTRY seam.
        #
        #     HYBRID MTM (mtm-retrieval-design.md §1.2/§1.3): ONE sparse producer, built here and
        #     threaded into BOTH sides that need it — this adapter (write side: the document term
        #     weights stamped onto every point) and `RecallService` below (read side: the query
        #     term weights). One instance, one config, so the two can never be enabled apart.
        #     `RecallSettings.sparse_enabled=False` (the default) makes this `None` and the whole
        #     engine is byte-identical to its dense-only self.
        self.sparse_encoder = build_sparse_encoder(self._engine_settings.recall)
        self.mtm: MtmTierRepository = STORE_REGISTRY.build(
            "vector",
            storage.vector.backend,
            dim=self.embedder.dimension,
            sparse_encoder=self.sparse_encoder,
            **self._vector_cfg(storage.vector),
        )
        self._register_closer(self.mtm, "_qdrant")

        # (4) LTM (graph role: falkordb, the MANDATORY graph engine) — same seam.
        self.ltm: GraphStorePort = STORE_REGISTRY.build(
            "graph", storage.graph.backend, **self._graph_cfg(storage.graph)
        )
        self._register_closer(self.ltm, "_db.connection")
        # D-5 (ARCHITECTURE-CONFORMANCE.md entity_uids MTM payload): wire the just-built vector
        # adapter into the just-built graph adapter's optional `EntityUidsSink` seam so
        # `upsert_fact` can backfill `entity_uids` onto an already-promoted MTM point once it
        # resolves subject/object entities. Duck-typed (`set_mtm_entity_sink`/`set_entity_uids`
        # are BOTH structural, no shared base class) so a non-Qdrant `storage.vector.backend`
        # (pgvector/chroma/faiss — none implement `set_entity_uids` today) or a non-FalkorDB
        # future graph backend simply leaves this a no-op, never a hard failure.
        set_sink = getattr(self.ltm, "set_mtm_entity_sink", None)
        if callable(set_sink) and hasattr(self.mtm, "set_entity_uids"):
            set_sink(self.mtm)

        # (5) control-plane (relational) — built through the STORE_REGISTRY seam (reuse). Off the
        #     ingest/recall critical path this slice; disposed on close.
        self.control = STORE_REGISTRY.build(
            "relational", storage.relational.backend, **self._relational_cfg(storage.relational)
        )
        self._register_closer(self.control, "_engine")

        # (5b) ContextRepository (NEW — software-arch spec §5/§6, l.260-263/l.340): the artifact
        #      provenance-root store `PersistRawArtifactStage` writes through. Built through the
        #      SAME STORE_REGISTRY seam as every other role (factories.py's new `artifact` role,
        #      not one of `MANDATORY_ROLES` — a filesystem adapter, no network client to close).
        self.artifacts: ContextRepository = STORE_REGISTRY.build(
            "artifact", storage.artifact.backend, **self._artifact_cfg(storage.artifact)
        )

        # (6) The MODEL LAYER. ENG-115a: `model_router` is now ALWAYS built — from the SHIPPED
        #     multi-provider catalog (anthropic/openai/azure/deepseek/moonshot + the keyless local
        #     endpoints), narrowed to the providers whose credentials actually resolve. Before
        #     this, `ModelCatalogSettings()` was empty and this root only ever called
        #     `default_local_catalog()`, so the plane shipped ONE embedder and ZERO LLM
        #     deployments — measured: `build_model_router` on the wired defaults raised
        #     `RegistryError: task 'answer' maps to model-group 'gpt-5-chat' which has no
        #     deployment`. That contradicted `CLAUDE.md:18-27` (FULL-LOCAL is complete and good)
        #     and is gate G6.
        #
        #     `self.llm` is DELIBERATELY still governed by `storage.llm`: the router existing and
        #     `LocalMemory`'s LLM-dependent verbs being armed are two different decisions, and
        #     flipping the second one silently would turn every heuristic-mode caller into a
        #     caller of a local endpoint that may not be running. `llm=None` ⇒ heuristic mode,
        #     byte-for-byte as before; a configured profile ⇒ the router, pinned to that profile.
        # `self.embedder` (built at (1) above) is INJECTED — the model layer must NOT resolve a
        # second `EmbeddingPort` of its own from `ModelSettings.embed_backend`. See
        # `_build_plane_router` and `build_model_router`'s docstring for the defect this closes.
        self.model_router: ModelRouter = self._build_plane_router(storage.llm, self.embedder)
        self.llm: ModelRouter | None = self.model_router if storage.llm is not None else None
        # …and its TEARDOWN, which did not exist. `close()` below promises to "release every store
        # connection this container opened"; the MODEL layer was outside that promise entirely, so
        # litellm's process-global logging worker — a `while True: await queue.get()` consumer our
        # first async model call starts — was left pending on a loop nobody would close it on, and
        # died as `RuntimeError: Event loop is closed` out of `asyncio.Queue.get`'s bare `except:`
        # (nine such unraisables measured on the VM's full mu-core suite; see
        # `LiteLLMRouterAdapter.aclose` for the mechanism). `insert(0, …)` so the LIFO drain runs it
        # LAST, and that position was CORRECTED by measurement: appended (drained FIRST) it still
        # leaked, because litellm ends every async call with a bare
        # `asyncio.create_task(_client_async_logging_helper(...))` that RE-STARTS the worker during
        # any await that follows — and the remaining store closers are exactly such awaits.
        # `model_router` is always built (ENG-115a) and `aclose()` is a no-op on a router that
        # never made a call, so this is unconditional.
        self._closers.insert(0, self.model_router.aclose)
        # C2: `settings=` threaded from the WIRED `EngineSettings.extraction` — previously bare
        # (`HeuristicSpoExtractor()`), so `MU_EXTRACTION__MIN_TOKENS`/`MU_EXTRACTION__…`-vocab
        # overrides never reached the DEFAULT (heuristic, `storage.llm is None`) extraction path,
        # only the LLM path below (which already threaded it via `model_copy`).
        self._extractor: FactExtractorPort = HeuristicSpoExtractor(
            settings=self._engine_settings.extraction
        )
        if storage.llm is not None:
            # C1: base on the WIRED `EngineSettings.extraction` (so `MU_EXTRACTION__MIN_TOKENS`/
            # `MU_EXTRACTION__CHUNK_TOKEN_RATIO` reach this extractor too) — `max_tokens`/
            # `temperature` still come from the SLM profile (the per-call params for THIS specific
            # deployed model), same as before.
            self._extractor = LlmFactExtractor(
                self.model_router,
                model_group=storage.llm.model_group,
                settings=self._engine_settings.extraction.model_copy(
                    update={
                        "max_tokens": storage.llm.max_tokens,
                        "temperature": storage.llm.temperature,
                    }
                ),
            )

        # (7) platform singletons + the three content-free observability sinks (DEV-STANDARDS
        #     rule 4) — built once here from ObservabilitySettings and threaded into every service.
        self._clock = SystemClock()
        self._bus = InprocBus()
        # (7c) AD-24 — the tier-count cache behind ``MemoryLifecycleManager.get_state``'s
        #      stm/mtm/ltm counts, which were a hardcoded, FALSE ``0,0,0``. Built HERE, ONCE, and
        #      attached to the SAME ``InprocBus`` above that ingest/distill/promotion/demotion/
        #      retention already publish onto — not inside ``build_lifecycle_manager``, which is a
        #      FACTORY that returns a fresh manager per call while ``InprocBus._handlers`` is a
        #      plain list that is never pruned (subscribing per manager would leak a handler per
        #      call). Its ``detach`` goes into ``self._closers`` so ``close()`` unsubscribes it.
        #      Zero API keys, zero daemon: the events it folds are published in-process by the
        #      services this same container builds (the FULL-LOCAL boundary rule).
        #
        #      Settings are INJECTABLE (``tier_counts=`` above) and, failing that, come from
        #      ``TierCountSettings()`` — a ``BaseSettings`` reading ``MU_TIER_COUNTS_*``, so both
        #      memory bounds are reachable by an operator (DEV-STANDARDS rule 3; the first cut was
        #      a bare ``BaseModel`` with no seam and an unreachable ~4 GB ceiling).
        #
        #      ``stm_window_s`` must track the STM adapter's real TTL, because STM expiry publishes
        #      no event and that window is the only thing keeping ``stm_count`` from converging to
        #      "captures since process start". Two of the KV backends expose it in settings
        #      (``InMemoryKvSettings``/``MemcachedSettings``: ``default_ttl_s``) and it is read from
        #      there; ``RedisSettings``/``ValkeySettings`` do NOT — ``RedisStmAdapter.__init__``
        #      builds a bare ``RedisMapper()`` whose ``default_ttl_s=3600`` no config can reach
        #      (``storage/adapters/redis_stm.py``/``storage/mappers/redis_mapper.py``). That is a
        #      real config gap in ``storage/**``, owned by another lane and REPORTED, not papered
        #      over here: on those backends this falls back to ``TierCountSettings``' own 3600s
        #      default, which matches the mapper today, and an operator who changes one must set
        #      ``MU_TIER_COUNTS_STM_WINDOW_S`` to match.
        _kv_ttl_s = getattr(self._settings.storage.cache, "default_ttl_s", None)
        self._tier_count_settings: TierCountSettings = (
            tier_counts
            if tier_counts is not None
            else (
                TierCountSettings(stm_window_s=float(_kv_ttl_s))
                if _kv_ttl_s is not None
                else TierCountSettings()
            )
        )
        self.tier_counts = TierCountCache(settings=self._tier_count_settings, clock=self._clock)
        self.tier_counts.attach(self._bus)
        self._closers.append(self.tier_counts.detach)

        self.tracer: Tracer = build_tracer(enabled=self._obs.otel_enabled, service_name="mu-local")
        self.metrics: MetricSink = build_metrics(enabled=self._obs.metrics_enabled)
        # C3: `settings=` threaded from the WIRED `EngineSettings.observability` so
        # `MU_OBSERVABILITY__DURABLE_AUDIT_QUEUE_MAX` reaches `_DurableAuditLog`'s bounded queue
        # whenever a durable sink IS configured (`build_audit`'s own docstring: this arg is a
        # no-op when `durable_sink=None`, unchanged here — no durable sink is wired yet).
        self.audit: AuditLog = build_audit(
            enabled=self._obs.audit_enabled, settings=self._engine_settings.observability
        )

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
        #
        #      C1 (CONFIG-AND-DATA-FIX-PLAN.md §1.2): every OTHER field (`salience`, `retention`,
        #      promote/demote thresholds, cadence, adjudication budget, …) now comes from the
        #      WIRED `EngineSettings.lifecycle` (env-overridable, e.g.
        #      `MU_LIFECYCLE__SALIENCE__W_RECENCY`, `MU_LIFECYCLE__PROMOTE_STM_MTM`) instead of a
        #      bare `LifecycleSettings()`; only `manager_mode` keeps this composition-root's
        #      deliberate MANUAL narrowing (a structural fact about this daemonless container, not
        #      an operator-tunable knob — unchanged from before this fix).
        self.lifecycle_settings: LifecycleSettings = (
            lifecycle
            or self._engine_settings.lifecycle.model_copy(
                update={"manager_mode": ManagerModeSettings(default_mode=ManagerMode.MANUAL.value)}
            )
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
        #      mock) for a PENDING/MANUAL-parked verdict — used unless a durable
        #      ``ConflictRecordRepository`` was injected via the ``conflict_records`` ctor param
        #      (design §13 item 6a/6b; no durable implementation exists in-tree yet, item 6b).
        #      Every daemonless/embedded caller here omits it and gets this in-process default,
        #      unchanged (AG-1).
        #
        #      C1: from the WIRED `EngineSettings.distill` (`MU_DISTILL__SUPERSEDE_CONFIDENCE`,
        #      `MU_DISTILL__USE_LLM_ADJUDICATOR`, …) instead of a bare `DistillSettings()`.
        self._distill_settings = self._engine_settings.distill
        # ONE conflict inbox per container, hoisted out of the adjudicator branch below.
        # ``MemoryHealthService`` READS this inbox (through ``PendingConflictEdgeReader``) to
        # raise the CONFLICTING / LOW_CONFIDENCE flags, and ``ConflictAdjudicator`` WRITES its
        # parked verdicts into it. Constructing it inline inside the adjudicator branch would
        # have given the health view a SECOND, permanently empty inbox in LLM mode and no
        # shared object at all in heuristic mode — a health lens that structurally could never
        # see a conflict (DEV-STANDARDS rule 9: one composition root, one instance per plane).
        self._conflict_records: ConflictRecordRepository = (
            self._injected_conflict_records or InMemoryConflictRecordRepository()
        )
        self.conflict_adjudicator: ConflictAdjudicator | None = None
        if self.llm is not None and self._distill_settings.use_llm_adjudicator:
            # C3: `settings=` threaded from the WIRED `EngineSettings.lifecycle`
            # (`MU_LIFECYCLE__ADJUDICATION_BUDGET_PER_SWEEP`, `MU_LIFECYCLE__
            # ADJUDICATION_DEGRADE_THRESHOLD_S`, `MU_LIFECYCLE__ADJUDICATOR_MAX_TOKENS`,
            # `MU_LIFECYCLE__ADJUDICATOR_TEMPERATURE`) instead of `build_conflict_adjudicator`
            # omitting `settings=` entirely (-> a bare `ConflictAdjudicatorSettings()` fallback).
            self.conflict_adjudicator = build_conflict_adjudicator(
                use_llm=True,
                router=self.llm,
                settings=conflict_adjudicator_settings_from_lifecycle(self.lifecycle_settings),
                clock=self._clock,
                bus=self._bus,
                conflict_records=self._conflict_records,
            )

        # (8) application services — each facade verb delegates to exactly one of these; each gets
        #     the wired sinks so its meaningful op emits spans/metrics/audit (never silently no-op).
        # C1: `settings=` threaded from the WIRED `EngineSettings.ingest` — previously omitted
        # entirely, so `IngestService`/`DeterministicPromoteStage` fell back to a bare
        # `IngestSettings()` (`services/ingest.py:96`) and `importance_promote`/`mention_promote`/
        # `stm_ttl_s` were unreachable from the environment (plan §1.1 Group A).
        self.ingest = IngestService(
            stm=self.stm,
            mtm=self.mtm,
            embedder=self.embedder,
            bus=self._bus,
            ledger=self._ledger,
            clock=self._clock,
            settings=self._engine_settings.ingest,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
            # NEW (software-arch spec §6, l.340-341): threading a real ContextRepository turns ON
            # `PersistRawArtifactStage` — every capture becomes kind=REFERENCE targeting a
            # persisted ContextArtifact (see `services/ingest.py`'s `artifacts=None` docstring for
            # why this stays optional at the service layer; this composition root always opts in).
            artifacts=self.artifacts,
        )
        self.distill = DistillPipeline(
            ltm=self.ltm,
            extractor=self._extractor,
            clock=self._clock,
            settings=self._distill_settings,
            mtm=self.mtm,
            # Third arm of the cross-store supersession (memory-layer-design.md §7.2 step 5):
            # without this the superseded loser stays live in the STM recency window and the
            # recency floor re-surfaces it as a top recall hit.
            stm=self.stm,
            bus=self._bus,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
            adjudicator=self.conflict_adjudicator,
        )
        # C1: from the WIRED `EngineSettings.recall` (`MU_RECALL__WEIGHT_MTM`,
        # `MU_RECALL__RRF_K`, `MU_RECALL__FLOOR_PROTECT_LIMIT`, …) — the exact class the
        # `02fbed9` `recency_floor_limit` bug lived in, previously unreachable from bare
        # `RecallSettings()`.
        recall_settings = self._engine_settings.recall
        fusion = ReciprocalRankFusion()
        ranker = ThreeChannelRecallRanker(
            stm=self.stm,
            mtm=self.mtm,
            ltm=self.ltm,
            fusion=fusion,
            settings=recall_settings,
            clock=self._clock,
            # D1 (data-quality assessment §3.1): the SAME embedder the query is embedded with at
            # the RecallService façade — the ranker reuses it to score STM candidate content
            # against the query vector (`recall_settings.stm_scoring`, default "embed").
            embedder=self.embedder,
            # ACCURACY-PLAN-0831.md item 6: `self.model_router` is ALWAYS built (ENG-115a, gate
            # G6 — see the (6) comment above), so this plane's rerank gate is armed exactly as
            # unconditionally as its context-budget width derivation already is
            # (`context_budget=self.model_router` below) — never gated on `self.llm`/`storage.llm`
            # for the same reason: `AdaptiveRerankGate.apply` never calls the model until a real
            # `recall()` needs it, and `settings.recall.rerank_enabled` (default True, env-
            # overridable `MU_RECALL__RERANK_ENABLED=false`) is the one A/B off-switch this needs.
            reranker=self.model_router,
        )
        authz = RecallAuthorizationFilter(
            tenancy=DefaultTenancyGuard(), authorized_ids=PrincipalAuthorizedIdsResolver()
        )
        self.recall = RecallService(
            embedder=self.embedder,
            private_ranker=ranker,
            # §1.3 "the M2 resolution": the façade encodes the sparse query at the SAME boundary
            # it embeds the dense one — the same instance the MTM adapter writes with, above.
            sparse_encoder=self.sparse_encoder,
            shared_recall=LocalNullSharedRecall(clock=self._clock),
            authz=authz,
            fusion=fusion,
            settings=recall_settings,
            clock=self._clock,
            metrics=self.metrics,
            tracer=self.tracer,
            # ACCURACY-PLAN-0831.md item 4 / CLAUDE.md boundary rule ("FULL-LOCAL must stay good
            # on the same mechanism"): `self.model_router` is ALWAYS built (ENG-115a, gate G6 —
            # see the (6) comment above), so THIS plane's recall width derivation is armed exactly
            # as unconditionally as chunking's identical `max_input_tokens` lookup already is —
            # never gated on `self.llm`/`storage.llm` (the SLM-profile-only, synthesis-capable
            # seam), because deriving a width is a pure catalog/registry metadata read, not a live
            # model call: it costs nothing to leave armed even when no deployment's credentials
            # actually resolved (`derive_recall_limit`'s own ceiling clamp bounds the worst case —
            # an unknown/unresolved group lands at the SAME validated `max_derived_limit` a
            # correctly-configured large-context model would, never an unbounded guess).
            context_budget=self.model_router,
        )

        # (9) the MemoryRepository FAÇADE + the two services that could not exist without it
        #     (CANONICAL §6-P2; memory-health-pinning-spec §5.1/§5.2).
        #
        #     ``MemoryHealthService`` and ``PinService`` both take ``repo: MemoryRepository`` as a
        #     REQUIRED keyword with no default, and nothing in this repo implemented that Protocol
        #     — so neither class could be constructed at all, and mu-client's ``/health`` /
        #     ``/pin`` / ``/unpin`` IPC routes returned a named 503 while its MCP tools raised
        #     ``ServiceNotWiredError``. Building the façade alone would not have changed that:
        #     this container built NEITHER service, so the surfaces had nothing to be handed.
        #     Both are constructed here, over the SAME stm/mtm/ltm adapters every other service
        #     shares (never a second set), which is what makes the feature live rather than merely
        #     present.
        self._tier_router = TierRouter(
            (
                TierLeg(Tier.STM, self.stm, backend=storage.kv.backend),
                TierLeg(Tier.MTM, self.mtm, backend=storage.vector.backend),
                TierLeg(Tier.LTM, self.ltm, backend=storage.graph.backend),
            )
        )
        self.memory: TieredMemoryRepository = TieredMemoryRepository(
            router=self._tier_router,
            # CANONICAL §6-P2 puts the embed step at the façade boundary; the SAME live embedder
            # every other arm uses, so `semantic` cannot drift onto a second model.
            embedder=self.embedder,
        )
        #     WHICH of the two services can exist is a BINDING question, answered at BOOT.
        #
        #     ``TierRouter.missing_capabilities()`` names every bound backend that has no walk
        #     primitive and every one that has no id-stable pin upsert. Three of the five vector
        #     backends (pgvector/chroma/faiss) genuinely have neither. On such a binding a
        #     constructed ``MemoryHealthService`` would raise ``TierCapabilityUnavailableError``
        #     on EVERY call, forever — and that class does NOT subclass
        #     ``TierRepositoryUnavailableError``, so ``MemoryHealthService._walk``'s degrade branch
        #     never fires and mu-client's ``/health`` route (which wraps ``assess`` in no ``try``)
        #     would close the connection with no reply. A service that can only ever throw is not
        #     a wired service; the house ABSENCE rule says leave it absent, and every surface
        #     already answers a NAMED, content-free "not wired" degrade for exactly that
        #     (``daemon/ipc.py`` ``health_service_not_wired``). So the gap decides, rather than
        #     being logged and then ignored.
        #
        #     Reported and NOT raised: such a backend still serves ingest and recall perfectly
        #     well, so refusing the whole container would take a working deployment down over two
        #     surfaces it never used.
        #
        #     REPORTED DESIGN DELTA, not silently absorbed: a partial view over the tiers that CAN
        #     enumerate would be strictly better than absence here, and the ``partial`` flag plus
        #     the ``tiers=`` narrowing already exist to express it — but CANONICAL §2 models
        #     exactly ONE memory-tier degrade reason (``LTM_UNAVAILABLE``), so "MTM cannot
        #     enumerate at all" has no sanctioned partial answer to fall back to. Inventing one
        #     here would be a silent deviation from the design set.
        gaps = self._tier_router.missing_capabilities()
        cannot_enumerate = tuple(gap for gap in gaps if ":enumerate(" in gap)
        cannot_pin = tuple(gap for gap in gaps if ":set_pinned(" in gap)
        if gaps:
            _log.warning(
                "memory_repository_capability_gap",
                missing=list(gaps),
                impact="health/pin surfaces refuse by name on this binding",
            )
        health_settings = HealthSettings()
        #: ``None`` on a binding whose vector backend cannot walk a partition — the health lens is
        #: a partition walk and nothing else, so without one there is no view to serve.
        self.health: MemoryHealthService | None = (
            None
            if cannot_enumerate
            else MemoryHealthService(
                repo=self.memory,
                assessor=HeuristicV1Assessor(health_settings),
                conflicts=PendingConflictEdgeReader(self._conflict_records),
                settings=health_settings,
                clock=self._clock,
                bus=self._bus,
                tracer=self.tracer,
                metrics=self.metrics,
                audit=self.audit,
            )
        )
        #: ``None`` when any bound tier cannot apply the id-stable pin upsert, AND when the
        #: partition cannot be walked — ``PinService._assert_within_pin_bound`` counts the
        #: partition's existing pins with one bounded ``enumerate`` page on EVERY pin, so a
        #: binding that cannot enumerate cannot enforce the pin-explosion bound either.
        self.pin: PinService | None = (
            None
            if (cannot_pin or cannot_enumerate)
            else PinService(
                repo=self.memory,
                bus=self._bus,
                settings=PinSettings(),
                clock=self._clock,
                tracer=self.tracer,
                metrics=self.metrics,
                audit=self.audit,
            )
        )

        # ---------------------------------------------------------------- (10) PERSONA (§5.2)
        #: The private user-model (``persona-design.md``). ``None`` when persona is disabled or
        #: when NO model is wired: spec line 103's slot tagger IS ``models.classify_model``, so a
        #: zero-model deployment has no evidence to build a persona from, and the house ABSENCE
        #: rule (the same one that leaves ``health``/``pin`` ``None`` above) says leave it absent
        #: rather than composed-and-inputless. ``build_persona`` also SUBSCRIBES the two bus
        #: seams — ``MemoryPromoted`` -> the sync incremental queue, ``ConsolidationCompleted`` /
        #: ``SleeptimeTick`` -> the sleep-time cadence — onto the SAME bus every other service
        #: publishes to (DEV-STANDARDS rule 9), which is what makes persona actually run.
        self.persona: PersonaWiring | None = build_persona(
            memory=self.memory,
            router=self.llm,
            bus=self._bus,
            clock=self._clock,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
        )
        if self.persona is not None:
            # §5.2's topic-affinity prior: a DECORATOR over the finished ranked read, so persona
            # provably runs after the authorized top-k and can only re-order survivors.
            #
            # The ``cast`` is a KNOWN, REPORTED lie to the type-checker, not a shortcut.
            # ``mu_engine.surface.facade.LocalContainerLike`` declares ``recall: RecallService``
            # as an invariant attribute, so a structurally-typed wrapper does not satisfy it;
            # that Protocol is in a file this lane does not own AND one the ``persona-off-the-
            # hot-path`` import contract forbids from naming persona at all. The wrapper
            # implements ``RecallService``'s ENTIRE public surface (``recall`` and
            # ``recall_degraded``) and ``test_persona_shaping_unit`` asserts that against the
            # real class, so the cast is safe today and fails visibly the day it is not. The fix
            # is one line in ``facade.py`` — reported, not made here.
            self.recall = cast("RecallService", self.persona.shaped(self.recall))

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
        for ``self.distill`` — never a second instance. ``mtm=self.mtm`` threads the SAME
        ``MtmTierRepository`` this container already built so the manager's automatic sweep can
        ENUMERATE stale MTM points (``scan_for_demotion``) and drive the demotion forgetting curve
        with no caller-supplied window (S2 tier-lifecycle completion). ``retention=`` threads a
        REAL ``RetentionService`` over the SAME ``self.ltm`` graph adapter — but ONLY when that
        adapter is a ``FalkorLtmAdapter`` (the only graph backend that implements the three
        ``LtmRetentionStorePort`` capabilities ``facts_by_state``/``chain_head_state``/
        ``gc_delete``); a non-Falkor graph backend (none ship today) leaves ``retention=None`` (an
        honest skip, never a fabricated no-op adapter). ``warm_cache`` is
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
            # D1 (STATE-AND-DEFECTS-0829.md): the SAME embedder instance the ingest-time gate
            # (`DeterministicPromoteStage`) and the recall ranker already use — never a second
            # one (DEV-STANDARDS rule 9).
            embedder=self.embedder,
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
        # Validity-first LTM retention (S2-01, ADR 0035): a REAL RetentionService over the SAME
        # graph adapter, wired ONLY when it is a FalkorLtmAdapter — the one graph backend that
        # implements the LtmRetentionStorePort capabilities (facts_by_state/chain_head_state/
        # gc_delete). A non-Falkor backend (none ship today) leaves this None — an honest skip
        # (manager.py:594's `if self._retention is not None` simply does not fire), never a
        # fabricated no-op adapter in the production path.
        from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter

        retention: RetentionService | None = None
        if isinstance(self.ltm, FalkorLtmAdapter):
            retention = RetentionService(
                ltm=self.ltm,
                ltm_retention=self.ltm,  # SAME real adapter — no test-only store in prod
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
            mtm=self.mtm,  # SAME MtmTierRepository — powers scan_for_demotion auto-drive
            retention=retention,  # REAL RetentionService (FalkorDB) or honest None
            conflict=self.conflict_adjudicator,  # SAME instance self.distill was built with
            mode_gate=self.mode_gate,
            bus=self._bus,
            settings=self.lifecycle_settings,
            lease=lease,
            runner=runner,
            clock=clock or self._clock,
            warm_cache=warm_cache,
            counts=self.tier_counts,  # AD-24: SAME cache, attached to the SAME bus (see §7c)
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

    def _build_embedder(self, choice: BackendChoice) -> EmbeddingPort:
        # C2: the catalog backing this embedder is now the WIRED `EngineSettings.model_catalog`
        # (`MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND`/`MU_MODEL_CATALOG__DEFAULT_MINILM_PATH`, and
        # now also `MU_MODEL_CATALOG__HTTP_EMBED_*` for the opt-in HTTP/VM backend — settings.py
        # `default_local_catalog`), not a bare call. `choice.backend` (from `StorageSettings.
        # embedding`, the storage-layer BACKEND-SELECTION knob, `MU_EMBED_BACKEND` via
        # `mu_client.config.ClientSettings.embed_backend`) selects WHICH registered key resolves
        # — "minilm_local" (default, in-process) or "minilm_vm_http" (the VM endpoint); which
        # concrete `EmbeddingPort` implementation that key resolves to is `build_embedder`'s
        # config-type dispatch (embedding.py), not decided here.
        catalog = default_local_catalog(self._engine_settings.model_catalog)
        embedder = build_embedder(choice.backend, catalog)
        if not isinstance(embedder, EmbeddingPort):  # fail-loud, never a silent None
            raise BackendUnavailableError(
                f"embedding backend {choice.backend!r} did not resolve to an EmbeddingPort"
            )
        return embedder

    def _build_plane_router(
        self, profile: ModelProfileSettings | None, embedder: EmbeddingPort
    ) -> ModelRouter:
        """Build the plane's REAL ``ModelRouter`` through the ONE model-layer entry point.

        Two shapes, one call site:
          * ``profile is None`` (the DEFAULT) — ``build_plane_router`` resolves the SHIPPED
            catalog, probes the secret seam (``MU_MODEL_CATALOG__SECRETS_DIR`` + the environment)
            and activates only the providers whose credentials exist. Every local provider carries
            no ``credential_ref``, so a box with zero keys still gets a complete table; the
            per-task fields the operator did not set adopt the logical ``ModelGroup``s
            (capability / cost / latency / context / local-vs-remote).
          * a configured profile — the same entry point with the profile's ONE row overlaid on an
            EMPTY source, pinning every task to it (``_resolve_profile_layer``).

        C1: threads `chunk_token_ratio` from the WIRED `EngineSettings.extraction` (was a bare
        `LongTextChunker()`/hardcoded `* 3 // 4` in `chunking.py`, plan §1.1 Group A).

        `embedder` is THIS container's own `EmbeddingPort` — the same instance already injected
        into ingest/recall/rank/promote — handed to the model layer so it does not resolve a
        SECOND one. Before this, the container selected its embedder from `StorageSettings.
        embedding.backend` (`MU_EMBED_BACKEND`) while `build_model_router` independently resolved
        `EngineSettings.model.embed_backend` (`MU_MODEL__EMBED_BACKEND`). The two disagreed by
        default, so pointing the laptop daemon at the VM embed endpoint still imported torch and
        loaded MiniLM in-process for a `ModelRouter.embed` with zero callers — measured
        2026-08-31 at 1139 MB RSS and two live `SentenceTransformer` instances, with
        `model_router_built` reporting `embed_backend=minilm_local` while every real embed went
        over HTTP. ONE container, ONE embedder, ONE closer (`_register_closer` at (1)).

        This is deliberately NOT mirrored in `mu_engine_server.composition`: on the server the
        engine and the model are on the SAME host, so an in-process embedder there is the correct
        deployment, not an accident. The defect is specific to a composition root whose embedding
        was moved OFF-box.
        """
        ratio = self._engine_settings.extraction.chunk_token_ratio
        catalog = self._engine_settings.model_catalog
        if profile is None:
            return build_plane_router(
                models=self._engine_settings.model,
                catalog=catalog,
                chunk_token_ratio=ratio,
                resolver=build_plane_secret_resolver(catalog),
                embedder=embedder,
            )
        layer = _resolve_profile_layer(profile, self._engine_settings.model, catalog)
        return build_model_router(
            models=layer.models,
            catalog=layer.catalog,
            secret_resolver=_profile_resolver(profile, catalog),
            chunk_token_ratio=ratio,
            embedder=embedder,
        )

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

    def _artifact_cfg(self, choice: BackendChoice) -> dict[str, Any]:
        """``STORE_REGISTRY.build("artifact", ...)`` kwargs — the ``filesystem`` factory
        (``mu_engine.storage.factories._build_artifact_fs``) already falls back to the central
        ``ArtifactFsSettings.content_root`` when ``cfg`` doesn't supply one, same pattern as
        ``valkey``/``chroma`` above, so an empty override dict is the common case."""
        return dict(choice.config)

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
