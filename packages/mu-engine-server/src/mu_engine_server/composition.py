"""``EngineServerApp`` — the C4 composition root (build-plan §4 C4).

**The boundary problem this module exists to solve.** ``SurfaceFacade`` (``mu_engine.surface.
facade``) wraps a ``LocalContainerLike`` structural Protocol — deliberately NOT ``mu_local.
composition.LocalContainer`` itself, because ``mu_engine_server`` may import ``{mu_contracts,
mu_engine}`` ONLY (design §8, ``.importlinter`` ``mu-engine-server-boundary``: forbidden
``mu_server``/``mu_client``/``mu_local``). ``LocalContainerLike``'s own docstring names exactly
which attributes satisfy it (``ingest``/``distill``/``stm``/``recall``/``mode_gate``/``llm``) and
says a real ``LocalContainer`` "satisfies this Protocol with ZERO adapter code" — but building
one AT ALL still means going through ``mu_local.composition.LocalContainer.__init__``, which this
package cannot import.

The resolution (per this task's own instruction): **read how ``mu_local/composition.py:230-378``
assembles ``LocalContainer`` and replicate the assembly here, directly against ``mu_engine`` +
``mu_contracts`` primitives** — never against ``mu_local.config.StorageSettings``/``BackendChoice``
(themselves plain pydantic models with zero ``mu_local``-only logic, but physically homed in a
package this module cannot import) and never against ``mu_local.shared_null.LocalNullSharedRecall``
(a 20-line empty-arm adapter with zero ``mu_local``-only logic either — reimplemented below as
:class:`_NullSharedRecall`, byte-identical behaviour, PORTed not imported, exactly as
``SurfaceFacade`` itself already PORTs ``LocalMemory._ns``/``_render_context`` across this same
boundary rather than importing them).

**Everything else the real assembly needs — ``STORE_REGISTRY``, ``IngestService``,
``DistillPipeline``, ``RecallService`` + its ranker/fusion/authz collaborators, ``ManagerModeGate``,
``ModelRouter``, the three observability sinks, ``InprocBus``/``SystemClock`` — is ALREADY
``mu_engine``-native** (``mu_local/composition.py``'s own imports prove this: only
``mu_local.config``/``mu_local.errors``/``mu_local.shared_null`` are ``mu_local``-homed; every
other name it imports is ``mu_contracts.*`` or ``mu_engine.*``). So no piece of the REAL assembly
is actually missing from this package's reach — the boundary is resolvable, not a genuine gap.
Nothing here is FLAGGED as a boundary problem because nothing was found to be one.

**The one deliberate narrowing vs. ``LocalContainer``**: this composition root always binds
EXACTLY Valkey (kv) + Qdrant (vector) + FalkorDB (graph) + sqlite (relational, off the
ingest/recall critical path, same as ``LocalContainer``'s own default) — never a config-selected
alternative backend (contrast ``mu_local.config.StorageSettings``'s full multi-backend generality,
``BackendChoice``). ``mu-engine-server`` is a single deployable shape (design §2.2, "single-tenant
by construction"); ``EngineServerSettings`` (:mod:`mu_engine_server.settings`) exposes each of
those four stores' HOST/PORT as the tunable surface, not the backend CHOICE itself.

**The durability point (item 6c, this task's actual assignment)**: unlike ``LocalContainer``'s own
in-process defaults (``InMemoryStageLedger`` / ``InMemoryConflictRecordRepository`` — spec'd as the
"embedded, daemonless" floor, ``mu_local/composition.py:234-252``), this composition root ALWAYS
injects the durable Redis-shaped adapters (``RedisStageLedger``, ``mu_engine/pipelines/ledger.py:
117`` — EXISTS; ``RedisConflictRecordRepository``, C1, ``mu_engine/lifecycle/conflict_redis.py`` —
EXISTS) against the SAME configured Valkey endpoint the STM tier repository itself binds. A
promote/distill crash mid-flight replays against durable state, not an empty in-process dict that
died with the process — the entire reason ``LocalContainer.__init__`` accepts ``stage_ledger``/
``conflict_records`` as an injection seam in the first place (its own docstring: "a durable caller
(e.g. mu-engine-server, Stage C) threads its own ... through here").
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import FastAPI
from redis.asyncio import Redis

from mu_contracts.domain.errors import BackendUnavailableError
from mu_contracts.domain.model.memory import Namespace as _Namespace
from mu_contracts.domain.model.memory import Tier as _Tier  # PERSONA (§5.2) tier legs
from mu_contracts.domain.model.memory import Visibility as _NsVisibility
from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.time import Clock
from mu_engine.config import EngineSettings, get_engine_settings
from mu_engine.lifecycle.conflict import (
    ConflictAdjudicator,
    build_conflict_adjudicator,
    conflict_adjudicator_settings_from_lifecycle,
)
from mu_engine.lifecycle.conflict_redis import ConflictRedisSettings, RedisConflictRecordRepository
from mu_engine.lifecycle.demotion import DemotionService
from mu_engine.lifecycle.manager import MemoryLifecycleManager
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate
from mu_engine.lifecycle.promotion import PromotionService
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import ManagerModeSettings
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.pipelines.ledger import RedisStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import build_audit, build_metrics, build_tracer
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.providers.catalog import ModelDeployment, ModelKind, ProviderKind, ProviderRecord
from mu_engine.providers.embedding import SentenceTransformerEmbedder, build_embedder
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
from mu_engine.services.conflict.policy_resolver import ConflictPolicyResolver
from mu_engine.services.conflict.ports import (
    InMemoryMemoryConflictPolicyStore,
    InMemoryNamespaceConflictPolicyStore,
    RecordBackedResolutionQueue,
)
from mu_engine.services.conflict.resolution import ConflictResolutionService
from mu_engine.services.extract import (
    FactExtractorPort,
    HeuristicSpoExtractor,
    LlmFactExtractor,
)
from mu_engine.services.ingest import IngestService
from mu_engine.services.memory.repository import TieredMemoryRepository
from mu_engine.services.memory.router import TierLeg, TierRouter
from mu_engine.services.persona import PersonaWiring, build_persona
from mu_engine.services.recall import (
    PrincipalAuthorizedIdsResolver,
    RecallAuthorizationFilter,
    RecallService,
    ReciprocalRankFusion,
    ThreeChannelRecallRanker,
)
from mu_engine.services.recall.dto import RecallChannels, RecallQuery, RecallResult
from mu_engine.storage.domain.namespace import Visibility
from mu_engine.storage.factories import STORE_REGISTRY
from mu_engine.storage.ports import (
    ContextRepository,
    GraphStorePort,
    MtmTierRepository,
    StmTierRepository,
)
from mu_engine.storage.registry import assert_mandatory_roles
from mu_engine.surface.facade import SurfaceFacade
from mu_engine_server.app import build_app
from mu_engine_server.auth import make_bearer_verifier, require_bearer_token
from mu_engine_server.lifecycle_runner import EngineLifecycleSweepRunner
from mu_engine_server.settings import EngineServerSettings, SlmProfile, load_settings

__all__ = ["EngineContainer", "EngineServerApp"]

# Backends this composition root always selects, per role (the deliberate narrowing described
# in the module docstring — never a config-selected alternative, unlike LocalContainer/
# StorageSettings).
_KV_BACKEND = "valkey"
_VECTOR_BACKEND = "qdrant"
_GRAPH_BACKEND = "falkordb"
_RELATIONAL_BACKEND = "sqlite"  # off the ingest/recall critical path, same as LocalContainer
_EMBEDDING_BACKEND = "minilm_local"
# ContextRepository role (NEW — software-arch spec §5): not one of the four FIXED roles above
# (never MANDATORY, module docstring) — a filesystem adapter, no dedicated EngineServerSettings
# endpoint of its own; the factory falls back to the central ArtifactFsSettings.content_root.
_ARTIFACT_BACKEND = "filesystem"


class _WorkspaceDefaultModeResolver:
    """PORT of ``mu_local.composition._WorkspaceDefaultModeResolver`` (that class has zero
    ``mu_local``-only logic — a one-field constant resolver — but is private to a package this
    module cannot import). Resolves the SAME single configured default for every namespace this
    single-tenant server serves (design §2.2)."""

    def __init__(self, settings: Any) -> None:
        self._mode = ManagerMode(settings.default_mode)

    def resolve(self, ns: Any) -> ManagerMode:
        return self._mode


class _NullSharedRecall:
    """PORT of ``mu_local.shared_null.LocalNullSharedRecall`` (module docstring — zero
    ``mu_local``-only logic, private to a package this module cannot import). ``mu-engine-server``
    is LOCAL-mode, single-tenant (design §2.2): there is no shared plane to query, so this arm
    always returns an empty, non-degraded result and the recall fusion collapses to the private
    arm alone — byte-identical behaviour to what ``LocalContainer`` wires today."""

    def __init__(self, *, clock: Clock) -> None:
        self._clock = clock

    async def recall(
        self, q: RecallQuery, *, caller_identity_set: CallerIdentitySet
    ) -> RecallResult:
        del caller_identity_set  # no shared plane ⇒ the caller identity set authorizes nothing
        return RecallResult(
            namespace=q.namespace,
            items=[],
            channels_run=RecallChannels(stm=False, mtm=False, ltm=False),
            degraded=None,
            generated_at=self._clock.now(),
        )


#: The secret-seam NAME the configured SLM profile's key is registered under (ENG-118: a NAME in
#: the catalog, the VALUE only inside the resolver). Never a value, never in `extra_params`.
_PROFILE_CREDENTIAL_REF = "mu_engine_server_llm_api_key"


def _profile_rows(profile: SlmProfile) -> tuple[ProviderRecord, ModelDeployment]:
    """The ONE ``ProviderKind.LOCAL_HTTP`` row the configured SLM profile contributes.

    ENG-118 (mirrors the identical fix in ``mu_local.composition``): the key no longer travels in
    ``ModelDeployment.extra_params`` — ``model-layer-spec §4`` reserves that field for
    ``api_version`` and nothing else. The provider carries a ``credential_ref`` NAME and the
    root's :class:`SecretSeamResolver` holds the value, so ``registry.compile_model_list``
    resolves it through the one sanctioned seam (``registry.py:123-124``).

    REPORTED, NOT FIXED HERE (outside this lane's file ownership): ``SlmProfile.provider`` still
    defaults to ``"openai"`` (``settings.py:103``) and ``api_key`` to a literal placeholder
    (``:105``). With the ``openai`` prefix litellm falls back to ``get_secret("OPENAI_API_KEY")``
    when no key is supplied, i.e. a deployment with a cloud key in its environment would send it
    to ``127.0.0.1`` — measured, see ``mu_local/config.py``'s ``provider`` comment. The keyless
    fix is ``provider="hosted_vllm"`` + ``api_key=None``, which ``mu_local.config`` now defaults.
    """
    provider = ProviderRecord(
        key=profile.provider_key,
        kind=ProviderKind.LOCAL_HTTP,
        litellm_provider=profile.provider,
        api_base=profile.base_url,
        credential_ref=_PROFILE_CREDENTIAL_REF if profile.api_key else None,
        is_local=True,
    )
    deployment = ModelDeployment(
        model_group=profile.model_group,
        provider_key=profile.provider_key,
        model_id=f"{profile.provider}/{profile.model}",
        kind=ModelKind.LLM,
    )
    return provider, deployment


def _profile_models(profile: SlmProfile, models: ModelSettings) -> ModelSettings:
    """Pin every LLM task at the profile's ONE model-group (unchanged behaviour, C1 preserved)."""
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


def _profile_resolver(profile: SlmProfile, catalog: ModelCatalogSettings) -> SecretSeamResolver:
    """The plane's secret seam plus, when the profile carries a key VALUE, that value under its
    NAME. The value never enters the catalog data, `extra_params`, or a log line."""
    overrides = {_PROFILE_CREDENTIAL_REF: profile.api_key} if profile.api_key else None
    return build_plane_secret_resolver(catalog, overrides=overrides)


def _resolve_profile_layer(
    profile: SlmProfile, models: ModelSettings, catalog: ModelCatalogSettings
) -> PlaneModelLayer:
    """A configured profile means *"pin every task to this deployment"*, so its source is EMPTY
    with the profile row overlaid — the shipped multi-provider table would add groups no task can
    reach. The plane DEFAULT (``llm.enabled=False``) takes the SHIPPED path instead."""
    provider, deployment = _profile_rows(profile)
    return resolve_plane_model_layer(
        models=_profile_models(profile, models),
        catalog=catalog.model_copy(update={"source": CatalogSource.EMPTY}),
        resolver=_profile_resolver(profile, catalog),
        overlay_providers=[provider],
        overlay_deployments=[deployment],
    )


def _build_llm_catalog(profile: SlmProfile) -> tuple[ModelSettings, ModelCatalogSettings]:
    """PORT of ``mu_local.composition._build_llm_catalog`` (module-level, reads only its argument
    plus the WIRED ``get_engine_settings()``), kept so the two roots stay diff-able."""
    engine_settings = get_engine_settings()
    layer = _resolve_profile_layer(profile, engine_settings.model, engine_settings.model_catalog)
    return layer.models, layer.catalog


class EngineContainer:
    """The ``LocalContainerLike``-satisfying composition root ``mu-engine-server`` builds directly
    from ``mu_engine`` primitives — NO ``mu_local`` import anywhere in this class (module
    docstring). Structurally satisfies ``mu_engine.surface.facade.LocalContainerLike`` with ZERO
    adapter code: ``ingest``/``distill``/``stm``/``recall``/``mode_gate``/``llm`` are exactly the
    attribute names and types that Protocol names.

    Assembly order mirrors ``mu_local.composition.LocalContainer.__init__`` (``mu-local/
    composition.py:196-406``) step-for-step, so a future diff between the two stays easy to audit;
    the numbered comments below are the SAME step numbers that file uses.
    """

    def __init__(
        self, settings: EngineServerSettings, *, engine_settings: EngineSettings | None = None
    ) -> None:
        self._settings = settings
        # CONFIG-AND-DATA-FIX-PLAN.md §1.2 C1: the SAME central ``EngineSettings`` root
        # ``mu_local.composition.LocalContainer`` reads (C0) — endpoints stay on
        # ``EngineServerSettings`` (module docstring), but every intelligence knob (`recall`,
        # `distill`, `ingest`, `lifecycle`, `model`, `ledger`) is pulled FROM this instance below
        # instead of being constructed bare, so the SAME `MU_RECALL__…`/`MU_LIFECYCLE__…`/
        # `MU_INGEST__…`/`MU_DISTILL__…`/`MU_MODEL__…`/`MU_LEDGER__…` env overrides reach this
        # deployable too. Injectable for tests, mirrors `settings` above.
        self._engine_settings: EngineSettings = engine_settings or get_engine_settings()
        self._closers: list[Callable[[], Awaitable[None]]] = []

        # (0) fail-loud mandatory-role validation — this root's four backends are FIXED (module
        #     docstring), so this call can never actually fail; kept for parity/defense-in-depth
        #     with LocalContainer, and because assert_mandatory_roles also refuses a graph-fold
        #     backend key, which is a real (if currently unreachable) misconfiguration class.
        assert_mandatory_roles(
            {
                "relational": _RELATIONAL_BACKEND,
                "vector": _VECTOR_BACKEND,
                "graph": _GRAPH_BACKEND,
            },
            plane=Visibility.PRIVATE,
        )

        # (1) the REAL local embedder (offline MiniLM) — dimension read FROM the live model.
        self.embedder: SentenceTransformerEmbedder = self._build_embedder(self._engine_settings)

        # (2) STM (kv role: valkey, durable — appendonly yes per docker-compose.dev.yml) — built
        #     through the SAME STORE_REGISTRY seam LocalContainer uses.
        self.stm: StmTierRepository = STORE_REGISTRY.build(
            "kv", _KV_BACKEND, url=settings.valkey.url
        )
        self._register_closer(self.stm, "_redis", "_mc")

        # (2b) item 6c — the DURABILITY POINT this task exists to land: a raw Redis client against
        #      the SAME configured Valkey endpoint, backing RedisStageLedger + C1's
        #      RedisConflictRecordRepository. A SEPARATE client instance from `self.stm`'s own
        #      (neither RedisStmAdapter/ValkeyStmAdapter nor the STORE_REGISTRY factory exposes its
        #      wrapped client for reuse) — registered as its own closer below.
        self.redis: Redis = Redis.from_url(settings.valkey.url)
        self._register_closer(self.redis)
        # C1: from the WIRED `EngineSettings.ledger` instead of a bare `LedgerSettings()`.
        self.ledger: RedisStageLedger = RedisStageLedger(
            self.redis, key_prefix=settings.ledger_key_prefix, settings=self._engine_settings.ledger
        )
        self.conflict_records: RedisConflictRecordRepository = RedisConflictRecordRepository(
            self.redis, key_prefix=settings.conflict_key_prefix, settings=ConflictRedisSettings()
        )

        # (3) MTM (vector role: qdrant) — dim from the LIVE embedder.
        #     HYBRID MTM (mtm-retrieval-design.md §1.2/§1.3): ONE sparse producer for the write
        #     side (here) and the read side (`RecallService`, below) — see mu-local's identical
        #     wiring for why they must be the same instance. `None` when
        #     `RecallSettings.sparse_enabled` is off, which is the default.
        self.sparse_encoder = build_sparse_encoder(self._engine_settings.recall)
        self.mtm: MtmTierRepository = STORE_REGISTRY.build(
            "vector",
            _VECTOR_BACKEND,
            url=settings.qdrant.url,
            dim=self.embedder.dimension,
            sparse_encoder=self.sparse_encoder,
        )
        self._register_closer(self.mtm, "_qdrant")

        # (4) LTM (graph role: falkordb, MANDATORY).
        self.ltm: GraphStorePort = STORE_REGISTRY.build(
            "graph", _GRAPH_BACKEND, host=settings.falkordb.host, port=settings.falkordb.port
        )
        self._register_closer(self.ltm, "_db.connection")
        # D-5 (ARCHITECTURE-CONFORMANCE.md entity_uids MTM payload) — same duck-typed wiring as
        # ``mu_local.composition.LocalContainer``; a no-op if either side doesn't implement the
        # structural seam (`set_mtm_entity_sink`/`set_entity_uids`).
        set_sink = getattr(self.ltm, "set_mtm_entity_sink", None)
        if callable(set_sink) and hasattr(self.mtm, "set_entity_uids"):
            set_sink(self.mtm)

        # (5) control-plane (relational, sqlite in-memory — off the ingest/recall critical path,
        #     same default LocalContainer itself ships). Disposed on close.
        self.control = STORE_REGISTRY.build("relational", _RELATIONAL_BACKEND)
        self._register_closer(self.control, "_engine")

        # (5b) ContextRepository (NEW — software-arch spec §5/§6, l.260-263/l.340), mirrors
        #      LocalContainer's own step (5b): the artifact provenance-root store
        #      `PersistRawArtifactStage` writes through. No network client to close.
        self.artifacts: ContextRepository = STORE_REGISTRY.build("artifact", _ARTIFACT_BACKEND)

        # (6) LLM: settings.llm.enabled (default True, unlike LocalContainer's llm=None default —
        #     see SlmProfile's own docstring for why) ⇒ a REAL ModelRouter over the dev SLM +
        #     LlmFactExtractor for DISTILL's SPO extraction; disabled ⇒ heuristic mode, unchanged.
        #     ENG-115a / gate G6: `model_router` is ALWAYS built now. With `llm.enabled=False` it
        #     resolves the SHIPPED multi-provider catalog (anthropic/openai/azure/deepseek/
        #     moonshot + the keyless local endpoints) narrowed by a credential probe, instead of
        #     the plane having no model layer at all; `self.llm` — and therefore the LLM extractor
        #     and the conflict adjudicator — stays governed by `llm.enabled`, unchanged.
        self.model_router: ModelRouter = self._build_plane_router(
            settings.llm if settings.llm.enabled else None
        )
        self.llm: ModelRouter | None = self.model_router if settings.llm.enabled else None
        # (6a) …and its TEARDOWN, which did not exist. `close()` below promises to "release every
        #      store connection this container opened"; the MODEL layer was outside that promise
        #      entirely, so litellm's process-global logging worker — a `while True: await
        #      queue.get()` consumer that our first async model call starts — was left pending on
        #      a loop nobody would close it on. MEASURED on the VM's full mu-core suite: nine
        #      `RuntimeError: Event loop is closed` unraisables out of `asyncio.Queue.get`, plus
        #      dropped `Logging.async_success_handler` coroutines (see
        #      `LiteLLMRouterAdapter.aclose`). `insert(0, …)` so the LIFO drain runs it LAST, and
        #      that position was CORRECTED by measurement, not chosen: appended (i.e. drained
        #      first) it still leaked, because litellm ends every async call with a bare
        #      `asyncio.create_task(_client_async_logging_helper(...))` — a deferred task that
        #      RE-STARTS the worker during any await that follows, and the remaining store closers
        #      are exactly such awaits. Draining last leaves nothing after it to restart it.
        #      `self.model_router` is always built (ENG-115a), and `aclose()` is a no-op on a
        #      router that never made a call, so this is unconditional.
        self._closers.insert(0, self.model_router.aclose)
        # C2 (mirrors `mu_local.composition`): `settings=` threaded from the WIRED
        # `EngineSettings.extraction` — previously bare, so `MU_EXTRACTION__MIN_TOKENS`/vocab
        # overrides never reached the DEFAULT (heuristic) extraction path.
        self._extractor: FactExtractorPort = HeuristicSpoExtractor(
            settings=self._engine_settings.extraction
        )
        if settings.llm.enabled:
            # C1: base on the WIRED `EngineSettings.extraction` (mirrors the identical fix in
            # `mu_local.composition`) — `max_tokens`/`temperature` still come from the SLM
            # profile (the per-call params for THIS deployed model), unchanged.
            self._extractor = LlmFactExtractor(
                self.model_router,
                model_group=settings.llm.model_group,
                settings=self._engine_settings.extraction.model_copy(
                    update={
                        "max_tokens": settings.llm.max_tokens,
                        "temperature": settings.llm.temperature,
                    }
                ),
            )

        # (7) platform singletons + the three content-free observability sinks — always ON (this
        #     is a real network-reachable server, never a bare-unit-test context).
        self._clock = SystemClock()
        self._bus = InprocBus()
        # AD-24 — `mu_engine.lifecycle.counts.TierCountCache` is DELIBERATELY NOT WIRED HERE, and
        # this comment exists so nobody wires it without reading why. It IS wired on mu-local
        # (`mu_local/composition.py`), so `GET /profile` (`routes/lifecycle.py:64-74`) reports
        # `counts_basis=UNOBSERVED` with zeros on this plane — an honest "we did not look", which
        # is exactly the value AD-24 asks for when nothing has looked.
        #
        # An earlier cut DID attach it here, on the argument that leaving it unwired would "make
        # that endpoint lie". Measured, the wiring was the worse lie on this plane, for two
        # reasons that no amount of care inside the cache can fix:
        #   (a) `self._bus = InprocBus()` above is PER-PROCESS and CANONICAL §4.1 makes the bus
        #       plane-local, so two uvicorn workers can never converge: two identical `GET /profile`
        #       calls would return DIFFERENT numbers under the same badge, and a plain restart
        #       resets one of them to zero;
        #   (b) `max_tracked_prefixes` is an LRU by WRITE recency across every tenant of a hosted
        #       plane, so busy tenants evict quiet ones and most users would read UNOBSERVED anyway
        #       — the feature degrading to the stub at precisely the scale this plane exists for.
        # Wiring it here needs a SHARED counter (a Valkey/Postgres cardinality this container can
        # read), not a second copy of an in-process cache. Until then, uniform and honest beats
        # per-replica and confident.
        self.tracer = build_tracer(enabled=True, service_name="mu-engine-server")
        self.metrics = build_metrics(enabled=True)
        # C3: `settings=` threaded from the WIRED `EngineSettings.observability` so
        # `MU_OBSERVABILITY__DURABLE_AUDIT_QUEUE_MAX` reaches `_DurableAuditLog`'s bounded queue
        # whenever a durable sink is configured (mirrors `mu_local.composition`).
        self.audit = build_audit(enabled=True, settings=self._engine_settings.observability)

        # (7b) manager-mode gate — MANUAL default (mirrors LocalContainer's own narrowing for the
        #      MANUAL VERB surface: `LocalMemory.consolidate()`-shaped calls / `POST
        #      /lifecycle/enforce` still work unmanaged-gate-free under MANUAL/HYBRID). T2 fix
        #      (CONFIG-AND-DATA-FIX-PLAN.md): this composition root NOW also starts an automatic
        #      background sweep runner (`EngineLifecycleSweepRunner`, step 9 below,
        #      `start_lifecycle_sweep()`/`stop_lifecycle_sweep()`) — unlike `LocalContainer`, which
        #      still ships none. That runner's own `sweep_user(prefix)` calls are internal,
        #      engine-driven triggers (`manual=False`, the default) — per
        #      `MemoryLifecycleManager.sweep_user`'s own docstring, "never mode-gated: MANAGED
        #      means exactly 'the engine drives this automatically'" — so they run unaffected by
        #      whichever `manager_mode` a deployment configures; only `manual=True` verbs (an
        #      external `POST /lifecycle/enforce`, a manual `consolidate()`) are ever gated.
        # C1: every OTHER lifecycle field wired from the WIRED `EngineSettings.lifecycle`
        # (`MU_LIFECYCLE__SALIENCE__W_RECENCY`, `MU_LIFECYCLE__PROMOTE_STM_MTM`, …); only
        # `manager_mode` keeps this composition root's deliberate MANUAL narrowing (structural,
        # not operator-tunable — mirrors `mu_local.composition.LocalContainer`, unchanged).
        self.lifecycle_settings = self._engine_settings.lifecycle.model_copy(
            update={"manager_mode": ManagerModeSettings(default_mode=ManagerMode.MANUAL.value)}
        )
        self.mode_gate: ManagerModeGate = ManagerModeGate(
            self.lifecycle_settings.manager_mode,
            _WorkspaceDefaultModeResolver(self.lifecycle_settings.manager_mode),
        )

        # (7c) LLM-judged conflict adjudicator, gated the same way DISTILL's extractor is — wired
        #      against C1's DURABLE RedisConflictRecordRepository (item 6c), never the in-process
        #      InMemoryConflictRecordRepository LocalContainer defaults to.
        # C1: from the WIRED `EngineSettings.distill` instead of a bare `DistillSettings()`.
        self._distill_settings = self._engine_settings.distill

        # (7b) conflict-resolution-async-design.md §4.1 — the most-specific-wins policy chain.
        #      Without a resolver, `ConflictAdjudicator` falls back to a policy fixed at
        #      construction time, which structurally cannot express "per-memory beats
        #      per-namespace": every conflict in the deployment resolved under one hardcoded
        #      `ConflictResolutionPolicy()`, and the per-namespace knob §4.1 line 155 calls
        #      "the primary knob the owner asked for" could not be set at all.
        #
        #      Step 3 (`settings.conflict.default_policy`) now comes from the WIRED
        #      `EngineSettings.conflict`, so `MU_CONFLICT__DEFAULT_POLICY__MODE=manual` reaches
        #      this deployable — the same C1 treatment every other subtree above got.
        #
        #      Steps 1 and 2 are the in-process stores: real adapters, not stubs (their own
        #      docstrings: "the sanctioned LOCAL-plane defaults"), namespace-scoped on
        #      `ns.to_prefix()`. They are the SAME instances handed to `ConflictResolutionService`
        #      below, so a policy written through `PUT /conflict-policy` is the policy the next
        #      detection reads. A durable control-plane row is the multi-tenant server's concern
        #      (mu-server), not this single-tenant deployable's.
        self._namespace_conflict_policies = InMemoryNamespaceConflictPolicyStore()
        self._memory_conflict_policies = InMemoryMemoryConflictPolicyStore()
        self.conflict_policy_resolver: ConflictPolicyResolver = ConflictPolicyResolver(
            settings=self._engine_settings.conflict,
            namespace_policies=self._namespace_conflict_policies,
            memory_policies=self._memory_conflict_policies,
        )

        # (7b-2) The §5 write actions + the §5-line-218 resolve queue. The queue is the
        #        RECORD-BACKED one, never the in-process dict: its durability IS the durable
        #        `RedisConflictRecordRepository` above, so a decision accepted and then lost to a
        #        restart is re-derived and applied on the next sweep instead of leaving a record
        #        that says RESOLVED while both items stay active forever.
        self.conflict_resolution_queue: RecordBackedResolutionQueue = RecordBackedResolutionQueue(
            self.conflict_records
        )
        self.conflict_resolution: ConflictResolutionService = ConflictResolutionService(
            records=self.conflict_records,
            queue=self.conflict_resolution_queue,
            clock=self._clock,
            bus=self._bus,
            namespace_policies=self._namespace_conflict_policies,
            memory_policies=self._memory_conflict_policies,
        )

        self.conflict_adjudicator: ConflictAdjudicator | None = None
        if self.llm is not None and self._distill_settings.use_llm_adjudicator:
            # C3 (mirrors `mu_local.composition`): `settings=` threaded from the WIRED
            # `EngineSettings.lifecycle` instead of `build_conflict_adjudicator` omitting
            # `settings=` entirely (-> a bare `ConflictAdjudicatorSettings()` fallback).
            self.conflict_adjudicator = build_conflict_adjudicator(
                use_llm=True,
                router=self.llm,
                settings=conflict_adjudicator_settings_from_lifecycle(self.lifecycle_settings),
                clock=self._clock,
                bus=self._bus,
                conflict_records=self.conflict_records,
                # §4.1 — the resolver WINS over the constructor-time `policy`, which is what
                # makes "per-memory beats per-namespace beats workspace-default" reachable.
                policy_resolver=self.conflict_policy_resolver,
            )

        # (8) application services — each SurfaceFacade verb delegates to exactly one of these.
        # C1: `settings=` threaded from the WIRED `EngineSettings.ingest` — previously omitted
        # entirely (same gap as `mu_local.composition`), so `importance_promote`/
        # `mention_promote`/`stm_ttl_s` were unreachable from the environment.
        self.ingest = IngestService(
            stm=self.stm,
            mtm=self.mtm,
            embedder=self.embedder,
            bus=self._bus,
            ledger=self.ledger,
            clock=self._clock,
            settings=self._engine_settings.ingest,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
            # NEW (software-arch spec §6, l.340-341) — mirrors LocalContainer's own wiring.
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
            # ResolveConflictStage's two arms (conflict-async §2 table). Without these the
            # machinery above is inert: a human decision recorded by `conflict_resolution` would
            # never be applied to any store, and an AUTOMATIC supersession would never close its
            # own `ConflictRecord`, leaving `AUTO_RESOLVED`/`origin=auto` unreachable.
            resolution_queue=self.conflict_resolution_queue,
            conflict_apply=self.conflict_resolution,
        )
        # C1: from the WIRED `EngineSettings.recall` instead of a bare `RecallSettings()` — the
        # exact class the `02fbed9` `recency_floor_limit` bug lived in.
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
            # ACCURACY-PLAN-0831.md item 6 (mirrors `mu_local.composition`'s identical wiring, C6
            # "never edit one without the other"): `self.model_router` is always built (ENG-115a),
            # so this plane's rerank gate is armed unconditionally — see the sibling comment in
            # `mu_local/composition.py` for the full reasoning.
            reranker=self.model_router,
        )
        authz = RecallAuthorizationFilter(
            tenancy=DefaultTenancyGuard(), authorized_ids=PrincipalAuthorizedIdsResolver()
        )
        self.recall = RecallService(
            embedder=self.embedder,
            private_ranker=ranker,
            # §1.3 "M2 resolution": the façade encodes the sparse query at the same boundary it
            # embeds the dense one — the SAME instance the MTM adapter writes with, above.
            sparse_encoder=self.sparse_encoder,
            shared_recall=_NullSharedRecall(clock=self._clock),
            authz=authz,
            fusion=fusion,
            settings=recall_settings,
            clock=self._clock,
            metrics=self.metrics,
            tracer=self.tracer,
            # ACCURACY-PLAN-0831.md item 4 (mirrors `mu_local.composition`'s identical wiring, C6
            # "never edit one without the other"): `self.model_router` is always built (ENG-115a),
            # so this plane's width derivation is armed unconditionally — a pure catalog metadata
            # read, never gated on `self.llm`/`settings.llm.enabled` (see this file's own (6)
            # comment for why chunking's identical lookup is already unconditional).
            context_budget=self.model_router,
        )

        # (9) T2 fix (CONFIG-AND-DATA-FIX-PLAN.md) — the real MemoryLifecycleManager (MLM) + the
        #     in-process automatic sweep runner (`lifecycle_runner.py`'s own module docstring has
        #     the full root-cause + design citation). Mirrors `LocalContainer.build_lifecycle_
        #     manager`'s assembly (`mu_local/composition.py:507-597`) step-for-step: SAME
        #     `stm`/`mtm`/`distill`/`bus`/`clock`/`mode_gate`/`conflict_adjudicator` instances this
        #     container already built above — never a second, independently-constructed set
        #     (DEV-STANDARDS rule 9: one composition root, one set of adapters). Unlike
        #     `LocalContainer` (which only EXPOSES `build_lifecycle_manager()` as a factory a
        #     daemonless embedded caller may or may not invoke), this always-on server ALWAYS
        #     builds one — `lifecycle_manager=None` in `EngineServerApp.__init__` (a tracked gap
        #     noted in this module's own prior revision) is now closed.
        lifecycle_salience = SalienceStrategy(self.lifecycle_settings.salience)
        lifecycle_promotion = PromotionService(
            mtm=self.mtm,
            distill=self.distill,  # SAME object LocalMemory-shaped `consolidate()` delegates to
            salience=lifecycle_salience,
            # D1 (STATE-AND-DEFECTS-0829.md): the SAME embedder instance the ingest-time gate
            # (`DeterministicPromoteStage`, wired above) and the recall ranker already use —
            # never a second one (DEV-STANDARDS rule 9).
            embedder=self.embedder,
            stm=self.stm,
            settings=self.lifecycle_settings,
            clock=self._clock,
            bus=self._bus,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
        )
        lifecycle_demotion = DemotionService(
            stm=self.stm,
            mtm_remove=self.mtm,
            salience=lifecycle_salience,
            settings=self.lifecycle_settings,
            clock=self._clock,
            bus=self._bus,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
        )
        self.lifecycle_manager: MemoryLifecycleManager = MemoryLifecycleManager(
            salience=lifecycle_salience,
            promotion=lifecycle_promotion,
            demotion=lifecycle_demotion,
            distill=self.distill,
            conflict=self.conflict_adjudicator,  # SAME instance self.distill was built with
            mode_gate=self.mode_gate,
            bus=self._bus,
            settings=self.lifecycle_settings,
            clock=self._clock,
            # counts=…: intentionally omitted on this plane — see the AD-24 note above.
        )
        self.lifecycle_runner: EngineLifecycleSweepRunner = EngineLifecycleSweepRunner(
            bus=self._bus,
            lifecycle_manager=self.lifecycle_manager,
            settings=settings.lifecycle_sweep,
        )
        self._lifecycle_runner_task: asyncio.Task[None] | None = None

        # ---------------------------------------------------------------- (10) PERSONA (§5.2)
        #: PORT of ``LocalContainer``'s own persona block, one plane over — same
        #: ``build_persona`` call, same ABSENCE rule (``None`` when persona is disabled or no
        #: model is wired, because spec line 103's slot tagger IS ``models.classify_model``), and
        #: the same two bus subscriptions, so the two planes cannot drift into two personas.
        #:
        #: **The partition reader is built HERE and is persona's own**, unlike mu-local's, which
        #: reuses the container's shared ``MemoryRepository`` façade. This container has no such
        #: façade: nothing on this plane built a ``TierRouter``/``TieredMemoryRepository`` (a real
        #: asymmetry between the two roots — ``health``/``pin`` are consequently absent here too).
        #: ``TierRouter`` is a stateless dispatcher over the stm/mtm/ltm adapters THIS container
        #: already built, so this opens no second connection and no second adapter set
        #: (DEV-STANDARDS rule 9 is about instances of state, and this holds none). It is named
        #: privately so that when this plane grows the shared façade, persona switches to it
        #: without a name collision — REPORTED as the right follow-up, not silently absorbed.
        self._persona_memory: TieredMemoryRepository = TieredMemoryRepository(
            router=TierRouter(
                (
                    TierLeg(_Tier.STM, self.stm, backend=_KV_BACKEND),
                    TierLeg(_Tier.MTM, self.mtm, backend=_VECTOR_BACKEND),
                    TierLeg(_Tier.LTM, self.ltm, backend=_GRAPH_BACKEND),
                )
            ),
            embedder=self.embedder,
        )
        self.persona: PersonaWiring | None = build_persona(
            memory=self._persona_memory,
            router=self.llm,
            bus=self._bus,
            clock=self._clock,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
        )
        if self.persona is not None:
            # §5.2's topic-affinity prior, as a DECORATOR over the finished ranked read. The
            # ``cast`` carries the identical, reported caveat mu-local's does: ``mu_engine.
            # surface.facade.LocalContainerLike`` declares ``recall: RecallService`` as an
            # invariant attribute, and that file is neither this lane's to edit nor allowed to
            # name persona at all. The wrapper implements that class's entire public surface and
            # ``test_persona_shaping_unit`` asserts it.
            self.recall = cast("RecallService", self.persona.shaped(self.recall))

    async def start_lifecycle_sweep(self) -> None:
        """Starts `self.lifecycle_runner.run()` as a background `asyncio.Task` (T2 wiring) — a
        no-op (never a silent double-start) if a task is already running. Called by
        `EngineServerApp` at app-lifecycle startup (`docker/serve.py`'s lifespan)."""
        if self._lifecycle_runner_task is not None and not self._lifecycle_runner_task.done():
            return
        self._lifecycle_runner_task = asyncio.create_task(
            self.lifecycle_runner.run(), name="lifecycle-sweep-runner"
        )

    async def stop_lifecycle_sweep(self) -> None:
        """Signals the runner to stop and awaits its background task's clean exit (ordered
        shutdown, mirrors `close()`'s own best-effort discipline) — a no-op if never started."""
        await self.lifecycle_runner.stop()
        if self._lifecycle_runner_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._lifecycle_runner_task
            self._lifecycle_runner_task = None

    @property
    def bus(self) -> EventBusPort:
        """The SAME real ``InprocBus`` instance threaded into ``IngestService``/``DistillPipeline``/
        ``self.lifecycle_runner`` above — PORT of ``LocalContainer.bus``'s identical accessor
        (``mu_local/composition.py``: "a caller that needs to subscribe to THIS container's real
        event stream ... must observe the identical bus captured memories publish onto, never a
        second, independently-constructed ``InprocBus``"). Exists for the same reason there: an
        external caller (a test verifying the T2 sweep, an operator script) that wants to observe
        this process's real capture events needs this exact instance, not a fresh one."""
        return self._bus

    async def close(self) -> None:
        """Release every store connection this container opened (LIFO), best-effort per client —
        identical discipline to ``LocalContainer.close`` (``mu-local/composition.py:408-413``).
        Stops the lifecycle sweep runner FIRST (best-effort) so no in-flight sweep races a store
        connection this call is about to release."""
        with contextlib.suppress(Exception):
            await self.stop_lifecycle_sweep()
        for closer in reversed(self._closers):
            with contextlib.suppress(Exception):
                await closer()
        self._closers.clear()

    async def health(self) -> dict[str, str]:
        """Best-effort liveness probe of the durable Valkey client (item 6c's own durability
        point) + the STM tier repository — a NAMED per-store status map, never a bare boolean
        (DEV-STANDARDS: no silent/aggregate-only health signal). Kept on ``EngineContainer``
        itself (not ``EngineServerApp``) so it can read ``self.redis``/``self.stm`` directly
        rather than an outer caller reaching into this class's internals."""
        status: dict[str, str] = {}
        try:
            await self.redis.ping()
            status["valkey"] = "ok"
        except Exception as exc:
            status["valkey"] = f"error: {exc}"
        try:
            await self.stm.get(_HEALTH_PROBE_NS, "__health__")  # exercises the live connection
            status["stm"] = "ok"
        except Exception as exc:
            status["stm"] = f"error: {exc}"
        return status

    # ------------------------------------------------------------------------------------ helpers
    @staticmethod
    def _build_embedder(engine_settings: EngineSettings) -> SentenceTransformerEmbedder:
        # C2 (mirrors `mu_local.composition`): the catalog backing this embedder is now the
        # WIRED `EngineSettings.model_catalog` (`MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND`/
        # `MU_MODEL_CATALOG__DEFAULT_MINILM_PATH`), not a bare `default_local_catalog()` call.
        catalog = default_local_catalog(engine_settings.model_catalog)
        embedder = build_embedder(_EMBEDDING_BACKEND, catalog)
        if not isinstance(embedder, SentenceTransformerEmbedder):
            raise BackendUnavailableError(
                f"embedding backend {_EMBEDDING_BACKEND!r} did not resolve to a local embedder"
            )
        return embedder

    def _build_plane_router(self, profile: SlmProfile | None) -> ModelRouter:
        """The plane's REAL ``ModelRouter`` through the ONE model-layer entry point (mirrors
        ``mu_local.composition.LocalContainer._build_plane_router``).

        ``profile is None`` (``llm.enabled=False``) ⇒ the SHIPPED catalog + the credential probe;
        a profile ⇒ that ONE deployment, every task pinned to it. C1: `chunk_token_ratio` still
        threaded from the WIRED `EngineSettings.extraction`.
        """
        ratio = self._engine_settings.extraction.chunk_token_ratio
        catalog = self._engine_settings.model_catalog
        if profile is None:
            return build_plane_router(
                models=self._engine_settings.model,
                catalog=catalog,
                chunk_token_ratio=ratio,
                resolver=build_plane_secret_resolver(catalog),
            )
        layer = _resolve_profile_layer(profile, self._engine_settings.model, catalog)
        return build_model_router(
            models=layer.models,
            catalog=layer.catalog,
            secret_resolver=_profile_resolver(profile, catalog),
            chunk_token_ratio=ratio,
        )

    def _register_closer(self, adapter: object, *client_paths: str) -> None:
        closer = self._resolve_closer(adapter, *client_paths)
        if closer is not None:
            self._closers.append(closer)

    @staticmethod
    def _resolve_closer(
        adapter: object, *client_paths: str
    ) -> Callable[[], Awaitable[None]] | None:
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


class EngineServerApp:
    """start/health/shutdown lifecycle over an :class:`EngineContainer` + the C2 ``FastAPI`` app
    (build-plan §4 C4). The ONE object a real deployment (uvicorn) or a test constructs.

    ``app`` is built eagerly at construction time (C2's ``build_app`` is a pure, synchronous
    factory — no I/O of its own); ``EngineContainer`` construction IS where the real I/O-adjacent
    work happens (vendor client construction — connections are lazy on first use for every
    adapter here, matching ``LocalContainer``'s own "cheap, synchronous construction" discipline).
    """

    def __init__(self, settings: EngineServerSettings | None = None) -> None:
        self.settings: EngineServerSettings = settings or load_settings()
        self.container: EngineContainer = EngineContainer(self.settings)
        self.facade: SurfaceFacade = SurfaceFacade(
            self.container, workspace=self.settings.workspace, namespace=self.settings.namespace
        )
        verifier = make_bearer_verifier(self.settings.token_path)
        self.app: FastAPI = build_app(
            self.facade,
            # T2 fix (CONFIG-AND-DATA-FIX-PLAN.md): a real MemoryLifecycleManager is now always
            # composed (EngineContainer step 9) — the prior tracked gap ("MemoryLifecycleManager
            # wiring — tracked gap, not this task") is closed. `GET /profile`/`POST
            # /lifecycle/enforce`/`GET /lifecycle/events` (routes/lifecycle.py) stop 501-ing as a
            # direct consequence.
            lifecycle_manager=self.container.lifecycle_manager,
            workspace=self.settings.workspace,
            namespace=self.settings.namespace,
        )
        # Swap C3's default (arg > MU_ENGINE_SERVER_TOKEN_PATH env > ~/.memory-universe/...)
        # verifier for one bound to THIS settings tree's token_path — same FastAPI override
        # mechanism app.py's own docstring names ("app.dependency_overrides[require_bearer_token]
        # = ..."), so a caller that configures `settings.token_path` differently from the env var
        # gets that value honored without any code-level change.
        self.app.dependency_overrides[require_bearer_token] = verifier

    async def health(self) -> dict[str, str]:
        """Delegates to :meth:`EngineContainer.health` — the per-store status map C4's caller
        (a ``/health`` route, a readiness probe) reads."""
        return await self.container.health()

    async def start(self) -> None:
        """Starts the automatic lifecycle-sweep runner (T2 fix) as a background task — called at
        app-lifecycle startup (`docker/serve.py`'s lifespan `try` block, before `yield`). A no-op
        (never blocks, never raises) when `EngineServerSettings.lifecycle_sweep.enabled` is False —
        `EngineLifecycleSweepRunner.run()` itself returns immediately in that case (see its own
        docstring), so the background task this schedules completes right away rather than looping
        forever doing nothing."""
        await self.container.start_lifecycle_sweep()

    async def shutdown(self) -> None:
        """Release every store connection this app's container opened (LIFO, best-effort). Stops
        the lifecycle-sweep runner first (`EngineContainer.close`'s own ordering)."""
        await self.container.close()


# A fixed, never-written probe namespace for EngineServerApp.health()'s STM read — module-level
# (not a bare literal inline in the method, DEV-STANDARDS rule 3) since it never varies per call.
_HEALTH_PROBE_NS = _Namespace(
    org="mu-engine-server",
    workspace="health",
    user="probe",
    session="probe",
    visibility=_NsVisibility.PRIVATE,
)
