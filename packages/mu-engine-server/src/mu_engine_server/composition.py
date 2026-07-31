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

import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI
from redis.asyncio import Redis

from mu_contracts.domain.errors import BackendUnavailableError
from mu_contracts.domain.model.memory import Namespace as _Namespace
from mu_contracts.domain.model.memory import Visibility as _NsVisibility
from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_contracts.ports.time import Clock
from mu_engine.config import EngineSettings, get_engine_settings
from mu_engine.lifecycle.conflict import ConflictAdjudicator, build_conflict_adjudicator
from mu_engine.lifecycle.conflict_redis import ConflictRedisSettings, RedisConflictRecordRepository
from mu_engine.lifecycle.mode_gate import ManagerMode, ManagerModeGate
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
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings, default_local_catalog
from mu_engine.services.extract import (
    FactExtractorPort,
    HeuristicSpoExtractor,
    LlmFactExtractor,
)
from mu_engine.services.ingest import IngestService
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


def _build_llm_catalog(profile: SlmProfile) -> tuple[ModelSettings, ModelCatalogSettings]:
    """PORT of ``mu_local.composition._build_llm_catalog`` (module-level function, no ``mu_local``
    state — reads only its ``profile`` argument), layering ONE ``ProviderKind.LOCAL_HTTP``
    deployment onto ``default_local_catalog()`` — the same shape the reference SLM integration
    test builds (``mu-engine/tests/pipelines/test_distill_llm_slm_int.py:163-196``).

    CONFIG-AND-DATA-FIX-PLAN.md §1.2 C1 (mirrors the identical fix in ``mu_local.composition``):
    ``ModelSettings`` is derived from the WIRED ``get_engine_settings().model`` via ``model_copy``
    instead of constructed bare, so ``MU_MODEL__MAX_OUTPUT_TOKENS``/``MU_MODEL__TEMPERATURE`` are
    reachable — ``profile.max_tokens``/``profile.temperature`` stay the SLM's own per-call
    extraction params (threaded into ``ExtractionSettings`` separately, unchanged).
    """
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
        extra_params={"api_key": profile.api_key},
    )
    catalog_base = default_local_catalog()
    catalog = catalog_base.model_copy(update={"providers": [provider], "deployments": [deployment]})
    model_defaults = get_engine_settings().model
    models = model_defaults.model_copy(
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
    return models, catalog


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
        self.embedder: SentenceTransformerEmbedder = self._build_embedder()

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
        self.mtm: MtmTierRepository = STORE_REGISTRY.build(
            "vector", _VECTOR_BACKEND, url=settings.qdrant.url, dim=self.embedder.dimension
        )
        self._register_closer(self.mtm, "_qdrant")

        # (4) LTM (graph role: falkordb, MANDATORY).
        self.ltm: GraphStorePort = STORE_REGISTRY.build(
            "graph", _GRAPH_BACKEND, host=settings.falkordb.host, port=settings.falkordb.port
        )
        self._register_closer(self.ltm, "_db.connection")

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
        self.llm: ModelRouter | None = None
        self._extractor: FactExtractorPort = HeuristicSpoExtractor()
        if settings.llm.enabled:
            self.llm = self._build_llm_router(settings.llm)
            # C1: base on the WIRED `EngineSettings.extraction` (mirrors the identical fix in
            # `mu_local.composition`) — `max_tokens`/`temperature` still come from the SLM
            # profile (the per-call params for THIS deployed model), unchanged.
            self._extractor = LlmFactExtractor(
                self.llm,
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
        self.tracer = build_tracer(enabled=True, service_name="mu-engine-server")
        self.metrics = build_metrics(enabled=True)
        self.audit = build_audit(enabled=True)

        # (7b) manager-mode gate — MANUAL default (mirrors LocalContainer's own narrowing:
        #      no MaintenanceLoop/daemon runs the auto sweep here either).
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
        self.conflict_adjudicator: ConflictAdjudicator | None = None
        if self.llm is not None and self._distill_settings.use_llm_adjudicator:
            self.conflict_adjudicator = build_conflict_adjudicator(
                use_llm=True,
                router=self.llm,
                clock=self._clock,
                bus=self._bus,
                conflict_records=self.conflict_records,
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
            bus=self._bus,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
            adjudicator=self.conflict_adjudicator,
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
        )
        authz = RecallAuthorizationFilter(
            tenancy=DefaultTenancyGuard(), authorized_ids=PrincipalAuthorizedIdsResolver()
        )
        self.recall = RecallService(
            embedder=self.embedder,
            private_ranker=ranker,
            shared_recall=_NullSharedRecall(clock=self._clock),
            authz=authz,
            fusion=fusion,
            settings=recall_settings,
            clock=self._clock,
            metrics=self.metrics,
            tracer=self.tracer,
        )

    async def close(self) -> None:
        """Release every store connection this container opened (LIFO), best-effort per client —
        identical discipline to ``LocalContainer.close`` (``mu-local/composition.py:408-413``)."""
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
    def _build_embedder() -> SentenceTransformerEmbedder:
        embedder = build_embedder(_EMBEDDING_BACKEND, default_local_catalog())
        if not isinstance(embedder, SentenceTransformerEmbedder):
            raise BackendUnavailableError(
                f"embedding backend {_EMBEDDING_BACKEND!r} did not resolve to a local embedder"
            )
        return embedder

    def _build_llm_router(self, profile: SlmProfile) -> ModelRouter:
        """C1: threads `chunk_token_ratio` from the WIRED `EngineSettings.extraction` (mirrors
        the identical fix in `mu_local.composition`) — no longer a `@staticmethod` so it can read
        `self._engine_settings`."""
        models, catalog = _build_llm_catalog(profile)
        return build_model_router(
            models=models,
            catalog=catalog,
            chunk_token_ratio=self._engine_settings.extraction.chunk_token_ratio,
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
            lifecycle_manager=None,  # MemoryLifecycleManager wiring — tracked gap, not this task
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

    async def shutdown(self) -> None:
        """Release every store connection this app's container opened (LIFO, best-effort)."""
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
