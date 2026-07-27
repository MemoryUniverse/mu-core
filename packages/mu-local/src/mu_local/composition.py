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

NO LLM this phase (Azure PARKED): ``llm`` is ``None`` ⇒ heuristic mode. Ingest promotion and the
DISTILL extractor are deterministic (real-integration-testable now); any synthesis verb refuses
loudly (:class:`~mu_local.errors.LlmNotConfiguredError`).
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

from mu_contracts.config import Settings, get_settings
from mu_contracts.ports.observability import AuditLog, MetricSink, Tracer
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.pipelines.ledger import InMemoryStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.observability import build_audit, build_metrics, build_tracer
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.providers.embedding import SentenceTransformerEmbedder, build_embedder
from mu_engine.providers.settings import default_local_catalog
from mu_engine.services.extract import HeuristicSpoExtractor
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
from mu_local.config import BackendChoice, ObservabilitySettings, StorageSettings
from mu_local.errors import BackendUnavailableError
from mu_local.shared_null import LocalNullSharedRecall

__all__ = ["LocalContainer"]

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


class LocalContainer:
    """The ONE place adapters bind for the embedded LOCAL engine (spec §2.2)."""

    def __init__(
        self,
        storage: StorageSettings,
        *,
        settings: Settings | None = None,
        observability: ObservabilitySettings | None = None,
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

        # (6) LLM PARKED ⇒ heuristic mode.
        self.llm: object | None = None

        # (7) platform singletons + the three content-free observability sinks (DEV-STANDARDS
        #     rule 4) — built once here from ObservabilitySettings and threaded into every service.
        self._clock = SystemClock()
        self._bus = InprocBus()
        self.tracer: Tracer = build_tracer(enabled=self._obs.otel_enabled, service_name="mu-local")
        self.metrics: MetricSink = build_metrics(enabled=self._obs.metrics_enabled)
        self.audit: AuditLog = build_audit(enabled=self._obs.audit_enabled)

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
            extractor=HeuristicSpoExtractor(),
            clock=self._clock,
            mtm=self.mtm,
            tracer=self.tracer,
            metrics=self.metrics,
            audit=self.audit,
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
