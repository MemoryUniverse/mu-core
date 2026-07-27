"""``LocalContainer`` — the ONE composition root for the embedded LOCAL engine (spec §2.2).

PORT of ``mem0.Memory.__init__`` (``other_repos/mem0/mem0/memory/main.py:172-204``): build every
store + the embedder + the LLM from config in one place, then hand the wired graph to the facade.
Re-expressed over MU ports + the fail-loud registry, with mem0's ``graph optional`` branch
(``main.py:199-204``) REMOVED — graph is MANDATORY (CANONICAL storage; spec §2.2/§3.1). This is
the SOLE owner of the composition + the one default set (APPLY-PLAN B-4); ``mu-client``'s
``LocalContainer`` and the daemon ``EngineHost`` WRAP this, add no engine logic.

Backend SELECTION + mandatory-role validation reuse the phase-0 engine registry
(``mu_engine.storage.registry`` / ``factories.STORE_REGISTRY``). The network-store CLIENTS are
constructed here (not hidden inside a registry factory) so the facade owns their async lifecycle
and ``close()`` deterministically releases every connection (DEV-STANDARDS resource management) —
the exact wiring the engine's own REAL-container integration fixtures use. One Redis connection is
shared by STM and the pipeline ledger (matching those fixtures).

NO LLM this phase (Azure PARKED): ``llm`` is ``None`` ⇒ heuristic mode. Ingest promotion and the
DISTILL extractor are deterministic (real-integration-testable now); any synthesis verb refuses
loudly (:class:`~mu_local.errors.LlmNotConfiguredError`).
"""
# mypy: disable-error-code="arg-type"
# ^ The SAME systemic decorator-typing gap the engine's own integration wiring documents and
#   suppresses (see mu-engine tests/services/test_recall_federate_live_int.py header): every
#   `@retry_io`-decorated adapter method is annotated to return `Awaitable[R]`, which WIDENS it
#   past the `Coroutine[...]` an async Protocol method requires, so a concrete adapter
#   (RedisStmAdapter / QdrantMtmAdapter / FalkorLtmAdapter / RedisStageLedger) fails the
#   structural match at THIS composition site. A `platform/decorators.py` typing bug, out of this
#   file's ownership, NEVER a runtime issue — the objects ARE the ports at runtime (the REAL
#   container integration test proves it).

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable

from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings, get_settings
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.pipelines.ledger import InMemoryStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
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
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.namespace import Visibility
from mu_engine.storage.factories import STORE_REGISTRY
from mu_engine.storage.registry import assert_mandatory_roles
from mu_local.config import BackendChoice, StorageSettings
from mu_local.errors import BackendUnavailableError
from mu_local.shared_null import LocalNullSharedRecall

__all__ = ["LocalContainer"]

# The backends the phase-0 registry actually ships, per role. Selecting anything else is a NAMED
# fail-loud refusal (spec §7) — never a silent fallback to a different backend.
_SUPPORTED_KV = frozenset({"redis"})
_SUPPORTED_VECTOR = frozenset({"qdrant"})
_SUPPORTED_GRAPH = frozenset({"falkordb"})
_SUPPORTED_RELATIONAL = frozenset({"sqlite", "postgres"})
_SUPPORTED_EMBEDDING = frozenset({"minilm_local"})


class LocalContainer:
    """The ONE place adapters bind for the embedded LOCAL engine (spec §2.2)."""

    def __init__(self, storage: StorageSettings, *, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._closers: list[Callable[[], Awaitable[None]]] = []

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

        # (2) STM (redis). The durable KV write is the facade's durability floor; the pipeline
        #     ledger is IN-PROCESS (InMemoryStageLedger) — a cross-PROCESS durable ledger is a
        #     daemon concern (mu-client), not this daemonless single-session facade. An in-process
        #     ledger also keeps promote idempotency scoped to THIS instance, so a content_hash
        #     (which the engine keys content-only, not per-η) never collides across tenants/runs.
        redis_client = Redis.from_url(self._redis_url(storage.kv), decode_responses=False)
        self._closers.append(redis_client.aclose)
        self.stm = RedisStmAdapter(redis_client)
        self._ledger = InMemoryStageLedger()

        # (3) MTM (qdrant) — dim from the LIVE embedder.
        qdrant_client = AsyncQdrantClient(url=self._qdrant_url(storage.vector))
        self._closers.append(qdrant_client.close)
        self.mtm = QdrantMtmAdapter(qdrant_client, dim=self.embedder.dimension)

        # (4) LTM (falkordb) — the MANDATORY graph engine.
        host, port = self._falkor_endpoint(storage.graph)
        falkor_db = FalkorDB(host=host, port=port)
        self._closers.append(falkor_db.connection.aclose)
        self.ltm = FalkorLtmAdapter(falkor_db)

        # (5) control-plane (relational) — built through the STORE_REGISTRY seam (reuse). Off the
        #     ingest/recall critical path this slice; disposed on close.
        self.control = STORE_REGISTRY.build(
            "relational", storage.relational.backend, **self._relational_cfg(storage.relational)
        )
        engine = getattr(self.control, "_engine", None)
        if engine is not None:
            self._closers.append(engine.dispose)

        # (6) LLM PARKED ⇒ heuristic mode.
        self.llm: object | None = None

        # (7) platform singletons.
        self._clock = SystemClock()
        self._bus = InprocBus()

        # (8) application services — each facade verb delegates to exactly one of these.
        self.ingest = IngestService(
            stm=self.stm,
            mtm=self.mtm,
            embedder=self.embedder,
            bus=self._bus,
            ledger=self._ledger,
            clock=self._clock,
        )
        self.distill = DistillPipeline(
            ltm=self.ltm, extractor=HeuristicSpoExtractor(), clock=self._clock, mtm=self.mtm
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
        """NAMED fail-loud refusal of a backend the phase-0 registry does not ship (spec §7)."""
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
                    f"role {role!r} backend {backend!r} is not shipped by the phase-0 registry "
                    f"(available: {sorted(supported)}); the embedded zero-infra floor "
                    "(in-proc KV / FAISS / embedded Kùzu) is a tracked, not-yet-built gap — "
                    "no silent fallback"
                )

    def _build_embedder(self, choice: BackendChoice) -> SentenceTransformerEmbedder:
        embedder = build_embedder(choice.backend, default_local_catalog())
        if not isinstance(embedder, SentenceTransformerEmbedder):  # fail-loud, never a silent None
            raise BackendUnavailableError(
                f"embedding backend {choice.backend!r} did not resolve to a local embedder"
            )
        return embedder

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
        return {}  # sqlite factory default = in-memory
