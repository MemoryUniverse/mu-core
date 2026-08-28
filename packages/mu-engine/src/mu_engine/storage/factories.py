"""Default ``StoreRegistry`` population — one factory per (role, backend).

The ONE place adapters bind to real clients (``storage-pluggable §4.2/§4.3``). Each factory
reads its knobs from a ``BackendChoice.config``-shaped dict (``dsn``/``url``/``host``/
``port``/``dim``), NEVER a hardcoded literal — the values flow from the central Settings
tree (DEV-STANDARDS rule 3). Registered on import so ``STORE_REGISTRY.build(role, backend,
**cfg)`` resolves.

EXTENSION SEAM (graph role, owner override 2026-07-27): a future graph driver — e.g.
``neo4j``, ``kuzu``, ``ladybug``, ``neptune`` — registers with ZERO change to the engine or
to this file's existing entries by adding one more block of this exact shape::

    @STORE_REGISTRY.register("graph", "neo4j")  # new (role, backend) key
    def _build_neo4j(**cfg: Any) -> SomeNeo4jGraphAdapter:
        driver = Neo4jDriver(uri=cfg["uri"], auth=(cfg["user"], cfg["password"]))
        return SomeNeo4jGraphAdapter(driver)  # must satisfy GraphStorePort

The engine, ``mu_engine.storage.ports.GraphStorePort``, and every caller only ever import
the Protocol; none of them import ``falkordb`` (or would import ``neo4j``) directly — the
vendor client lives ONLY inside the adapter + this factory. Selecting the new backend is then
a config change (``StorageSettings.graph.backend = "neo4j"``), never a code change.

VECTOR ROLE (owner override 2026-07-27: "genuinely MULTI-backend NOW, mem0 pattern"): the
``vector`` role now self-registers FIVE backends the exact same way — ``qdrant`` (reference,
SHARED-safe), ``pgvector`` (SHARED-safe, SQL WHERE + HNSW), ``chroma`` (embedded floor, partial
filter — widened, never incomplete), ``faiss`` (brute-force, PRIVATE-plane ONLY — the registry's
``_BRUTE_FORCE_VECTOR_BACKENDS`` gate refuses it on SHARED at build time), and ``weaviate`` (ADR
0050: native multi-tenancy, the SHARED-plane MTM backend — filterable, gated on that ADR's own
gate 0 authz-completeness spike before any migration is scheduled). Every one of them implements
``MtmTierRepository`` and none of them is imported by the engine directly — only through this
factory, exactly like the graph seam above. Selecting a different vector backend is a
``StorageSettings.vector`` / ``BackendChoice`` config change, never a code change.

WEAVIATE FACTORY NOTE (delta CLOSED): ``WeaviateSettings`` now exists on the central ``Settings``
tree, mirroring ``QdrantSettings``/``PgVectorSettings``, so ``_build_weaviate`` resolves EVERY
knob it threads — connection (``host``/``http_port``/``http_secure``/``grpc_*``), the I/O budget,
and the two recall over-fetch tunables (``semantic_overfetch_factor``/
``semantic_overfetch_max_extra``) — from that subtree, with ``cfg`` (a ``BackendChoice.config``
override) taking precedence, exactly like every other vector factory. The adapter's module
constants survive only as its documented CONSTRUCTOR DEFAULTS for a caller that builds it
directly; nothing reachable through the registry is a fixed literal any more (DEV-STANDARDS
rule 3).

KV + RELATIONAL ROLES (owner stage 2026-07-27, same mem0-pattern multi-backend requirement):
``kv`` now self-registers FOUR backends — ``redis``/``valkey`` (wire-identical, real
``mu-dev-cache`` — recreated as ``valkey/valkey:8-alpine``), ``memory`` (embedded, in-process
floor, D2), ``memcached`` (real ``mu-dev-memcached``, CAS-emulated recency floor, D6).
``relational`` adds ``mysql`` (real ``mu-dev-mysql``, ``asyncmy`` async driver) alongside the
existing ``postgres``/``sqlite`` — the SAME ``RelationalControlPlaneAdapter`` class binds to
all three dialects (dialect-aware upsert seam, spec §2.1/§3.1, D7). Every new backend is
selected purely by a ``StorageSettings`` config value, never a code change at the call site.

ARTIFACT ROLE (NEW — software-arch spec §5 ``ContextRepository``, l.260-263): a fifth role,
``artifact``, self-registers its first backend, ``filesystem`` (``content_fs.py`` — the
LOCAL-plane provenance-root store `PersistRawArtifactStage` (``pipelines/concrete/ingest.py``)
writes through). NOT mandatory (``StoreRegistry.MANDATORY_ROLES`` is unchanged: relational +
vector + graph) — a composition root that omits it simply runs without the reference-capture
stage, byte-identical to before this role existed. Selecting a future versioned backend
(``content_git`` — spec l.437, "ported from Letta Context Repositories") is, exactly like every
role above, a config change (``StorageSettings.artifact`` / a new ``BackendChoice``), never an
engine change.
"""

from __future__ import annotations

from typing import Any

import aiomcache
import weaviate
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import create_async_engine

from mu_contracts.config import get_settings
from mu_engine.config.engine_settings import get_engine_settings
from mu_engine.storage.adapters.chroma_mtm import ChromaMtmAdapter
from mu_engine.storage.adapters.content_fs import FsContextRepositoryAdapter
from mu_engine.storage.adapters.faiss_mtm import FaissMtmAdapter
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.memcached_stm import MemcachedStmAdapter
from mu_engine.storage.adapters.memory_stm import InMemoryStmAdapter
from mu_engine.storage.adapters.pgvector_mtm import PgVectorMtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.adapters.relational_control import RelationalControlPlaneAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.adapters.weaviate_mtm import WeaviateMtmAdapter
from mu_engine.storage.registry import StoreRegistry

__all__ = ["STORE_REGISTRY"]

STORE_REGISTRY = StoreRegistry()


@STORE_REGISTRY.register("relational", "postgres")
def _build_postgres(**cfg: Any) -> RelationalControlPlaneAdapter:
    pg_settings = get_settings().storage.postgres
    engine = create_async_engine(cfg["dsn"], pool_pre_ping=True)
    return RelationalControlPlaneAdapter(
        engine,
        store_io_timeout_s=cfg.get("store_io_timeout_s", pg_settings.store_io_timeout_s),
    )


@STORE_REGISTRY.register("relational", "sqlite")
def _build_sqlite(**cfg: Any) -> RelationalControlPlaneAdapter:
    """SQLite is the embedded/test-only relational floor with no dedicated ``SQLiteSettings``
    subtree (unlike postgres/mysql) — its I/O timeout falls back to the adapter's own named
    constructor default (never a bare literal) when ``cfg`` doesn't override it.
    """
    engine = create_async_engine(cfg.get("dsn", "sqlite+aiosqlite:///:memory:"))
    kwargs: dict[str, Any] = {}
    if "store_io_timeout_s" in cfg:
        kwargs["store_io_timeout_s"] = cfg["store_io_timeout_s"]
    return RelationalControlPlaneAdapter(engine, **kwargs)


@STORE_REGISTRY.register("relational", "mysql")
def _build_mysql(**cfg: Any) -> RelationalControlPlaneAdapter:
    """MySQL/MariaDB relational backend (``storage-pluggable-spec.md §3.1`` "new | dev's stack
    is MySQL"), ``asyncmy`` async driver. Falls back to the central ``MySQLSettings`` when
    ``cfg`` doesn't supply a ``dsn`` (DEV-STANDARDS rule 3), exactly like the ``pgvector``/
    ``chroma`` vector factories below fall back to their own settings subtree.
    """
    mysql_settings = get_settings().storage.mysql
    dsn = cfg.get("dsn") or mysql_settings.dsn
    engine = create_async_engine(dsn, pool_pre_ping=True)
    return RelationalControlPlaneAdapter(
        engine,
        store_io_timeout_s=cfg.get("store_io_timeout_s", mysql_settings.store_io_timeout_s),
    )


@STORE_REGISTRY.register("vector", "qdrant")
def _build_qdrant(*, dim: int, **cfg: Any) -> QdrantMtmAdapter:
    qdrant_settings = get_settings().storage.vector
    client = AsyncQdrantClient(url=cfg["url"], prefer_grpc=cfg.get("prefer_grpc", False))
    return QdrantMtmAdapter(
        client,
        dim=dim,
        store_io_timeout_s=cfg.get("store_io_timeout_s", qdrant_settings.store_io_timeout_s),
    )


@STORE_REGISTRY.register("graph", "falkordb")
def _build_falkordb(**cfg: Any) -> FalkorLtmAdapter:
    """Builds the CANONICAL graph adapter (owner override: graph role = FalkorDB, KEPT).

    Instantiated THROUGH this registry seam like every other role's factory — the composition
    root selects ``(graph, "falkordb")`` by config key, never constructs ``FalkorLtmAdapter``
    from a hardcoded import path. The adapter's own tunables (``resolve_entity`` shortlist
    size / similarity band, the per-attempt I/O timeout) are DI-threaded here from the central
    Settings tree (``FalkorDBSettings``), with ``cfg`` (a ``BackendChoice.config`` override)
    taking precedence — see this module's EXTENSION SEAM note above for how a future
    (graph, "neo4j"|"kuzu"|"ladybug"|"neptune") backend registers alongside this one.
    """
    graph_settings = get_settings().storage.graph
    # AD-110 — THIS FACTORY DOES NO I/O, and that is the whole point of the closure below.
    #
    # ``FalkorDB.__init__`` builds a synchronous probe connection (``Is_Cluster`` ->
    # ``redis.Redis(...).info(section="server")``, even on the ``falkordb.asyncio`` class — see
    # ``falkordb/asyncio/cluster.py``). Every composition root builds its stores from inside an
    # ASGI ``lifespan`` coroutine (``mu_server.app.lifespan`` -> ``SharedContainer`` -> this
    # factory), so constructing ``FalkorDB`` HERE ran that blocking probe on the event loop
    # thread. Bounding it by ``socket_timeout`` (the previous fix, kept below) turned an
    # unbounded wedge into a bounded stall; it did not stop it being a stall, and
    # DEV-STANDARDS' async sharpener says *"no blocking/sync I/O in the event loop"* — not
    # *"briefly"*.
    #
    # So the connect is DEFERRED, not moved: ``STORE_REGISTRY.build("graph", "falkordb", ...)``
    # now returns after pure attribute assignment, and ``FalkorLtmAdapter._ensure_db`` runs this
    # closure once, on first use, inside ``asyncio.to_thread``. The timeouts still matter there
    # — they are what bounds the probe's thread — and they still come from the store's own
    # configured budget rather than a new literal (DEV-STANDARDS rule 3).
    store_io_timeout_s = cfg.get("store_io_timeout_s", graph_settings.store_io_timeout_s)
    host = cfg["host"]
    port = cfg["port"]

    def _connect() -> FalkorDB:  # type: ignore[no-any-unimported]  # falkordb ships no stubs
        return FalkorDB(
            host=host,
            port=port,
            socket_timeout=store_io_timeout_s,
            socket_connect_timeout=store_io_timeout_s,
        )

    return FalkorLtmAdapter(
        db_factory=_connect,
        shortlist_size=cfg.get("shortlist_size", graph_settings.entity_shortlist_size),
        similarity_threshold=cfg.get(
            "similarity_threshold", graph_settings.entity_similarity_threshold
        ),
        store_io_timeout_s=store_io_timeout_s,
    )


@STORE_REGISTRY.register("kv", "redis")
def _build_redis(**cfg: Any) -> RedisStmAdapter:
    redis_settings = get_settings().storage.cache
    client = Redis.from_url(cfg["url"], decode_responses=True)
    return RedisStmAdapter(
        client,
        store_io_timeout_s=cfg.get("store_io_timeout_s", redis_settings.store_io_timeout_s),
        # D4 write-time dedup toggle (conformance D-8) — DI-threaded from the mu-engine
        # intelligence-knob root (``EngineSettings.ingest.stm_dedup``, env
        # ``MU_INGEST__STM_DEDUP``), never hardcoded (DEV-STANDARDS rule 3).
        stm_dedup_enabled=cfg.get("stm_dedup_enabled", get_engine_settings().ingest.stm_dedup),
    )


@STORE_REGISTRY.register("kv", "valkey")
def _build_valkey(**cfg: Any) -> ValkeyStmAdapter:
    """The DECIDED KV/STM backend, registered under its OWN key (owner stage 2026-07-27,
    ``storage-pluggable-spec.md §3.2``) — same ``redis-py`` client (wire-identical), falls back
    to the central ``ValkeySettings`` when ``cfg`` doesn't supply a ``url`` (DEV-STANDARDS rule 3).
    """
    valkey_settings = get_settings().storage.valkey
    url = cfg.get("url") or valkey_settings.url
    client = Redis.from_url(url, decode_responses=True)
    return ValkeyStmAdapter(
        client,
        store_io_timeout_s=cfg.get("store_io_timeout_s", valkey_settings.store_io_timeout_s),
        # D4 write-time dedup toggle (conformance D-8) — same knob as ``_build_redis`` (SAME
        # ``EngineSettings.ingest.stm_dedup``; the wire-identical backends share one env knob).
        stm_dedup_enabled=cfg.get("stm_dedup_enabled", get_engine_settings().ingest.stm_dedup),
    )


@STORE_REGISTRY.register("kv", "memory")
def _build_memory_kv(**cfg: Any) -> InMemoryStmAdapter:
    """Embedded, zero-server KV floor (``storage-pluggable-spec.md §1``/§3.2 ``memory``; D2).
    Bounds fall back to the central ``InMemoryKvSettings`` when ``cfg`` doesn't override them.
    """
    mem_settings = get_settings().storage.kv_memory
    return InMemoryStmAdapter(
        max_items_per_namespace=int(
            cfg.get("max_items_per_namespace", mem_settings.max_items_per_namespace)
        ),
        default_ttl_s=cfg.get("default_ttl_s", mem_settings.default_ttl_s),
        # D4 write-time dedup toggle (conformance D-8) — parity with the redis/valkey factories.
        stm_dedup_enabled=cfg.get("stm_dedup_enabled", get_engine_settings().ingest.stm_dedup),
    )


@STORE_REGISTRY.register("kv", "memcached")
def _build_memcached(**cfg: Any) -> MemcachedStmAdapter:
    """Memcached KV backend (``storage-pluggable-spec.md §3.2`` "new | dev only has Memcached"),
    real ``mu-dev-memcached`` container via ``aiomcache``. No native sorted set — the CAS-guarded
    recency-list emulation (degrade D6) is tuned by ``MemcachedSettings`` (``recency_cap``,
    ``cas_max_attempts``, ``default_ttl_s``), overridable per ``cfg`` (DEV-STANDARDS rule 3).
    """
    mc_settings = get_settings().storage.memcached
    host = cfg.get("host", mc_settings.host)
    port = int(cfg.get("port", mc_settings.port))
    client = aiomcache.Client(host, port)
    return MemcachedStmAdapter(
        client,
        recency_cap=int(cfg.get("recency_cap", mc_settings.recency_cap)),
        cas_max_attempts=int(cfg.get("cas_max_attempts", mc_settings.cas_max_attempts)),
        default_ttl_s=int(cfg.get("default_ttl_s", mc_settings.default_ttl_s)),
        store_io_timeout_s=float(cfg.get("store_io_timeout_s", mc_settings.store_io_timeout_s)),
    )


@STORE_REGISTRY.register("vector", "pgvector")
def _build_pgvector(*, dim: int, **cfg: Any) -> PgVectorMtmAdapter:
    """mem0 ``VectorStoreFactory`` pattern (``factory.py:164-205``) — the pgvector alt MTM backend,
    SHARED-safe (``storage-pluggable-spec.md §3.3``: SQL ``WHERE`` + HNSW filter BEFORE ANN
    truncation). Connection knobs fall back to the central ``PgVectorSettings`` when ``cfg`` (a
    ``BackendChoice.config`` override) doesn't supply them (DEV-STANDARDS rule 3).
    """
    pgv_settings = get_settings().storage.pgvector
    dsn = cfg.get("dsn") or pgv_settings.dsn
    return PgVectorMtmAdapter(
        dsn=str(dsn),
        dim=dim,
        hnsw=bool(cfg.get("hnsw", pgv_settings.hnsw)),
        min_size=int(cfg.get("min_size", pgv_settings.pool_min_size)),
        max_size=int(cfg.get("max_size", pgv_settings.pool_max_size)),
        store_io_timeout_s=float(cfg.get("store_io_timeout_s", pgv_settings.store_io_timeout_s)),
    )


@STORE_REGISTRY.register("vector", "chroma")
def _build_chroma(*, dim: int, **cfg: Any) -> ChromaMtmAdapter:
    """Embedded Chroma MTM floor (``storage-pluggable-spec.md §1``: "Vector / MTM ... Chroma
    (embedded, ``path=``) — filterable, single-node"). No server/container; the on-disk path
    falls back to the central ``ChromaSettings`` when ``cfg`` doesn't supply one.
    """
    chroma_settings = get_settings().storage.chroma
    path = cfg.get("path") or chroma_settings.path
    return ChromaMtmAdapter(
        path=str(path),
        dim=dim,
        store_io_timeout_s=float(cfg.get("store_io_timeout_s", chroma_settings.store_io_timeout_s)),
        min_widen=int(cfg.get("min_widen", chroma_settings.min_widen)),
    )


@STORE_REGISTRY.register("artifact", "filesystem")
def _build_artifact_fs(**cfg: Any) -> FsContextRepositoryAdapter:
    """Builds the ``ContextRepository`` / artifact-content adapter (this task's new role —
    software-arch spec §5, l.260-263). NOT one of ``StoreRegistry.MANDATORY_ROLES`` (registry.py)
    — a caller/composition root that never wires ``artifact`` simply never gets
    ``PersistRawArtifactStage`` (the caller opts in by threading ``artifacts=`` into
    ``IngestService``); this factory only governs WHICH backend binds when it does. Falls back to
    the central ``ArtifactFsSettings.content_root`` when ``cfg`` doesn't supply one (DEV-STANDARDS
    rule 3), same pattern as ``_build_chroma``/``_build_faiss`` above.
    """
    artifact_settings = get_settings().storage.artifact
    content_root = cfg.get("content_root") or artifact_settings.content_root
    return FsContextRepositoryAdapter(content_root=str(content_root))


@STORE_REGISTRY.register("vector", "weaviate")
def _build_weaviate(*, dim: int, **cfg: Any) -> WeaviateMtmAdapter:
    """ADR 0050: Weaviate, native multi-tenancy, the SHARED-plane MTM vector backend.

    ``skip_init_checks=True`` is UNCONDITIONAL, not caller-configurable — this adapter never
    issues a single gRPC call (see ``weaviate_mtm.py``'s module docstring for why: verified live,
    every gRPC-backed SDK method hangs against an HTTP-only deployment), so gating connection
    success on a gRPC health probe would refuse a perfectly usable REST-only deployment for a
    capability this adapter does not use. ``grpc_host``/``grpc_port`` are still required
    constructor args of ``use_async_with_custom`` even though never dialed; they default to the
    same host and Weaviate's own conventional gRPC port so a caller need not think about them.

    Every other knob resolves ``cfg`` (a ``BackendChoice.config`` override) FIRST, then the
    central ``WeaviateSettings`` subtree — connection, the I/O budget, and the two recall
    over-fetch tunables — so nothing reachable through the registry is a fixed literal
    (DEV-STANDARDS rule 3; see this module's WEAVIATE FACTORY NOTE).
    """
    wv_settings = get_settings().storage.weaviate
    host = str(cfg.get("host") or wv_settings.host)
    http_port = int(cfg.get("http_port", wv_settings.http_port))
    http_secure = bool(cfg.get("http_secure", wv_settings.http_secure))
    client = weaviate.use_async_with_custom(
        http_host=host,
        http_port=http_port,
        http_secure=http_secure,
        # ``grpc_host=None`` in Settings means "same host as HTTP" — resolved here, not in the
        # adapter, so the never-dialed gRPC args stay a factory concern (see WeaviateSettings).
        grpc_host=str(cfg.get("grpc_host") or wv_settings.grpc_host or host),
        grpc_port=int(cfg.get("grpc_port", wv_settings.grpc_port)),
        grpc_secure=bool(cfg.get("grpc_secure", wv_settings.grpc_secure)),
        skip_init_checks=True,
    )
    return WeaviateMtmAdapter(
        client,
        http_url=f"{'https' if http_secure else 'http'}://{host}:{http_port}",
        dim=dim,
        store_io_timeout_s=float(cfg.get("store_io_timeout_s", wv_settings.store_io_timeout_s)),
        # The two WORD-tokenization over-fetch knobs — DI-threaded like every other adapter
        # tunable (DEV-STANDARDS rule 3). Before this, they were reachable ONLY by constructing
        # the adapter directly, i.e. fixed constants for every caller coming through the
        # registry.
        semantic_overfetch_factor=int(
            cfg.get("semantic_overfetch_factor", wv_settings.semantic_overfetch_factor)
        ),
        semantic_overfetch_max_extra=int(
            cfg.get("semantic_overfetch_max_extra", wv_settings.semantic_overfetch_max_extra)
        ),
    )


@STORE_REGISTRY.register("vector", "faiss")
def _build_faiss(*, dim: int, **cfg: Any) -> FaissMtmAdapter:
    """Brute-force in-proc MTM floor. PRIVATE-plane ONLY (degrade rule D3) — the registry (and this
    adapter, defense-in-depth) refuse it on the SHARED plane
    (``mu_engine.storage.registry._BRUTE_FORCE_VECTOR_BACKENDS``).
    """
    faiss_settings = get_settings().storage.faiss
    path = cfg.get("path") if "path" in cfg else faiss_settings.path
    return FaissMtmAdapter(path=str(path) if path else None, dim=dim)
