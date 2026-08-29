"""Integration-test fixtures — REAL mu-dev-* containers, ZERO mocks (DEV-STANDARDS).

Every store client here connects to a real running container whose host port comes from the
central Settings tree (``.env.test`` -> ``get_settings()``), never a hardcoded literal.
Tests isolate themselves with a unique ``org``/``workspace``/``session`` per test so no
cross-test contamination, and tear down the collections/graphs/keys they create.

If a container is not up, the fixture NEVER fakes it (DEV-STANDARDS: "if a real dependency
isn't up, the test is BLOCKED (**reported**), never faked"). The reported half is the part that
was missing: raising the driver's own connection error makes a machine-state fact arrive as a
~40-line setup ERROR that reads exactly like a broken tier. ``mysql_engine`` below therefore
translates *connection* failure — and only connection failure — into a one-line named report that
says which container, which endpoint, and the command that starts it, with
``MU_REQUIRE_MYSQL=1`` to turn that report into a hard FAILURE where the container was supposed
to be up. Every other fixture here still raises the raw error; same latent noise, different
owner's item.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Callable

import aiomcache
import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from mu_contracts.config import Settings
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name
from mu_engine.storage.relational.schema import Base

VECTOR_DIM = 8  # tiny deterministic vectors — no ML dep needed to test store/filter/recall


@pytest.fixture(scope="session")
def settings() -> Settings:
    # single env-boundary read; .env.test wires the mu-dev-* host ports.
    return Settings()


@pytest.fixture
def uid() -> str:
    return uuid.uuid4().hex[:12]


class NamespaceFactory:
    """A per-test-unique namespace factory (isolation by construction) that RECORDS every
    ``Namespace`` it produces on ``.created`` — so a teardown fixture can compute the EXACT
    physical partitions a test actually touched from the real objects, instead of reconstructing a
    guess from ``uid`` + assumed defaults (a guess drifts silently the moment a test overrides
    ``workspace``; see ``qdrant_teardown_collections``). Structurally satisfies
    ``Callable[..., Namespace]``, so every existing call site typed that way is unaffected."""

    def __init__(self, uid: str) -> None:
        self._uid = uid
        self.created: list[Namespace] = []

    def __call__(
        self,
        *,
        visibility: Visibility = Visibility.PRIVATE,
        user: str = "u1",
        session: str = "s1",
        workspace: str | None = None,
    ) -> Namespace:
        ws = workspace or f"ws{self._uid}"
        if visibility is Visibility.SHARED:
            ns = Namespace.shared(org=f"org{self._uid}", workspace=ws, session=session)
        else:
            ns = Namespace(
                org=f"org{self._uid}",
                workspace=ws,
                user=user,
                session=session,
                visibility=Visibility.PRIVATE,
            )
        self.created.append(ns)
        return ns


@pytest.fixture
def make_ns(uid: str) -> NamespaceFactory:
    return NamespaceFactory(uid)


@pytest.fixture
def make_item() -> Callable[..., MemoryItem]:
    """A MemoryItem factory with a deterministic tiny embedding derived from the content."""

    def _make(
        ns: Namespace,
        content: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        polarity: Polarity = Polarity.POSITIVE,
        authorized_ids: list[str] | None = None,
        memory_id: str | None = None,
    ) -> MemoryItem:
        seed = sum(ord(c) for c in content)
        embedding = [((seed + i) % 17) / 17.0 for i in range(VECTOR_DIM)]
        meta = {"authorized_ids": authorized_ids} if authorized_ids is not None else {}
        kwargs = {}
        if memory_id is not None:
            kwargs["id"] = memory_id
        return MemoryItem(
            content=content,
            kind=MemoryKind.PROPOSITION,
            namespace=ns,
            owner_id=ns.user if ns.visibility is Visibility.PRIVATE else "owner1",
            workspace_id=ns.workspace,
            session_id=ns.session,
            subject=subject,
            predicate=predicate,
            object=obj,
            polarity=polarity,
            embedding=embedding,
            embedding_model="test-fixture",
            metadata=meta,
            **kwargs,
        )

    return _make


@pytest.fixture
def qdrant_teardown_collections(make_ns: NamespaceFactory) -> Callable[[], list[str]]:
    """The Qdrant collection names this test ACTUALLY occupied — computed through the REAL mapper
    from every ``Namespace`` ``make_ns`` produced during the test (``make_ns.created``), never
    reconstructed from ``uid`` + assumed defaults.

    ``qdrant_mapper.collection_name`` HASHES ``org``+``workspace`` together (see that function's
    docstring), so ``uid`` no longer appears as a literal substring inside the collection name the
    way it did when the name was a raw ``__``-joined string, and a
    ``name.startswith("mu_mtm__org")`` sweep can no longer find them. The PREVIOUS fix for that
    reconstructed exactly two names (PRIVATE + SHARED) from ``org=f"org{uid}"``/
    ``workspace=f"ws{uid}"`` — hardcoded assumptions that silently stop matching the moment a test
    calls ``make_ns(workspace=...)`` with an override, or creates only one visibility, or creates
    more than two namespaces: the recomputed names would miss the real collection with no error,
    exactly the kind of silent teardown gap that let 283 orphaned collections accumulate before
    this session's cleanup. Reading the actual ``.created`` list instead cannot drift out of sync
    with what the test really made, because it IS what the test really made — not a second,
    independent guess at it.
    """

    def _names() -> list[str]:
        # dedup (multiple namespaces can legitimately hash to the same collection, e.g. two
        # PRIVATE namespaces differing only by user/session) without losing any distinct target.
        return list({collection_name(ns, VECTOR_DIM) for ns in make_ns.created})

    return _names


@pytest_asyncio.fixture
async def redis_client(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=True)
    await client.ping()  # fail-loud if the container is down
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def qdrant_client(settings: Settings) -> AsyncIterator[AsyncQdrantClient]:
    client = AsyncQdrantClient(url=settings.storage.vector.url)
    await client.get_collections()  # fail-loud
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def falkor_db(settings: Settings) -> AsyncIterator[FalkorDB]:
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    # fail-loud probe
    await db.select_graph("_probe").query("RETURN 1")
    yield db


@pytest_asyncio.fixture
async def pg_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(settings.storage.postgres.dsn, pool_pre_ping=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)  # idempotent (checkfirst)
    try:
        yield engine
    finally:
        await engine.dispose()


#: Turns the ``mu-dev-mysql``-is-absent SKIP below into a hard FAILURE. Set it wherever the
#: container is *supposed* to be up — a CI lane, or a VM run whose provisioning claims to have
#: started it — so "the tier did not run" cannot be mistaken for "the tier passed". Named for
#: the store, not for a lane, so a future ``MU_REQUIRE_<STORE>`` reads the same way.
MYSQL_REQUIRED_ENV = "MU_REQUIRE_MYSQL"


def _mysql_absent(settings: Settings, exc: BaseException) -> str:
    """The one message a missing ``mu-dev-mysql`` gets, whether it skips or fails.

    It names the store, the endpoint it looked at, the driver's own words, and the command that
    fixes it — everything the reader needs without opening a file. The DSN is NOT interpolated:
    it carries the dev password, and a skip line is printed into logs and CI summaries.

    The last sentence differs by mode on purpose. Telling a reader who ALREADY set
    ``MU_REQUIRE_MYSQL=1`` to set it is advice they have taken, printed on the run where it fired
    — the reader would reasonably conclude the flag did nothing. In that mode the sentence states
    which flag turned this into a hard stop instead.
    """
    where = f"{settings.storage.mysql.host}:{settings.storage.mysql.port}"
    cause = getattr(exc, "orig", exc)
    verdict = (
        f"{MYSQL_REQUIRED_ENV}=1 is set, so this is a FAILURE and not a skip."
        if os.environ.get(MYSQL_REQUIRED_ENV)
        else f"Set {MYSQL_REQUIRED_ENV}=1 to make this a FAILURE instead of a skip."
    )
    return (
        f"mu-dev-mysql is not reachable at {where} ({type(cause).__name__}: {cause}). "
        f"Start it with `docker compose -f docker-compose.dev.yml up -d mysql` "
        f"(host port from MU_STORAGE__MYSQL__* in .env.test). {verdict}"
    )


@pytest_asyncio.fixture
async def mysql_engine(settings: Settings) -> AsyncIterator[AsyncEngine]:
    """The REAL ``mu-dev-mysql`` relational engine — and a NAMED report when it is not there.

    **Why the probe is separate from ``create_all``.** This fixture used to open the engine and
    run the DDL in one breath, so a container that was simply not started surfaced as four
    setup ERRORs, each ~40 lines of SQLAlchemy pool internals with
    ``Can't connect to MySQL server on 'localhost' ([Errno 111] ...)`` as the last line. That is
    not the "BLOCKED (reported)" DEV-STANDARDS asks for — it is indistinguishable, at a glance,
    from the tier being broken, and it is the same shape as the 41 push-tier tests that once
    vanished into a green headline: a machine-state fact wearing the costume of a code fact.

    So connectivity is established FIRST, on its own, and only that failure is translated. A
    ``create_all`` that fails once MySQL *is* answering is a real dialect/DDL defect (this schema
    binds Postgres, SQLite and MySQL — see ``relational/schema.py``) and is deliberately left to
    raise untouched: it must never be swallowed by an environment excuse.
    """
    engine = create_async_engine(settings.storage.mysql.dsn, pool_pre_ping=True)
    try:
        absent: str | None = None
        try:
            async with engine.connect() as conn:
                await conn.execute(text("select 1"))
        except (OperationalError, OSError) as exc:
            # Reached only when the SERVER never answered: refused/unreachable/unauthenticated.
            # A live server that rejects the *schema* raises a different class and is not caught.
            absent = _mysql_absent(settings, exc)
        # Raised OUTSIDE the ``except`` on purpose. Raising in the handler makes Python chain the
        # driver exception onto it, and pytest then prints the whole
        # "The above exception was the direct cause of ..." tower ABOVE the message — measured:
        # the require-flag run reported the named sentence buried under 3 chained frames per test,
        # i.e. exactly the noise this fixture exists to remove. Here the sentence is the report.
        if absent is not None:
            if os.environ.get(MYSQL_REQUIRED_ENV):
                pytest.fail(absent, pytrace=False)
            pytest.skip(absent)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)  # idempotent (checkfirst)
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def valkey_client(settings: Settings) -> AsyncIterator[Redis]:
    client: Redis = Redis.from_url(settings.storage.valkey.url, decode_responses=True)
    await client.ping()  # fail-loud if the container is down
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def memcached_client(settings: Settings) -> AsyncIterator[aiomcache.Client]:
    client = aiomcache.Client(settings.storage.memcached.host, settings.storage.memcached.port)
    await client.version()  # fail-loud if the container is down
    try:
        yield client
    finally:
        await client.close()
