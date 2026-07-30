"""RecallService federate-live — REAL mu-dev-qdrant + mu-dev-falkordb + mu-dev-cache, REAL MiniLM
embedder, ZERO mocks (DEV-STANDARDS: non-negotiable).

The acceptance the task names: store items across the PRIVATE own-partition and the authorized
SHARED partition, then ``recall(query)`` returns the RIGHT FUSED set with NAMESPACE ISOLATION on
live stores —
  * private-own ⊕ authorized-shared are fused into ONE ranked result;
  * an UNAUTHORIZED shared item (Model-A ``authorized_ids`` mismatch) never surfaces;
  * a SUPERSEDED item (``state != active``) never surfaces;
  * a DIFFERENT-tenant item (different session ``to_prefix()`` partition) never surfaces;
  * a duplicate body present in BOTH planes collapses to ONE via ``content_hash`` dedup;
  * an STM recency-floor member is present and flagged ``is_floor``.

Plus the NAMED shared-arm degrade (``SHARED_RECALL_UNAVAILABLE``): the private arm is returned
intact and LABELLED, never a silent private-only drop (recall-service-design §1.6).
"""
# mypy: disable-error-code="no-any-unimported, arg-type"
# ^ (1) ``no-any-unimported`` — ``falkordb`` ships no stubs, so the FalkorDB param type is ``Any``
#   under ``disallow_any_unimported`` (the adapter itself carries the same ``# type: ignore`` on
#   its ctor). (2) ``arg-type`` — this test wires an ``object``-typed local fake into
#   ``RecallService`` (``shared_recall=``). NOTE: the former ``@retry_io`` decorator-typing gap this
#   suppression ALSO used to cover is RESOLVED — ``retry_io`` now returns
#   ``Callable[P, Coroutine[Any, Any, R]]``, so a decorated adapter method structurally satisfies
#   its async ``Protocol`` with no suppression at any composition site (mu-local ``composition.py``
#   no longer needs its file-level disable).

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.domain.events import DegradeReason
from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.providers.embedding import SentenceTransformerEmbedder
from mu_engine.services.recall import (
    InProcessSharedRecall,
    PrincipalAuthorizedIdsResolver,
    RecallAuthorizationFilter,
    RecallQuery,
    RecallResult,
    RecallService,
    RecallSettings,
    ReciprocalRankFusion,
    SharedRecallUnavailableError,
    ThreeChannelRecallRanker,
)
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.integration

ItemFactory = Callable[..., Awaitable[MemoryItem]]


@pytest_asyncio.fixture
async def falkor_db(settings) -> AsyncIterator[FalkorDB]:  # type: ignore[no-untyped-def]
    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    await db.select_graph("_probe").query("RETURN 1")  # fail-loud if the container is down
    yield db


@pytest.fixture
def make_item(embedder: SentenceTransformerEmbedder) -> ItemFactory:
    """Build a MemoryItem carrying a REAL MiniLM embedding of its content."""

    async def _make(
        ns: Namespace,
        content: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        state: MemoryState = MemoryState.ACTIVE,
        tier: MemoryTier = MemoryTier.MTM,
        authorized_ids: list[str] | None = None,
        valid_at: datetime | None = None,
    ) -> MemoryItem:
        vec = (await embedder.embed([content]))[0]
        meta = {"authorized_ids": authorized_ids} if authorized_ids is not None else {}
        return MemoryItem(
            content=content,
            kind=MemoryKind.PROPOSITION,
            namespace=ns,
            state=state,
            tier=tier,
            owner_id=ns.user if ns.visibility is Visibility.PRIVATE else "owner1",
            workspace_id=ns.workspace,
            session_id=ns.session,
            subject=subject,
            predicate=predicate,
            object=obj,
            embedding=vec,
            embedding_model=embedder.model_name,
            metadata=meta,
            valid_at=valid_at,
        )

    return _make


def _build_service(
    *,
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
    shared_recall: object | None = None,
) -> RecallService:
    """Wire a RecallService over the REAL stores. Both arms share the same adapters — the arm
    difference is the η partition + the caller identity set (CANONICAL §7.9 "one fuse impl")."""
    stm = RedisStmAdapter(redis_client)
    mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
    ltm = FalkorLtmAdapter(falkor_db)
    fusion = ReciprocalRankFusion()
    settings = RecallSettings()
    clock = SystemClock()
    ranker = ThreeChannelRecallRanker(
        stm=stm, mtm=mtm, ltm=ltm, fusion=fusion, settings=settings, clock=clock
    )
    shared = shared_recall or InProcessSharedRecall(ranker=ranker, embedder=embedder)
    authz = RecallAuthorizationFilter(
        tenancy=DefaultTenancyGuard(), authorized_ids=PrincipalAuthorizedIdsResolver()
    )
    return RecallService(
        embedder=embedder,
        private_ranker=ranker,
        shared_recall=shared,
        authz=authz,
        fusion=fusion,
        settings=settings,
        clock=clock,
    )


async def _teardown(
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    redis_client: Redis,
    *,
    workspace: str,
    dim: int,
    user: str,
) -> None:
    for visibility in (Visibility.PRIVATE, Visibility.SHARED):
        coll = f"mu_mtm__{workspace}__{visibility.value}__{dim}"
        if await qdrant_client.collection_exists(coll):
            await qdrant_client.delete_collection(coll)
    for graph in (f"mu_g__{workspace}__{user}", f"mu_g__{workspace}__shared"):
        with contextlib.suppress(Exception):  # best-effort drop; a missing graph raises
            await falkor_db.select_graph(graph).delete()
    keys = [k async for k in redis_client.scan_iter(match=f"mu/mu/*{workspace}*".encode())]
    if keys:
        await redis_client.delete(*keys)


async def test_federate_live_recall_fuses_and_isolates(
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
    uid: str,
    make_item: ItemFactory,
) -> None:
    org, ws, session = f"org{uid}", f"ws{uid}", "s1"
    priv = Namespace(
        org=org, workspace=ws, user="u1", session=session, visibility=Visibility.PRIVATE
    )
    shared = Namespace.shared(org=org, workspace=ws, session=session)
    other = Namespace(  # a DIFFERENT user ⇒ a genuinely different tenant.
        # NOTE (S1-04 / BQ3 / ADR 0030 integrate-phase fix): this used to be user="u1",
        # session="other" — a different SESSION of the SAME user. That is no longer a tenant
        # boundary: qdrant_mtm's PRIVATE-own federated-recall path (session_scope=None, the
        # new default) intentionally matches on the truncated, session-less user prefix, so a
        # same-user/different-session item is SUPPOSED to surface now (that federation is
        # exhaustively covered by test_qdrant_mtm_session_scope_int.py). Genuine tenant
        # isolation is keyed on `user` (and org/workspace), so this fixture now differs by
        # user to keep testing the real isolation invariant instead of an assumption Stage-1
        # intentionally overturned.
        org=org, workspace=ws, user="u2", session=session, visibility=Visibility.PRIVATE
    )
    caller = ClientScope(
        principal_id="u1", org_id=org, workspace_id=ws, session_id=session, agent_principal_id="u1"
    )
    now = datetime.now(UTC)

    stm = RedisStmAdapter(redis_client)
    mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
    ltm = FalkorLtmAdapter(falkor_db)
    service = _build_service(
        redis_client=redis_client,
        qdrant_client=qdrant_client,
        falkor_db=falkor_db,
        embedder=embedder,
    )

    dup_text = "The onboarding runbook lives in the shared drive"
    try:
        # --- PRIVATE arm data (own partition, no authorized_ids gate) ---
        p_color = await make_item(priv, "My favorite color is teal")
        p_pet = await make_item(priv, "I have a dog named Rex")
        p_fact = await make_item(
            priv,
            "Paris is the capital of France",
            subject="Paris",
            predicate="capital_of",
            obj="France",
            valid_at=now,
            tier=MemoryTier.LTM,
        )
        p_old = await make_item(  # SUPERSEDED — must be filtered by state='active'
            priv, "My favorite color is orange", state=MemoryState.SUPERSEDED
        )
        p_dup = await make_item(priv, dup_text)  # a pulled shared→local copy
        await mtm.upsert(p_color)
        await stm.put(p_color)  # also in the STM recency floor
        await mtm.upsert(p_pet)
        await ltm.upsert_fact(p_fact)
        await mtm.upsert(p_old)
        await mtm.upsert(p_dup)

        # --- SHARED arm data (Model-A authorized_ids stamped per point) ---
        s_team = await make_item(shared, "The team standup is at 9am", authorized_ids=["u1"])
        s_secret = await make_item(  # NOT authorized to u1 — must never surface
            shared, "Confidential acquisition target is Acme", authorized_ids=["someone-else"]
        )
        s_dup = await make_item(shared, dup_text, authorized_ids=["u1"])  # live shared origin
        await mtm.upsert(s_team)
        await mtm.upsert(s_secret)
        await mtm.upsert(s_dup)

        # --- other-tenant data (different user) — must never surface ---
        o_item = await make_item(other, "The favorite color over there is crimson")
        await mtm.upsert(o_item)

        q = RecallQuery(
            namespace=priv,
            text="what is my favorite color and when is the team standup",
            limit=10,
        )
        result: RecallResult = await service.recall(caller, q)

        ids = set(result.memory_ids)
        contents = [it.content for it in result.items]

        # fused: authorized private + authorized shared both present in ONE result
        assert p_color.id in ids, "private own MTM hit missing"
        assert s_team.id in ids, "authorized shared hit missing from the fused result"
        # isolation — Model-A authorized_ids: an unauthorized shared item never surfaces
        assert s_secret.id not in ids, "UNAUTHORIZED shared item leaked (Model-A breach)"
        # isolation — state='active': a superseded item never surfaces
        assert p_old.id not in ids, "superseded item resurfaced (state filter breach)"
        # isolation — physical to_prefix partition: a different session never surfaces
        assert o_item.id not in ids, "cross-tenant item leaked (tenancy breach)"
        # dedup — the same body in both planes collapses to exactly one ranked item
        assert contents.count(dup_text) == 1, "content_hash dedup failed (double-count)"
        # recency floor — the STM member is present and flagged
        floor_ids = {it.memory_id for it in result.items if it.is_floor}
        assert p_color.id in floor_ids, "STM recency-floor member not protected/flagged"
        # LTM bi-temporal fact fused in from the graph arm
        assert p_fact.id in ids, "valid LTM graph fact missing from the fused result"
        # a clean, non-degraded federated read across all three channels
        assert result.degraded is None
        assert result.channels_run.mtm and result.channels_run.ltm
        # every returned η belongs to the caller's own or the authorized shared partition
        assert {it.namespace.visibility for it in result.items} <= {
            Visibility.PRIVATE,
            Visibility.SHARED,
        }
    finally:
        await _teardown(
            qdrant_client, falkor_db, redis_client, workspace=ws, dim=embedder.dimension, user="u1"
        )


async def test_shared_arm_unavailable_is_named_degrade_not_silent_drop(
    redis_client: Redis,
    qdrant_client: AsyncQdrantClient,
    falkor_db: FalkorDB,
    embedder: SentenceTransformerEmbedder,
    uid: str,
    make_item: ItemFactory,
) -> None:
    """The shared plane being unreachable is a NAMED degrade: the private arm is returned intact and
    labelled ``SHARED_RECALL_UNAVAILABLE`` — never a silent private-only union (§1.6/§5.2)."""

    class _UnavailableSharedRecall:
        """The faithful 'shared plane unreachable' adapter — raises the named error, exactly as the
        REST/in-process adapter does when the shared source cannot be served (not a mock of a
        working dependency; it IS the failure contract under test)."""

        async def recall(
            self, q: RecallQuery, *, caller_identity_set: CallerIdentitySet
        ) -> RecallResult:
            raise SharedRecallUnavailableError("shared plane down")

    org, ws, session = f"org{uid}", f"ws{uid}", "s1"
    priv = Namespace(
        org=org, workspace=ws, user="u1", session=session, visibility=Visibility.PRIVATE
    )
    caller = ClientScope(
        principal_id="u1", org_id=org, workspace_id=ws, session_id=session, agent_principal_id="u1"
    )
    mtm = QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)
    service = _build_service(
        redis_client=redis_client,
        qdrant_client=qdrant_client,
        falkor_db=falkor_db,
        embedder=embedder,
        shared_recall=_UnavailableSharedRecall(),
    )
    try:
        p_item = await make_item(priv, "My recovery email is set in the private vault")
        await mtm.upsert(p_item)

        result = await service.recall(
            caller, RecallQuery(namespace=priv, text="what is my recovery email", limit=10)
        )

        assert result.degraded is DegradeReason.SHARED_RECALL_UNAVAILABLE, "degrade not named"
        assert p_item.id in set(result.memory_ids), "private arm dropped on shared-arm failure"
    finally:
        await _teardown(
            qdrant_client, falkor_db, redis_client, workspace=ws, dim=embedder.dimension, user="u1"
        )
