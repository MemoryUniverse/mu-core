"""The two-arm federate-live experiment: does fusing private ⊕ shared beat either arm alone?

``RecallService.recall`` fuses a PRIVATE arm and a SHARED arm with RRF and reports one list
(``services/recall/service.py:119-127``). Whether that fuse actually HELPS has never been
measured. It is a claim with a cheap decisive test: split one conversation's turns across the two
planes so that neither plane alone holds all the gold evidence, then score three configurations
over the identical query set —

    private-only   channels run against the PRIVATE η only (shared arm stubbed empty)
    shared-only    channels run against the SHARED η only (private arm empty)
    fused          the shipped ``RecallService`` with both arms live

If the fuse is working, ``fused`` recall@k must dominate both single arms on queries whose gold
evidence straddles the split, because each arm can only ever return its own half.

WHY the plane split is done with a direct ``MtmTierRepository.upsert`` rather than a write verb:
``LocalMemory`` is private-plane-only BY CONSTRUCTION — it raises ``PlaneFieldRejectedError`` on a
non-None ``visibility`` (``local_memory.py`` ``validate_plane_fields``), because the shared write
path lives in ``mu-server``. mu-core nonetheless OWNS the shared READ arm
(``InProcessSharedRecall``), which is exactly what is under test here. So the shared corpus is
placed with the same primitive the server's write path ultimately calls, carrying the Model-A
``authorized_ids`` a governed room would stamp (``qdrant_mapper.py:85-88``), and the READ path
exercised is the shipped one, unmodified.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from mu_eval.corpus import TurnIndex
from mu_eval.locomo import Conversation, Turn
from mu_eval.metrics import QueryScores, aggregate, score_query

__all__ = ["ArmComparison", "run_two_arm_fuse"]


class ArmComparison(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    ks: tuple[int, ...]
    corpus_private: int
    corpus_shared: int
    queries_scored: int
    straddling_queries: int
    private_only: dict[str, dict[int, float]]
    shared_only: dict[str, dict[int, float]]
    fused: dict[str, dict[int, float]]
    fused_beats_best_single_arm: dict[int, bool]


class _EmptyRanker:
    """A private arm that is present, healthy and empty — the shared-only control.

    Satisfies the ``RecallRanker`` structural protocol (``key`` + ``rank``), so
    ``RecallService`` runs completely unmodified: the isolation is achieved by giving one arm
    nothing to return, never by editing the service under test.
    """

    key = "eval_empty_v1"

    def __init__(self, clock: Any) -> None:
        self._clock = clock

    async def rank(
        self,
        ns: Any,
        query: str,
        query_vec: Any,
        *,
        limit: int,
        channels: Any,
        caller_identity_set: Any,
    ) -> Any:
        from mu_engine.services.recall.dto import RecallResult

        return RecallResult(
            namespace=ns,
            items=[],
            channels_run=channels,
            degraded=None,
            generated_at=self._clock.now(),
        )


class _EmptySharedRecall:
    """A shared arm that is present, healthy and empty — the private-only control.

    Deliberately NOT ``LocalNullSharedRecall``: that is the shipped LOCAL-mode stub and using it
    would make this control a test of a different code path. This one has the same shape and
    returns an empty result for the SHARED η, which is what "the room holds nothing" looks like.
    """

    def __init__(self, clock: Any) -> None:
        self._clock = clock

    async def recall(self, q: Any, *, caller_identity_set: frozenset[str]) -> Any:
        from mu_engine.services.recall.dto import RecallResult
        from mu_engine.storage.domain.namespace import Namespace

        shared_ns = Namespace.shared(
            org=q.namespace.org, workspace=q.namespace.workspace, session=q.namespace.session
        )
        return RecallResult(
            namespace=shared_ns,
            items=[],
            channels_run=q.channels,
            degraded=None,
            generated_at=self._clock.now(),
        )


@contextlib.asynccontextmanager
async def _container(tag: str, settings: object | None) -> AsyncIterator[Any]:
    from mu_local.composition import LocalContainer
    from mu_local.config import StorageSettings

    container = LocalContainer(StorageSettings(), settings=settings)  # type: ignore[arg-type]
    try:
        yield container
    finally:
        from mu_eval.corpus import _teardown

        with contextlib.suppress(Exception):
            await _teardown(tag, settings)
        await container.close()


async def _put_shared(container: Any, *, ns: Any, body: str, principal: str) -> None:
    """Place one turn on the SHARED plane, Model-A stamped, in the vector tier."""
    from mu_engine.storage.domain.memory import MemoryItem, MemoryTier

    vector = (await container.embedder.embed([body]))[0]
    item = MemoryItem(
        content=body,
        tier=MemoryTier.MTM,
        namespace=ns,
        owner_id=principal,
        workspace_id=ns.workspace,
        session_id=ns.session,
        embedding=vector,
        metadata={"authorized_ids": [principal]},
    )
    await container.mtm.upsert(item)


async def run_two_arm_fuse(
    conversation: Conversation,
    *,
    run_id: str,
    ks: Sequence[int] = (1, 3, 5, 10),
    recall_limit: int | None = None,
    importance: float = 0.9,
    max_queries: int | None = None,
    settings: object | None = None,
) -> ArmComparison:
    from mu_engine.platform.tenancy import DefaultTenancyGuard
    from mu_engine.services.recall.authz import (
        PrincipalAuthorizedIdsResolver,
        RecallAuthorizationFilter,
    )
    from mu_engine.services.recall.dto import RecallQuery
    from mu_engine.services.recall.service import RecallService
    from mu_engine.services.recall.shared_port import InProcessSharedRecall
    from mu_engine.storage.domain.namespace import (
        Namespace,
        Visibility,
    )

    ks = tuple(sorted(ks))
    limit = recall_limit or max(ks)
    tag = f"{run_id}{uuid.uuid4().hex[:6]}"
    principal = "evaluser"
    session = "evalsession"

    async with _container(tag, settings) as container:
        private_ns = Namespace(
            org=f"org{tag}",
            workspace=f"ws{tag}",
            user=principal,
            session=session,
            visibility=Visibility.PRIVATE,
        )
        shared_ns = Namespace.shared(
            org=private_ns.org, workspace=private_ns.workspace, session=session
        )

        # --- corpus split: even-indexed turns PRIVATE, odd-indexed turns SHARED ---------------
        index = TurnIndex()
        private_turns: list[Turn] = []
        shared_turns: list[Turn] = []
        for i, turn in enumerate(conversation.turns):
            index.add(turn)
            (private_turns if i % 2 == 0 else shared_turns).append(turn)

        # Private half goes through the SAME write verb the product ships.
        from mu_local import LocalMemory

        memory = LocalMemory(
            workspace=private_ns.workspace, namespace=private_ns.org, settings=settings
        )
        try:
            for turn in private_turns:
                await memory.add(
                    turn.ingest_text,
                    user=principal,
                    session=session,
                    importance_score=importance,
                )
        finally:
            await memory.aclose()

        for turn in shared_turns:
            await _put_shared(container, ns=shared_ns, body=turn.ingest_text, principal=principal)

        # --- three recall configurations over the identical query set ------------------------
        recall_settings = container._engine_settings.recall
        authz = RecallAuthorizationFilter(
            tenancy=DefaultTenancyGuard(), authorized_ids=PrincipalAuthorizedIdsResolver()
        )
        from mu_engine.services.recall.fusion import ReciprocalRankFusion
        from mu_engine.services.recall.ranker import ThreeChannelRecallRanker

        fusion = ReciprocalRankFusion()
        base_ranker = ThreeChannelRecallRanker(
            stm=container.stm,
            mtm=container.mtm,
            ltm=container.ltm,
            fusion=fusion,
            settings=recall_settings,
            clock=container._clock,
            embedder=container.embedder,
        )

        def service(private_ranker: Any, shared: Any) -> Any:
            return RecallService(
                embedder=container.embedder,
                private_ranker=private_ranker,
                shared_recall=shared,
                authz=authz,
                fusion=fusion,
                settings=recall_settings,
                clock=container._clock,
            )

        live_shared = InProcessSharedRecall(ranker=base_ranker, embedder=container.embedder)
        empty_shared = _EmptySharedRecall(container._clock)
        empty_private = _EmptyRanker(container._clock)

        # Each arm is isolated by EMPTYING the other one, never by post-filtering the fused list:
        # a post-filter would let the arm being suppressed still consume `limit` slots, so
        # "shared-only" would silently be measuring "shared, minus whatever private crowded out"
        # and the comparison against `fused` would be rigged in the fuse's favour.
        svc_fused = service(base_ranker, live_shared)
        svc_private_only = service(base_ranker, empty_shared)
        svc_shared_only = service(empty_private, live_shared)

        from mu_contracts.domain.model.scope import ClientScope

        scope = ClientScope(
            principal_id=principal,
            org_id=private_ns.org,
            workspace_id=private_ns.workspace,
            session_id=session,
            agent_principal_id=principal,
        )

        private_ids = {t.dia_id for t in private_turns}
        shared_ids = {t.dia_id for t in shared_turns}
        known = private_ids | shared_ids

        rows_private: list[QueryScores] = []
        rows_shared: list[QueryScores] = []
        rows_fused: list[QueryScores] = []
        straddling = 0

        queries = [q for q in conversation.queries if not q.is_adversarial and q.evidence]
        if max_queries is not None:
            queries = queries[:max_queries]

        for query in queries:
            gold = {e for e in query.evidence if e in known}
            if not gold:
                continue
            if gold & private_ids and gold & shared_ids:
                straddling += 1
            q = RecallQuery(namespace=private_ns, text=query.question, limit=limit)

            for svc, rows in (
                (svc_private_only, rows_private),
                (svc_shared_only, rows_shared),
                (svc_fused, rows_fused),
            ):
                result = await svc.recall(scope, q)
                ranked = _rank_ids(result.items, index, gold)
                rows.append(
                    score_query(
                        query_id=query.query_id,
                        category=query.category,
                        retrieved=ranked,
                        gold=sorted(gold),
                        ks=ks,
                    )
                )

        agg_private = aggregate(rows_private, ks)
        agg_shared = aggregate(rows_shared, ks)
        agg_fused = aggregate(rows_fused, ks)
        return ArmComparison(
            sample_id=conversation.sample_id,
            ks=ks,
            corpus_private=len(private_turns),
            corpus_shared=len(shared_turns),
            queries_scored=len(rows_fused),
            straddling_queries=straddling,
            private_only=agg_private,
            shared_only=agg_shared,
            fused=agg_fused,
            fused_beats_best_single_arm={
                k: agg_fused["recall"][k]
                > max(agg_private["recall"][k], agg_shared["recall"][k]) + 1e-9
                for k in ks
            },
        )


def _rank_ids(items: Sequence[Any], index: TurnIndex, gold: set[str]) -> list[str]:
    ranked: list[str] = []
    for position, item in enumerate(items):
        candidates = index.resolve(item.content)
        hit = next((c for c in candidates if c in gold), None)
        ranked.append(hit or (candidates[0] if candidates else f"__unknown__#{position}"))
    return ranked
