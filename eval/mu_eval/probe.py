"""Score-provenance probe — the ``0.0000`` investigation, reduced to one runnable command.

The only recall transcript in this repo (``demos/mu-live-demo/out/h3-mu-recall.txt``) prints
``0.0000`` as the score of every hit. A scorer that returns zero for everything is either a
DISPLAY bug (the ranking is fine, the wrong number reaches the surface) or a RANKING bug (the
ordering really is score-free), and which one it is changes everything downstream. Nothing in the
repo distinguishes them, so this probe does, by reading BOTH sides of the same recall:

  * the RAW channel scores the store adapters return (``MtmTierRepository.semantic`` cosine,
    ``LtmTierRepository.graph_recall`` reciprocal rank, ``StmTierRepository.recent``), and
  * the ``fused_score`` the same items carry when they arrive at the caller-facing
    ``RecallItemView``,

for the identical query, in one process, against the real stores. It also recomputes the RRF
fusion over the raw channels so the three numbers can be laid side by side.
"""

from __future__ import annotations

import uuid
from typing import Any

__all__ = ["probe_scores"]

_TURNS = [
    "Ada lives in Paris and commutes by bicycle.",
    "Ada works at Acme as a staff engineer.",
    "Bo adopted a rescue dog named Pepper last spring.",
    "The deploy passphrase for the ZEPHYR release is violet-anchor-77.",
    "Melanie is training for a half marathon in October.",
    "Caroline went to an LGBTQ support group on 7 May.",
    "The team standup moved to 09:30 on Tuesdays.",
    "Bo prefers oat milk in coffee, never soy.",
]
_QUERY = "What is the deploy passphrase for the ZEPHYR release?"


async def probe_scores(*, settings: object | None = None, query: str = _QUERY) -> dict[str, Any]:
    """Ingest a tiny known corpus, then read the same recall at three depths. Returns a record."""
    from mu_contracts.domain.model.scope import ClientScope
    from mu_engine.services.recall.dto import RecallQuery
    from mu_engine.services.recall.fusion import reciprocal_rank_fusion
    from mu_engine.storage.domain.namespace import Namespace, Visibility
    from mu_eval.corpus import _teardown
    from mu_local import LocalMemory
    from mu_local.composition import LocalContainer
    from mu_local.config import StorageSettings

    tag = f"probe{uuid.uuid4().hex[:8]}"
    user, session = "probeuser", "probesession"
    record: dict[str, Any] = {"tag": tag, "query": query, "corpus": len(_TURNS)}

    memory = LocalMemory(workspace=f"ws{tag}", namespace=f"org{tag}", settings=settings)
    try:
        for turn in _TURNS:
            await memory.add(turn, user=user, session=session, importance_score=0.9)
        surface = await memory.recall(query, user=user, session=session, limit=10)
        record["surface_items"] = [
            {
                "tier": str(item.tier),
                "channel": item.channel,
                "is_floor": item.is_floor,
                "fused_score": item.fused_score,
                "rerank_score": item.rerank_score,
                "content": item.content[:60],
            }
            for item in surface.items
        ]
        record["surface_degraded"] = None if surface.degraded is None else str(surface.degraded)

        # The SAME corpus read from a DIFFERENT session. This is the condition the repo's only
        # recall transcript was produced under (demos/mu-live-demo/run_hook_demo.sh recalls with
        # no session after the capture ran under Claude's own generated session id), and it is
        # the one that prints `mtm/mtm` rows: the STM window is session-scoped, so it is EMPTY
        # here and the MTM dense arm is the only channel with hits.
        cross = await memory.recall(query, user=user, session="a-different-session", limit=10)
        record["cross_session_items"] = [
            {
                "tier": str(item.tier),
                "channel": item.channel,
                "is_floor": item.is_floor,
                "fused_score": item.fused_score,
                "content": item.content[:60],
            }
            for item in cross.items
        ]
    finally:
        await memory.aclose()

    # --- the same query, one layer down: raw channel scores + the RRF number itself -------------
    container = LocalContainer(StorageSettings(), settings=settings)  # type: ignore[arg-type]
    try:
        ns = Namespace(
            org=f"org{tag}",
            workspace=f"ws{tag}",
            user=user,
            session=session,
            visibility=Visibility.PRIVATE,
        )
        vector = (await container.embedder.embed([query]))[0]
        stm = await container.stm.recent(ns, limit=10, caller_identity_set=None)
        mtm = await container.mtm.semantic(ns, vector, limit=20, caller_identity_set=None)
        ltm = await container.ltm.graph_recall(ns, limit=20, caller_identity_set=None)
        record["raw_channels"] = {
            "stm": [{"score": s.score, "content": s.item.content[:60]} for s in stm],
            "mtm": [{"score": s.score, "content": s.item.content[:60]} for s in mtm],
            "ltm": [{"score": s.score, "content": s.item.content[:60]} for s in ltm],
        }
        fused = reciprocal_rank_fusion(
            [stm, mtm, ltm],
            key=lambda s: s.item.id,
            weights=[1.0, 1.0, 1.0],
            k=60,
        )
        record["rrf_scores"] = [
            {"rrf": score, "content": scored.item.content[:60]} for scored, score in fused[:10]
        ]

        # And what the SERVICE returns, so the surface value can be attributed to a layer.
        scope = ClientScope(
            principal_id=user,
            org_id=ns.org,
            workspace_id=ns.workspace,
            session_id=session,
            agent_principal_id=user,
        )
        engine_result = await container.recall.recall(
            scope, RecallQuery(namespace=ns, text=query, limit=10)
        )
        record["engine_items"] = [
            {
                "channel": item.channel,
                "fused_score": item.fused_score,
                "is_floor": item.is_floor,
                "content": item.content[:60],
            }
            for item in engine_result.items
        ]
    finally:
        await _teardown(tag, settings)
        await container.close()

    return record
