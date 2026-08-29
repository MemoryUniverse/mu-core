"""Zero-vector probe — WHICH write path puts a memory into Qdrant with no embedding.

Reading the code says there are two ways a memory reaches the MTM vector tier and only one of
them embeds:

  * ``DeterministicPromoteStage._execute`` (ingest-time, fires when
    ``importance >= IngestSettings.importance_promote``) calls
    ``self._embedder.embed([activity.atomic_fact_text()])`` and puts the vector on the promoted
    copy — ``pipelines/concrete/ingest.py:437-450``;
  * ``LifecyclePromotionService._promote_to_mtm`` (the maintenance sweep, and the targeted
    ``promote`` verb) does ``item.model_copy(deep=True)`` of the **STM** item and upserts it —
    ``lifecycle/promotion.py:479-490``. The STM item never carried an embedding
    (``_build_memory_item`` in ingest.py sets none), and nothing adds one here.

and that the mapper turns a missing embedding into a ZERO VECTOR rather than refusing:

    vector = item.embedding if item.embedding is not None else [0.0] * self.dim
    -- storage/mappers/qdrant_mapper.py:89

Cosine similarity against a zero vector is 0 for every query, so such a point scores exactly
0.0000 forever and its position inside the MTM channel is arbitrary. This probe demonstrates it
end to end on the real stores rather than asserting it from a read: two identical bodies, one
promoted by each path, then the stored vectors and the recall scores are printed side by side.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

__all__ = ["probe_promotion_paths"]

_BODY_INGEST = "The ingest-promoted fact: the ZEPHYR release passphrase is violet-anchor-77."
_BODY_LIFECYCLE = "The lifecycle-promoted fact: the ZEPHYR release owner is Dilan."
_QUERY = "What is the ZEPHYR release passphrase and who owns it?"


async def probe_promotion_paths(*, settings: object | None = None) -> dict[str, Any]:
    from qdrant_client import AsyncQdrantClient

    from mu_contracts.config import Settings
    from mu_engine.storage.domain.namespace import Namespace, Visibility
    from mu_engine.storage.mappers.qdrant_mapper import collection_name, point_id
    from mu_eval.corpus import _teardown
    from mu_local import LocalMemory

    tag = f"probe{uuid.uuid4().hex[:8]}"
    user, session = "probeuser", "probesession"
    record: dict[str, Any] = {"tag": tag, "query": _QUERY}

    memory = LocalMemory(workspace=f"ws{tag}", namespace=f"org{tag}", settings=settings)
    try:
        # (a) ingest-time promotion — importance over the gate, so DeterministicPromoteStage fires.
        ingest_side = await memory.add(
            _BODY_INGEST, user=user, session=session, importance_score=0.9
        )
        # (b) lifecycle promotion — importance UNDER the gate so nothing promotes at ingest, then
        #     the targeted `promote` verb drives LifecyclePromotionService's copy-on-write.
        lifecycle_side = await memory.add(
            _BODY_LIFECYCLE, user=user, session=session, importance_score=0.1
        )
        record["ingest_promoted_at_write"] = bool(getattr(ingest_side, "promoted", False))
        record["lifecycle_promoted_at_write"] = bool(getattr(lifecycle_side, "promoted", False))
        verb = await memory.promote(
            lifecycle_side.memory_id, to_tier="mtm", user=user, session=session
        )
        record["promote_verb"] = {
            "memory_id": verb.memory_id,
            "from_tier": verb.from_tier,
            "to_tier": verb.to_tier,
        }
        await asyncio.sleep(1.0)  # qdrant applies upserts asynchronously

        # Read the two points' STORED vectors straight out of Qdrant.
        cfg = settings if isinstance(settings, Settings) else Settings()
        ns = Namespace(
            org=f"org{tag}",
            workspace=f"ws{tag}",
            user=user,
            session=session,
            visibility=Visibility.PRIVATE,
        )
        client = AsyncQdrantClient(url=cfg.storage.vector.url)
        try:
            name = collection_name(ns, 384)
            points = await client.retrieve(
                collection_name=name,
                ids=[point_id(ingest_side.memory_id), point_id(lifecycle_side.memory_id)],
                with_vectors=True,
                with_payload=True,
            )
            record["stored_points"] = [
                {
                    "content": str(p.payload.get("content", ""))[:60] if p.payload else "",
                    "vector_dim": len(p.vector or []) if isinstance(p.vector, list) else None,
                    "vector_nonzero": (
                        sum(1 for v in p.vector if v != 0.0) if isinstance(p.vector, list) else None
                    ),
                }
                for p in points
            ]
        finally:
            await client.close()

        # And what recall reports for each — cross-session so the MTM arm is the only live channel.
        result = await memory.recall(_QUERY, user=user, session="fresh-session", limit=10)
        record["recall_items"] = [
            {
                "tier": str(item.tier),
                "channel": item.channel,
                "fused_score": item.fused_score,
                "content": item.content[:60],
            }
            for item in result.items
        ]
    finally:
        await _teardown(tag, settings)
        await memory.aclose()
    return record
