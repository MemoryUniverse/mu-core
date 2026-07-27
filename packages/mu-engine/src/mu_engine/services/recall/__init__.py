"""``services/recall`` — the consolidated federate-live RECALL read path (recall-service-design.md).

One ``RecallService.recall`` federates private-own ⊕ authorized-shared in a single call: embed the
query once via the REAL local ``EmbeddingPort``, run the 3-channel ranker (STM recency floor / MTM
qdrant dense with Model-A ``authorized_ids`` + ``state='active'`` / LTM falkordb bi-temporal graph)
on the PRIVATE arm, run the authorized ``SharedRecallPort`` on the SHARED arm, RRF-fuse the two,
``content_hash``-dedup, protect the recency floor, and return a ``RecallResult``. Fully async, pure
read (emits nothing on the bus), NAMED degrades only. NO LLM (ANSWER/INJECT deferred).
"""

from __future__ import annotations

from mu_engine.services.recall.authz import (
    AuthorizedIdsResolver,
    PrincipalAuthorizedIdsResolver,
    RecallAuthorizationFilter,
)
from mu_engine.services.recall.dto import (
    RecallChannels,
    RecallItemView,
    RecallMode,
    RecallQuery,
    RecallResult,
    RecallSettings,
)
from mu_engine.services.recall.fusion import (
    FusionStrategy,
    ReciprocalRankFusion,
    dedup_by_content_hash,
    reciprocal_rank_fusion,
)
from mu_engine.services.recall.ranker import RecallRanker, ThreeChannelRecallRanker
from mu_engine.services.recall.service import RecallService
from mu_engine.services.recall.shared_port import (
    InProcessSharedRecall,
    SharedRecallPort,
    SharedRecallUnavailableError,
)

__all__ = [
    "AuthorizedIdsResolver",
    "FusionStrategy",
    "InProcessSharedRecall",
    "PrincipalAuthorizedIdsResolver",
    "RecallAuthorizationFilter",
    "RecallChannels",
    "RecallItemView",
    "RecallMode",
    "RecallQuery",
    "RecallRanker",
    "RecallResult",
    "RecallService",
    "RecallSettings",
    "ReciprocalRankFusion",
    "SharedRecallPort",
    "SharedRecallUnavailableError",
    "ThreeChannelRecallRanker",
    "dedup_by_content_hash",
    "reciprocal_rank_fusion",
]
