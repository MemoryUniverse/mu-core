"""services/ — the engine application use-cases + baseline strategies.

SCAFFOLD ONLY. IngestService, RecallService, PromotionService, DemotionService,
ConflictService, ComposeService; and the OPEN
baseline strategies (WeightedSalienceStrategy, ThresholdPromotionStrategy,
EbbinghausDemotionStrategy, BiTemporalConflictResolver, ThreeChannelRecallRanker —
PACKAGING-v2 §2.1). Ported per CODE-ADOPTION-METHODOLOGY (mem0 diff loop, Graphiti
bi-temporal, MemoryBank decay).

SHIPPED (this slice): the INGEST use-case — ``IngestService.remember(activity)`` + its result
DTO + the deterministic-promotion ``IngestSettings``.

SHIPPED (memory-repository slice): the ``MemoryRepository`` façade + ``TierRouter``, in
``mu_engine.services.memory`` — no longer scaffold. It is what makes ``MemoryHealthService`` and
``PinService`` constructible: both take ``repo: MemoryRepository`` as a REQUIRED keyword, and
until this landed the Protocol had no implementer anywhere, so mu-client's ``/health`` /
``/pin`` / ``/unpin`` IPC routes returned a named 503 and its MCP tools raised
``ServiceNotWiredError`` in every configuration. ``mu-client/src/mu_client/memory_health.py``
lines 29-38 cite THIS docstring's old wording as the authority for that degraded behaviour and
now need updating in turn (reported — mu-client is a separate repo).
"""

from mu_engine.services.extract import (
    ExtractedFact,
    FactExtractorPort,
    HeuristicSpoExtractor,
    LlmFactExtractor,
    build_extractor,
    decompose_to_spo,
)
from mu_engine.services.ingest import IngestResult, IngestService
from mu_engine.services.settings import IngestSettings

__all__ = [
    "ExtractedFact",
    "FactExtractorPort",
    "HeuristicSpoExtractor",
    "IngestResult",
    "IngestService",
    "IngestSettings",
    "LlmFactExtractor",
    "build_extractor",
    "decompose_to_spo",
]
