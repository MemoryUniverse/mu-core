"""services/ — the engine application use-cases + baseline strategies.

SCAFFOLD ONLY. IngestService, RecallService, PromotionService, DemotionService,
ConflictService, ComposeService; MemoryRepository facade + TierRouter; and the OPEN
baseline strategies (WeightedSalienceStrategy, ThresholdPromotionStrategy,
EbbinghausDemotionStrategy, BiTemporalConflictResolver, ThreeChannelRecallRanker —
PACKAGING-v2 §2.1). Ported per CODE-ADOPTION-METHODOLOGY (mem0 diff loop, Graphiti
bi-temporal, MemoryBank decay).

SHIPPED (this slice): the INGEST use-case — ``IngestService.remember(activity)`` + its result
DTO + the deterministic-promotion ``IngestSettings``.
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
