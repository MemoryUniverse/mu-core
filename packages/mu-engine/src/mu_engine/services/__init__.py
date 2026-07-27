"""services/ — the engine application use-cases + baseline strategies.

SCAFFOLD ONLY. IngestService, RecallService, PromotionService, DemotionService,
ConflictService, ComposeService; MemoryRepository facade + TierRouter; and the OPEN
baseline strategies (WeightedSalienceStrategy, ThresholdPromotionStrategy,
EbbinghausDemotionStrategy, BiTemporalConflictResolver, ThreeChannelRecallRanker —
PACKAGING-v2 §2.1). Ported per CODE-ADOPTION-METHODOLOGY (mem0 diff loop, Graphiti
bi-temporal, MemoryBank decay).
"""

__all__: list[str] = []
