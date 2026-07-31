"""lifecycle/ — the Memory Lifecycle Manager (MLM) subtree (mu_engine).

Stage-0 (S0-01..S0-10) contracts + engine-foundation slices, PLUS Stage-1 (S1-01/S1-02/S1-03) —
the real promotion/demotion/orchestrator services — have landed and are re-exported below with an
explicit ``__all__`` (mypy ``no_implicit_reexport``), mirroring ``mu_engine.services``' pattern:

- ``mode_gate.py``   (S0-03) — ``ManagerMode``/``ManagerModeGate``/``ManagerOwnsLifecycleError``/
                      ``ModePolicyResolver``.
- ``settings.py``    (S0-07) — the ``LifecycleSettings`` central-config tree + its sub-trees.
- ``salience.py``    (S0-08) — ``SalienceStrategy``.
- ``dto.py``         (S0-09) — ``TransitionKind``/``SalienceInputs``/``ModelVerdict``/
                      ``LifecycleStateView``.
- ``explain.py``     (S0-09) — ``ExplainRecord``.
- ``mtm_graph.py``   (S0-10) — ``MtmWorkingGraphSettings``/``MtmWorkingGraphService`` (RESERVED
                      seam, disabled). ``MtmWorkingGraphSettings`` is defined ONCE here (this
                      module) and re-exported by ``settings.py`` — not redefined there (DRY,
                      integrate-phase fix: S0-07 and S0-10 independently drafted the class).
- ``promotion.py``   (S1-01) — ``PromotionService``/``PromotionReport``/``PromotionOutcome``/
                      ``PromotionOutcomeKind``.
- ``demotion.py``    (S1-02) — ``DemotionService``/``DemotionReport``/``DemotionOutcome``/
                      ``MtmRemovalPort``.
- ``manager.py``     (S1-03) — ``MemoryLifecycleManager`` (the orchestrator) + its local seam
                      Protocols (``RetentionServicePort``/``ConflictAdjudicatorPort``/
                      ``WarmRecallCacheServicePort``) and ``RenderedContext``.
- ``retention.py``   (S2-01) — ``RetentionService`` (validity-first LTM archival + GC + COLD
                      slide, ADR 0035) + its local seam Protocol ``LtmRetentionStorePort`` and
                      DTOs ``RetentionOutcome``/``RetentionReport``. Satisfies ``manager.py``'s
                      ``RetentionServicePort`` seam structurally (no import of ``manager.py``).
- ``conflict.py``    (S3-01) — ``ConflictAdjudicator`` (LLM-judged supersession, ADR 0037) +
                      its policy/settings/verdict/port types and ``build_conflict_adjudicator``.
                      Landed at Stage-3 integrate; re-exported here like every other slice.
- ``conflict_redis.py`` (SDK build-plan §4 Stage-C C1) — ``RedisConflictRecordRepository`` +
                      ``ConflictRedisSettings``, the durable Valkey/Redis
                      ``ConflictRecordRepository`` adapter (the SHARED/persistent counterpart to
                      ``conflict.py``'s ``InMemoryConflictRecordRepository``).

Stage-3 integrate (this same pass) also retired the ``MtmRemovalPort`` back-compat alias
``demotion.py`` used to export (``mu_engine.lifecycle.demotion.MtmRemovalPort`` was always just
``MtmTierRepository`` by another name, CF-2/MLM-STAGE2-CARRYOVER.md) — every caller now passes a
real ``MtmTierRepository``-conforming instance (e.g. ``QdrantMtmAdapter``) directly, so the alias
had no remaining importer and is gone, not merely deprecated.
"""

from mu_engine.lifecycle.conflict import (
    AdjudicationBudget,
    AdjudicationKind,
    AdjudicationVerdict,
    ConflictAdjudicationPort,
    ConflictAdjudicator,
    ConflictAdjudicatorSettings,
    ConflictEventPublisher,
    ConflictResolutionPolicy,
    InMemoryConflictRecordRepository,
    ResolutionMode,
    ResolutionStrategy,
    build_conflict_adjudicator,
    compute_conflict_id,
)
from mu_engine.lifecycle.conflict_redis import ConflictRedisSettings, RedisConflictRecordRepository
from mu_engine.lifecycle.demotion import (
    DemotionOutcome,
    DemotionReport,
    DemotionService,
)
from mu_engine.lifecycle.dto import (
    LifecycleStateView,
    ModelVerdict,
    SalienceInputs,
    TransitionKind,
)
from mu_engine.lifecycle.explain import ExplainRecord
from mu_engine.lifecycle.manager import (
    ConflictAdjudicatorPort,
    MemoryLifecycleManager,
    RenderedContext,
    RetentionServicePort,
    WarmRecallCacheServicePort,
)
from mu_engine.lifecycle.mode_gate import (
    ManagerMode,
    ManagerModeGate,
    ManagerOwnsLifecycleError,
    ModePolicyResolver,
)
from mu_engine.lifecycle.mtm_graph import MtmWorkingGraphService, MtmWorkingGraphSettings
from mu_engine.lifecycle.promotion import (
    PromotionOutcome,
    PromotionOutcomeKind,
    PromotionReport,
    PromotionService,
)
from mu_engine.lifecycle.retention import (
    LtmRetentionStorePort,
    RetentionOutcome,
    RetentionReport,
    RetentionService,
)
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import (
    HostedMirrorConsent,
    LifecycleSettings,
    ManagerModeSettings,
    OwnershipSettings,
    RetentionSettings,
    SalienceSettings,
)

__all__ = [
    "AdjudicationBudget",
    "AdjudicationKind",
    "AdjudicationVerdict",
    "ConflictAdjudicationPort",
    "ConflictAdjudicator",
    "ConflictAdjudicatorPort",
    "ConflictAdjudicatorSettings",
    "ConflictEventPublisher",
    "ConflictRedisSettings",
    "ConflictResolutionPolicy",
    "DemotionOutcome",
    "DemotionReport",
    "DemotionService",
    "ExplainRecord",
    "HostedMirrorConsent",
    "InMemoryConflictRecordRepository",
    "LifecycleSettings",
    "LifecycleStateView",
    "LtmRetentionStorePort",
    "ManagerMode",
    "ManagerModeGate",
    "ManagerModeSettings",
    "ManagerOwnsLifecycleError",
    "MemoryLifecycleManager",
    "ModePolicyResolver",
    "ModelVerdict",
    "MtmWorkingGraphService",
    "MtmWorkingGraphSettings",
    "OwnershipSettings",
    "PromotionOutcome",
    "PromotionOutcomeKind",
    "PromotionReport",
    "PromotionService",
    "RedisConflictRecordRepository",
    "RenderedContext",
    "ResolutionMode",
    "ResolutionStrategy",
    "RetentionOutcome",
    "RetentionReport",
    "RetentionService",
    "RetentionServicePort",
    "RetentionSettings",
    "SalienceInputs",
    "SalienceSettings",
    "SalienceStrategy",
    "TransitionKind",
    "WarmRecallCacheServicePort",
    "build_conflict_adjudicator",
    "compute_conflict_id",
]
