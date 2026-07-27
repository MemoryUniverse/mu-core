"""pipelines/ — the staged pipeline framework + engine concrete stages.

SCAFFOLD ONLY. INGEST / DISTILL / LIFECYCLE / PROMOTE / DEMOTE stages, the generic
Temporal runner, sleeptime/promotion engine workflows, the stage registry. Fully async
with structured concurrency + bounded queues/backpressure (DEV-STANDARDS async correctness).

SHIPPED (this slice): the ``base`` framework (Stage/BaseStage/StageOutcome/PipelineContext/
Pipeline), the ``StageLedger`` idempotency substrate (in-memory + Redis), and the concrete
CAPTURE->INGEST stages.
"""

from mu_engine.pipelines.base import (
    BaseStage,
    HaltPolicy,
    Pipeline,
    PipelineContext,
    Stage,
    StageOutcome,
    StageStatus,
)
from mu_engine.pipelines.distill import (
    DistillAction,
    DistillActionKind,
    DistillPipeline,
    DistillReport,
    DistillSettings,
    InProcessWriterLease,
    WriterLeasePort,
)
from mu_engine.pipelines.ledger import (
    InMemoryStageLedger,
    RedisStageLedger,
    StageLedger,
    StageLedgerRecord,
)

__all__ = [
    "BaseStage",
    "DistillAction",
    "DistillActionKind",
    "DistillPipeline",
    "DistillReport",
    "DistillSettings",
    "HaltPolicy",
    "InMemoryStageLedger",
    "InProcessWriterLease",
    "Pipeline",
    "PipelineContext",
    "RedisStageLedger",
    "Stage",
    "StageLedger",
    "StageLedgerRecord",
    "StageOutcome",
    "StageStatus",
    "WriterLeasePort",
]
