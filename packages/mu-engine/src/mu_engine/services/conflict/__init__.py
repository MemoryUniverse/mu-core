"""``services/conflict`` — the ASYNCHRONOUS, off-write-path conflict lane.

Authority: ``conflict-resolution-async-design.md`` (§4 policy, §4.1 precedence, §4.2 strategies,
§5 manual surface, §6 recall semantics, §7 cross-device convergence) · CANONICAL §7.17 item 4a
(the total order) · §7.20.

Two things live under this package and they are deliberately different concerns:

* ``order`` — every replica re-deriving the SAME winner over an already-replicated delta set,
  with no coordination. ONE pure function pair (``total_order_key`` for deltas,
  ``total_order_key_items`` for live items) sharing one seven-term chain.
* everything else — one replica DECIDING, and a human OVERRIDING: the policy precedence chain,
  the deterministic pickers, the resolve service, the inbox projector, the recall semantics, and
  the sticky-manual convergence.

``ConflictAdjudicator`` (the LLM-judged single-replica verdict, ADR 0037) stays in
``mu_engine.lifecycle.conflict``, next to the ``ConflictLifecyclePolicy`` FSM it drives.

**Nothing in this package writes memory state.** The resolve service holds no writer, the recall
policy holds no repository, and the strategies are pure functions. Resolution is applied by
``ResolveConflictStage`` on the background worker under the §7.5 writer lease — keeping the
"writer never waits" invariant structural rather than a promise.
"""

from mu_engine.services.conflict.convergence import (
    ConvergenceOutcome,
    converge_pair,
    manual_reinstate_delta,
    manual_supersede_delta,
)
from mu_engine.services.conflict.inbox import ConflictInboxProjector
from mu_engine.services.conflict.order import total_order_key, total_order_key_items
from mu_engine.services.conflict.policy_resolver import ConflictPolicyResolver
from mu_engine.services.conflict.ports import (
    ConflictMemberHydration,
    ConflictMemberHydrator,
    InMemoryConflictResolutionQueue,
    InMemoryMemoryConflictPolicyStore,
    InMemoryNamespaceConflictPolicyStore,
    MemoryConflictPolicyStore,
    NamespaceConflictPolicyStore,
    RecordBackedResolutionQueue,
    ResolutionIntent,
    ResolutionQueue,
    UnappliedConflictRecordReader,
    WritableMemoryConflictPolicyStore,
)
from mu_engine.services.conflict.recall import (
    PendingConflictRecallPolicy,
    RecallConflictOutcome,
)
from mu_engine.services.conflict.resolution import (
    ConflictResolutionService,
    ManualDecision,
    ManualDecisionKind,
)
from mu_engine.services.conflict.settings import ConflictSettings
from mu_engine.services.conflict.strategies import (
    SOURCE_TRUST_RANK,
    AutoResolution,
    AutoWinnerPicker,
    ConfidencePicker,
    ProvenancePicker,
    RecencyPicker,
    picker_for,
    recommended_resolution_kind,
    resolve_automatically,
)

__all__ = [
    "SOURCE_TRUST_RANK",
    "AutoResolution",
    "AutoWinnerPicker",
    "ConfidencePicker",
    "ConflictInboxProjector",
    "ConflictMemberHydration",
    "ConflictMemberHydrator",
    "ConflictPolicyResolver",
    "ConflictResolutionService",
    "ConflictSettings",
    "ConvergenceOutcome",
    "InMemoryConflictResolutionQueue",
    "InMemoryMemoryConflictPolicyStore",
    "InMemoryNamespaceConflictPolicyStore",
    "ManualDecision",
    "ManualDecisionKind",
    "MemoryConflictPolicyStore",
    "NamespaceConflictPolicyStore",
    "PendingConflictRecallPolicy",
    "ProvenancePicker",
    "RecallConflictOutcome",
    "RecencyPicker",
    "RecordBackedResolutionQueue",
    "ResolutionIntent",
    "ResolutionQueue",
    "UnappliedConflictRecordReader",
    "WritableMemoryConflictPolicyStore",
    "converge_pair",
    "manual_reinstate_delta",
    "manual_supersede_delta",
    "picker_for",
    "recommended_resolution_kind",
    "resolve_automatically",
    "total_order_key",
    "total_order_key_items",
]
