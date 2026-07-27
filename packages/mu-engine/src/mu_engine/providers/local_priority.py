"""L4 — Local-model PRIORITY predicate (model-layer-spec §2.4).

We own ONE decision: *is this task allowed to prefer a local model?* The EXECUTION (trying
local first, failing over to remote) is delegated entirely to LiteLLM's `order:` field — local
deployments compile to `order: 1`, remote to `order: 2`, and the Router does the ordered
failover (research §1.6 layer 2). We write NO custom router (research §1.3 anti-duplication).
"""

from __future__ import annotations

from mu_engine.providers.catalog import Task

__all__ = ["LOCAL_ORDER", "REMOTE_ORDER", "LocalPriorityPolicy"]

# The two `order:` tiers LiteLLM honours via `enable_pre_call_checks=True` (model-layer-spec §2.4).
LOCAL_ORDER = 1
REMOTE_ORDER = 2


class LocalPriorityPolicy:
    """Owns the 'is this task local-capable' predicate. Ordering is DONE BY LiteLLM via `order:`."""

    def __init__(self, *, local_capable_tasks: frozenset[Task], enabled: bool) -> None:
        self._local_tasks = local_capable_tasks
        self._enabled = enabled

    def prefers_local(self, task: Task) -> bool:
        """True iff local priority is on AND this task may be served locally (§2.4)."""
        return self._enabled and task in self._local_tasks

    def order_for(self, *, is_local: bool, task: Task) -> int:
        """The `order:` a deployment gets: local deployments of a local-capable task → tier 1;
        everything else → tier 2. This is the whole mechanism — the failover is the Router's."""
        if is_local and self.prefers_local(task):
            return LOCAL_ORDER
        return REMOTE_ORDER
