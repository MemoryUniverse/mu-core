"""``ConflictPolicyResolver`` — §4.1 policy resolution, most-specific wins.

Authority: ``conflict-resolution-async-design.md`` §4.1 (lines 150-160).

**The precedence, in the order it is evaluated:**

1. **Per-memory override** — if ANY member of the conflict carries a ``conflict_policy_ref``, it
   wins. *"a user may say 'my dietary facts are hand-curated (MANUAL); my scratch notes
   auto-resolve'"* (line 154).
2. **Per-namespace policy** — *"the primary knob the owner asked for"* (line 155).
3. **Workspace/global default** — ``ConflictSettings.default_policy`` (line 156).

**Why the direction is the whole point.** Inverting steps 1 and 2 — letting a namespace default
beat a per-memory override — is a silent defect of exactly the shape this project keeps finding:
nothing errors, nothing logs, the system simply auto-supersedes a fact the user had marked
hand-curated. It is invisible until the fact is gone. So the order is asserted by
``test_a_per_memory_override_beats_the_namespace_default`` and the inversion is mutation-proven
to turn it RED.

**Mixed-mode within one namespace is DELIBERATE, not a leak** (line 160): a namespace can be
AUTOMATIC by default with a handful of MANUAL-pinned sensitive facts. That is what step 1
existing at all means.

**Two members, two overrides — which wins?** The spec says "if ANY member ... carries a
``conflict_policy_ref``, it wins" and stops there. When BOTH members carry one and they DISAGREE,
:meth:`for_conflict` takes the more CONSERVATIVE of the two rather than the first found:
``MANUAL`` beats ``AUTOMATIC``, because the alternative is deciding a user's hand-curated fact by
iteration order over a member tuple. Recorded as a spec gap (line 154 does not rule on it) rather
than left to whichever id sorted first.
"""

from __future__ import annotations

from mu_contracts.domain.model.conflict import ConflictResolutionMode
from mu_contracts.domain.model.memory import Namespace
from mu_engine.lifecycle.conflict import ConflictResolutionPolicy
from mu_engine.services.conflict.ports import (
    MemoryConflictPolicyStore,
    NamespaceConflictPolicyStore,
)
from mu_engine.services.conflict.settings import ConflictSettings

__all__ = ["ConflictPolicyResolver"]


class ConflictPolicyResolver:
    """Resolve the effective :class:`ConflictResolutionPolicy` for one conflict.

    Pure precedence over two injected readers plus a settings default; holds no state, performs
    no writes, and is safe to call concurrently. Both stores are OPTIONAL: an un-wired store is
    simply a step the chain skips, so a FULL-LOCAL deployment with neither configured resolves
    every conflict to the settings default with zero I/O and zero API keys.
    """

    def __init__(
        self,
        *,
        settings: ConflictSettings | None = None,
        namespace_policies: NamespaceConflictPolicyStore | None = None,
        memory_policies: MemoryConflictPolicyStore | None = None,
    ) -> None:
        self._settings = settings or ConflictSettings()
        self._namespace_policies = namespace_policies
        self._memory_policies = memory_policies

    async def for_conflict(
        self, ns: Namespace, member_ids: tuple[str, ...]
    ) -> ConflictResolutionPolicy:
        """The effective policy, resolved most-specific-first.

        Called ONCE per conflict by the detect side, and its result snapshotted onto the
        ``ConflictRecord`` (spec line 158), so a later namespace-policy change governs new
        detections and ``REOPENED`` ones but never retroactively re-resolves a settled conflict.
        """
        override = await self._member_override(ns, member_ids)
        if override is not None:
            return override
        if self._namespace_policies is not None:
            namespace_policy = await self._namespace_policies.policy_for(ns)
            if namespace_policy is not None:
                return namespace_policy
        return self._settings.default_policy

    async def _member_override(
        self, ns: Namespace, member_ids: tuple[str, ...]
    ) -> ConflictResolutionPolicy | None:
        """Step 1. ``None`` when no member carries an override.

        Members are read in a DETERMINISTIC order (sorted by id) so that two replicas resolving
        the same conflict read the same overrides in the same order — the conservative tie-break
        below makes the outcome order-independent anyway, but determinism here means the I/O
        pattern is reproducible too.
        """
        if self._memory_policies is None:
            return None
        chosen: ConflictResolutionPolicy | None = None
        for memory_id in sorted(member_ids):
            candidate = await self._memory_policies.override_for(ns, memory_id)
            if candidate is None:
                continue
            if chosen is None:
                chosen = candidate
                continue
            chosen = _more_conservative(chosen, candidate)
        return chosen


def _more_conservative(
    a: ConflictResolutionPolicy, b: ConflictResolutionPolicy
) -> ConflictResolutionPolicy:
    """The MANUAL one, or ``a`` when both agree on mode.

    "Conservative" here means "the one that does not let the machine decide". Two disagreeing
    per-memory overrides is genuinely under-specified (see the module docstring); resolving it
    toward MANUAL is the only direction that cannot destroy a fact the user hand-curated, and it
    is order-independent, so it holds however the member tuple was built.
    """
    if a.mode is ConflictResolutionMode.MANUAL:
        return a
    if b.mode is ConflictResolutionMode.MANUAL:
        return b
    return a
