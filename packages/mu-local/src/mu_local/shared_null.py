"""``LocalNullSharedRecall`` — the LOCAL-mode shared arm: there is NO shared plane.

``RecallService`` federates a private-own arm ⊕ a shared arm in one call (CANONICAL §7.9). In
SERVER mode the shared arm is the hosted moat (grants/rooms/authorized_ids). In LOCAL mode the
shared plane does not exist (spec §1/§3.2, T10: mu-local exposes NO cross-user primitive), so this
adapter returns an EMPTY result — it reads no SHARED partition and constructs no grant.

This is NOT a mock of a working dependency: it IS the LOCAL-mode shared-arm contract (there is
nothing to reach). It satisfies the :class:`SharedRecallPort` structural protocol so the ONE
``RecallService`` runs unchanged in-process, and the fusion collapses to the private arm alone.
"""

from __future__ import annotations

from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_contracts.ports.time import Clock
from mu_engine.services.recall.dto import RecallChannels, RecallQuery, RecallResult

__all__ = ["LocalNullSharedRecall"]


class LocalNullSharedRecall:
    """Empty shared arm for the single-tenant embed (SharedRecallPort, spec §1/§3.2)."""

    def __init__(self, *, clock: Clock) -> None:
        self._clock = clock

    async def recall(
        self, q: RecallQuery, *, caller_identity_set: CallerIdentitySet
    ) -> RecallResult:
        """Return an empty, non-degraded result: LOCAL mode has no shared plane to query."""
        del caller_identity_set  # no shared plane ⇒ the caller identity set authorizes nothing
        return RecallResult(
            namespace=q.namespace,
            items=[],
            channels_run=RecallChannels(stm=False, mtm=False, ltm=False),
            degraded=None,
            generated_at=self._clock.now(),
        )
