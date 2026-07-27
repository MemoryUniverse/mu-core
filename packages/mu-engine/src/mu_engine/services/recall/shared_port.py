"""``SharedRecallPort`` — the SHARED arm of federate-live recall (recall-service-design.md §1.6).

A recall on a PRIVATE session federates private-own ⊕ authorized-shared in ONE call (CANONICAL
§7.9, X-B1). The shared arm is this port: an authorized recall over the caller's SHARED partitions
with Model-A ``MatchAny(authorized_ids, caller_identity_set)`` + ``state='active'`` applied AT THE
SOURCE — never a Python post-filter, never a cross-plane store handle opened by the private side.

**One port, two adapters, zero recall-side branching (§4.2a, CANONICAL §7.9/X11):**
  * ``LocalContainer`` — the shared arm rides the daemon→server authorized REST channel (NEVER a
    daemon subscription to the SHARED bus). That REST adapter lands in ``mu-client``.
  * ``SharedContainer`` — the SAME Protocol bound to the in-process shared implementation
    (:class:`InProcessSharedRecall`): no REST hop, identical signature, identical authz.

A shared-arm failure is a NAMED degrade (``SHARED_RECALL_UNAVAILABLE``), never a silent private-only
drop — signalled by raising :class:`SharedRecallUnavailableError`, which the service catches and
maps to ``RecallResult.with_degrade`` (§1.6). The reason does NOT change with the execution site.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_contracts.domain.errors import BackendUnavailableError, StoreUnavailableError
from mu_contracts.domain.model.recall import CallerIdentitySet
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.services.recall.dto import RecallQuery, RecallResult
from mu_engine.services.recall.ranker import RecallRanker
from mu_engine.storage.domain.namespace import Namespace

__all__ = ["InProcessSharedRecall", "SharedRecallPort", "SharedRecallUnavailableError"]


class SharedRecallUnavailableError(BackendUnavailableError):
    """The shared arm could not be served (§1.6). The service maps this to the NAMED
    ``DegradeReason.SHARED_RECALL_UNAVAILABLE`` and returns the private arm intact — the caller is
    told the shared half is missing, never silently served a partial union."""


@runtime_checkable
class SharedRecallPort(Protocol):
    """The narrow shared-arm view (§1.6, rooms §4.2), threaded with the caller identity set so the
    SERVER applies Model-A authz at the source. ``q`` carries the SHARED η to read."""

    async def recall(
        self, q: RecallQuery, *, caller_identity_set: CallerIdentitySet
    ) -> RecallResult: ...


class InProcessSharedRecall:
    """The ``SharedContainer`` adapter: the shared arm runs in-process over the SHARED-plane tier
    repos (§4.2a). Embeds the query with the SAME EmbeddingPort seam the private arm uses (§6-P5;
    the remote REST adapter embeds server-side — this one embeds in-process, warm-once), then ranks
    the SHARED partition with the caller identity set. Store failures surface as the NAMED
    :class:`SharedRecallUnavailableError` (never a silent empty)."""

    def __init__(self, *, ranker: RecallRanker, embedder: EmbeddingPort) -> None:
        self._ranker = ranker
        self._embedder = embedder

    async def recall(
        self, q: RecallQuery, *, caller_identity_set: CallerIdentitySet
    ) -> RecallResult:
        shared_ns = Namespace.shared(
            org=q.namespace.org, workspace=q.namespace.workspace, session=q.namespace.session
        )
        try:
            vectors = await self._embedder.embed([q.text])
            return await self._ranker.rank(
                shared_ns,
                q.text,
                vectors[0],
                limit=q.limit,
                channels=q.channels,
                caller_identity_set=caller_identity_set,
            )
        except StoreUnavailableError as exc:  # a real shared-plane store failure → NAMED degrade
            raise SharedRecallUnavailableError("shared recall unavailable") from exc
