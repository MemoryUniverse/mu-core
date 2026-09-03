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
from mu_contracts.ports.model import SparseEncoderPort
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
    :class:`SharedRecallUnavailableError` (never a silent empty).

    AD-222 FOLLOW-UP (verified in this session): every shipped composition root wires this arm's
    `RecallService` counterpart with a `SparseEncoderPort` and threads the resulting `SparseQuery`
    into the PRIVATE ranker call (`RecallService.recall`, "M2 resolution") — but this port's own
    `recall()` called `self._ranker.rank(...)` with no `sparse_query=` at all, defaulting it to
    `None` three frames deep in the SAME `ThreeChannelRecallRanker` the private arm uses. The
    result: with `sparse_enabled=True`, a private-session federated recall got hybrid dense+sparse
    on its private half and silently fell back to dense-only on its shared half — an asymmetry
    that showed up nowhere in AD-222's measurement (the LoCoMo eval harness never has shared data)
    and was not among AD-222's three recorded gaps. `sparse_encoder` mirrors `RecallService.
    __init__`'s own optional param exactly: `None` (the default) makes this byte-identical to the
    pre-fix behaviour, so no committed measurement changes."""

    def __init__(
        self,
        *,
        ranker: RecallRanker,
        embedder: EmbeddingPort,
        sparse_encoder: SparseEncoderPort | None = None,
    ) -> None:
        self._ranker = ranker
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder

    async def recall(
        self, q: RecallQuery, *, caller_identity_set: CallerIdentitySet
    ) -> RecallResult:
        shared_ns = Namespace.shared(
            org=q.namespace.org, workspace=q.namespace.workspace, session=q.namespace.session
        )
        # ACCURACY-PLAN-0831.md item 4: `RecallQuery.limit` became `int | None` (width
        # derivation, `RecallService._effective_limit`) — but that resolution happens ONCE, at
        # `RecallService.recall`'s own top, BEFORE either arm (including this in-process shared
        # one) is ever called; every caller reaching this port therefore hands it an
        # already-concrete `limit`. Guarded rather than assumed: a future caller of this port that
        # skips `RecallService` entirely gets a NAMED, loud failure here, not a `TypeError` three
        # frames down inside the ranker (§5 "re-raise loud, not a silent partial").
        limit = q.limit
        if limit is None:
            raise ValueError(
                "InProcessSharedRecall.recall requires an already-resolved RecallQuery.limit — "
                "RecallService.recall resolves width derivation before calling either arm; a "
                "caller reaching this port directly must resolve `limit` first"
            )
        try:
            vectors = await self._embedder.embed([q.text])
            # SAME "M2 resolution" the private arm's façade applies (`RecallService.recall`):
            # encode the sparse query once, here, at this arm's own boundary — `None` when no
            # encoder is configured, leaving the MTM arm dense-only exactly as before this fix.
            sparse_query = (
                self._sparse_encoder.encode_query(q.text)
                if self._sparse_encoder is not None
                else None
            )
            return await self._ranker.rank(
                shared_ns,
                q.text,
                vectors[0],
                limit=limit,
                channels=q.channels,
                caller_identity_set=caller_identity_set,
                sparse_query=sparse_query,
            )
        except StoreUnavailableError as exc:  # a real shared-plane store failure → NAMED degrade
            raise SharedRecallUnavailableError("shared recall unavailable") from exc
