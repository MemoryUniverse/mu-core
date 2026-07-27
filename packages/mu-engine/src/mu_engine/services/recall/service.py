"""``RecallService`` — the consolidated CQRS read side (recall-service-design.md §1.2 + §1.6).

App-singleton, stateless, side-effect-free. ``recall`` is federate-live (CANONICAL §7.9, X-B1): a
PRIVATE-session recall fans out to TWO arms in a SINGLE call and fuses them ONCE —

    private-own (own to_prefix partition, authorized_ids=None)
      ⊕  authorized-shared (SharedRecallPort, Model-A MatchAny at the source)
      →  RRF fuse  →  content_hash dedup  →  recency-floor protect  →  RecallResult

with NO ahead-of-time pull/subscribe/materialisation in front of it (X-M3: the automation gap
closes by construction). The query is embedded ONCE at this façade boundary via the
``EmbeddingPort`` seam (§6-P2/m4) — the tier repos never see a raw string. NO LLM on this path:
ANSWER/INJECT modes (synthesis / additionalContext render) are DEFERRED (Azure PARKED; ranked real).

The ONE mutation any read here can trigger is the idempotent read-stat write-back the stores
already do (never a tier transition, §5). Degrades are NAMED, never silent:
  * shared arm fails → ``SHARED_RECALL_UNAVAILABLE`` (private arm returned intact + labelled);
  * LTM store down in an arm → ``LTM_UNAVAILABLE`` (MTM+floor returned + labelled).
Both increment the content-free operator metric; the user signal is ``RecallResult.degraded``.
"""

from __future__ import annotations

from collections.abc import Mapping

from mu_contracts.domain.events import DegradeReason
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.observability import MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.platform.observability import NoopMetricSink, NoopTracer, sanitize_label_value
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.services.recall.authz import RecallAuthorizationFilter
from mu_engine.services.recall.dto import (
    RecallChannels,
    RecallItemView,
    RecallQuery,
    RecallResult,
    RecallSettings,
)
from mu_engine.services.recall.fusion import FusionStrategy, dedup_by_content_hash
from mu_engine.services.recall.ranker import RecallRanker
from mu_engine.services.recall.shared_port import SharedRecallPort, SharedRecallUnavailableError
from mu_engine.storage.domain.namespace import Namespace

__all__ = ["RecallService"]

_DEGRADED_METRIC = "mu_degraded_mode_total"
_SPAN = "recall.federate_live"


class RecallService:
    """The consolidated read service (§1.2). Registered identically in both composition roots; no
    mode branch — THIN/HYBRID differ only in which private-arm repos + which SharedRecallPort
    adapter are injected (§4.2a, CANONICAL §7.9/X11)."""

    def __init__(
        self,
        *,
        embedder: EmbeddingPort,
        private_ranker: RecallRanker,
        shared_recall: SharedRecallPort,
        authz: RecallAuthorizationFilter,
        fusion: FusionStrategy,
        settings: RecallSettings,
        clock: Clock,
        metrics: MetricSink | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._embedder = embedder
        self._private_ranker = private_ranker
        self._shared_recall = shared_recall
        self._authz = authz
        self._fusion = fusion
        self._settings = settings
        self._clock = clock
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._tracer: Tracer = tracer or NoopTracer()

    async def recall(self, scope: ClientScope, q: RecallQuery) -> RecallResult:
        """Federate-live RANKED recall over private-own ⊕ authorized-shared (§1.6).

        Ordered, fail-loud stages: resolve scope → embed once → private arm → shared arm (named
        degrade on failure) → RRF fuse → content_hash dedup → floor-protect → belt authz →
        RecallResult. Emits nothing on the bus (a pure read, §1.2 step 5)."""
        with self._tracer.span(_SPAN, attributes={"strategy": self._settings.strategy}):
            private_ns = q.namespace
            shared_ns = Namespace.shared(
                org=private_ns.org, workspace=private_ns.workspace, session=private_ns.session
            )

            # (1) resolve scope + the caller identity set for EACH arm (Layer 1 assert + Layer 2).
            _pns, private_caller = await self._authz.resolve(scope, private_ns)
            _sns, shared_caller = await self._authz.resolve(scope, shared_ns)

            # (2) embed the query ONCE at the façade boundary (§6-P2/m4).
            query_vec = (await self._embedder.embed([q.text]))[0]

            # (3) PRIVATE arm — own partition, authorized_ids=None (§1.4 Layer 1 authorizes it).
            private = await self._private_ranker.rank(
                private_ns,
                q.text,
                query_vec,
                limit=q.limit,
                channels=q.channels,
                caller_identity_set=private_caller,
            )

            # (4) SHARED arm — authorized at the source; failure is a NAMED degrade, not a drop.
            #     A SHARED η always resolves a caller set (§1.4); an empty set (defensive default)
            #     authorizes NOTHING server-side — the safe direction, never an over-broad match.
            if shared_caller is None:
                shared_caller = frozenset()
            try:
                shared = await self._shared_recall.recall(
                    q.for_namespace(shared_ns), caller_identity_set=shared_caller
                )
            except SharedRecallUnavailableError:
                return self._degrade_private_only(scope, private)

            # (5) RRF fuse the two ALREADY-authorized arms by tier-stable id, then (6) dedup by
            #     content_hash so a pulled shared→local copy + its live origin never double-count.
            fused_pairs = self._fusion.fuse(
                [private.items, shared.items],
                id_of=lambda v: v.memory_id,
                weights=[self._settings.weight_private, self._settings.weight_shared],
                k=self._settings.rrf_k,
            )
            deduped = dedup_by_content_hash([view for view, _score in fused_pairs])

            # (7) floor-protect: a floor member (from either arm) is reorderable, never evicted.
            items = _protect_floor(deduped, limit=q.limit)

            # (8) belt-and-suspenders: re-assert tenancy on every distinct η that survived.
            self._authz.assert_items(frozenset(v.namespace for v in items), scope)

            degraded = private.degraded or shared.degraded
            if degraded is not None:
                self._emit_degraded("ltm")
            return RecallResult(
                namespace=private_ns,
                items=items,
                channels_run=_union_channels(private.channels_run, shared.channels_run),
                degraded=degraded,
                generated_at=self._clock.now(),
            )

    async def recall_degraded(
        self, scope: ClientScope, q: RecallQuery, *, without: set[str], reason: DegradeReason
    ) -> RecallResult:
        """Explicit named channel-subset recall (§1.2/§5): drop the named channels up front and
        LABEL the result. ``without`` names channels (``"stm"``/``"mtm"``/``"ltm"``)."""
        narrowed = RecallChannels(
            stm=q.channels.stm and "stm" not in without,
            mtm=q.channels.mtm and "mtm" not in without,
            ltm=q.channels.ltm and "ltm" not in without,
        )
        result = await self.recall(scope, q.model_copy(update={"channels": narrowed}))
        self._emit_degraded("recall")
        return result.with_degrade(reason)

    def _degrade_private_only(self, scope: ClientScope, private: RecallResult) -> RecallResult:
        """Shared-arm failure → return the private arm intact + the NAMED reason (§1.6/§5.2). The
        reason does NOT change with the execution site (CANONICAL §7.9/X11)."""
        self._authz.assert_items(frozenset(v.namespace for v in private.items), scope)
        self._emit_degraded("shared-recall")
        items = _protect_floor(dedup_by_content_hash(private.items), limit=len(private.items) or 1)
        return RecallResult(
            namespace=private.namespace,
            items=items,
            channels_run=private.channels_run,
            degraded=DegradeReason.SHARED_RECALL_UNAVAILABLE,
            generated_at=self._clock.now(),
        )

    def _emit_degraded(self, component: str) -> None:
        labels: Mapping[str, str] = {"component": sanitize_label_value(component)}
        self._metrics.inc(_DEGRADED_METRIC, labels=labels)


def _protect_floor(items: list[RecallItemView], *, limit: int) -> list[RecallItemView]:
    """Floor members lead and are never evicted; the fused tail fills up to ``limit`` (§1.3)."""
    floor = [v for v in items if v.is_floor]
    floor_ids = {v.memory_id for v in floor}
    tail = [v for v in items if v.memory_id not in floor_ids]
    room = max(0, limit - len(floor))
    return [*floor, *tail[:room]]


def _union_channels(a: RecallChannels, b: RecallChannels) -> RecallChannels:
    return RecallChannels(stm=a.stm or b.stm, mtm=a.mtm or b.mtm, ltm=a.ltm or b.ltm)


# NOTE (tracked gap, explicit — DEV-STANDARDS "no silent stubs"): ANSWER mode (recall + LLM
# synthesis, §3) and INJECT mode (additionalContext render + artifact hydration, §2) are DEFERRED
# this phase — the box cannot reach Azure and no synthesis is done heuristically. They fold in
# behind an injected AnswerSynthesizer / InjectRenderer without changing recall()'s federate core.
