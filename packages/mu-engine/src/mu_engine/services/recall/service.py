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

import structlog

from mu_contracts.contracts.defaults import DEFAULT_RECALL_LIMIT
from mu_contracts.domain.events import DegradeReason
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.model import SparseEncoderPort
from mu_contracts.ports.observability import MetricSink, Tracer
from mu_contracts.ports.time import Clock
from mu_engine.platform.observability import NoopMetricSink, NoopTracer, sanitize_label_value
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.providers.catalog import Task
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
from mu_engine.services.recall.width import ContextBudgetPort, derive_recall_limit
from mu_engine.storage.domain.namespace import Namespace

__all__ = ["RecallService"]

_DEGRADED_METRIC = "mu_degraded_mode_total"
_SPAN = "recall.federate_live"
_log = structlog.get_logger("mu_engine.services.recall.service")


class RecallService:
    """The consolidated read service (§1.2). Registered identically in both composition roots; no
    mode branch — THIN/HYBRID differ only in which private-arm repos + which SharedRecallPort
    adapter are injected (§4.2a, CANONICAL §7.9/X11)."""

    def __init__(
        self,
        *,
        embedder: EmbeddingPort,
        private_ranker: RecallRanker,
        sparse_encoder: SparseEncoderPort | None = None,
        shared_recall: SharedRecallPort,
        authz: RecallAuthorizationFilter,
        fusion: FusionStrategy,
        settings: RecallSettings,
        clock: Clock,
        metrics: MetricSink | None = None,
        tracer: Tracer | None = None,
        context_budget: ContextBudgetPort | None = None,
    ) -> None:
        self._embedder = embedder
        # mtm-retrieval-design.md §1.3, "the M2 resolution": this façade is the ONE place that
        # already holds the EmbeddingPort, so it is also where the SPARSE encoder lives. It
        # encodes the query to a `SparseQuery` value object here, at the boundary, and threads it
        # down beside `query_vec` — so BM25 gets the query TOKENS it needs without any tier repo
        # ever receiving raw text (CANONICAL §6-P2/m4). `None` (the default, and what the
        # composition root wires whenever `RecallSettings.sparse_enabled` is off) leaves every
        # arm dense-only and byte-identical to the pre-hybrid engine.
        self._sparse_encoder = sparse_encoder
        self._private_ranker = private_ranker
        self._shared_recall = shared_recall
        self._authz = authz
        self._fusion = fusion
        self._settings = settings
        self._clock = clock
        self._metrics: MetricSink = metrics or NoopMetricSink()
        self._tracer: Tracer = tracer or NoopTracer()
        # ACCURACY-PLAN-0831.md item 4 (`services/recall/width.py`): the ONE optional seam width
        # derivation needs — a real :class:`~mu_engine.providers.model_router.ModelRouter` (or any
        # other :class:`~mu_engine.services.recall.width.ContextBudgetPort`) satisfies it
        # structurally. ``None`` is the shipped default and a fully legitimate composition-root
        # choice (a plane with no model layer wired at all) — a ``RecallQuery`` with no explicit
        # ``limit`` then falls back to the static wire default, never an invented number.
        self._context_budget = context_budget

    async def recall(self, scope: ClientScope, q: RecallQuery) -> RecallResult:
        """Federate-live RANKED recall over private-own ⊕ authorized-shared (§1.6).

        Ordered, fail-loud stages: resolve scope → embed once → private arm → shared arm (named
        degrade on failure) → RRF fuse → content_hash dedup → floor-protect → belt authz →
        RecallResult. Emits nothing on the bus (a pure read, §1.2 step 5)."""
        with self._tracer.span(_SPAN, attributes={"strategy": self._settings.strategy}):
            # (0) resolve the width ONCE, before either arm runs, so both the private and shared
            #     arms (and the floor-protect slice below) see the SAME concrete limit — deriving
            #     it independently per arm risked two arms disagreeing on a width neither caller
            #     asked for. An explicit `q.limit` always short-circuits derivation (§ "explicit
            #     config still wins"). A local `int` (not `q.limit` re-read later) so every
            #     downstream use is unambiguously concrete, for the type checker and the reader
            #     both — `q` itself is also re-pointed at it so `q.for_namespace(shared_ns)` below
            #     hands the SHARED arm the identical resolved width, never a second `None`.
            effective_limit: int = q.limit if q.limit is not None else self._effective_limit()
            q = q.model_copy(update={"limit": effective_limit})
            private_ns = q.namespace
            shared_ns = Namespace.shared(
                org=private_ns.org, workspace=private_ns.workspace, session=private_ns.session
            )

            # (1) resolve scope + the caller identity set for EACH arm (Layer 1 assert + Layer 2).
            _pns, private_caller = await self._authz.resolve(scope, private_ns)
            _sns, shared_caller = await self._authz.resolve(scope, shared_ns)

            # (2) embed the query ONCE at the façade boundary (§6-P2/m4).
            query_vec = (await self._embedder.embed([q.text]))[0]
            #     ... and, when the hybrid MTM arm is wired, encode the SPARSE query once here
            #     too, for the same reason the dense embed happens once: both arms of the same
            #     intra-MTM fuse must be derived from the same query text, and neither belongs
            #     in a tier adapter.
            sparse_query = (
                self._sparse_encoder.encode_query(q.text)
                if self._sparse_encoder is not None
                else None
            )

            # (3) PRIVATE arm — own partition, authorized_ids=None (§1.4 Layer 1 authorizes it).
            private = await self._private_ranker.rank(
                private_ns,
                q.text,
                query_vec,
                limit=effective_limit,
                channels=q.channels,
                caller_identity_set=private_caller,
                sparse_query=sparse_query,
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
            # D2 (STATE-AND-DEFECTS-0829.md): this fuse's OWN rrf_score used to be discarded
            # (`for view, _score in fused_pairs`), leaving each view's `fused_score` at whatever
            # its SOURCE arm's single-arm fuse already computed — the wrong number one federation
            # layer up from where the same bug lived in `ranker.py`. Re-stamp it with the score
            # THIS fuse actually produced.
            fused_views = [
                view.model_copy(update={"fused_score": rrf_score})
                for view, rrf_score in fused_pairs
            ]
            deduped = dedup_by_content_hash(fused_views)

            # (7) floor-protect: a floor member (from either arm) is reorderable, never evicted.
            items = _protect_floor(deduped, limit=effective_limit)

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

    def _effective_limit(self) -> int:
        """ACCURACY-PLAN-0831.md item 4: the width a caller gets when it did not name one —
        derived from the consuming model's own context budget when both the settings opt-in
        (``derive_limit_from_budget``, default True) and a real :class:`~mu_engine.services.
        recall.width.ContextBudgetPort` are present; ``DEFAULT_RECALL_LIMIT`` (the SAME static
        wire default this engine always shipped) otherwise — "no invented numbers" (`mu_engine.
        providers.shipped_catalog`'s own rule, applied here to width instead of a context window).

        Content-free by construction (CANONICAL §3.1): every value logged below is a token count
        or an item count, never memory content."""
        s = self._settings
        if not s.derive_limit_from_budget or self._context_budget is None:
            _log.debug(
                "recall.width_static",
                limit=DEFAULT_RECALL_LIMIT,
                reason="derive_limit_from_budget=false"
                if not s.derive_limit_from_budget
                else "no_context_budget_port",
            )
            return DEFAULT_RECALL_LIMIT
        max_input_tokens = self._context_budget.context_window(Task.ANSWER)
        limit = derive_recall_limit(
            max_input_tokens=max_input_tokens,
            prompt_reserve_tokens=s.prompt_reserve_tokens,
            answer_reserve_tokens=s.answer_reserve_tokens,
            tokens_per_memory=s.tokens_per_memory,
            min_limit=s.min_derived_limit,
            max_limit=s.max_derived_limit,
        )
        _log.debug(
            "recall.width_derived",
            limit=limit,
            max_input_tokens=max_input_tokens,
            tokens_per_memory=s.tokens_per_memory,
            min_derived_limit=s.min_derived_limit,
            max_derived_limit=s.max_derived_limit,
        )
        return limit


def _protect_floor(items: list[RecallItemView], *, limit: int) -> list[RecallItemView]:
    """Floor members are never evicted; ``items`` already carries the fused rank order (§1.3) —
    THIS fuse's own, over the two ALREADY-authorized arms, one layer up from the in-arm fuse each
    ``ThreeChannelRecallRanker`` already ran.

    **D3 (STATE-AND-DEFECTS-0829.md) — the SAME "floor leads unconditionally" shape lived here
    too, one federation layer up**, and RETRIEVAL-EVAL-0829.md §5.4 measured its cost directly: a
    private-plane STM floor member force-led the two-arm fuse's result, so at k<=5 the FUSED
    recall was WORSE than the shared arm used alone (e.g. ``recall@3``: private-only 0.0000,
    shared-only 0.1723, fused 0.0000 — conv-26, 149 queries) even though federation genuinely
    found evidence neither arm could reach alone (fused beat both at k=10). ``is_floor`` is a
    MEMBERSHIP flag (which ids are protected), never a position instruction — a member keeps
    whatever rank THIS fuse actually gave it; only a member ranked outside the returned window is
    rescued at the END, not the front, exactly mirroring ``ranker.py``'s in-arm
    :func:`~mu_engine.services.recall.ranker._merge_floor` fix (never edit one without the
    other — the SAME shape, DEV-STANDARDS rule 6)."""
    head = items[:limit]
    head_ids = {v.memory_id for v in head}
    rescued = [v for v in items if v.is_floor and v.memory_id not in head_ids]
    room = max(0, limit - len(rescued))
    return [*items[:room], *rescued]


def _union_channels(a: RecallChannels, b: RecallChannels) -> RecallChannels:
    return RecallChannels(stm=a.stm or b.stm, mtm=a.mtm or b.mtm, ltm=a.ltm or b.ltm)


# NOTE (tracked gap, explicit — DEV-STANDARDS "no silent stubs"): ANSWER mode (recall + LLM
# synthesis, §3) and INJECT mode (additionalContext render + artifact hydration, §2) are DEFERRED
# this phase — the box cannot reach Azure and no synthesis is done heuristically. They fold in
# behind an injected AnswerSynthesizer / InjectRenderer without changing recall()'s federate core.
