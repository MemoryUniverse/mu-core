"""``TieredMemoryRepository`` — the application-facing ``MemoryRepository`` façade.

Authority: CANONICAL-CONTRACTS.md §6 row P2 (the façade over the engine-internal tier repos,
behind a ``TierRouter``), ``memory-layer-design.md`` §2 (the façade/tier layering paragraph — who
embeds and where), ``memory-health-pinning-spec.md`` §3.1 lines 160-179 (``set_pinned`` /
``enumerate``).

**What this class actually is.** The design set describes a ``TierRouter`` over the
CANONICAL-pinned ``MemoryTierRepository`` family in ``mu_contracts.ports.memory``. That family has
no implementers and never has: every adapter in this repo binds to the SHIPPED family in
``mu_engine.storage.ports`` (``StmTierRepository.put/get/recent/evict``,
``MtmTierRepository.upsert/get/semantic/…``, ``GraphStorePort.upsert_fact/get_fact/…``), whose own
RE-HOME NOTE (``storage/ports.py`` lines 7-10) explains it was defined there "because that package
is a scaffold this phase" — a statement that is now stale. So this façade is an ADAPTER, not
merely a router: it implements the published ``MemoryRepository`` Protocol directly over the
shipped ports plus the model translation in ``translation.py``. Growing the published family onto
ten adapter classes instead would be an order-of-magnitude larger change, and three of those
adapters could not implement it at all. Recorded as an architecture delta, not decided silently.

**The consumers this must satisfy** are ``MemoryHealthService`` (``services/health/service.py``)
and ``PinService`` (``services/pin/service.py``), which between them call exactly three of the
seven methods — ``enumerate``, ``get``, ``set_pinned`` — and whose call sites, not a guessed shape,
are what the behaviour here is built against. The other four are implemented rather than stubbed
(the house absence rule forbids ``NotImplementedError``), and each says below what it is really
built on.

**Read-purity.** ``MemoryHealthService`` holds no write port at all; that is what makes its
side-effect-freedom structural. ``enumerate`` here honours the same discipline: no reinforcement,
no ``access_count`` bump, no tier transition, no ``last_seen`` write-back.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import cast

import structlog

from mu_contracts.domain.errors import (
    LlmNotConfiguredError,
    PinPartiallyAppliedError,
    PinTargetNotFoundError,
    TierCapabilityUnavailableError,
    TierRepositoryUnavailableError,
)
from mu_contracts.domain.model.memory import MemoryItem, Namespace, State, Tier
from mu_contracts.domain.model.recall import CallerIdentitySet, Scored
from mu_contracts.ports.model import EmbeddingPort
from mu_engine.services.memory.cursor import (
    MAX_STALLED_PAGES,
    TierCursor,
    decode_cursor,
    encode_cursor,
)
from mu_engine.services.memory.router import TierLeg, TierRouter
from mu_engine.services.memory.translation import (
    to_contract_item,
    to_contract_scored,
    to_engine_item,
    to_engine_states,
)
from mu_engine.storage.domain.memory import MemoryItem as EngineMemoryItem

__all__ = ["TieredMemoryRepository"]

_log = structlog.get_logger("mu_engine.services.memory.repository")

#: The per-tier cursor value meaning "this leg has not been started yet".
#:
#: The composite cursor uses ABSENCE to mean EXHAUSTED, so it needs a distinct token for a leg
#: that a page ran out of budget before reaching. Without it, a continuation page could not tell
#: "the LTM walk finished" from "the LTM walk never began", and one of those two readings silently
#: drops a tier's memories out of the caller's view for the rest of the walk.
_UNSTARTED = ""


class TieredMemoryRepository:
    """Implements ``mu_contracts.ports.memory.MemoryRepository`` over the shipped tier adapters."""

    def __init__(
        self,
        *,
        router: TierRouter,
        embedder: EmbeddingPort | None = None,
    ) -> None:
        self._router = router
        # `semantic` is the ONE method that needs a model, and only for the embed step CANONICAL
        # §6-P2 places at this boundary. `enumerate`/`set_pinned` must never reach for one:
        # memory-health-pinning-spec §3.2 line 181 states outright "No LLMProviderPort, no
        # EmbeddingPort — health and pin are fully deterministic", and says so specifically to
        # forbid a future model dependency creeping onto those paths. Optional here so a
        # health/pin-only binding constructs with zero model wiring at all.
        self._embedder = embedder

    # ------------------------------------------------------------------------------- reads --
    async def get(self, ns: Namespace, id: str) -> MemoryItem | None:
        """First resident copy of ``id``, across every tier.

        ``PinService._resolve`` calls this before every pin and reads ``.state`` and ``.pinned``
        off the result, so "first resident copy" has to mean the tier the item actually lives in.
        The router's walk order (STM -> MTM -> LTM) is the promotion order, and a memory sits in
        exactly one tier outside a transient promotion window.

        The parameter is named ``id`` because the Protocol names it ``id``; renaming it here would
        break every keyword call site against the published port.
        """
        found = await self._router.fan_get(ns, id)
        return None if found is None else to_contract_item(found)

    async def enumerate(
        self,
        ns: Namespace,
        *,
        states: frozenset[State],
        tiers: frozenset[Tier] | None,
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        """One BOUNDED page of ``ns``'s partition, walked across tiers in order.

        **Sequential across tiers, not concurrent — and that is a deliberate correctness choice,
        against the house preference for ``TaskGroup`` fan-out.** The three tiers page on three
        incompatible, OPAQUE, positional cursors (a ZSET rank, a Qdrant scroll offset, a keyset
        ``m.id``). A concurrent fan-out would have to ask each leg for a full ``limit``, merge, and
        truncate — and then it could not compose a correct continuation, because there is no way
        to advance an opaque positional cursor *partially*. It would either re-serve the truncated
        rows (a walk that loops) or drop them (a walk with holes). Draining one tier at a time
        makes the composite cursor exact: at most one leg is ever mid-page.

        Concurrency is still used where it composes — ``set_pinned``, ``get`` and ``by_artifact``
        all fan out under a ``TaskGroup``/``gather``, because none of them carries a cursor.

        **A short page with a continuation is a normal answer**, not a bug: the ``states``/
        ``pinned`` predicates reject rows a store cannot filter server-side, and each leg is capped
        by how much it INSPECTS. The caller pages on ``next_cursor``, which is ``None`` iff every
        walked tier is exhausted.

        **Degrade is a RAISE, not an internal fallback.** ``MemoryHealthService._walk`` catches
        ``TierRepositoryUnavailableError`` and retries with ``tiers={STM, MTM}``, emitting the
        named ``DegradedModeEntered(LTM_UNAVAILABLE)`` and marking the view ``partial``. Silently
        skipping a dead tier here would rob it of that decision and hand back a page
        indistinguishable from a complete one — the exact defect the ``partial`` flag exists to
        prevent. So a leg that cannot serve propagates, and the consumer owns the degrade.

        **``tiers`` narrows which tiers THIS PAGE reads — not which tiers the walk covers.** The
        one caller that narrows is the degraded retry, and it narrows for a page, not for a walk:
        the tier it skipped is still part of the partition and its rows are still owed. So a
        narrowed page reads fewer stores but the continuation still accounts for every bound tier,
        and ``next_cursor is None`` keeps meaning what ``ports/memory.py`` lines 78-80 say it
        means — the WALK is exhausted — rather than the weaker "the tiers you asked for this time
        are exhausted", which would let a caller mistake a partial read for a complete one.

        **A tier NARROWED AWAY mid-walk keeps its position; it is never dropped.** The degraded
        retry re-issues the SAME cursor with ``tiers={STM, MTM}``, so the token arriving here can
        name a tier this call will not walk. Rebuilding the continuation only from the legs that
        WERE walked would delete that tier's position from the token, and every later page would
        then read ``positions.get(LTM) is None`` as "the LTM walk finished earlier" — a walk that
        ends ``next_cursor=None`` (which ``ports/memory.py`` lines 78-80 define as EXHAUSTED) over
        a tier that was never read. That is the same lie the raise above exists to prevent, only
        deferred by one page. So an un-walked BOUND tier's position is carried forward verbatim
        (``carried`` below) and the walk resumes it as soon as the caller stops narrowing.

        **A page that cannot make progress is tolerated ONCE, then refused — never looped.** If
        every leg this call may walk is already exhausted while a narrowed-away tier still has an
        unread position, this page has nothing to give. Returning ``None`` would claim
        completeness over an unread tier (the lie), and returning the token unconditionally would
        hand back the same empty page forever if that tier never comes back (the loop). So the
        FIRST such page returns empty with the position preserved — that is exactly the round trip
        in which ``MemoryHealthService._walk`` re-attempts un-narrowed and a recovered tier
        resumes — and a SECOND consecutive one raises ``TierRepositoryUnavailableError`` naming
        the unread tiers, at which point the narrowed retry also fails and the error propagates
        loud, exactly as that method's own docstring says a second failure must.
        """
        if limit <= 0:
            return [], None
        legs = self._router.legs_for(tiers)
        # Capability is checked for EVERY applicable leg BEFORE any store is touched, so a binding
        # that cannot enumerate is refused by name instead of returning a partial page that looks
        # complete.
        for leg in legs:
            leg.as_enumerator()
        positions, stalls = self._starting_positions(ns, cursor)
        engine_states = to_engine_states(states)

        walked = {leg.tier for leg in legs}
        bound = {leg.tier for leg in self._router.legs}
        # Positions for tiers this deployment binds but this CALL was narrowed away from. A tier
        # the deployment does not bind at all is dropped instead: no leg can ever resume it, so
        # carrying it would make the walk non-terminating for no gain.
        carried: TierCursor = {
            tier: position
            for tier, position in positions.items()
            if tier in bound and tier not in walked
        }
        if carried and not any(positions.get(leg.tier) is not None for leg in legs):
            if stalls >= MAX_STALLED_PAGES:
                raise TierRepositoryUnavailableError(
                    "the walk cannot continue: every walkable tier is exhausted while "
                    + ", ".join(sorted(tier.value for tier in carried))
                    + " is still unread"
                )
            return [], encode_cursor(ns, carried, stalls=stalls + 1)

        collected: list[tuple[Tier, EngineMemoryItem]] = []
        next_positions: TierCursor = dict(carried)
        for leg in legs:
            position = positions.get(leg.tier)
            if position is None:
                continue  # this tier's walk finished on an earlier page.
            if len(collected) >= limit:
                next_positions[leg.tier] = position  # not reached this page; carry it forward.
                continue
            page, leg_cursor = await self._router.enumerate_leg(
                leg,
                ns,
                states=engine_states,
                pinned=pinned,
                cursor=None if position == _UNSTARTED else position,
                limit=limit - len(collected),
            )
            collected.extend((leg.tier, item) for item in page)
            if leg_cursor is not None:
                next_positions[leg.tier] = leg_cursor
        items = [to_contract_item(item) for item in self._router.dedupe(collected)]
        # ``stalls=0``: this page consulted at least one store, so the walk is moving again.
        return items, encode_cursor(ns, next_positions)

    def _starting_positions(self, ns: Namespace, cursor: str | None) -> tuple[TierCursor, int]:
        """Where each BOUND leg resumes — every bound leg, not only the ones this page may read.

        A continuation trusts the token for POSITION only — the partition is always re-derived
        from the authorized ``ns`` — and a leg the token does not mention is one whose walk is
        already complete.

        **A fresh call seeds every BOUND tier, even the ones ``tiers=`` narrows away, and that is
        load-bearing rather than sloppy.** The first page of a walk is exactly where
        ``MemoryHealthService._walk``'s degraded retry arrives with ``cursor=None`` and
        ``tiers={STM, MTM}``. Seeding only the narrowed legs there would mean the LTM leg was
        never in the token at all, so no later page could carry it forward and the tier would be
        absent from the entire walk — the same defect as dropping it from a continuation, entered
        one page earlier.
        """
        if cursor is None:
            return {leg.tier: _UNSTARTED for leg in self._router.legs}, 0
        return decode_cursor(ns, cursor)

    async def semantic(
        self, ns: Namespace, query: str, *, k: int, authorized_ids: CallerIdentitySet
    ) -> list[Scored[MemoryItem]]:
        """Dense recall over the MTM tier, embedding the query at THIS boundary.

        CANONICAL §6-P2 and ``memory-layer-design.md`` §2 both put the embed step here and not in
        the tier repo: ``MtmTierRepository.semantic`` deliberately takes an already-embedded
        vector and a ``CallerIdentitySet`` and "never sees a raw query string". This is that step.

        Scope: the single dense MTM arm the published Protocol's signature describes — one
        ``query``, one ``k``, one ``authorized_ids``, no sparse parameter and no fusion inputs.
        The three-channel fused recall path (STM floor ⊕ MTM dense/sparse ⊕ LTM graph, RRF,
        federate-live) belongs to ``RecallService``/``ThreeChannelRecallRanker``, which
        ``memory-layer`` §2 itself names as the in-subsystem caller that already does this job.
        Re-deriving fusion here would put a second, drifting answer to the same question behind
        the same façade.

        **``authorized_ids`` is passed THROUGH, never coerced — an EMPTY set is a real answer.**
        ``CallerIdentitySet`` is a bare ``frozenset[str]`` (``domain/model/recall.py:27``) with no
        non-empty constraint, so ``frozenset()`` is legal input and means "this caller is
        authorized for nothing". The MTM adapters gate the Model-A clause on
        ``caller_identity_set is not None`` (``qdrant_mtm.py:402``, ``falkor_ltm.py:561``,
        ``weaviate_mtm.py:657``, ``pgvector_mtm.py:213``, ``chroma_mtm.py:167``), so ``None`` is
        their sentinel for *omit the filter entirely*. Turning the empty set into ``None`` would
        therefore hand a caller authorized for NOTHING every memory in a SHARED room. This is the
        rule ``RecallService`` states in words two files away and applies in the opposite
        direction (``services/recall/service.py:108-112`` — *"an empty set (defensive default)
        authorizes NOTHING server-side — the safe direction, never an over-broad match"*), and it
        coerces ``None -> frozenset()``. The façade does the same, never the inverse.
        """
        embedder = self._embedder
        if embedder is None:
            # Named refusal, never a silent empty result set: an empty recall is indistinguishable
            # from "you have no matching memories", which would be a wrong answer rather than an
            # unavailable one.
            raise LlmNotConfiguredError(
                "MemoryRepository.semantic needs an EmbeddingPort at the façade boundary "
                "(CANONICAL §6-P2); this repository was built without one"
            )
        leg = self._leg(Tier.MTM)
        searcher = getattr(leg.store, "semantic", None)
        if not callable(searcher):
            raise TierCapabilityUnavailableError(
                f"the mtm backend {leg.backend!r} cannot serve a semantic query"
            )
        vectors = await embedder.embed([query])
        scored = await self._router.guarded(
            leg,
            lambda: searcher(
                ns,
                vectors[0],
                limit=k,
                # NOT ``or None``: an empty caller set must stay empty (deny-all), because the
                # adapters read ``None`` as "no Model-A filter at all". See the docstring above.
                caller_identity_set=frozenset(authorized_ids),
            ),
        )
        return [to_contract_scored(hit) for hit in scored]

    async def by_artifact(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]:
        """Every memory referencing ``artifact_id``, fanned across the tiers that INDEX it.

        Answered by the LTM graph (a ``REFERENCES`` edge traversal off the merged ``:Artifact``
        node) and the MTM vector tier (a server-side filter on the already-indexed ``artifact_ref``
        payload key) — never a label scan on either, which is what the port means by "the
        FIRST-CLASS reverse lookup ... never a scan".

        **Reported coverage gap: the STM tier does not participate**, because it has no reverse
        ``artifact_ref`` index at all — only a payload blob and a recency ZSET. A memory that is
        still STM-resident and references an artifact is therefore NOT counted here. That matters
        for artifact GC-eligibility (≥1 live ref ⇒ not GC-eligible, memory-layer §2 lines 312-321),
        so it is named rather than absorbed: the fix is an ``mu/{prefix}:stm:artifact:{id}`` set
        maintained by the STM write path, which is a change to a hot, heavily-tested write path
        and belongs in its own unit of work.
        """
        found = await self._router.fan_by_artifact(ns, artifact_id)
        return [to_contract_item(item) for item in found]

    # ------------------------------------------------------------------------------ writes --
    async def add(self, item: MemoryItem) -> None:
        """Persist ``item`` into the tier it names.

        A plain repository write — the store-level half of an ingest, not the ingest itself.
        Extraction, distillation, dedup policy and promotion belong to ``IngestService`` /
        ``DistillPipeline``; a repository's ``add`` that reached for those would invert the
        layering and give the engine two competing write paths.
        """
        engine_item = to_engine_item(item)
        leg = self._leg(item.tier)
        writer = _writer_for(leg)
        if writer is None:
            raise TierCapabilityUnavailableError(
                f"the {leg.tier.value} backend {leg.backend!r} exposes no write primitive"
            )
        await self._router.guarded(leg, lambda: writer(engine_item))

    async def set_pinned(
        self,
        ns: Namespace,
        id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None = None,
    ) -> int:
        """Apply the pin group across EVERY tier holding ``id``, id-stably. Returns the version.

        **What the stores can actually guarantee, stated plainly: convergence, not atomicity.**
        Each leg is individually atomic (a Redis ``MULTI``, a Qdrant ``set_payload`` on one point,
        one Cypher ``SET``). Across the three there is no shared transaction, no two-phase commit
        and no distributed log — and ``IdempotentWriteScope``, which ``PinService`` wraps this
        call in, is an outbox for the event BUS, not a transaction over the stores. The port asks
        for the pin to be "applied across every store the item resides in", which is a convergence
        obligation, and that is the one being met.

        The behaviour is therefore chosen, not incidental:

        * **Residency and the write are ONE round trip per tier.** Each leg's ``set_pinned``
          returns ``None`` when it does not hold the id. A separate probe-then-write would add a
          hop per tier AND a TOCTOU window in which a promotion moves the item between the two —
          the window ``PinService`` cannot close itself, since its own ``get`` already happened.
        * **No tier holds it** -> ``PinTargetNotFoundError``. The port requires this explicitly:
          "a missing id raises ``PinTargetNotFoundError`` — never a silent no-op".
        * **Every leg fails** -> ``TierRepositoryUnavailableError``. Nothing landed, so this is an
          outage, not a partial apply, and calling it partial would send an operator looking for
          divergence that does not exist.
        * **Some landed, some failed** -> ``PinPartiallyAppliedError``, naming both sets. It is a
          RAISE so that ``PinService._commit``'s write scope never commits and no
          ``MemoryPinned``/``MemoryUnpinned`` is published: an event announcing a pin that only
          half-landed is worse than no event.
        * **The landed legs are NOT rolled back.** A compensating write can fail the same way the
          original leg did, turning one inconsistency into two. ``set_pinned`` is a full
          field-group upsert with a caller-supplied ``at``/``by``/``reason``, so re-running the
          whole call converges — retry is the recovery, and the landed legs simply re-converge.

        **WHEN a partial apply is actually reachable — corrected after measuring it.** An earlier
        version of this docstring claimed that "an STM-only item pinned while the LTM store is
        down reports a partial apply". It does not, and the reason matters: ``PinService._resolve``
        (``services/pin/service.py:231``) calls ``get`` BEFORE this method, ``get`` fans out under
        an ``asyncio.TaskGroup`` (``router.py:179``), and one leg's failure cancels its siblings
        and propagates. So a whole-tier outage fails at the READ step with
        ``TierRepositoryUnavailableError`` and nothing is ever written — verified by running it.
        A partial apply is reachable from a per-item failure or from a store that dies BETWEEN
        the ``get`` and the write, not from a tier that is simply down.

        The consequence, stated because it is a real availability property and not an accident:
        **pin and unpin are unavailable while ANY one bound tier store is down**, even for an item
        resident wholly in a healthy tier. That is deliberate here rather than repaired, and the
        reason is that CANONICAL models NO degrade for pin — the one memory-tier degrade it names
        (``LTM_UNAVAILABLE``) is a READ degrade the health lens owns. Making ``get`` answer from a
        surviving leg would let ``set_pinned`` proceed and land a write on one store while another
        is unreachable, i.e. it would trade a clean, fully-retryable refusal for a half-landed
        write whose event never publishes. Choosing between those is an owner decision about a
        contract CANONICAL does not yet contain, so it is REPORTED, not invented here.

        Within that, a failed leg is still counted as possibly-resident: a store that could not
        answer cannot tell us whether it held the id, and assuming a silent tier held nothing
        would report success over a pin that genuinely failed to reach the tier the item lives in.

        **The two directions are not equally safe.** For a pin, a leftover pinned leg is merely
        un-GC-able until reconciled. For an UNPIN it strands the row as permanently GC-ineligible
        — exactly what ``PinService`` documents unpin as existing to prevent. The raised error
        carries the direction so the two can be told apart; a genuine reconciliation path for the
        unpin case is NOT built here and is reported as an open owner decision.
        """
        legs = self._router.legs
        for leg in legs:
            leg.as_pin_writer()  # refuse an incapable binding BEFORE writing anything.
        applied, failed = await self._router.fan_set_pinned(
            ns, id, pinned, at=at, by=by, reason=reason, legs=legs
        )
        if not applied and not failed:
            # Non-enumerating denial: the message never echoes the id, so a probe cannot use pin
            # to test for existence (the discipline PinService._resolve already follows).
            raise PinTargetNotFoundError("not found")
        if failed and not applied:
            raise TierRepositoryUnavailableError(
                "no tier could apply the pin: " + ", ".join(sorted(t.value for t in failed))
            )
        if failed:
            landed = frozenset(t.value for t in applied)
            missed = frozenset(t.value for t in failed)
            _log.error(
                "pin_partially_applied",
                ns=ns.to_prefix(),
                memory_id=id,
                pinned=pinned,
                applied=sorted(landed),
                failed=sorted(missed),
            )
            raise PinPartiallyAppliedError(
                f"pin applied on {sorted(landed)} but not on {sorted(missed)}",
                applied=landed,
                failed=missed,
                pinned=pinned,
            )
        # Every leg that held the id wrote the SAME derived version (each bumps its own copy's
        # counter), so they agree outside a transient promotion window where one tier's copy is
        # older. The max is the truthful "newest revision now in the system".
        return max(applied.values())

    # ---------------------------------------------------------------------------- internals --
    def _leg(self, tier: Tier) -> TierLeg:
        for leg in self._router.legs:
            if leg.tier is tier:
                return leg
        raise TierCapabilityUnavailableError(f"no {tier.value} tier is bound in this deployment")


def _writer_for(leg: TierLeg) -> Callable[[EngineMemoryItem], Awaitable[object]] | None:
    """The write primitive on this leg's adapter, whatever the shipped port calls it.

    ``StmTierRepository.put`` / ``MtmTierRepository.upsert`` / ``GraphStorePort.upsert_fact`` are
    the same operation under three names, with no common base declaring it — resolved structurally
    rather than by widening the shipped ports.
    """
    for name in ("put", "upsert", "upsert_fact"):
        candidate: object = getattr(leg.store, name, None)
        if callable(candidate):
            return cast("Callable[[EngineMemoryItem], Awaitable[object]]", candidate)
    return None
