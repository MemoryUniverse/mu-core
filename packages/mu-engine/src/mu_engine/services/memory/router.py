"""``TierRouter`` — the STM/MTM/LTM fan-out CANONICAL §6-P2 puts under the façade.

Authority: CANONICAL-CONTRACTS.md §6 row P2 — *"``MemoryRepository`` is the application-facing
façade; ``Stm/Mtm/LtmTierRepository`` are engine-internal behind a ``TierRouter``"* — and
``memory-health-pinning-spec.md`` §3.1 line 179, which names this class as the thing the façade's
``set_pinned``/``enumerate`` fan across.

The router owns four things the façade above it should not have to think about:

* **which tiers can answer at all** — the capability gate (:meth:`legs_for`),
* **store failure translation** — raw client exceptions become ONE named domain error,
* **concurrency** where it is correct (``TaskGroup`` for the by-id fan-outs),
* **de-duplication** across tiers, which is load-bearing for correctness, not cosmetic.

Everything here speaks the ENGINE record. The crossing to the published one happens above, in
``repository.py``, via ``translation.py`` — one boundary, one place.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TypeVar

import structlog

from mu_contracts.domain.errors import (
    MemoryUniverseError,
    TierCapabilityUnavailableError,
    TierRepositoryUnavailableError,
)
from mu_contracts.domain.model.memory import Tier
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.tier_capabilities import TierEnumerationPort, TierPinPort

__all__ = ["TierLeg", "TierRouter"]

_log = structlog.get_logger("mu_engine.services.memory.router")

T = TypeVar("T")

#: The tier walk order. STM -> MTM -> LTM is the promotion order (CANONICAL §7.1), so a
#: sequential walk in this order visits a memory in the tier it is most likely to still be
#: resident in first, and — because ``enumerate`` de-duplicates on the tier-stable id — the copy
#: that wins a tie is the one from the tier closest to where the item actually lives.
TIER_ORDER: tuple[Tier, ...] = (Tier.STM, Tier.MTM, Tier.LTM)


class TierLeg:
    """One tier's store adapter plus what it is structurally able to do.

    Capability is probed with ``isinstance`` against the two ``runtime_checkable`` Protocols,
    which for a Protocol means "has these methods". That is exactly the right question: three of
    the five vector backends (``pgvector``, ``chroma``, ``faiss``) genuinely have no enumeration
    or point-get primitive, and a deployment bound to one of them must be TOLD that ``/health``
    and ``/pin`` cannot be served rather than being served an empty page that reads as "your
    partition is fine".
    """

    def __init__(self, tier: Tier, store: object, *, backend: str | None = None) -> None:
        self.tier = tier
        self.store = store
        #: A name for the bound backend, used only in refusal messages so an operator learns
        #: WHICH backend cannot serve, not merely that something cannot.
        self.backend = backend or type(store).__name__
        self.enumerable = isinstance(store, TierEnumerationPort)
        self.pinnable = isinstance(store, TierPinPort)

    def as_enumerator(self) -> TierEnumerationPort:
        if not isinstance(self.store, TierEnumerationPort):
            raise TierCapabilityUnavailableError(
                f"the {self.tier.value} backend {self.backend!r} cannot enumerate a partition "
                "(no bounded, paginated walk primitive); /health and the pin bound cannot be "
                "served on this binding"
            )
        return self.store

    def as_pin_writer(self) -> TierPinPort:
        if not isinstance(self.store, TierPinPort):
            raise TierCapabilityUnavailableError(
                f"the {self.tier.value} backend {self.backend!r} cannot apply an id-stable pin "
                "upsert; /pin and /unpin cannot be served on this binding"
            )
        return self.store


class TierRouter:
    """Routes one façade call across the tier legs it applies to."""

    def __init__(self, legs: tuple[TierLeg, ...]) -> None:
        by_tier = {leg.tier: leg for leg in legs}
        self._legs = tuple(by_tier[tier] for tier in TIER_ORDER if tier in by_tier)

    @property
    def legs(self) -> tuple[TierLeg, ...]:
        return self._legs

    def legs_for(self, tiers: frozenset[Tier] | None) -> tuple[TierLeg, ...]:
        """The legs a call applies to, in walk order.

        ``tiers=None`` means every bound tier (``ports/memory.py`` line 74's ``tiers`` filter),
        which is what ``MemoryHealthService`` passes on its first attempt; the narrowed
        ``{STM, MTM}`` set is what it retries with after an LTM failure.
        """
        if tiers is None:
            return self._legs
        return tuple(leg for leg in self._legs if leg.tier in tiers)

    def missing_capabilities(self) -> tuple[str, ...]:
        """Human-readable names of the capabilities no bound backend can serve.

        Reported by the composition root at BUILD time so a deployment on a backend that cannot
        enumerate learns it at boot, next to the other fail-loud binding refusals
        (``StoreRegistry.assert_mandatory_roles``), rather than on the first ``/health`` call.
        """
        gaps: list[str] = []
        for leg in self._legs:
            if not leg.enumerable:
                gaps.append(f"{leg.tier.value}:enumerate({leg.backend})")
            if not leg.pinnable:
                gaps.append(f"{leg.tier.value}:set_pinned({leg.backend})")
        return tuple(gaps)

    # ------------------------------------------------------------------ failure translation --
    async def guarded(self, leg: TierLeg, call: Callable[[], Awaitable[T]]) -> T:
        """Run one leg's store call, translating any store failure into ONE named domain error.

        **This method is why the degrade path works at all.** ``MemoryHealthService._walk``
        catches ``mu_contracts.domain.errors.TierRepositoryUnavailableError`` and NOTHING else.
        Almost nothing in the storage layer raises it: ``retry_io`` retries and re-raises, so a
        Qdrant/FalkorDB/Redis outage arrives as a RAW client exception, and there is a SECOND,
        unrelated ``TierRepositoryUnavailableError`` in ``mu_engine.storage.errors`` that
        subclasses ``StorageError(Exception)`` — not ``MemoryUniverseError``, despite its own
        docstring claiming otherwise (``storage/errors.py:28``, reported). Either would sail past
        the degrade branch and fail the whole request. The façade is the translation point.

        Domain errors pass through UNTOUCHED — a ``PinTargetNotFoundError`` is an answer, not an
        outage, and wrapping it would turn a precise refusal into a false degrade.

        Everything else is wrapped WITH ITS CAUSE CHAINED (``from exc``). Chaining is what keeps
        this honest: the class asserts only "this tier could not serve", while the traceback still
        carries the real exception, so a genuine bug (a ``TypeError`` in an adapter, say) is
        neither hidden nor silently re-labelled as an infrastructure problem.
        """
        try:
            return await call()
        except asyncio.CancelledError:
            raise
        except MemoryUniverseError:
            raise
        except Exception as exc:
            _log.warning(
                "tier_store_unavailable",
                tier=leg.tier.value,
                backend=leg.backend,
                error_type=type(exc).__name__,
            )
            raise TierRepositoryUnavailableError(
                f"the {leg.tier.value} tier store ({leg.backend}) could not serve this call"
            ) from exc

    # ------------------------------------------------------------------------- the fan-outs --
    async def fan_get(self, ns: Namespace, memory_id: str) -> MemoryItem | None:
        """First resident copy of ``memory_id``, searched across every tier concurrently.

        Under ``asyncio.TaskGroup`` (DEV-STANDARDS rule 1, structured concurrency): a failing leg
        cancels its siblings rather than leaving orphaned in-flight store I/O, and the error
        surfaces instead of being averaged away into a ``None`` that would read as "no such
        memory". A point-get is the one shape where concurrency is unambiguously correct — the
        legs share no ordering and no cursor.
        """
        legs = self._legs
        if not legs:
            return None
        tasks: list[asyncio.Task[MemoryItem | None]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._get_one(leg, ns, memory_id)) for leg in legs]
        except* MemoryUniverseError as group:
            leaf = _first_leaf(group)
            # Re-chained to its OWN cause — the raw store error ``guarded`` recorded — rather
            # than to the ``ExceptionGroup`` scaffolding or to ``None``. ``from None`` would
            # discard the root cause outright, which is the one thing this translation must
            # not do.
            raise leaf from leaf.__cause__
        for task in tasks:
            found = task.result()
            if found is not None:
                return found
        return None

    async def _get_one(self, leg: TierLeg, ns: Namespace, memory_id: str) -> MemoryItem | None:
        getter = _point_getter(leg)
        if getter is None:
            return None
        return await self.guarded(leg, lambda: getter(ns, memory_id))

    async def fan_set_pinned(
        self,
        ns: Namespace,
        memory_id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None,
        legs: tuple[TierLeg, ...],
    ) -> tuple[dict[Tier, int], frozenset[Tier]]:
        """Apply the pin group on EVERY leg concurrently; report what landed and what did not.

        Returns ``(applied, failed)`` where ``applied`` maps each tier that WROTE to the new
        version it wrote, and ``failed`` names the tiers whose leg errored. A tier that simply
        does not hold the id appears in neither — it returned ``None``, which is an answer, not a
        failure.

        **Every leg is awaited before anything is decided.** ``asyncio.gather(...,
        return_exceptions=True)`` rather than a ``TaskGroup`` here, deliberately and against the
        house default: a ``TaskGroup`` cancels its siblings on the first failure, which would
        leave the remaining legs in an unknown, half-issued state — the one outcome a cross-store
        write must never produce, because the caller then cannot say which stores it has to
        reconcile. Short-circuiting a fan-out that MUTATES is how partial writes become invisible.
        """
        results = await asyncio.gather(
            *(
                self._pin_one(leg, ns, memory_id, pinned, at=at, by=by, reason=reason)
                for leg in legs
            ),
            return_exceptions=True,
        )
        applied: dict[Tier, int] = {}
        failed: set[Tier] = set()
        cancelled: BaseException | None = None
        for leg, outcome in zip(legs, results, strict=True):
            if isinstance(outcome, asyncio.CancelledError):
                # Cancellation is NOT a store failure and must never be reported as a partial
                # apply; it is re-raised after the loop so the whole call unwinds (rule 1:
                # CancelledError propagates, never swallowed).
                cancelled = outcome
            elif isinstance(outcome, BaseException):
                failed.add(leg.tier)
            elif outcome is not None:
                applied[leg.tier] = outcome
        if cancelled is not None:
            raise cancelled
        return applied, frozenset(failed)

    async def _pin_one(
        self,
        leg: TierLeg,
        ns: Namespace,
        memory_id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None,
    ) -> int | None:
        writer = leg.as_pin_writer()
        return await self.guarded(
            leg, lambda: writer.set_pinned(ns, memory_id, pinned, at=at, by=by, reason=reason)
        )

    async def fan_by_artifact(self, ns: Namespace, artifact_id: str) -> list[MemoryItem]:
        """Every memory referencing ``artifact_id``, across the tiers that can answer by INDEX.

        Concurrent (``TaskGroup``) and de-duplicated on the tier-stable id, so an artifact whose
        memories straddle a promotion boundary is counted once — which matters, because this is
        the reference-count authority for artifact GC-eligibility (memory-layer §2 lines 312-321).
        """
        legs = [leg for leg in self._legs if _artifact_reader(leg) is not None]
        if not legs:
            return []
        tasks: list[tuple[TierLeg, asyncio.Task[list[MemoryItem]]]] = []
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    (leg, tg.create_task(self._by_artifact_one(leg, ns, artifact_id)))
                    for leg in legs
                ]
        except* MemoryUniverseError as group:
            leaf = _first_leaf(group)
            # Re-chained to its OWN cause — the raw store error ``guarded`` recorded — rather
            # than to the ``ExceptionGroup`` scaffolding or to ``None``. ``from None`` would
            # discard the root cause outright, which is the one thing this translation must
            # not do.
            raise leaf from leaf.__cause__
        return self.dedupe([(leg.tier, item) for leg, task in tasks for item in task.result()])

    async def _by_artifact_one(
        self, leg: TierLeg, ns: Namespace, artifact_id: str
    ) -> list[MemoryItem]:
        reader = _artifact_reader(leg)
        if reader is None:
            return []
        return await self.guarded(leg, lambda: reader(ns, artifact_id))

    async def enumerate_leg(
        self,
        leg: TierLeg,
        ns: Namespace,
        *,
        states: frozenset[MemoryState],
        pinned: bool | None,
        cursor: str | None,
        limit: int,
    ) -> tuple[list[MemoryItem], str | None]:
        """One leg's bounded page, capability-gated and failure-translated."""
        enumerator = leg.as_enumerator()
        return await self.guarded(
            leg,
            lambda: enumerator.enumerate_page(
                ns, states=states, pinned=pinned, cursor=cursor, limit=limit
            ),
        )

    @staticmethod
    def dedupe(candidates: list[tuple[Tier, MemoryItem]]) -> list[MemoryItem]:
        """Collapse the same memory id appearing in more than one tier down to ONE row.

        **Required for correctness, and unstated by both the Protocol and the spec.** The id is
        tier-stable by construction (CANONICAL §7.1), and a promotion or distillation window
        genuinely leaves a copy in two tiers at once. Without this,
        ``PinService._assert_within_pin_bound`` — which counts ``len(page)`` from ONE
        ``enumerate(pinned=True)`` round trip against the namespace pin limit — would count such
        an item twice and start refusing pins the partition has room for, and
        ``MemoryHealthSummary.total`` would over-report the same way.

        The winner is the copy whose OWN ``item.tier`` matches the tier that produced it: that is
        the row sitting where it claims to live, so a stale duplicate left behind by an
        in-progress tier move loses to the real one. Failing that, walk order (STM -> MTM -> LTM)
        breaks the tie, which keeps the result deterministic — a health view that reordered
        itself between two identical calls would be unreadable.
        """
        chosen: dict[str, tuple[bool, MemoryItem]] = {}
        for tier, item in candidates:
            resident = _tier_matches(item, tier)
            existing = chosen.get(item.id)
            if existing is None or (resident and not existing[0]):
                chosen[item.id] = (resident, item)
        return [item for _resident, item in chosen.values()]


def _tier_matches(item: MemoryItem, tier: Tier) -> bool:
    """Whether ``item`` claims to live in the tier that returned it. Compared on the enum VALUE
    because the item carries the engine ``MemoryTier`` and the router speaks the published
    ``Tier``; the two are separate enums over the same axis."""
    return item.tier.value == tier.value


def _point_getter(leg: TierLeg) -> Callable[[Namespace, str], Awaitable[MemoryItem | None]] | None:
    """The by-id read on this leg's adapter, whatever that adapter calls it.

    The three shipped tier ports name the same operation three different ways —
    ``StmTierRepository.get``, ``MtmTierRepository.get``, ``GraphStorePort.get_fact`` — and no
    common base declares it. Resolved structurally here rather than by widening the shipped ports,
    which would force the method onto backends that do not have it. ``None`` means this backend
    cannot point-get at all (``pgvector``/``chroma``/``faiss``), which ``fan_get`` treats as "not
    found HERE" — correct, because a tier that cannot be asked genuinely cannot answer, and the
    pin path's own capability gate refuses those bindings by name before any write.
    """
    getter = getattr(leg.store, "get", None) or getattr(leg.store, "get_fact", None)
    return getter if callable(getter) else None


def _artifact_reader(
    leg: TierLeg,
) -> Callable[[Namespace, str], Awaitable[list[MemoryItem]]] | None:
    reader = getattr(leg.store, "by_artifact", None)
    return reader if callable(reader) else None


def _first_leaf(group: BaseExceptionGroup[MemoryUniverseError]) -> MemoryUniverseError:
    """The first real exception out of a ``TaskGroup``'s ``ExceptionGroup``.

    A ``TaskGroup`` always wraps, so without this a consumer sees an ``ExceptionGroup`` where it
    is looking for ``TierRepositoryUnavailableError`` — and ``MemoryHealthService._walk``'s
    degrade branch, which catches exactly that class and nothing else, would never fire. This is
    the same unwrap ``ThreeChannelRecallRanker.rank`` performs, for the same reason.

    The caller re-raises it ``from leaf.__cause__``, never ``from None``: the leaf already
    carries the raw store error that ``guarded`` chained onto it, and ``from None`` would clear
    exactly that link — turning an honest translation into a discarded root cause.
    """
    leaf: BaseException = group.exceptions[0]
    while isinstance(leaf, BaseExceptionGroup):
        leaf = leaf.exceptions[0]
    if not isinstance(leaf, MemoryUniverseError):  # pragma: no cover - narrowed by ``except*``
        raise leaf
    return leaf
