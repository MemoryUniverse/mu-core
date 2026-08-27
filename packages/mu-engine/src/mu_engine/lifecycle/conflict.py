"""``ConflictAdjudicator`` — the LLM-judged supersession decision (S3-01; ADR 0037).

Authority: ``memory-lifecycle-manager-spec.md`` §8 DECISION 8 (lines 315-328, including the S1
adjudication-budget amendment), §13.2 fix #2 (lines 408-412, the MTM cross-store degrade wrap
folded in alongside this same root issue), AC-3.1/AC-3.2 (§14.1, lines 441-442);
``CANONICAL-CONTRACTS.md`` §7.20 (lines 822-832, ``ResolveConflictStage`` split from
``ReconcileConflictsStage`` + the ``ConflictResolutionPolicy`` product type). Routes through the
EXISTING seam ``mu_engine.providers.model_router.build_model_router`` -> ``ModelRouter.generate``
via the already-landed ``Task.CONFLICT_ADJUDICATION`` catalog entry (S0-05,
``tests/providers/test_conflict_adjudication_task.py``) — no new provider wiring, per ADR 0037's
own text ("no new adjudicator field is introduced").

**Placement (spec §8).** ``DistillPipeline``'s existing ``PolarityCardinalityHeuristic``
(``_contradicts``, ``distill.py:437-446``) becomes the **candidate gate**: the mem0-style
top-``candidate_k`` same-subject/predicate gather ``DistillPipeline._reconcile`` already runs
(minus the exact-identical NOOP match) is the survivor set handed to this adjudicator — NOT just
the subset the heuristic itself would have flagged as a direct/functional contradiction. This
matters for the "genuine semantic contradiction the heuristic alone would miss" acceptance item:
a non-functional predicate whose objects are still substantively incompatible passes the gate (it
is same-subject/predicate, non-identical) even though ``_contradicts()`` alone would call it
COEXIST — the LLM call sees it and can correctly override the heuristic's blind spot.
``_contradicts()`` keeps its OLD role too, but demoted to the **degrade floor** (spec §8 "Degrade
honesty"): the decision used when the LLM is disabled (``ConflictAdjudicatorSettings`` /
``DistillSettings.use_llm_adjudicator=False``), down (any exception from the router), or the
per-sweep budget (S1, AC-3.2) is spent.

**Async split (CANONICAL §7.20).** ``DistillPipeline._reconcile`` is the
``ReconcileConflictsStage``-equivalent: it ONLY gathers candidates + runs the free heuristic — no
model call, no write, ever (structurally impossible: it holds no reference to this module's
``ConflictAdjudicator``). ``ConflictAdjudicator.adjudicate`` + ``DistillPipeline._resolve`` are the
``ResolveConflictStage``-equivalent — the ONLY place that may await the (slow, HARD-tier,
``models.adjudicate_model``) LLM call, and the ONLY place that writes. A reconcile pass for the
whole window always completes before any resolve call is even issued (see ``distill.py:_distill``)
— reconcile is demonstrably never blocked waiting on an adjudication.

**Degrade honesty + never-fabricate (spec §8).** Every LLM-unavailable path (disabled, down,
budget-exhausted) falls through to the SAME named ``DegradeReason``s
(``CONFLICT_ADJUDICATION_DEGRADED`` for "was expected but unavailable this call",
``LLM_UNAVAILABLE_HEURISTIC`` for "deliberately configured off") the heuristic degrade floor
resolves. If even that deterministic floor cannot pick a winner — a bi-temporal ``valid_at`` TIE
between the incoming winner and a contradicting candidate, which the ORIGINAL pre-ADR-0037 code
silently broke in the winner's favor (``>`` not ``>=``) — this adjudicator refuses to guess:
``AdjudicationKind.PENDING`` + ``apply=False``, and (when a ``ConflictRecordRepository`` is wired)
a ``ConflictRecord(state=MANUAL_PENDING)`` is parked in the conflict-inbox (CANONICAL §7.20 C9)
instead. ``DistillPipeline`` with NO ``ConflictAdjudicator`` wired at all reproduces the ORIGINAL
tie-goes-to-the-winner behavior exactly (``_heuristic_only_verdict`` in ``distill.py``) — 100%
backward compatible with every pre-existing heuristic-only test.

**``ConflictResolutionPolicy`` (CANONICAL §7.20).** ``{AUTOMATIC, MANUAL} x {RECENCY, CONFIDENCE,
PROVENANCE}``. ``AUTOMATIC`` applies the verdict inline (the resolve stage above). ``MANUAL``
parks EVERY non-COEXIST verdict as a ``ConflictRecord`` instead of auto-applying it, regardless of
whether the verdict came from the LLM or the heuristic floor — "MANUAL parks a ConflictRecord in
the conflict-inbox" (spec §8 / CANONICAL §7.20), never auto-writes.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.events import DegradedModeEntered, DegradeReason
from mu_contracts.domain.model.conflict import ConflictRecord, ConflictState
from mu_contracts.ports.governance import ConflictRecordRepository
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.policy import LifecyclePolicy
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.pipelines.distill import asserted_later
from mu_engine.platform.clock import SystemClock
from mu_engine.providers._contracts import Completion, Message, MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace

__all__ = [
    "AdjudicationBudget",
    "AdjudicationKind",
    "AdjudicationVerdict",
    "ConflictAdjudicationPort",
    "ConflictAdjudicator",
    "ConflictAdjudicatorSettings",
    "ConflictEventPublisher",
    "ConflictResolutionPolicy",
    "InMemoryConflictRecordRepository",
    "ResolutionMode",
    "ResolutionStrategy",
    "build_conflict_adjudicator",
    "compute_conflict_id",
    "conflict_adjudicator_settings_from_lifecycle",
]

_log = structlog.get_logger("mu_engine.lifecycle.conflict")


# --------------------------------------------------------------------------------------- policy
class ResolutionMode(StrEnum):
    """CANONICAL §7.20: ``AUTOMATIC`` resolves inline in the async resolve stage; ``MANUAL`` parks
    a ``ConflictRecord`` in the conflict-inbox instead of auto-applying."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"


class ResolutionStrategy(StrEnum):
    """CANONICAL §7.20 tie-break axis (informational tag on the policy / ``ConflictRecord``
    snapshot; ``RECENCY`` is the one axis this adjudicator's degrade floor actually implements
    today — ``CONFIDENCE``/``PROVENANCE`` are reserved for a future ``ConflictResolutionPolicy``
    consumer, same "declared, not yet a second code path" discipline as spec §10's RESERVED
    states)."""

    RECENCY = "recency"
    CONFIDENCE = "confidence"
    PROVENANCE = "provenance"


class ConflictResolutionPolicy(BaseModel):
    """``{AUTOMATIC, MANUAL} x {RECENCY, CONFIDENCE, PROVENANCE}`` (CANONICAL §7.20)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ResolutionMode = ResolutionMode.AUTOMATIC
    strategy: ResolutionStrategy = ResolutionStrategy.RECENCY


# --------------------------------------------------------------------------------------- settings
class ConflictAdjudicatorSettings(BaseModel):
    """Adjudicator-local knobs (DEV-STANDARDS rule 3 tracked-seam pattern, mirrors
    ``ExtractionSettings``). ``adjudication_budget_per_sweep``/``adjudication_degrade_threshold_s``
    mirror ``LifecycleSettings``' own fields of the same name (S1, spec §16) — the composition
    root threads the ``LifecycleSettings`` values in here when it wires a real
    ``ConflictAdjudicator`` (tracked seam; not yet a ``Settings.lifecycle`` sibling, same as
    ``LifecycleSettings`` itself)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    adjudication_budget_per_sweep: int = Field(default=50, ge=0)
    adjudication_degrade_threshold_s: float = Field(default=30.0, ge=0.0)
    max_tokens: int = 256
    temperature: float = 0.0


def conflict_adjudicator_settings_from_lifecycle(
    lifecycle: LifecycleSettings,
) -> ConflictAdjudicatorSettings:
    """CONFIG-AND-DATA-FIX-PLAN.md §1.2 C3: builds the ``ConflictAdjudicatorSettings`` this
    module's ``build_conflict_adjudicator(..., settings=...)`` takes FROM the WIRED
    ``EngineSettings.lifecycle`` (``get_engine_settings().lifecycle``) instead of the composition
    root omitting ``settings=`` entirely (-> ``ConflictAdjudicator.__init__``'s own bare
    ``ConflictAdjudicatorSettings()`` fallback, ``:353`` — the exact orphan-knob shape this plan
    closes). A pure, side-effect-free mapping (no I/O, unit-testable without a container) so both
    ``mu_local.composition.LocalContainer`` and ``mu_engine_server.composition.EngineContainer``
    share ONE translation instead of duplicating it (DEV-STANDARDS rule 7).

    ``adjudication_budget_per_sweep``/``adjudication_degrade_threshold_s`` are the two fields
    ``LifecycleSettings`` already carried (mirrored across both classes by the ORIGINAL spec —
    ``ConflictAdjudicatorSettings``' own docstring); ``max_tokens``/``temperature`` are the two
    ``adjudicator_*``-prefixed fields C3 added to ``LifecycleSettings`` for exactly this purpose —
    every adjudicator knob is now reachable via the ONE ``MU_LIFECYCLE__…`` env family, never a
    second ``MU_CONFLICT_ADJUDICATOR__…`` prefix for the same conceptual budget.
    """
    return ConflictAdjudicatorSettings(
        adjudication_budget_per_sweep=lifecycle.adjudication_budget_per_sweep,
        adjudication_degrade_threshold_s=lifecycle.adjudication_degrade_threshold_s,
        max_tokens=lifecycle.adjudicator_max_tokens,
        temperature=lifecycle.adjudicator_temperature,
    )


# --------------------------------------------------------------------------------------- budget
class AdjudicationBudget:
    """Mutable per-sweep-tick counter (spec §8 S1 / AC-3.2). Deliberately NOT a pydantic model —
    it is a stateful runtime counter, not a value object (DEV-STANDARDS rule 2 targets domain
    DTOs; this mirrors the plain-class shape already used for ``InProcessWriterLease``'s internal
    lock table). Constructed fresh once per sweep tick via ``ConflictAdjudicator.new_budget()`` —
    NEVER reused across ``DistillPipeline.distill()`` calls, so a slow prior tick can't starve the
    next one and vice versa."""

    __slots__ = ("_clock", "_consumed", "_degrade_threshold_s", "_limit", "_started_at")

    def __init__(self, *, limit: int, degrade_threshold_s: float, clock: Clock) -> None:
        self._limit = limit
        self._degrade_threshold_s = degrade_threshold_s
        self._clock = clock
        self._consumed = 0
        self._started_at = clock.now()

    @property
    def consumed(self) -> int:
        return self._consumed

    def has_capacity(self) -> bool:
        """False once EITHER the call-count cap OR the wall-clock budget (Clock-measured, never
        ``time.time()``) is exhausted — the two S1 knobs, either one ends the tick's LLM spend."""
        if self._consumed >= self._limit:
            return False
        elapsed_s = (self._clock.now() - self._started_at).total_seconds()
        return elapsed_s < self._degrade_threshold_s

    def consume(self) -> None:
        """Reserves one call's worth of budget. Called BEFORE the LLM call is issued (spec §8: a
        call that is attempted — even if it then fails — still spent quota/time)."""
        self._consumed += 1


# --------------------------------------------------------------------------------------- verdict
class AdjudicationKind(StrEnum):
    """The adjudicator's decision space for ONE (winner, candidate) pair — a strict subset of
    ``DistillPipeline``'s own ``DistillActionKind`` vocabulary (ADD/NOOP are decided upstream of
    this call, never here) PLUS ``PENDING`` (a genuinely undecidable tie, never returned by the
    LLM path — only the heuristic degrade floor can reach it). Deliberately NOT imported from
    ``mu_engine.pipelines.distill`` to avoid a distill.py<->conflict.py import cycle (distill.py
    imports THIS module, not the reverse) — ``DistillPipeline._resolve`` maps this vocabulary onto
    ``DistillActionKind`` at the one call site that needs both."""

    SUPERSEDE = "supersede"  # the winner replaces the candidate (candidate invalidated)
    SELF_EXPIRE = "self_expire"  # the candidate is more authoritative; the winner self-expires
    COEXIST = "coexist"  # no real contradiction; both stay active
    PENDING = "pending"  # neither the LLM nor the heuristic floor could decide — never a guess


class AdjudicationVerdict(BaseModel):
    """One adjudication outcome (spec §8). ``apply=False`` means "do not write anything for this
    pair" — either it was parked (``conflict_record`` set, MANUAL policy or an undecidable tie
    under AUTOMATIC) or is otherwise withheld; the caller MUST treat ``apply=False`` identically
    to COEXIST at the storage layer (no invalidate, no state flip) — never fabricate a winner."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    kind: AdjudicationKind
    apply: bool
    used_llm: bool
    #: True iff the automatic winner-picker was REFUSED because the item it would have made the
    #: loser is ``pinned`` (memory-health §6.4; CANONICAL §7.17 item 4a(b)). ``kind`` is forced to
    #: ``PENDING`` and ``apply`` to ``False`` in that case, so the pinned item stays ACTIVE and
    #: the conflict is PARKED rather than lost.
    pin_blocked: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    model_id: str | None = None
    conflict_record: ConflictRecord | None = None


# --------------------------------------------------------------------------------------- ports
@runtime_checkable
class ConflictAdjudicationPort(Protocol):
    """The narrow slice of ``ModelRouter`` this adjudicator depends on (DEV-STANDARDS rule 6
    strategy/port pattern, mirrors ``services.extract.FactExtractorPort`` depending on the
    narrower ``LLMProviderPort`` rather than the concrete router). Structurally satisfied by
    ``mu_engine.providers.model_router.ModelRouter.generate`` — routes
    ``Task.CONFLICT_ADJUDICATION`` to ``models.adjudicate_model`` (the HARD tier; local SLM
    excluded, ADR 0037) via the SAME ``TaskClassMapper`` table every other ``Task`` uses (S0-05)."""

    async def generate(
        self,
        task: Task,
        messages: list[Message],
        *,
        override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: str | None = None,
    ) -> Completion: ...


@runtime_checkable
class ConflictEventPublisher(Protocol):
    """Content-free publish seam (mirrors ``pipelines.distill.EventPublisher`` exactly — a
    separate declaration, not an import, to keep this module's only dependency direction on
    ``distill.py`` at zero: ``distill.py`` imports this module, never the reverse)."""

    async def publish(self, event: DegradedModeEntered) -> None: ...


class InMemoryConflictRecordRepository:
    """A REAL in-process ``ConflictRecordRepository`` (LOCAL-plane default) — same "legitimate
    in-process adapter, not a test mock" pattern as ``pipelines.distill.InProcessWriterLease`` /
    ``pipelines.ledger.InMemoryStageLedger``. A durable (SQLite/Postgres) adapter is the SHARED/
    persistent-client concern (spec §2.9 relational mirror); this is the sanctioned LOCAL default
    the composition root wires when no durable store is configured."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], ConflictRecord] = {}

    async def add(self, record: ConflictRecord) -> None:
        self._records[(record.namespace.to_prefix(), record.conflict_id)] = record

    async def get(self, ns: Namespace, conflict_id: str) -> ConflictRecord | None:
        return self._records.get((ns.to_prefix(), conflict_id))

    async def upsert(self, record: ConflictRecord) -> None:
        await self.add(record)

    async def pending(self, ns: Namespace) -> list[ConflictRecord]:
        prefix = ns.to_prefix()
        return [
            r
            for (p, _), r in self._records.items()
            if p == prefix and r.state is ConflictState.MANUAL_PENDING
        ]


def compute_conflict_id(ns: Namespace, member_ids: tuple[str, ...], predicate_key: str) -> str:
    """``sha256(ns.to_prefix|sorted(member_ids)|predicate_key)[:24]`` (``ConflictRecord`` docstring,
    ``mu_contracts.domain.model.conflict`` — idempotent, same conflict re-detected -> same id)."""
    key = f"{ns.to_prefix()}|{'|'.join(sorted(member_ids))}|{predicate_key}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


# --------------------------------------------------------------------------------------- prompt
_ADJUDICATE_SYSTEM = (
    "You are a memory-conflict adjudicator for a personal-memory system. You are given an "
    "EXISTING recorded proposition and a NEW incoming proposition about the same subject and "
    "predicate. Decide the relationship between them:\n"
    '- "supersede": the NEW proposition replaces/contradicts the EXISTING one (the existing '
    "fact is no longer true).\n"
    '- "self_expire": the EXISTING proposition is actually the more current/authoritative one; '
    "the NEW one should be treated as stale instead.\n"
    '- "coexist": both can be true at once; there is no real contradiction.\n'
    'Respond with ONLY a JSON object: {"verdict": "supersede"|"self_expire"|"coexist", '
    '"confidence": <0.0-1.0>, "reason": "<short reason>"}.'
)


def _build_prompt(winner: MemoryItem, candidate: MemoryItem) -> str:
    """Deliberately content-bearing (a model prompt, not a ``DomainEvent``/``ConflictRecord`` —
    the content-free rule binds those two, never an LLM prompt; mirrors
    ``LlmFactExtractor.extract`` sending raw text to the model)."""
    return (
        f"EXISTING (recorded_at={_valid_at(candidate).isoformat()}): "
        f"{candidate.subject or ''} {candidate.predicate or ''} {candidate.object or ''} "
        f"-- {candidate.content!r}\n"
        f"NEW (recorded_at={_valid_at(winner).isoformat()}): "
        f"{winner.subject or ''} {winner.predicate or ''} {winner.object or ''} "
        f"-- {winner.content!r}"
    )


def _parse_verdict(raw: str) -> tuple[AdjudicationKind, float, str]:
    """Parse the model's ``{"verdict": ..., "confidence": ..., "reason": ...}`` JSON, tolerant of
    code fences (mirrors ``services.extract._parse_mem0_facts``). Any malformed/unknown verdict
    raises (``ValueError``/``KeyError``/``json.JSONDecodeError``) — the caller treats that
    identically to a model outage (degrade to the heuristic floor), never a fabricated guess."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", cleaned).strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("adjudicator response is not a JSON object")
    verdict_raw = str(payload["verdict"]).strip().lower()
    kind = AdjudicationKind(verdict_raw)  # raises ValueError on "pending" or any unknown token
    if kind is AdjudicationKind.PENDING:
        raise ValueError("the LLM adjudicator may not itself return PENDING")
    confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5))))
    reason = str(payload.get("reason") or "llm_adjudicated")
    return kind, confidence, reason


def _valid_at(item: MemoryItem) -> datetime:
    """A fact's world-time validity start (mirrors ``pipelines.distill._valid_at`` exactly —
    duplicated rather than imported to keep this module's dependency on ``distill.py`` at zero)."""
    if item.valid_at is None:  # defensive — promotion always sets it
        return item.created_at
    return item.valid_at


# --------------------------------------------------------------------------------------- service
class ConflictAdjudicator:
    """LLM-judged supersession over ``Task.CONFLICT_ADJUDICATION`` (spec §8). See module docstring
    for the full placement/degrade/policy contract."""

    def __init__(
        self,
        *,
        router: ConflictAdjudicationPort | None = None,
        policy: ConflictResolutionPolicy | None = None,
        settings: ConflictAdjudicatorSettings | None = None,
        clock: Clock | None = None,
        bus: ConflictEventPublisher | None = None,
        conflict_records: ConflictRecordRepository | None = None,
    ) -> None:
        self._router = router
        self._policy = policy or ConflictResolutionPolicy()
        self._settings = settings or ConflictAdjudicatorSettings()
        self._clock: Clock = clock or SystemClock()
        self._bus = bus
        self._conflict_records = conflict_records
        # The central FSM/pin guard (memory-health §6.1). Consulted — never re-decided here — so
        # the "a pinned item is never the auto-loser" rule has exactly one owner.
        self._lifecycle = LifecyclePolicy()

    def new_budget(self) -> AdjudicationBudget:
        """One fresh budget per sweep tick — ``DistillPipeline._distill`` calls this ONCE per
        ``distill()`` invocation, never per-candidate (AC-3.2's "a sweep tick" IS one ``distill()``
        call over one window)."""
        return AdjudicationBudget(
            limit=self._settings.adjudication_budget_per_sweep,
            degrade_threshold_s=self._settings.adjudication_degrade_threshold_s,
            clock=self._clock,
        )

    async def adjudicate(
        self,
        *,
        ns: Namespace,
        winner: MemoryItem,
        candidate: MemoryItem,
        heuristic_contradicts: bool,
        budget: AdjudicationBudget,
    ) -> AdjudicationVerdict:
        """One (winner, candidate) pair's verdict. Never raises on an LLM failure — every failure
        mode (disabled, down, budget-exhausted, unparseable) degrades to the deterministic
        heuristic floor + a named ``DegradedModeEntered`` (spec §8 "Degrade honesty"), and the only
        further-undecidable case (a bi-temporal tie) parks rather than guesses."""
        used_llm = False
        model_id: str | None = None
        kind: AdjudicationKind
        confidence: float
        reason: str

        if self._router is None:
            await self._emit_degrade(
                DegradeReason.LLM_UNAVAILABLE_HEURISTIC, detail="adjudicator_disabled"
            )
            kind, confidence, reason = self._heuristic_verdict(
                winner, candidate, heuristic_contradicts
            )
        elif not budget.has_capacity():
            await self._emit_degrade(
                DegradeReason.CONFLICT_ADJUDICATION_DEGRADED, detail="budget_exhausted"
            )
            kind, confidence, reason = self._heuristic_verdict(
                winner, candidate, heuristic_contradicts
            )
        else:
            budget.consume()  # reserved before the call — a failed attempt still spent quota (§8)
            try:
                completion = await self._router.generate(
                    Task.CONFLICT_ADJUDICATION,
                    [
                        Message(role=MessageRole.SYSTEM, content=_ADJUDICATE_SYSTEM),
                        Message(role=MessageRole.USER, content=_build_prompt(winner, candidate)),
                    ],
                    max_tokens=self._settings.max_tokens,
                    temperature=self._settings.temperature,
                    response_format="json_object",
                )
                kind, confidence, reason = _parse_verdict(completion.text)
                used_llm = True
                model_id = completion.model_id
            except Exception as exc:  # ModelGroupUnavailableError, parse failure, etc. — no guess
                await self._emit_degrade(
                    DegradeReason.CONFLICT_ADJUDICATION_DEGRADED, detail=type(exc).__name__
                )
                kind, confidence, reason = self._heuristic_verdict(
                    winner, candidate, heuristic_contradicts
                )

        # memory-health §6.4 — the pinned side is NEVER the automatic loser (CANONICAL §7.17
        # item 4a(b)). Applied AFTER the verdict is formed, deliberately: the adjudicator still
        # records what it thought, so the conflict is parked with an honest proposed winner
        # instead of being silently dropped.
        pin_blocked = self._pin_blocks_loser(winner, candidate, kind)
        if pin_blocked:
            kind = AdjudicationKind.PENDING
            reason = "pin_blocked_pinned_item_is_never_the_auto_loser"

        apply = self._decide_apply(kind)
        conflict_record: ConflictRecord | None = None
        if not apply:
            conflict_record = await self._park(
                ns=ns,
                winner=winner,
                candidate=candidate,
                kind=kind,
                confidence=confidence,
                used_llm=used_llm,
                pin_blocked=pin_blocked,
            )
        return AdjudicationVerdict(
            kind=kind,
            apply=apply,
            used_llm=used_llm,
            confidence=confidence,
            reason=reason,
            model_id=model_id,
            conflict_record=conflict_record,
            pin_blocked=pin_blocked,
        )

    def _pin_blocks_loser(
        self, winner: MemoryItem, candidate: MemoryItem, kind: AdjudicationKind
    ) -> bool:
        """Would applying ``kind`` drive a PINNED item out of ``ACTIVE`` automatically?

        ``COEXIST``/``PENDING`` assert no loser at all, so there is nothing to block.

        **The loser is NOT read off ``kind``.** ``DistillPipeline._resolve`` — the only caller,
        and the only writer of ``state=SUPERSEDED`` in the engine — deliberately discards this
        verdict's SELF_EXPIRE-vs-SUPERSEDE polarity and re-derives the direction from ASSERTION
        recency (``distill.asserted_later``; the BUG1 fix, because the small adjudicator model
        inverts the EXISTING-vs-NEW framing often enough to have been live-reproduced). Keying
        the pin check on ``kind`` would therefore consult the pin of an item that is not the one
        about to be written, and a pinned memory would be superseded whenever the two disagreed.
        The SAME predicate the writer uses is used here, so the two can never diverge.

        The pin answer itself is the CENTRAL guard's, not this class's: ``LifecyclePolicy.permits``
        owns the "pinned ⇒ no automatic adjudicated exit" rule, so an explicit, user-driven
        supersede of a pinned item stays legal exactly as CANONICAL §7.10 requires. This check is
        a BACK-STOP that makes the parked ``ConflictRecord`` honest (``pin_blocked``, which the
        health view projects); the load-bearing refusal is the writer's own, at the write site.
        """
        if kind not in (AdjudicationKind.SUPERSEDE, AdjudicationKind.SELF_EXPIRE):
            return False
        loser = winner if asserted_later(candidate, winner) else candidate
        return not self._lifecycle.permits(loser, MemoryState.SUPERSEDED)

    def _decide_apply(self, kind: AdjudicationKind) -> bool:
        if kind is AdjudicationKind.PENDING:
            return False  # genuinely undecidable — never a guess, regardless of policy
        if kind is AdjudicationKind.COEXIST:
            return True  # nothing to park — no winner/loser relationship was even asserted
        return self._policy.mode is ResolutionMode.AUTOMATIC

    def _heuristic_verdict(
        self, winner: MemoryItem, candidate: MemoryItem, contradicts: bool
    ) -> tuple[AdjudicationKind, float, str]:
        """The deterministic degrade floor (``PolarityCardinalityHeuristic``'s OLD final-verdict
        role, spec §8). Unlike the pre-ADR-0037 code (which broke a tie toward the winner via a
        bare ``>`` comparison), a genuine tie here is PENDING — "even the heuristic can't decide"
        (spec §8) — never a silent default winner.

        BUG1 FIX (data-quality re-assessment §2 "SUPERSESSION WINNER INVERTED"): compares
        ``created_at`` (ASSERTION recency — the source STM message's real capture instant), NEVER
        ``valid_at`` (the extracted bi-temporal world-time). ``distill.py``'s own ``_resolve``
        re-derives this SAME direction from ``created_at`` unconditionally regardless of what this
        method returns (never trusts this method's SELF_EXPIRE/SUPERSEDE polarity blind) — this
        method is kept consistent with that authoritative rule so its OWN return value (``kind``,
        used for e.g. the parked ``ConflictRecord``'s ``proposed_winner_id``) is never itself a
        second, diverging source of truth."""
        if not contradicts:
            return AdjudicationKind.COEXIST, 1.0, "heuristic_no_contradiction"
        c_at, w_at = candidate.created_at, winner.created_at
        if c_at > w_at:
            return AdjudicationKind.SELF_EXPIRE, 1.0, "heuristic_candidate_more_recent"
        if w_at > c_at:
            return AdjudicationKind.SUPERSEDE, 1.0, "heuristic_winner_more_recent"
        return AdjudicationKind.PENDING, 0.0, "heuristic_validity_tie_undecidable"

    async def _park(
        self,
        *,
        ns: Namespace,
        winner: MemoryItem,
        candidate: MemoryItem,
        kind: AdjudicationKind,
        confidence: float,
        used_llm: bool,
        pin_blocked: bool = False,
    ) -> ConflictRecord:
        proposed_winner_id = (
            winner.id
            if kind is AdjudicationKind.SUPERSEDE
            else candidate.id
            if kind is AdjudicationKind.SELF_EXPIRE
            else None
        )
        member_ids = (winner.id, candidate.id)
        record = ConflictRecord(
            conflict_id=compute_conflict_id(ns, member_ids, winner.predicate or ""),
            namespace=ns,
            member_ids=member_ids,
            predicate_key=winner.predicate or "",
            method="llm_adjudicator" if used_llm else "polarity_cardinality_heuristic",
            detected_confidence=confidence,
            proposed_winner_id=proposed_winner_id,
            state=ConflictState.MANUAL_PENDING,
            detected_at=self._clock.now(),
            pin_blocked=pin_blocked,
        )
        if self._conflict_records is not None:
            try:
                await self._conflict_records.add(record)
            except Exception:  # best-effort inbox write — the correct non-apply decision already
                # holds regardless (no fabricated winner either way); log loud, never crash the
                # sweep over the audit surface (DEV-STANDARDS rule 8 — logged, not silent).
                _log.error("conflict_record_park_failed", conflict_id=record.conflict_id)
        return record

    async def _emit_degrade(self, reason: DegradeReason, *, detail: str) -> None:
        if self._bus is None:
            return
        await self._bus.publish(
            DegradedModeEntered(
                component="conflict", mode="heuristic_only", reason=reason, detail=detail
            )
        )


def build_conflict_adjudicator(
    *,
    use_llm: bool,
    router: ConflictAdjudicationPort | None = None,
    policy: ConflictResolutionPolicy | None = None,
    settings: ConflictAdjudicatorSettings | None = None,
    clock: Clock | None = None,
    bus: ConflictEventPublisher | None = None,
    conflict_records: ConflictRecordRepository | None = None,
) -> ConflictAdjudicator | None:
    """Resolve the configured adjudicator (DEV-STANDARDS rule 6 strategy selection; mirrors
    ``services.extract.build_extractor``'s shape/fail-loud contract exactly).

    ``use_llm=False`` (``DistillSettings.use_llm_adjudicator=False`` / ``LifecycleSettings
    .use_llm_adjudicator=False``) -> ``None``: ``DistillPipeline`` then uses its OWN
    ``_heuristic_only_verdict`` degrade floor (100% pre-ADR-0037 backward-compatible — ties go to
    the winner), not this module's stricter tie-is-PENDING floor. ``use_llm=True`` requires a
    wired ``ConflictAdjudicationPort`` (``ModelRouter.generate``) — a missing router is a wiring
    bug and raises, never a silent downgrade.
    """
    if not use_llm:
        return None
    if router is None:
        raise ValueError("LLM adjudicator selected (use_llm=True) but no router wired")
    return ConflictAdjudicator(
        router=router,
        policy=policy,
        settings=settings,
        clock=clock,
        bus=bus,
        conflict_records=conflict_records,
    )
