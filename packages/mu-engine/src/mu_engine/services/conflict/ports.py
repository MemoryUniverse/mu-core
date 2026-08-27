"""The conflict lane's ports, plus their LOCAL in-process adapters.

Authority: ``conflict-resolution-async-design.md`` §4.1 (lines 150-160 — the two policy
lookups the precedence chain reads), §5 (lines 168-220 — hydration at render time, and
*"enqueues ``ResolveConflictStage``"*), proposed contract change 5 (line 312).

**Why these live in ``mu_engine`` and not ``mu_contracts/ports/conflict.py``.** Every one of them
is parameterised by ``ConflictResolutionPolicy``, which is an ENGINE type
(``mu_engine.lifecycle.conflict``). ``mu_contracts`` may not import ``mu_engine`` — the
``contracts-imports-nothing-in-project`` import-linter contract is a CI gate — so a policy port
in ``mu_contracts/ports/`` is structurally impossible without first moving the policy type, which
would break every existing ``from mu_engine.lifecycle.conflict import ConflictResolutionPolicy``
call site in another lane's composition root. Recorded as a delta against spec line 312, which
places all the new conflict ports in ``ports/conflict.py``.

**Every port here ships a real adapter.** No ``NotImplementedError`` stubs (house rule): the
in-memory adapters are the sanctioned LOCAL-plane defaults, the same "legitimate in-process
adapter, not a test mock" pattern as ``InMemoryConflictRecordRepository`` and
``InProcessWriterLease``. The durable SHARED-plane counterparts (a control-plane policy row, the
SQLite outbox for the resolve queue) are composition-root concerns in the client/server repos.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mu_contracts.domain.model.conflict import (
    MERGED_TEXT_REF_PATTERN,
    ConflictRecord,
    ConflictResolutionKind,
    ContentFreeModel,
    MergedTextRef,
)
from mu_contracts.domain.model.memory import Namespace
from mu_engine.lifecycle.conflict import ConflictResolutionPolicy

__all__ = [
    "MERGED_TEXT_REF_PATTERN",
    "ConflictMemberHydration",
    "ConflictMemberHydrator",
    "InMemoryConflictResolutionQueue",
    "InMemoryMemoryConflictPolicyStore",
    "InMemoryNamespaceConflictPolicyStore",
    "MemoryConflictPolicyStore",
    "MergedTextRef",
    "NamespaceConflictPolicyStore",
    "RecordBackedResolutionQueue",
    "ResolutionIntent",
    "ResolutionQueue",
    "UnappliedConflictRecordReader",
    "WritableMemoryConflictPolicyStore",
    "intent_from_record",
]


# ------------------------------------------------------------------------------------ policy
@runtime_checkable
class MemoryConflictPolicyStore(Protocol):
    """§4.1 step 1 — the PER-MEMORY override, the most specific rule there is.

    ``override_for`` answers for ONE memory id and returns ``None`` when that memory carries no
    override, which is the overwhelmingly common case. The resolver asks it for every member of
    the conflict, so a MANUAL override on either side is enough to make the whole conflict
    manual: the user who hand-curated one of the two facts is exactly the user who must not have
    the other one silently supersede it.

    **Storage delta (reported).** Spec line 154 puts this on the item itself, as a
    ``conflict_policy_ref`` on ``MemoryItem``/``MemoryKind``/tag. ``MemoryItem`` has no such
    field and lives in the storage lane's files, so the lookup is a port here instead of a
    column read. The PRECEDENCE logic — the part that is easy to get quietly wrong — is
    independent of which of the two it is.
    """

    async def override_for(
        self, ns: Namespace, memory_id: str
    ) -> ConflictResolutionPolicy | None: ...


@runtime_checkable
class WritableMemoryConflictPolicyStore(MemoryConflictPolicyStore, Protocol):
    """A :class:`MemoryConflictPolicyStore` that the §5 line 216 policy surface can also WRITE.

    Split from the read port deliberately: the §4.1 resolver needs only ``override_for`` and must
    not be handed a writer it could accidentally use, while the resolve service needs both. The
    alternative — one port plus a ``hasattr`` check at the call site — would move a capability
    question out of the type system and into runtime, which is precisely the "silent stub"
    shape the house rules forbid.
    """

    async def set_override(
        self, ns: Namespace, memory_id: str, policy: ConflictResolutionPolicy | None
    ) -> None: ...


@runtime_checkable
class NamespaceConflictPolicyStore(Protocol):
    """§4.1 step 2 — the PER-NAMESPACE policy ("the primary knob the owner asked for", line 155).

    A control-plane row on SHARED, daemon settings on LOCAL; ``None`` means the namespace has
    never been configured and the workspace/global default applies.
    """

    async def policy_for(self, ns: Namespace) -> ConflictResolutionPolicy | None: ...

    async def set_policy(self, ns: Namespace, policy: ConflictResolutionPolicy) -> None: ...


class InMemoryNamespaceConflictPolicyStore:
    """The LOCAL-plane :class:`NamespaceConflictPolicyStore` (daemon settings, in process).

    Keyed on ``ns.to_prefix()`` — namespace-scoped like every other store access (CLAUDE.md rule
    4), so a policy set on one user's partition can never be read from another's.
    """

    def __init__(self) -> None:
        self._by_prefix: dict[str, ConflictResolutionPolicy] = {}

    async def policy_for(self, ns: Namespace) -> ConflictResolutionPolicy | None:
        return self._by_prefix.get(ns.to_prefix())

    async def set_policy(self, ns: Namespace, policy: ConflictResolutionPolicy) -> None:
        self._by_prefix[ns.to_prefix()] = policy


class InMemoryMemoryConflictPolicyStore:
    """The LOCAL-plane :class:`MemoryConflictPolicyStore` (per-memory overrides, in process)."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], ConflictResolutionPolicy] = {}

    async def override_for(self, ns: Namespace, memory_id: str) -> ConflictResolutionPolicy | None:
        return self._by_key.get((ns.to_prefix(), memory_id))

    async def set_override(
        self, ns: Namespace, memory_id: str, policy: ConflictResolutionPolicy | None
    ) -> None:
        """``None`` CLEARS the override (spec line 216's ``PUT .../conflict-policy`` "set/clear")
        — the memory falls back to its namespace's policy rather than to a frozen copy of it."""
        key = (ns.to_prefix(), memory_id)
        if policy is None:
            self._by_key.pop(key, None)
            return
        self._by_key[key] = policy


# ------------------------------------------------------------------------------- hydration
class ConflictMemberHydration(BaseModel):
    """What the inbox needs about ONE member that the content-free record cannot carry.

    Deliberately a narrow DTO rather than a whole ``MemoryItem``: the inbox needs a body, a
    tier, a validity stamp and a source label, and handing the projector a full item would let
    far more than that leak into a render-only view by accident.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    tier: str = Field(min_length=1)
    valid_at: str | None = None  # ISO-8601; parsed by the projector, kept transport-plain here
    valid_at_inferred: bool = False
    provenance_id: str = Field(min_length=1)
    source_label: str | None = None
    pinned: bool = False


@runtime_checkable
class ConflictMemberHydrator(Protocol):
    """§5 line 176 / §3 line 107 — hydrate bodies BY ID at render time, from the owning store.

    Bounded by the caller's id set exactly like ``ConflictEdgeReader.edges_for``: the inbox
    renders one page, so this is asked once for that page's members and never scans a partition.
    Ids that cannot be hydrated are simply absent from the result — the projector renders the
    conflict without them rather than failing the whole inbox (spec's named-marker doctrine).
    """

    async def hydrate(
        self, ns: Namespace, memory_ids: frozenset[str]
    ) -> dict[str, ConflictMemberHydration]: ...


# ----------------------------------------------------------------------------- resolve queue
class ResolutionIntent(ContentFreeModel):
    """The durable record of a decision that has been ACCEPTED but not yet APPLIED (spec line
    218: *"A resolve action **does not execute inline** ... and **enqueues**
    ``ResolveConflictStage`` ... The HTTP call returns immediately"*).

    Content-free, and now ENFORCED rather than asserted. This DTO travels further than any other
    in the lane — it is the payload handed to a durable queue (the SQLite outbox on the client),
    it is logged, and §7 reflects the decision into a sync delta — yet it was the ONE conflict
    DTO that did not inherit :class:`ContentFreeModel`, and the lane's own content-free test
    parametrized every travelling DTO except this one. ``merged_text_ref`` is a REFERENCE to a
    composed draft in the owning store (spec line 212), never the draft's text, and
    :data:`MergedTextRef` makes that a check instead of a promise.
    """

    conflict_id: str = Field(min_length=1)
    namespace: Namespace
    kind: ConflictResolutionKind
    winner_id: str | None = None
    loser_ids: tuple[str, ...] = ()
    merged_text_ref: MergedTextRef | None = None
    resolved_by: str = Field(min_length=1)

    @model_validator(mode="after")
    def _losers_require_a_winner(self) -> ResolutionIntent:
        """A winner-less intent may name NO losers, and every loser must not be the winner.

        This is the invariant a live defect violated: a ``KEEP_BOTH`` decision — which means
        *"both remain active"* (``ConflictResolutionKind.KEEP_BOTH``, spec §5 line 211) — was
        enqueued with ``winner_id=None`` and ``loser_ids=(a, b)``, because the losers were
        computed as "every member that is not the winner". ``ResolveConflictStage``'s whole job
        is to supersede ``loser_ids``, so honouring that intent would invalidate BOTH items from
        a decision that means keep both. The service no longer enqueues ``KEEP_BOTH`` at all
        (nothing to apply); this validator is the structural back-stop, so no future caller can
        rebuild the same shape.
        """
        if self.winner_id is None and self.loser_ids:
            raise ValueError("a winner-less resolution intent cannot name losers")
        if self.winner_id is not None and self.winner_id in self.loser_ids:
            raise ValueError("the winner cannot also be a loser")
        if self.kind is ConflictResolutionKind.MERGE and self.merged_text_ref is None:
            raise ValueError("a merge intent requires a merged_text_ref")
        return self


@runtime_checkable
class ResolutionQueue(Protocol):
    """The hand-off from the request path to ``ResolveConflictStage``.

    This is the seam that keeps the §1 invariant true for MANUAL resolution: the caller's request
    ends at ``enqueue``, and the cross-store ``IdempotentWriteScope`` supersession happens later
    on the background worker under the writer lease. A resolve service holding a writer would be
    a SECOND write path, which §2 line 54 forbids ("never a third lease-free write path").
    """

    async def enqueue(self, intent: ResolutionIntent) -> None: ...


@runtime_checkable
class UnappliedConflictRecordReader(Protocol):
    """A ``ConflictRecordRepository`` that can also answer "what has been decided but not yet
    applied?" — ``mu_engine.lifecycle.conflict.awaits_apply`` over the durable records.

    Declared HERE, structurally, rather than added to
    ``mu_contracts.ports.governance.ConflictRecordRepository``: that port is another lane's file.
    Both shipped adapters (``InMemoryConflictRecordRepository``,
    ``RedisConflictRecordRepository``) implement ``awaiting_apply`` and therefore satisfy this
    Protocol without inheritance.
    """

    async def awaiting_apply(self, ns: Namespace) -> list[ConflictRecord]: ...


class RecordBackedResolutionQueue:
    """The RECOVERABLE :class:`ResolutionQueue` — its durability is the ``ConflictRecord`` itself.

    **The failure this exists to remove.** ``ConflictResolutionService.resolve`` moves the record
    to a terminal state and hands the apply-intent to a queue. With
    :class:`InMemoryConflictResolutionQueue` — a plain dict — the intent dies with the process,
    while the record durably says ``RESOLVED``. The conflict FSM then routes ``RESOLVED`` only to
    ``REOPENED`` and ``ACTIONABLE_STATES`` excludes it, so nothing can re-drive the decision and
    both contending items stay ``state='active'`` forever. A record that SAYS resolved while
    nothing was applied, unreachable from every surface, is worse than an unresolved conflict.

    So this queue stores nothing of its own. ``enqueue`` is a no-op **by construction, not by
    omission**: by the time the service calls it, the decision is already durable on the record
    (``resolution_kind`` + ``resolved_winner_id`` + ``resolution_applied_at is None``), which
    spec line 218 names as *"the durable intent"*. ``drain`` RE-DERIVES the intents from those
    records, so a crash anywhere between the decision and the apply costs a delay, never a
    decision. The worker closes the loop with ``ConflictResolutionService.mark_applied``, which
    stamps ``resolution_applied_at`` and drops the record out of this query — the same
    at-least-once + idempotent-apply discipline the outbox uses everywhere else.

    ``InMemoryConflictResolutionQueue`` remains the in-process fast path for a single-process
    FULL-LOCAL daemon that drains within the same run; this is what a composition root wires when
    the decision must survive the process, and the two compose (a worker can drain both).
    """

    def __init__(self, records: UnappliedConflictRecordReader) -> None:
        self._records = records

    async def enqueue(self, intent: ResolutionIntent) -> None:
        """Deliberately does nothing: the record written immediately before this call IS the
        queue entry. Kept on the :class:`ResolutionQueue` seam so a composition root can swap
        this in for the in-process queue with no change at the call site."""

    async def drain(self, ns: Namespace) -> tuple[ResolutionIntent, ...]:
        """Every decision in ``ns`` that still needs applying, oldest decision first.

        Non-destructive: a record leaves this set only when ``mark_applied`` stamps it, so an
        intent handed out and then lost to a crash is handed out again on the next drain.
        """
        records = sorted(
            await self._records.awaiting_apply(ns),
            key=lambda r: (r.resolved_at or r.detected_at, r.conflict_id),
        )
        return tuple(intent_from_record(r) for r in records)


def intent_from_record(record: ConflictRecord) -> ResolutionIntent:
    """Rebuild the apply-intent from the durable record — the ONE place that mapping lives.

    ``ConflictResolutionService.resolve`` builds the same shape for the in-process queue; both go
    through this function so a recovered intent and a freshly-enqueued one can never disagree
    about who the losers are.
    """
    if record.resolution_kind is None:
        raise ValueError("a queued resolution must name a resolution kind")
    winner_id = record.resolved_winner_id
    losers = () if winner_id is None else tuple(i for i in record.member_ids if i != winner_id)
    return ResolutionIntent(
        conflict_id=record.conflict_id,
        namespace=record.namespace,
        kind=record.resolution_kind,
        winner_id=winner_id,
        loser_ids=losers,
        merged_text_ref=record.merged_text_ref,
        resolved_by=record.resolved_by or "unknown",
    )


class InMemoryConflictResolutionQueue:
    """The LOCAL-plane :class:`ResolutionQueue` — an in-process, FIFO, namespace-scoped queue.

    Idempotent on ``conflict_id``: re-submitting the same decision (a retried HTTP call, a
    replayed IPC frame) replaces the pending intent rather than enqueueing a second apply, so a
    double-click cannot produce two supersessions. Later decisions on the same conflict legally
    supersede earlier ones — the record's ``resolution_*`` fields are the durable intent (spec
    line 218) and the last one written is what the stage should apply.

    ``drain`` is what a worker calls; it empties the queue and returns the batch in insertion
    order, so a caller can neither lose an intent it was handed nor see one twice.
    """

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], ResolutionIntent] = {}

    async def enqueue(self, intent: ResolutionIntent) -> None:
        self._pending[(intent.namespace.to_prefix(), intent.conflict_id)] = intent

    async def drain(self, ns: Namespace | None = None) -> tuple[ResolutionIntent, ...]:
        prefix = None if ns is None else ns.to_prefix()
        taken = [key for key in self._pending if prefix is None or key[0] == prefix]
        return tuple(self._pending.pop(key) for key in taken)

    def __len__(self) -> int:
        return len(self._pending)
