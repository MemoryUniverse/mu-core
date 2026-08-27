"""``ConflictResolutionService`` — the §5 write actions, structurally off the write path.

Authority: ``conflict-resolution-async-design.md`` §5 (lines 207-220 — the resolve/reopen/policy
actions and *"A resolve action **does not execute inline**"*), §3.1 (the FSM it routes every
state change through), §4.1 (the policy setters), §8 (the events it emits).

**The invariant this class is shaped to keep.** §1 invariant 1 says the writer never waits for
conflict work, and §2 line 54 forbids *"a third lease-free write path"*. So this service holds no
memory-write port, no tier repository, no writer lease and no model router — it CANNOT supersede
anything, in the same structural way ``MemoryHealthService`` cannot reinforce anything. What it
does is: authorize, validate the FSM edge, record the durable INTENT on the ``ConflictRecord``,
enqueue it for ``ResolveConflictStage``, emit the event, return. The cross-store
``IdempotentWriteScope`` supersession happens later, on the background worker, under the writer
lease (§7.5).

**Which surface this is, and which it is not.** The three read/write FACES in §5 lines 202-217
(REST ``/v1/conflicts``, daemon IPC ``/conflicts`` + ``mu conflicts``, MCP
``memory.local.conflicts``) live in ``mu-server`` and ``mu-client`` — different repos, not this
lane's files. This is the engine-side service all three call, so the authorization, the FSM
validation and the audit stamp happen ONCE, in mu-core, and cannot drift between the three faces.
Reported: those three faces do not exist yet in either repo.

**What is NOT built here, and why (reported, not stubbed).**
``ResolveConflictStage`` itself — the consumer that drains :class:`ResolutionQueue` and performs
the supersession — belongs in ``pipelines/distill.py``, which is another lane's file: the shipped
detect/resolve split is two PRIVATE methods on ``DistillPipeline`` (``_reconcile``/``_resolve``),
not named stages, and standing up a parallel writer here to avoid touching it would create
exactly the second write path §2 forbids. The queue and the durable intent on the record are the
complete hand-off; the drain side is a one-file change in the distill lane.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

import structlog
from pydantic import Field

from mu_contracts.domain.errors import ConflictUnresolvedError
from mu_contracts.domain.events import (
    ConflictDismissed,
    ConflictPolicyChanged,
    ConflictResolved,
)
from mu_contracts.domain.model.conflict import (
    ConflictRecord,
    ConflictResolutionKind,
    ConflictState,
    ContentFreeModel,
    ResolutionOrigin,
)
from mu_contracts.domain.model.memory import Namespace
from mu_contracts.domain.model.scope import ClientScope
from mu_contracts.ports.governance import ConflictRecordRepository
from mu_contracts.ports.security import TenancyGuard
from mu_contracts.ports.time import Clock
from mu_engine.lifecycle.conflict import ConflictResolutionPolicy
from mu_engine.lifecycle.conflict_events import ConflictEventSink, publish_content_free
from mu_engine.lifecycle.conflict_policy import (
    ConflictLifecyclePolicy,
    TransitionTrigger,
)
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.services.conflict.ports import (
    MergedTextRef,
    NamespaceConflictPolicyStore,
    ResolutionQueue,
    WritableMemoryConflictPolicyStore,
    intent_from_record,
)

__all__ = ["ConflictResolutionService", "ManualDecision", "ManualDecisionKind"]

_log = structlog.get_logger("mu_engine.services.conflict.resolution")

_OP_RESOLVE = "memory.conflict_resolve"
_OP_REOPEN = "memory.conflict_reopen"
_OP_POLICY = "memory.conflict_policy_set"


class ManualDecisionKind(StrEnum):
    """The §5 line 209-213 resolve-action vocabulary.

    A SUPERSET of ``ConflictResolutionKind`` by exactly one member: ``DISMISS``. They are
    genuinely different axes and collapsing them would lose information — a dismissal is "there
    was never a conflict here", which is not a WAY of resolving one; it produces
    ``state=DISMISSED`` with ``resolution_kind=None`` and the no-reopen-on-same-hashes protection
    (§3 line 106), which no resolution kind carries.
    """

    SUPERSEDE = "supersede"
    KEEP_BOTH = "keep_both"
    MERGE = "merge"
    QUARANTINE = "quarantine"
    DISMISS = "dismiss"


#: ``ManualDecisionKind`` -> the ``ConflictResolutionKind`` recorded on the record. ``DISMISS``
#: maps to ``None``: dismissal is a state, not a resolution kind (see above).
_DECISION_TO_KIND: dict[ManualDecisionKind, ConflictResolutionKind | None] = {
    ManualDecisionKind.SUPERSEDE: ConflictResolutionKind.SUPERSEDE,
    ManualDecisionKind.KEEP_BOTH: ConflictResolutionKind.KEEP_BOTH,
    ManualDecisionKind.MERGE: ConflictResolutionKind.MERGE,
    ManualDecisionKind.QUARANTINE: ConflictResolutionKind.QUARANTINE,
    ManualDecisionKind.DISMISS: None,
}

#: The kinds that name a single surviving item. ``KEEP_BOTH`` and ``DISMISS`` deliberately do
#: not: nothing is superseded, so there is no winner to name and no ``REINSTATE`` to ever undo.
_KINDS_REQUIRING_WINNER: frozenset[ManualDecisionKind] = frozenset(
    {ManualDecisionKind.SUPERSEDE, ManualDecisionKind.MERGE, ManualDecisionKind.QUARANTINE}
)

#: Decisions that ``ResolveConflictStage`` has NO work for, so nothing is enqueued.
#:
#: ``DISMISS`` is obvious — "there was never a conflict" supersedes nothing. ``KEEP_BOTH`` was
#: not, and was a live defect: it names no winner, so "every member that is not the winner"
#: computed BOTH members as ``loser_ids`` and handed the stage — whose job is to supersede
#: ``loser_ids`` — an instruction to invalidate both items, from the one decision that means
#: *"both remain active"* (``ConflictResolutionKind.KEEP_BOTH``; spec §5 line 211, §6). Nothing
#: needs applying for a coexisting outcome: neither item moves, the record carries the outcome,
#: and the inbox reads the record.
_KINDS_WITH_NOTHING_TO_APPLY: frozenset[ManualDecisionKind] = frozenset(
    {ManualDecisionKind.DISMISS, ManualDecisionKind.KEEP_BOTH}
)


class ManualDecision(ContentFreeModel):
    """One human decision, as the surfaces submit it (§5 lines 209-213).

    Inherits the CONTENT-FREE guard (``ContentFreeModel``) on purpose: a decision command travels
    the same distance a record does — it is logged, audited, and (§7) reflected into a sync delta
    — so a field carrying the merged TEXT rather than a ref to it would leak exactly as far.
    ``merged_text_ref`` is spec line 212's ``"<draft>"`` REFERENCE into the owning store.
    """

    kind: ManualDecisionKind
    winner_id: str | None = None
    #: A REFERENCE into the owning store, shape-checked by :data:`MergedTextRef` — the SAME
    #: annotated type ``ResolutionIntent`` uses. Two things were wrong with the bare ``str`` it
    #: replaces. (1) The name ``merged_text_ref`` walks past the field-NAME content-free guard
    #: this class inherits, so it was the one field through which raw memory text could reach a
    #: durable queue, a log and a sync delta — a runtime probe put a full sentence, passport
    #: number included, into the queued payload verbatim. (2) It was unbounded here while the
    #: intent bounded it at 512, so an over-long ref failed validation only AFTER ``resolve``
    #: had durably moved the record to ``RESOLVED``, losing the human's decision with no event,
    #: no queue entry and no way back. One shared type closes both.
    merged_text_ref: MergedTextRef | None = None
    #: Principal id of the human. AUDIT only — never an authz principal, and deliberately NOT one
    #: of the §7.17 total-order terms: *who* resolved a conflict must not change *which* item
    #: wins, or two replicas with different views of the actor would diverge (the reasoning
    #: ``PrivateDelta.resolved_by`` already records). Bounded ``min_length=1`` to MATCH
    #: ``ResolutionIntent.resolved_by``: every bound this DTO shares with the intent must be the
    #: same bound, or a value that passes here and fails there re-creates the two-step-commit
    #: loss the shared ``MergedTextRef`` type was introduced to close.
    resolved_by: str = Field(min_length=1)


class ConflictResolutionService:
    """Accept, validate and durably record a human conflict decision. Never applies it."""

    def __init__(
        self,
        *,
        records: ConflictRecordRepository,
        queue: ResolutionQueue,
        clock: Clock | None = None,
        bus: ConflictEventSink | None = None,
        tenancy: TenancyGuard | None = None,
        namespace_policies: NamespaceConflictPolicyStore | None = None,
        memory_policies: WritableMemoryConflictPolicyStore | None = None,
    ) -> None:
        self._records = records
        self._queue = queue
        self._clock: Clock = clock or SystemClock()
        self._bus = bus
        self._tenancy: TenancyGuard = tenancy or DefaultTenancyGuard()
        self._namespace_policies = namespace_policies
        self._memory_policies = memory_policies
        self._fsm = ConflictLifecyclePolicy()

    # ------------------------------------------------------------------ resolve / dismiss
    async def resolve(
        self,
        scope: ClientScope,
        ns: Namespace,
        conflict_id: str,
        decision: ManualDecision,
    ) -> ConflictRecord:
        """Record a human decision and enqueue it. Returns the record's NEW state immediately.

        Order of operations is the contract, and each step guards the next:

        1. **Authorize.** ``TenancyGuard.assert_scope`` — a shared-plane resolution passes the
           same boundary check as any shared write (§5 line 220). Non-enumerating: a caller
           outside the partition gets the fixed NOT_FOUND envelope, never a hint that the
           conflict exists.
        2. **Load.** An unknown ``conflict_id`` raises ``ConflictUnresolvedError`` — the same
           non-enumerating shape, so a probe cannot distinguish "denied" from "absent".
        3. **Validate the edge**, through :class:`ConflictLifecyclePolicy` with
           ``trigger=EXPLICIT``. This is what makes it impossible for this surface to write
           ``AUTO_RESOLVED`` (automatic-only) and impossible for a sweep to write ``RESOLVED``
           (explicit-only). A record that is not actionable — already resolved, already
           dismissed, still ``DETECTED`` — is refused, not silently overwritten.
        4. **Record the durable intent** on the ``ConflictRecord`` (``resolution_kind``,
           ``resolved_winner_id``, ``resolution_origin=MANUAL``, ``resolved_by``, ``resolved_at``).
           This upsert is the point of no return: from here the decision survives a crash and the
           stage can be replayed idempotently.
        5. **Enqueue** for ``ResolveConflictStage`` — never applied here.
        6. **Emit** the §8 bookend and return.
        """
        self._tenancy.assert_scope(scope, ns, _OP_RESOLVE)
        record = await self._load_actionable(ns, conflict_id)
        self._validate_decision(record, decision)

        to_state = (
            ConflictState.DISMISSED
            if decision.kind is ManualDecisionKind.DISMISS
            else ConflictState.RESOLVED
        )
        self._fsm.assert_transition(record, to_state, trigger=TransitionTrigger.EXPLICIT)
        needs_apply = decision.kind not in _KINDS_WITH_NOTHING_TO_APPLY

        resolution_kind = _DECISION_TO_KIND[decision.kind]
        winner_id = decision.winner_id
        resolved = record.model_copy(
            update={
                "state": to_state,
                "resolution_kind": resolution_kind,
                "resolved_winner_id": winner_id,
                "resolution_origin": ResolutionOrigin.MANUAL,
                "resolved_by": decision.resolved_by,
                "resolved_at": self._clock.now(),
                # An APPLY-BEARING decision is not applied yet; a NOTHING-TO-APPLY one is
                # complete the moment it is recorded, so it is stamped applied here and never
                # sits in `awaiting_apply` waiting for a stage with no work to do.
                "resolution_applied_at": None if needs_apply else self._clock.now(),
                # The ref lives on the RECORD too, not only on the queue payload: the record is
                # spec line 218's "durable intent", and an intent recovered from a record with no
                # ref would be a merge with nothing to merge.
                "merged_text_ref": decision.merged_text_ref,
            }
        )

        # Build the intent BEFORE the record write. Step 4's docstring calls the upsert "the
        # point of no return", and it is: `ACTIONABLE_STATES` excludes `RESOLVED`, so once the
        # record is terminal `_load_actionable` refuses every retry. Constructing the intent
        # afterwards meant any validation failure on it (an over-long `merged_text_ref` was the
        # live one) discarded the human's decision with the record already durably RESOLVED, the
        # queue empty, and no event — a loss invisible even in this method's own log line, which
        # sits after the enqueue. Validate first, commit second.
        # ONE mapping record -> intent (`intent_from_record`), shared with
        # `RecordBackedResolutionQueue.drain`, so a freshly-enqueued intent and one recovered
        # after a crash can never disagree about who the losers are.
        intent = intent_from_record(resolved) if needs_apply else None

        await self._records.upsert(resolved)
        if intent is not None:
            await self._queue.enqueue(intent)
        await self._emit_decision(resolved, decision)
        _log.info(
            "conflict_decision_recorded",
            conflict_id=resolved.conflict_id,
            state=resolved.state.value,
            kind=decision.kind.value,
            ns=ns.to_prefix(),
        )
        return resolved

    async def reopen(
        self,
        scope: ClientScope,
        ns: Namespace,
        conflict_id: str,
        *,
        trigger: TransitionTrigger = TransitionTrigger.EXPLICIT,
        current_member_hashes: tuple[str, ...] = (),
    ) -> ConflictRecord:
        """§5 line 214 — force re-adjudication under the CURRENT policy.

        Two callers, one path. An EXPLICIT reopen is the human pressing the button and is always
        allowed on any legal edge. An AUTOMATIC reopen is the detect side finding a genuinely new
        contradicting delta, and it is additionally gated by
        ``ConflictLifecyclePolicy.may_reopen_dismissed``: a ``DISMISSED`` record is re-opened only
        when a member's ``content_hash`` actually changed (§3 line 106). Without that gate a
        source that keeps re-asserting a dismissed fact would resurrect the conflict on every
        sweep and the dismiss button would mean nothing.

        Reopening deliberately does NOT clear ``resolution_origin``/``resolved_by``: a manual
        decision stays on the record and stays sticky in cross-device re-derivation (§7 line 262)
        while the reopened conflict is re-reviewed. A later automatic delta must never flip a
        human's answer — it may only ask them again.
        """
        self._tenancy.assert_scope(scope, ns, _OP_REOPEN)
        record = await self._get(ns, conflict_id)
        if trigger is TransitionTrigger.AUTOMATIC and not self._fsm.may_reopen_dismissed(
            record, current_member_hashes
        ):
            return record
        self._fsm.assert_transition(record, ConflictState.REOPENED, trigger=trigger)
        reopened = record.model_copy(update={"state": ConflictState.REOPENED})
        await self._records.upsert(reopened)
        return reopened

    # -------------------------------------------------------------- the APPLY-side callbacks
    async def mark_applied(
        self,
        ns: Namespace,
        conflict_id: str,
        *,
        superseded_valid_at: datetime | None = None,
    ) -> ConflictRecord:
        """Close the loop: ``ResolveConflictStage`` calls this once the supersession has LANDED.

        Until it does, the record sits in ``awaiting_apply`` and
        :class:`~mu_engine.services.conflict.ports.RecordBackedResolutionQueue` keeps handing the
        intent back out — at-least-once delivery whose stopping condition is a durable fact, not
        an in-process dict. That is the whole difference between "the apply is delayed" and the
        shipped behaviour, where a crash between the decision and the apply left a record saying
        ``RESOLVED`` that no surface could see and no code path could re-drive.

        NO ``TenancyGuard`` check: this is not a caller-facing surface. It is invoked by the
        background worker that already holds the plane-qualified writer lease for ``ns`` and has
        just written the supersession; asking a ``ClientScope`` of it would mean inventing one
        for a machine.

        Idempotent — an already-applied record is returned unchanged, so a redelivered intent
        cannot double-stamp.
        """
        record = await self._get(ns, conflict_id)
        if record.resolution_applied_at is not None:
            return record
        applied = record.model_copy(
            update={
                "resolution_applied_at": self._clock.now(),
                "superseded_valid_at": superseded_valid_at or record.superseded_valid_at,
            }
        )
        await self._records.upsert(applied)
        _log.info(
            "conflict_resolution_applied",
            conflict_id=applied.conflict_id,
            kind=applied.resolution_kind.value if applied.resolution_kind else None,
            ns=ns.to_prefix(),
        )
        return applied

    async def record_automatic_resolution(
        self,
        ns: Namespace,
        conflict_id: str,
        *,
        winner_id: str,
        kind: ConflictResolutionKind = ConflictResolutionKind.SUPERSEDE,
        superseded_valid_at: datetime | None = None,
    ) -> ConflictRecord:
        """The AUTOMATIC lane's counterpart to :meth:`resolve` — ``DETECTED -> AUTO_RESOLVED``.

        **Why this exists.** ``ConflictState.AUTO_RESOLVED`` and ``ResolutionOrigin.AUTO`` were
        declared, documented and put in the FSM table, and nothing in the source tree ever wrote
        either: the adjudicator opened a record only on the PARK branch, so an auto-superseded
        memory had no conflict aggregate at all, ``ConflictResolved`` could only ever say
        ``origin=manual``, and spec line 117 — which makes ``resolution_origin`` the entire
        distinction between ``AUTO_RESOLVED`` and ``RESOLVED`` — described a state no production
        path could reach. ``ConflictAdjudicator`` now opens the record in ``DETECTED`` on the
        automatic lane; this closes it once the supersession has landed.

        Called with ``trigger=AUTOMATIC``, which is what makes ``AUTOMATIC_ONLY_EDGES`` mean
        something: a human surface cannot forge ``AUTO_RESOLVED`` and destroy the origin audit,
        and this cannot forge ``RESOLVED``. The apply has already happened by the time it is
        called, so ``resolution_applied_at`` is stamped here and the record never enters
        ``awaiting_apply`` — the automatic lane's supersession is written under the same lease
        that decided it, so there is no gap to recover from.

        **Wiring gap, REPORTED not stubbed:** ``DistillPipeline._resolve``
        (``mu_engine/pipelines/distill.py``) is the only site that writes ``state=SUPERSEDED``
        automatically, and it is another lane's file. It must call this after the write lands,
        with ``verdict.conflict_record.conflict_id``.
        """
        record = await self._get(ns, conflict_id)
        if record.state is ConflictState.AUTO_RESOLVED:
            return record  # idempotent — a replayed sweep must not re-stamp the audit
        if winner_id not in record.member_ids:
            raise ConflictUnresolvedError("winner_id is not a member of this conflict")
        self._fsm.assert_transition(
            record, ConflictState.AUTO_RESOLVED, trigger=TransitionTrigger.AUTOMATIC
        )
        now = self._clock.now()
        resolved = record.model_copy(
            update={
                "state": ConflictState.AUTO_RESOLVED,
                "resolution_kind": kind,
                "resolved_winner_id": winner_id,
                "resolution_origin": ResolutionOrigin.AUTO,
                "resolved_at": now,
                "resolution_applied_at": now,
                "superseded_valid_at": superseded_valid_at,
            }
        )
        await self._records.upsert(resolved)
        await publish_content_free(
            self._bus,
            ConflictResolved(
                namespace=ns,
                conflict_id=resolved.conflict_id,
                winner_id=winner_id,
                loser_ids=[i for i in resolved.member_ids if i != winner_id],
                resolution_origin=ResolutionOrigin.AUTO,
            ),
        )
        return resolved

    # ------------------------------------------------------------------------- policy setters
    async def set_namespace_policy(
        self, scope: ClientScope, ns: Namespace, policy: ConflictResolutionPolicy
    ) -> None:
        """§5 line 215 — ``PUT /v1/namespaces/{ns}/conflict-policy``.

        Governs NEW detections and ``REOPENED`` ones only. It does not retroactively re-resolve
        already-``RESOLVED``/``AUTO_RESOLVED`` conflicts (spec line 158: *"no surprise mass-flips"*)
        — which is why every record snapshots the policy that governed it at detection time, and
        why this method touches the policy store and nothing else.
        """
        self._tenancy.assert_scope(scope, ns, _OP_POLICY)
        if self._namespace_policies is None:
            raise ConflictUnresolvedError("no namespace conflict-policy store is wired")
        await self._namespace_policies.set_policy(ns, policy)
        await self._emit_policy_changed(ns, policy)

    async def set_memory_policy(
        self,
        scope: ClientScope,
        ns: Namespace,
        memory_id: str,
        policy: ConflictResolutionPolicy | None,
    ) -> None:
        """§5 line 216 — ``PUT /v1/memories/{id}/conflict-policy``; ``None`` clears the override."""
        self._tenancy.assert_scope(scope, ns, _OP_POLICY)
        if self._memory_policies is None:
            raise ConflictUnresolvedError("no writable per-memory conflict-policy store is wired")
        await self._memory_policies.set_override(ns, memory_id, policy)
        if policy is not None:
            await self._emit_policy_changed(ns, policy)

    # ------------------------------------------------------------------------------ internals
    async def _get(self, ns: Namespace, conflict_id: str) -> ConflictRecord:
        record = await self._records.get(ns, conflict_id)
        if record is None:
            # Non-enumerating (CANONICAL §1 rule 5 / ports.security): the message never echoes
            # the requested id back, so a probe cannot distinguish denied from absent.
            raise ConflictUnresolvedError("not found")
        return record

    async def _load_actionable(self, ns: Namespace, conflict_id: str) -> ConflictRecord:
        record = await self._get(ns, conflict_id)
        if not self._fsm.is_actionable(record):
            raise ConflictUnresolvedError(
                f"conflict is not awaiting a decision (state={record.state.value})"
            )
        return record

    @staticmethod
    def _validate_decision(record: ConflictRecord, decision: ManualDecision) -> None:
        """Refuse a decision that names no winner when the kind needs one, or names a
        non-member.

        The second check is the load-bearing one: without it a caller could 'resolve' a conflict
        in favour of an item that is not part of it, and the stage would later supersede both
        real members in favour of an unrelated memory.
        """
        if decision.kind in _KINDS_REQUIRING_WINNER:
            if decision.winner_id is None:
                raise ConflictUnresolvedError(f"{decision.kind.value} requires a winner_id")
            if decision.winner_id not in record.member_ids:
                raise ConflictUnresolvedError("winner_id is not a member of this conflict")
        elif decision.winner_id is not None:
            raise ConflictUnresolvedError(f"{decision.kind.value} does not take a winner_id")
        if decision.kind is ManualDecisionKind.MERGE and decision.merged_text_ref is None:
            raise ConflictUnresolvedError("merge requires a merged_text_ref")

    async def _emit_decision(self, record: ConflictRecord, decision: ManualDecision) -> None:
        """§8 lines 279-280.

        ``ConflictDismissed`` for a dismissal; ``ConflictResolved`` otherwise — but ONLY when the
        decision named a winner. **Reported contract gap:** the ratified ``ConflictResolved``
        (CANONICAL §5 / ``events.py``) declares ``winner_id: str`` as REQUIRED, so a winner-less
        resolution — ``KEEP_BOTH``, the coexisting outcome §6 is built around — cannot be
        expressed by the event at all. Emitting an empty ``winner_id`` would put a false winner
        on the bus, so nothing is emitted and the gap is reported instead of papered over; the
        record itself still carries the outcome and the inbox projector reads the record.
        """
        if decision.kind is ManualDecisionKind.DISMISS:
            await publish_content_free(
                self._bus,
                ConflictDismissed(
                    namespace=record.namespace,
                    conflict_id=record.conflict_id,
                    by=decision.resolved_by,
                ),
            )
            return
        if record.resolved_winner_id is None:
            _log.info(
                "conflict_resolved_event_suppressed_winnerless",
                conflict_id=record.conflict_id,
                kind=decision.kind.value,
            )
            return
        await publish_content_free(
            self._bus,
            ConflictResolved(
                namespace=record.namespace,
                conflict_id=record.conflict_id,
                winner_id=record.resolved_winner_id,
                loser_ids=[i for i in record.member_ids if i != record.resolved_winner_id],
                resolution_origin=ResolutionOrigin.MANUAL,
            ),
        )

    async def _emit_policy_changed(self, ns: Namespace, policy: ConflictResolutionPolicy) -> None:
        """§8 line 281. **Reported information loss:** the design's field list is
        ``{namespace, scope, target_id, mode, strategy}``; the RATIFIED event (CANONICAL §5) is
        ``{namespace, policy}``, which drops ``scope``/``target_id`` — so "which memory's override
        changed" is unrecoverable from the event. CANONICAL wins and the code follows it; the loss
        is flagged to the CANONICAL owner rather than worked around by smuggling the target id
        into the ``policy`` string."""
        await publish_content_free(
            self._bus, ConflictPolicyChanged(namespace=ns, policy=policy.mode.value)
        )
