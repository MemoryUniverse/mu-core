"""CAPTURE->INGEST concrete stages (engine-core-spec §5 + §6.4; PIPELINES-design.md §4.1).

PORT of the prototype ``local_client/service.py:268`` ``remember()`` write path
(``/home/user/hackathon/memory_universe/...``): STM durable write keyed by ``activity_id``, then a
deterministic STM->MTM promotion that embeds the ATOMIC-FACT vector and upserts to the vector store
keyed by ``content_hash``. NO LLM on this path (the mem0 fact-extraction / diff-loop is the DISTILL
slice). Fully async; the store I/O is bounded by the adapters' ``@retry_io``.

Stage list (engine-core-spec §6.4 CAPTURE->INGEST; software-arch spec §6 ``IngestService.ingest``,
l.338-342, provides the numbering ``PersistRawArtifactStage`` now fills in as step 1):
0. ``PersistRawArtifactStage`` — persist raw as a ContextArtifact (provenance root, spec l.340)
1. ``WriteStmStage``          — STM add, kind=REFERENCE -> artifact_ref (spec l.341), key
                                 ``activity_id`` (M12)                  -> ``MemoryCaptured``
2. ``DeterministicPromoteStage`` — MTM upsert, key ``content_hash``    -> ``MemoryPromoted``
3. ``EmitIngestCompletedStage``  — fan-out trigger                     -> ``IngestCompleted``
4. ``EnqueueEnrichmentStage``    — S2 write-time enrichment (ADR-0055, AD-241): durable, best-
                                    -effort enqueue of the just-written memory onto
                                    ``EnrichmentQueuePort`` for a background worker
                                    (``pipelines/enrichment_worker.py``) to enrich AFTER this
                                    call returns. Still NO LLM on this path — this stage only
                                    writes a small job row; the LLM call happens later, off the
                                    request path. Optional (only present when the composition
                                    root threads an ``enrichment_queue``, same backward-compatible
                                    precedent as ``artifacts``/``PersistRawArtifactStage`` above);
                                    NEVER raises — a full/unavailable queue is a named
                                    ``StageDegraded``, never a failed ``add()``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from hashlib import sha256
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mu_contracts.domain.events import (
    DegradeReason,
    DomainEvent,
    IngestCompleted,
    MemoryCaptured,
    MemoryPromoted,
    StageDegraded,
)
from mu_contracts.domain.model.authorized_ids import AUTHORIZED_IDS_KEY, validate_stamp_subjects
from mu_contracts.domain.model.enrichment import EnrichmentJob
from mu_contracts.domain.model.memory import Namespace, Tier, Visibility
from mu_contracts.ports.enrichment import EnrichmentQueuePort
from mu_contracts.ports.time import Clock
from mu_engine.pipelines.base import BaseStage, PipelineContext, StageOutcome, StageStatus
from mu_engine.pipelines.errors import StageExecutionError
from mu_engine.pipelines.ledger import StageLedger
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.services.extract import extract_valid_at
from mu_engine.services.settings import IngestSettings
from mu_engine.storage.domain.artifact import ArtifactKind, ContextArtifact
from mu_engine.storage.domain.memory import (
    FactObjectKind,
    MemoryItem,
    MemoryKind,
    MemorySource,
    MemoryState,
    MemoryTier,
    Polarity,
)
from mu_engine.storage.ports import ContextRepository, MtmTierRepository, StmTierRepository

__all__ = [
    "DeterministicPromoteStage",
    "EmitIngestCompletedStage",
    "EnqueueEnrichmentStage",
    "IngestActivity",
    "PersistRawArtifactStage",
    "WriteStmStage",
    "activity_id_for",
]

_log = structlog.get_logger("mu_engine.pipelines.concrete.ingest")


class IngestActivity(BaseModel):
    """The INGEST pipeline input — one captured, salient host activity (S1 ``RawActivity`` essence,
    ``data-extraction-methodology §1.1``). Carries its resolved η + the two deterministic-promotion
    signals (``importance``, explicit ``promote``). ``text`` is the salient slice (echoes dropped
    upstream). A structured ``subject/predicate/object`` triple, when present, is what the atomic
    fact embeds (mem0 ``embed(new_mem)`` / Graphiti ``embed(edge.fact)``; ``mtm_qdrant.py:222``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: Namespace
    host: str = Field(min_length=1)
    session_offset: str = Field(min_length=1)  # source offset — the M12 replay discriminator
    kind: str = "user_message"  # ActivityKind
    text: str = Field(min_length=1)  # salient content
    # S1b (TRACE-0923.md §7/§6.2, ADR pending): the conversational-order key, forwarded verbatim
    # onto `MemoryItem.turn_seq` below (see that field's own docstring for why `session_offset`
    # above — deliberately RANDOM, "never a pure M12 replay" — cannot double as this). `None`
    # (the default) when the caller assigns none; the read path degrades gracefully.
    turn_seq: int | None = Field(default=None, ge=0)

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    promote: bool = False
    source: MemorySource = MemorySource.USER

    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    object_kind: FactObjectKind | None = None
    polarity: Polarity = Polarity.POSITIVE

    owner_id: str | None = None  # defaults to the η principal when omitted

    # AD-294: set by `IngestService.remember` (never by a caller — the field exists so the flag
    # can ride the SAME frozen object into `_build_memory_item` below) when `text` was
    # credential-shaped under `CredentialPolicy.MARK`/`REDACT`. Content-free: a boolean, never the
    # matched value. `False` for every pre-existing caller/test — additive, backward-compatible.
    credential_shaped: bool = False

    #: Model-A stamp for a SHARED write — the EXPLODED principal ids permitted to read this item
    #: (CANONICAL §7.4: *"It is STAMPED at write/sync time from the session participant set +
    #: materialized ACL rows"*; this is that write-time stamp, and the recall filter's only input).
    #:
    #: The caller resolves the session's participant roster and passes it; the engine never
    #: invents one, because the engine cannot see a roster — membership is a governance fact that
    #: lives on the plane that owns the session. `None` on a SHARED η therefore writes an
    #: UNSTAMPED row, which every Model-A reader DENIES to everyone (`storage/authz.py`,
    #: `qdrant_mtm`, `falkor_ltm`): fail-closed, never a room-readable row nobody authorized.
    #: MUST be `None` on a PRIVATE η — §7.4: *"PRIVATE items are isolated by
    #: `Namespace.to_prefix()` partitioning, not via `authorized_ids`"*, and a private item
    #: *"never enters a grant/ACL/`authorized_ids` until published"* (§1 rule 6).
    authorized_ids: frozenset[str] | None = None

    @model_validator(mode="after")
    def _stamp_is_shared_only_and_principal_only(self) -> IngestActivity:
        """Validate the stamp AT THE BOUNDARY (DEV-STANDARDS rule 2), not at the store.

        Two refusals, both from §7.4: a role/session/device id may never be stamped (it is an
        offboarding hole / ACL bypass — `validate_stamp_subjects`), and a PRIVATE item may never
        carry a stamp at all (its authorization is the partition key, and a stamp there would be
        a second, contradicting answer to "who may read this")."""
        if self.authorized_ids is None:
            return self
        if self.namespace.visibility is not Visibility.SHARED:
            raise ValueError(
                "authorized_ids is a SHARED-plane stamp (CANONICAL §7.4): a PRIVATE item is "
                "isolated by Namespace.to_prefix() and never enters an ACL until published"
            )
        validate_stamp_subjects(self.authorized_ids)
        return self

    def atomic_fact_text(self) -> str:
        """The text the MTM vector embeds — the ``{subject} {predicate} {object}`` triple when
        structured, else the whole salient utterance (engine-core-spec §5 atomic-fact embedding)."""
        if self.subject and self.predicate and self.object:
            return f"{self.subject} {self.predicate} {self.object}"
        return self.text


def activity_id_for(activity: IngestActivity) -> str:
    """``activity_id = sha256(host|session|source_offset|kind)`` (CANONICAL §8-M12).

    The discriminator is the SOURCE OFFSET, not ``content_hash``: a genuine second occurrence at a
    new offset is a legitimate reinforcement (kept), while a pure replay of the same offset is a
    no-op (dropped). Mirrors ``S1-...:156`` + the outbox ``UNIQUE(activity_id)`` rule.
    """
    basis = "\x1f".join(
        (activity.host, activity.namespace.session, activity.session_offset, activity.kind)
    )
    return sha256(basis.encode("utf-8")).hexdigest()


def _build_memory_item(
    activity: IngestActivity,
    *,
    at: datetime,
    artifact_ref: str | None = None,
    provenance_id: str | None = None,
) -> MemoryItem:
    """Mint the STM ``MemoryItem`` ONCE (CANONICAL §7.1 — id carried unchanged across tiers).

    ``content_hash`` is derived by the model from content + triple; it is DISTINCT from ``id``.

    ``artifact_ref``/``provenance_id`` (NEW, software-arch spec §6 ``IngestService.ingest`` step
    2, l.341: "write STM MemoryItems (kind=reference -> artifact)"): when
    ``PersistRawArtifactStage`` ran ahead of this stage in the pipeline, it threads the minted
    :class:`~mu_engine.storage.domain.artifact.ContextArtifact`'s id/provenance stream through
    ``ctx.state`` — the STM capture record becomes ``kind=REFERENCE`` targeting it, per the
    design's ``MemoryKind`` split (``storage/domain/memory.py`` "proposition = inline
    self-contained fact; reference = handle to a ContextArtifact"). Both default to ``None`` so
    this function stays byte-compatible for any OTHER caller that mints a plain, artifact-less
    ``PROPOSITION`` directly (e.g. the content-hash-only fallback calls in this module and in
    ``services/ingest.py``, and any pre-existing test that calls it bare).
    """
    ns = activity.namespace
    owner = activity.owner_id or ns.user
    object_kind = activity.object_kind
    if activity.object is not None and object_kind is None:
        object_kind = FactObjectKind.LITERAL
    kwargs: dict[str, Any] = {}
    if provenance_id is not None:
        kwargs["provenance_id"] = provenance_id
    if activity.authorized_ids is not None:
        # The Model-A stamp is written ONCE, here, onto the item minted ONCE (§7.1 id-stability),
        # so it rides the SAME object into every tier: STM's JSON row, the Qdrant payload keyword
        # index (`qdrant_mapper.py:85-88` reads exactly this key), and the FalkorDB list property
        # (`graph_mapper.py`). Stamping per-tier instead would let one tier answer a security
        # question differently from another — the state `CompositeStampWriter` exists to prevent
        # on the re-stamp path.
        kwargs["metadata"] = {AUTHORIZED_IDS_KEY: sorted(activity.authorized_ids)}
    return MemoryItem(
        content=activity.text,
        kind=MemoryKind.REFERENCE if artifact_ref else MemoryKind.PROPOSITION,
        tier=MemoryTier.STM,
        state=MemoryState.ACTIVE,
        namespace=ns,
        owner_id=owner,
        workspace_id=ns.workspace,
        session_id=ns.session,
        created_at=at,
        updated_at=at,
        # AD-308 follow-up (team-lead review, 2026-09-26): the ONE mint point for a captured
        # MemoryItem (CANONICAL §7.1) is the right place for the temporal signal a raw capture's
        # OWN text states ("...yesterday", "...on 8 May, 2023") to survive at all — before this,
        # `valid_at` stayed unset for every STM/MTM item unconditionally (date resolution only
        # ever ran during MTM->LTM DISTILL, which AD-310/311 measured never wins a recall slot).
        # `extract_valid_at` (`services/extract.py`) is LLM-free (this stage's own module
        # docstring: "NO LLM on this path") and never alters `content` — only a NEW, additive
        # field. `DeterministicPromoteStage`'s STM->MTM promotion is a `model_copy` that carries
        # this field forward unchanged (`ingest.py:_execute` below), so the date reaches the tier
        # that actually serves recall slots without any change to promotion itself.
        valid_at=extract_valid_at(activity.text, now=at),
        importance_score=activity.importance,
        source=activity.source,
        turn_seq=activity.turn_seq,
        subject=activity.subject,
        predicate=activity.predicate,
        object=activity.object,
        object_kind=object_kind,
        polarity=activity.polarity,
        artifact_ref=artifact_ref,
        credential_shaped=activity.credential_shaped,  # AD-294 — content-free flag only
        **kwargs,
    )


def _activity(ctx: PipelineContext) -> IngestActivity:
    activity = ctx.state.get("activity")
    if not isinstance(activity, IngestActivity):
        raise StageExecutionError("ingest", "ctx.state['activity'] is not an IngestActivity")
    return activity


def artifact_id_for(activity: IngestActivity) -> str:
    """Deterministic artifact id (software-arch spec §6, l.340) — the SAME discriminator basis
    as :func:`activity_id_for` (this one raw activity has exactly one provenance root), prefixed
    so it never collides with a memory id in a log/index that mixes both kinds."""
    return f"art_{activity_id_for(activity)}"


class PersistRawArtifactStage(BaseStage):
    """Stage 0 — persist the raw activity as a ContextArtifact provenance root (software-arch
    spec §6 ``IngestService.ingest`` step 1, l.340: "persist raw as ContextArtifact(s)
    (provenance)"). Runs BEFORE ``WriteStmStage`` so the minted artifact id/provenance stream can
    be threaded onto the STM ``MemoryItem``'s ``artifact_ref``/``provenance_id`` (step 2, l.341).

    Idempotency (PIPELINES §2.2): NOT ledger-gated (``idempotency_key`` returns ``""``, mirroring
    ``DeterministicPromoteStage``'s own "not every stage needs the ledger" precedent, l.235-239 of
    this module) — the underlying write is ALREADY idempotent by construction: both the artifact
    id (:func:`artifact_id_for`) and the blob path (content-hash-addressed, ``content_fs.py``)
    are deterministic functions of the activity, so a crash-replay simply re-derives and re-``put``
    s the IDENTICAL artifact (same id, same bytes) — a harmless overwrite, never a duplicate,
    and never a re-mint of a fresh random id (CANONICAL §7.1 id-stability, applied here to the
    artifact side of the provenance pair).
    """

    name = "persist_raw_artifact"

    def __init__(self, *, artifacts: ContextRepository, ledger: StageLedger, clock: Clock) -> None:
        super().__init__(ledger=ledger, clock=clock)
        self._artifacts = artifacts

    def idempotency_key(self, ctx: PipelineContext) -> str:
        return ""  # not ledger-gated — see class docstring (the store write is self-idempotent).

    async def _execute(self, ctx: PipelineContext) -> StageOutcome:
        activity = _activity(ctx)
        blob = activity.text.encode("utf-8")
        content_hash = sha256(blob).hexdigest()
        provenance_id = f"prov_{activity_id_for(activity)}"
        ns = activity.namespace
        art = ContextArtifact(
            id=artifact_id_for(activity),
            namespace=ns,
            kind=ArtifactKind.TRANSCRIPT,
            version=content_hash,
            uri=f"artifact://{ns.to_prefix()}/{content_hash}",
            content_hash=content_hash,
            provenance_id=provenance_id,
        )
        stored = await self._artifacts.put(art, blob)
        return StageOutcome(
            status=StageStatus.OK,
            produced={
                "artifact": stored,
                "artifact_id": stored.id,
                "artifact_provenance_id": stored.provenance_id,
            },
        )


class WriteStmStage(BaseStage):
    """Stage 1 — durable STM write, idempotent on ``activity_id`` (engine-core-spec §6.4).

    The writer NEVER waits for conflict detection (conflict-resolution-async §1 invariant 1): this
    stage returns on the STM-durable write; promotion + distill run downstream. Mints the id ONCE
    and carries it forward in ``ctx.state`` (``memory_ids``) so every later tier reuses it — UNLESS
    the STM adapter's own D4 write-time content-hash index reports that id was never actually kept
    (a dedup hit against a DIFFERENT, still-resident row for the SAME content in this namespace),
    in which case ``_execute`` re-stamps the RESIDENT id before it is carried forward
    (return-idempotency — ``add()`` now returns the SAME id across repeat identical-content calls
    in one namespace, DATA-QUALITY-REASSESSMENT §3 "add() idempotency"). This is per-``activity_id``
    NOT ledger-gated (a fresh ``session_offset`` is always a fresh ``_execute``, per the class name
    above) — the content-level idempotency lives entirely inside the STM store's own dedup index,
    never this stage's own ledger key.
    """

    name = "write_stm"

    def __init__(self, *, stm: StmTierRepository, ledger: StageLedger, clock: Clock) -> None:
        super().__init__(ledger=ledger, clock=clock)
        self._stm = stm

    def idempotency_key(self, ctx: PipelineContext) -> str:
        return activity_id_for(_activity(ctx))

    async def _execute(self, ctx: PipelineContext) -> StageOutcome:
        activity = _activity(ctx)
        # NEW (software-arch spec §6, l.341): when PersistRawArtifactStage ran ahead of this one
        # (the normal pipeline order below), ctx.state carries the minted ContextArtifact's id +
        # shared provenance stream — this capture record becomes kind=REFERENCE targeting it.
        item = _build_memory_item(
            activity,
            at=self._clock.now(),
            artifact_ref=ctx.state.get("artifact_id"),
            provenance_id=ctx.state.get("artifact_provenance_id"),
        )
        # RETURN-IDEMPOTENCY (``add()`` return contract, DATA-QUALITY-REASSESSMENT §3 "add()
        # idempotency" / the D4 report): ``put()`` reports the RESIDENT memory id —
        # ``item.id`` on a fresh write, or a DIFFERENT, still-resident id when the D4 write-time
        # content-hash index (``ports.py``'s ``StmTierRepository.put`` docstring) already holds
        # this exact content in this namespace. D4 landed the STORE-level dedup (one physical STM
        # row) without ever surfacing it here, so ``IngestService.remember`` kept minting+
        # returning a fresh id the store had already discarded — a caller-visible id that did not
        # correspond to any physical row. Re-stamp ``item`` onto the id the store actually kept
        # (never a second, independently-fetched copy) so every downstream consumer of THIS
        # stage's ``produced`` — the ``MemoryCaptured`` event, ``IngestResult.memory_id``, and
        # ``DeterministicPromoteStage``'s own id-stability (CANONICAL §7.1: a later promotion of
        # this content writes to the SAME MTM point, never a stray new one) — sees ONE consistent
        # identity. Gated entirely by the STM adapter's own ``stm_dedup_enabled``
        # (``MU_INGEST__STM_DEDUP``): dedup off -> ``put()`` always returns ``item.id`` -> this is
        # a no-op and ``add()`` reverts to minting a fresh id every call, unchanged.
        resident_id = await self._stm.put(item)
        if resident_id != item.id:
            # D3 fix (AD-266b): a dedup hit means `resident_id` names the WINNER row
            # `RedisStmAdapter._bump_if_duplicate` just bumped `mention_count` on — re-stamping
            # `item.id` alone onto this stage's OWN fresh copy (the pre-fix behaviour) would carry
            # that fresh copy's stale `mention_count=1` forward into `ctx.state["stm_item"]`,
            # which is exactly what `DeterministicPromoteStage._resolve_item`'s fast path reads
            # to decide `mention_promote` — silently hiding the very bump this stage exists to
            # surface. Re-read the ACTUAL resident row instead of re-stamping a field onto the
            # discarded one; a `None` (raced away between the store's own dedup write and this
            # read) falls back to the previous re-stamp so the pipeline still has SOME id to carry
            # forward rather than raising on a narrow, already-benign race.
            resident = await self._stm.get(activity.namespace, resident_id)
            item = resident if resident is not None else item.model_copy(update={"id": resident_id})
        return StageOutcome(
            status=StageStatus.OK,
            produced={"stm_item": item, "memory_ids": [item.id], "content_hash": item.content_hash},
            events=[MemoryCaptured(namespace=activity.namespace, ids=[item.id], tier=Tier.STM)],
            idempotency_key=self.idempotency_key(ctx),
        )

    async def reconstruct_produced(
        self, ctx: PipelineContext, events: Sequence[DomainEvent]
    ) -> dict[str, Any]:
        """(F1 — crash-replay resume) On a ledger hit the STM write is durable but ``ctx.state``
        (``stm_item``/``memory_ids``/``content_hash``) was only ever populated in the crashed
        process's memory — ``DeterministicPromoteStage._resolve_item``'s recovery path (this
        module, ``_resolve_item``) needs it back to promote instead of raising "no STM item".

        Re-reads the SAME durable STM record by the id carried in the recorded ``MemoryCaptured``
        event — NEVER calls ``_build_memory_item`` again, which would mint a fresh random ``id``
        (``storage/domain/memory.py::_memory_id``) and silently duplicate the already-durable
        write (the exact failure this hook exists to prevent, CANONICAL §7.1 id-stability).
        """
        activity = _activity(ctx)
        ids = [mid for event in events if isinstance(event, MemoryCaptured) for mid in event.ids]
        if not ids:
            # Recorded events don't carry a MemoryCaptured (should be impossible for this stage —
            # its own _execute always emits exactly one) — nothing to rehydrate.
            return {}
        item = await self._stm.get(activity.namespace, ids[0])
        if item is None:
            # Ledger says complete but the durable STM record is gone (evicted/corrupted store).
            # Fail loud rather than silently re-minting a new id under the same recorded name.
            raise StageExecutionError(
                self.name,
                f"ledger-complete for {ids[0]!r} but the STM record is missing "
                "(durability invariant violated — refusing to mint a replacement id)",
            )
        return {"stm_item": item, "memory_ids": [item.id], "content_hash": item.content_hash}


class DeterministicPromoteStage(BaseStage):
    """Stage 2 — deterministic STM->MTM promotion, idempotent on ``content_hash`` (§6.4).

    NO LLM. Promotes on three LLM-free signals available at capture (the ``service.py:335``
    rule): explicit ``promote``, ``importance >= importance_promote``, OR
    ``mention_count >= mention_promote`` (D3 fix, AD-266b — the third arm this class's own
    docstring used to report as unwired "minus the mention_count arm which needs the
    content-hash-indexed STM lookup"; that lookup (D4, ``redis_stm.py::_bump_if_duplicate``)
    shipped, and ``WriteStmStage`` now hands this stage the WINNER row's true, bumped
    ``mention_count`` via ``ctx.state["stm_item"]`` — see :meth:`_mention_count`). When it
    promotes it embeds the ATOMIC-FACT vector via the REAL local ``EmbeddingPort`` and upserts
    the promoted copy (same id) to MTM. When it does not promote it is a normal ``OK`` no-op
    with NO ledger mark (a later reinforcement may still promote) and NO event.
    """

    name = "deterministic_promote"

    def __init__(
        self,
        *,
        stm: StmTierRepository,
        mtm: MtmTierRepository,
        embedder: EmbeddingPort,
        settings: IngestSettings,
        ledger: StageLedger,
        clock: Clock,
    ) -> None:
        super().__init__(ledger=ledger, clock=clock)
        self._stm = stm
        self._mtm = mtm
        self._embedder = embedder
        self._settings = settings

    def _mention_count(self, ctx: PipelineContext) -> int:
        """The STM winner row's CURRENT ``mention_count`` (D3 fix, AD-266b) — read from
        ``ctx.state["stm_item"]``, which ``WriteStmStage`` populates BEFORE this stage runs on
        every path (a fresh ``_execute``, a dedup-hit re-fetch of the winner, AND a ledger-hit
        ``reconstruct_produced`` replay — that method's own docstring: it re-reads the durable
        STM record, never re-mints). Defaults to ``1`` (a never-deduplicated item's true count)
        when ``stm_item`` is absent for any reason, never a raise — this arm degrading to
        unavailable must never block the ``explicit``/``importance`` arms above it."""
        item = ctx.state.get("stm_item")
        return item.mention_count if isinstance(item, MemoryItem) else 1

    def _promotion_reason(self, activity: IngestActivity, *, mention_count: int = 1) -> str | None:
        if activity.promote:
            return "explicit"
        if activity.importance >= self._settings.importance_promote:
            return "importance"
        if mention_count >= self._settings.mention_promote:
            return "mention_count"
        return None

    def idempotency_key(self, ctx: PipelineContext) -> str:
        # Ledger-gate ONLY when we will actually promote; a non-promotion is not a completed write,
        # so it must not be marked (empty key => BaseStage runs but never marks — PIPELINES §2.2).
        activity = _activity(ctx)
        reason = self._promotion_reason(activity, mention_count=self._mention_count(ctx))
        if reason is None:
            return ""
        content_hash = (
            ctx.state.get("content_hash")
            or _build_memory_item(
                activity,
                at=self._clock.now(),
                artifact_ref=ctx.state.get("artifact_id"),
                provenance_id=ctx.state.get("artifact_provenance_id"),
            ).content_hash
        )
        # NAMESPACE-SCOPED (AG-2 / design §13 item 6d): the bare content_hash collides across
        # tenants — two DIFFERENT users adding identical content+triple into one instance would
        # share one ledger key, so the second user's promote SKIPPED("ledger-hit") and silently
        # never upserted under their own namespace. Prefix with the full η partition
        # (Namespace.to_prefix(), CANONICAL §1 rule 5) so the ledger key is per-tenant, matching
        # the physical MTM key-space every store adapter already partitions on. Chosen over
        # folding η into ``compute_content_hash`` (storage/domain/memory.py) because that basis is
        # also the STORED ``content_hash`` field (dedupe/version key across the whole record, spec
        # §2.4) — widening it would change the on-disk hash for every existing item, whereas the
        # ledger key is a private, throwaway idempotency token local to this stage.
        return f"{activity.namespace.to_prefix()}:{content_hash}"

    async def _resolve_item(self, ctx: PipelineContext, activity: IngestActivity) -> MemoryItem:
        # Fast path: the STM stage handed the item over in-process. Recovery path (a crash between
        # STM and MTM, then a retry): re-read the durable STM record by its stable id.
        item = ctx.state.get("stm_item")
        if isinstance(item, MemoryItem):
            return item
        memory_ids = ctx.state.get("memory_ids") or []
        if memory_ids:
            recovered = await self._stm.get(activity.namespace, memory_ids[0])
            if recovered is not None:
                return recovered
        raise StageExecutionError(self.name, "no STM item to promote (missing id / STM record)")

    async def _execute(self, ctx: PipelineContext) -> StageOutcome:
        activity = _activity(ctx)
        reason = self._promotion_reason(activity, mention_count=self._mention_count(ctx))
        if reason is None:
            return StageOutcome(status=StageStatus.OK, produced={"promoted": False})

        item = await self._resolve_item(ctx, activity)
        vectors = await self._embedder.embed([activity.atomic_fact_text()])
        promoted = item.model_copy(
            update={
                "tier": MemoryTier.MTM,
                "state": MemoryState.ACTIVE,
                "embedding": list(vectors[0]),
                "embedding_model": self._embedder.model_name,
                "last_seen": self._clock.now(),
                "updated_at": self._clock.now(),
            }
        )
        await self._mtm.upsert(promoted)
        return StageOutcome(
            status=StageStatus.OK,
            produced={"promoted": True, "mtm_item": promoted},
            events=[
                MemoryPromoted(
                    namespace=activity.namespace,
                    id=promoted.id,
                    frm=Tier.STM,
                    to=Tier.MTM,
                    reason=reason,
                )
            ],
            idempotency_key=self.idempotency_key(ctx),
        )


class EmitIngestCompletedStage(BaseStage):
    """Stage 3 — the DISTILL fan-out trigger (engine-core-spec §6.4). Idempotent on the
    ``correlation_id`` so a replay re-emits the recorded ``IngestCompleted`` rather than nothing."""

    name = "emit_ingest_completed"

    def idempotency_key(self, ctx: PipelineContext) -> str:
        return f"ingest-completed:{ctx.correlation_id}"

    async def _execute(self, ctx: PipelineContext) -> StageOutcome:
        memory_ids = list(ctx.state.get("memory_ids") or [])
        return StageOutcome(
            status=StageStatus.OK,
            produced={},
            events=[IngestCompleted(namespace=ctx.namespace, memory_ids=memory_ids)],
            idempotency_key=self.idempotency_key(ctx),
        )


class EnqueueEnrichmentStage(BaseStage):
    """Stage 4 — S2 write-time enrichment enqueue (ADR-0055; AD-241; engine-core-spec §6.4).

    The owner's shape, in one stage: ``add()`` (this pipeline) stays synchronous and LLM-free —
    this stage does exactly ONE cheap durable write (a small job row: ids + namespace + hashes,
    never content, see ``mu_contracts.domain.model.enrichment`` module docstring) and returns. The
    LLM call happens later, off this request, in ``pipelines.enrichment_worker.EnrichmentWorker``.

    **NOT ledger-gated** (``idempotency_key`` -> ``""``, the SAME "not every stage needs the
    ledger" precedent ``PersistRawArtifactStage`` documents): the underlying write is ALREADY
    idempotent by construction — ``EnrichmentQueuePort.submit`` is idempotent on ``job.job_id``,
    which is itself a deterministic function of ``memory_id`` (:func:`_enrichment_job_id`). A
    crash-replay of this stage submits the byte-identical row; ``INSERT OR IGNORE`` (the shipped
    adapters) makes that a harmless no-op, never a duplicate job.

    **NEVER raises, NEVER fails ``add()``** (the owner's explicit rule: "a failed or slow
    enrichment must never degrade the unenriched memory"). Every way this stage can fail to
    enqueue — the queue rejecting for backpressure, or the port raising outright — becomes a
    ``DEGRADED`` :class:`StageOutcome` carrying a named ``StageDegraded`` event (DEV-STANDARDS: a
    degrade is a NAMED reason + emitted event, never a bare fallback), and the pipeline proceeds
    exactly as if this stage were absent. The raw memory this pipeline already wrote (STM, and
    MTM if promoted) is completely unaffected either way.
    """

    name = "enqueue_enrichment"

    def __init__(
        self,
        *,
        queue: EnrichmentQueuePort,
        ledger: StageLedger,
        clock: Clock,
    ) -> None:
        super().__init__(ledger=ledger, clock=clock)
        self._queue = queue

    def idempotency_key(self, ctx: PipelineContext) -> str:
        return ""  # not ledger-gated — see class docstring (submit() is self-idempotent).

    async def _execute(self, ctx: PipelineContext) -> StageOutcome:
        memory_ids = list(ctx.state.get("memory_ids") or [])
        if not memory_ids:
            # Nothing was written upstream (should be unreachable — WriteStmStage always mints
            # one) — no memory to enrich, quietly OK.
            return StageOutcome(status=StageStatus.OK, produced={"enrichment_enqueued": False})
        content_hash = str(ctx.state.get("content_hash") or "")
        job = EnrichmentJob(
            job_id=_enrichment_job_id(memory_ids[0]),
            namespace_parts=ctx.namespace.parts(),
            memory_id=memory_ids[0],
            content_hash=content_hash,
            enqueued_at=self._clock.now(),
        )
        try:
            accepted = await self._queue.submit(job)
        except Exception as exc:  # broad except is deliberate — see class docstring
            _log.warning(
                "enrichment_enqueue_failed",
                pipeline=ctx.pipeline,
                stage=self.name,
                memory_id=job.memory_id,
                error=type(exc).__name__,  # content-free: exception class name only
            )
            return StageOutcome(
                status=StageStatus.DEGRADED,
                reason="enrichment_enqueue_failed",
                produced={"enrichment_enqueued": False},
                events=[
                    StageDegraded(
                        pipeline=ctx.pipeline,
                        stage=self.name,
                        reason=DegradeReason.DURABLE_SUBSTRATE_DOWN.value,
                    )
                ],
            )
        if not accepted:
            # Bounded queue at capacity (backpressure) — a burst of writes sheds enrichment
            # rather than growing the job log unboundedly (DEV-STANDARDS: bounded queues, never
            # unbounded). The memory itself was already written by the earlier stages; only its
            # future enrichment is skipped.
            return StageOutcome(
                status=StageStatus.DEGRADED,
                reason="enrichment_queue_full",
                produced={"enrichment_enqueued": False},
                events=[
                    StageDegraded(
                        pipeline=ctx.pipeline,
                        stage=self.name,
                        reason=DegradeReason.BACKPRESSURE_SHED.value,
                    )
                ],
            )
        return StageOutcome(
            status=StageStatus.OK,
            produced={"enrichment_enqueued": True, "enrichment_job_id": job.job_id},
        )


def _enrichment_job_id(memory_id: str) -> str:
    """Deterministic job id (§7.1 id-stability applied to the enrichment queue): one memory is
    enriched (at most) once, and a crash-replayed enqueue resubmits the identical row."""
    return f"enr_{memory_id}"
