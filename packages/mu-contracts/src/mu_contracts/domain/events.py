"""The frozen DOMAIN EVENT CATALOG + DegradeReason (CANONICAL-CONTRACTS.md §2/§3/§5).

Every event is declared ONCE here and imported by both planes — the generated seam that lets
the compiler (not prose) enforce the single vocabulary (CANONICAL §"Generated seam"). All
payloads are CONTENT-FREE (§3): ids, content hashes, counts, enum values, timestamps, and
``Namespace``/``ShareableRef`` refs only — never a message body, prompt, memory text, or secret.

DEV-STANDARDS deviation (recorded): CANONICAL §2/§3 illustrate events as frozen dataclasses.
DEV-STANDARDS rule 2 is BINDING — "pydantic v2 for all models/DTOs; never ``dataclass``" — and
events are DTOs, so they are frozen pydantic v2 models here. The FIELD SET and freezing are
identical to the pinned shapes; only the base machinery differs. The content-free rule is
enforced STRUCTURALLY: ``DomainEvent.__pydantic_init_subclass__`` refuses any subclass with a
content-bearing field name (``body``/``text``/``content``/``message``/…) — a construction rule,
not a review guideline (§3.1, platform-layer0 §8.3).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mu_contracts.domain.model.conflict import ResolutionOrigin
from mu_contracts.domain.model.device import ClientMode, DevicePlatform, DeviceState, PrivacyTier
from mu_contracts.domain.model.device_sync import SyncOp
from mu_contracts.domain.model.governance import Direction, ShareableRef, TransferState
from mu_contracts.domain.model.memory import Namespace, State, Tier
from mu_contracts.domain.model.room import MessageKind, ParticipantKind
from mu_contracts.domain.model.scope import AgentKind

__all__ = [
    "RESERVED_REASONS",
    "SYNC_CLASS_ALWAYS",
    "SYNC_CLASS_WHEN_DEVICE_SYNC",
    "AccessDenied",
    "ActivityCaptured",
    "AgentChunkProduced",
    "AgentEnrolled",
    "AgentRetired",
    "AgentVisibilityGranted",
    "AgentVisibilityRevoked",
    "ApiKeyIssued",
    "ApiKeyRevoked",
    "ApiKeyRotated",
    "AskAnswered",
    "BackpressureRejected",
    "BoundAgentDispatchCompleted",
    "BoundAgentDispatchFailed",
    "BoundAgentDispatchStarted",
    "BoundAgentDispatched",
    "BoundAgentResulted",
    "BoundAgentSuppressed",
    "BoundAgentUnavailable",
    "BoundaryViolationBlocked",
    "CaptureBatchReceived",
    "CaptureSourceHalted",
    "CaptureSourceStarted",
    "ComposeRequested",
    "ComposedContextStale",
    "ConflictDetected",
    "ConflictDismissed",
    "ConflictPolicyChanged",
    "ConflictResolutionPending",
    "ConflictResolved",
    "ConsolidationCompleted",
    "ConsolidationDeferred",
    "ContextComposed",
    "ContextIndexed",
    "ContextInjected",
    "ContextPulled",
    "ContextShared",
    "DegradeReason",
    "DegradedModeEntered",
    "DeviceActivated",
    "DeviceEnrolled",
    "DeviceKeyEpochRotated",
    "DeviceRevoked",
    "DomainEvent",
    "DurableSubstrateUnavailable",
    "FactDropped",
    "FactsExtracted",
    "GrantDerived",
    "GrantExpired",
    "GrantIssued",
    "GrantRevoked",
    "HostInjectionRendered",
    "HostInjectionSkipped",
    "HostPreCompact",
    "IngestCompleted",
    "MemoryCaptured",
    "MemoryDemoted",
    "MemoryGarbageCollected",
    "MemoryHealthAlert",
    "MemoryPinned",
    "MemoryPromoted",
    "MemoryQuarantined",
    "MemorySuperseded",
    "MemoryUnpinned",
    "NoteLinkSkipped",
    "NotificationCreated",
    "OrgProvisioned",
    "OutboxRecordDeadLettered",
    "PacketProposed",
    "PacketTransitioned",
    "ParticipantJoined",
    "ParticipantLeft",
    "PersonaUpdated",
    "PipelineCompleted",
    "PipelineHalted",
    "PipelineShed",
    "PipelineStarted",
    "PreCompactIntercepted",
    "PresenceChanged",
    "PrivateDeltaAppended",
    "PrivateDeltaApplied",
    "PrivateHostedRead",
    "ProvenanceRecorded",
    "RecallCacheRefreshed",
    "RevokeCascadeCompleted",
    "RevokeCascadeStarted",
    "RevokeCascadeStepFailed",
    "RoomClosed",
    "RoomContextInjected",
    "RoomLogUnavailable",
    "RoomMessageCaptured",
    "RoomMessagePosted",
    "RoomOpened",
    "RoomSubscriberLagged",
    "RoomTransportDegraded",
    "SalienceRecomputed",
    "SharedAgentSessionOpened",
    "SharedAgentSessionPromoted",
    "SharedMemoryUpserted",
    "SleeptimeTick",
    "StageDegraded",
    "SubagentSpawned",
    "SubscriptionCreated",
    "SubscriptionMatched",
    "SurfaceDegraded",
    "TrustLedgerAppended",
    "WorkspaceCreated",
]

# Field names that would smuggle raw memory content onto a content-free bus (CANONICAL §3.1).
_FORBIDDEN_EVENT_FIELDS = frozenset(
    {"body", "text", "content", "message", "prompt", "raw", "blob", "secret"}
)


class DomainEvent(BaseModel):
    """Base of the frozen event catalog. Frozen + ``extra='forbid'``; enforces the content-free
    field-name rule on every subclass at class-definition time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        offenders = _FORBIDDEN_EVENT_FIELDS & set(cls.model_fields)
        if offenders:
            raise TypeError(
                f"{cls.__name__} declares content-bearing field(s) {sorted(offenders)}; "
                "DomainEvent payloads are content-free (CANONICAL §3.1) — carry a content_hash "
                "and read the body from the owning store by id."
            )


class DegradeReason(StrEnum):
    """The one closed union of degrade reasons across every subsystem (CANONICAL §2). A degrade
    is a NAMED reason + emitted event + metric, never a bare fallback. SYNC-CLASS members are the
    ONLY user-visible reasons (via ``SyncStatusView``); everything else is operator-only."""

    # --- memory-layer ---
    LTM_UNAVAILABLE = "ltm_unavailable"
    DATE_EXTRACTION_FALLBACK = "date_extraction_fallback"
    LTM_GRAPH_LATENCY_CAP = "ltm_graph_latency_cap"
    # --- daemon (S1/S2) ---
    DAEMON_UNREACHABLE_SPOOLED = "daemon_unreachable_spooled"  # SYNC-CLASS
    SERVER_UNREACHABLE = "server_unreachable"  # SYNC-CLASS when component="device_sync"
    RECALL_CORE_DOWN = "recall_core_down"
    STALE_INJECTION = "stale_injection"
    LTM_EXTRACTION_DEFERRED = "ltm_extraction_deferred"
    UNSUPPORTED_HOST_FORMAT = "unsupported_host_format"
    CAPTURE_SCHEMA_DRIFT = "capture_schema_drift"
    # --- platform / workflow ---
    TEMPORAL_UNAVAILABLE = "temporal_unavailable"
    # --- PIPELINES ---
    NOTE_LINK_SKIPPED = "note_link_skipped"
    FACT_DROPPED = "fact_dropped"
    BACKPRESSURE_SHED = "backpressure_shed"
    DURABLE_SUBSTRATE_DOWN = "durable_substrate_down"
    # --- governance ---
    REVOKE_CASCADE_DISABLED = "revoke_cascade_disabled"
    REVOKE_ACK_PENDING = "revoke_ack_pending"
    # --- rooms ---
    ROOM_LOG_UNAVAILABLE = "room_log_unavailable"
    ROOM_SUBSCRIBER_LAGGED = "room_subscriber_lagged"
    ROOM_TRANSPORT_DEGRADED = "room_transport_degraded"
    BOUND_AGENT_UNAVAILABLE = "bound_agent_unavailable"
    PRESENCE_STORE_DOWN = "presence_store_down"
    # --- surface ---
    SURFACE_COMPONENT_DOWN = "surface_component_down"
    # --- cross-visibility recall (federate-live, §7.9) ---
    SHARED_RECALL_UNAVAILABLE = "shared_recall_unavailable"
    ARTIFACT_HYDRATION_BUDGET = "artifact_hydration_budget"
    # --- topology / hosting — RESERVED, NOT EMITTED (E2E tier deferred, §7.18) ---
    E2E_CLIENT_SIDE_RECALL = "e2e_client_side_recall"  # RESERVED
    E2E_NO_SERVER_EMBED = "e2e_no_server_embed"  # RESERVED
    E2E_NO_SERVER_CONSOLIDATION = "e2e_no_server_consolidation"  # RESERVED
    HOSTED_E2E_STORAGE_ONLY = "hosted_e2e_storage_only"  # RESERVED
    # --- device registry / sync (§7.11, §7.14-§7.17) ---
    SYNC_BACKFILL_WINDOW_EXCEEDED = "sync_backfill_window_exceeded"  # SYNC-CLASS
    DEVICE_WIPE_ACK_PENDING = "device_wipe_ack_pending"  # SYNC-CLASS
    KEY_ROTATION_PENDING = "key_rotation_pending"  # RESERVED with §7.18
    SYNC_STALLED = "sync_stalled"  # SYNC-CLASS
    SYNC_DELTA_DEAD_LETTERED = "sync_delta_dead_lettered"  # SYNC-CLASS
    PRIMARY_SYNC_LAG = "primary_sync_lag"  # SYNC-CLASS
    # --- gateway ---
    GATEWAY_UPSTREAM_UNAVAILABLE = "gateway_upstream_unavailable"
    GATEWAY_REGION_UNAVAILABLE = "gateway_region_unavailable"
    # --- S1 install-time self-check (§7.13) ---
    HOST_WIRING_ABSENT = "host_wiring_absent"
    # --- storage-pluggable backends (operator-consumer, MetricSink only) ---
    KV_NONDURABLE = "kv_nondurable"
    RECENCY_ZSET_UNAVAILABLE = "recency_zset_unavailable"
    LLM_UNAVAILABLE_HEURISTIC = "llm_unavailable_heuristic"
    VECTOR_SINGLE_NODE = "vector_single_node"
    RELATIONAL_INDEX_REDUCED = "relational_index_reduced"
    # --- model-layer (§7.2 ModelSettings; operator) ---
    MODEL_GROUP_UNAVAILABLE = "model_group_unavailable"
    MODEL_LOCAL_UNAVAILABLE_FELL_BACK = "model_local_unavailable_fell_back"
    # --- engine-core conflict (§7.20 async conflict; operator) ---
    CONFLICT_ADJUDICATION_DEGRADED = "conflict_adjudication_degraded"
    CONFLICT_PENDING_SUPPRESSED = "conflict_pending_suppressed"
    CONFLICT_MANUAL_BACKLOG = "conflict_manual_backlog"
    # write-after-read Qdrant point visibility lag on cross-store supersede-invalidate;
    # bounded retry next sweep, the loser stays SUPERSEDED in the graph (source of truth)
    MTM_INVALIDATE_POINT_ABSENT = "mtm_invalidate_point_absent"  # ADR 0037
    # --- trust-ledger (§7.25; operator) ---
    TRUST_LEDGER_UNAVAILABLE = "trust_ledger_unavailable"
    # --- lifecycle / hosted-mirror consent (§4, §16; X4 consent gate, operator) ---
    # server is barred from taking the user-grained lifecycle-sweep lease on an offline
    # PRIVATE device because hosted_mirror_consent=NOT_CONSENTED; memory ages only once
    # the device reconnects, never a silent freeze (spec §4, AC-4.4)
    OFFLINE_FAILOVER_NOT_CONSENTED = "offline_failover_not_consented"  # ADR 0032


# The closed set of user-visible reasons (CANONICAL §2 / platform-layer0 §12). The only way to
# make a degrade user-visible is membership here; there is no ad-hoc user-facing error channel.
SYNC_CLASS_ALWAYS: frozenset[DegradeReason] = frozenset(
    {
        DegradeReason.SYNC_STALLED,
        DegradeReason.SYNC_DELTA_DEAD_LETTERED,
        DegradeReason.SYNC_BACKFILL_WINDOW_EXCEEDED,
        DegradeReason.PRIMARY_SYNC_LAG,
        DegradeReason.DEVICE_WIPE_ACK_PENDING,
        DegradeReason.DAEMON_UNREACHABLE_SPOOLED,
    }
)
# SERVER_UNREACHABLE is user-visible ONLY when component == "device_sync".
SYNC_CLASS_WHEN_DEVICE_SYNC: frozenset[DegradeReason] = frozenset(
    {DegradeReason.SERVER_UNREACHABLE}
)
# The RESERVED reasons that MUST have zero emit sites while §7.18 holds (conformance test).
RESERVED_REASONS: frozenset[DegradeReason] = frozenset(
    {
        DegradeReason.E2E_CLIENT_SIDE_RECALL,
        DegradeReason.E2E_NO_SERVER_EMBED,
        DegradeReason.E2E_NO_SERVER_CONSOLIDATION,
        DegradeReason.HOSTED_E2E_STORAGE_ONLY,
        DegradeReason.KEY_ROTATION_PENDING,
    }
)


# =============================================================================================
# §2 — the single most-emitted event
# =============================================================================================
class DegradedModeEntered(DomainEvent):
    """The frozen 4-field degrade event (CANONICAL §2). Emittable identically by every subsystem;
    platform is the sole OPERATOR consumer (never the sole author). ``detail`` is optional free
    text and NEVER raw memory content (allowed here — it is not a content-bearing field name)."""

    component: str  # what degraded: "ltm" | "mtm" | "capture:codex" | "temporal" | ...
    mode: str  # the NAMED degraded path: "recall_mtm_only" | "inline_dispatch" | ...
    reason: DegradeReason
    detail: str | None = None

    def user_visible(self) -> bool:
        return self.reason in SYNC_CLASS_ALWAYS or (
            self.reason in SYNC_CLASS_WHEN_DEVICE_SYNC and self.component == "device_sync"
        )


# =============================================================================================
# §5.1 — core lifecycle (capture → distill → promote)
# =============================================================================================
class ActivityCaptured(DomainEvent):
    activity_id: str
    host: str
    kind: str
    session_id: str


class MemoryCaptured(DomainEvent):
    namespace: Namespace
    ids: list[str]
    tier: Tier = Tier.STM


class IngestCompleted(DomainEvent):
    namespace: Namespace
    memory_ids: list[str]


class SalienceRecomputed(DomainEvent):
    namespace: Namespace
    id: str
    score: float
    strength: float


class MemoryPromoted(DomainEvent):
    namespace: Namespace
    id: str
    frm: Tier
    to: Tier
    reason: str


class MemoryDemoted(DomainEvent):
    namespace: Namespace
    id: str
    tier: Tier
    # Additive (ADR 0034 / PROPOSED P5, R5): distinguishes an actual tier-down move
    # (MTM->STM forgetting-curve demotion) from pure archival. ``None`` means
    # "no tier change — this is an archival-only demotion" (the pre-existing
    # to_state=ARCHIVED behavior); the demotion service (S1-02) sets a concrete
    # ``Tier`` whenever the transition is a real tier-down move, so misusing
    # to_state=ARCHIVED for a tier move is no longer necessary. Optional with a
    # ``None`` default (not a bare additive Tier) so every existing
    # ``MemoryDemoted(...)`` call site — which only ever modeled archival —
    # keeps constructing without change.
    to_tier: Tier | None = None
    to_state: State = State.ARCHIVED
    retention: float


class MemoryGarbageCollected(DomainEvent):
    namespace: Namespace
    id: str
    prior_state: State


class ConflictDetected(DomainEvent):
    namespace: Namespace
    incoming_id: str
    candidate_ids: list[str]
    method: str


class MemorySuperseded(DomainEvent):
    namespace: Namespace
    loser_id: str
    winner_id: str
    valid_at: datetime


class MemoryQuarantined(DomainEvent):
    namespace: Namespace
    id: str
    reason: str
    confidence: float


class FactsExtracted(DomainEvent):
    namespace: Namespace
    count: int


class ConsolidationCompleted(DomainEvent):
    namespace: Namespace
    facts_n: int
    superseded_n: int


class ConsolidationDeferred(DomainEvent):
    namespace: Namespace


class SleeptimeTick(DomainEvent):
    namespace: Namespace


# =============================================================================================
# §5.2 — host / capture / injection (S1 daemon — LOCAL only)
# =============================================================================================
class HostPreCompact(DomainEvent):
    session_id: str
    transcript_path: str  # a filesystem path, not memory content
    cwd: str


class PreCompactIntercepted(DomainEvent):
    session_id: str
    pending_flushed: int


class CaptureSourceStarted(DomainEvent):
    host: str
    schema_version: str


class CaptureSourceHalted(DomainEvent):
    host: str
    schema_version: str
    raw_sample_sha256: str  # a hash of the drifted sample — content-free


class OutboxRecordDeadLettered(DomainEvent):
    seq: int
    reason: str


class RecallCacheRefreshed(DomainEvent):
    session_id: str
    etag: str


class HostInjectionRendered(DomainEvent):
    session_id: str
    staleness: float


class HostInjectionSkipped(DomainEvent):
    session_id: str
    reason: str


class BoundAgentDispatchStarted(DomainEvent):
    session: str
    principal: str
    correlation_id: str


class BoundAgentDispatchCompleted(DomainEvent):
    session: str
    principal: str
    correlation_id: str


class BoundAgentDispatchFailed(DomainEvent):
    session: str
    principal: str
    correlation_id: str
    reason: str


# =============================================================================================
# §5.3 — room events (S3 sessions-live-rooms — SHARED only)
# =============================================================================================
class RoomOpened(DomainEvent):
    room_id: str
    workspace_id: str
    namespace_id: str
    floor_policy: str


class RoomClosed(DomainEvent):
    room_id: str


class ParticipantJoined(DomainEvent):
    room_id: str
    principal_id: str
    kind: ParticipantKind


class ParticipantLeft(DomainEvent):
    room_id: str
    principal_id: str


class PresenceChanged(DomainEvent):
    room_id: str
    principal_id: str
    state: str


class RoomMessagePosted(DomainEvent):
    """The canonical content-free room message notification (CANONICAL §3.2) — NO body.
    Consumers read the full ``RoomMessage`` from ``RoomLogRepository.get(room_id, seq)``."""

    room_id: str  # == session_id
    seq: int  # per-room monotonic ordering authority
    author_principal_id: str
    author_kind: ParticipantKind
    kind: MessageKind
    correlation_id: str
    content_hash: str  # dedupe / provenance; NOT the body


class AgentChunkProduced(DomainEvent):
    session: str
    correlation_id: str
    chunk_seq: int
    content_hash: str  # ephemeral, NOT logged


class BoundAgentDispatched(DomainEvent):
    room_id: str
    agent_principal_id: str
    correlation_id: str


class BoundAgentResulted(DomainEvent):
    room_id: str
    agent_principal_id: str
    correlation_id: str


class BoundAgentSuppressed(DomainEvent):
    room_id: str
    agent_principal_id: str
    reason: str


class RoomContextInjected(DomainEvent):
    room_id: str
    seq: int
    source_memory_ids: list[str]
    mode: str


class RoomMessageCaptured(DomainEvent):
    room_id: str
    seq: int
    captured_memory_ids: list[str]


class RoomTransportDegraded(DomainEvent):
    room_id: str
    principal_id: str
    transport_kind: str


class RoomSubscriberLagged(DomainEvent):
    room_id: str
    principal_id: str
    dropped_count: int


class RoomLogUnavailable(DomainEvent):
    room_id: str


class BoundAgentUnavailable(DomainEvent):
    room_id: str
    agent_principal_id: str


# =============================================================================================
# §5.4 — governance / transfer (the moat — SHARED only)
# =============================================================================================
class ComposeRequested(DomainEvent):
    namespace: Namespace
    memory_ids: list[str]
    artifact_ids: list[str]
    intent: str  # short label, NOT the generated body


class ContextComposed(DomainEvent):
    composed_id: str
    source_refs: list[ShareableRef]


class ContextIndexed(DomainEvent):
    index_id: str
    namespace: Namespace


class ComposedContextStale(DomainEvent):
    """A source of an immutable ``ComposedContext`` left the live set (CANONICAL §7.10 [G8];
    ``governance-transfer-core-spec.md:476``, listed among §13's emitted events at :875).

    **Observational by contract.** On ``MemorySuperseded``/``MemoryDemoted``/
    ``MemoryGarbageCollected`` of a source, governance records ``sources_superseded_count`` and
    emits this. The composed snapshot **stays immutable** and **no downstream grant is auto-
    severed** — that is a policy toggle (§13-Q3), not this event's job. Only the freshness marker
    changes, which is why the count rides the event and no state does.

    Declared here because §13 lists it among the governance events emitted and ``events.py`` did
    not have it (AD-69), so the one signal that a bundle has gone stale had nowhere to be
    published.
    """

    composed_id: str
    #: How many of this bundle's sources have left the live set. A COUNT, never the source bodies
    #: — the refs are recoverable from the ``COMPOSED_FROM`` provenance fan-out (spec:476).
    sources_superseded_count: int = Field(ge=1)


class ContextShared(DomainEvent):
    grant_id: str
    to_session: str
    object_ref: ShareableRef


class ContextPulled(DomainEvent):
    grant_id: str
    object_ref: ShareableRef
    to_local_session: str


class GrantIssued(DomainEvent):
    grant_id: str
    object_ref: ShareableRef
    grantor: str
    grantee: str
    direction: Direction


class GrantDerived(DomainEvent):
    grant_id: str
    parent_grant_id: str
    depth: int


class GrantRevoked(DomainEvent):
    grant_id: str
    cascade_root_grant_id: str
    by: str


class GrantExpired(DomainEvent):
    grant_id: str


class ProvenanceRecorded(DomainEvent):
    stream_id: str
    version: int
    action: str


class AccessDenied(DomainEvent):
    subject: str
    permission: str
    object_ref: ShareableRef
    reason: str


class BoundaryViolationBlocked(DomainEvent):
    object_ref: ShareableRef
    visibility: str
    where: str


class PacketProposed(DomainEvent):
    packet_id: str
    source_session: str
    target_sessions: list[str]
    policy_id: str


class PacketTransitioned(DomainEvent):
    packet_id: str
    to: TransferState


class SubscriptionCreated(DomainEvent):
    grant_id: str
    source_session: str


class SubscriptionMatched(DomainEvent):
    grant_id: str
    index_id: str


class RevokeCascadeStarted(DomainEvent):
    root_grant_id: str
    descendant_count: int


class RevokeCascadeCompleted(DomainEvent):
    """The cascade's terminal receipt-material (AD-42).

    ⚠ **"Completed" must never hide a still-live LOCAL copy or an unpurged cache.** Until this
    change the event carried ``revoked_count`` alone, so the two counts that decide whether the
    revoke actually SETTLED had no home on it: ``governance-transfer-core-spec.md:762`` emits
    ``RevokeCascadeCompleted(root, revoked_count, ack_pending_count, cache_entries_purged)``,
    ``design-governance-transfer.puml:659`` draws exactly that triple, and
    ``trust-ledger-spec.md:366`` requires ``counts={"revoked","ack_pending","cache_purged"}`` to be
    DERIVABLE from this event so the projector can append ``REVOKE_SETTLED`` and build the
    receipt. With the fields absent, the producer routed them onto the trust-ledger row instead
    and said so in a code comment — the receipt's honesty (``state=PARTIAL`` when
    ``ack_pending > 0``, trust-ledger:388-389) depended on a side channel.

    Both counts default to ``0`` so every existing producer keeps constructing unchanged, and
    ``0`` is the honest reading of what those producers measure: no ack outstanding, nothing
    purged. A count is content-free (CANONICAL §3).
    """

    root_grant_id: str
    revoked_count: int
    #: LOCAL copies whose daemon never returned ``revoke_ack`` within ``revoke_ack_budget_s``
    #: (spec:867). ``> 0`` is what forces the receipt to ``PARTIAL`` — the event may not claim
    #: "revoked everywhere" when it was not.
    ack_pending_count: int = Field(default=0, ge=0)
    #: Warm-cache entries ACTUALLY purged — "reflects only actual purges" (spec:868), so a purge
    #: that exhausted its attempts lowers this number rather than being rounded up into success.
    cache_entries_purged: int = Field(default=0, ge=0)


class RevokeCascadeStepFailed(DomainEvent):
    root_grant_id: str
    grant_id: str
    error: str


# =============================================================================================
# §5.5 — surface + framework + platform
# =============================================================================================
class SharedMemoryUpserted(DomainEvent):
    memory_id: str
    superseded_id: str | None = None


class AskAnswered(DomainEvent):
    session_id: str
    memory_ids: list[str]
    model: str


class ContextInjected(DomainEvent):
    session_id: str
    memory_ids: list[str]
    tokens: int


class CaptureBatchReceived(DomainEvent):
    count: int


class SurfaceDegraded(DomainEvent):
    component: str
    reason: str


class PipelineStarted(DomainEvent):
    pipeline: str
    namespace: Namespace


class PipelineCompleted(DomainEvent):
    pipeline: str
    namespace: Namespace


class PipelineHalted(DomainEvent):
    pipeline: str
    namespace: Namespace


class PipelineShed(DomainEvent):
    pipeline: str
    namespace: Namespace


class StageDegraded(DomainEvent):
    pipeline: str
    stage: str
    reason: str


class NoteLinkSkipped(DomainEvent):
    pipeline: str
    stage: str
    reason: str


class FactDropped(DomainEvent):
    pipeline: str
    stage: str
    reason: str


class DurableSubstrateUnavailable(DomainEvent):
    pipeline: str
    namespace: Namespace


class BackpressureRejected(DomainEvent):
    pipeline: str
    namespace: Namespace


# =============================================================================================
# §5.5a — device registry + private sync
# =============================================================================================
# ``org_id`` on all five rows below closes **O-21** (phase-3 spec §14), with the matching
# ``CANONICAL-CONTRACTS.md`` §5 payload amendment authored in the same change. The gap was real,
# not cosmetic: ``org`` is the BILLING/tenancy root (ADR 0026) and ``workspace_id`` alone does not
# identify it, so on a multi-tenant plane a metering or audit consumer of these events could not
# attribute a device lifecycle to the org that pays for it — and a consumer that tried would have
# to join back to a control-plane table these events exist to avoid needing. It is the first
# field on each payload because it is the first term of ``device_id``'s own hash
# (``CANONICAL:647``: *"keyed on `org` post-ADR-0026"*).
class DeviceEnrolled(DomainEvent):
    device_id: str
    org_id: str
    workspace_id: str
    principal_id: str
    platform: DevicePlatform
    client_mode: ClientMode
    privacy_tier: PrivacyTier
    state: DeviceState


class DeviceActivated(DomainEvent):
    device_id: str
    org_id: str
    workspace_id: str
    principal_id: str
    at_seq: int


class DeviceRevoked(DomainEvent):
    device_id: str
    org_id: str
    workspace_id: str
    principal_id: str
    by: str
    #: A :class:`~mu_contracts.domain.model.device.RevokeReason` VALUE, never free text (D-34).
    #: Typed ``str`` because the event catalog pins it so; the producing service face takes the
    #: closed enum, which is what makes the guarantee enforceable. ``_FORBIDDEN_EVENT_FIELDS`` is
    #: a field-NAME guard and cannot see an operator-supplied sentence in a field called
    #: ``reason`` — so the discipline lives at the producer, and an acceptance gate asserts it.
    reason: str


class DeviceKeyEpochRotated(DomainEvent):
    """RESERVED with §7.18 — not emitted while E2E is deferred."""

    org_id: str
    workspace_id: str
    principal_id: str
    epoch: int
    member_device_ids: list[str]


class PrivateDeltaAppended(DomainEvent):
    org_id: str
    workspace_id: str
    principal_id: str
    seq: int
    op: SyncOp
    memory_id: str
    content_hash: str
    origin: str  # "device" | "server" — the appender, DO NOT branch on it for resolution


class PrivateDeltaApplied(DomainEvent):
    #: O-21's fix, extended to the sixth row of this block. It was left off when the five rows
    #: above gained it, and the asymmetry was flagged in review: ``org`` is the BILLING/tenancy
    #: root (ADR 0026) and ``workspace_id`` alone does not identify it, so a metering or audit
    #: consumer of this event could not attribute an applied delta to the org that pays for it.
    #: Free to add today precisely because the event has ZERO producers — the projector is build
    #: step 9 — so there is no payload in flight to migrate.
    org_id: str
    workspace_id: str
    principal_id: str
    device_id: str
    seq: int


# =============================================================================================
# §5.7 — new subsystem rows (Groups 2/5)
# =============================================================================================
class ConflictResolutionPending(DomainEvent):
    namespace: Namespace
    conflict_id: str
    incoming_id: str
    candidate_ids: list[str]
    policy: str


class ConflictResolved(DomainEvent):
    namespace: Namespace
    conflict_id: str
    winner_id: str
    loser_ids: list[str]
    resolution_origin: ResolutionOrigin


class ConflictDismissed(DomainEvent):
    namespace: Namespace
    conflict_id: str
    by: str


class ConflictPolicyChanged(DomainEvent):
    namespace: Namespace
    policy: str


class PersonaUpdated(DomainEvent):
    namespace: Namespace
    version: int
    slots_changed: int
    brief_etag: str


class SharedAgentSessionOpened(DomainEvent):
    session_id: str
    org_id: str
    owner_principal_ids: list[str]


class SharedAgentSessionPromoted(DomainEvent):
    session_id: str
    from_owner_set: list[str]
    to_owner_set: list[str]


class AgentEnrolled(DomainEvent):
    agent_principal_id: str
    org_id: str
    kind: AgentKind


class SubagentSpawned(DomainEvent):
    agent_principal_id: str
    parent_agent_principal_id: str
    agent_path: str


class AgentVisibilityGranted(DomainEvent):
    agent_principal_id: str
    object_ref: ShareableRef


class AgentVisibilityRevoked(DomainEvent):
    agent_principal_id: str
    object_ref: ShareableRef


class AgentRetired(DomainEvent):
    agent_principal_id: str
    reason: str


class MemoryPinned(DomainEvent):
    namespace: Namespace
    id: str
    by: str


class MemoryUnpinned(DomainEvent):
    namespace: Namespace
    id: str
    by: str


class NotificationCreated(DomainEvent):
    notification_id: str
    principal_id: str
    category: str
    kind: str
    seq: int


class MemoryHealthAlert(DomainEvent):
    namespace: Namespace
    signal: str
    count: int


class TrustLedgerAppended(DomainEvent):
    org_id: str
    seq: int
    action: str


class ApiKeyIssued(DomainEvent):
    org_id: str
    workspace_id: str
    key_id: str
    by: str


class ApiKeyRotated(DomainEvent):
    org_id: str
    workspace_id: str
    key_id: str
    rotated_from_key_id: str


class ApiKeyRevoked(DomainEvent):
    org_id: str
    workspace_id: str
    key_id: str
    by: str
    reason: str


class OrgProvisioned(DomainEvent):
    org_id: str
    residency_region: str


class WorkspaceCreated(DomainEvent):
    org_id: str
    workspace_id: str
    by: str


class PrivateHostedRead(DomainEvent):
    """The audited-no-look event (CANONICAL §1 rule 6 / §7.18): every server read of a
    ``visibility=PRIVATE`` hosted partition emits this (content-free — no body)."""

    org_id: str
    principal_id: str
    object_ref: ShareableRef
    purpose: str
