"""The MemoryUniverseError hierarchy (platform-layer0-spec §6; CANONICAL-CONTRACTS §1.3).

ONE typed error hierarchy. Adapters RAISE typed errors; the *application* decides what to do
(platform-layer0-spec §6, §12). ``except: pass`` and silent fallback are BANNED — an unresolved
failure is either a raise or a NAMED degrade (see ``mu_engine.platform.degradation``), never a
swallow (DEV-STANDARDS rule 8: fail-loud, no silent fallback).

Home: ``mu-contracts`` (platform-layer0-spec §0.2) — errors are part of the shared wire vocabulary
so every plane/package maps the SAME hierarchy. This module imports only the stdlib (contracts is
pydantic-only; plain ``Exception`` subclasses carry no pydantic dependency).
"""

from __future__ import annotations

__all__ = [
    "AuthorizationError",
    "BackendUnavailableError",
    "BoundaryViolationError",
    "BudgetExceededError",
    "BusUnavailableError",
    "ConflictUnresolvedError",
    "DomainError",
    "DuplicateComponentError",
    "GovernanceError",
    "GrantExpiredError",
    "GrantNotFoundError",
    "GrantRevokedError",
    "HybridMirrorRequiredError",
    "IllegalConflictTransitionError",
    "IllegalTransitionError",
    "InvalidBackendError",
    "InvalidTransferStateError",
    "LlmNotConfiguredError",
    "MandatoryBackendMissingError",
    "MemoryLayerError",
    "MemoryNotFoundError",
    "MemoryUniverseError",
    "NamespaceIsolationError",
    "PacketExpiredError",
    "PermissionNarrowingError",
    "PinAuthorizationError",
    "PinLimitExceededError",
    "PinPartiallyAppliedError",
    "PinTargetNotFoundError",
    "PinTargetNotPinnableError",
    "PinnedTransitionBlocked",
    "PlaneFieldRejectedError",
    "PrivacyTierConflictError",
    "PrivacyTierNotAvailableError",
    "ProvenanceIntegrityError",
    "ProviderError",
    "ReshareDepthExceededError",
    "RevocationConflictError",
    "SchemaDriftError",
    "SettingsValidationError",
    "StoreUnavailableError",
    "SubModelProviderDisabledError",
    "SurfaceVerbNotImplementedError",
    "TenancyViolationError",
    "TierCapabilityUnavailableError",
    "TierRepositoryUnavailableError",
    "UnknownBackendError",
    "UnknownComponentError",
    "UnknownPrincipalError",
    "UnknownRoleError",
    "VectorNotFilterableError",
    "WorkflowUnavailableError",
]


class MemoryUniverseError(Exception):
    """Root of the typed error hierarchy (platform-layer0-spec §6)."""


# Subsystem specs that say ``class X(DomainError)`` (governance, sync-devices) mean this
# exact root; ``DomainError`` is a spelling alias, not a distinct class.
DomainError = MemoryUniverseError


# ── wiring / config ────────────────────────────────────────────────────────────────────────
class UnknownComponentError(MemoryUniverseError):
    """A registry lookup for an unregistered key (fail-loud; never a silent default) — §7."""


class DuplicateComponentError(MemoryUniverseError):
    """A registry registration collided with an existing key — §7."""


class SettingsValidationError(MemoryUniverseError):
    """A bad/forbidden config value; raised AT LOAD, never defaulted silently — §1.1."""


class BackendUnavailableError(MemoryUniverseError):
    """The selected backend's optional extra is not installed (mu-local-and-sdk §)."""


# ── governance / transfer — the ONE hierarchy every refusal on this plane hangs from ───────
#
# ⚠ **``GovernanceError`` is declared HERE, above the security section, and that placement is the
# fix, not an accident of ordering (AD-89).** Two authorities disagreed about
# ``AuthorizationError``'s parent: ``platform-layer0-spec.md:320`` writes
# ``class AuthorizationError(MemoryUniverseError)`` and
# ``governance-transfer-core-spec.md:840-870`` writes ``class AuthorizationError(GovernanceError)``
# inside ONE hierarchy. The shipped code followed the first, so ``AuthorizationError`` and
# ``RevocationConflictError`` — between them the governance plane's most common refusal and its
# concurrency refusal — were NOT ``GovernanceError`` subclasses, and a consumer mapping
# ``GovernanceError`` to an HTTP status caught neither. Every one of those refusals reached a
# client as a bare 500.
#
# The specs are reconciled the ONLY way that loses nothing: ``GovernanceError`` becomes an
# INTERMEDIATE root, so ``AuthorizationError`` is still a ``MemoryUniverseError`` (layer-0's
# sentence stays true — it is a strictly weaker claim) and is ALSO a ``GovernanceError``
# (governance's sentence becomes true). Nothing that caught the old base stops catching it.
#
# ⚠ **The blast radius is real and is stated so it is not discovered later:**
# ``NamespaceIsolationError`` (a.k.a. ``TenancyViolationError``),
# ``SubModelProviderDisabledError``, ``PrivacyTierNotAvailableError``,
# ``PrivacyTierConflictError`` and ``HybridMirrorRequiredError`` all descend from
# ``AuthorizationError`` and therefore become ``GovernanceError`` subclasses too. That is
# semantically right — each IS a governance refusal — but it makes ORDER load-bearing in any
# error→status table: ``mu_engine.platform.exceptions._ERROR_TABLE`` is walked in sequence and
# ``AuthorizationError`` is its FIRST row, so tenancy denials keep collapsing to the
# non-enumerating ``404 NOT_FOUND`` and a later ``GovernanceError`` row could never shadow them.
# **Any table that adds a ``GovernanceError`` row MUST keep it BELOW ``AuthorizationError``**, or
# a probe regains the ability to tell "denied" from "absent".
class GovernanceError(MemoryUniverseError):
    """Base of the governance/transfer errors (governance-transfer-core-spec §13)."""


# ── security ───────────────────────────────────────────────────────────────────────────────
class AuthorizationError(GovernanceError):
    """Base for every authorization denial (platform-layer0-spec §6; re-parented under
    :class:`GovernanceError` per governance-transfer-core-spec §13 — see the note above)."""


class NamespaceIsolationError(AuthorizationError):
    """Cross-tenant read/write (a.k.a. TenancyViolationError) — §5. Denials use the
    non-enumerating ``safe_error_response(NOT_FOUND)`` envelope so a probe cannot distinguish
    'denied' from 'absent' (§5, §14)."""


# Governance/tenancy docs use this spelling for the same class.
TenancyViolationError = NamespaceIsolationError


class SubModelProviderDisabledError(AuthorizationError):
    """The submodel provider was selected without the ToS gate — §1 (S5)."""


class PrivacyTierNotAvailableError(AuthorizationError):
    """An E2E-tier enroll/refusal (E2E is RESERVED, refused at load) — §1."""


class PrivacyTierConflictError(AuthorizationError):
    """An enroll request's ``privacy_tier`` differs from the principal's existing fleet tier
    (fleet uniformity, sync-devices-gateway §A.5)."""


class HybridMirrorRequiredError(AuthorizationError):
    """A ``thin`` secondary enrolled under a mirror-disabled ``full_local`` primary
    (sync-devices-gateway §A.5)."""


class RevocationConflictError(GovernanceError):
    """A concurrent revoke lost a compare-and-set (sync-devices-gateway §A.5).

    Re-parented under :class:`GovernanceError` (AD-89): ``governance-transfer-core-spec.md:850``
    lists it in the governance hierarchy and maps it to ``409``. It sat directly under
    ``MemoryUniverseError``, so the one refusal a revoke cascade races on was invisible to a
    governance handler — a bare 500 on the exact path whose whole promise is a receipt.
    """


# ── infra ──────────────────────────────────────────────────────────────────────────────────
class StoreUnavailableError(MemoryUniverseError):
    """A backing store is unreachable — §14 (MTM loud; LTM degrades named)."""


class TierRepositoryUnavailableError(StoreUnavailableError):
    """A specific tier repository (STM/MTM/LTM) is down; the caller decides degrade
    (engine-core §13, memory-layer §)."""


class BusUnavailableError(MemoryUniverseError):
    """The event bus is unreachable — §8."""


class WorkflowUnavailableError(MemoryUniverseError):
    """The durable workflow runner (Temporal) is unreachable — §9 (~TemporalUnavailableError).
    On SHARED this may downgrade to a NAMED inline-dispatch degrade; never a claimed guarantee."""


class ProviderError(MemoryUniverseError):
    """An LLM/embedding provider call failed — §14 (answer path denies loud)."""


# ── domain / data quality ──────────────────────────────────────────────────────────────────
class SchemaDriftError(MemoryUniverseError):
    """A capture adapter saw an unexpected schema; HALTS that source loudly — §6, §14
    (``daemon.schema_drift_policy='halt'``)."""


class ConflictUnresolvedError(MemoryUniverseError):
    """A conflict could not be resolved by the adjudication path — §6."""


class BudgetExceededError(MemoryUniverseError):
    """A pinned latency/token/size budget was exceeded — §6."""


class IllegalTransitionError(MemoryUniverseError):
    """An illegal memory-lifecycle FSM edge (memory-layer §1.1; ``LifecyclePolicy`` fail-loud)."""


class IllegalConflictTransitionError(MemoryUniverseError):
    """An illegal conflict-lifecycle FSM edge (engine-core §7.6)."""


class MemoryLayerError(MemoryUniverseError):
    """Base of the memory-layer subsystem errors (memory-layer §)."""


# ── pinning (memory-health-pinning-spec §9, lines 372-378) ───────────────────────────────────
class PinAuthorizationError(MemoryLayerError):
    """The caller does not own the partition it tried to pin in, or tried to pin a SHARED-origin
    item while ``PinSettings.allow_shared_origin_pin`` is off (spec §5.2 step 1).

    Pin is RETENTION, never ACCESS (CANONICAL §7.26): this error means "you may not change this
    item's retention", never "you may not read it" — raising it neither widens nor narrows any
    caller's read set.
    """


class PinTargetNotFoundError(MemoryLayerError):
    """``PinRequest.memory_id`` does not resolve in the target partition (spec §5.2 step 3).

    Carries no id in its message (the non-enumerating-denial discipline of
    ``NamespaceIsolationError`` / ``mu_engine.platform.exceptions.safe_error_response``): a probe
    must not be able to distinguish "absent" from "not yours".
    """


class PinTargetNotPinnableError(MemoryLayerError):
    """The target resolves but has already LEFT the store's live set — ``SUPERSEDED`` /
    ``EXPIRED`` / ``DELETED`` (spec §5.2 step 3; the ``PinService.PINNABLE_STATES`` set).

    Pin is a RETENTION override, so pinning a settled exit is meaningless AND harmful: a pinned
    row is unconditionally GC-ineligible (CANONICAL §7.10), so the pin would strand a dead row in
    the graph forever with no live counterpart for the owner to act on. Refused loud rather than
    accepted as a no-op.

    NOT in the spec's §9 error list (lines 372-378) — reported as a spec addition, because §5.2
    step 3 recognises only "not found" and the shipped ``PINNABLE_STATES`` constant had no
    enforcement behind it at all.
    """


class PinLimitExceededError(MemoryLayerError):
    """The partition already holds ``PinSettings.max_pins_per_namespace`` pins (spec §5.2 step 2
    — the pin-explosion guard). Refused loud; never a silent no-op."""


class PinPartiallyAppliedError(MemoryLayerError):
    """A cross-store pin landed on SOME of the tiers holding the id and failed on others.

    **This error exists because the guarantee the spec asks for is convergence, not atomicity,
    and the difference has to be visible rather than papered over.** ``set_pinned`` is specified
    as an id-stable upsert "applied across every store the item resides in"
    (``ports/memory.py:52-56``). The stores are Redis/Valkey, Qdrant and FalkorDB — three
    separate network services with no shared transaction, no two-phase commit and no distributed
    log in front of them. ``IdempotentWriteScope`` buffers DOMAIN EVENTS and publishes them after
    the write step; it is an outbox for the bus, not a transaction across the stores. Each store
    can make its OWN leg atomic (a Redis ``MULTI``, a Qdrant ``set_payload`` on one point, a
    single Cypher ``SET``) and nothing can make the three atomic together. So a partial apply is
    not a hazard to be avoided — it is a guaranteed operating condition, and this is its name.

    **Raised, not swallowed, so the event never outruns the write.** ``PinService._commit`` runs
    the repository call as the write step of an ``IdempotentWriteScope`` and publishes
    ``MemoryPinned``/``MemoryUnpinned`` only if that step returns. Reporting a partial apply as
    success would publish a pin event for a pin that half-landed, which is strictly worse than no
    event: downstream converges on a state no store agrees with.

    **The landed legs are deliberately NOT rolled back.** A compensating write can fail exactly
    the way the original leg did, turning one inconsistency into two, and ``set_pinned`` is a
    full-field-group upsert with a caller-supplied ``at``/``by``/``reason`` — re-running it
    converges. The honest recovery is therefore "retry the whole call", and the landed legs
    simply re-converge. Leaving them is convergent, not lazy.

    **The two directions are NOT equally safe, and this type says which one you are in.** For a
    PIN, a leftover pinned leg is the conservative direction: a spuriously-pinned row is merely
    un-GC-able until reconciled, never lost data. For an UNPIN it is the dangerous direction —
    a leftover pinned leg strands the row as permanently GC-ineligible, which is precisely what
    ``PinService`` documents unpin as existing to prevent. ``applied``/``failed`` carry the tier
    names (content-free: tier enum values and the memory id, never memory text) so an operator
    can tell the two apart and reconcile, instead of inferring it from a bare failure.
    """

    def __init__(
        self, message: str, *, applied: frozenset[str], failed: frozenset[str], pinned: bool
    ) -> None:
        super().__init__(message)
        #: Tier names whose leg accepted the write (content-free — ``Tier`` values only).
        self.applied = applied
        #: Tier names whose leg refused or errored.
        self.failed = failed
        #: The DIRECTION that half-landed: ``True`` = pin (conservative leftover), ``False`` =
        #: unpin (a stranded, GC-ineligible leftover — the direction that needs reconciling).
        self.pinned = pinned


class TierCapabilityUnavailableError(MemoryUniverseError):
    """A bound backend cannot serve a capability the tier-repository port requires.

    Distinct from :class:`TierRepositoryUnavailableError`, and the distinction is the point:
    that one means "this store is DOWN, retry or degrade"; this one means "this store is UP and
    will never be able to answer, because the backend you bound has no such primitive". Three of
    the five vector backends (``pgvector``, ``chroma``, ``faiss``) expose only
    ``upsert``/``semantic``/``invalidate`` — no point-get and no enumeration primitive of any
    kind — so on those deployments the bounded partition walk and the cross-store pin genuinely
    cannot be served.

    Raised rather than answered with an empty page, because an empty page is indistinguishable
    from "this partition is healthy and holds nothing", which is exactly the silent wrong answer
    DEV-STANDARDS rule 8 forbids. The house absence rule applies: the capability is reported
    missing, never stubbed into a lie.
    """


# Naming note: the SPEC names this type verbatim (memory-health-pinning-spec §9, line 377)
# and CANONICAL §7.26 refers to it by that name; renaming it to `...Error` would fork the
# vocabulary between the design set and the code, which DEV-STANDARDS' spec-driven rule forbids.
class PinnedTransitionBlocked(IllegalTransitionError):  # noqa: N818
    """An EXIT edge was attempted on a pinned item without ``force_unpinned`` (spec §6.1).

    Subclasses ``IllegalTransitionError`` deliberately: a pinned item's exit edge is not a
    special case of policy, it is an ILLEGAL FSM EDGE, and every existing handler that maps
    ``IllegalTransitionError`` keeps working unchanged. An AUTOMATIC sweep never lets this
    escalate — it asks :meth:`LifecyclePolicy.permits` first and treats ``False`` as
    "skip, keep" (spec §9, line 381); the raise is reserved for a caller that wrongly drove an
    exit transition directly, which is a bug and must fail loud.
    """


# ── governance / transfer (governance-transfer-core §13) ─────────────────────────────────────
# ``GovernanceError`` itself is declared ABOVE the security section — see the note there.
class GrantNotFoundError(GovernanceError):
    """A grant id does not resolve."""


class GrantRevokedError(GovernanceError):
    """An action targeted an already-terminal (revoked) grant."""


class GrantExpiredError(GovernanceError):
    """An action targeted an expired grant."""


class PermissionNarrowingError(GovernanceError):
    """A re-share tried to widen permissions (monotonic narrowing violated)."""


class ReshareDepthExceededError(GovernanceError):
    """A re-share exceeded ``policy.max_reshare_depth``."""


# The six §13 refusals that were declared in the spec and existed in NO repo (AD-69). Until they
# landed here the transfer plane declared them locally, which put six members of a "single"
# hierarchy behind the commercial boundary where no open-plane caller could name them.
class BoundaryViolationError(GovernanceError):
    """A PRIVATE object crossed, or was about to cross, into the shared plane — or a shared write
    was routed at a private store (``BoundaryGuard.assert_shareable`` /
    ``assert_shared_plane_target``, spec:840-848).

    **FAIL LOUD, never a filter.** This is the one boundary the whole two-plane split exists to
    hold: a silent drop here would look identical to "nothing to share" while the object had
    already been classified for transit.
    """


class ProvenanceIntegrityError(GovernanceError):
    """A provenance-ledger version or hash conflict — an append whose ``expected_version`` did not
    match the stream head, or a chain that does not fold (spec:849).

    The ledger is append-only and per-stream monotonic; a version conflict means two writers
    believed they held the same head, and accepting either would produce an audit chain that
    silently lost a fact.
    """


class InvalidTransferStateError(GovernanceError):
    """An illegal ``ContextPacket`` FSM edge (spec:851; ``409``). The legal edges are the ONLY
    edges — a guard rejects everything else rather than tolerating a state the receipt path
    cannot explain."""


class PacketExpiredError(GovernanceError):
    """A packet was advanced after ``expires_at`` (spec:852; ``409``).

    Distinct from :class:`InvalidTransferStateError` on purpose: the requested EDGE was legal and
    the CLOCK refused it, and a caller that cannot tell those apart cannot tell "you may not do
    this" from "you are too late".
    """


class UnknownPrincipalError(GovernanceError):
    """A grant, ACL row or role membership named a principal that does not resolve (spec:853)."""


class UnknownRoleError(GovernanceError):
    """A role grant or membership named a role that does not resolve (spec:854).

    Separate from :class:`UnknownPrincipalError` because the accept-time explosion (§3) treats the
    two differently: an unknown PRINCIPAL yields no ACL row, while an unknown ROLE would silently
    yield ZERO rows for a grant that looked issued — an access that appears granted and grants
    nothing.
    """


# ── storage-pluggable build-time refusals (fail-loud, storage-pluggable §7) ──────────────────
class MandatoryBackendMissingError(BackendUnavailableError):
    """A mandatory store role (relational/vector/GRAPH) is unbound at container build."""


class InvalidBackendError(BackendUnavailableError):
    """An illegal backend selection (e.g. a graph fold — GRAPH is mandatory)."""


class VectorNotFilterableError(BackendUnavailableError):
    """A non-filterable (brute-force) vector store was selected for the SHARED plane — the one
    hard authz-completeness refusal (storage-pluggable §7)."""


class UnknownBackendError(UnknownComponentError):
    """An unknown store-backend registry key (storage-pluggable §7)."""


# ── plane-gating (design §2.5 "Unified verb surface"; build-plan Stage B ruling 1) ───────────
class PlaneFieldRejectedError(MemoryUniverseError):
    """A canonical-signature field was supplied for a plane the caller has not configured
    (sdk-engine-server-design.md §2.5: "Supplying a field that doesn't apply to the currently-
    configured plane is a REJECTION, not a silent no-op"). Raised by
    :func:`mu_contracts.validation.plane_gate.validate_plane_fields` — the ONE validator both
    ``LocalMemory`` (mu-local) and ``MemoryClient`` (mu-sdk-python) call, so a shared-plane field
    (``visibility``/``subject``/``predicate``/``object``) supplied with no shared plane configured,
    or a private-plane field (``user``/``session``/``agent``) supplied under a shared-only
    configuration, fails the SAME named way on both surfaces — never a silent drop (DEV-STANDARDS
    rule 8: fail-loud)."""

    def __init__(self, field: str, *, plane: str, reason: str) -> None:
        self.field = field
        self.plane = plane
        super().__init__(f"field {field!r} requires the {plane!r} plane to be configured: {reason}")


# ── verb surface / synthesis refusals (CO-3: folded from mu_engine.surface.facade /
# mu_local.errors duplicates into this ONE canonical home, per this module's own docstring
# "ONE typed error hierarchy" + "Home: mu-contracts") ─────────────────────────────────────────
class SurfaceVerbNotImplementedError(MemoryUniverseError):
    """A canonical verb has no engine-side implementation to delegate to yet (build-queue §13
    item 5). Raised by ``SurfaceFacade.promote``/``.demote``/``.share`` (mu-engine) and mapped by
    ``mu_engine_server.errors`` to HTTP 501 — NAMED so every layer imports ONE exception family,
    never a bare ``NotImplementedError`` (not part of the ``MemoryUniverseError`` wire hierarchy)
    and never a silent no-op (DEV-STANDARDS rule 8: fail-loud, no silent fallback)."""

    def __init__(self, verb: str, *, reason: str) -> None:
        self.verb = verb
        super().__init__(f"SurfaceFacade.{verb}() is not implemented: {reason}")


class MemoryNotFoundError(MemoryUniverseError):
    """A TARGETED lifecycle verb (``promote``/``demote``/``update``/``delete`` — the surface
    facade's single-memory verbs) was given a ``memory_id`` that resides in NONE of the tiers it
    can act on within the caller's η partition. NAMED (not a bare ``KeyError``/``None`` return) so
    every layer maps it the SAME way — HTTP 404 on the wire (``mu_engine_server.errors``), an SDK
    ``NotFoundError`` — and never a silent no-op (DEV-STANDARDS rule 8: fail-loud). The verbs guard
    a nonexistent id with THIS, exactly as they guard an invalid ``to_tier`` with a ``ValueError``
    (mapped to 400) — an honest error, never a fake success."""

    def __init__(self, memory_id: str, *, reason: str = "not resident in any tier") -> None:
        self.memory_id = memory_id
        super().__init__(f"memory {memory_id!r} not found: {reason}")


class LlmNotConfiguredError(MemoryUniverseError):
    """``ask()`` / adjudication was called while the container is in heuristic mode (``llm=None``).

    Neither ``mu_local.local_memory.LocalMemory`` nor ``mu_engine.surface.facade.SurfaceFacade``
    ever answers with an empty/degraded synthesis (spec §7, T7): ``add``/``recall``/``context``
    still work heuristically (no-LLM extraction + deterministic assembly), but a synthesis verb
    refuses loudly until an LLM backend is configured. Both surfaces raise this SAME class."""
