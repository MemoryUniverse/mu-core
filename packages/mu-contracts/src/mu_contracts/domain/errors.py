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
    "MandatoryBackendMissingError",
    "MemoryLayerError",
    "MemoryUniverseError",
    "NamespaceIsolationError",
    "PermissionNarrowingError",
    "PlaneFieldRejectedError",
    "PrivacyTierConflictError",
    "PrivacyTierNotAvailableError",
    "ProviderError",
    "ReshareDepthExceededError",
    "RevocationConflictError",
    "SchemaDriftError",
    "SettingsValidationError",
    "StoreUnavailableError",
    "SubModelProviderDisabledError",
    "TenancyViolationError",
    "TierRepositoryUnavailableError",
    "UnknownBackendError",
    "UnknownComponentError",
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


# ── security ───────────────────────────────────────────────────────────────────────────────
class AuthorizationError(MemoryUniverseError):
    """Base for every authorization denial (platform-layer0-spec §6)."""


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


class RevocationConflictError(MemoryUniverseError):
    """A concurrent revoke lost a compare-and-set (sync-devices-gateway §A.5)."""


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


# ── governance / transfer (governance-transfer-core §7) ──────────────────────────────────────
class GovernanceError(MemoryUniverseError):
    """Base of the governance/transfer errors."""


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
