"""Memory Lifecycle Manager (MLM) central-config tree (spec §16 verbatim field set;
DEV-STANDARDS rule 3 — no threshold/interval/weight is ever hardcoded at a call site).

This follows the exact **"tracked seam" convention** already established by
``mu_engine.services.settings.IngestSettings`` and
``mu_engine.pipelines.distill.DistillSettings``: a plain, frozen ``pydantic.BaseModel`` subtree
that the owning pipeline/manager takes as an explicit constructor argument, with every default
sourced from the spec rather than inlined at a call site. It is declared here as the sanctioned
central-config home for the lifecycle subsystem, but it is **NOT YET wired as a
``Settings.lifecycle`` sibling field** on the central ``mu_contracts.config.settings.Settings``
root — that composition-root wiring (and the ``MU_LIFECYCLE__`` env-prefix activation it implies,
mirroring the demo override pattern at ``mma/mma/memory/controller.py:405,412``
(``MU_LIFECYCLE__MAINTENANCE_INTERVAL_S=60``)) is explicitly out of scope for this slice (see
plan doc §0) and lands when the composition root threads ``LifecycleSettings`` into
``MemoryLifecycleManager`` (spec §17).

Field set, defaults, and sub-tree shape are copied verbatim from
``docs/superpowers/design/memory-lifecycle-manager-spec.md`` §16.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mu_engine.lifecycle.mtm_graph import MtmWorkingGraphSettings

__all__ = [
    "HostedMirrorConsent",
    "LifecycleSettings",
    "ManagerModeSettings",
    "MtmWorkingGraphSettings",
    "OwnershipSettings",
    "RetentionSettings",
    "SalienceSettings",
]


class SalienceSettings(BaseModel):
    """Salience-score weights (spec §16; §6 "rel DROPPED off the sweep" — weights sum to 1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    w_recency: float = 0.5
    w_usage: float = 0.2
    w_importance: float = 0.3
    recency_half_life_h: float = 24.0
    usage_cap: int = 10


class RetentionSettings(BaseModel):
    """Per-retention-class knobs (spec §9) — PER-CLASS, not one global window.

    PERMANENT has no knobs here (never archived/GC'd — only explicit supersede/delete).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # EPHEMERAL: slack past invalid_at before the SELF_EXPIRE state flip (reuses invalid_at,
    # no valid_until field — F3).
    ephemeral_grace_s: int = 0
    # DURABLE: + low-importance + inactive → COLD.
    durable_cold_after_d: int = 180
    durable_cold_importance_max: float = 0.2
    # dead (SUPERSEDED/EXPIRED) → GC once chain head dead.
    gc_history_window_d: int = 365
    reactivate_on_recall: bool = True
    respect_pins: bool = True  # pinned ⇒ GC-ineligible (CANONICAL §7.10)


class ManagerModeSettings(BaseModel):
    """Manager-mode gate defaults (spec §3).

    NOTE: ``ManagerMode`` itself (the 3-member StrEnum: ``MANAGED``/``MANUAL``/``HYBRID`` per
    spec §3) is owned by ``mu_engine.lifecycle.mode_gate`` (S0-03, alongside ``ManagerModeGate``
    and ``ManagerOwnsLifecycleError``) — a sibling Stage-0 slice not yet landed as this module
    was authored. ``default_mode`` is typed ``str`` here (value = ``ManagerMode.MANAGED``) rather
    than importing the not-yet-existing enum, to avoid a premature cross-task import; this
    subtree never redefines the enum (DRY) and the integrate phase should re-type this field to
    ``ManagerMode`` once ``mode_gate.py`` lands.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Workspace default; per-ns/per-memory override (ManagerMode.MANAGED value, spec §3).
    default_mode: str = "managed"
    enforce_engine_side: bool = True  # SDK selects, engine enforces — never client-authorized


class HostedMirrorConsent(StrEnum):
    """Consent boundary for a server-side offline sweep of a user's mirrored data (spec §4, X4)."""

    NOT_CONSENTED = "not_consented"  # server MUST NOT sweep this user's data offline
    IMPLICIT_VIA_HYBRID = "implicit_via_hybrid"  # DEFAULT — 2nd-device provisioning implies consent
    CONSENTED = "consented"  # explicit opt-in independent of device count


class OwnershipSettings(BaseModel):
    """Sweep-lease + hosted-mirror-consent knobs (spec §4/§4b)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Governs BOTH lifecycle-sweep-lease grains (§4b): the local SqliteWalLeaseAdapter row AND
    # the hub Redis SETNX key — same sweep_user code, same worst-case-duration bound, one TTL.
    # offline > this ⇒ server also takes over as primary (§4 failover).
    lease_ttl_s: int = 900
    # S4: renew well under lease_ttl_s/2 — a live holder never lapses (applies to both grains).
    lease_heartbeat_s: int = 240
    handback_reconcile: bool = True  # content-hash reconcile + REINSTATE on reconnect
    hosted_mirror_consent: HostedMirrorConsent = HostedMirrorConsent.IMPLICIT_VIA_HYBRID


class LifecycleSettings(BaseModel):
    """The Memory Lifecycle Manager central-config tree (spec §16 verbatim field set).

    Same "tracked seam" pattern as ``IngestSettings``/``DistillSettings`` (see module docstring):
    a plain frozen ``BaseModel``, not yet a ``Settings`` sibling field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True

    # --- runner cadence / discovery (§7) ---
    maintenance_interval_s: int = Field(default=86_400, ge=1)  # 24h prod
    full_scan_interval_s: int = Field(default=86_400, ge=1)  # slow full-scan backstop
    batch_size: int = Field(default=20, ge=1)  # event-driven per-user fast-fire threshold
    session_idle_s: int = Field(default=900, ge=1)  # (ii) session-boundary idle timer
    max_users_per_sweep: int = Field(default=500, ge=1)  # shared-box RAM guard (bounded enum.)
    max_items_per_user_sweep: int = Field(default=2_000, ge=1)

    # --- backpressure (§7 S3 / AC-1.2 — GAPSWEEP BLOCKER 3 fix) ---
    # AC-1.2's named bound: max allowed increase in capture-ack p99 latency (ms) while a sweep
    # runs concurrently, vs. capture-ack p99 with no sweep running. Measured via the injected
    # Clock (§19), not time.time(). Exceeding this budget is the AC-1.2 failure condition (§14.1).
    capture_ack_p99_delta_budget_ms: int = Field(default=50, ge=0)

    # --- pre-TTL rescue cadence (§7b — MAJOR 4 fix) ---
    # Independent cadence for the pre-TTL rescue scan, decoupled from maintenance_interval_s (24h
    # prod default would never land inside a 300s pre_ttl_window_s before Redis TTL-deletes the
    # item). Invariant (§7b S-4 arithmetic pass): pre_ttl_scan_interval_s <= pre_ttl_window_s / 2
    # (120<=150 here) -- guarantees >=2 scan ticks land inside every item's pre_ttl_window_s-wide
    # rescue window regardless of phase. (NOT a comparison against stm_ttl_s - pre_ttl_window_s --
    # that quantity is the window's OFFSET, not its WIDTH, and bounding against it does not
    # guarantee a tick lands inside the window.)
    pre_ttl_scan_interval_s: int = Field(default=120, ge=1)

    # --- promotion gates (§7b) ---
    promote_stm_mtm: float = Field(default=0.7, ge=0.0, le=1.0)
    promote_mtm_ltm: float = Field(default=0.9, ge=0.0, le=1.0)
    promote_min_age_h: float = Field(default=24.0, ge=0.0)
    pre_ttl_window_s: int = Field(default=300, ge=1)  # last-chance salience rescue before STM TTL

    # --- demotion (§7b) ---
    demote_mtm: float = Field(default=0.3, ge=0.0, le=1.0)
    demotion_enabled: bool = True
    ltm_demotion_enabled: bool = False  # RESERVED (§10)

    # --- quarantine (RESERVED §10) ---
    quarantine_ttl_d: int = Field(default=7, ge=0)

    # --- conflict (§8) ---
    use_llm_adjudicator: bool = True  # False ⇒ heuristic-only (LLM_UNAVAILABLE_HEURISTIC)
    adjudication_budget_per_sweep: int = Field(default=50, ge=0)  # S1: hard cap on
    # Task.CONFLICT_ADJUDICATION calls / sweep tick
    adjudication_degrade_threshold_s: float = Field(default=30.0, ge=0.0)  # S1: per-sweep
    # wall-clock budget, Clock-measured (§19)

    # --- observability + metering (§20) ---
    config_version: str = "v1"  # S6: tag on every job/explain/usage record this tick produces
    policy_version: str = "v1"  # S6: tag for the ManagerModeSettings/RetentionSettings gen active

    # --- sub-trees ---
    salience: SalienceSettings = Field(default_factory=SalienceSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    manager_mode: ManagerModeSettings = Field(default_factory=ManagerModeSettings)
    ownership: OwnershipSettings = Field(default_factory=OwnershipSettings)
    mtm_working_graph: MtmWorkingGraphSettings = Field(default_factory=MtmWorkingGraphSettings)
