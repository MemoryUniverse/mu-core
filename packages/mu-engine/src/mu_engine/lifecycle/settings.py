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

from mu_engine.lifecycle.centrality import CentralitySettings
from mu_engine.lifecycle.mtm_graph import MtmWorkingGraphSettings

__all__ = [
    "CentralitySettings",
    "HostedMirrorConsent",
    "LifecycleSettings",
    "ManagerModeSettings",
    "MtmWorkingGraphSettings",
    "OwnershipSettings",
    "RetentionSettings",
    "SalienceSettings",
]


class SalienceSettings(BaseModel):
    """Salience-score weights (spec §16; §6 "rel DROPPED off the sweep" — weights sum to 1).

    **The three ratified weights are UNCHANGED at 0.5 / 0.2 / 0.3.** The A4 structural-salience
    amendment (``lifecycle/centrality.py``;
    ``docs/superpowers/design/research-graphify-adoption.md`` §4/§6 item A4) adds ``cen`` as a
    BLEND SHARE (:attr:`w_centrality`) rather than as a fourth
    entry in a renormalising weighted mean:

    ```
    base = w_recency*rec + w_usage*use + w_importance*imp      # these three sum to 1, always
    S    = base                                    if cen ABSENT
         = (1 - w_centrality)*base + w_centrality*cen   if cen PRESENT
    ```

    So the amendment is still a RE-WEIGHTING and never an append: with ``cen`` present the
    EFFECTIVE four-term vector is ``(0.45, 0.18, 0.27, 0.10)``, which sums to 1 and preserves the
    ratified 0.5 : 0.2 : 0.3 proportions exactly (5 : 2 : 3 either way). ``rel`` stays dropped —
    this amendment does not re-open spec §6's resolution of DRAFT §9 Q1.

    **Why a blend and not four declared weights.** The four-weight form has to divide by the sum of
    the PRESENT weights, and ``(0.45*rec + 0.18*use + 0.27*imp)/0.90`` is not bit-identical to
    ``0.5*rec + 0.2*use + 0.3*imp``: measured, 46,942 of 112,211 grid points differ in the last
    ulp and 20 of them cross one of the three ABSOLUTE gates below. Because ``cen`` is absent on
    every install with no centrality service wired, that would have silently re-decided
    promote/demote for FULL-LOCAL users. The blend form leaves the absent branch returning ``base``
    untouched, so those gates stay calibrated for real and not merely approximately.

    **The sum-to-1 invariant is NOT enforced at construction, deliberately, and that is a
    REPORTED gap rather than an oversight.** A ``model_validator`` rejecting a three-weight vector
    that misses 1.0 was written and then removed: it makes a single-field env override impossible,
    because ``MU_LIFECYCLE__SALIENCE__W_RECENCY=0.9`` alone leaves the other two at their defaults
    and sums to 1.4. That override is exactly what
    ``tests/config/test_engine_settings_unit.py:115`` exercises (proving three-level env nesting
    resolves), and enforcing the sum turned it red. An operator can therefore still configure a
    vector that does not sum to 1 and silently mis-calibrate the three ABSOLUTE gates below — a
    real pre-existing hazard this amendment did not create and must not silently re-scope into.
    What IS pinned, by literal assertions in ``tests/lifecycle/test_salience_centrality_unit.py``,
    is every SHIPPED weight magnitude, so no later edit can rescale the vector unnoticed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    w_recency: float = Field(default=0.5, ge=0.0)
    w_usage: float = Field(default=0.2, ge=0.0)
    w_importance: float = Field(default=0.3, ge=0.0)
    #: A4 structural-salience BLEND SHARE — the fraction of S(m) the structural term takes when it
    #: is present (effective weight 0.10, the smallest in the four-term vector). Small on purpose:
    #: this is the newest and least-validated signal, and 0.10 is small enough that a fact's
    #: structural position adjusts its rank without ever overriding recency or importance — the
    #: whole span of the term, cen=0 to cen=1, moves S by at most 0.10, which cannot on its own
    #: carry an item across the 0.4-wide gap between ``demote_mtm`` and ``promote_stm_mtm``.
    #: Range-bounded to [0, 1] so ``S = (1-w)*base + w*cen`` stays a convex combination and AC-1's
    #: unit-interval guarantee is structural.
    w_centrality: float = Field(default=0.10, ge=0.0, le=1.0)
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
    # FAULT-HUNT-0924 F1 fix (ADR 0054): the ONLY way a demoted item ever returns to MTM is by
    # re-crossing THIS gate from STM (there is no separate "un-demote" verb — DemotionService's
    # own pre-demotion rescue, `score >= demote_mtm`, is a different mechanism that only ever
    # prevents a still-in-MTM item from demoting in the first place). The achievable ceiling for
    # an item the gate has ALREADY demoted, re-scored after a full recall-driven usage rescue
    # (`use` raised from whatever it was to `usage_cap`), is bounded by
    # `demote_mtm + w_usage` = 0.3 + 0.2 = 0.5 (probe: FAULT-HUNT-0924.md §1 F1 — measured
    # EXACTLY 0.5000 at the demotion instant for every importance, because a demotion candidate's
    # score-without-usage sits right at `demote_mtm` by construction). The shipped 0.7 default put
    # the gate 0.2 ABOVE that ceiling — arithmetically unreachable for 100% of demoted items,
    # regardless of how many times they were re-recalled. 0.45 sits with real margin under the 0.5
    # ceiling (crossable by an item demoted moderately below the gate, not only a borderline one)
    # while still requiring genuine re-engagement: one recall alone only raises `access_count` by
    # 1, adding `w_usage/usage_cap` = 0.02 to the score with the shipped `usage_cap=10` — nowhere
    # near enough on its own, so "rescued" still means "recalled repeatedly," not "recalled once."
    # This does not touch `w_recency`/`w_usage`/`w_importance` (pinned, literal-tested elsewhere)
    # — only where the SAME score is compared against for THIS gate. Lowering it also makes the
    # ordinary (non-rescue) STM->MTM backstop somewhat more permissive; that is intentional and
    # bounded by DemotionService's own forgetting curve on the other side (ADR 0034's "MTM growth
    # is bounded by demotion, not by starving promotion").
    #
    # 0.45, not 0.5 (the theoretical ceiling) or lower: it must ALSO stay far enough above
    # `demote_mtm` (0.3) that `SalienceSettings.w_centrality` (0.10, the A4 structural-salience
    # blend share) cannot, on its own, carry an item across this gate — the stated A4 design
    # invariant (`test_the_whole_span_of_the_term_cannot_carry_an_item_across_the_gates`) is that
    # centrality adjusts rank and never overrides the other terms; a gap <= w_centrality would
    # quietly violate that for this one gate. 0.45 leaves a 0.15 margin above `demote_mtm`
    # (> 0.10) while staying under the 0.5 rescue ceiling.
    promote_stm_mtm: float = Field(default=0.45, ge=0.0, le=1.0)

    # ADR 0058 verify pass (2026-09-24): 0.9 survived ADR 0054's F2 fix (`score_for_ltm_gate`
    # drops recency) arithmetically reachable in theory but empirically DEAD for the corpus the
    # product actually writes. Exhaustive grid over `score_for_ltm_gate = (w_usage*use +
    # w_importance*imp)/(w_usage+w_importance)` = `0.4*use + 0.6*imp` at shipped `w_usage=0.2`/
    # `w_importance=0.3` (importance 0..1 step 0.01 x access_count 0..60, `usage_cap=10`):
    #     max score over the whole grid = 1.0 (imp=1.0, access_count>=10)
    #     lowest importance EVER admitted at 0.9 = 0.84, and only at access_count >= 10
    # Every importance the shipped capture path actually writes is below that floor:
    # `IngestSettings.importance_promote` default 0.5 (`mu_contracts.../memory.py:277`),
    # `mu-client` `thinking_finding_importance` 0.55, `thinking_decision_importance` 0.70
    # (`mu_client/config.py:184-185`) — UNREACHABLE at ANY access_count. Measured end to end
    # (`docs/tracking/eval-runs/2026-09-24-verify-graph-tier-contribution.md`): 1,051 facts
    # extracted from three real conversations, 0 ever admitted through the periodic gate. A gate
    # sitting above the entire distribution the product writes is not a gate, it is an off switch
    # (ADR 0058's own words) — recalibrated here against that distribution, not picked to feel
    # right.
    #
    # Recalibrated to 0.6 — the SAME bar `importance_promote` already uses for the STM->MTM ingest
    # gate (`services/settings.py:24`), so "worth consolidating to the durable graph" is pinned to
    # "at least as validated as a fact that reached MTM on importance alone, PLUS demonstrated
    # re-use," not a number chosen in isolation. Re-solving the grid at 0.6:
    #     imp=1.00 (explicit max/pinned importance): admitted at access_count=0 (immediately)
    #     imp=0.90:                                  admitted at access_count>=2   (use>=0.15)
    #     imp=0.70 (thinking_decision_importance):    admitted at access_count>=5   (use>=0.45)
    #     imp=0.55 (thinking_finding_importance):     admitted at access_count>=7   (use>=0.675)
    #     imp=0.50 (ingest/importance_promote floor):  admitted at access_count>=8   (use>=0.75)
    # Every stamp the shipped capture path writes is now reachable through real re-engagement
    # (access_count in single digits, well under `usage_cap=10`) rather than "never, at any
    # access_count" — while still requiring genuine, repeated recall for anything below the
    # `importance_promote` floor, so this is a recalibration of an off switch into a gate, not a
    # removal of the gate. See `tests/lifecycle/test_promotion_int.py` and
    # `tests/lifecycle/test_salience_ltm_gate_calibration_unit.py` for the grid re-run against the
    # SHIPPED default (no half-life/usage_cap override) and the mutation check.
    promote_mtm_ltm: float = Field(default=0.6, ge=0.0, le=1.0)
    promote_min_age_h: float = Field(default=24.0, ge=0.0)
    pre_ttl_window_s: int = Field(default=300, ge=1)  # last-chance salience rescue before STM TTL

    # --- demotion (§7b) ---
    demote_mtm: float = Field(default=0.3, ge=0.0, le=1.0)
    demotion_enabled: bool = True
    ltm_demotion_enabled: bool = False  # RESERVED (§10)

    # FAULT-HUNT-0924 F1 fix (ADR 0054): the TTL stamped on the write-ahead STM copy
    # `DemotionService._demote_one` writes when an MTM point demotes. This is DELIBERATELY a
    # separate knob from `IngestSettings.stm_ttl_s` (the fresh-capture buffer TTL, 3600s) — before
    # this fix `DemotionService` had no TTL override at all and silently inherited that same
    # 3600s "unprocessed capture" TTL for a memory that had already survived days in MTM, giving
    # every demoted memory a hard, silent, ~1-hour-from-demotion horizon (measured: importance
    # 0.5 -> demotable after 1.74 real days, then dead again ~1h after that — a cliff, not a
    # policy). A demoted memory is not scratch space; it is a real memory the forgetting curve
    # de-prioritized, and it deserves real time for `promote_stm_mtm`'s re-crossing (above) or the
    # `pre_ttl_scan_interval_s` rescue scan to actually catch it. Default: 30 days — long enough
    # that a demoted-and-never-touched-again memory is a deliberate, documented, CONFIGURABLE
    # retention decision instead of an emergent one-hour accident; short enough that a namespace's
    # STM tier does not grow unbounded with cold copies (Valkey TTL is still the eventual floor,
    # per ADR 0034 "the TTL becomes a floor, not the decision-maker").
    demoted_stm_ttl_s: int = Field(default=2_592_000, ge=1)

    # --- quarantine (RESERVED §10) ---
    quarantine_ttl_d: int = Field(default=7, ge=0)

    # --- conflict (§8) ---
    use_llm_adjudicator: bool = True  # False ⇒ heuristic-only (LLM_UNAVAILABLE_HEURISTIC)
    adjudication_budget_per_sweep: int = Field(default=50, ge=0)  # S1: hard cap on
    # Task.CONFLICT_ADJUDICATION calls / sweep tick
    adjudication_degrade_threshold_s: float = Field(default=30.0, ge=0.0)  # S1: per-sweep
    # wall-clock budget, Clock-measured (§19)
    # CONFIG-AND-DATA-FIX-PLAN.md §1.2 C3: the OTHER two ``lifecycle.conflict.
    # ConflictAdjudicatorSettings`` fields (``max_tokens``/``temperature``) had no home on this
    # tree at all (only ``adjudication_budget_per_sweep``/``adjudication_degrade_threshold_s``
    # mirrored across both classes) — added here so a composition root can build the FULL
    # ``ConflictAdjudicatorSettings`` from ONE wired subtree
    # (``mu_engine.lifecycle.conflict.conflict_adjudicator_settings_from_lifecycle``) and every
    # adjudicator knob is reachable via ``MU_LIFECYCLE__…`` (this subtree's own env prefix),
    # never a second env-prefix family for the SAME conceptual budget.
    adjudicator_max_tokens: int = Field(default=256, ge=1)
    adjudicator_temperature: float = Field(default=0.0, ge=0.0)

    # --- observability + metering (§20) ---
    config_version: str = "v1"  # S6: tag on every job/explain/usage record this tick produces
    policy_version: str = "v1"  # S6: tag for the ManagerModeSettings/RetentionSettings gen active

    # --- sub-trees ---
    salience: SalienceSettings = Field(default_factory=SalienceSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)
    manager_mode: ManagerModeSettings = Field(default_factory=ManagerModeSettings)
    ownership: OwnershipSettings = Field(default_factory=OwnershipSettings)
    mtm_working_graph: MtmWorkingGraphSettings = Field(default_factory=MtmWorkingGraphSettings)
    centrality: CentralitySettings = Field(default_factory=CentralitySettings)
