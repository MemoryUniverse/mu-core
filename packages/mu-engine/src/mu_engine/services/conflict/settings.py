"""``ConflictSettings`` — the conflict subtree of the central config.

Authority: ``conflict-resolution-async-design.md`` §4 (lines 142-147) + §8 line 273
(``manual_backlog_alert``) + proposed contract change 8 (line 318). Same "tracked seam"
convention as ``HealthSettings``/``PinSettings``/``LifecycleSettings``: a frozen pydantic subtree
with real defaults, wired by the composition root; CANONICAL §7.27's ``Settings`` root wiring is
an outstanding, reported delta shared with those three.

**Why the detection knobs are re-homed here (and what that does NOT do).** ``candidate_k``,
``supersede_confidence`` and ``refine_confidence`` already exist as fields on
``DistillSettings`` (``pipelines/distill.py``). They are the §4 line 143-145 field set verbatim,
so they are declared here too — as the ONE named home the design gives them — and
:meth:`from_distill` maps the shipped values in rather than duplicating the numbers. This module
does NOT edit ``DistillSettings`` (another lane's file) and does not change any behaviour of the
distill pipeline; it gives the composition root a single ``ConflictSettings`` to thread, and
records the duplication as a delta to close from the distill side later.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mu_engine.lifecycle.conflict import ConflictResolutionPolicy

__all__ = ["ConflictSettings"]


class ConflictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Detection strategy tag (spec line 143). Informational provenance carried onto
    #: ``ConflictRecord.method``-adjacent audit; not a dispatch key.
    strategy: str = Field(default="bitemporal_v1", min_length=1)
    #: Mirrors ``DistillSettings.supersede_confidence`` / ``.refine_confidence`` (spec lines
    #: 144-145). NOTE these are the same two numbers as ``ConflictResolutionPolicy
    #: .auto_min_confidence`` / ``.quarantine_below`` under different names — three names, two
    #: thresholds. The POLICY pair is the one any decision path reads (it is per-namespace and
    #: per-memory overridable; these two are global), and the duplication is REPORTED rather
    #: than resolved unilaterally across two lanes' files.
    supersede_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    refine_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Top-k candidate gather bound (spec line 146). Mirrors ``DistillSettings.candidate_k``.
    candidate_k: int = Field(default=5, ge=1)
    #: The workspace/global default — step 3 of the §4.1 precedence chain, the floor every
    #: resolution lands on when no per-memory and no per-namespace policy is set.
    default_policy: ConflictResolutionPolicy = ConflictResolutionPolicy()
    #: Pending manual conflicts above this count for one principal raise
    #: ``CONFLICT_MANUAL_BACKLOG`` (spec §8 line 273) — a soft, user-visible nudge routed to the
    #: inbox surface, never a hard failure. ``0`` disables the nudge entirely.
    manual_backlog_alert: int = Field(default=25, ge=0)
