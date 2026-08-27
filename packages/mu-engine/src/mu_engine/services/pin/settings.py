"""``PinSettings`` — the pinning central-config subtree.

Authority: ``memory-health-pinning-spec.md`` §8 (lines 359-362), verbatim field set + defaults.
Same "tracked seam" convention as ``HealthSettings``/``LifecycleSettings``; CANONICAL §7.27's
``Settings`` root wiring is an outstanding, reported delta.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["PinSettings"]


class PinSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = True
    #: Pin-explosion guard (spec §5.2 step 2). Bounds BOTH the number of pins and the size of the
    #: single ``enumerate`` page the bound check reads, so the check itself can never scan.
    max_pins_per_namespace: int = Field(default=500, ge=1)
    #: v1 default OFF (spec line 362): you pin YOUR OWN items. A member who wants a shared fact
    #: pinned pins their pulled LOCAL copy, which is a PRIVATE item after Import (CANONICAL §7.7).
    #:
    #: NECESSARY BUT NOT SUFFICIENT. Spec §5.2 step 1 line 265 makes SHARED-origin pin a
    #: CONJUNCTION — this flag AND "the caller is the item's origin principal (provenance ORIGIN,
    #: §7.10) or a workspace admin". mu-core has no provenance-ORIGIN reader and no admin role
    #: (both are ``mu-server`` governance), so the second conjunct evaluates FALSE here and
    #: ``PinService`` refuses a SHARED-origin pin even with this ON. Turning it on therefore
    #: changes only the refusal MESSAGE on this plane; it can never grant pin over another
    #: member's shared item. See ``PinService._refuse_shared_origin_pin``.
    allow_shared_origin_pin: bool = False
