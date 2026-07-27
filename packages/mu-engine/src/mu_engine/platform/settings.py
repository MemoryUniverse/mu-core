"""Platform-layer0 cross-cutting knobs — ``ObservabilitySettings`` + ``RetrySettings``
(platform-layer0-spec §9 typed-retry / §11 observability).

Declared here as a tracked seam (same pattern as ``services/settings.py``'s ``IngestSettings``
and ``pipelines/distill.py``'s ``DistillSettings``): the central ``Settings`` tree does not yet
carry the platform subtree, so callers take these explicitly and the composition root wires
``settings.platform.observability`` / ``settings.platform.retry`` when it lands — no re-shape.
DEV-STANDARDS rule 3: no bounded-queue size or retry/backoff number is hardcoded in a decorator
or sink LOGIC path; every one of them flows from here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["ObservabilitySettings", "RetrySettings"]


class ObservabilitySettings(BaseModel):
    """Content-free observability sink knobs (platform-layer0-spec §11)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bounded backpressure for the durable audit queue (DEV-STANDARDS: bounded queues, never
    # unbounded) — :class:`mu_engine.platform.observability._DurableAuditLog`.
    durable_audit_queue_max: int = 1024


class RetrySettings(BaseModel):
    """``retry_io`` decorator defaults (platform-layer0-spec §9 typed-retry).

    These are the ONE central function's own tunables (DEV-STANDARDS rule 7: a cross-cutting
    concern lives in one decorator, not copy-pasted) — modeled here so the defaults are a named,
    Settings-shaped object rather than bare literals scattered at the decorator's signature.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = 3
    base_delay_s: float = 0.05
    max_delay_s: float = 2.0
