"""``RecallResult`` family — the canonical ranked-hit-list DTOs for ``recall`` and (via
:class:`~mu_contracts.contracts.views.ContextView`) ``build_context`` (design §2.5; build-plan
Stage B item 3; ``SDK-BUILD-DECISIONS.md`` Decision B).

RE-HOMED from ``mu_sdk.models.recall`` (`mu-sdk-python/src/mu_sdk/models/recall.py`) — the WIRE
shape won as canonical over the embedded ``mu_local.views.MemoryListView``/``MemoryRecordView``
pair (Decision B: "``RecallResult`` (wire), a strict superset of ``MemoryListView``" — nothing
embedded recall returns is lost, see the decision doc's field-by-field diff). ``RecallItemView``
here is ALSO the single canonical per-hit item DTO the decision doc's cross-cutting section calls
for (picked over the thinner ``mu_local.views.MemoryRecordView``, which it retires) — used for
BOTH ``RecallResult.items`` and ``ContextView.items``.

**Two ``RecallResult``/``RecallItemView`` pairs exist elsewhere in the tree — do not confuse them:**
``mu_engine.services.recall.dto`` (`mu-core/packages/mu-engine/src/mu_engine/services/recall/
dto.py:90,108`) is the INTERNAL engine-native shape (``RecallItemView`` there additionally carries
``content_hash`` + a per-item ``namespace``, used for federate-dedup across the private/shared
recall arms, `dto.py:92,98,101`). Decision B recommends the SURFACE item drop those two
federate-internal fields; this module implements that recommendation — they are intentionally
ABSENT below. ``mu_engine.services.recall.dto`` is untouched by this change (a different package,
not owned by this task) and stays the engine's internal read-path shape; ``SurfaceFacade``
(``mu-engine/surface/facade.py``) still returns THAT shape today (Stage A ruling 2, provisional) —
reconciling the facade's return type to this canonical one is Stage B's B1/B2 wiring work, not done
by this module.

``RecallChannels``/``RecallMode`` are moved alongside ``RecallResult`` (not left behind in
``mu_sdk``) because ``RecallResult.channels_run: RecallChannels`` cannot type-check without them in
the same importable closure a boundary-respecting caller can reach.

**``RecallRequest`` is deliberately NOT moved here** — it is a request-only wire shape (not a
per-verb RETURN DTO, out of this task's A4-winners scope) and stays in ``mu_sdk.models.recall``;
that module should import ``RecallChannels``/``RecallMode`` from here instead of re-declaring them
once B2 lands (not done by this task — B0 owns only ``mu-contracts``).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.events import DegradeReason
from mu_contracts.domain.model.memory import Namespace, Tier

__all__ = [
    "RecallChannels",
    "RecallItemView",
    "RecallMode",
    "RecallResult",
]


class RecallChannels(BaseModel):
    """Which channels a recall runs (recall-service-design.md:78-82). A degraded recall drops
    one; :attr:`RecallResult.channels_run` reports what actually ran."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stm: bool = True
    mtm: bool = True
    ltm: bool = True


class RecallMode(StrEnum):
    """recall-service-design.md:84-87."""

    RANKED = "ranked"  # default: full 3-channel ranked list
    ANSWER = "answer"  # recall -> LLM synthesis (ask path)
    INJECT = "inject"  # recall -> rendered additionalContext bundle


class RecallItemView(BaseModel):
    """One ranked hit — the single canonical hit-item shape shared by ``RecallResult.items`` and
    ``ContextView.items`` (Decision B cross-cutting section). Deliberately WITHOUT the engine-
    internal ``content_hash``/per-item ``namespace`` fields ``mu_engine.services.recall.dto.
    RecallItemView`` carries for federate-dedup (module docstring) — those are an implementation
    detail of the recall service's private/shared-arm merge, not part of the surface a caller
    reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    content: str
    tier: Tier
    channel: str  # "stm" | "mtm" | "ltm" — provenance of the hit
    fused_score: float
    rerank_score: float | None = None
    is_floor: bool = False
    artifact_ref: str | None = None


class RecallResult(BaseModel):
    """The ranked read result — the canonical return DTO for ``recall(...)`` (Decision B: wins
    over the embedded ``MemoryListView`` as a strict superset; nothing embedded recall returns is
    lost, see the decision doc's field-by-field diff)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    namespace: Namespace
    items: list[RecallItemView]
    channels_run: RecallChannels
    degraded: DegradeReason | None = None
    generated_at: datetime

    @property
    def memory_ids(self) -> list[str]:
        """The content-free projection (recall-service-design.md:122-124)."""
        return [item.memory_id for item in self.items]
