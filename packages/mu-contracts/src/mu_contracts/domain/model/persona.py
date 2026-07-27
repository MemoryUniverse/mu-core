"""Persona vocabulary — PersonaSlot / SlotValue / PersonaProfile (PRIVATE-only).

Authority: persona-design.md §2.1/§3.1. The objective/subjective slot taxonomy is adopted
verbatim in shape from MemOS (``OR/MemOS/src/memos/mem_reader/memory.py:83-110``) because it
classifies at extraction time into stable keys. ``PersonaProfile`` is keyed on
``Namespace.to_prefix()`` and is structurally PRIVATE — the ``_must_be_private`` validator makes
it impossible to persist a persona on a SHARED/room partition (persona is voice/relevance, never
access — the §5.4 authorization firewall).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mu_contracts.domain.model.memory import Namespace, Visibility

__all__ = ["PersonaProfile", "PersonaSlot", "SlotValue"]


class PersonaSlot(StrEnum):
    """The MemOS objective/subjective slot schema (``memory.py:85-110``). OCEAN/MBTI is NOT the
    top-level schema — it is one optional value carried inside the ``PERSONALITY`` slot."""

    # --- objective (slow-changing facts about the user) ---
    NICKNAME = "nickname"
    PERSONALITY = "personality"
    OCCUPATION = "occupation"
    EXPERTISE = "expertise"
    LANGUAGE = "language"
    PREFERENCE = "preference"
    HOBBY = "hobby"
    GOAL = "goal"
    # --- subjective (fast-changing interaction preferences) ---
    RESPONSE_STYLE = "response_style"
    LANGUAGE_STYLE = "language_style"
    INFORMATION_DENSITY = "information_density"
    INTERACTION_PACE = "interaction_pace"
    FOLLOWED_TOPIC = "followed_topic"
    ROLE_PREFERENCE = "role_preference"


class SlotValue(BaseModel):
    """One resolved persona-slot value with provenance + optional decay (persona §2.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: str
    confidence: float = Field(ge=0.0, le=1.0)  # MemOS confidence_score; Wang-1996 trust weight
    support_ids: tuple[str, ...]  # MemoryItem.ids that evidence this slot (provenance)
    updated_at: datetime
    decay_half_life_h: float | None = None  # subjective slots decay; objective slots None (§3.3)


class PersonaProfile(BaseModel):
    """The per-user persona portrait (persona §3.1). PRIVATE-only by construction."""

    model_config = ConfigDict(extra="forbid")

    namespace: Namespace  # PRIVATE only; keyed on (org, workspace, user)
    slots: dict[PersonaSlot, SlotValue] = Field(default_factory=dict)
    overall_brief: str  # capped portrait+strategy string (MemoryBank overall_personality)
    brief_etag: str  # sha256(overall_brief) — inject skip-if-unchanged (recall §2.2)
    version: int = Field(ge=0)  # monotonic; bumped every rebuild (optimistic concurrency)
    rebuilt_at: datetime
    source_memory_count: int = Field(ge=0)  # how many items fed this build (obs / staleness)

    @field_validator("namespace")
    @classmethod
    def _must_be_private(cls, ns: Namespace) -> Namespace:
        if ns.visibility is not Visibility.PRIVATE:
            raise ValueError(
                "PersonaProfile is PRIVATE-only; a room/shared partition has no persona"
            )
        return ns
