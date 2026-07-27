"""Room vocabulary — ParticipantKind / MessageKind + the stored RoomMessage.

Authority: CANONICAL §3.2 (the content-free ``RoomMessagePosted`` event carries ``content_hash``
+ ``author_kind``/``kind`` enums — NEVER a body; consumers read the full ``RoomMessage`` from the
``RoomLogRepository`` by ``(room_id, seq)``), rooms/sessions-live-rooms spec.

``RoomMessage`` is a STORED domain object (read from the durable room log), so it legitimately
carries ``body`` — the content-free discipline binds the BUS/events, not the store (CANONICAL §3.1).
``ParticipantKind`` is UNCHANGED — ``SUBAGENT`` is an ``AgentKind`` projected to
``bound_agent``/``shared_agent``, never a new ``ParticipantKind`` value (CANONICAL §5.7, rooms A1).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["MessageKind", "ParticipantKind", "RoomMessage"]


class ParticipantKind(StrEnum):
    HUMAN = "human"
    SHARED_AGENT = "shared_agent"
    BOUND_AGENT = "bound_agent"


class MessageKind(StrEnum):
    UTTERANCE = "utterance"
    AGENT_RESULT = "agent_result"
    SYSTEM = "system"
    CONTEXT_INJECTED = "context_injected"


class RoomMessage(BaseModel):
    """The durable room-log record. ``seq`` is the per-room monotonic ordering authority
    (contiguity apply-rule, CANONICAL §7.6). Read from the store by ``(room_id, seq)`` — the
    content-free ``RoomMessagePosted`` event only carries the ``content_hash`` locator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    room_id: str = Field(min_length=1)  # == session_id
    seq: int = Field(ge=0)
    author_principal_id: str = Field(min_length=1)
    author_kind: ParticipantKind
    kind: MessageKind
    correlation_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)  # dedupe / provenance
    body: str  # stored body (legitimate on the STORE object; never on the bus)
    posted_at: datetime
