"""Clock — the injected time port (platform-layer0-spec §4; CANONICAL §6-P1).

One canonical port; ``rooms.ClockPort`` is deleted. ``Clock.now()`` is allowed in activities
and in-process stages but FORBIDDEN in a Temporal workflow body (which uses ``workflow.now()``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

__all__ = ["Clock"]


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:  # timezone-aware UTC
        ...
