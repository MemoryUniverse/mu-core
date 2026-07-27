"""Clock implementations — the injected time port's concrete adapters (platform-layer0-spec §4).

The ``Clock`` Protocol is owned by ``mu-contracts`` (``ports/time.py``). :class:`SystemClock` is the
app singleton (``build_clock``); tests inject :class:`FrozenClock` / :class:`OffsetClock` for
determinism (DEV-STANDARDS: no wall-clock in tests without a seed).

Determinism boundary (spec §4): ``Clock.now()`` is allowed in activities and in-process stages but
FORBIDDEN in a Temporal workflow body (which uses ``workflow.now()``). A CI grep enforces that; this
module only provides the adapters.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from mu_contracts.config.settings import Settings
from mu_contracts.ports.time import Clock

__all__ = ["FrozenClock", "OffsetClock", "SystemClock", "build_clock"]


class SystemClock:
    """Wall-clock, timezone-aware UTC (the production :class:`Clock`)."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class FrozenClock:
    """A fixed instant — deterministic tests (spec §4). ``advance`` moves it explicitly."""

    def __init__(self, at: datetime) -> None:
        self._at = at if at.tzinfo is not None else at.replace(tzinfo=UTC)

    def now(self) -> datetime:
        return self._at

    def advance(self, delta: timedelta) -> None:
        self._at = self._at + delta


class OffsetClock:
    """A real clock shifted by a fixed offset — skew simulation (sync tests)."""

    def __init__(self, offset: timedelta, *, base: Clock | None = None) -> None:
        self._offset = offset
        self._base: Clock = base or SystemClock()

    def now(self) -> datetime:
        return self._base.now() + self._offset


def build_clock(settings: Settings) -> Clock:
    """The app-singleton clock (spec §4). Signature takes ``Settings`` for registry-uniformity even
    though the system clock needs no config (a future test/offset clock is selected via config)."""
    del settings
    return SystemClock()
