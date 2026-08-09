"""Library-local error types for ``mu-local`` (mu-local-and-sdk-spec §7 / §Contract-changes 5).

All degrades are NAMED, never a bare ``except`` (CANONICAL §2, Layer-0 "no silent fallbacks").
These extend the frozen ``mu_contracts.domain.errors.MemoryUniverseError`` hierarchy — additive,
no envelope change. ``BackendUnavailableError`` and ``LlmNotConfiguredError`` already exist in
mu-contracts (CO-3 folded the latter's former mu-local-local duplicate into that ONE canonical
home, alongside ``mu_engine.surface.facade``'s former duplicate); both are re-exported here so
callers import the whole mu-local failure surface from one place.
"""

from __future__ import annotations

from mu_contracts.domain.errors import BackendUnavailableError, LlmNotConfiguredError

__all__ = [
    "BackendUnavailableError",
    "LlmNotConfiguredError",
]
