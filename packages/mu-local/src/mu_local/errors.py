"""Library-local error types for ``mu-local`` (mu-local-and-sdk-spec §7 / §Contract-changes 5).

All degrades are NAMED, never a bare ``except`` (CANONICAL §2, Layer-0 "no silent fallbacks").
These extend the frozen ``mu_contracts.domain.errors.MemoryUniverseError`` hierarchy — additive,
no envelope change. ``BackendUnavailableError`` already exists in mu-contracts (a selected backend
extra is missing); it is re-exported here so callers import the whole mu-local failure surface from
one place.
"""

from __future__ import annotations

from mu_contracts.domain.errors import BackendUnavailableError, MemoryUniverseError

__all__ = [
    "BackendUnavailableError",
    "LlmNotConfiguredError",
]


class LlmNotConfiguredError(MemoryUniverseError):
    """``ask()`` / adjudication was called while the container is in heuristic mode (``llm=None``).

    mu-local NEVER answers with an empty/degraded synthesis (spec §7, T7): ``add``/``recall``/
    ``context`` still work heuristically (no-LLM extraction + deterministic assembly), but a
    synthesis verb refuses loudly until an LLM backend is configured.
    """
