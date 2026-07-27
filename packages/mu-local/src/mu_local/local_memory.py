"""LocalMemory — the embedded in-process engine facade (mu-local-and-sdk-spec §1/§CC-1).

STUB ONLY (scaffold). The full facade — `add()` / `recall()` / `ask()` over an in-process
LocalContainer that supplies all three mandatory storage roles (vector / relational /
graph — graph is MANDATORY, CANONICAL storage) and the OPEN baseline strategies from
mu-engine — is implemented in a later phase. It NEVER imports mu-server (CI: mu-local-no-server).

The real facade will be fully async with correct cancellation handling (DEV-STANDARDS rule 1)
and built via DI on the LocalContainer composition root (rule 9). No method is implemented
here yet — this is a tracked, explicit gap, not a silent stub.
"""

from __future__ import annotations

from mu_contracts.config import Settings, get_settings

__all__ = ["LocalMemory"]


class LocalMemory:
    """Embedded FULL-LOCAL memory entrypoint. Scaffold: wiring only, no engine logic yet."""

    def __init__(self, settings: Settings | None = None) -> None:
        # Config from the central boundary (rule 3: no hardcoding); DI-friendly override.
        self._settings: Settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings
