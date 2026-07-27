"""mu-local — the embedded, daemonless, in-process engine facade (PACKAGING-v2 §1.3).

Depends on {mu-contracts, mu-engine}; NEVER mu-server (CI: mu-local-no-server / import-linter
mu-local-layers). The library form of FULL-LOCAL minus the daemon. Two public objects — the
:class:`LocalMemory` facade and its :class:`LocalContainer` composition root — plus config + the
result views + the library-local error types.
"""

from mu_local.composition import LocalContainer
from mu_local.config import BackendChoice, ModelProfileSettings, StorageSettings
from mu_local.errors import BackendUnavailableError, LlmNotConfiguredError
from mu_local.local_memory import LocalMemory
from mu_local.views import (
    ConsolidateView,
    ContextView,
    MemoryListView,
    MemoryRecordView,
    MemoryWriteResult,
)

__all__ = [
    "BackendChoice",
    "BackendUnavailableError",
    "ConsolidateView",
    "ContextView",
    "LlmNotConfiguredError",
    "LocalContainer",
    "LocalMemory",
    "MemoryListView",
    "MemoryRecordView",
    "MemoryWriteResult",
    "ModelProfileSettings",
    "StorageSettings",
    "__version__",
]

__version__ = "0.1.0"
