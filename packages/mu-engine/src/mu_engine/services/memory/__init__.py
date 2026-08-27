"""The ``MemoryRepository`` façade + its ``TierRouter`` (CANONICAL §6-P2).

``services/__init__.py`` listed "MemoryRepository facade + TierRouter" as unbuilt SCAFFOLD; this
package is that unit. It is what makes ``MemoryHealthService`` and ``PinService`` constructible,
and therefore what makes mu-client's ``/health`` / ``/pin`` / ``/unpin`` surfaces answer instead of
returning their named 503.
"""

from mu_engine.services.memory.repository import TieredMemoryRepository
from mu_engine.services.memory.router import TierLeg, TierRouter

__all__ = ["TierLeg", "TierRouter", "TieredMemoryRepository"]
