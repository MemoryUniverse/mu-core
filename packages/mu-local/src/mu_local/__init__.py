"""mu-local — the embedded, daemonless, in-process engine facade (PACKAGING-v2 §1.3).

Depends on {mu-contracts, mu-engine}; NEVER mu-server. The library form of FULL-LOCAL
minus the daemon.
"""

from mu_local.local_memory import LocalMemory

__all__ = ["LocalMemory", "__version__"]

__version__ = "0.1.0"
