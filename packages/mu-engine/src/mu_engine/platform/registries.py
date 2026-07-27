"""The named app-singleton registries (platform-layer0-spec §7).

Registration lives beside each adapter (``@bus_registry.register("redis_streams")``); importing a
plane container imports the adapter packages, running the decorators (spec §7). These are the five
process-wide registries; the role-keyed ``StoreRegistry`` (storage-pluggable §4.2) wraps the base
:class:`Registry` separately.

The ``provider``/``store``/``algorithm`` registries are typed ``Registry[object]`` at Layer-0: their
concrete port types (``LLMProviderPort``/store roles/strategy ports) are populated by their owning
phases; Layer-0 only creates the sockets. ``bus``/``workflow`` are typed to their contracts ports
because the Layer-0 composition root builds from them directly.
"""

from __future__ import annotations

from mu_contracts.ports.bus import EventBusPort
from mu_contracts.ports.workflow import WorkflowRunnerPort
from mu_engine.platform.registry import Registry

__all__ = [
    "algorithm_registry",
    "bus_registry",
    "provider_registry",
    "store_registry",
    "workflow_registry",
]

#: LLM/embedding provider adapters (model-layer phase populates).
provider_registry: Registry[object] = Registry("provider")
#: Store adapters (storage phase populates via its role-keyed StoreRegistry wrapper).
store_registry: Registry[object] = Registry("store")
#: Swappable strategy ports — the crown-jewel substitution seam (spec §7, PACKAGING-v2 §2.2/§3).
algorithm_registry: Registry[object] = Registry("algorithm")
#: Event bus adapters (``inproc`` for tests, ``redis_streams`` default — spec §8.3).
bus_registry: Registry[EventBusPort] = Registry("bus")
#: Workflow runner adapters (``inline`` on LOCAL, ``temporal`` on SHARED — spec §9).
workflow_registry: Registry[WorkflowRunnerPort] = Registry("workflow")
