"""``StoreRegistry`` — the pluggable role->backend->factory selection mechanism.

PORT of mem0's ``provider_to_class`` dict + ``load_class`` dynamic dispatch
(``/home/user/D/abstract_project/mma/other_repos/mem0/mem0/utils/factory.py:22-25,164-229``)
onto a fail-loud registry: an unknown key raises a NAMED ``UnknownBackendError`` (not a bare
``ValueError``) — ``storage-pluggable-spec.md §4.2``. Adapters self-register per (role,
backend); ``register_provider`` (mem0 ``factory.py:111-123``) lets a community adapter bind a
novel backend without forking.

``_assert_mandatory_roles`` is the build-time fail-loud refusal (spec §5/§7, owner overrides
1+2): vector + graph + relational MUST be bound; a brute-force vector backend on the SHARED
plane is REFUSED (D3, the one hard authz-completeness refusal).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mu_engine.storage.domain.namespace import Visibility
from mu_engine.storage.errors import (
    InvalidBackendError,
    MandatoryBackendMissingError,
    UnknownBackendError,
    VectorNotFilterableError,
)

__all__ = [
    "MANDATORY_ROLES",
    "StoreRegistry",
    "assert_mandatory_roles",
]

Factory = Callable[..., Any]

MANDATORY_ROLES = ("relational", "vector", "graph")

# graph fold values are invalid — graph is MANDATORY (owner override 1).
_FORBIDDEN_GRAPH_BACKENDS = frozenset({"none", "sqlfold", "pgvector-fold"})
# vector backends that cannot filter before ANN truncation -> PRIVATE-only (D3).
_BRUTE_FORCE_VECTOR_BACKENDS = frozenset({"faiss", "numpy"})


class StoreRegistry:
    """role -> { backend_key -> factory callable }, fail-loud (spec §4.2)."""

    def __init__(self) -> None:
        self._factories: dict[str, dict[str, Factory]] = {}

    def register(self, role: str, backend: str) -> Callable[[Factory], Factory]:
        """Decorator: bind a factory to a (role, backend) key (mem0 factory.py dict)."""

        def _decorate(factory: Factory) -> Factory:
            self.register_provider(role, backend, factory)
            return factory

        return _decorate

    def register_provider(self, role: str, backend: str, factory: Factory) -> None:
        """Programmatic registration (mem0 factory.py:111-123 extensibility seam)."""
        self._factories.setdefault(role, {})[backend] = factory

    def build(self, role: str, backend: str, /, **kwargs: Any) -> Any:
        """Instantiate the adapter bound to (role, backend); NAMED raise on miss."""
        role_map = self._factories.get(role)
        if role_map is None or backend not in role_map:
            raise UnknownBackendError(
                f"no backend {backend!r} registered for role {role!r} "
                f"(known: {sorted(role_map or {})})"
            )
        return role_map[backend](**kwargs)

    def known(self, role: str) -> frozenset[str]:
        return frozenset(self._factories.get(role, {}))


def assert_mandatory_roles(bound: dict[str, str], *, plane: Visibility) -> None:
    """Build-time fail-loud refusal (spec §5/§7). ``bound`` maps role -> chosen backend key.

    Refuses: a missing mandatory role; a graph fold; a brute-force vector on SHARED.
    """
    for role in MANDATORY_ROLES:
        if not bound.get(role):
            raise MandatoryBackendMissingError(
                f"mandatory storage role {role!r} has no bound backend "
                "(vector + graph + relational are all required; owner override 2)"
            )
    graph_backend = bound["graph"]
    if graph_backend in _FORBIDDEN_GRAPH_BACKENDS:
        raise InvalidBackendError(
            f"graph.backend={graph_backend!r} is invalid — graph is MANDATORY, "
            "no sqlfold/pgvector-fold/none (owner override 1)"
        )
    if plane is Visibility.SHARED and bound["vector"] in _BRUTE_FORCE_VECTOR_BACKENDS:
        raise VectorNotFilterableError(
            f"vector.backend={bound['vector']!r} cannot filter before ANN truncation; "
            "refused on the SHARED plane (D3 authz-completeness boundary)"
        )
