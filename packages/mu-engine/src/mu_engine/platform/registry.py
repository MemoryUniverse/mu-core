"""``Registry[T]`` — the Strategy + Registry pattern, fail-loud (platform-layer0-spec §7).

Ported from mem0's open-registration factory (CODE-ADOPTION-METHODOLOGY):
  * ``provider_to_class`` dict + lookup — ``other_repos/mem0/mem0/utils/factory.py:35``.
  * unknown-key RAISE (never a silent default) — ``factory.py:71-72``.
  * open registration (``register_provider``) — ``factory.py:112-123``.
  * name listing — ``factory.py:125-129`` (``get_supported_providers``).

DELIBERATE DEVIATIONS (CODE-ADOPTION step 4):
  1. mem0 raises a bare ``ValueError``; we raise the TYPED ``UnknownComponentError`` /
     ``DuplicateComponentError`` so the global handler maps them (spec §6/§7).
  2. mem0 registers ``(class_path, config_class)`` string tuples resolved by ``load_class``;
     we register a LAZY factory ``Callable[[Settings], T]`` so there is **no import-time socket**
     (spec §7: the socket opens only at ``LifecycleManager.start()``), and the registry is
     TYPED/generic instead of a stringly-typed dict.
  3. duplicate registration RAISES here (mem0 silently overwrites) — the anti-silent-fallback
     rule applied to wiring (spec §7).

Home: ``mu-engine`` (``platform/registry.py``, spec §0.2). The role-keyed ``StoreRegistry``
specialization (``storage-pluggable-spec §4.2``) and the named app-singleton registries
(``provider_registry``/``bus_registry``/… — spec §7) build on top of THIS base and land with
the composition root once their port types exist.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from mu_contracts.config.settings import Settings
from mu_contracts.domain.errors import DuplicateComponentError, UnknownComponentError

__all__ = ["Registry"]

T = TypeVar("T")

#: A lazy factory: given the injected ``Settings`` (never process env — spec §1.1) it constructs
#: the component. Lazy so registration at import opens no socket (spec §7).
Factory = Callable[[Settings], T]


class Registry(Generic[T]):
    """A typed, fail-loud open registry of ``key -> Factory[T]`` (spec §7).

    Fail-loud contract (NON-NEGOTIABLE, spec §7): an unknown key raises ``UnknownComponentError``
    (never a default); a duplicate key raises ``DuplicateComponentError``. This is the
    anti-silent-fallback rule (DEV-STANDARDS rule 8) applied to wiring.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._factories: dict[str, Factory[T]] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, key: str) -> Callable[[Factory[T]], Factory[T]]:
        """Decorator registering ``factory`` under ``key``. Runs at import (beside each adapter);
        opens no socket. Duplicate ``key`` fails loud (spec §7, deviation #3)."""

        def deco(factory: Factory[T]) -> Factory[T]:
            if key in self._factories:
                raise DuplicateComponentError(f"{self._name}:{key}")
            self._factories[key] = factory
            return factory

        return deco

    def register_factory(self, key: str, factory: Factory[T]) -> None:
        """Imperative form of :meth:`register` (same fail-loud contract), for call-site wiring."""
        if key in self._factories:
            raise DuplicateComponentError(f"{self._name}:{key}")
        self._factories[key] = factory

    def create(self, key: str, settings: Settings) -> T:
        """Build the component for ``key`` (the socket opens here, not at import). Unknown key
        raises ``UnknownComponentError`` listing the known keys (ported mem0 factory.py:71-72,
        deviation #1)."""
        try:
            factory = self._factories[key]
        except KeyError as exc:
            raise UnknownComponentError(
                f"{self._name}:{key} (known: {sorted(self._factories)})"
            ) from exc
        return factory(settings)

    def is_registered(self, key: str) -> bool:
        return key in self._factories

    def names(self) -> tuple[str, ...]:
        """The registered keys, stably sorted (ported mem0 factory.py:125-129)."""
        return tuple(sorted(self._factories))
