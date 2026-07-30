"""``ManagerModeGate`` — the engine-side authority for the three manager modes (ADR 0031; spec §3,
lines 131-161; ``PROPOSED-CANONICAL-ADDITIONS-mlm.md`` P2 lines 56-62;
``2026-07-22-memory-universe-software-architecture.md`` lines 396-402).

A ``manager_mode`` (central-config, resolved **memory ▷ namespace ▷ workspace-default**) decides
how much control a caller keeps over the three manual lifecycle verbs (``consolidate``/``promote``/
``demote``). The **ENGINE enforces**; the SDK/client/server merely *select* a mode and carry it on
the wire — they cannot self-authorize (DEV-STANDARDS rule 8: central exception handling, fail-loud,
no silent fallback). This module is that one gate, consulted at the entry of every lifecycle write
verb, in ``mu-engine`` so no surface (REST/MCP/SDK/CLI) can bypass it.

| Mode        | Manual verb             | Semantics                                          |
|-------------|--------------------------|-----------------------------------------------------|
| ``MANUAL``  | allowed                 | full control; caller drives the sweep (daemonless) |
| ``MANAGED`` | **409** (this raises)   | auto only; manual lifecycle verbs refused, loud    |
| ``HYBRID``  | allowed (under a lease) | auto + manual coexist; lease is ADR 0032's concern |

``ModePolicyResolver`` (spec §17a, lines 665-669) is the ONE seam this gate depends on: an injected
Protocol that walks the memory ▷ namespace ▷ workspace-default resolution order and returns the
single effective ``ManagerMode`` for a given ``Namespace``. The gate itself is a pure decision
function over that already-resolved mode — it is deliberately NOT a settings-reader for the
resolution order itself (that order's concrete implementation is a composition-root concern,
out of this Stage-0, container-free task).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from mu_contracts.domain.errors import MemoryUniverseError
from mu_contracts.domain.model.memory import Namespace
from mu_engine.lifecycle.settings import ManagerModeSettings

__all__ = [
    "ManagerMode",
    "ManagerModeGate",
    "ManagerOwnsLifecycleError",
    "ModePolicyResolver",
]


class ManagerMode(StrEnum):
    """The three manager modes (ADR 0031; spec §3)."""

    MANUAL = "manual"
    MANAGED = "managed"
    HYBRID = "hybrid"


class ManagerOwnsLifecycleError(MemoryUniverseError):
    """Raised when a manual lifecycle verb is attempted against a ``MANAGED`` namespace (ADR 0031;
    spec §3) — the surface maps this to **HTTP 409** (``2026-07-22-...-architecture.md:402``,
    ``PROPOSED-CANONICAL-ADDITIONS-mlm.md`` P2). Carries ``ns``/``verb`` (namespace identifiers and
    the attempted verb name — never memory content) so the caller/surface can build a precise,
    content-free 409 response without re-deriving what was refused."""

    def __init__(self, *, ns: Namespace, verb: str) -> None:
        self.ns = ns
        self.verb = verb
        super().__init__(
            f"manager owns the lifecycle: verb {verb!r} refused for ns={ns.to_prefix()!r} "
            "(namespace resolves to MANAGED — auto sweep only)"
        )


class ModePolicyResolver(Protocol):
    """Injected into :class:`ManagerModeGate` (spec §3) so the gate stays a pure decision function,
    never a settings-reader itself. Resolves the effective :class:`ManagerMode` by walking
    **memory ▷ namespace ▷ workspace-default** (spec §3's stated resolution order) — the ONE method
    this port needs (spec §17a, lines 665-669)."""

    def resolve(self, ns: Namespace) -> ManagerMode: ...


class ManagerModeGate:
    """Engine-side authority (spec §3). Resolves the effective mode via the injected
    :class:`ModePolicyResolver` and admits/refuses a manual lifecycle verb. The SDK/client/server
    SELECT (carry ``manager_mode`` as a routing hint on the wire); the ENGINE enforces here —
    ``SurfaceFacade``/``LocalMemory`` call :meth:`assert_manual_allowed` before delegating, so the
    client can never self-authorize (ADR 0031)."""

    def __init__(self, settings: ManagerModeSettings, resolver: ModePolicyResolver) -> None:
        self._settings = settings
        self._resolver = resolver

    def assert_manual_allowed(self, ns: Namespace, verb: str) -> None:
        """Raise :class:`ManagerOwnsLifecycleError` iff the resolved mode is ``MANAGED``; pass
        silently for ``MANUAL``/``HYBRID`` (spec §3 — HYBRID additionally runs under the
        user-grained lifecycle lease, ADR 0032, which is this gate's caller's concern, not this
        gate's).

        ``settings.enforce_engine_side=False`` is the ADR 0031 rollback switch (rollout/rollback
        section): it bypasses enforcement entirely so a MANAGED-by-default rollout can be turned
        off without a data change — this is the ONE thing the gate itself reads off ``settings``;
        the resolution order (memory ▷ namespace ▷ workspace-default) is entirely the injected
        :class:`ModePolicyResolver`'s job, never re-derived here (DRY, no second derivation).
        """
        if not self._settings.enforce_engine_side:
            return
        mode = self._resolver.resolve(ns)
        if mode is ManagerMode.MANAGED:
            raise ManagerOwnsLifecycleError(ns=ns, verb=verb)
        # MANUAL & HYBRID: allowed; HYBRID additionally acquires the lifecycle lease (§4b) —
        # the caller's responsibility, not this gate's.
