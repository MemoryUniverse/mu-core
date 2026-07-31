"""Structural (PEP 544) route-layer ports — build-plan §4 C2, item 2.

``mu_engine_server.routes`` never imports the concrete :class:`~mu_engine.surface.facade.
SurfaceFacade`/:class:`~mu_engine.lifecycle.manager.MemoryLifecycleManager` classes directly in
its FUNCTION SIGNATURES — it depends on these two narrow Protocols instead, mirroring the SAME
discipline :class:`~mu_engine.surface.facade.LocalContainerLike` already uses one layer down
(``mu-engine/surface/facade.py``: "the composition root is taken as a constructor argument —
dependency injected — never built here"). Two reasons, not one:

1. **C4 dependency injection** (build-plan §4: "the app takes its SurfaceFacade/composition via
   dependency injection from C4's EngineServerApp") — :func:`mu_engine_server.app.build_app` binds
   against these Protocols, so C4 can hand it a real ``SurfaceFacade``/``MemoryLifecycleManager``
   (both satisfy these structurally, zero adapter code) OR any other object with the same shape.
2. **C2's own unit-tier route tests** (this task's REAL-TEST GATE) construct a hand-built fake
   satisfying :class:`MemorySurfacePort` with zero real infra/composition-root construction — the
   identical "fake ``LocalContainerLike``" pattern ``mu-engine/tests/test_surface_facade_int.py``
   already uses for the layer below.

``MemorySurfacePort`` is the exact subset of :class:`~mu_engine.surface.facade.SurfaceFacade`'s
public verbs the §2.1 route inventory serves (``add``/``recall``/``get``/``consolidate``/
``build_context``/``promote``/``demote``) — ``ask``/``share`` are deliberately absent (no route in
§2.1 calls them). ``LifecycleSurfacePort`` is the exact subset of
:class:`~mu_engine.lifecycle.manager.MemoryLifecycleManager`'s public API the ``GET /profile``/
``POST /lifecycle/enforce``/``GET /lifecycle/events`` routes call (``get_state``/``sweep_user``/
``explain``) — see ``routes/lifecycle.py``'s own module docstring for why exactly these three verbs
were picked to back that trio (a mapping the design's route table names but does not itself
resolve, since ``mu_engine.lifecycle.manager`` has no route-shaped ``profile``/``enforce``/
``events`` methods of its own).
"""

from __future__ import annotations

from typing import Any, NoReturn, Protocol

from mu_contracts.contracts.memory import MemoryResponse
from mu_contracts.contracts.recall import RecallResult
from mu_contracts.contracts.views import ConsolidateView, ContextView, MemoryWriteResult
from mu_contracts.domain.model.lifecycle import JobHandle, UserPrefix
from mu_contracts.domain.model.memory import Namespace
from mu_engine.lifecycle.dto import LifecycleStateView
from mu_engine.lifecycle.explain import ExplainRecord
from mu_engine.storage.domain.memory import MemoryTier

__all__ = ["LifecycleSurfacePort", "MemorySurfacePort"]


class MemorySurfacePort(Protocol):
    """The verb subset of :class:`~mu_engine.surface.facade.SurfaceFacade` the §2.1 memory/context
    routes call. A real ``SurfaceFacade`` satisfies this structurally with zero adapter code."""

    async def add(
        self,
        content: str | dict[str, Any] | list[dict[str, Any]],
        *,
        user: str,
        session: str | None,
    ) -> MemoryWriteResult: ...

    async def recall(
        self,
        query: str,
        *,
        user: str,
        session: str | None,
        tier: MemoryTier | None,
        limit: int,
    ) -> RecallResult: ...

    async def get(
        self, memory_id: str, *, user: str, session: str | None
    ) -> MemoryResponse | None: ...

    async def consolidate(
        self, *, user: str, session: str | None, limit: int
    ) -> ConsolidateView: ...

    async def build_context(
        self,
        query: str,
        *,
        user: str,
        session: str | None,
        limit: int,
        max_chars: int | None,
    ) -> ContextView: ...

    async def promote(self, memory_id: str, *, user: str, session: str | None) -> NoReturn: ...

    async def demote(self, memory_id: str, *, user: str, session: str | None) -> NoReturn: ...


class LifecycleSurfacePort(Protocol):
    """The verb subset of :class:`~mu_engine.lifecycle.manager.MemoryLifecycleManager` the §2.1
    ``profile``/``lifecycle.*`` routes call. A real ``MemoryLifecycleManager`` satisfies this
    structurally with zero adapter code. ``None`` (no lifecycle manager injected) is a valid,
    named-501 configuration — see ``routes/lifecycle.py``."""

    def get_state(self, ns: Namespace) -> LifecycleStateView: ...

    def explain(self, ns: Namespace, memory_id: str) -> list[ExplainRecord]: ...

    async def sweep_user(self, prefix: UserPrefix, *, manual: bool = False) -> JobHandle: ...
