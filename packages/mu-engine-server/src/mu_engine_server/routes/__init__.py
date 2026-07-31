"""§2.1 route inventory — one module per resource family (build-plan §4 C2, item 2).

``memories.py`` — ``POST /memories``, ``GET /memories/{id}``, ``POST /v1/memories/recall``,
``POST /v1/memories/consolidate``, ``POST /v1/memories/{id}/promote|demote``.
``context.py`` — ``POST /v1/context/window`` (the private-plane ``build_context`` twin ONLY; the
3 shared-plane ``context.*`` governed-transfer routes are absent by design, §2.1 REVIEW-3 C1).
``lifecycle.py`` — ``GET /profile``, ``POST /lifecycle/enforce``, ``GET /lifecycle/events``.

Every router in this package is assembled by :func:`mu_engine_server.app.build_app`, never
imported/mounted anywhere else — this package has no module-level ``FastAPI()`` instance of its
own (design §2.3: the app is C4's composition target, not a module-import side effect).
"""

from __future__ import annotations

__all__: list[str] = []
