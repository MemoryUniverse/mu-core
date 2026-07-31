"""mu-engine-server — thin single-tenant HTTP server over the ``SurfaceFacade`` (``mu-engine``),
backed by durable stores (design ``2026-07-31-sdk-engine-server-design.md`` §2.1).

Import boundary (HARD, CI: ``mu-engine-server-boundary``): ``mu_engine_server`` may import
``{mu_contracts, mu_engine}`` ONLY — never ``mu_local``, never ``mu_server``, never ``mu_client``.

Scaffold stage (build-plan §4 C0): no app/route/auth code lands here yet — that is C1-C4. This
module intentionally has no public surface until the FastAPI app (C2) and its composition root
(C4) exist.
"""

from __future__ import annotations

__all__: list[str] = ["__version__"]

__version__ = "0.1.0"
