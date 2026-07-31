"""surface/ — the ONE canonical verb surface over the embedded LOCAL composition root.

Landed (build-queue §13 item 1): ``facade.py`` — ``SurfaceFacade`` (``add``/``recall``/``get``/
``consolidate``/``ask``, honest ``501``-shaped refusals for ``build_context``/``promote``/
``demote``/``share`` pending build-queue §13 item 3), its structural composition-root Protocol
(``LocalContainerLike``), and the two locally-defined named errors
(``SurfaceVerbNotImplementedError``/``LlmNotConfiguredError``) it raises. See ``facade.py``'s
module docstring for the full placement rationale and import-boundary discipline.
"""

from mu_engine.surface.facade import (
    LlmNotConfiguredError,
    LocalContainerLike,
    SurfaceFacade,
    SurfaceVerbNotImplementedError,
)

__all__ = [
    "LlmNotConfiguredError",
    "LocalContainerLike",
    "SurfaceFacade",
    "SurfaceVerbNotImplementedError",
]
