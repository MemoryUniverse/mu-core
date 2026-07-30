"""lifecycle/ — the Memory Lifecycle Manager (MLM) subtree (mu_engine).

Stage-0 (S0-01..S0-10) contracts + engine-foundation slices have landed and are re-exported
below with an explicit ``__all__`` (mypy ``no_implicit_reexport``), mirroring
``mu_engine.services``' pattern:

- ``mode_gate.py``   (S0-03) — ``ManagerMode``/``ManagerModeGate``/``ManagerOwnsLifecycleError``/
                      ``ModePolicyResolver``.
- ``settings.py``    (S0-07) — the ``LifecycleSettings`` central-config tree + its sub-trees.
- ``salience.py``    (S0-08) — ``SalienceStrategy``.
- ``dto.py``         (S0-09) — ``TransitionKind``/``SalienceInputs``/``ModelVerdict``/
                      ``LifecycleStateView``.
- ``explain.py``     (S0-09) — ``ExplainRecord``.
- ``mtm_graph.py``   (S0-10) — ``MtmWorkingGraphSettings``/``MtmWorkingGraphService`` (RESERVED
                      seam, disabled). ``MtmWorkingGraphSettings`` is defined ONCE here (this
                      module) and re-exported by ``settings.py`` — not redefined there (DRY,
                      integrate-phase fix: S0-07 and S0-10 independently drafted the class).

Stage-1+ slices (``promotion.py``, ``demotion.py``, ``manager.py``, ``retention.py``,
``conflict.py``) are NOT yet landed and are intentionally absent from this aggregate — they join
this re-export as their own stage merges.
"""

from mu_engine.lifecycle.dto import (
    LifecycleStateView,
    ModelVerdict,
    SalienceInputs,
    TransitionKind,
)
from mu_engine.lifecycle.explain import ExplainRecord
from mu_engine.lifecycle.mode_gate import (
    ManagerMode,
    ManagerModeGate,
    ManagerOwnsLifecycleError,
    ModePolicyResolver,
)
from mu_engine.lifecycle.mtm_graph import MtmWorkingGraphService, MtmWorkingGraphSettings
from mu_engine.lifecycle.salience import SalienceStrategy
from mu_engine.lifecycle.settings import (
    HostedMirrorConsent,
    LifecycleSettings,
    ManagerModeSettings,
    OwnershipSettings,
    RetentionSettings,
    SalienceSettings,
)

__all__ = [
    "ExplainRecord",
    "HostedMirrorConsent",
    "LifecycleSettings",
    "LifecycleStateView",
    "ManagerMode",
    "ManagerModeGate",
    "ManagerModeSettings",
    "ManagerOwnsLifecycleError",
    "ModePolicyResolver",
    "ModelVerdict",
    "MtmWorkingGraphService",
    "MtmWorkingGraphSettings",
    "OwnershipSettings",
    "RetentionSettings",
    "SalienceInputs",
    "SalienceSettings",
    "SalienceStrategy",
    "TransitionKind",
]
