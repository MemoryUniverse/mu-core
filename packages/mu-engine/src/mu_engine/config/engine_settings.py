"""``EngineSettings`` — the intelligence-knob aggregator (CONFIG-AND-DATA-FIX-PLAN.md PART 0 §0.3,
PART 1 §1.2 stage C0).

**The problem this fixes.** ``mu-engine`` already ships good named settings classes
(``RecallSettings``, ``LifecycleSettings``, ``IngestSettings``, ``DistillSettings``,
``ExtractionSettings``, ``ModelSettings``, ``ModelCatalogSettings``, ``RetrySettings``,
``ObservabilitySettings``, ``LedgerSettings``) — every threshold/weight/timeout is already a named
``pydantic.BaseModel`` field, never a bare literal at a call site (DEV-STANDARDS rule 3). But each
one is a plain, ORPHAN ``BaseModel``: composition roots construct it bare (``RecallSettings()``,
``DistillSettings()``, ...) with no path from the environment to its fields. The
``recency_floor_limit=10`` regression fixed in ``02fbed9`` (a hardcoded value inside exactly this
class of orphan settings, colliding with an unrelated hardcoded ``10`` elsewhere) is the canonical
symptom: a knob nobody could see or override until it broke recall.

**Why a NEW aggregator here, not a field on `mu_contracts.config.Settings`.** ``Settings`` (the
``MU_`` root) lives in ``mu-contracts`` — the BOTTOM layer (``.importlinter`` ``core-layers``:
``mu_engine`` sits ABOVE ``mu_contracts``; ``contracts-imports-nothing-in-project`` forbids
``mu_contracts`` from importing ``mu_engine`` OR ``mu_local``). Every orphan class above lives IN
``mu-engine``. Mounting them directly as sibling fields on ``mu_contracts.config.Settings`` would
force ``mu-contracts`` to import ``mu-engine`` — a dependency inversion the layering contract
forbids. The resolution (plan §0.3): a SECOND ``BaseSettings`` root, owned by ``mu-engine`` itself,
that mounts ONLY the subtrees that live in this package. ``mu_contracts.Settings`` keeps
``runtime_mode``/``storage`` (the truly cross-cutting, contracts-level knobs); this class owns
every intelligence knob. Composition roots read whichever root owns the knob they need. Both roots
share the SAME ``MU_`` env-prefix family and the SAME ``__`` nested-delimiter convention (this
class's ``model_config`` is deliberately identical in shape to ``mu_contracts.config.settings.
Settings.model_config``), so from the operator's view it is one flat env namespace even though it
is two Python roots — e.g. ``MU_RECALL__WEIGHT_MTM=1.4``, ``MU_LIFECYCLE__PROMOTE_STM_MTM=0.65``,
``MU_MODEL__MAX_OUTPUT_TOKENS=2048`` all resolve through THIS class; ``MU_STORAGE__POSTGRES__PORT``
resolves through ``mu_contracts.config.settings.Settings``.

**Scope of this slice (C0 — no behavior change).** This module ONLY wires the mount points and the
cached accessor. Every field keeps its EXACT current bare default (each subtree is constructed via
its own zero-arg ``default_factory``, so ``EngineSettings().recall == RecallSettings()`` field-for-
field) — no threshold/weight changes, no composition-root wiring yet. Threading the wired instance
into ``mu-local``/``mu-engine-server`` composition roots (so pipelines stop calling
``RecallSettings()``/``DistillSettings()`` bare) is stage **C1** (Group A — highest risk), which
depends on this class existing first.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.pipelines.distill import DistillSettings
from mu_engine.pipelines.ledger import LedgerSettings
from mu_engine.platform.settings import ObservabilitySettings, RetrySettings
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings
from mu_engine.services.conflict.settings import ConflictSettings
from mu_engine.services.extract import ExtractionSettings
from mu_engine.services.recall.dto import RecallSettings
from mu_engine.services.settings import IngestSettings

__all__ = ["EngineSettings", "get_engine_settings"]


class EngineSettings(BaseSettings):
    """The ONE intelligence-knob ``BaseSettings`` root for ``mu-engine`` (plan §0.3).

    Every field is one of the pre-existing "tracked seam" ``BaseModel`` subtrees, unmodified,
    mounted via ``Field(default_factory=...)`` so each keeps its own independent, freshly-
    constructed default (no shared mutable default, mirrors ``providers.settings.
    default_local_catalog``'s own factory-not-constant discipline). Env override example:
    ``MU_RECALL__WEIGHT_MTM=1.4`` (nested delimiter ``__`` reaches ``recall.weight_mtm``).
    """

    model_config = SettingsConfigDict(
        env_prefix="MU_",
        env_nested_delimiter="__",
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    recall: RecallSettings = Field(default_factory=RecallSettings)
    distill: DistillSettings = Field(default_factory=DistillSettings)
    #: ``conflict-resolution-async-design.md`` §4.1 step 3 — *"Workspace/global default —
    #: ``settings.conflict.default_policy``"*. Until this mount existed there was NO path from
    #: the environment to it: ``ConflictPolicyResolver`` fell back to a freshly-constructed
    #: ``ConflictSettings()`` on every conflict, so step 3 of the precedence chain always
    #: resolved to a hardcoded ``ConflictResolutionPolicy()`` and the workspace-level knob the
    #: chain's last step names could not be set at all.
    #:
    #: Mounted HERE and not on ``mu_contracts.config.settings.Settings``, for this class's own
    #: founding reason (module docstring): ``ConflictSettings`` lives in ``mu-engine`` and its
    #: ``default_policy`` field is typed ``mu_engine.lifecycle.conflict.ConflictResolutionPolicy``,
    #: so mounting it on the contracts root would force ``mu_contracts`` to import ``mu_engine``
    #: — exactly what the ``contracts-imports-nothing-in-project`` import-linter contract forbids.
    #: Env reach: ``MU_CONFLICT__DEFAULT_POLICY__MODE=manual``.
    conflict: ConflictSettings = Field(default_factory=ConflictSettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    lifecycle: LifecycleSettings = Field(default_factory=LifecycleSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    model_catalog: ModelCatalogSettings = Field(default_factory=ModelCatalogSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    ledger: LedgerSettings = Field(default_factory=LedgerSettings)


@lru_cache
def get_engine_settings() -> EngineSettings:
    """The single, cached read of the ``mu-engine`` intelligence-knob env boundary.

    Mirrors ``mu_contracts.config.settings.get_settings`` exactly (same ``@lru_cache`` discipline
    — construct ``EngineSettings()`` here, never elsewhere, so the environment is read exactly
    once per process). Tests that need a fresh read after mutating ``os.environ`` must call
    ``get_engine_settings.cache_clear()`` first (same contract ``get_settings`` already implies).
    """
    return EngineSettings()
