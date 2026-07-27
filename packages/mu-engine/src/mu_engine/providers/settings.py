"""Model-layer settings — `ModelSettings` (CANONICAL §7.2) + `ModelCatalogSettings` (§4).

CANONICAL §7.2 pins `ModelSettings`' canonical home at `config/settings.py` (platform-layer0),
with `mu-contracts/contracts` re-exporting it; `ModelCatalogSettings` is a SIBLING under
`Settings` owned by the model layer (§7.27, model-layer-spec §4).

PLACEMENT NOTE (tracked seam): those homes are still scaffold-empty and owned by other
phases. To stay inside this phase's `mu_engine/providers/` ownership boundary, the model layer
declares both models here with the EXACT §7.2 field names. `build_model_router` (model_router.py)
takes `models` + `catalog` explicitly, so the plane composition root wires them from
`settings.models` / `settings.model_catalog` once the central tree carries them — no re-shape.

Defaults live in the pydantic models (the sanctioned central-config home — DEV-STANDARDS rule 3);
no model id, host, or threshold is hardcoded in any code path.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from mu_engine.providers.catalog import (
    ModelDeployment,
    ModelKind,
    ProviderRecord,
    Task,
    WarmLocalConfig,
)

__all__ = [
    "ModelCatalogSettings",
    "ModelSettings",
    "RouterSettings",
    "default_local_catalog",
]

# The default local embedder backend key + model — the ONE active embedding backend this
# phase ships (works offline). Declared as a Settings default (central-config home), not a
# literal in a code path.
_DEFAULT_EMBED_BACKEND = "minilm_local"
_DEFAULT_MINILM_PATH = "sentence-transformers/all-MiniLM-L6-v2"


class ModelSettings(BaseModel):
    """CANONICAL §7.2 — the ONE per-task model config. Field names are frozen by CANONICAL.

    Every per-task field value is a **model-group name** that MUST exist in the catalog
    (validated at composition, §4). `embed_backend` is the `EmbeddingPort` registry key (the
    seam, R19); `embed_model` is only the internal id of a REMOTE embedder adapter.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str = "azure"  # provider_registry key
    answer_model: str = "gpt-5-chat"
    adjudicate_model: str = "gpt-5-chat"
    hard_extract_model: str = "gpt-5-chat"  # DISTILL fact-extraction AND conflict/date-extraction
    routine_extract_model: str = "gpt-4o"
    summarize_model: str = "gpt-4o"
    classify_model: str = "gpt-4.1-mini"
    rerank_model: str = "gpt-4.1-mini"  # a model-GROUP name, not necessarily an LLM (§CC-4)
    embed_backend: str = _DEFAULT_EMBED_BACKEND  # EmbeddingPort registry key — THE seam (R19)
    embed_model: str = "text-embedding-3-large"  # internal id of a REMOTE embedder adapter only
    model_override: str | None = None
    max_output_tokens: int = 4096
    temperature: float = 0.2


class RouterSettings(BaseModel):
    """`litellm.Router` knobs (model-layer-spec §4). Every value is delegated INTO the Router;
    the model layer implements none of health/cooldown/fallback (research §1.3)."""

    model_config = ConfigDict(extra="forbid")

    num_retries: int = 2
    timeout_s: float = 60.0
    cooldown_s: float = 60.0
    allowed_fails: int = 3
    health_interval_s: int = 300
    background_health_checks: bool = True
    strategy: str = "simple-shuffle"
    fallbacks: list[dict[str, list[str]]] = Field(default_factory=list)  # cross-group chains
    ctx_fallbacks: list[dict[str, list[str]]] = Field(default_factory=list)  # ctx-window fallbacks


class ModelCatalogSettings(BaseModel):
    """The many-to-many catalog + router knobs (model-layer-spec §4). `settings.model_catalog`.

    `embedders` maps an `embed_backend` key → the local embedder config it resolves to (the
    dedicated `EmbeddingPort` seam, CANONICAL §6-P5). `warm_local` is the L5 in-process LLM
    singleton list (empty on THIN). Per-task model-group names in `ModelSettings` MUST match a
    deployment group here (validated at startup — fail-loud, §5).
    """

    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderRecord] = Field(default_factory=list)
    deployments: list[ModelDeployment] = Field(default_factory=list)
    warm_local: list[WarmLocalConfig] = Field(default_factory=list)  # L5 in-proc LLM singletons
    embedders: dict[str, WarmLocalConfig] = Field(default_factory=dict)  # embed_backend -> config
    local_capable_tasks: list[Task] = Field(
        default_factory=lambda: [
            Task.EMBED,
            Task.RERANK,
            Task.CLASSIFY,
            Task.ROUTINE_EXTRACT,
            Task.SUMMARIZE,
        ]
    )
    local_priority_enabled: bool = True
    router: RouterSettings = Field(default_factory=RouterSettings)


def default_local_catalog() -> ModelCatalogSettings:
    """The LocalContainer default: an offline MiniLM embedder as the active embed backend, no
    cloud providers (the user adds them). This is the plane default described in §4 / §6 — the
    daemon ships the local embedder; THIN clients ship an empty catalog and route to server.

    NOTE this is a factory (not a module constant) so each call yields a fresh, independent
    catalog — no shared mutable default.
    """
    return ModelCatalogSettings(
        embedders={
            _DEFAULT_EMBED_BACKEND: WarmLocalConfig(
                model_id=_DEFAULT_EMBED_BACKEND,
                kind=ModelKind.EMBED,
                model_load_path=_DEFAULT_MINILM_PATH,
                normalize_embeddings=True,
            )
        },
    )
