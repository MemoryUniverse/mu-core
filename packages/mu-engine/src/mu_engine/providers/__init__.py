"""providers/ — model access behind the LLMProviderPort / EmbeddingPort (model-layer-spec).

The seven thin layers over LiteLLM (the adopted substrate — we delegate health/cooldown/fallback
into `litellm.Router`, never reimplement them; research §1.3):
  L1 registry.py         — ProviderRegistry + ModelRegistry (many-to-many → litellm model_list)
  L2 litellm_provider.py — LiteLLMRouterAdapter (owns ONE Router; delegation seam)
  L3 task_map.py         — Task→model-group mapper (+ override + default class)
  L4 local_priority.py   — local-PRIORITY predicate (mechanism = litellm `order:`)
  L5 warm_local.py       — startup-warm in-process models (ported MemOS HFSingletonLLM+HFLLM)
  L6 chunking.py         — long-text map-reduce over litellm token primitives
  L7 model_router.py     — DI ModelRouter façade implementing the canonical ports

The plane wiring (ENG-115a) sits above those seven:
  shipped_settings.py    — `ShippedCatalogSettings`, mounted on `ModelCatalogSettings.shipped`
  shipped_catalog.py     — the DECLARED multi-provider table (data, not machinery)
  secrets.py             — `SecretSeamResolver`: `credential_ref` NAME -> value (overrides/file/env)
  plane.py               — `build_plane_router`: settings -> probe -> ACTIVE catalog -> ModelRouter
This is what each composition root calls; before it existed the table above had no caller and the
plane shipped one embedder and zero LLM deployments.

Embedding (CANONICAL §6-P5) is a DEDICATED seam (embedding.py) — an offline sentence-transformers
MiniLM is the ONE active embedding backend, selected by `models.embed_backend`.

Ported per CODE-ADOPTION-METHODOLOGY (LiteLLM in-process CustomLLM path; MemOS HFSingletonLLM).
"""

from mu_engine.providers._contracts import (
    Chunk,
    Completion,
    DegradedModeEntered,
    DegradeEmitterPort,
    DegradeReason,
    EmbeddingPort,
    LLMProviderPort,
    Message,
    MessageRole,
    ModelGroupUnavailableError,
    ModelLayerError,
    RerankHit,
    RerankProviderPort,
    StreamingCompletionPort,
    Usage,
    Vector,
)
from mu_engine.providers.catalog import (
    HttpEmbedConfig,
    ModelDeployment,
    ModelKind,
    ProviderKind,
    ProviderRecord,
    Task,
    WarmLocalConfig,
)
from mu_engine.providers.embedding import (
    HttpEmbedder,
    HttpEmbedError,
    SentenceTransformerEmbedder,
    build_embedder,
)
from mu_engine.providers.model_router import ModelRouter, build_model_router
from mu_engine.providers.plane import (
    PlaneModelLayer,
    build_plane_router,
    build_plane_secret_resolver,
    resolve_plane_model_layer,
    resolve_task_models,
)
from mu_engine.providers.registry import ProviderModelRegistry, RegistryError, SecretResolver
from mu_engine.providers.secrets import SecretSeamResolver
from mu_engine.providers.settings import (
    CatalogSource,
    LocalFallbackPosture,
    ModelCatalogSettings,
    ModelSettings,
    RouterSettings,
    TaskDefaults,
    default_local_catalog,
)
from mu_engine.providers.shipped_catalog import (
    CredentialProbe,
    LegacyModelGroup,
    ModelGroup,
    ProviderKey,
    ShippedCatalogSettings,
    active_catalog,
    group_tasks,
    recommended_model_settings,
    resolvable_credential_refs,
    shipped_catalog,
    shipped_deployments,
    shipped_providers,
    shipped_router_fallbacks,
    shipped_warm_local,
)
from mu_engine.providers.task_map import TaskClassMapper

__all__ = [
    "CatalogSource",
    "Chunk",
    "Completion",
    "CredentialProbe",
    "DegradeEmitterPort",
    "DegradeReason",
    "DegradedModeEntered",
    "EmbeddingPort",
    "HttpEmbedConfig",
    "HttpEmbedError",
    "HttpEmbedder",
    "LLMProviderPort",
    "LegacyModelGroup",
    "LocalFallbackPosture",
    "Message",
    "MessageRole",
    "ModelCatalogSettings",
    "ModelDeployment",
    "ModelGroup",
    "ModelGroupUnavailableError",
    "ModelKind",
    "ModelLayerError",
    "ModelRouter",
    "ModelSettings",
    "PlaneModelLayer",
    "ProviderKey",
    "ProviderKind",
    "ProviderModelRegistry",
    "ProviderRecord",
    "RegistryError",
    "RerankHit",
    "RerankProviderPort",
    "RouterSettings",
    "SecretResolver",
    "SecretSeamResolver",
    "SentenceTransformerEmbedder",
    "ShippedCatalogSettings",
    "StreamingCompletionPort",
    "Task",
    "TaskClassMapper",
    "TaskDefaults",
    "Usage",
    "Vector",
    "WarmLocalConfig",
    "active_catalog",
    "build_embedder",
    "build_model_router",
    "build_plane_router",
    "build_plane_secret_resolver",
    "default_local_catalog",
    "group_tasks",
    "recommended_model_settings",
    "resolvable_credential_refs",
    "resolve_plane_model_layer",
    "resolve_task_models",
    "shipped_catalog",
    "shipped_deployments",
    "shipped_providers",
    "shipped_router_fallbacks",
    "shipped_warm_local",
]
