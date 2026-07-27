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
    ModelDeployment,
    ModelKind,
    ProviderKind,
    ProviderRecord,
    Task,
    WarmLocalConfig,
)
from mu_engine.providers.embedding import SentenceTransformerEmbedder, build_embedder
from mu_engine.providers.model_router import ModelRouter, build_model_router
from mu_engine.providers.registry import ProviderModelRegistry, RegistryError
from mu_engine.providers.settings import (
    ModelCatalogSettings,
    ModelSettings,
    RouterSettings,
    default_local_catalog,
)
from mu_engine.providers.task_map import TaskClassMapper

__all__ = [
    "Chunk",
    "Completion",
    "DegradeEmitterPort",
    "DegradeReason",
    "DegradedModeEntered",
    "EmbeddingPort",
    "LLMProviderPort",
    "Message",
    "MessageRole",
    "ModelCatalogSettings",
    "ModelDeployment",
    "ModelGroupUnavailableError",
    "ModelKind",
    "ModelLayerError",
    "ModelRouter",
    "ModelSettings",
    "ProviderKind",
    "ProviderModelRegistry",
    "ProviderRecord",
    "RegistryError",
    "RerankHit",
    "RerankProviderPort",
    "RouterSettings",
    "SentenceTransformerEmbedder",
    "StreamingCompletionPort",
    "Task",
    "TaskClassMapper",
    "Usage",
    "Vector",
    "WarmLocalConfig",
    "build_embedder",
    "build_model_router",
    "default_local_catalog",
]
