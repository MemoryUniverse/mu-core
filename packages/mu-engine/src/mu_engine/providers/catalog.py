"""Catalog value shapes — the many-to-many model↔provider vocabulary (model-layer-spec §2.1).

These are the frozen pydantic records the L1 registry (`registry.py`) compiles into LiteLLM's
flat `model_list`, plus the `Task` taxonomy (§2.3) LiteLLM does not carry, and the warm-local
loader config (§2.5). Split into its own module (no internal imports) so `settings.py`,
`task_map.py`, `local_priority.py`, and `registry.py` share one definition with no import cycle.

Ported vocabulary source: mem0 `LlmFactory.provider_to_class` registry
(`other_repos/mem0/mem0/utils/factory.py:35`) and MemOS `LLMFactory.backend_to_class`
(`other_repos/MemOS/src/memos/llms/factory.py:18-24`), generalized from one-to-one
provider→class into a many-to-many model_group↔provider catalog (model-layer-spec §2.1).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ModelDeployment",
    "ModelKind",
    "ProviderKind",
    "ProviderRecord",
    "Task",
    "WarmLocalConfig",
]


class Task(StrEnum):
    """The task taxonomy (model-layer-spec §2.3). Each maps to a `ModelSettings` field.

    LiteLLM has model groups but no task concept — we own this enum (research §5.1).
    """

    ANSWER = "answer"  # -> ModelSettings.answer_model
    ADJUDICATE = "adjudicate"  # -> adjudicate_model (conflict resolution)
    HARD_EXTRACT = "hard_extract"  # -> hard_extract_model (DISTILL + conflict/date extract)
    ROUTINE_EXTRACT = "routine_extract"  # -> routine_extract_model
    SUMMARIZE = "summarize"  # -> summarize_model
    CLASSIFY = "classify"  # -> classify_model
    RERANK = "rerank"  # -> rerank_model
    EMBED = "embed"  # -> embed_backend (EmbeddingPort seam) / embed_model


class ProviderKind(StrEnum):
    """How a provider is reached (model-layer-spec §2.1)."""

    REMOTE = "remote"  # cloud OpenAI-format provider (azure/openai/anthropic/bedrock/...)
    LOCAL_HTTP = "local_http"  # localhost OpenAI-compatible endpoint (ollama/vllm/tgi/tei)
    LOCAL_INPROC = "local_inproc"  # in-process warm singleton via CustomLLM (L5), tag "mu-local"


class ModelKind(StrEnum):
    """One taxonomy for what a deployment serves (model-layer-spec §2.1, §2.7)."""

    LLM = "llm"
    EMBED = "embed"
    RERANK = "rerank"


class ProviderRecord(BaseModel):
    """A provider the plane can reach (model-layer-spec §2.1 `ProviderRecord`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str  # registry key, e.g. "azure", "openai", "ollama_local", "mu-local"
    kind: ProviderKind
    litellm_provider: str  # the litellm provider prefix ("azure", "openai", "ollama", "mu-local")
    api_base: str | None = None  # localhost:11434 etc.; None for cloud (litellm defaults)
    credential_ref: str | None = None  # name under Settings(secrets_dir=...); NEVER inline a key
    is_local: bool = False  # feeds the local-priority predicate (L4)


class ModelDeployment(BaseModel):
    """One (model_group, provider) binding (model-layer-spec §2.1 `ModelDeployment`).

    MANY deployments may share one `model_group` (the many side); a provider may serve M
    groups. `priority_order` is LiteLLM's `order:` (1 = local/preferred, 2 = remote fallback);
    it is STAMPED by the L1 compiler from the L4 predicate, not authored per row unless the
    operator overrides it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_group: str  # the LOGICAL name callers/tasks reference ("gpt-4.1-mini", "slm-fast")
    provider_key: str  # -> ProviderRecord.key (the many side)
    model_id: str  # provider-native id ("azure/gpt-4.1-mini", "mu-local/phi-4-mini")
    kind: ModelKind = ModelKind.LLM
    priority_order: int = 2  # LiteLLM `order:` — default remote-fallback tier (L4 re-stamps local)
    max_input_tokens: int | None = None  # for chunking (L6); else litellm.get_max_tokens
    weight: int | None = None  # optional LB weight within a group
    extra_params: dict[str, Any] = Field(default_factory=dict)  # passthrough litellm_params


class WarmLocalConfig(BaseModel):
    """Config for an in-process warm singleton (model-layer-spec §2.5 L5) OR an embed backend.

    Faithful-port note (CODE-ADOPTION step 4): the LLM fields mirror MemOS `HFLLMConfig` usage
    in `other_repos/MemOS/src/memos/llms/hf.py` — `model_name_or_path` (→ `model_load_path`),
    `max_tokens`, `temperature`, `top_k`, `top_p`, `do_sample`, `add_generation_prompt`,
    `remove_think_prefix` (hf.py:36-54, 66-67, 105-125). DEVIATION: the singleton config key
    derives from `model_id` (our deployment id) rather than MemOS `model_name_or_path`
    (hf_singleton.py:69) — recorded here per the methodology. The EMBED slot
    (`normalize_embeddings`) is our extension of the same singleton to a sentence-transformers
    embedder (kind=EMBED) — a documented deviation from MemOS's causal-only loader.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str  # the litellm deployment id AND the singleton key (deviation from MemOS)
    kind: ModelKind = ModelKind.LLM
    model_load_path: str  # HF repo/path (causal LM) or sentence-transformers model id (embed)
    device_map: str = "auto"  # hf.py:41
    torch_dtype: str = "auto"  # hf.py:41
    # --- causal-LM generation params (hf.py) ---
    max_tokens: int = 512  # hf.py:105 max_new_tokens
    temperature: float = 1.0  # hf.py:50,109 (1.0 = no temperature warper)
    top_k: int = 0  # hf.py:51,110
    top_p: float = 1.0  # hf.py:53,111
    do_sample: bool = True  # hf.py:106,108
    add_generation_prompt: bool = True  # hf.py:67
    remove_think_prefix: bool = False  # hf.py:124
    # --- embed params (our extension) ---
    normalize_embeddings: bool = True
    device: str | None = None  # sentence-transformers device override; None = auto
