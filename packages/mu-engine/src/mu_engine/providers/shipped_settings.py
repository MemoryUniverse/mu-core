"""``ShippedCatalogSettings`` — the environment-dependent knobs of the SHIPPED catalog.

Split out of :mod:`mu_engine.providers.shipped_catalog` (2026-08-28, ENG-115a wiring) for ONE
structural reason: :class:`~mu_engine.providers.settings.ModelCatalogSettings` now MOUNTS this
class as its ``shipped`` subtree, so ``MU_MODEL_CATALOG__SHIPPED__LOCAL_HTTP_API_BASE`` (and every
other knob below) reaches the catalog a composition root actually builds — one env namespace, the
sanctioned central-config home (DEV-STANDARDS rule 3). ``settings.py`` importing the class from
``shipped_catalog.py`` would be a cycle (``shipped_catalog`` imports ``ModelCatalogSettings``),
so the DATA class lives here, importing only :mod:`mu_engine.providers.catalog` — the same
"no internal imports" discipline ``catalog.py`` itself follows.

``shipped_catalog`` re-exports the name, so every existing
``from mu_engine.providers.shipped_catalog import ShippedCatalogSettings`` keeps working.

Secret VALUES never appear here — only ``*_credential_ref`` NAMES (house rule 1,
model-layer-spec §4).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ShippedCatalogSettings"]


class ShippedCatalogSettings(BaseModel):
    """The environment-dependent knobs of the shipped catalog (DEV-STANDARDS rule 3).

    Secret VALUES never appear here — only `*_credential_ref` NAMES, which are looked up through
    the `SecretResolver` seam at compile time (`registry.py:123-124`).
    """

    model_config = ConfigDict(extra="forbid")

    # --- secret-seam NAMES (never values; HARD house rule 1 / model-layer-spec §4) -------------
    azure_credential_ref: str = "azure_openai_api_key"
    openai_credential_ref: str = "openai_api_key"
    anthropic_credential_ref: str = "anthropic_api_key"
    deepseek_credential_ref: str = "deepseek_api_key"
    moonshot_credential_ref: str = "moonshot_api_key"

    # --- Azure: deployment names are chosen PER TENANT, so they are config, not table data -----
    azure_enabled: bool = True
    azure_api_base: str | None = None  # https://<resource>.openai.azure.com — per tenant
    azure_api_version: str | None = None  # passed through as `api_version` (never a key)
    azure_frontier_deployment: str = "gpt-5-chat"
    azure_balanced_deployment: str = "gpt-4o"
    azure_small_deployment: str = "gpt-4.1-mini"
    azure_embed_deployment: str = "text-embedding-3-large"

    # --- the OpenAI-compatible LOCAL HTTP chat endpoint (ollama's default port) ----------------
    # This is what keeps FULL-LOCAL whole: it needs no credential, so it survives
    # `active_catalog()` with zero keys, and it routes through litellm's `hosted_vllm` prefix so
    # a keyless call actually reaches the endpoint (house rule 2).
    local_http_enabled: bool = True
    local_http_api_base: str = "http://127.0.0.1:11434/v1"
    local_chat_model: str = "qwen2.5:7b-instruct"
    local_fast_model: str = "qwen2.5:3b-instruct"
    local_tiny_model: str = "qwen2.5:0.5b-instruct"

    # --- the local embed + rerank endpoint -----------------------------------------------------
    # It must speak the Cohere-shaped `/rerank` and the OpenAI-shaped `/v1/embeddings` that
    # litellm's `hosted_vllm` handlers post to (vLLM >= 0.7 and Infinity both do). NOTE: HF
    # text-embeddings-inference's native `/rerank` is a DIFFERENT wire shape (`{query, texts}`)
    # and litellm ships no adapter for it — hence this is not named "tei".
    local_embed_rerank_enabled: bool = True
    local_embed_rerank_api_base: str = "http://127.0.0.1:8080/v1"
    local_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    local_embed_model: str = "BAAI/bge-m3"

    # --- may a local model serve a group whose task PREFERS a remote model? --------------------
    # Default False. See house rule 3: `order:` cannot express "last resort", so with this off a
    # local row is placed only where it compiles to `order: 1`, and `mu-chat`'s local sibling is
    # reached through a cross-group fallback instead. Turning it ON is the posture of a box with
    # NO remote credentials at all: the local rows become PRIMARY members of `mu-chat`,
    # `mu-reason-hard` and the legacy `gpt-5-chat` group so those groups still resolve — which
    # for the adjudicating tasks is a deliberate, LOGGED deviation from ADR 0037 (whose decision
    # is to degrade to the deterministic heuristic rather than to a weaker model).
    local_serves_remote_preferred_groups: bool = False

    # --- the in-process warm SLM (L5). OFF by default: enabling it LOADS WEIGHTS at composition
    #     (`build_model_router` constructs a `WarmLocalSingleton` per entry, model_router.py:330).
    #     Absence is the house rule — while it is off, neither the provider nor its deployments
    #     nor the `warm_local` config are emitted, so no handler-less `mu-local/` prefix dangles.
    warm_local_enabled: bool = False
    warm_local_model_id: str = "mu-local/qwen2.5-0.5b-instruct"  # also the L5 singleton key
    warm_local_load_path: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # --- operator escape hatch: drop whole providers without touching the table ---------------
    disabled_provider_keys: tuple[str, ...] = Field(default_factory=tuple)
