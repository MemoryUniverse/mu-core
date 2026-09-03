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

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mu_engine.providers.catalog import (
    HttpEmbedConfig,
    ModelDeployment,
    ModelKind,
    ProviderRecord,
    Task,
    WarmLocalConfig,
)
from mu_engine.providers.shipped_settings import ShippedCatalogSettings

__all__ = [
    "CatalogSource",
    "LocalFallbackPosture",
    "ModelCatalogSettings",
    "ModelSettings",
    "RouterSettings",
    "TaskDefaults",
    "default_local_catalog",
]


class CatalogSource(StrEnum):
    """WHICH table `ModelCatalogSettings` resolves to at composition (MVP-SPEC §8-Q8).

    Q8 asked whether the ~915-line multi-provider table is opt-in or the default and recorded
    that *"today it is neither — it is a table with no caller, which is how design starts to
    depreciate."* This enum is the answer, written down: **SHIPPED is the default**, so a plane
    built from bare settings resolves every task out of the box (gate G6), and EMPTY is the
    named, one-env-var way back to the pre-2026-08-28 posture.
    """

    SHIPPED = "shipped"  # DEFAULT — `shipped_catalog()` narrowed by the credential probe
    EMPTY = "empty"  # providers/deployments exactly as configured (the pre-wiring posture)


class TaskDefaults(StrEnum):
    """WHERE the per-task fields of `ModelSettings` point when the operator did not set them.

    `ModelSettings` stays the ONE user-configurable seam (CANONICAL §7.2): a field the operator
    set — in code or via `MU_MODEL__ANSWER_MODEL` — is NEVER overwritten, whichever value this
    carries (`ModelSettings.model_fields_set` is the discriminator).
    """

    RECOMMENDED = "recommended"  # DEFAULT — `recommended_model_settings()`'s logical ModelGroups
    LEGACY = "legacy"  # `ModelSettings`' own `gpt-*` field defaults, untouched


class LocalFallbackPosture(StrEnum):
    """Whether the local rows may serve the REMOTE-PREFERRED groups
    (`ShippedCatalogSettings.local_serves_remote_preferred_groups`).

    That flag is the shipped catalog's OWN answer to a box with no cloud keys: with it off, a
    keyless plane's `mu-reason-hard` / `gpt-5-chat` groups are empty and `ProviderModelRegistry`
    refuses to start (model-layer-spec §5, and the behaviour
    `test_shipped_catalog_unit.py::test_without_the_flag_a_keyless_box_fails_loud_on_the_hard_tier`
    pins). A plane that must boot with ZERO credentials therefore has to make a choice, and
    leaving the choice implicit is how it got made by accident.

    AUTO makes it explicit and evidence-driven: adopt the no-remote-credentials posture **iff the
    credential probe found no remote credential at all**, and say so in a named, content-free
    event. For the adjudicating groups that is the deliberate ADR 0037 deviation the flag's own
    docstring describes — logged, never silent, and `NEVER` keeps the fail-loud original.
    """

    AUTO = "auto"  # DEFAULT — adopt it only when zero remote credentials resolved
    NEVER = "never"  # keep the shipped default: a keyless box fails loud on the hard tier
    ALWAYS = "always"  # local rows are primary even where remote credentials exist


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
    # Fallback token ceiling when neither the deployment catalog nor litellm knows a model
    # group's context window (``ModelRouter.max_input_tokens``, model-layer-spec §2.7) — a
    # conservative modern default, never a bare literal in the router's logic (rule 3).
    default_context_window: int = 128_000


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
    #: embed_backend -> config. A `WarmLocalConfig` value resolves through the in-process
    #: `SentenceTransformerEmbedder`; an `HttpEmbedConfig` value resolves through `HttpEmbedder`
    #: (an HTTP-reachable embed endpoint, e.g. the VM-hosted `all-minilm` service) — see
    #: `embedding.build_embedder`'s dispatch. Both keep the SAME `EmbeddingPort` seam; which one
    #: activates for a given key is a catalog-content question, not a code branch.
    embedders: dict[str, WarmLocalConfig | HttpEmbedConfig] = Field(default_factory=dict)
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
    # CONFIG-AND-DATA-FIX-PLAN.md §1.1 Group B: the offline-embedder default was a bare module
    # const (`_DEFAULT_EMBED_BACKEND`/`_DEFAULT_MINILM_PATH` above), read only by
    # `default_local_catalog()`'s own body — never reachable from the environment. Promoted here
    # so `MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND`/`MU_MODEL_CATALOG__DEFAULT_MINILM_PATH` (via
    # `EngineSettings.model_catalog`, C0) actually reach `default_local_catalog()` when a
    # composition root passes its WIRED `ModelCatalogSettings` in (see that function's docstring).
    default_embed_backend: str = _DEFAULT_EMBED_BACKEND
    default_minilm_path: str = _DEFAULT_MINILM_PATH

    # --- optional HTTP embed backend (owner's ask: embedding on the VPS, not the laptop) -------
    # ADDITIVE to the in-process entry above, never a replacement of it: `default_local_catalog`
    # registers this SECOND `embedders` entry (an `HttpEmbedConfig`, keyed by
    # `http_embed_backend_key`) only when `http_embed_api_base` is set — a keyless/unset box gets
    # byte-identical behaviour to before this field existed. Which key actually SERVES a given
    # composition (`ModelSettings.embed_backend` / `mu_local.config.StorageSettings.embedding.
    # backend`) is a separate selection, left at its own default — so the offline MiniLM singleton
    # stays the code-level default (CLAUDE.md boundary rule: FULL-LOCAL must work with no VM).
    # Every field is `MU_MODEL_CATALOG__HTTP_EMBED_*` (nested delimiter `__`, same subtree).
    http_embed_backend_key: str = "minilm_vm_http"  # the embedders/embed_backend registry key
    http_embed_api_base: str | None = None  # e.g. "http://127.0.0.1:11435/v1"; None = not offered
    http_embed_model: str = "all-minilm"  # Ollama's GGUF conversion of all-MiniLM-L6-v2 — 384-dim
    http_embed_dimension: int = 384  # MUST match the live Qdrant collections' `__384` suffix
    http_embed_timeout_s: float = 10.0
    http_embed_batch_size: int = 64

    # --- ENG-115a: the wiring knobs. Everything below is reachable as `MU_MODEL_CATALOG__*`
    #     because `EngineSettings.model_catalog` mounts THIS class (config/engine_settings.py:99).
    #: WHICH table to resolve (see :class:`CatalogSource`). SHIPPED is the default — MVP-SPEC
    #: §8-Q8 answer (a). `providers`/`deployments` above stay the operator's own explicit rows and
    #: are OVERLAID on the shipped table, never replaced by it.
    source: CatalogSource = CatalogSource.SHIPPED
    #: The environment-dependent knobs of the shipped table (endpoints, Azure deployment names,
    #: secret-seam NAMES). `MU_MODEL_CATALOG__SHIPPED__LOCAL_HTTP_API_BASE=...` etc.
    shipped: ShippedCatalogSettings = Field(default_factory=ShippedCatalogSettings)
    #: Where UNSET per-task fields point (see :class:`TaskDefaults`).
    task_defaults: TaskDefaults = TaskDefaults.RECOMMENDED
    #: The keyless-box posture (see :class:`LocalFallbackPosture`).
    local_fallback: LocalFallbackPosture = LocalFallbackPosture.AUTO
    #: Root of the secret seam a `credential_ref` NAME is looked up under — one file per ref
    #: (`/run/secrets/<ref>`, model-layer-spec §4). `None` ⇒ the environment alone is the seam.
    #: A VALUE never appears in this tree; only the directory that holds them.
    secrets_dir: str | None = None
    #: Also consult the environment for a `credential_ref` (upper-cased) when no secret file
    #: exists. This is how `ANTHROPIC_API_KEY` in the operator's shell activates the anthropic
    #: provider with no config at all — the ordinary way every one of these vendors is configured.
    credentials_from_env: bool = True


def default_local_catalog(catalog: ModelCatalogSettings | None = None) -> ModelCatalogSettings:
    """The LocalContainer default: an offline MiniLM embedder as the active embed backend, no
    cloud providers (the user adds them). This is the plane default described in §4 / §6 — the
    daemon ships the local embedder; THIN clients ship an empty catalog and route to server.

    CONFIG-AND-DATA-FIX-PLAN.md §1.2 C2: ``catalog`` is an optional BASE — when a composition
    root passes its WIRED ``get_engine_settings().model_catalog`` (C0/C1), every OTHER subtree on
    it (``router``, ``providers``, ``deployments``, ``warm_local``, ``local_capable_tasks``,
    ``local_priority_enabled``) is env-overridable (``MU_MODEL_CATALOG__ROUTER__…``, etc.) and
    flows through UNCHANGED; only ``embedders`` is derived here, keyed by
    ``catalog.default_embed_backend``/``catalog.default_minilm_path`` (also env-overridable, same
    subtree) instead of a bare module constant. ``catalog=None`` (every call site before this
    change) reproduces the EXACT prior behavior — ``ModelCatalogSettings()``'s own bare defaults —
    so this is a no-drift, backward-compatible signature widening.

    NOTE this always returns a FRESH object (``model_copy``, never mutates ``catalog``) so each
    call yields an independent catalog — no shared mutable default.
    """
    base = catalog if catalog is not None else ModelCatalogSettings()
    backend = base.default_embed_backend
    embedders: dict[str, WarmLocalConfig | HttpEmbedConfig] = {
        backend: WarmLocalConfig(
            model_id=backend,
            kind=ModelKind.EMBED,
            model_load_path=base.default_minilm_path,
            normalize_embeddings=True,
        )
    }
    # ADDITIVE, opt-in: only registered when the operator actually named an endpoint
    # (`MU_MODEL_CATALOG__HTTP_EMBED_API_BASE`). An unset box gets exactly the dict above —
    # byte-identical to before this backend existed (CLAUDE.md: in-process stays the default).
    if base.http_embed_api_base is not None:
        embedders[base.http_embed_backend_key] = HttpEmbedConfig(
            model_id=base.http_embed_backend_key,
            kind=ModelKind.EMBED,
            api_base=base.http_embed_api_base,
            model=base.http_embed_model,
            dimension=base.http_embed_dimension,
            timeout_s=base.http_embed_timeout_s,
            batch_size=base.http_embed_batch_size,
        )
    return base.model_copy(update={"embedders": embedders})
