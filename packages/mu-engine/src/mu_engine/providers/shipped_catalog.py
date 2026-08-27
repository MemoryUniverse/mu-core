"""The SHIPPED multi-provider catalog — MVP-BUILD-PLAN.md Phase 4 ("full model layer").

This module is **data, not machinery**. The many-to-many seam already exists
(`catalog.ModelDeployment.model_group`, compiled by `registry.ProviderModelRegistry` into
LiteLLM's flat `model_list`); Phase 4 only asked for the missing *table*: the plane shipped
GPT-only defaults and an EMPTY `ModelCatalogSettings.providers` / `.deployments`
(`settings.py:100-101`), so out of the box no task group resolved to anything at all.

The shape (model-layer-spec §2.1, §4):

    Task  ->  LOGICAL MODEL GROUP  ->  MANY ModelDeployments across MANY providers

`ModelGroup` names the logical tiers the `Task` taxonomy actually needs (hard reasoning /
answer / routine extraction / summarize / classify / rerank / embed). Each group is served by
several providers at once — Anthropic, OpenAI, Azure OpenAI, DeepSeek, Moonshot (Kimi), an
OpenAI-compatible LOCAL HTTP endpoint (vLLM / ollama / TGI), a local embed+rerank endpoint, and
optionally the in-process warm `mu-local` singleton (L5). LiteLLM owns health/cooldown/failover
*within* a group (`litellm_provider.py:207-217`); the local-first tier is stamped by L4
(`registry._order_for` -> `local_priority.LocalPriorityPolicy`).

Four house rules are load-bearing here:

1. **NEVER inline an API key.** Every credentialed provider carries only a
   `ProviderRecord.credential_ref` — a NAME under the secret seam (`catalog.py:77`,
   model-layer-spec §4). `extra_params` is used for `api_version` only, never for a key.
2. **FULL-LOCAL is a complete, good system — never a crippled baseline** (`CLAUDE.md`), and
   "complete" means CALLABLE, not merely present in a table. Every local row therefore uses
   LiteLLM's `hosted_vllm` OpenAI-compatible prefix, which (a) is the only local prefix whose
   chat / embedding / **rerank** handlers all exist in this litellm version, and (b) substitutes
   `"fake-api-key"` when no key is configured
   (`litellm/llms/hosted_vllm/chat/transformation.py:125`) instead of demanding one. The obvious
   alternative — declaring the local endpoint as `litellm_provider="openai"` — is WRONG twice
   over and was measured, not guessed: the OpenAI handler rejects a keyless call before opening a
   socket (`OpenAIException - Missing credentials`), and when the operator DOES have an
   `OPENAI_API_KEY` in the environment litellm silently puts that real cloud secret in the
   `Authorization` header of every localhost request (`litellm/main.py` openai branch:
   `api_key or ... or get_secret("OPENAI_API_KEY")`). `hosted_vllm` reads only
   `HOSTED_VLLM_API_KEY`, so the local seam is credential-free on the wire as well as in the
   table. LiteLLM's rerank dispatch has no `openai` branch at all
   (`litellm/rerank_api/main.py:515` -> `Unsupported provider: openai`).
3. **`order:` has only TWO tiers, so it cannot express "last resort."** LiteLLM keeps only the
   MINIMUM order tier (`litellm/utils.py:4496` "Default: pick min order group") and then
   `simple-shuffle` picks UNIFORMLY inside it. An `order: 2` local row in a group that also
   holds frontier rows is therefore a coin-flip *sibling* of Claude/GPT-4.1, not a fallback. So
   a local row is placed IN a group only where local is genuinely PREFERRED (the group's task is
   in `local_capable_tasks`, i.e. it compiles to `order: 1`). Where a group's task prefers a
   remote model, the local row lives in a SIBLING group (`mu-chat-local`) reached only through a
   LiteLLM cross-group fallback (`RouterSettings.fallbacks`) — i.e. after the primary group is
   exhausted, which is what "last resort" actually means.
   **ADR 0037 goes further for the HARD tier**: adjudication must degrade to the deterministic
   `PolarityCardinalityHeuristic` + `DegradedModeEntered`, *never* to a weaker model. So
   `mu-reason-hard` (and the legacy `gpt-5-chat` group, which the shipped `ModelSettings`
   defaults point ADJUDICATE at) gets no local row and no local fallback chain at all —
   unless the operator explicitly sets `local_serves_remote_preferred_groups`, which is logged.
4. **No invented numbers.** `max_input_tokens` is declared ONLY where the context window is
   unambiguous (a vendor-published figure, or a size named in the model id itself). Everywhere
   else the field is left `None` and `ModelRouter._max_input_tokens` falls back to
   `litellm.get_max_tokens` and then `RouterSettings.default_context_window`
   (`model_router.py:260-272`) — an omitted number, never a guessed one.

Environment-dependent values (endpoints, Azure deployment names, secret-seam names, the local
model tags) live on `ShippedCatalogSettings` — a pydantic model, the sanctioned central-config
home (DEV-STANDARDS rule 3, same pattern as `settings.py`). Vendor-fixed model ids are table
data below.

KNOWN GAP, deliberately NOT papered over: `ModelSettings.rerank_model` defaults to the
`"gpt-4.1-mini"` group (`settings.py:60`), which is a chat group — no chat deployment can serve
`Router.arerank` on ANY provider. This catalog does not smuggle a cross-encoder into a chat
group to hide that; `recommended_model_settings()` points RERANK at the real `mu-rerank`
cross-encoder group instead, and the legacy default remains the (pre-existing) placeholder it
always was.

NOTHING here changes an existing default: `ModelCatalogSettings()` and `default_local_catalog()`
are untouched (they still ship an empty catalog + the offline MiniLM embedder), and
`ModelSettings`' per-task fields still default to the `gpt-*` group names — which this table
declares as REAL groups, so those defaults keep resolving. Adopting the logical groups is one
opt-in call: `recommended_model_settings()`.
"""

from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum
from typing import Protocol

import structlog
from pydantic import BaseModel, ConfigDict, Field

from mu_engine.providers.catalog import (
    ModelDeployment,
    ModelKind,
    ProviderKind,
    ProviderRecord,
    Task,
    WarmLocalConfig,
)
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings, default_local_catalog

__all__ = [
    "CredentialProbe",
    "LegacyModelGroup",
    "ModelGroup",
    "ProviderKey",
    "ShippedCatalogSettings",
    "active_catalog",
    "group_tasks",
    "recommended_model_settings",
    "resolvable_credential_refs",
    "shipped_catalog",
    "shipped_deployments",
    "shipped_providers",
    "shipped_router_fallbacks",
    "shipped_warm_local",
]

log = structlog.get_logger("mu_engine.providers")


class ModelGroup(StrEnum):
    """The LOGICAL model groups — the `Task`-facing names, chosen on capability tier.

    A group is a *capability contract*, not a vendor model: "the strongest reasoning model this
    plane can reach", "the cheapest classifier". The catalog binds each to many providers so a
    single vendor outage, rate limit, or missing key never removes a capability.
    """

    REASON_HARD = "mu-reason-hard"  # ADJUDICATE / CONFLICT_ADJUDICATION / HARD_EXTRACT — frontier
    CHAT = "mu-chat"  # ANSWER — strong conversational, long context
    CHAT_LOCAL = "mu-chat-local"  # the LAST-RESORT local sibling of CHAT (house rule 3)
    EXTRACT_FAST = "mu-extract-fast"  # ROUTINE_EXTRACT — cheap + fast + structured output
    SUMMARIZE = "mu-summarize"  # SUMMARIZE — long context, cheap
    CLASSIFY = "mu-classify"  # CLASSIFY — the smallest/fastest model that can label
    RERANK = "mu-rerank"  # RERANK — dedicated cross-encoder rerankers (not an LLM)
    EMBED = "mu-embed"  # EMBED — dense embedders (the EmbeddingPort seam is separate, §6-P5)


class LegacyModelGroup(StrEnum):
    """The GPT-only group names `ModelSettings` still DEFAULTS to (`settings.py:53-62`).

    They are declared as real groups (Azure + OpenAI, plus a local row wherever the tasks that
    point at the group are local-capable) so the shipped catalog is drop-in for the current
    defaults. Migrating the defaults to `ModelGroup` is a separate, deliberate change — see
    `recommended_model_settings`.
    """

    GPT_5_CHAT = "gpt-5-chat"  # answer_model / adjudicate_model / hard_extract_model
    GPT_4O = "gpt-4o"  # routine_extract_model / summarize_model
    GPT_41_MINI = "gpt-4.1-mini"  # classify_model / rerank_model
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"  # embed_model


class ProviderKey(StrEnum):
    """`ProviderRecord.key` — the registry key each deployment's `provider_key` points at."""

    AZURE = "azure"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"
    LOCAL_OPENAI_HTTP = "local_openai_http"  # vLLM / ollama / TGI, OpenAI-compatible chat
    LOCAL_EMBED_RERANK_HTTP = "local_embed_rerank_http"  # local embedders + cross-encoders
    MU_LOCAL = "mu-local"  # in-process warm singleton (L5 CustomLLM prefix)


# --- the LiteLLM prefix every LOCAL row routes with (house rule 2) ----------------------------
# `hosted_vllm` is litellm's generic OpenAI-compatible LOCAL provider: chat, embeddings AND
# rerank handlers all exist for it, and its key resolution is `HOSTED_VLLM_API_KEY or
# "fake-api-key"` — it never reaches for the operator's OPENAI_API_KEY. This is protocol data
# (which litellm handler to dispatch to), not an operator knob, so it lives here, not in config.
_LOCAL_LITELLM_PREFIX = "hosted_vllm"

# --- vendor-FIXED model ids (table data; the vendor owns these strings, not the operator) ------
# Every id below is the provider-native id LiteLLM routes with, prefixed by the provider's
# `litellm_provider`. Azure ids are NOT here: an Azure deployment name is chosen per tenant, so
# it lives on `ShippedCatalogSettings` instead.
_ANTHROPIC_FRONTIER = "claude-opus-4-1"
_ANTHROPIC_BALANCED = "claude-sonnet-4-5"
_ANTHROPIC_FAST = "claude-haiku-4-5"
_OPENAI_FRONTIER = "gpt-4.1"
_OPENAI_BALANCED = "gpt-4o"
_OPENAI_FAST = "gpt-4o-mini"
_OPENAI_SMALL = "gpt-4.1-mini"
_OPENAI_EMBED = "text-embedding-3-large"
_DEEPSEEK_REASONER = "deepseek-reasoner"
_DEEPSEEK_CHAT = "deepseek-chat"
_MOONSHOT_LONG = "moonshot-v1-128k"
_MOONSHOT_MID = "moonshot-v1-32k"

# --- context windows: ONLY the ones that are unambiguous (house rule 4) ------------------------
# Anthropic Claude publishes a 200K-token window across the current model line; OpenAI's 4o line
# publishes 128K; the Moonshot ids state their own window in the id ("-32k"/"-128k"); DeepSeek
# publishes 64K. Nothing else is declared — omitted, never guessed.
_CTX_CLAUDE = 200_000
_CTX_GPT_4O = 128_000
_CTX_MOONSHOT_128K = 131_072
_CTX_MOONSHOT_32K = 32_768
_CTX_DEEPSEEK = 64_000


# --- which Task(s) each group serves ----------------------------------------------------------
# The inverse of `recommended_model_settings()` (logical groups) and of `ModelSettings`' own
# field defaults (legacy groups). It is what decides WHERE a local row may be placed: a local row
# belongs IN a group only if some task pointing at that group is local-capable, because that is
# exactly the condition under which `registry._order_for` stamps it `order: 1` (house rule 3).
_GROUP_TASKS: dict[str, frozenset[Task]] = {
    ModelGroup.REASON_HARD.value: frozenset(
        {Task.ADJUDICATE, Task.CONFLICT_ADJUDICATION, Task.HARD_EXTRACT}
    ),
    ModelGroup.CHAT.value: frozenset({Task.ANSWER}),
    ModelGroup.EXTRACT_FAST.value: frozenset({Task.ROUTINE_EXTRACT}),
    ModelGroup.SUMMARIZE.value: frozenset({Task.SUMMARIZE}),
    ModelGroup.CLASSIFY.value: frozenset({Task.CLASSIFY}),
    ModelGroup.RERANK.value: frozenset({Task.RERANK}),
    ModelGroup.EMBED.value: frozenset({Task.EMBED}),
    LegacyModelGroup.GPT_5_CHAT.value: frozenset(
        {Task.ANSWER, Task.ADJUDICATE, Task.CONFLICT_ADJUDICATION, Task.HARD_EXTRACT}
    ),
    LegacyModelGroup.GPT_4O.value: frozenset({Task.ROUTINE_EXTRACT, Task.SUMMARIZE}),
    LegacyModelGroup.GPT_41_MINI.value: frozenset({Task.CLASSIFY, Task.RERANK}),
    LegacyModelGroup.TEXT_EMBEDDING_3_LARGE.value: frozenset({Task.EMBED}),
}

# The HARD tier: ADR 0037 forbids substituting a weaker model for the adjudicator (degrade to the
# deterministic heuristic instead), so these groups get no local row and no local fallback chain.
_ADJUDICATING_TASKS: frozenset[Task] = frozenset({Task.ADJUDICATE, Task.CONFLICT_ADJUDICATION})


def group_tasks(model_group: str) -> frozenset[Task]:
    """The `Task`s the shipped defaults point at `model_group` (empty for an unknown group)."""
    return _GROUP_TASKS.get(model_group, frozenset())


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


class CredentialProbe(Protocol):
    """Structural view of `registry.SecretResolver` — just enough to ASK whether a ref resolves.

    A `Protocol` (not `registry.SecretResolver`, which is a nominal class) so any composition
    root's resolver satisfies it structurally without importing the model layer's base class.
    """

    def resolve(self, credential_ref: str) -> str: ...


# ---------------------------------------------------------------------------------------------
# The declared catalog
# ---------------------------------------------------------------------------------------------
def shipped_providers(cfg: ShippedCatalogSettings | None = None) -> list[ProviderRecord]:
    """Every provider the shipped catalog can reach, credentials named but NEVER inlined.

    A provider is emitted only if its feature flag is on and its key is not in
    `disabled_provider_keys` — absence, not a disabled stub (house rule 5).
    """
    c = cfg if cfg is not None else ShippedCatalogSettings()
    disabled = frozenset(c.disabled_provider_keys)
    records: list[ProviderRecord] = [
        ProviderRecord(
            key=ProviderKey.OPENAI,
            kind=ProviderKind.REMOTE,
            litellm_provider="openai",
            credential_ref=c.openai_credential_ref,
        ),
        ProviderRecord(
            key=ProviderKey.ANTHROPIC,
            kind=ProviderKind.REMOTE,
            litellm_provider="anthropic",
            credential_ref=c.anthropic_credential_ref,
        ),
        ProviderRecord(
            key=ProviderKey.DEEPSEEK,
            kind=ProviderKind.REMOTE,
            litellm_provider="deepseek",
            credential_ref=c.deepseek_credential_ref,
        ),
        ProviderRecord(
            key=ProviderKey.MOONSHOT,
            kind=ProviderKind.REMOTE,
            litellm_provider="moonshot",
            credential_ref=c.moonshot_credential_ref,
        ),
    ]
    if c.azure_enabled:
        records.append(
            ProviderRecord(
                key=ProviderKey.AZURE,
                kind=ProviderKind.REMOTE,
                litellm_provider="azure",
                api_base=c.azure_api_base,
                credential_ref=c.azure_credential_ref,
            )
        )
    if c.local_http_enabled:
        records.append(
            ProviderRecord(
                key=ProviderKey.LOCAL_OPENAI_HTTP,
                kind=ProviderKind.LOCAL_HTTP,
                litellm_provider=_LOCAL_LITELLM_PREFIX,  # keyless-capable local handler (rule 2)
                api_base=c.local_http_api_base,
                credential_ref=None,  # localhost — no secret, so it survives with zero keys
                is_local=True,
            )
        )
    if c.local_embed_rerank_enabled:
        records.append(
            ProviderRecord(
                key=ProviderKey.LOCAL_EMBED_RERANK_HTTP,
                kind=ProviderKind.LOCAL_HTTP,
                litellm_provider=_LOCAL_LITELLM_PREFIX,  # the ONLY local rerank handler litellm has
                api_base=c.local_embed_rerank_api_base,
                credential_ref=None,
                is_local=True,
            )
        )
    if c.warm_local_enabled:
        records.append(
            ProviderRecord(
                key=ProviderKey.MU_LOCAL,
                kind=ProviderKind.LOCAL_INPROC,
                litellm_provider=c.warm_local_model_id.split("/", 1)[0],
                credential_ref=None,
                is_local=True,
            )
        )
    return [r for r in records if r.key not in disabled]


def shipped_warm_local(cfg: ShippedCatalogSettings | None = None) -> list[WarmLocalConfig]:
    """The L5 in-process singletons — empty unless `warm_local_enabled`, because each entry
    loads real weights inside `build_model_router` (model_router.py:328-335)."""
    c = cfg if cfg is not None else ShippedCatalogSettings()
    if not c.warm_local_enabled or ProviderKey.MU_LOCAL in frozenset(c.disabled_provider_keys):
        return []
    return [
        WarmLocalConfig(
            model_id=c.warm_local_model_id,
            kind=ModelKind.LLM,
            model_load_path=c.warm_local_load_path,
        )
    ]


def _local_capable(group: str, local_capable_tasks: Collection[Task]) -> bool:
    """True iff SOME task pointing at `group` may prefer a local model — the exact condition
    under which `registry._order_for` stamps a local row in that group `order: 1` (rule 3)."""
    wanted = frozenset(local_capable_tasks)
    return bool(_GROUP_TASKS.get(group, frozenset()) & wanted)


def shipped_deployments(
    cfg: ShippedCatalogSettings | None = None,
    *,
    local_capable_tasks: Collection[Task] | None = None,
) -> list[ModelDeployment]:
    """The many-to-many table: LOGICAL GROUP x PROVIDER.

    Ordering within this list is irrelevant to routing — LiteLLM routes by the `order:` tier the
    L1 compiler stamps from L4 (`registry._order_for`), not by list position. The rows are
    grouped by `model_group` purely for readability.

    `local_capable_tasks` is the SAME list the composition root puts on
    `ModelCatalogSettings.local_capable_tasks` (`shipped_catalog` passes its base's). It decides
    WHERE a local row may sit (house rule 3): in-group where it compiles to `order: 1`, in the
    `mu-chat-local` sibling otherwise. It defaults to `ModelCatalogSettings()`'s own default set.

    Rows whose provider was not emitted by `shipped_providers` are filtered out at the end, so a
    disabled provider leaves no dangling `provider_key` (which `registry._validate` would
    otherwise reject as an unknown provider).
    """
    c = cfg if cfg is not None else ShippedCatalogSettings()
    capable = (
        local_capable_tasks
        if local_capable_tasks is not None
        else ModelCatalogSettings().local_capable_tasks
    )
    azure_extra = {"api_version": c.azure_api_version} if c.azure_api_version else {}

    def azure(group: str, deployment: str, *, kind: ModelKind = ModelKind.LLM) -> ModelDeployment:
        return ModelDeployment(
            model_group=group,
            provider_key=ProviderKey.AZURE,
            model_id=f"azure/{deployment}",
            kind=kind,
            extra_params=dict(azure_extra),  # api_version ONLY — never a key
        )

    def local_llm(group: str, model: str) -> ModelDeployment:
        return ModelDeployment(
            model_group=group,
            provider_key=ProviderKey.LOCAL_OPENAI_HTTP,
            model_id=f"{_LOCAL_LITELLM_PREFIX}/{model}",
        )

    def local_embed_rerank(group: str, model: str, *, kind: ModelKind) -> ModelDeployment:
        return ModelDeployment(
            model_group=group,
            provider_key=ProviderKey.LOCAL_EMBED_RERANK_HTTP,
            model_id=f"{_LOCAL_LITELLM_PREFIX}/{model}",
            kind=kind,
        )

    rows: list[ModelDeployment] = [
        # ---- mu-reason-hard: ADJUDICATE / CONFLICT_ADJUDICATION / HARD_EXTRACT ----------------
        # Frontier tier, REMOTE-ONLY by decision (ADR 0037: when the HARD model is unreachable
        # the adjudicator degrades to the deterministic PolarityCardinalityHeuristic, it does NOT
        # substitute a weaker model). A local row here would not be a "fallback": it would share
        # `order: 2` with every frontier row and win a uniform shuffle ~1 call in N.
        ModelDeployment(
            model_group=ModelGroup.REASON_HARD,
            provider_key=ProviderKey.ANTHROPIC,
            model_id=f"anthropic/{_ANTHROPIC_FRONTIER}",
            max_input_tokens=_CTX_CLAUDE,
        ),
        ModelDeployment(
            model_group=ModelGroup.REASON_HARD,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_FRONTIER}",
        ),
        ModelDeployment(
            model_group=ModelGroup.REASON_HARD,
            provider_key=ProviderKey.DEEPSEEK,
            model_id=f"deepseek/{_DEEPSEEK_REASONER}",
            max_input_tokens=_CTX_DEEPSEEK,
        ),
        azure(ModelGroup.REASON_HARD, c.azure_frontier_deployment),
        # ---- mu-chat: ANSWER -----------------------------------------------------------------
        ModelDeployment(
            model_group=ModelGroup.CHAT,
            provider_key=ProviderKey.ANTHROPIC,
            model_id=f"anthropic/{_ANTHROPIC_BALANCED}",
            max_input_tokens=_CTX_CLAUDE,
        ),
        ModelDeployment(
            model_group=ModelGroup.CHAT,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_BALANCED}",
            max_input_tokens=_CTX_GPT_4O,
        ),
        ModelDeployment(
            model_group=ModelGroup.CHAT,
            provider_key=ProviderKey.DEEPSEEK,
            model_id=f"deepseek/{_DEEPSEEK_CHAT}",
            max_input_tokens=_CTX_DEEPSEEK,
        ),
        ModelDeployment(
            model_group=ModelGroup.CHAT,
            provider_key=ProviderKey.MOONSHOT,
            model_id=f"moonshot/{_MOONSHOT_LONG}",
            max_input_tokens=_CTX_MOONSHOT_128K,
        ),
        azure(ModelGroup.CHAT, c.azure_frontier_deployment),
        # ---- mu-extract-fast: ROUTINE_EXTRACT (local-capable -> local row gets order 1) -------
        ModelDeployment(
            model_group=ModelGroup.EXTRACT_FAST,
            provider_key=ProviderKey.ANTHROPIC,
            model_id=f"anthropic/{_ANTHROPIC_FAST}",
            max_input_tokens=_CTX_CLAUDE,
        ),
        ModelDeployment(
            model_group=ModelGroup.EXTRACT_FAST,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_FAST}",
            max_input_tokens=_CTX_GPT_4O,
        ),
        ModelDeployment(
            model_group=ModelGroup.EXTRACT_FAST,
            provider_key=ProviderKey.DEEPSEEK,
            model_id=f"deepseek/{_DEEPSEEK_CHAT}",
            max_input_tokens=_CTX_DEEPSEEK,
        ),
        azure(ModelGroup.EXTRACT_FAST, c.azure_small_deployment),
        # ---- mu-summarize: SUMMARIZE (long context, cheap; local-capable) --------------------
        ModelDeployment(
            model_group=ModelGroup.SUMMARIZE,
            provider_key=ProviderKey.ANTHROPIC,
            model_id=f"anthropic/{_ANTHROPIC_FAST}",
            max_input_tokens=_CTX_CLAUDE,
        ),
        ModelDeployment(
            model_group=ModelGroup.SUMMARIZE,
            provider_key=ProviderKey.MOONSHOT,
            model_id=f"moonshot/{_MOONSHOT_MID}",
            max_input_tokens=_CTX_MOONSHOT_32K,
        ),
        ModelDeployment(
            model_group=ModelGroup.SUMMARIZE,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_FAST}",
            max_input_tokens=_CTX_GPT_4O,
        ),
        azure(ModelGroup.SUMMARIZE, c.azure_balanced_deployment),
        # ---- mu-classify: CLASSIFY (smallest + fastest; local-capable) ------------------------
        ModelDeployment(
            model_group=ModelGroup.CLASSIFY,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_FAST}",
            max_input_tokens=_CTX_GPT_4O,
        ),
        ModelDeployment(
            model_group=ModelGroup.CLASSIFY,
            provider_key=ProviderKey.ANTHROPIC,
            model_id=f"anthropic/{_ANTHROPIC_FAST}",
            max_input_tokens=_CTX_CLAUDE,
        ),
        azure(ModelGroup.CLASSIFY, c.azure_small_deployment),
        # ---- mu-rerank: RERANK — a dedicated cross-encoder, NOT an LLM (settings.py:60 §CC-4).
        #      No cloud provider in this catalog serves a reranker, so this group is local-only;
        #      when the endpoint is down the recall RerankGate degrades (model-layer-spec §5).
        local_embed_rerank(ModelGroup.RERANK, c.local_rerank_model, kind=ModelKind.RERANK),
        # ---- mu-embed: EMBED — declared for completeness; the PRIMARY embed path is the
        #      dedicated EmbeddingPort seam (`catalog.embedders`, §6-P5), which is why
        #      `registry._validate` skips Task.EMBED entirely (registry.py:83-84).
        local_embed_rerank(ModelGroup.EMBED, c.local_embed_model, kind=ModelKind.EMBED),
        ModelDeployment(
            model_group=ModelGroup.EMBED,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_EMBED}",
            kind=ModelKind.EMBED,
        ),
        azure(ModelGroup.EMBED, c.azure_embed_deployment, kind=ModelKind.EMBED),
        # ---- legacy compatibility groups: the names `ModelSettings` still defaults to ---------
        # Real groups (not aliases): the literal vendor model, plus a local row wherever the
        # tasks pointing at the group are local-capable (rule 3). `gpt-5-chat` also serves
        # ADJUDICATE under the shipped defaults, so it follows `mu-reason-hard`: remote-only.
        azure(LegacyModelGroup.GPT_5_CHAT, c.azure_frontier_deployment),
        ModelDeployment(
            model_group=LegacyModelGroup.GPT_5_CHAT,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_FRONTIER}",
        ),
        azure(LegacyModelGroup.GPT_4O, c.azure_balanced_deployment),
        ModelDeployment(
            model_group=LegacyModelGroup.GPT_4O,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_BALANCED}",
            max_input_tokens=_CTX_GPT_4O,
        ),
        azure(LegacyModelGroup.GPT_41_MINI, c.azure_small_deployment),
        ModelDeployment(
            model_group=LegacyModelGroup.GPT_41_MINI,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_SMALL}",
        ),
        azure(
            LegacyModelGroup.TEXT_EMBEDDING_3_LARGE, c.azure_embed_deployment, kind=ModelKind.EMBED
        ),
        ModelDeployment(
            model_group=LegacyModelGroup.TEXT_EMBEDDING_3_LARGE,
            provider_key=ProviderKey.OPENAI,
            model_id=f"openai/{_OPENAI_EMBED}",
            kind=ModelKind.EMBED,
        ),
        local_embed_rerank(
            LegacyModelGroup.TEXT_EMBEDDING_3_LARGE, c.local_embed_model, kind=ModelKind.EMBED
        ),
    ]

    # ---- LOCAL chat rows: placed where they are genuinely PREFERRED (house rule 3) ------------
    local_chat_rows: list[tuple[str, str]] = [
        (ModelGroup.EXTRACT_FAST.value, c.local_fast_model),
        (ModelGroup.SUMMARIZE.value, c.local_chat_model),
        (ModelGroup.CLASSIFY.value, c.local_tiny_model),
        (LegacyModelGroup.GPT_4O.value, c.local_fast_model),
        (LegacyModelGroup.GPT_41_MINI.value, c.local_tiny_model),
    ]
    rows.extend(
        local_llm(group, model)
        for group, model in local_chat_rows
        if _local_capable(group, capable)
    )

    # ---- the REMOTE-preferred groups' local rows ----------------------------------------------
    remote_preferred = [
        (ModelGroup.CHAT.value, c.local_chat_model),
        (ModelGroup.REASON_HARD.value, c.local_chat_model),
        (LegacyModelGroup.GPT_5_CHAT.value, c.local_chat_model),
    ]
    if c.local_serves_remote_preferred_groups:
        # The no-remote-credentials posture: the local rows become PRIMARY so these groups still
        # resolve. Named + content-free, because for the adjudicating groups this is a deliberate
        # deviation from ADR 0037 and must never be silent.
        adjudicating = sorted(
            g for g, _ in remote_preferred if group_tasks(g) & _ADJUDICATING_TASKS
        )
        log.info(
            "model_catalog_local_serves_remote_preferred_groups",
            groups=sorted(g for g, _ in remote_preferred),
            adjudicating_groups=adjudicating,  # ADR 0037 deviation, opted into by config
        )
        rows.extend(local_llm(group, model) for group, model in remote_preferred)
    else:
        # `mu-chat` keeps a TRUE last-resort local sibling: a separate group no task points at,
        # reached only through the cross-group fallback chain in `shipped_router_fallbacks`.
        # `mu-reason-hard` / `gpt-5-chat` get no such sibling — ADR 0037 degrades to the
        # deterministic heuristic instead of to a weaker model.
        rows.append(local_llm(ModelGroup.CHAT_LOCAL.value, c.local_chat_model))

    if c.warm_local_enabled and ProviderKey.MU_LOCAL not in frozenset(c.disabled_provider_keys):
        # The warm in-process SLM serves the small/fast tiers it can actually do well — and only
        # where those tiers are local-capable, same rule as every other local row (rule 3).
        rows.extend(
            ModelDeployment(
                model_group=group,
                provider_key=ProviderKey.MU_LOCAL,
                model_id=c.warm_local_model_id,
            )
            for group in (ModelGroup.CLASSIFY.value, ModelGroup.EXTRACT_FAST.value)
            if _local_capable(group, capable)
        )

    known = {p.key for p in shipped_providers(c)}
    return [row for row in rows if row.provider_key in known]


def shipped_router_fallbacks(
    cfg: ShippedCatalogSettings | None = None,
) -> list[dict[str, list[str]]]:
    """The LiteLLM cross-group fallback chains (`RouterSettings.fallbacks`, settings.py:81).

    This is the ONLY mechanism in the stack that expresses "last resort": a chain is followed
    after the primary group is exhausted, whereas `order:` only partitions ONE group into tiers
    and litellm then shuffles uniformly inside the surviving tier (house rule 3).

    Exactly one chain is shipped: `mu-chat` -> `mu-chat-local`, so ANSWER survives a total
    remote outage without ever competing with a frontier model on a healthy box. Deliberately
    NO chain for `mu-reason-hard` / `gpt-5-chat` — ADR 0037 requires adjudication to degrade to
    the deterministic heuristic, not to a weaker model.
    """
    c = cfg if cfg is not None else ShippedCatalogSettings()
    if c.local_serves_remote_preferred_groups or not c.local_http_enabled:
        return []  # the local rows are primary (or absent) — nothing to fall back TO
    if ProviderKey.LOCAL_OPENAI_HTTP in frozenset(c.disabled_provider_keys):
        return []
    return [{ModelGroup.CHAT.value: [ModelGroup.CHAT_LOCAL.value]}]


def shipped_catalog(
    base: ModelCatalogSettings | None = None,
    *,
    cfg: ShippedCatalogSettings | None = None,
) -> ModelCatalogSettings:
    """The DECLARED catalog: `base` with the shipped providers/deployments/warm-local filled in.

    `base` defaults to `default_local_catalog()` so the returned catalog already carries the
    offline MiniLM embedder as the active `embed_backend` — i.e. the result is immediately
    usable with zero API keys. A composition root should pass its WIRED
    `default_local_catalog(get_engine_settings().model_catalog)` instead, so every env-overridable
    subtree (`router`, `local_capable_tasks`, `local_priority_enabled`, …) flows through.

    The base's `local_capable_tasks` decides where local rows are placed (house rule 3), and the
    shipped fallback chains are APPENDED to `base.router.fallbacks` — an operator chain already
    keyed on a group always wins, so nothing configured is overwritten.

    Returns a FRESH object (`model_copy`) — `base` is never mutated.

    This is the DECLARED catalog, not the ACTIVE one: it still contains providers whose
    credentials may be absent. Pass it through `active_catalog` before building the router.
    """
    catalog = base if base is not None else default_local_catalog()
    configured = {key for chain in catalog.router.fallbacks for key in chain}
    chains = [c for c in shipped_router_fallbacks(cfg) if not (set(c) & configured)]
    return catalog.model_copy(
        update={
            "providers": shipped_providers(cfg),
            "deployments": shipped_deployments(
                cfg, local_capable_tasks=catalog.local_capable_tasks
            ),
            "warm_local": shipped_warm_local(cfg),
            "router": catalog.router.model_copy(
                update={"fallbacks": [*catalog.router.fallbacks, *chains]}
            ),
        }
    )


# ---------------------------------------------------------------------------------------------
# Credential-aware activation
# ---------------------------------------------------------------------------------------------
def active_catalog(
    catalog: ModelCatalogSettings,
    *,
    available_credentials: Collection[str],
) -> ModelCatalogSettings:
    """Narrow a DECLARED catalog to the providers whose credentials are actually available.

    `available_credentials` is a collection of `credential_ref` NAMES that resolve — never the
    secret values, which do not enter this module at all (house rule 1). Build it with
    `resolvable_credential_refs`.

    Rules:
      * a provider with `credential_ref is None` (every local / in-process one) ALWAYS survives —
        that is what keeps FULL-LOCAL a complete system with zero API keys;
      * a credentialed provider survives iff its ref is in `available_credentials`;
      * a dropped provider takes its deployments with it, so no row is left pointing at an
        unknown `provider_key` (which `registry._validate` rejects).

    A group that credential absence emptied ADOPTS its DECLARED fallback chain
    (`router.fallbacks`, written by `shipped_router_fallbacks`): the chain is precisely the
    catalog's declared answer to "this group is unavailable", so honouring it here is following
    declared data, not inventing a route. Only a group the catalog already declared can adopt
    one, only from a chain target that still has rows, and the promotion is a named event. This
    is what lets a box with ZERO keys still answer: `mu-chat` adopts `mu-chat-local`. It is also
    why `mu-reason-hard` / `gpt-5-chat` deliberately have NO chain — ADR 0037 wants adjudication
    to degrade to the deterministic heuristic, so those groups stay empty and fail loud.

    Exclusion is neither a hard failure nor a silent success: it is named, content-free
    observability — `model_catalog_provider_excluded` for the providers that went,
    `model_catalog_fallback_promoted` for each adopted chain, and `model_catalog_group_emptied`
    for any group still left with nothing. An emptied group that a task points at makes
    `ProviderModelRegistry` fail loud at composition, as designed (model-layer-spec §5); this
    function never papers over that, it names it first.

    Returns a FRESH object; `catalog` is never mutated.
    """
    available = frozenset(available_credentials)
    kept: list[ProviderRecord] = []
    dropped: list[ProviderRecord] = []
    for provider in catalog.providers:
        if provider.credential_ref is None or provider.credential_ref in available:
            kept.append(provider)
        else:
            dropped.append(provider)

    dropped_keys = {p.key for p in dropped}
    deployments = [d for d in catalog.deployments if d.provider_key not in dropped_keys]
    if dropped:
        log.info(
            "model_catalog_provider_excluded",  # content-free: names + counts, never a value
            providers=sorted(dropped_keys),
            missing_credential_refs=sorted(
                p.credential_ref for p in dropped if p.credential_ref is not None
            ),
            deployments_dropped=len(catalog.deployments) - len(deployments),
            providers_active=len(kept),
        )
    deployments = [*deployments, *_promoted_fallback_rows(catalog, deployments)]
    emptied = sorted(
        {d.model_group for d in catalog.deployments} - {d.model_group for d in deployments}
    )
    if emptied:
        log.warning(
            "model_catalog_group_emptied",  # a task pointing here will fail loud at composition
            groups=emptied,
        )
    return catalog.model_copy(update={"providers": kept, "deployments": deployments})


def _promoted_fallback_rows(
    catalog: ModelCatalogSettings, surviving: list[ModelDeployment]
) -> list[ModelDeployment]:
    """Rows a DECLARED fallback chain contributes to a group activation emptied (see
    `active_catalog`). Never invents a group, never overrides a group that still has rows."""
    declared_groups = {d.model_group for d in catalog.deployments}
    by_group: dict[str, list[ModelDeployment]] = {}
    for dep in surviving:
        by_group.setdefault(dep.model_group, []).append(dep)

    promoted: list[ModelDeployment] = []
    for chain in catalog.router.fallbacks:
        for group, alternatives in chain.items():
            if group not in declared_groups or by_group.get(group):
                continue
            for alternative in alternatives:
                rows = by_group.get(alternative)
                if not rows:
                    continue
                promoted.extend(r.model_copy(update={"model_group": group}) for r in rows)
                log.info(
                    "model_catalog_fallback_promoted",  # content-free: group names + a count
                    group=group,
                    source_group=alternative,
                    rows=len(rows),
                )
                break
    return promoted


def resolvable_credential_refs(
    catalog: ModelCatalogSettings,
    resolver: CredentialProbe,
) -> frozenset[str]:
    """PROBE the secret seam: which of the catalog's `credential_ref`s actually resolve?

    The seam (`registry.SecretResolver`) can only `resolve()`, and a missing secret is signalled
    by raising — the concrete exception type is the resolver's business (`RegistryError`,
    `KeyError`, `FileNotFoundError`, …). So absence is detected by attempting the lookup. This
    is a deliberate, NAMED probe, not a swallowed error: every miss is logged with the ref name
    and the exception TYPE only (never its message, which can carry a path or a value), and the
    caller sees the miss as an excluded provider in `active_catalog`.

    `BaseException` (so `KeyboardInterrupt`/`CancelledError`) is never caught.
    """
    refs = sorted({p.credential_ref for p in catalog.providers if p.credential_ref is not None})
    present: set[str] = set()
    for ref in refs:
        try:
            value = resolver.resolve(ref)
        except Exception as exc:
            log.info(
                "model_catalog_credential_absent", credential_ref=ref, cause=type(exc).__name__
            )
            continue
        if value:
            present.add(ref)
        else:
            log.info("model_catalog_credential_absent", credential_ref=ref, cause="empty")
    return frozenset(present)


# ---------------------------------------------------------------------------------------------
# Task -> group mapping (the user-configurable seam stays `ModelSettings`)
# ---------------------------------------------------------------------------------------------
def recommended_model_settings(base: ModelSettings | None = None) -> ModelSettings:
    """Point every per-task field at a LOGICAL `ModelGroup` instead of a vendor model name.

    The seam is unchanged — `ModelSettings`' per-task fields (CANONICAL §7.2) are still the ONE
    place a task's model is configured, still env-overridable via `MU_MODEL__*`, and still
    per-call overridable via `TaskClassMapper.group_for(override=…)`. This helper only supplies
    a principled DEFAULT SET for them:

      ANSWER                                  -> mu-chat          (capability + context)
      ADJUDICATE / CONFLICT_ADJUDICATION      -> mu-reason-hard   (frontier; ADR 0037)
      HARD_EXTRACT                            -> mu-reason-hard
      ROUTINE_EXTRACT                         -> mu-extract-fast  (cost + latency, local-first)
      SUMMARIZE                               -> mu-summarize     (context + cost, local-first)
      CLASSIFY                                -> mu-classify      (latency, local-first)
      RERANK                                  -> mu-rerank        (dedicated cross-encoder)
      EMBED (`embed_model`)                   -> mu-embed

    This is also the ONE fix for the pre-existing `rerank_model="gpt-4.1-mini"` default
    (`settings.py:60`): a chat group cannot serve `Router.arerank` on any provider, and
    `mu-rerank` is a real cross-encoder group.

    `embed_backend` is deliberately NOT touched: it selects the dedicated `EmbeddingPort`
    adapter (`minilm_local`), which is a different seam from a routed deployment (§6-P5).

    Returns a FRESH object; `base` is never mutated.
    """
    models = base if base is not None else ModelSettings()
    return models.model_copy(
        update={
            "answer_model": ModelGroup.CHAT.value,
            "adjudicate_model": ModelGroup.REASON_HARD.value,
            "hard_extract_model": ModelGroup.REASON_HARD.value,
            "routine_extract_model": ModelGroup.EXTRACT_FAST.value,
            "summarize_model": ModelGroup.SUMMARIZE.value,
            "classify_model": ModelGroup.CLASSIFY.value,
            "rerank_model": ModelGroup.RERANK.value,
            "embed_model": ModelGroup.EMBED.value,
        }
    )
