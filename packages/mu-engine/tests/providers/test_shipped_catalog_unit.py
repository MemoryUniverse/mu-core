"""The SHIPPED multi-provider catalog — MVP-BUILD-PLAN.md Phase 4.

Offline by construction: every assertion runs against plain constructed pydantic objects, the
L1 `ProviderModelRegistry` (the exact validation + compilation `build_model_router` performs,
minus the `litellm.Router` construction and the MiniLM load), and litellm's PURE dispatch
resolvers (`get_llm_provider`, `ProviderConfigManager.get_provider_rerank_config`) — the
functions that decide, before any socket is opened, whether a compiled row can be called at all.
No network, no store, no weights.

Why the litellm-resolver assertions are here and not in an integration test: a catalog row that
LOOKS fine is worthless if litellm refuses to dispatch it. Both of those resolvers reject
`openai/…` local rows deterministically and offline (`Unsupported provider: openai` for rerank;
`Missing credentials` for a keyless chat call), which is exactly the class of defect a
"the group has >= 1 deployment" assertion cannot see.
"""

from __future__ import annotations

import pytest
import structlog
from structlog.testing import capture_logs

from mu_engine.providers.catalog import ModelKind, ProviderKind, Task
from mu_engine.providers.local_priority import LOCAL_ORDER, REMOTE_ORDER, LocalPriorityPolicy
from mu_engine.providers.registry import ProviderModelRegistry, RegistryError
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings, default_local_catalog
from mu_engine.providers.shipped_catalog import (
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

pytestmark = pytest.mark.unit

NO_KEYS: frozenset[str] = frozenset()


def _registry(
    catalog: ModelCatalogSettings,
    models: ModelSettings,
    *,
    resolver: object | None = None,
) -> ProviderModelRegistry:
    """Exactly what `build_model_router` (model_router.py:339-348) constructs — no Router, no
    embedder, so this stays fully offline."""
    task_map = TaskClassMapper(models)
    return ProviderModelRegistry(
        catalog.providers,
        catalog.deployments,
        local_policy=LocalPriorityPolicy(
            local_capable_tasks=frozenset(catalog.local_capable_tasks),
            enabled=catalog.local_priority_enabled,
        ),
        task_groups=task_map.task_groups(),
        secret_resolver=resolver,
    )


def _groups(catalog: ModelCatalogSettings) -> dict[str, list[str]]:
    """model_group -> the provider keys serving it."""
    out: dict[str, list[str]] = {}
    for dep in catalog.deployments:
        out.setdefault(dep.model_group, []).append(dep.provider_key)
    return out


def _local_keys(catalog: ModelCatalogSettings) -> set[str]:
    return {p.key for p in catalog.providers if p.is_local}


# ---------------------------------------------------------------------------------------------
# A. the shipped table: many providers -> one logical group
# ---------------------------------------------------------------------------------------------
def test_every_named_provider_is_declared_with_the_right_kind_and_locality() -> None:
    by_key = {p.key: p for p in shipped_providers()}

    assert set(by_key) == {
        ProviderKey.AZURE,
        ProviderKey.OPENAI,
        ProviderKey.ANTHROPIC,
        ProviderKey.DEEPSEEK,
        ProviderKey.MOONSHOT,
        ProviderKey.LOCAL_OPENAI_HTTP,
        ProviderKey.LOCAL_EMBED_RERANK_HTTP,
    }  # mu-local is absent until warm_local_enabled — absence, not a disabled stub
    for key in (
        ProviderKey.AZURE,
        ProviderKey.OPENAI,
        ProviderKey.ANTHROPIC,
        ProviderKey.DEEPSEEK,
        ProviderKey.MOONSHOT,
    ):
        assert by_key[key].kind is ProviderKind.REMOTE
        assert by_key[key].is_local is False
        assert by_key[key].credential_ref is not None  # a remote provider needs a named secret
    for key in (ProviderKey.LOCAL_OPENAI_HTTP, ProviderKey.LOCAL_EMBED_RERANK_HTTP):
        assert by_key[key].kind is ProviderKind.LOCAL_HTTP
        assert by_key[key].is_local is True
        assert by_key[key].credential_ref is None  # localhost needs no secret
        assert by_key[key].api_base is not None
        # ...and the prefix that makes "no secret" WORK on the wire (house rule 2). `openai`
        # here would demand OPENAI_API_KEY and, when one is present, ship it to localhost.
        assert by_key[key].litellm_provider == "hosted_vllm"
    # the litellm prefixes the compiler routes with
    assert by_key[ProviderKey.ANTHROPIC].litellm_provider == "anthropic"
    assert by_key[ProviderKey.DEEPSEEK].litellm_provider == "deepseek"
    assert by_key[ProviderKey.MOONSHOT].litellm_provider == "moonshot"
    assert by_key[ProviderKey.AZURE].litellm_provider == "azure"


def test_every_logical_group_is_served_by_several_providers_at_once() -> None:
    """The many-to-many seam is the POINT: one `model_group`, N `ModelDeployment`s across N
    providers — LiteLLM then owns health/cooldown/failover between them."""
    groups = _groups(shipped_catalog())

    for group in (
        ModelGroup.REASON_HARD,
        ModelGroup.CHAT,
        ModelGroup.EXTRACT_FAST,
        ModelGroup.SUMMARIZE,
        ModelGroup.CLASSIFY,
    ):
        serving = groups[group.value]
        assert len(serving) >= 3, f"{group} is not multi-provider: {serving}"
        assert len(set(serving)) == len(serving), f"{group} has a duplicated provider row"
    # the frontier group really does reach three different vendors + azure
    assert {ProviderKey.ANTHROPIC, ProviderKey.OPENAI, ProviderKey.DEEPSEEK} <= set(
        groups[ModelGroup.REASON_HARD.value]
    )


# --- the assertions that separate "declared" from "callable" ----------------------------------
def _local_rows(catalog: ModelCatalogSettings) -> list[tuple[str, str, str, ModelKind]]:
    """(group, model_id, api_base, kind) for every row served by a LOCAL provider."""
    bases = {p.key: p.api_base for p in catalog.providers}
    return [
        (d.model_group, d.model_id, bases[d.provider_key] or "", d.kind)
        for d in catalog.deployments
        if d.provider_key in _local_keys(catalog)
    ]


def test_every_local_row_resolves_to_a_keyless_litellm_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FULL-LOCAL means CALLABLE with zero keys. litellm's `get_llm_provider` is the pure
    resolver every call goes through first: it must hand the local row a usable key WITHOUT
    reading the operator's cloud secret. A sentinel OPENAI_API_KEY is planted precisely because
    an `openai/…` local row would silently pick it up and put it on a localhost request."""
    from litellm import get_llm_provider

    sentinel = "sk-PROBE-SENTINEL-must-never-reach-localhost"
    monkeypatch.setenv("OPENAI_API_KEY", sentinel)
    monkeypatch.delenv("HOSTED_VLLM_API_KEY", raising=False)
    catalog = shipped_catalog(cfg=ShippedCatalogSettings(local_serves_remote_preferred_groups=True))

    rows = _local_rows(catalog)
    assert rows
    for group, model_id, api_base, _kind in rows:
        _model, provider, dynamic_key, _base = get_llm_provider(
            model=model_id, api_base=api_base or None
        )
        assert provider == "hosted_vllm", f"{group}/{model_id} routes to {provider}"
        # a key litellm can use without any configuration...
        assert dynamic_key, f"{group}/{model_id} would be called with no credentials at all"
        # ...and NOT the operator's real cloud secret
        assert dynamic_key != sentinel, f"{group}/{model_id} leaks OPENAI_API_KEY to {api_base}"


def test_the_rerank_group_dispatches_to_a_real_rerank_handler() -> None:
    """`ModelRouter.rerank` calls `Router.arerank(model=<group>)` (model_router.py:238-240).
    litellm's rerank dispatch supports only a fixed provider set; anything else raises
    `Unsupported provider` BEFORE any I/O, which a "the group is non-empty" check cannot see.
    A default `CohereRerankConfig` on a non-cohere provider is litellm's own sentinel for
    'unsupported' (`litellm/rerank_api/main.py:505-515`)."""
    import litellm
    from litellm import get_llm_provider
    from litellm.utils import ProviderConfigManager

    catalog = shipped_catalog()
    rerank_rows = [d for d in catalog.deployments if d.kind is ModelKind.RERANK]
    assert rerank_rows, "the catalog declares no reranker at all"
    bases = {p.key: p.api_base for p in catalog.providers}

    for dep in rerank_rows:
        _model, provider, _key, _base = get_llm_provider(
            model=dep.model_id, api_base=bases[dep.provider_key]
        )
        config = ProviderConfigManager.get_provider_rerank_config(
            model=dep.model_id,
            provider=litellm.LlmProviders(provider),
            api_base=bases[dep.provider_key],
            present_version_params=[],
        )
        assert not isinstance(
            config, litellm.CohereRerankConfig | litellm.CohereRerankV2Config
        ), f"{dep.model_group}/{dep.model_id} -> litellm would raise Unsupported provider"
        assert isinstance(config, litellm.HostedVLLMRerankConfig)


def test_every_task_group_is_reachable_with_zero_keys_or_is_frontier_by_decision() -> None:
    """The house rule and its ONE ratified exception, stated as an invariant rather than as
    prose: with no credentials at all, every task group under `recommended_model_settings()`
    still has a local deployment — except the HARD tier, which ADR 0037 keeps frontier-only."""
    local_only = active_catalog(shipped_catalog(), available_credentials=NO_KEYS)
    served = _groups(local_only)
    table = TaskClassMapper(recommended_model_settings()).task_groups()

    frontier_only = {ModelGroup.REASON_HARD.value}
    for task, group in table.items():
        if group in frontier_only:
            assert group not in served, f"{group} kept a local row despite ADR 0037"
            continue
        assert served.get(group), f"task {task.value} -> group {group} is empty with zero keys"


def test_max_input_tokens_is_declared_only_where_it_is_known() -> None:
    """Never an invented number: the field is set for the vendor-published windows and left
    `None` everywhere else, so `ModelRouter._max_input_tokens` falls back to litellm.

    Keyed by (group, model_id): 13 model_ids repeat across groups, so a `model_id`-keyed dict
    would silently drop 19 of the 40 rows and let a shadowed row carry any number at all —
    and `registry.max_input_tokens()` takes `max()` across a GROUP, so a wrong number on a
    shadowed row really does move a live chunking budget."""
    deps = shipped_catalog().deployments
    declared = {(d.model_group, d.model_id): d.max_input_tokens for d in deps}
    assert len(declared) == len(deps), "two rows share a (group, model_id) key"

    # EVERY row carrying a window, exhaustively — not a sample.
    with_window = {k: v for k, v in declared.items() if v is not None}
    assert with_window == {
        (ModelGroup.REASON_HARD.value, "anthropic/claude-opus-4-1"): 200_000,
        (ModelGroup.REASON_HARD.value, "deepseek/deepseek-reasoner"): 64_000,
        (ModelGroup.CHAT.value, "anthropic/claude-sonnet-4-5"): 200_000,
        (ModelGroup.CHAT.value, "openai/gpt-4o"): 128_000,
        (ModelGroup.CHAT.value, "deepseek/deepseek-chat"): 64_000,
        (ModelGroup.CHAT.value, "moonshot/moonshot-v1-128k"): 131_072,
        (ModelGroup.EXTRACT_FAST.value, "anthropic/claude-haiku-4-5"): 200_000,
        (ModelGroup.EXTRACT_FAST.value, "openai/gpt-4o-mini"): 128_000,
        (ModelGroup.EXTRACT_FAST.value, "deepseek/deepseek-chat"): 64_000,
        (ModelGroup.SUMMARIZE.value, "anthropic/claude-haiku-4-5"): 200_000,
        (ModelGroup.SUMMARIZE.value, "moonshot/moonshot-v1-32k"): 32_768,
        (ModelGroup.SUMMARIZE.value, "openai/gpt-4o-mini"): 128_000,
        (ModelGroup.CLASSIFY.value, "openai/gpt-4o-mini"): 128_000,
        (ModelGroup.CLASSIFY.value, "anthropic/claude-haiku-4-5"): 200_000,
        (LegacyModelGroup.GPT_4O.value, "openai/gpt-4o"): 128_000,
    }
    # unknown/tenant-defined windows are OMITTED, not guessed
    assert declared[(ModelGroup.REASON_HARD.value, "openai/gpt-4.1")] is None
    assert all(d.max_input_tokens is None for d in deps if d.model_id.startswith("azure/"))
    assert all(d.max_input_tokens is None for d in deps if d.model_id.startswith("hosted_vllm/"))


def test_rerank_and_embed_groups_declare_their_real_model_kind() -> None:
    catalog = shipped_catalog()
    by_group = {d.model_group: d for d in catalog.deployments if d.kind is not ModelKind.LLM}

    assert by_group[ModelGroup.RERANK.value].kind is ModelKind.RERANK
    assert by_group[ModelGroup.EMBED.value].kind is ModelKind.EMBED
    assert all(
        d.kind is ModelKind.EMBED
        for d in catalog.deployments
        if d.model_group == LegacyModelGroup.TEXT_EMBEDDING_3_LARGE.value
    )
    assert all(
        d.kind is ModelKind.LLM
        for d in catalog.deployments
        if d.model_group
        in {ModelGroup.CHAT.value, ModelGroup.REASON_HARD.value, ModelGroup.CHAT_LOCAL.value}
    )


# ---------------------------------------------------------------------------------------------
# B. task -> group mapping (the ModelSettings seam is unchanged)
# ---------------------------------------------------------------------------------------------
def test_the_shipped_catalog_satisfies_the_current_gpt_only_defaults() -> None:
    """Deliverable B's safe half: the existing `ModelSettings()` defaults are left alone, and the
    `gpt-*` names they point at are declared as REAL groups — so nothing regresses."""
    catalog = shipped_catalog()

    reg = _registry(catalog, ModelSettings(), resolver=_Resolver())  # raises if a group is empty

    rows = {r["model_name"] for r in reg.compile_model_list()}
    assert {g.value for g in LegacyModelGroup} <= rows


def test_recommended_model_settings_points_every_task_at_a_logical_group() -> None:
    models = recommended_model_settings()
    table = TaskClassMapper(models).task_groups()

    assert table[Task.ANSWER] == ModelGroup.CHAT.value
    assert table[Task.ADJUDICATE] == ModelGroup.REASON_HARD.value
    assert table[Task.CONFLICT_ADJUDICATION] == ModelGroup.REASON_HARD.value  # ADR 0037
    assert table[Task.HARD_EXTRACT] == ModelGroup.REASON_HARD.value
    assert table[Task.ROUTINE_EXTRACT] == ModelGroup.EXTRACT_FAST.value
    assert table[Task.SUMMARIZE] == ModelGroup.SUMMARIZE.value
    assert table[Task.CLASSIFY] == ModelGroup.CLASSIFY.value
    assert table[Task.RERANK] == ModelGroup.RERANK.value
    assert table[Task.EMBED] == ModelGroup.EMBED.value
    # the EmbeddingPort seam (a different mechanism, §6-P5) is deliberately NOT repointed
    assert models.embed_backend == ModelSettings().embed_backend
    # user-configurable: it is still just `ModelSettings`, so a per-task override still wins
    assert (
        TaskClassMapper(
            recommended_model_settings(ModelSettings(classify_model="my-own-group"))
        ).task_groups()[Task.CLASSIFY]
        == ModelGroup.CLASSIFY.value
    )
    assert (
        TaskClassMapper(recommended_model_settings()).group_for(Task.CLASSIFY, override="ad-hoc")
        == "ad-hoc"
    )


def test_the_group_task_table_is_the_inverse_of_the_recommended_settings() -> None:
    """`group_tasks` decides WHERE a local row may sit, so it must not drift from the mapping it
    claims to invert (and from `ModelSettings`' own legacy field defaults)."""
    for models, groups in (
        (recommended_model_settings(), {g.value for g in ModelGroup}),
        (ModelSettings(), {g.value for g in LegacyModelGroup}),
    ):
        table = TaskClassMapper(models).task_groups()
        for group in groups:
            expected = {task for task, g in table.items() if g == group}
            if group == ModelGroup.CHAT_LOCAL.value:
                assert not expected  # the fallback sibling is referenced by NO task, by design
                continue
            assert group_tasks(group) == expected, group


def test_recommended_settings_validate_against_the_shipped_catalog() -> None:
    reg = _registry(shipped_catalog(), recommended_model_settings(), resolver=_Resolver())

    assert {r["model_name"] for r in reg.compile_model_list()} >= {
        g.value for g in ModelGroup if g is not ModelGroup.EMBED
    }


def test_recommended_model_settings_does_not_mutate_its_base() -> None:
    base = ModelSettings()

    out = recommended_model_settings(base)

    assert out is not base
    assert base.answer_model == "gpt-5-chat"  # the pinned CANONICAL default, untouched


# ---------------------------------------------------------------------------------------------
# C. WHERE a local row is allowed to sit (house rule 3 + ADR 0037)
# ---------------------------------------------------------------------------------------------
def test_no_local_row_ever_shares_a_tier_with_a_remote_row() -> None:
    """`order:` has two tiers and litellm keeps only the MIN one, then shuffles UNIFORMLY inside
    it (`litellm/utils.py:4496`). So a local row that sits in a group beside remote rows at the
    SAME order is a coin-flip sibling of a frontier model, not a fallback. The invariant: in any
    mixed group, the local rows are tier 1 and the remote rows are tier 2."""
    for models in (recommended_model_settings(), ModelSettings()):
        catalog = shipped_catalog()
        reg = _registry(catalog, models, resolver=_Resolver())
        rows = reg.compile_model_list()
        local_keys = _local_keys(catalog)
        by_group: dict[str, list[tuple[bool, int]]] = {}
        for dep, row in zip(catalog.deployments, rows, strict=True):
            by_group.setdefault(dep.model_group, []).append(
                (dep.provider_key in local_keys, row["litellm_params"]["order"])
            )
        referenced = set(TaskClassMapper(models).task_groups().values())
        for group, entries in by_group.items():
            if group not in referenced:
                continue  # an unreferenced group (the fallback sibling) has no tier to share
            kinds = {is_local for is_local, _ in entries}
            if kinds != {True, False}:
                continue  # not a mixed group
            for is_local, order in entries:
                assert order == (
                    LOCAL_ORDER if is_local else REMOTE_ORDER
                ), f"{group}: local={is_local} compiled to order {order} — uniform shuffle"


def test_the_hard_reasoning_groups_carry_no_local_deployment() -> None:
    """ADR 0037: 'Use the local SLM for adjudication. Rejected.' The model being unreachable must
    fall through to the deterministic `PolarityCardinalityHeuristic` + `DegradedModeEntered`,
    never to a weaker model — so the adjudicating groups hold no local row and no local fallback
    chain. `gpt-5-chat` is here too: the shipped `ModelSettings` defaults point ADJUDICATE at it."""
    catalog = shipped_catalog()
    local_keys = _local_keys(catalog)

    for group in (ModelGroup.REASON_HARD.value, LegacyModelGroup.GPT_5_CHAT.value):
        assert group_tasks(group) & {Task.ADJUDICATE, Task.CONFLICT_ADJUDICATION}
        serving = {d.provider_key for d in catalog.deployments if d.model_group == group}
        assert serving
        assert not serving & local_keys, f"{group} routes adjudication to a local model"
    chained = {key for chain in catalog.router.fallbacks for key in chain}
    assert ModelGroup.REASON_HARD.value not in chained
    assert LegacyModelGroup.GPT_5_CHAT.value not in chained


def test_answer_keeps_a_true_last_resort_local_sibling() -> None:
    """ANSWER is not adjudication: it may fall back to local, but only AFTER the primary group is
    exhausted. The only mechanism in the stack that expresses that is a cross-group fallback
    chain — `order:` cannot (see the tier test above)."""
    cfg = ShippedCatalogSettings()
    catalog = shipped_catalog(cfg=cfg)

    assert catalog.router.fallbacks == [{ModelGroup.CHAT.value: [ModelGroup.CHAT_LOCAL.value]}]
    sibling = [d for d in catalog.deployments if d.model_group == ModelGroup.CHAT_LOCAL.value]
    assert [d.model_id for d in sibling] == [f"hosted_vllm/{cfg.local_chat_model}"]
    assert all(d.provider_key == ProviderKey.LOCAL_OPENAI_HTTP for d in sibling)
    # ...and the primary CHAT group itself stays remote-only, so no shuffle can pick local
    assert not {d.provider_key for d in catalog.deployments if d.model_group == ModelGroup.CHAT} & (
        _local_keys(catalog)
    )


def test_fallback_promotion_never_invents_a_group_the_catalog_did_not_declare() -> None:
    """Promotion follows DECLARED data, it does not create routes. A chain whose head is a group
    the catalog never declared must stay unrealised — otherwise an arbitrary `router.fallbacks`
    entry could conjure a model group into existence during activation."""
    declared = shipped_catalog()
    with_ghost = declared.model_copy(
        update={
            "router": declared.router.model_copy(
                update={
                    "fallbacks": [
                        *declared.router.fallbacks,
                        {"a-group-nobody-declared": [ModelGroup.CHAT_LOCAL.value]},
                    ]
                }
            )
        }
    )

    active = active_catalog(with_ghost, available_credentials=NO_KEYS)

    assert all(d.model_group != "a-group-nobody-declared" for d in active.deployments)
    # ...while the DECLARED chain in the same list is still honoured (not vacuous)
    assert any(d.model_group == ModelGroup.CHAT.value for d in active.deployments)


def test_an_operator_fallback_chain_is_never_overwritten() -> None:
    base = default_local_catalog(
        ModelCatalogSettings(
            router=ModelCatalogSettings().router.model_copy(
                update={"fallbacks": [{ModelGroup.CHAT.value: ["my-own-group"]}]}
            )
        )
    )

    catalog = shipped_catalog(base)

    assert catalog.router.fallbacks == [{ModelGroup.CHAT.value: ["my-own-group"]}]


def test_local_placement_follows_local_capable_tasks() -> None:
    """The placement rule is not a hardcoded list — it reads the SAME
    `ModelCatalogSettings.local_capable_tasks` seam L4 stamps `order:` from."""
    without_summarize = [
        t for t in ModelCatalogSettings().local_capable_tasks if t is not Task.SUMMARIZE
    ]
    base = default_local_catalog(ModelCatalogSettings(local_capable_tasks=without_summarize))

    catalog = shipped_catalog(base)

    serving = {d.provider_key for d in catalog.deployments if d.model_group == ModelGroup.SUMMARIZE}
    assert not serving & _local_keys(catalog)
    # ...while a task still in the set keeps its local row
    classify = {d.provider_key for d in catalog.deployments if d.model_group == ModelGroup.CLASSIFY}
    assert classify & _local_keys(catalog)


def test_the_no_remote_credentials_posture_is_explicit_and_logged() -> None:
    """A box with NO cloud keys can still resolve the remote-preferred groups — but only by
    setting the flag, and the ADR 0037 deviation that implies is a NAMED event, never silent."""
    cfg = ShippedCatalogSettings(local_serves_remote_preferred_groups=True)

    with capture_logs() as events:
        catalog = shipped_catalog(cfg=cfg)

    named = [
        e for e in events if e["event"] == "model_catalog_local_serves_remote_preferred_groups"
    ]
    assert len(named) == 1
    assert named[0]["adjudicating_groups"] == [
        LegacyModelGroup.GPT_5_CHAT.value,
        ModelGroup.REASON_HARD.value,
    ]
    assert catalog.router.fallbacks == []  # the local rows are primary — nothing to fall back TO
    assert shipped_router_fallbacks(cfg) == []
    local_only = active_catalog(catalog, available_credentials=NO_KEYS)
    for models in (ModelSettings(), recommended_model_settings()):
        _registry(local_only, models)  # raises RegistryError if any task group went empty


def test_without_the_flag_a_keyless_box_fails_loud_on_the_hard_tier() -> None:
    """The other half of the same decision: with zero keys and the flag OFF, the adjudicating
    group is EMPTY and `ProviderModelRegistry` refuses to start, naming the group. That is the
    designed fail-loud (model-layer-spec §5) — never a silent downgrade to a 7B local model."""
    local_only = active_catalog(shipped_catalog(), available_credentials=NO_KEYS)

    with pytest.raises(RegistryError) as err:
        _registry(local_only, recommended_model_settings())

    assert ModelGroup.REASON_HARD.value in str(err.value)


# ---------------------------------------------------------------------------------------------
# D. credential-aware activation
# ---------------------------------------------------------------------------------------------
class _Resolver:
    """A SecretResolver that knows some refs and raises on the rest (the real seam's shape)."""

    VALUE = "SECRET-VALUE-do-not-leak"  # a fake secret for a hygiene test

    def __init__(self, known: frozenset[str] | None = None) -> None:
        self.known = known if known is not None else frozenset(_all_refs())

    def resolve(self, credential_ref: str) -> str:
        if credential_ref not in self.known:
            raise KeyError(credential_ref)
        return f"{self.VALUE}:{credential_ref}"


def _all_refs() -> set[str]:
    return {p.credential_ref for p in shipped_providers() if p.credential_ref is not None}


def test_absent_credentials_exclude_the_provider_and_its_deployments() -> None:
    declared = shipped_catalog()
    only_openai = {ShippedCatalogSettings().openai_credential_ref}

    active = active_catalog(declared, available_credentials=only_openai)

    keys = {p.key for p in active.providers}
    assert ProviderKey.OPENAI in keys
    assert ProviderKey.ANTHROPIC not in keys
    assert ProviderKey.AZURE not in keys
    assert ProviderKey.DEEPSEEK not in keys
    assert ProviderKey.MOONSHOT not in keys
    # local providers never carry a credential, so they always survive
    assert {ProviderKey.LOCAL_OPENAI_HTTP, ProviderKey.LOCAL_EMBED_RERANK_HTTP} <= keys
    # the excluded providers take their rows with them — no dangling provider_key
    assert {d.provider_key for d in active.deployments} <= keys
    assert len(active.deployments) < len(declared.deployments)


def test_activation_is_not_a_hard_failure_and_not_a_silent_success() -> None:
    """Excluding a provider must leave a catalog the registry still accepts, AND must SAY what it
    excluded. Both halves are asserted: the surviving rows carry no `api_key` (nothing was
    resolved that could not be), and the named, content-free events actually fire."""
    declared = shipped_catalog()

    with capture_logs() as events:
        active = active_catalog(declared, available_credentials=NO_KEYS)

    # the HARD tier has no local row by decision (ADR 0037) — a keyless box must say explicitly
    # what it wants there; everything else resolves untouched.
    models = recommended_model_settings().model_copy(
        update={
            "adjudicate_model": ModelGroup.CHAT.value,
            "hard_extract_model": ModelGroup.CHAT.value,
        }
    )
    reg = _registry(active, models)  # no resolver needed: nothing carries a ref
    rows = reg.compile_model_list()
    assert rows
    assert all("api_key" not in r["litellm_params"] for r in rows)

    excluded = [e for e in events if e["event"] == "model_catalog_provider_excluded"]
    assert len(excluded) == 1
    assert excluded[0]["providers"] == sorted(
        [
            ProviderKey.ANTHROPIC.value,
            ProviderKey.AZURE.value,
            ProviderKey.DEEPSEEK.value,
            ProviderKey.MOONSHOT.value,
            ProviderKey.OPENAI.value,
        ]
    )
    assert excluded[0]["missing_credential_refs"] == sorted(_all_refs())
    assert excluded[0]["providers_active"] == 2
    # counted BEFORE any fallback promotion: exactly the remote-provider rows
    assert excluded[0]["deployments_dropped"] == len(declared.deployments) - len(
        _local_rows(declared)
    )
    # no secret VALUE is ever in the event, only names
    assert _Resolver.VALUE not in repr(events)

    promoted = [e for e in events if e["event"] == "model_catalog_fallback_promoted"]
    assert [(e["group"], e["source_group"], e["rows"]) for e in promoted] == [
        (ModelGroup.CHAT.value, ModelGroup.CHAT_LOCAL.value, 1)
    ]
    emptied = [e for e in events if e["event"] == "model_catalog_group_emptied"]
    assert len(emptied) == 1
    # mu-chat is NOT here: it adopted its declared local chain. The two adjudicating groups are,
    # because ADR 0037 gives them no chain to adopt.
    assert emptied[0]["groups"] == [
        LegacyModelGroup.GPT_5_CHAT.value,
        ModelGroup.REASON_HARD.value,
    ]


def test_activation_says_nothing_when_nothing_was_excluded() -> None:
    """The events are a SIGNAL, not noise: a fully-credentialed activation is silent."""
    with capture_logs() as events:
        active_catalog(shipped_catalog(), available_credentials=_all_refs())

    assert [e["event"] for e in events] == []


def test_activation_does_not_mutate_the_declared_catalog() -> None:
    declared = shipped_catalog()
    before = len(declared.providers), len(declared.deployments)

    active = active_catalog(declared, available_credentials=NO_KEYS)

    assert active is not declared
    assert (len(declared.providers), len(declared.deployments)) == before
    assert len(active.providers) < len(declared.providers)


def test_all_credentials_present_keeps_the_whole_catalog() -> None:
    declared = shipped_catalog()

    active = active_catalog(declared, available_credentials=_all_refs())

    assert [p.key for p in active.providers] == [p.key for p in declared.providers]
    assert len(active.deployments) == len(declared.deployments)


def test_resolvable_credential_refs_probes_the_seam_without_raising() -> None:
    declared = shipped_catalog()
    cfg = ShippedCatalogSettings()
    partial = _Resolver(frozenset({cfg.anthropic_credential_ref, cfg.deepseek_credential_ref}))

    refs = resolvable_credential_refs(declared, partial)

    assert refs == {cfg.anthropic_credential_ref, cfg.deepseek_credential_ref}
    active = active_catalog(declared, available_credentials=refs)
    assert {p.key for p in active.providers} == {
        ProviderKey.ANTHROPIC,
        ProviderKey.DEEPSEEK,
        ProviderKey.LOCAL_OPENAI_HTTP,
        ProviderKey.LOCAL_EMBED_RERANK_HTTP,
    }


@pytest.mark.parametrize(
    "exc",
    [KeyError("nope"), FileNotFoundError("/run/secrets/x"), RuntimeError("vault down")],
    ids=["KeyError", "FileNotFoundError", "RuntimeError"],
)
def test_any_resolver_failure_counts_as_absent_and_is_named(exc: Exception) -> None:
    """The resolver's exception TYPE is its own business (`RegistryError`, `KeyError`,
    `FileNotFoundError`, …), so the probe catches broadly — and logs the type, never the
    message, which can carry a path or a value."""

    class _Raises:
        def resolve(self, credential_ref: str) -> str:
            raise exc

    with capture_logs() as events:
        refs = resolvable_credential_refs(shipped_catalog(), _Raises())

    assert refs == frozenset()
    absent = [e for e in events if e["event"] == "model_catalog_credential_absent"]
    assert len(absent) == len(_all_refs())
    assert {e["cause"] for e in absent} == {type(exc).__name__}
    assert {e["credential_ref"] for e in absent} == _all_refs()
    assert str(exc) not in repr(events)  # the MESSAGE never travels


def test_an_empty_string_secret_counts_as_absent() -> None:
    class _Blank:
        def resolve(self, credential_ref: str) -> str:
            return ""

    with capture_logs() as events:
        assert resolvable_credential_refs(shipped_catalog(), _Blank()) == frozenset()

    assert {e["cause"] for e in events if e["event"] == "model_catalog_credential_absent"} == {
        "empty"
    }


# ---------------------------------------------------------------------------------------------
# E. FULL-LOCAL with zero API keys
# ---------------------------------------------------------------------------------------------
def test_local_only_catalog_serves_every_reachable_task_with_no_keys_at_all() -> None:
    """Zero credentials, still a complete system for every task the HARD-tier decision leaves
    local: the group resolves AND every one of its rows dispatches to a keyless handler."""
    local_only = active_catalog(shipped_catalog(), available_credentials=NO_KEYS)

    assert local_only.providers
    assert all(p.is_local for p in local_only.providers)
    assert all(p.credential_ref is None for p in local_only.providers)
    # EVERY surviving row is a local one, and nothing dangles
    assert len(_local_rows(local_only)) == len(local_only.deployments)
    assert {d.provider_key for d in local_only.deployments} <= {p.key for p in local_only.providers}
    # ANSWER survives because `mu-chat` adopted its declared local chain during activation
    assert [d.model_id for d in local_only.deployments if d.model_group == ModelGroup.CHAT] == [
        f"hosted_vllm/{ShippedCatalogSettings().local_chat_model}"
    ]


def test_local_only_catalog_keeps_the_offline_embedder_backend() -> None:
    """`shipped_catalog()` builds on `default_local_catalog()`, so the MiniLM `EmbeddingPort`
    backend (the REAL embed path, §6-P5) survives activation with zero keys."""
    local_only = active_catalog(shipped_catalog(), available_credentials=NO_KEYS)

    assert set(local_only.embedders) == {ModelCatalogSettings().default_embed_backend}
    assert (
        local_only.embedders[ModelCatalogSettings().default_embed_backend].kind is ModelKind.EMBED
    )


def test_local_first_order_is_stamped_per_task_capability() -> None:
    """L4: a local deployment of a LOCAL-CAPABLE task is tier 1 (preferred); its remote siblings
    are tier 2."""
    catalog = shipped_catalog()
    reg = _registry(catalog, recommended_model_settings(), resolver=_Resolver())
    rows = reg.compile_model_list()
    orders = {
        (r["model_name"], r["litellm_params"]["model"]): r["litellm_params"]["order"] for r in rows
    }
    cfg = ShippedCatalogSettings()

    assert orders[(ModelGroup.CLASSIFY.value, f"hosted_vllm/{cfg.local_tiny_model}")] == LOCAL_ORDER
    assert (
        orders[(ModelGroup.EXTRACT_FAST.value, f"hosted_vllm/{cfg.local_fast_model}")]
        == LOCAL_ORDER
    )
    assert (
        orders[(ModelGroup.SUMMARIZE.value, f"hosted_vllm/{cfg.local_chat_model}")] == LOCAL_ORDER
    )
    assert orders[(ModelGroup.RERANK.value, f"hosted_vllm/{cfg.local_rerank_model}")] == LOCAL_ORDER
    # the remote siblings of a local-capable group are the fallback tier
    assert orders[(ModelGroup.CLASSIFY.value, "openai/gpt-4o-mini")] == REMOTE_ORDER
    assert orders[(ModelGroup.SUMMARIZE.value, "openai/gpt-4o-mini")] == REMOTE_ORDER


# ---------------------------------------------------------------------------------------------
# F. credential hygiene — no key value ever lives in the catalog
# ---------------------------------------------------------------------------------------------
def test_no_deployment_or_provider_carries_a_key_shaped_value() -> None:
    """HARD constraint 1. `ModelDeployment.extra_params` is an EXISTING working api_key channel
    (both composition roots use it today) — the shipped catalog must never use it that way."""
    cfg = ShippedCatalogSettings(azure_api_version="2024-10-21")
    catalog = shipped_catalog(cfg=cfg)
    banned = {"api_key", "api-key", "key", "token", "password", "secret", "authorization"}

    for dep in catalog.deployments:
        assert not banned & {k.lower() for k in dep.extra_params}, dep.model_id
        for value in dep.extra_params.values():
            assert value == "2024-10-21"  # api_version is the ONLY passthrough used
    for provider in catalog.providers:
        dumped = provider.model_dump()
        assert set(dumped) == {
            "key",
            "kind",
            "litellm_provider",
            "api_base",
            "credential_ref",
            "is_local",
        }
        # credential_ref is a NAME under the secret seam, resolved at compile time
        assert dumped["credential_ref"] in (None, *_named_refs(cfg))


def _named_refs(cfg: ShippedCatalogSettings) -> set[str]:
    return {
        cfg.azure_credential_ref,
        cfg.openai_credential_ref,
        cfg.anthropic_credential_ref,
        cfg.deepseek_credential_ref,
        cfg.moonshot_credential_ref,
    }


def test_the_resolved_secret_reaches_litellm_params_only_via_the_resolver() -> None:
    catalog = shipped_catalog()
    resolver = _Resolver()

    rows = _registry(catalog, ModelSettings(), resolver=resolver).compile_model_list()

    keyed = [r for r in rows if "api_key" in r["litellm_params"]]
    assert keyed  # remote rows DO get a key — but only through the seam
    assert all(r["litellm_params"]["api_key"].startswith(_Resolver.VALUE) for r in keyed)
    # ...and it came from the resolver, never from the declared catalog itself
    assert _Resolver.VALUE not in repr(catalog)


# ---------------------------------------------------------------------------------------------
# G. flags: absence is the house rule
# ---------------------------------------------------------------------------------------------
def test_warm_local_is_off_by_default_and_leaves_no_dangling_prefix() -> None:
    """An enabled `warm_local` entry LOADS WEIGHTS in `build_model_router`, so it is opt-in — and
    while it is off neither the provider nor its deployments exist."""
    assert shipped_warm_local() == []
    assert all(p.key != ProviderKey.MU_LOCAL for p in shipped_providers())
    assert all(d.provider_key != ProviderKey.MU_LOCAL for d in shipped_deployments())


def test_enabling_warm_local_adds_provider_config_and_deployments_together() -> None:
    cfg = ShippedCatalogSettings(warm_local_enabled=True)

    catalog = shipped_catalog(cfg=cfg)

    provider = next(p for p in catalog.providers if p.key == ProviderKey.MU_LOCAL)
    assert provider.kind is ProviderKind.LOCAL_INPROC
    assert provider.is_local is True
    assert provider.credential_ref is None
    assert provider.litellm_provider == cfg.warm_local_model_id.split("/", 1)[0]
    assert [w.model_id for w in catalog.warm_local] == [cfg.warm_local_model_id]
    served = {d.model_group for d in catalog.deployments if d.provider_key == ProviderKey.MU_LOCAL}
    assert served == {ModelGroup.CLASSIFY.value, ModelGroup.EXTRACT_FAST.value}


def test_disabling_mu_local_while_warm_local_is_enabled_removes_it_entirely() -> None:
    cfg = ShippedCatalogSettings(
        warm_local_enabled=True, disabled_provider_keys=(ProviderKey.MU_LOCAL.value,)
    )

    catalog = shipped_catalog(cfg=cfg)

    assert catalog.warm_local == []
    assert all(p.key != ProviderKey.MU_LOCAL for p in catalog.providers)
    assert all(d.provider_key != ProviderKey.MU_LOCAL for d in catalog.deployments)
    _registry(catalog, ModelSettings(), resolver=_Resolver())  # still a valid catalog


def test_disabling_a_provider_removes_it_and_every_row_that_referenced_it() -> None:
    cfg = ShippedCatalogSettings(disabled_provider_keys=(ProviderKey.ANTHROPIC.value,))

    catalog = shipped_catalog(cfg=cfg)

    assert all(p.key != ProviderKey.ANTHROPIC for p in catalog.providers)
    assert all(d.provider_key != ProviderKey.ANTHROPIC for d in catalog.deployments)
    _registry(catalog, ModelSettings(), resolver=_Resolver())  # still a valid catalog


@pytest.mark.parametrize(
    ("flag", "provider_key"),
    [
        ("azure_enabled", ProviderKey.AZURE),
        ("local_http_enabled", ProviderKey.LOCAL_OPENAI_HTTP),
        ("local_embed_rerank_enabled", ProviderKey.LOCAL_EMBED_RERANK_HTTP),
    ],
)
def test_each_provider_flag_actually_gates_its_provider_and_its_rows(
    flag: str, provider_key: ProviderKey
) -> None:
    """Every feature flag is exercised: turned off, the provider AND every row referencing it are
    absent (not a disabled stub), and no dangling `provider_key` survives for the registry."""
    cfg = ShippedCatalogSettings(**{flag: False})

    catalog = shipped_catalog(cfg=cfg)

    assert all(p.key != provider_key for p in catalog.providers)
    assert all(d.provider_key != provider_key for d in catalog.deployments)
    assert {d.provider_key for d in catalog.deployments} <= {p.key for p in catalog.providers}
    # ...and the ON case really does emit it, so the assertion above is not vacuous
    assert any(p.key == provider_key for p in shipped_providers())


def test_disabling_the_local_chat_endpoint_drops_its_fallback_chain_too() -> None:
    """A chain pointing at a group with no deployments would be a dangling reference for
    litellm; the sibling group and its chain appear and disappear together."""
    cfg = ShippedCatalogSettings(local_http_enabled=False)

    catalog = shipped_catalog(cfg=cfg)

    assert catalog.router.fallbacks == []
    assert all(d.model_group != ModelGroup.CHAT_LOCAL.value for d in catalog.deployments)
    assert shipped_catalog().router.fallbacks  # not vacuous


def test_every_environment_dependent_string_in_the_table_comes_from_config() -> None:
    """DEV-STANDARDS rule 3, enforced EXHAUSTIVELY rather than by sampling: every field of
    `ShippedCatalogSettings` is given a sentinel, and the whole azure/local half of the table is
    compared as a SET — so replacing any single config read with the literal it happens to
    default to goes red, including a field read from several rows."""
    cfg = ShippedCatalogSettings(
        azure_credential_ref="REF-AZURE",
        openai_credential_ref="REF-OPENAI",
        anthropic_credential_ref="REF-ANTHROPIC",
        deepseek_credential_ref="REF-DEEPSEEK",
        moonshot_credential_ref="REF-MOONSHOT",
        azure_api_base="https://AZURE-BASE",
        azure_api_version="AZURE-VERSION",
        azure_frontier_deployment="AZ-FRONTIER",
        azure_balanced_deployment="AZ-BALANCED",
        azure_small_deployment="AZ-SMALL",
        azure_embed_deployment="AZ-EMBED",
        local_http_api_base="http://LOCAL-CHAT-BASE/v1",
        local_chat_model="LOCAL-CHAT",
        local_fast_model="LOCAL-FAST",
        local_tiny_model="LOCAL-TINY",
        local_embed_rerank_api_base="http://LOCAL-ER-BASE/v1",
        local_rerank_model="LOCAL-RERANK",
        local_embed_model="LOCAL-EMBED",
        warm_local_enabled=True,
        warm_local_model_id="warm-prefix/WARM-ID",
        warm_local_load_path="org/WARM-PATH",
    )

    catalog = shipped_catalog(cfg=cfg)

    by_key = {p.key: p for p in catalog.providers}
    assert by_key[ProviderKey.AZURE].api_base == "https://AZURE-BASE"
    assert by_key[ProviderKey.LOCAL_OPENAI_HTTP].api_base == "http://LOCAL-CHAT-BASE/v1"
    assert by_key[ProviderKey.LOCAL_EMBED_RERANK_HTTP].api_base == "http://LOCAL-ER-BASE/v1"
    assert {p.credential_ref for p in catalog.providers if p.credential_ref} == {
        "REF-AZURE",
        "REF-OPENAI",
        "REF-ANTHROPIC",
        "REF-DEEPSEEK",
        "REF-MOONSHOT",
    }
    assert by_key[ProviderKey.MU_LOCAL].litellm_provider == "warm-prefix"
    assert [(w.model_id, w.model_load_path) for w in catalog.warm_local] == [
        ("warm-prefix/WARM-ID", "org/WARM-PATH")
    ]

    def rows_for(key: ProviderKey) -> set[tuple[str, str]]:
        return {(d.model_group, d.model_id) for d in catalog.deployments if d.provider_key == key}

    assert rows_for(ProviderKey.AZURE) == {
        (ModelGroup.REASON_HARD.value, "azure/AZ-FRONTIER"),
        (ModelGroup.CHAT.value, "azure/AZ-FRONTIER"),
        (ModelGroup.EXTRACT_FAST.value, "azure/AZ-SMALL"),
        (ModelGroup.SUMMARIZE.value, "azure/AZ-BALANCED"),
        (ModelGroup.CLASSIFY.value, "azure/AZ-SMALL"),
        (ModelGroup.EMBED.value, "azure/AZ-EMBED"),
        (LegacyModelGroup.GPT_5_CHAT.value, "azure/AZ-FRONTIER"),
        (LegacyModelGroup.GPT_4O.value, "azure/AZ-BALANCED"),
        (LegacyModelGroup.GPT_41_MINI.value, "azure/AZ-SMALL"),
        (LegacyModelGroup.TEXT_EMBEDDING_3_LARGE.value, "azure/AZ-EMBED"),
    }
    assert rows_for(ProviderKey.LOCAL_OPENAI_HTTP) == {
        (ModelGroup.EXTRACT_FAST.value, "hosted_vllm/LOCAL-FAST"),
        (ModelGroup.SUMMARIZE.value, "hosted_vllm/LOCAL-CHAT"),
        (ModelGroup.CLASSIFY.value, "hosted_vllm/LOCAL-TINY"),
        (LegacyModelGroup.GPT_4O.value, "hosted_vllm/LOCAL-FAST"),
        (LegacyModelGroup.GPT_41_MINI.value, "hosted_vllm/LOCAL-TINY"),
        (ModelGroup.CHAT_LOCAL.value, "hosted_vllm/LOCAL-CHAT"),
    }
    assert rows_for(ProviderKey.LOCAL_EMBED_RERANK_HTTP) == {
        (ModelGroup.RERANK.value, "hosted_vllm/LOCAL-RERANK"),
        (ModelGroup.EMBED.value, "hosted_vllm/LOCAL-EMBED"),
        (LegacyModelGroup.TEXT_EMBEDDING_3_LARGE.value, "hosted_vllm/LOCAL-EMBED"),
    }
    assert rows_for(ProviderKey.MU_LOCAL) == {
        (ModelGroup.CLASSIFY.value, "warm-prefix/WARM-ID"),
        (ModelGroup.EXTRACT_FAST.value, "warm-prefix/WARM-ID"),
    }
    assert {v for d in catalog.deployments for v in d.extra_params.values()} == {"AZURE-VERSION"}


# ---------------------------------------------------------------------------------------------
# H. the shipped catalog is OPT-IN — it changes no existing default
# ---------------------------------------------------------------------------------------------
def test_shipped_catalog_does_not_change_the_bare_defaults() -> None:
    """`ModelCatalogSettings()` / `default_local_catalog()` still ship an EMPTY catalog; adopting
    the table is an explicit call, so no composition root changes behaviour by accident."""
    assert ModelCatalogSettings().providers == []
    assert ModelCatalogSettings().deployments == []
    assert ModelCatalogSettings().router.fallbacks == []
    assert default_local_catalog().providers == []
    assert default_local_catalog().deployments == []
    assert shipped_catalog().providers


def test_shipped_catalog_preserves_every_other_subtree_of_its_base() -> None:
    base = default_local_catalog(
        ModelCatalogSettings(local_priority_enabled=False, local_capable_tasks=[Task.CLASSIFY])
    )

    catalog = shipped_catalog(base)

    assert catalog is not base
    assert catalog.local_priority_enabled is False
    assert catalog.local_capable_tasks == [Task.CLASSIFY]
    assert catalog.embedders == base.embedders
    assert base.providers == []  # the base object itself was never mutated
    assert base.router.fallbacks == []  # ...router subtree included


def test_structlog_capture_is_wired(caplog: pytest.LogCaptureFixture) -> None:
    """Guard for the log assertions above: prove `capture_logs` actually observes THIS module's
    logger, so a green log test can never mean 'the capture saw nothing'."""
    with capture_logs() as events:
        structlog.get_logger("mu_engine.providers").info("probe_event", n=1)

    assert [e["event"] for e in events] == ["probe_event"]
