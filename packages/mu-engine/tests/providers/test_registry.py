"""L1 registry — compile, order-stamping, fail-loud catalog (model-layer-spec §8 tests 2,9)."""

from __future__ import annotations

import pytest

from mu_engine.providers.catalog import (
    ModelDeployment,
    ModelKind,
    ProviderKind,
    ProviderRecord,
    Task,
)
from mu_engine.providers.local_priority import LOCAL_ORDER, REMOTE_ORDER, LocalPriorityPolicy
from mu_engine.providers.registry import ProviderModelRegistry, RegistryError

pytestmark = pytest.mark.unit


def _policy(enabled: bool = True) -> LocalPriorityPolicy:
    return LocalPriorityPolicy(
        local_capable_tasks=frozenset(
            {Task.EMBED, Task.RERANK, Task.CLASSIFY, Task.ROUTINE_EXTRACT, Task.SUMMARIZE}
        ),
        enabled=enabled,
    )


def _task_groups() -> dict[Task, str]:
    # classify is local-capable and maps to "slm-fast" (has a local + remote deployment)
    return {
        Task.ANSWER: "big",
        Task.CLASSIFY: "slm-fast",
        Task.ROUTINE_EXTRACT: "slm-fast",
        Task.EMBED: "embed-remote",
    }


def _mixed_registry() -> ProviderModelRegistry:
    providers = [
        ProviderRecord(key="azure", kind=ProviderKind.REMOTE, litellm_provider="azure"),
        ProviderRecord(
            key="mu-local",
            kind=ProviderKind.LOCAL_INPROC,
            litellm_provider="mu-local",
            is_local=True,
        ),
    ]
    deployments = [
        ModelDeployment(model_group="big", provider_key="azure", model_id="azure/gpt-5-chat"),
        ModelDeployment(
            model_group="slm-fast",
            provider_key="mu-local",
            model_id="mu-local/phi",
            max_input_tokens=8000,
        ),
        ModelDeployment(model_group="slm-fast", provider_key="azure", model_id="azure/gpt-4o"),
    ]
    return ProviderModelRegistry(
        providers, deployments, local_policy=_policy(), task_groups=_task_groups()
    )


def test_compile_stamps_local_order_one_and_remote_two() -> None:
    reg = _mixed_registry()
    rows = reg.compile_model_list()
    by_id = {r["litellm_params"]["model"]: r["litellm_params"]["order"] for r in rows}
    # slm-fast serves CLASSIFY/ROUTINE_EXTRACT (local-capable): local deployment → order 1
    assert by_id["mu-local/phi"] == LOCAL_ORDER
    # remote sibling in the same local-capable group → order 2
    assert by_id["azure/gpt-4o"] == REMOTE_ORDER
    # "big" serves ANSWER (not local-capable) → remote order 2
    assert by_id["azure/gpt-5-chat"] == REMOTE_ORDER


def test_local_priority_disabled_stamps_everything_remote() -> None:
    providers = [
        ProviderRecord(
            key="mu-local",
            kind=ProviderKind.LOCAL_INPROC,
            litellm_provider="mu-local",
            is_local=True,
        )
    ]
    deps = [
        ModelDeployment(model_group="slm-fast", provider_key="mu-local", model_id="mu-local/phi")
    ]
    reg = ProviderModelRegistry(
        providers,
        deps,
        local_policy=_policy(enabled=False),
        task_groups={Task.CLASSIFY: "slm-fast"},
    )
    assert reg.compile_model_list()[0]["litellm_params"]["order"] == REMOTE_ORDER


def test_is_local_group_and_max_input_tokens() -> None:
    reg = _mixed_registry()
    assert reg.is_local_group("slm-fast") is True
    assert reg.is_local_group("big") is False
    assert reg.max_input_tokens("slm-fast") == 8000  # largest declared across the group
    assert reg.max_input_tokens("big") is None


def test_unknown_provider_key_fails_loud() -> None:
    with pytest.raises(RegistryError, match="unknown provider_key"):
        ProviderModelRegistry(
            [ProviderRecord(key="azure", kind=ProviderKind.REMOTE, litellm_provider="azure")],
            [ModelDeployment(model_group="g", provider_key="ghost", model_id="ghost/x")],
            local_policy=_policy(),
            task_groups={Task.ANSWER: "g"},
        )


def test_task_group_with_no_deployment_fails_loud() -> None:
    with pytest.raises(RegistryError, match="no deployment"):
        ProviderModelRegistry(
            [ProviderRecord(key="azure", kind=ProviderKind.REMOTE, litellm_provider="azure")],
            [ModelDeployment(model_group="present", provider_key="azure", model_id="azure/x")],
            local_policy=_policy(),
            task_groups={Task.ANSWER: "present", Task.SUMMARIZE: "MISSING"},
        )


def test_embed_task_is_not_required_as_a_deployment() -> None:
    # EMBED is served by the dedicated EmbeddingPort seam, not a litellm deployment (§6-P5).
    reg = ProviderModelRegistry(
        [ProviderRecord(key="azure", kind=ProviderKind.REMOTE, litellm_provider="azure")],
        [ModelDeployment(model_group="g", provider_key="azure", model_id="azure/x")],
        local_policy=_policy(),
        task_groups={Task.ANSWER: "g", Task.EMBED: "embed-remote"},  # no embed-remote deployment
    )
    assert reg.compile_model_list()  # does not raise


def test_credential_ref_without_resolver_fails_loud() -> None:
    with pytest.raises(RegistryError, match="no SecretResolver"):
        ProviderModelRegistry(
            [
                ProviderRecord(
                    key="azure",
                    kind=ProviderKind.REMOTE,
                    litellm_provider="azure",
                    credential_ref="azure_key",
                )
            ],
            [ModelDeployment(model_group="g", provider_key="azure", model_id="azure/x")],
            local_policy=_policy(),
            task_groups={Task.ANSWER: "g"},
        ).compile_model_list()


def test_custom_provider_keys() -> None:
    assert _mixed_registry().custom_provider_keys() == {"mu-local"}


def test_rerank_kind_deployment_compiles() -> None:
    reg = ProviderModelRegistry(
        [
            ProviderRecord(
                key="tei",
                kind=ProviderKind.LOCAL_HTTP,
                litellm_provider="openai",
                api_base="http://localhost:8080",
                is_local=True,
            )
        ],
        [
            ModelDeployment(
                model_group="rr",
                provider_key="tei",
                model_id="openai/bge-reranker",
                kind=ModelKind.RERANK,
            )
        ],
        local_policy=_policy(),
        task_groups={Task.RERANK: "rr"},
    )
    row = reg.compile_model_list()[0]
    assert row["litellm_params"]["api_base"] == "http://localhost:8080"
    assert row["litellm_params"]["order"] == LOCAL_ORDER  # RERANK is local-capable
