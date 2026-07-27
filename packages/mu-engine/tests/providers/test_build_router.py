"""L7 composition root — full wiring, two-plane catalog, credential hygiene, port suites.

Spec §8 tests 9 (fail-loud catalog at composition), 10 (two-plane wiring), 11 (credential
hygiene), plus the LLMProviderPort/EmbeddingPort contract suites. The remote (Azure) LLM is
config-wired but NEVER called — only the local MiniLM embedder is exercised (offline).
"""

from __future__ import annotations

import pytest

from mu_engine.providers._contracts import EmbeddingPort, LLMProviderPort
from mu_engine.providers.catalog import (
    ModelDeployment,
    ModelKind,
    ProviderKind,
    ProviderRecord,
    Task,
    WarmLocalConfig,
)
from mu_engine.providers.model_router import ModelRouter, build_model_router
from mu_engine.providers.registry import RegistryError
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings
from mu_engine.providers.task_map import TaskClassMapper

pytestmark = pytest.mark.unit

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


class _Resolver:
    """A SecretResolver that returns a secret WITHOUT ever letting it stringify into a log/repr."""

    SECRET = "SECRET-AZURE-KEY-do-not-leak"  # noqa: S105 — a fake secret for a hygiene test

    def resolve(self, credential_ref: str) -> str:
        assert credential_ref == "azure_key"
        return self.SECRET


def test_build_wires_embedder_and_resolves_task_groups(
    minilm_catalog: ModelCatalogSettings, default_models: ModelSettings
) -> None:
    mr = build_model_router(
        models=default_models, catalog=minilm_catalog, secret_resolver=_Resolver()
    )
    # EmbeddingPort attrs come from the REAL MiniLM (never assumed)
    assert mr.dimension == 384
    assert mr.model_name == MINILM
    # task groups resolve to the wired (never-called) Azure groups
    tm = TaskClassMapper(default_models)
    assert tm.group_for(Task.ANSWER) == "gpt-5-chat"
    assert tm.group_for(Task.CLASSIFY) == "gpt-4.1-mini"


async def test_router_implements_both_canonical_ports(
    minilm_catalog: ModelCatalogSettings, default_models: ModelSettings
) -> None:
    mr = build_model_router(
        models=default_models, catalog=minilm_catalog, secret_resolver=_Resolver()
    )
    assert isinstance(mr, LLMProviderPort)  # complete(...)
    assert isinstance(mr, EmbeddingPort)  # embed(...) + model_name/dimension
    vecs = await mr.embed(["real embedding call", "second"])
    assert len(vecs) == 2
    assert all(len(v) == mr.dimension for v in vecs)


def test_missing_task_group_fails_loud_at_composition() -> None:
    """Spec §8 test 9 — a ModelSettings task group with zero deployments raises at build."""
    catalog = ModelCatalogSettings(
        providers=[ProviderRecord(key="azure", kind=ProviderKind.REMOTE, litellm_provider="azure")],
        deployments=[
            ModelDeployment(model_group="gpt-4o", provider_key="azure", model_id="azure/gpt-4o")
        ],
        embedders={
            "minilm_local": WarmLocalConfig(
                model_id="minilm_local", kind=ModelKind.EMBED, model_load_path=MINILM
            )
        },
    )
    # default ModelSettings.answer_model = "gpt-5-chat" has no deployment here
    with pytest.raises(RegistryError, match="no deployment"):
        build_model_router(models=ModelSettings(), catalog=catalog)


def test_unknown_embed_backend_fails_loud() -> None:
    catalog = ModelCatalogSettings(
        providers=[ProviderRecord(key="azure", kind=ProviderKind.REMOTE, litellm_provider="azure")],
        deployments=[
            ModelDeployment(model_group=g, provider_key="azure", model_id=f"azure/{g}")
            for g in {"gpt-5-chat", "gpt-4o", "gpt-4.1-mini"}
        ],
        # embedders empty → embed_backend "minilm_local" cannot resolve
    )
    from mu_engine.providers.embedding import EmbedderConfigError

    with pytest.raises(EmbedderConfigError):
        build_model_router(models=ModelSettings(), catalog=catalog)


def test_credential_hygiene_key_not_in_repr_or_events(
    minilm_catalog: ModelCatalogSettings, default_models: ModelSettings
) -> None:
    """Spec §8 test 11 — the resolved key never appears in repr or any emitted degrade detail."""
    from mu_engine.providers.observability import RecordingDegradeEmitter

    emitter = RecordingDegradeEmitter()
    mr = build_model_router(
        models=default_models,
        catalog=minilm_catalog,
        secret_resolver=_Resolver(),
        degrade_emitter=emitter,
    )
    assert _Resolver.SECRET not in repr(mr)
    # force a degrade and assert the secret is not in the (content-free) detail
    mr._emit_group_unavailable("gpt-4o", RuntimeError("boom"))
    assert all(_Resolver.SECRET not in (e.detail or "") for e in emitter.events)


def test_two_plane_same_class_different_catalog(default_models: ModelSettings) -> None:
    """Spec §8 test 10 — LocalContainer registers a warm mu-local order-1 deployment;
    SharedContainer registers cloud (order-2), empty warm_local. Same class, diff catalog."""
    # ---- LOCAL plane: a warm in-process deployment (backed by the cached MiniLM singleton to
    #      avoid downloading a causal LM; never CALLED — we assert the catalog/orders only) ----
    local_catalog = ModelCatalogSettings(
        providers=[
            ProviderRecord(
                key="mu-local",
                kind=ProviderKind.LOCAL_INPROC,
                litellm_provider="mu-local",
                is_local=True,
            )
        ],
        deployments=[
            ModelDeployment(
                model_group="slm-fast", provider_key="mu-local", model_id="mu-local/phi"
            )
        ],
        warm_local=[
            WarmLocalConfig(model_id="mu-local/phi", kind=ModelKind.EMBED, model_load_path=MINILM)
        ],
        embedders={
            "minilm_local": WarmLocalConfig(
                model_id="minilm_local", kind=ModelKind.EMBED, model_load_path=MINILM
            )
        },
        # only CLASSIFY is used on this thin local plane
        local_capable_tasks=[Task.CLASSIFY, Task.EMBED],
    )
    # Restrict the task table to what the local plane serves by using a mapper the registry sees.
    # build_model_router validates against ALL tasks, so give every task the one local group.
    local_models = ModelSettings(
        answer_model="slm-fast",
        adjudicate_model="slm-fast",
        hard_extract_model="slm-fast",
        routine_extract_model="slm-fast",
        summarize_model="slm-fast",
        classify_model="slm-fast",
        rerank_model="slm-fast",
    )
    local_mr = build_model_router(models=local_models, catalog=local_catalog)
    assert isinstance(local_mr, ModelRouter)
    assert local_mr._registry.is_local_group("slm-fast") is True
    local_orders = {
        r["litellm_params"]["model"]: r["litellm_params"]["order"]
        for r in local_mr._registry.compile_model_list()
    }
    assert local_orders["mu-local/phi"] == 1  # warm local at order 1
    assert len(local_catalog.warm_local) == 1

    # ---- SHARED plane: cloud deployments, empty warm_local ----
    shared_catalog = ModelCatalogSettings(
        providers=[ProviderRecord(key="azure", kind=ProviderKind.REMOTE, litellm_provider="azure")],
        deployments=[
            ModelDeployment(model_group=g, provider_key="azure", model_id=f"azure/{g}")
            for g in {"gpt-5-chat", "gpt-4o", "gpt-4.1-mini"}
        ],
        embedders={
            "minilm_local": WarmLocalConfig(
                model_id="minilm_local", kind=ModelKind.EMBED, model_load_path=MINILM
            )
        },
    )
    shared_mr = build_model_router(models=default_models, catalog=shared_catalog)
    assert isinstance(shared_mr, ModelRouter)  # same class
    assert len(shared_catalog.warm_local) == 0  # server prefers container serving over in-process
    shared_orders = {r["litellm_params"]["order"] for r in shared_mr._registry.compile_model_list()}
    assert shared_orders == {2}  # all cloud → remote tier
