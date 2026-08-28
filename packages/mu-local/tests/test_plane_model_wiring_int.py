"""ENG-115a's acceptance, made over the thing the spec actually names: **the container's router**.

MVP-SPEC §5.1 ENG-115a is explicit that the criterion which can fail is *"the same assertion made
against ``LocalContainer(...).model_router``"* — not against ``shipped_catalog()``, and not against
``resolve_plane_model_layer`` either. ``test_plane_model_wiring_unit.py`` proves the resolution
function; this file proves the COMPOSED container, built the way an embedded caller builds it,
against the live mu-dev-* stores. It is an integration test for exactly that reason: constructing
``LocalContainer`` opens real Valkey/Qdrant/FalkorDB clients, and stubbing them would move the
assertion back off the composed object and defeat the point.

Measured at ``dev/mlm-build@eaf6c00`` (before the wiring), from HEAD's own source::

    EngineSettings().model_catalog.deployments == []
    build_model_router(models=EngineSettings().model,
                       catalog=default_local_catalog(EngineSettings().model_catalog))
    -> RegistryError: task 'answer' maps to model-group 'gpt-5-chat' which has no deployment

so this container could not have been built at all with the wired defaults.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from mu_contracts.config import Settings
from mu_engine.config import get_engine_settings
from mu_engine.providers.catalog import ProviderKind, Task
from mu_local.composition import LocalContainer
from mu_local.config import StorageSettings

pytestmark = pytest.mark.integration

#: Every vendor env var the shipped catalog's credential probe consults. Cleared so this asserts
#: the ZERO-CREDENTIAL posture (gate G6) on a developer box that may well have real keys exported.
_VENDOR_ENV = (
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "MOONSHOT_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture
def keyless(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in _VENDOR_ENV:
        monkeypatch.delenv(name, raising=False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the MiniLM embedder is real, from the HF cache
    get_engine_settings.cache_clear()
    yield
    get_engine_settings.cache_clear()


async def test_the_composed_container_resolves_every_task_with_zero_credentials(
    settings: Settings, keyless: None
) -> None:
    """Gate G6 over `LocalContainer(...).model_router` — the object ENG-115a names."""
    container = LocalContainer(StorageSettings(), settings=settings)
    try:
        router = container.model_router
        registry = router._registry
        served = {row["model_name"] for row in registry.compile_model_list()}

        table = router._task_map.task_groups()
        assert set(table) == set(Task), "a Task is missing from the composed task map"
        for task, group in table.items():
            assert group in served, f"task {task.value} -> group {group} has no deployment"

        # ...and it got there with NOTHING configured: no compiled row carries a credential,
        # which is what makes this a FULL-LOCAL claim rather than a cloud one.
        assert all(
            row["litellm_params"].get("api_key") in (None, "")
            for row in registry.compile_model_list()
        )
    finally:
        await container.close()


async def test_the_container_router_is_local_only_and_llm_stays_heuristic(
    settings: Settings, keyless: None
) -> None:
    """The two halves of the wiring decision, asserted separately.

    The router is ALWAYS built (that is the fix), but `self.llm` — and therefore `LocalMemory`'s
    LLM-dependent verbs — is still governed by `StorageSettings.llm`. Arming those silently would
    turn every heuristic-mode caller into a caller of a local endpoint that may not be running.
    """
    container = LocalContainer(StorageSettings(), settings=settings)
    try:
        assert container.model_router is not None
        assert container.llm is None  # heuristic mode, byte-for-byte the prior default

        providers = container.model_router._registry._providers
        assert providers, "the composed registry has no providers"
        assert all(p.kind is ProviderKind.LOCAL_HTTP for p in providers.values())
        assert all(p.credential_ref is None for p in providers.values())
        assert all(p.litellm_provider == "hosted_vllm" for p in providers.values())
    finally:
        await container.close()
