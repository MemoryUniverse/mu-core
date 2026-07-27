"""Shared fixtures for the model-layer unit tests.

Forces HuggingFace/transformers OFFLINE so the real-model tests (MiniLM embedder, warm singleton)
are deterministic and never touch the network — the weights are already in the local HF cache.
These are `unit` tests (mocks permitted for the LLM path; the embedder is REAL, not a mock —
DEV-STANDARDS: a real local-embedder test).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from mu_engine.providers._contracts import Message
from mu_engine.providers.catalog import (
    ModelDeployment,
    ModelKind,
    ProviderKind,
    ProviderRecord,
    WarmLocalConfig,
)
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings
from mu_engine.providers.warm_local import WarmLocalCustomLLM, WarmLocalSingleton

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(autouse=True)
def _clear_singletons() -> Iterator[None]:
    """Each test starts with an empty warm-singleton table (the class cache is process-global)."""
    WarmLocalSingleton.clear_all()
    yield
    WarmLocalSingleton.clear_all()


@pytest.fixture
def embed_cfg() -> WarmLocalConfig:
    return WarmLocalConfig(model_id="minilm_local", kind=ModelKind.EMBED, model_load_path=MINILM)


@pytest.fixture
def minilm_catalog(embed_cfg: WarmLocalConfig) -> ModelCatalogSettings:
    """A catalog whose LLM task groups are wired to a (never-called) remote provider + the real
    local MiniLM embedder as the active embed backend. Mirrors the acceptance case: Azure
    config-wired but NOT reachable/called."""
    return ModelCatalogSettings(
        providers=[
            ProviderRecord(
                key="azure",
                kind=ProviderKind.REMOTE,
                litellm_provider="azure",
                credential_ref="azure_key",
                is_local=False,
            )
        ],
        deployments=[
            ModelDeployment(model_group=g, provider_key="azure", model_id=f"azure/{g}")
            for g in {"gpt-5-chat", "gpt-4o", "gpt-4.1-mini"}
        ],
        embedders={"minilm_local": embed_cfg},
    )


@pytest.fixture
def default_models() -> ModelSettings:
    return ModelSettings()


class StubSingleton:
    """A no-load stand-in for a warm LLM singleton (unit-only): returns a canned completion. Used
    to exercise the litellm CustomLLM wiring WITHOUT downloading a causal LM. The DCL/single-load
    behaviour is verified on the REAL MiniLM embedder in test_warm_singleton.py."""

    def __init__(self, tag: str = "LOCAL") -> None:
        self.tag = tag

    def generate(self, messages: Sequence[Message]) -> str:
        last = messages[-1].content if messages else ""
        return f"{self.tag}:{last}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]


class BoomSingleton:
    """A warm singleton whose generate always fails — to force order-failover / all-down paths."""

    def generate(self, messages: Sequence[Message]) -> str:
        raise RuntimeError("warm local down")


def make_handler(singleton: Any) -> Any:
    return WarmLocalCustomLLM(singleton)
