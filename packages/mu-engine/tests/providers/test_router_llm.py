"""L1+L2+L5b+L7 — many-to-many failover, all-down degrade, CustomLLM addressable-with-no-endpoint.

Spec §8 tests 1 (failover is LiteLLM's), 2 (local order-1 tried first, fails over), 5 (CustomLLM
addressable as a deployment, NO HTTP endpoint), 8 (exhausted group → degrade + typed raise).

These use REAL `litellm.Router` routing over in-process `CustomLLM` handlers (stub singletons for
the WEIGHT load only — the routing/failover/degrade under test is genuinely LiteLLM's + ours, no
network, no live LLM). Mocks are confined to the weight-load, permitted in a `unit` test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from mu_engine.providers._contracts import (
    Message,
    MessageRole,
    ModelGroupUnavailableError,
    Vector,
)
from mu_engine.providers.catalog import (
    ModelDeployment,
    ProviderKind,
    ProviderRecord,
    Task,
)
from mu_engine.providers.chunking import LongTextChunker
from mu_engine.providers.litellm_provider import LiteLLMRouterAdapter
from mu_engine.providers.local_priority import LocalPriorityPolicy
from mu_engine.providers.model_router import ModelRouter
from mu_engine.providers.observability import RecordingDegradeEmitter
from mu_engine.providers.registry import ProviderModelRegistry
from mu_engine.providers.settings import ModelSettings, RouterSettings
from mu_engine.providers.task_map import TaskClassMapper
from mu_engine.providers.warm_local import WarmLocalCustomLLM

from .conftest import BoomSingleton, StubSingleton

pytestmark = pytest.mark.unit

_USER = [Message(role=MessageRole.USER, content="hi")]


class _FakeEmbedder:
    """A trivial EmbeddingPort so ModelRouter can be built for LLM-only tests (real MiniLM is
    exercised in test_embedder.py; no need to reload it here)."""

    model_name = "fake"
    dimension = 3

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def _router(
    *,
    deployments: list[ModelDeployment],
    providers: list[ProviderRecord],
    handlers: dict[str, Any],
    task_groups: dict[Task, str],
    classify_group: str,
    emitter: RecordingDegradeEmitter | None = None,
) -> ModelRouter:
    policy = LocalPriorityPolicy(
        local_capable_tasks=frozenset({Task.CLASSIFY, Task.ROUTINE_EXTRACT}), enabled=True
    )
    reg = ProviderModelRegistry(
        providers, deployments, local_policy=policy, task_groups=task_groups
    )
    custom = [{"provider": k, "custom_handler": WarmLocalCustomLLM(v)} for k, v in handlers.items()]
    adapter = LiteLLMRouterAdapter(
        model_list=reg.compile_model_list(),
        router_settings=RouterSettings(
            background_health_checks=False, num_retries=3, cooldown_s=1, allowed_fails=1
        ),
        custom_handlers=custom,
    )
    models = ModelSettings(classify_model=classify_group)
    return ModelRouter(
        router=adapter,
        task_map=TaskClassMapper(models),
        chunker=LongTextChunker(),
        models=models,
        registry=reg,
        embedder=_FakeEmbedder(),
        degrade_emitter=emitter,
    )


async def test_customllm_addressable_with_no_endpoint() -> None:
    """Spec §8 test 5 — a warm CustomLLM is addressable as an ordinary deployment, no HTTP."""
    mr = _router(
        deployments=[
            ModelDeployment(model_group="slm", provider_key="mu-local", model_id="mu-local/phi")
        ],
        providers=[
            ProviderRecord(
                key="mu-local",
                kind=ProviderKind.LOCAL_INPROC,
                litellm_provider="mu-local",
                is_local=True,
            )
        ],
        handlers={"mu-local": StubSingleton("LOCAL")},
        task_groups={Task.CLASSIFY: "slm"},
        classify_group="slm",
    )
    out = await mr.generate(Task.CLASSIFY, _USER)
    assert out.text == "LOCAL:hi"
    assert out.model_group == "slm"
    assert out.model_id  # a real per-deployment litellm id was assigned (participated in routing)


async def test_order_failover_local_to_remote() -> None:
    """Spec §8 tests 1+2 — order-1 (local) fails, Router fails over to order-2; failover is
    LiteLLM's (we only registered the model_list + orders)."""
    mr = _router(
        deployments=[
            ModelDeployment(model_group="slm", provider_key="mu-boom", model_id="mu-boom/x"),
            ModelDeployment(model_group="slm", provider_key="mu-good", model_id="mu-good/x"),
        ],
        providers=[
            ProviderRecord(
                key="mu-boom",
                kind=ProviderKind.LOCAL_INPROC,
                litellm_provider="mu-boom",
                is_local=True,
            ),  # order 1
            ProviderRecord(
                key="mu-good",
                kind=ProviderKind.LOCAL_INPROC,
                litellm_provider="mu-good",
                is_local=False,
            ),  # order 2
        ],
        handlers={"mu-boom": BoomSingleton(), "mu-good": StubSingleton("REMOTE")},
        task_groups={Task.CLASSIFY: "slm"},
        classify_group="slm",
    )
    out = await mr.generate(Task.CLASSIFY, _USER)
    assert out.text == "REMOTE:hi"  # order-1 boom failed, order-2 good served


async def test_exhausted_group_emits_degrade_and_raises_typed() -> None:
    """Spec §8 test 8 — all deployments fail: one MODEL_GROUP_UNAVAILABLE degrade (content-free)
    + a typed ModelGroupUnavailableError; NEVER a fabricated completion."""
    emitter = RecordingDegradeEmitter()
    mr = _router(
        deployments=[
            ModelDeployment(model_group="slm", provider_key="mu-boom", model_id="mu-boom/x")
        ],
        providers=[
            ProviderRecord(
                key="mu-boom",
                kind=ProviderKind.LOCAL_INPROC,
                litellm_provider="mu-boom",
                is_local=True,
            )
        ],
        handlers={"mu-boom": BoomSingleton()},
        task_groups={Task.CLASSIFY: "slm"},
        classify_group="slm",
        emitter=emitter,
    )
    with pytest.raises(ModelGroupUnavailableError) as ei:
        await mr.generate(Task.CLASSIFY, _USER)
    assert ei.value.model_group == "slm"
    assert len(emitter.events) == 1
    ev = emitter.events[0]
    assert ev.reason.value == "model_group_unavailable"
    assert ev.component == "model:slm"
    # content-free detail: no prompt text ("hi") leaks into the event
    assert "hi" not in (ev.detail or "")
