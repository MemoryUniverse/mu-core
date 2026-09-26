"""AD-328 — a rerank call is never SENT to a group that serves no reranker.

Why this file exists, measured rather than reasoned. AD-326 flipped
``RecallSettings.rerank_enabled`` to ``True`` by default (the owner's ruling). On the VM, with a
configured local model profile, that turned ``packages/mu-local/tests/test_local_llm_slm_int.py::
test_add_distill_ask_over_real_slm_closes_the_llm_seam`` red — and NOT on the rerank itself, which
degrades cleanly (``recall.rerank_unavailable``, ``mode=rerank_disabled``). It died later, on a
perfectly ordinary chat call::

    litellm.types.router.RouterRateLimitError: No deployments available for selected model,
    Try again in 60.0 seconds. Passed model=mu-local-llm

A/B on that one file, same commit: trunk 3d02517 = 2 passed · rerank ON (shipped default) =
1 failed · ``MU_RECALL__RERANK_ENABLED=false`` = 2 passed.

The chain: a configured profile pins EVERY task group — ``rerank_model`` included — to its single
CHAT deployment (``mu_local.composition._profile_models``; that pin is a deliberate invariant with
its own test, and unpinning it makes composition fail loud, because the profile catalog is
``CatalogSource.EMPTY``: MEASURED ``RegistryError: task 'rerank' maps to model-group 'gpt-4.1-mini'
which has no deployment``). So an ``arerank`` went to a chat deployment, failed, and litellm cooled
that DEPLOYMENT down — taking the chat group down with it. The fix is therefore not to unpin and not
to widen the catalog, but to refuse the call: the registry has always known each deployment's
``kind`` and nothing asked it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

import pytest

from mu_engine.providers._contracts import ModelGroupUnavailableError, Vector
from mu_engine.providers.catalog import (
    ModelDeployment,
    ModelKind,
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

pytestmark = pytest.mark.unit

_GROUP = "mu-local-llm"  # the real group name from the measured failure


class _FakeEmbedder:
    model_name = "fake"
    dimension = 3

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [[0.0, 0.0, 0.0] for _ in texts]


class _ArerankSpy:
    """Records whether the REQUEST was issued — the thing the cooldown was collateral of."""

    def __init__(self, response: Any | None = None) -> None:
        self.calls = 0
        self._response = response

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls += 1
        if self._response is None:
            raise AssertionError("a rerank request was issued to a group that serves no reranker")
        return self._response


_OPEN: list[ModelRouter] = []


@pytest.fixture(autouse=True)
async def _close_routers() -> AsyncIterator[None]:
    """Same discipline as ``test_router_llm.py``: a ModelRouter holds a litellm logging worker
    bound to THIS event loop, so it is released on the loop that built it."""
    yield
    while _OPEN:
        await _OPEN.pop().aclose()


def _router_over(
    deployments: list[ModelDeployment], emitter: RecordingDegradeEmitter
) -> ModelRouter:
    providers = [
        ProviderRecord(
            key="mu-local",
            # LOCAL_HTTP with a litellm-known prefix: the measured failure's real shape (ollama on
            # 127.0.0.1) and, unlike LOCAL_INPROC, it needs no CustomLLM handler registered just to
            # BUILD the router — these tests never let a request reach the wire.
            kind=ProviderKind.LOCAL_HTTP,
            litellm_provider="ollama",
            api_base="http://127.0.0.1:11435",
            is_local=True,
        )
    ]
    reg = ProviderModelRegistry(
        providers,
        deployments,
        local_policy=LocalPriorityPolicy(
            local_capable_tasks=frozenset({Task.CLASSIFY}), enabled=True
        ),
        # The profile shape under test: every task pinned at the one group.
        task_groups={Task.CLASSIFY: _GROUP, Task.RERANK: _GROUP},
    )
    models = ModelSettings(classify_model=_GROUP, rerank_model=_GROUP)
    adapter = LiteLLMRouterAdapter(
        model_list=reg.compile_model_list(),
        router_settings=RouterSettings(background_health_checks=False, num_retries=0),
        custom_handlers=[],
    )
    router = ModelRouter(
        router=adapter,
        task_map=TaskClassMapper(models),
        chunker=LongTextChunker(),
        models=models,
        registry=reg,
        embedder=_FakeEmbedder(),
        degrade_emitter=emitter,
    )
    _OPEN.append(router)
    return router


def _chat_only() -> list[ModelDeployment]:
    return [
        ModelDeployment(
            model_group=_GROUP,
            provider_key="mu-local",
            model_id="ollama/qwen2.5:0.5b",
            kind=ModelKind.LLM,
        )
    ]


async def test_a_chat_only_group_is_refused_without_issuing_the_request() -> None:
    """The guard, stated as the failure it prevents: no request leaves, so no deployment can be
    cooled down, and the caller still gets the same typed error recall already handles."""
    emitter = RecordingDegradeEmitter()
    mr = _router_over(_chat_only(), emitter)
    spy = _ArerankSpy()
    mr._llm.router.arerank = spy  # type: ignore[method-assign]

    with pytest.raises(ModelGroupUnavailableError):
        await mr.rerank("q", ["a", "b"])

    assert spy.calls == 0, "the rerank request was issued anyway — the cooldown risk is back"
    reasons = [e.component for e in emitter.events]
    assert "reranker" in reasons, f"no reranker degrade signal was emitted: {emitter.events!r}"
    detail = " ".join(str(getattr(e, "detail", "")) for e in emitter.events)
    assert (
        _GROUP in detail and "rerank-kind" in detail
    ), f"the degrade signal does not say WHICH group and why: {detail!r}"


async def test_a_group_with_a_real_reranker_still_calls_the_model() -> None:
    """The other half — a guard that refused everything would be indistinguishable from deleting
    rerank. A RERANK-kind deployment in the same group must reach the model call.
    """
    emitter = RecordingDegradeEmitter()
    deployments = [
        *_chat_only(),
        ModelDeployment(
            model_group=_GROUP,
            provider_key="mu-local",
            model_id="ollama/bge-reranker-v2-m3",
            kind=ModelKind.RERANK,
        ),
    ]
    mr = _router_over(deployments, emitter)

    class _Resp:
        results: ClassVar[list[dict[str, Any]]] = [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]

    spy = _ArerankSpy(_Resp())
    mr._llm.router.arerank = spy  # type: ignore[method-assign]

    hits = await mr.rerank("q", ["a", "b"], top_n=2)

    assert spy.calls == 1, "the guard swallowed a call to a group that DOES serve rerank"
    assert [(h.index, h.score) for h in hits] == [(1, 0.9), (0, 0.1)]
    assert not [
        e for e in emitter.events if e.component == "reranker"
    ], f"a working rerank emitted a degrade signal: {emitter.events!r}"


def test_serves_rerank_reads_the_kind_not_the_group_name() -> None:
    """The registry predicate on its own: the group NAME is meaningless (the measured failure had
    a group called `mu-local-llm` serving rerank calls), only ``ModelDeployment.kind`` decides.
    """
    providers = [
        ProviderRecord(
            key="mu-local",
            # LOCAL_HTTP with a litellm-known prefix: the measured failure's real shape (ollama on
            # 127.0.0.1) and, unlike LOCAL_INPROC, it needs no CustomLLM handler registered just to
            # BUILD the router — these tests never let a request reach the wire.
            kind=ProviderKind.LOCAL_HTTP,
            litellm_provider="ollama",
            api_base="http://127.0.0.1:11435",
            is_local=True,
        )
    ]
    reg = ProviderModelRegistry(
        providers,
        _chat_only(),
        local_policy=LocalPriorityPolicy(
            local_capable_tasks=frozenset({Task.CLASSIFY}), enabled=True
        ),
        task_groups={Task.CLASSIFY: _GROUP, Task.RERANK: _GROUP},
    )
    assert reg.serves_rerank(_GROUP) is False
    assert reg.serves_rerank("a-group-that-does-not-exist") is False
