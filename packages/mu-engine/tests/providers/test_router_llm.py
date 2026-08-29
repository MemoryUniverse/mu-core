"""L1+L2+L5b+L7 — many-to-many failover, all-down degrade, CustomLLM addressable-with-no-endpoint.

Spec §8 tests 1 (failover is LiteLLM's), 2 (local order-1 tried first, fails over), 5 (CustomLLM
addressable as a deployment, NO HTTP endpoint), 8 (exhausted group → degrade + typed raise).

These use REAL `litellm.Router` routing over in-process `CustomLLM` handlers (stub singletons for
the WEIGHT load only — the routing/failover/degrade under test is genuinely LiteLLM's + ours, no
network, no live LLM). Mocks are confined to the weight-load, permitted in a `unit` test.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
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


#: Every :class:`ModelRouter` `_router` below hands out, so the autouse fixture can RELEASE them.
#: A router is not an inert value object: its `LiteLLMRouterAdapter` makes litellm start a
#: process-global logging worker (`while True: await queue.get()`) on the CURRENT event loop the
#: first time a call completes, and pytest-asyncio closes that loop at the end of this test. Left
#: unclosed, the consumer is collected against a dead loop and `asyncio.Queue.get`'s bare `except:`
#: raises ``RuntimeError: Event loop is closed`` — MEASURED as nine unraisable warnings on the VM's
#: full mu-core suite, and reproduced in this directory alone as
#: ``coroutine 'LoggingWorker._worker_loop' was never awaited`` (litellm dropping the previous
#: loop's worker at `logging_worker.py:74-76`). Constructing a resource in a test means releasing
#: it in that test; this list is how these module-level helper calls do that.
_OPEN_ROUTERS: list[ModelRouter] = []


@pytest.fixture(autouse=True)
async def _close_routers() -> AsyncIterator[None]:
    """Closes every router this module built, on the loop that built it, after each test."""
    yield
    while _OPEN_ROUTERS:
        await _OPEN_ROUTERS.pop().aclose()


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
    router = ModelRouter(
        router=adapter,
        task_map=TaskClassMapper(models),
        chunker=LongTextChunker(),
        models=models,
        registry=reg,
        embedder=_FakeEmbedder(),
        degrade_emitter=emitter,
    )
    _OPEN_ROUTERS.append(router)  # released by `_close_routers` — see that list's own comment
    return router


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


# ============================================================================================
# L2 teardown — the model layer must not leave a background consumer on a dying event loop.
#
# THE DEFECT THESE PIN. `LiteLLMRouterAdapter`'s async calls make litellm start a MODULE-GLOBAL
# `LoggingWorker` whose body is `while True: await self._queue.get()`
# (`litellm/litellm_core_utils/logging_worker.py:109-124`), bound to whatever loop was running.
# `ModelRouter` had NO teardown verb and no composition root registered one, so that consumer
# outlived its loop; when it was collected, `coroutine.close()` threw `GeneratorExit` into
# `await getter` inside `asyncio.Queue.get`, whose bare `except:` calls `getter.cancel()` ->
# `loop.call_soon()` -> ``RuntimeError: Event loop is closed`` (`asyncio/queues.py:158-160`).
# MEASURED on the VM's full mu-core suite before the fix: NINE unraisable warnings of exactly
# that shape. They were warnings, never failures, which is why they survived so long — so the
# guarantee is asserted here, where a regression is RED instead of decorative.
# ============================================================================================


def _worker() -> Any:
    """litellm's process-global logging worker. Imported inside the test module (not at import
    time) for the same reason `LiteLLMRouterAdapter` imports litellm lazily."""
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    return GLOBAL_LOGGING_WORKER


async def test_aclose_leaves_no_litellm_worker_pending_on_this_loop() -> None:
    """After `aclose()`, nothing of litellm's queue consumer is left alive on this loop.

    `_bound_loop is get_running_loop()` is the load-bearing half: it proves the worker really was
    (re)initialised on THIS loop during this test — i.e. that the teardown had something to do —
    so the three "is shut down" assertions below cannot pass vacuously on a router that never
    reached litellm at all. litellm ends an async call with a bare
    `asyncio.create_task(_client_async_logging_helper(...))` (`litellm/utils.py:1758-1760`), so
    that initialisation happens AFTER `generate()` returns; `aclose()` yields for it on purpose.
    """
    mr = _router(
        deployments=[
            ModelDeployment(model_group="slm", provider_key="mu-good", model_id="mu-good/x")
        ],
        providers=[
            ProviderRecord(
                key="mu-good",
                kind=ProviderKind.LOCAL_INPROC,
                litellm_provider="mu-good",
                is_local=True,
            )
        ],
        handlers={"mu-good": StubSingleton("LOCAL")},
        task_groups={Task.CLASSIFY: "slm"},
        classify_group="slm",
    )
    assert (await mr.generate(Task.CLASSIFY, _USER)).text == "LOCAL:hi"

    await mr.aclose()

    worker = _worker()
    assert worker._bound_loop is asyncio.get_running_loop(), (
        "litellm's logging worker was never initialised on this test's loop, so this test proved "
        "nothing about teardown — `aclose()` stopped yielding for litellm's deferred "
        "`create_task(_client_async_logging_helper(...))`."
    )
    assert worker._worker_task is None, (
        "aclose() left litellm's queue consumer alive. pytest-asyncio closes this loop next, and "
        "the pending `await queue.get()` becomes `RuntimeError: Event loop is closed` at "
        "collection time — the exact unraisable this fix removed."
    )
    assert not worker._running_tasks, "aclose() left in-flight litellm logging tasks alive"
    assert worker._queue is not None and worker._queue.empty()


async def test_aclose_runs_the_logging_coroutines_litellm_had_queued() -> None:
    """A logging coroutine handed to litellm BEFORE the close is run, not discarded.

    `stop()` alone does not give this: the worker dequeues an item into
    `asyncio.create_task(self._process_log_task(...))` (`logging_worker.py:120-124`) and `stop()`
    cancels exactly those in-flight tasks (`:344-354`), destroying the coroutine unrun — observed
    as ``coroutine 'Logging.async_success_handler' was never awaited``. `clear_queue()` cannot
    cover it either: by then the item is no longer IN the queue. Hence
    `LiteLLMRouterAdapter._flush_logging_queue`'s bounded `Queue.join()` BEFORE the stop.
    """
    mr = _router(
        deployments=[
            ModelDeployment(model_group="slm", provider_key="mu-good", model_id="mu-good/x")
        ],
        providers=[
            ProviderRecord(
                key="mu-good",
                kind=ProviderKind.LOCAL_INPROC,
                litellm_provider="mu-good",
                is_local=True,
            )
        ],
        handlers={"mu-good": StubSingleton("LOCAL")},
        task_groups={Task.CLASSIFY: "slm"},
        classify_group="slm",
    )
    ran = asyncio.Event()

    async def _marker() -> None:
        ran.set()

    # Exactly the call litellm makes on every completed async request (`litellm/utils.py:1089`).
    _worker().ensure_initialized_and_enqueue(_marker())

    await mr.aclose()

    assert ran.is_set(), (
        "aclose() dropped a logging coroutine litellm had already queued — teardown must FLUSH "
        "the worker (bounded Queue.join()) before cancelling it, not cancel it out from under "
        "work it had already taken."
    )
