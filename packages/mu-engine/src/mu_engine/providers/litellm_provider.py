"""L2 — `LiteLLMRouterAdapter`: the delegation seam (model-layer-spec §2.2).

Owns ONE `litellm.Router` and delegates ALL routing/health/cooldown/fallback/load-balancing into
it. It decides NONE of those (research §1.3, the single most important fit finding) — we add no
health port, no cooldown store, no failover loop. Thin async pass-throughs only.

Import is lazy (inside `__init__`) so the module imports without litellm present.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import structlog

from mu_engine.providers.settings import RouterSettings

__all__ = ["LiteLLMRouterAdapter"]

log = structlog.get_logger("mu_engine.providers")


class LiteLLMRouterAdapter:
    """Constructs and owns one `litellm.Router`; exposes thin async pass-throughs (§2.2)."""

    def __init__(
        self,
        *,
        model_list: list[dict[str, Any]],
        router_settings: RouterSettings,
        custom_handlers: list[dict[str, Any]] | None = None,
    ) -> None:
        import litellm as _litellm
        from litellm.utils import custom_llm_setup as _custom_llm_setup

        # litellm ships py.typed but its type surface (Literal routing strategies, message unions,
        # evolving Router overloads; Router is not in its explicit re-exports) is version-volatile;
        # we bind to it through Any-typed handles at THIS boundary only (the pyproject mypy
        # override's intent — loosen at the third-party edge), keeping our public surface typed.
        litellm: Any = _litellm
        custom_llm_setup: Any = _custom_llm_setup
        router_cls: Any = litellm.Router

        # L5 in-process warm handlers — register BEFORE the Router validates the model_list, else
        # get_llm_provider rejects the "mu-local/*" prefix (verified against litellm source:
        # custom providers are only recognised after custom_llm_setup() populates provider_list).
        if custom_handlers:
            litellm.custom_provider_map = custom_handlers
            custom_llm_setup()

        # Background health — proactive, NOT an on-demand probe at routing time (research §1.3).
        litellm.background_health_checks = router_settings.background_health_checks
        litellm.health_check_interval = router_settings.health_interval_s

        self._router: Any = router_cls(
            model_list=model_list,
            num_retries=router_settings.num_retries,
            timeout=router_settings.timeout_s,
            routing_strategy=router_settings.strategy,
            fallbacks=router_settings.fallbacks or None,
            context_window_fallbacks=router_settings.ctx_fallbacks or None,
            cooldown_time=router_settings.cooldown_s,
            allowed_fails=router_settings.allowed_fails,
            enable_pre_call_checks=True,  # honours `order:` + ctx-window pre-checks (§2.4)
        )

    @property
    def router(self) -> Any:
        """The underlying `litellm.Router` (for advanced callers/tests); do not reimplement its
        health/cooldown/fallback around it."""
        return self._router

    async def acompletion(
        self, *, model: str, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> Any:
        return await self._router.acompletion(model=model, messages=list(messages), **kwargs)

    async def aembedding(self, *, model: str, input: Sequence[str], **kwargs: Any) -> Any:
        return await self._router.aembedding(model=model, input=list(input), **kwargs)

    async def astreaming(
        self, *, model: str, messages: Sequence[dict[str, str]], **kwargs: Any
    ) -> Any:
        """Returns the litellm streaming wrapper (an async iterator of chunks)."""
        return await self._router.acompletion(
            model=model, messages=list(messages), stream=True, **kwargs
        )

    async def aclose(self) -> None:
        """Shut down the background machinery this adapter's litellm calls START but never owned.

        **The defect this closes.** Every `acompletion`/`aembedding` above reaches
        `litellm.utils._client_async_logging_helper`, which calls
        `GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(...)`
        (`litellm/utils.py:1087-1090`). That is a MODULE-GLOBAL singleton
        (`litellm/litellm_core_utils/logging_worker.py:522`) whose `start()` does
        `asyncio.create_task(self._worker_loop())` — a `while True: await self._queue.get()`
        consumer bound to whatever event loop happened to be running when OUR first async model
        call completed. Nothing in mu-core ever stopped it: `ModelRouter` had no teardown verb at
        all and neither composition root registered one, so the consumer OUTLIVED the loop that
        created it.

        That is not a cosmetic leak. When the owning loop closes with the task still pending, the
        task is collected, `coroutine.close()` throws `GeneratorExit` into `await getter` inside
        `asyncio.Queue.get`, and that method's BARE `except:` runs `getter.cancel()` —
        `Future.cancel()` → `loop.call_soon()` → `_check_closed()` →
        ``RuntimeError: Event loop is closed`` (`asyncio/queues.py:158-160`,
        `base_events.py:799,545`). MEASURED on the VM's full mu-core suite: nine
        `PytestUnraisableExceptionWarning`s of exactly that shape, plus a
        ``coroutine 'Logging.async_success_handler' was never awaited`` — i.e. audit/logging
        coroutines we had already handed to litellm were silently DROPPED, not run. In the server
        the same shape means uvicorn's shutdown can discard queued provider-logging work while
        `EngineContainer.close()` reports a clean teardown.

        **Why `stop()` and not a broad suppress.** `LoggingWorker.stop()`
        (`logging_worker.py:338-358`) is litellm's own supported shutdown: it `cancel()`s the
        worker task AND every in-flight logging task, then `await asyncio.gather(...)` — i.e. it
        cancels and AWAITS the consumer on the loop the consumer actually lives on, which is the
        one property that makes the teardown correct. It is also restartable by construction
        (`start()` re-creates the task whenever `_worker_task` is `None`/done), so stopping the
        process-global worker when one router closes cannot strand a second router that is still
        in use — its next call starts a fresh worker on ITS loop.

        **`stop()` alone is NOT enough, and that was measured, not assumed.** With only `stop()`
        the `_worker_loop` unraisable disappeared but a second warning stayed:
        ``coroutine 'Logging.async_success_handler' was never awaited``. Cause: `enqueue()` runs
        `put_nowait` immediately after `start()`, so on a short-lived loop the worker task is
        cancelled BEFORE it is ever scheduled — its `except asyncio.CancelledError: await
        self.clear_queue()` branch (`logging_worker.py:130-133`) never executes, and the queued
        coroutine is dropped with the queue. So this calls `clear_queue()` EXPLICITLY afterwards
        (`logging_worker.py:373-403`): it drains the queue and AWAITS each pending logging
        coroutine, bounded by litellm's own `MAX_ITERATIONS_TO_CLEAR_QUEUE`/
        `MAX_TIME_TO_CLEAR_QUEUE` (200 items / 5s, `litellm/constants.py:420-421`), so a wedged
        callback cannot hang a container teardown. Draining AFTER the stop, not before, is
        deliberate: with the consumer already cancelled nothing can re-enter the queue mid-drain,
        which makes the flush deterministic instead of racing the worker.

        `discard()` (`litellm/router.py:732-748`) then unhooks THIS Router instance's callbacks
        from litellm's process-global callback lists — the other half of "a router that can be
        constructed must be destructible". ORDER IS LOAD-BEARING: stop and drain FIRST, because
        the coroutines the drain runs are the very success/failure callbacks `discard()`
        unregisters; unhooking first would make the flush a no-op.

        **The post-call logging task is FIRE-AND-FORGET, so the close has to yield first.**
        `litellm/utils.py:1758-1760` ends an async call with a bare
        `asyncio.create_task(_client_async_logging_helper(...))` — the enqueue (and therefore the
        worker's `start()`) happens on a LATER loop iteration, after our `await acompletion(...)`
        has already returned. MEASURED, outside pytest, with a two-loop harness that mimics
        pytest-asyncio's loop-per-test: closing immediately found `_worker_task is None`, did
        nothing, and the worker was then started on the loop we were about to abandon — the leak
        survived the fix. `asyncio.sleep(0)` yields exactly one iteration, which is all those
        already-scheduled tasks need to reach the enqueue (`_client_async_logging_helper` awaits
        nothing before it). Two yields, not one, because the Router wraps a second litellm call
        that schedules its own helper.

        **A worker bound to ANOTHER loop is left alone, loudly.** `stop()` `gather`s the tasks it
        cancels, and a `_GatheringFuture` built from another loop's tasks raises
        ``RuntimeError: ... attached to a different loop`` — measured in the same harness. There is
        no way to cancel-and-await a task from a loop that is not its own, so the honest action is
        to skip that half and SAY SO (a content-free warning), not to pretend the teardown
        happened. After this change nothing in mu-core should reach that branch: both composition
        roots close the router before their loop goes away.

        Idempotent: `stop()` returns immediately when no worker was ever launched (the
        `llm.enabled=False` plane never makes a call, so it never starts one), `clear_queue()`
        returns immediately on a `None`/empty queue, and `discard()`'s removals are no-ops the
        second time.
        """
        # Lazy, like __init__'s: this module must import without litellm present. No try/except
        # around it — a pinned litellm that has lost `stop()` is a fail-loud upgrade break, not a
        # thing to silently skip (DEV-STANDARDS rule 8: no silent fallback).
        from litellm.litellm_core_utils.logging_worker import (
            GLOBAL_LOGGING_WORKER as _GLOBAL_LOGGING_WORKER,
        )

        # Same Any-typed-handle discipline as __init__ above: `clear_queue`/`_bound_loop` carry no
        # annotations in this litellm version (`no-untyped-call` under --strict), so the
        # third-party edge is loosened HERE and nowhere else — our own surface stays typed.
        worker: Any = _GLOBAL_LOGGING_WORKER

        # (1) let litellm's already-scheduled fire-and-forget logging tasks reach the queue.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # (2) FLUSH, then cancel-and-await the consumer — but ONLY if it lives on the loop we are
        #     closing from.
        # `_bound_loop` is private, and reached deliberately: litellm exposes no public accessor
        # for "which loop is this worker on", and the answer decides whether `stop()` is legal.
        bound = worker._bound_loop
        if bound is not None and bound is not asyncio.get_running_loop():
            log.warning(
                "litellm_logging_worker_bound_to_foreign_loop",
                detail="skipped flush/stop; a task cannot be cancelled from another loop",
            )
        else:
            await self._flush_logging_queue(worker)
            await worker.stop()
            await worker.clear_queue()

        # (3) unhook THIS Router's callbacks from litellm's process-global callback lists.
        self._router.discard()

    @staticmethod
    async def _flush_logging_queue(worker: Any) -> None:
        """Wait for the worker to FINISH the items it has already taken, before anything cancels it.

        Why this exists, measured rather than assumed. With only `stop()` + `clear_queue()` the
        drain found an EMPTY queue and a logging coroutine was still reported
        ``never awaited``: the worker had already dequeued the item into
        `asyncio.create_task(self._process_log_task(...))` (`logging_worker.py:120-124`), so it was
        no longer IN the queue but had not yet run — and `stop()` cancels exactly those in-flight
        tasks (`:344-354`), destroying the coroutine unrun. `clear_queue()` cannot help: it only
        looks at the queue.

        `Queue.join()` is the right primitive because `_process_log_task` calls `task_done()` in
        its `finally` (`:105`), so the join completes precisely when every enqueued item has been
        processed. It is bounded by litellm's OWN drain budget rather than a new magic number
        (`MAX_TIME_TO_CLEAR_QUEUE`, `litellm/constants.py:421`, env-overridable) — DEV-STANDARDS:
        a timeout on every wait, and no second source of truth for one budget. A timeout is
        reported, never swallowed: a logging callback that will not finish is a real condition an
        operator should see, and teardown continues either way (`stop()` below still cancels it).
        """
        queue = worker._queue
        if queue is None or (queue.empty() and not worker._running_tasks):
            return
        from litellm.constants import MAX_TIME_TO_CLEAR_QUEUE

        try:
            await asyncio.wait_for(queue.join(), timeout=float(MAX_TIME_TO_CLEAR_QUEUE))
        except TimeoutError:
            log.warning(
                "litellm_logging_worker_flush_timeout",  # content-free: no payload, ids or text
                timeout_s=float(MAX_TIME_TO_CLEAR_QUEUE),
            )
