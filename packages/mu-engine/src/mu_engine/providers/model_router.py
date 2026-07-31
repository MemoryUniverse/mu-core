"""L7 — the DI `ModelRouter` façade + composition root (model-layer-spec §2.7, §7).

The ONE public surface of the model layer. Implements the canonical `LLMProviderPort.complete`
(CANONICAL §6-P2) and `EmbeddingPort.embed` (§6-P5), plus the mu-core-internal ergonomic surface
(`generate`/`stream`/`rerank`) and the additive `StreamingCompletionPort`/`RerankProviderPort`
seams (§Contract-changes 2). Local priority is baked into each group's `order:` at compile time
(L4) — the façade does NOT re-decide it. No-silent-fallback (CANONICAL §2): an exhausted group
emits `DegradedModeEntered` and raises `ModelGroupUnavailableError`, never a fabricated output.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from mu_engine.providers._contracts import (
    Chunk,
    Completion,
    DegradedModeEntered,
    DegradeEmitterPort,
    DegradeReason,
    EmbeddingPort,
    Message,
    ModelGroupUnavailableError,
    RerankHit,
    Usage,
    Vector,
)
from mu_engine.providers.catalog import Task
from mu_engine.providers.chunking import LongTextChunker
from mu_engine.providers.embedding import build_embedder
from mu_engine.providers.litellm_provider import LiteLLMRouterAdapter
from mu_engine.providers.local_priority import LocalPriorityPolicy
from mu_engine.providers.observability import LoggingDegradeEmitter, log, traced
from mu_engine.providers.registry import ProviderModelRegistry
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings
from mu_engine.providers.task_map import TaskClassMapper
from mu_engine.providers.warm_local import WarmLocalCustomLLM, WarmLocalSingleton

__all__ = ["ModelRouter", "build_model_router"]

# Constructor DEFAULT only (DEV-STANDARDS rule 3: no hardcoded constant lives in router LOGIC).
# The live value is DI-threaded from the central Settings tree (``RouterSettings
# .default_context_window``) by :func:`build_model_router`; a bare ``ModelRouter(...)`` (e.g. in
# a unit test) still gets a sane, named default rather than a silent unconfigured 0/None.
_DEFAULT_CONTEXT_WINDOW = 128_000

# Mirrors ``chunking._DEFAULT_CHUNK_TOKEN_RATIO`` — the live value is DI-threaded from
# ``EngineSettings.extraction.chunk_token_ratio`` (CONFIG-AND-DATA-FIX-PLAN.md §1.1 Group A) by
# each composition root; a bare :func:`build_model_router` call (unit tests) still gets the same
# named default :class:`~mu_engine.providers.chunking.LongTextChunker` itself falls back to.
_DEFAULT_CHUNK_TOKEN_RATIO = 0.75


class ModelRouter:
    """The DI façade implementing the canonical ports (model-layer-spec §2.7)."""

    def __init__(
        self,
        *,
        router: LiteLLMRouterAdapter,
        task_map: TaskClassMapper,
        chunker: LongTextChunker,
        models: ModelSettings,
        registry: ProviderModelRegistry,
        embedder: EmbeddingPort,
        degrade_emitter: DegradeEmitterPort | None = None,
        default_context_window: int = _DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        self._llm = router
        self._task_map = task_map
        self._chunker = chunker
        self._models = models
        self._registry = registry
        self._embedder = embedder
        self._degrade = degrade_emitter or LoggingDegradeEmitter()
        self._default_context_window = default_context_window
        # EmbeddingPort attrs — read from the adapter, NEVER assumed (memory-layer §8).
        self.model_name: str = embedder.model_name
        self.dimension: int = embedder.dimension

    # ---- canonical LLMProviderPort surface (CANONICAL §6-P2) ------------------------------
    @traced("model.complete")
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: str | None = None,
    ) -> Completion:
        """Single-shot completion against a model-GROUP. Chunks via L6 if the input exceeds the
        group's largest context window; delegates routing/failover to the Router (L2)."""
        max_input = self._max_input_tokens(model)
        if self._chunker.needs_chunking(messages, max_input_tokens=max_input):
            text = await self._chunked_complete(
                messages, model=model, max_tokens=max_tokens, temperature=temperature
            )
            return Completion(text=text, model_group=model, model_id=f"{model}:map_reduce")
        return await self._one_shot(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
        )

    async def _one_shot(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        response_format: str | None,
    ) -> Completion:
        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if response_format is not None:
            kwargs["response_format"] = {"type": response_format}
        try:
            resp = await self._llm.acompletion(
                model=model, messages=self._as_wire(messages), **kwargs
            )
        except Exception as exc:
            self._emit_group_unavailable(model, exc)
            raise ModelGroupUnavailableError(model, cause=type(exc).__name__) from exc
        return self._parse(resp, group=model)

    async def _chunked_complete(
        self, messages: Sequence[Message], *, model: str, max_tokens: int, temperature: float
    ) -> str:
        async def _call(window: list[Message]) -> str:
            c = await self._one_shot(
                window,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=None,
            )
            return c.text

        reduce_group = self._task_map.group_for(self._chunker_reduce_task())

        async def _reduce(msgs: list[Message]) -> str:
            c = await self._one_shot(
                msgs,
                model=reduce_group,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=None,
            )
            return c.text

        return await self._chunker.map_reduce(
            messages,
            max_input_tokens=self._max_input_tokens(model),
            map_call=_call,
            reduce_call=_reduce,
        )

    @staticmethod
    def _chunker_reduce_task() -> Task:
        return Task.SUMMARIZE

    # ---- mu-core ergonomic surface --------------------------------------------------------
    @traced("model.generate")
    async def generate(
        self,
        task: Task,
        messages: Sequence[Message],
        *,
        override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: str | None = None,
    ) -> Completion:
        group = self._task_map.group_for(task, override=override)
        return await self.complete(
            messages,
            model=group,
            max_tokens=max_tokens if max_tokens is not None else self._models.max_output_tokens,
            temperature=temperature if temperature is not None else self._models.temperature,
            response_format=response_format,
        )

    async def stream(
        self,
        task: Task,
        messages: Sequence[Message],
        *,
        override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[Chunk]:
        """Stream a completion as incremental `Chunk`s, ending with `done=True`. Routes the SAME
        group as `generate` (§8 test 7)."""
        group = self._task_map.group_for(task, override=override)
        mt = max_tokens if max_tokens is not None else self._models.max_output_tokens
        temp = temperature if temperature is not None else self._models.temperature
        try:
            wrapper = await self._llm.astreaming(
                model=group, messages=self._as_wire(messages), max_tokens=mt, temperature=temp
            )
        except Exception as exc:
            self._emit_group_unavailable(group, exc)
            raise ModelGroupUnavailableError(group, cause=type(exc).__name__) from exc
        return self._iter_chunks(wrapper)

    @staticmethod
    async def _iter_chunks(wrapper: Any) -> AsyncIterator[Chunk]:
        index = 0
        async for part in wrapper:
            delta = ""
            choices = getattr(part, "choices", None) or []
            if choices:
                d = getattr(choices[0], "delta", None)
                delta = (getattr(d, "content", None) or "") if d is not None else ""
            if delta:
                yield Chunk(delta=delta, index=index, done=False)
                index += 1
        yield Chunk(delta="", index=index, done=True)

    # ---- EmbeddingPort seam (CANONICAL §6-P5) ---------------------------------------------
    @traced("model.embed")
    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        return await self._embedder.embed(texts)

    # ---- RerankProviderPort seam (§Contract-changes 2) ------------------------------------
    @traced("model.rerank")
    async def rerank(
        self, query: str, documents: Sequence[str], *, top_n: int | None = None
    ) -> list[RerankHit]:
        """Execute the rerank model call for the `Task.RERANK` group. The ranking/gating POLICY
        lives in recall (recall §1.5); this only runs the model call. Delegates to the Router's
        rerank; on an exhausted rerank group it co-emits `SURFACE_COMPONENT_DOWN` and raises —
        recall reverts to its floor-protected `merged` order (model-layer-spec §5)."""
        group = self._task_map.group_for(Task.RERANK)
        try:
            resp = await self._llm.router.arerank(  # delegate to litellm rerank
                model=group, query=query, documents=list(documents), top_n=top_n
            )
        except Exception as exc:
            self._degrade.emit(
                DegradedModeEntered(
                    component="reranker",
                    mode="rerank_disabled",
                    reason=DegradeReason.SURFACE_COMPONENT_DOWN,
                    detail=f"group={group}",
                )
            )
            raise ModelGroupUnavailableError(group, cause=type(exc).__name__) from exc
        results = getattr(resp, "results", None) or []
        return [
            RerankHit(index=int(r["index"]), score=float(r.get("relevance_score", 0.0)))
            for r in results
        ]

    # ---- helpers --------------------------------------------------------------------------
    def _max_input_tokens(self, model_group: str) -> int:
        explicit = self._registry.max_input_tokens(model_group)
        if explicit is not None:
            return explicit
        try:
            import litellm

            got = litellm.get_max_tokens(model_group)
            if got:
                return int(got)
        except Exception as exc:  # a logical group id litellm does not know
            log.debug("max_tokens_unknown_group", model_group=model_group, err=type(exc).__name__)
        return self._default_context_window

    @staticmethod
    def _as_wire(messages: Sequence[Message]) -> list[dict[str, str]]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    @staticmethod
    def _parse(resp: Any, *, group: str) -> Completion:
        choice = resp.choices[0]
        text = getattr(choice.message, "content", None) or ""
        usage_obj = getattr(resp, "usage", None)
        usage = Usage(
            prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
        )
        model_id = str(resp._hidden_params.get("model_id") or getattr(resp, "model", group))
        return Completion(
            text=text,
            model_group=group,
            model_id=model_id,
            usage=usage,
            finish_reason=getattr(choice, "finish_reason", None),
        )

    def _emit_group_unavailable(self, group: str, exc: Exception) -> None:
        self._degrade.emit(
            DegradedModeEntered(
                component=f"model:{group}",
                mode="model_unavailable",
                reason=DegradeReason.MODEL_GROUP_UNAVAILABLE,
                detail=f"group={group} cause={type(exc).__name__}",  # content-free
            )
        )


def build_model_router(
    *,
    models: ModelSettings,
    catalog: ModelCatalogSettings,
    degrade_emitter: DegradeEmitterPort | None = None,
    secret_resolver: Any | None = None,
    chunk_token_ratio: float = _DEFAULT_CHUNK_TOKEN_RATIO,
) -> ModelRouter:
    """Composition root (model-layer-spec §7). Constructs the whole layer from settings ONCE.

    Called by each plane's container with that plane's `settings.models` / `settings.model_catalog`
    (CANONICAL §4 — same code, different catalog). Warm weights load at this point (L5). Fails loud
    on an invalid catalog (L1).

    NOTE the spec sketches `build_model_router(settings)`; this takes `models`+`catalog` explicitly
    because the central `Settings` tree does not yet carry those siblings (tracked seam in
    settings.py). The plane root passes `settings.models` / `settings.model_catalog` unchanged.

    `chunk_token_ratio` (CONFIG-AND-DATA-FIX-PLAN.md §1.1 Group A) is the ONE knob threaded into
    the `LongTextChunker` this factory builds; each composition root passes its wired
    `EngineSettings.extraction.chunk_token_ratio` (`get_engine_settings()`), not a bare literal.
    """
    # L5: load warm singletons ONCE, wrap as CustomLLM handlers (one provider prefix per model_id).
    custom_handlers: list[dict[str, Any]] = []
    for w in catalog.warm_local:
        prefix = w.model_id.split("/", 1)[0]
        handler = WarmLocalCustomLLM(WarmLocalSingleton(w))
        custom_handlers.append({"provider": prefix, "custom_handler": handler})

    task_map = TaskClassMapper(models)
    # L1: registries -> compiled model_list (local-priority order: stamped here via L4).
    registry = ProviderModelRegistry(
        catalog.providers,
        catalog.deployments,
        local_policy=LocalPriorityPolicy(
            local_capable_tasks=frozenset(catalog.local_capable_tasks),
            enabled=catalog.local_priority_enabled,
        ),
        task_groups=task_map.task_groups(),
        secret_resolver=secret_resolver,
    )
    model_list = registry.compile_model_list()  # fails loud on invalid catalog

    # embed seam (dedicated EmbeddingPort, §6-P5) — the ONE active embedding backend.
    embedder = build_embedder(models.embed_backend, catalog)

    # L2: one litellm.Router, health/cooldown/fallback DELEGATED into it.
    router = LiteLLMRouterAdapter(
        model_list=model_list,
        router_settings=catalog.router,
        custom_handlers=custom_handlers or None,
    )
    log.info(
        "model_router_built",
        deployments=len(model_list),
        warm=len(custom_handlers),
        embed_backend=models.embed_backend,
        embed_dim=embedder.dimension,
    )
    return ModelRouter(
        router=router,
        task_map=task_map,
        chunker=LongTextChunker(chunk_token_ratio=chunk_token_ratio),
        models=models,
        registry=registry,
        embedder=embedder,
        degrade_emitter=degrade_emitter,
        default_context_window=catalog.router.default_context_window,
    )
