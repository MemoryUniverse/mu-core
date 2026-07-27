"""L2 — `LiteLLMRouterAdapter`: the delegation seam (model-layer-spec §2.2).

Owns ONE `litellm.Router` and delegates ALL routing/health/cooldown/fallback/load-balancing into
it. It decides NONE of those (research §1.3, the single most important fit finding) — we add no
health port, no cooldown store, no failover loop. Thin async pass-throughs only.

Import is lazy (inside `__init__`) so the module imports without litellm present.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mu_engine.providers.settings import RouterSettings

__all__ = ["LiteLLMRouterAdapter"]


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
