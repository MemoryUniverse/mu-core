"""L6 — Long-text map-reduce chunking (model-layer-spec §2.6).

LiteLLM gives the MEASURING primitives (`token_counter`, `get_max_tokens`); the map-reduce POLICY
is ours (research §5.4). This is a COMPLEMENT to LiteLLM's `context_window_fallbacks` (which bumps
one over-long call to a bigger-context deployment, left enabled in L2): chunking handles the
input-exceeds-largest-context case. It runs BEFORE the Router call in `ModelRouter.generate` when
`needs_chunking` is true.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from mu_engine.providers._contracts import Message, MessageRole

__all__ = ["LongTextChunker"]

# The reduce prompt is a template, not free-standing content — it carries no memory content, only
# a fixed instruction. Kept here (not a magic string in a loop) as the ONE reduce instruction.
_REDUCE_INSTRUCTION = (
    "The following are partial results produced from consecutive windows of a single long input. "
    "Combine them into one coherent result that satisfies the original request."
)


class LongTextChunker:
    """Token-budgeted map-reduce over the Router's own completion path (§2.6)."""

    def __init__(self, *, headroom_tokens: int = 512, count_model: str = "gpt-4o") -> None:
        # `count_model` only selects a TOKENIZER for measurement; it is never CALLED. gpt-4o's
        # tiktoken encoding is a safe measurement default and works offline.
        self._headroom = headroom_tokens
        self._count_model = count_model

    def _count(self, messages: Sequence[Message]) -> int:
        import litellm

        return int(
            litellm.token_counter(
                model=self._count_model,
                messages=[{"role": m.role.value, "content": m.content} for m in messages],
            )
        )

    def needs_chunking(self, messages: Sequence[Message], *, max_input_tokens: int) -> bool:
        """True iff the prompt token count exceeds the largest context minus headroom (§2.6)."""
        return self._count(messages) > (max_input_tokens - self._headroom)

    def _window_texts(self, text: str, *, budget_tokens: int) -> list[str]:
        """Split `text` into windows each <= `budget_tokens`, by token ids (encode/decode) with a
        word-split fallback if the tokenizer codec is unavailable."""
        import litellm

        try:
            ids = litellm.encode(model=self._count_model, text=text)
            id_list = list(ids)
            windows = [
                id_list[i : i + budget_tokens] for i in range(0, len(id_list), budget_tokens)
            ]
            return [litellm.decode(model=self._count_model, tokens=w) for w in windows]
        except Exception:
            # Fallback: greedy word-budget (approx 1 token ~= 0.75 words); deterministic.
            words = text.split()
            approx = max(1, int(budget_tokens * 3 // 4))
            return [" ".join(words[i : i + approx]) for i in range(0, len(words), approx)] or [""]

    async def map_reduce(
        self,
        messages: Sequence[Message],
        *,
        max_input_tokens: int,
        map_call: Callable[[list[Message]], Awaitable[str]],
        reduce_call: Callable[[list[Message]], Awaitable[str]],
    ) -> str:
        """MAP each window through `map_call`, then REDUCE with `reduce_call` (§2.6).

        The over-long content is taken from the LAST user message; all other messages
        (system/prior context) are preserved on every window so each MAP call is self-contained.
        """
        budget = max(1, max_input_tokens - self._headroom)
        msg_list = list(messages)
        # locate the last user message (the long input)
        long_idx = next(
            (i for i in range(len(msg_list) - 1, -1, -1) if msg_list[i].role is MessageRole.USER),
            len(msg_list) - 1,
        )
        preamble = msg_list[:long_idx]
        long_msg = msg_list[long_idx]
        windows = self._window_texts(long_msg.content, budget_tokens=budget)

        partials: list[str] = []
        for w in windows:
            window_messages = [*preamble, Message(role=MessageRole.USER, content=w)]
            partials.append(await map_call(window_messages))

        combined = "\n\n".join(f"[window {i + 1}]\n{p}" for i, p in enumerate(partials))
        reduce_messages = [
            Message(role=MessageRole.SYSTEM, content=_REDUCE_INSTRUCTION),
            Message(role=MessageRole.USER, content=combined),
        ]
        return await reduce_call(reduce_messages)
