"""A minimal OpenAI-compatible chat client for the judge arm — no new dependency.

``httpx`` is already in the resolved environment (the qdrant client depends on it), so the judge
needs nothing installed that a plain ``uv sync`` does not already provide. Deliberately NOT routed
through the engine's ``ModelRouter``: the judge is the MEASURING INSTRUMENT, and running it
through the same router the system under test uses would let a router change silently move the
metric.
"""

from __future__ import annotations

from typing import Any

__all__ = ["OpenAICompatChat"]


class OpenAICompatChat:
    """``ChatPort`` over any ``/v1/chat/completions`` endpoint (ollama, vLLM, OpenAI)."""

    def __init__(
        self, *, base_url: str, model: str, api_key: str = "unused", timeout: float = 120.0
    ) -> None:
        import httpx

        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        payload: Any = response.json()
        return str(payload["choices"][0]["message"]["content"])

    async def aclose(self) -> None:
        await self._client.aclose()
