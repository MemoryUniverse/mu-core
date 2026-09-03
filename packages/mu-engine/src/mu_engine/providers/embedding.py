"""The dedicated `EmbeddingPort` seam (CANONICAL §6-P5, model-layer-spec §2.7/§7).

`EmbeddingPort` is the PRIMARY embedding path (R19) — a dedicated adapter selected by
`models.embed_backend`, NOT a litellm-routed deployment. Two backends are registered:

  * `minilm_local` (the DEFAULT, in code): an in-process sentence-transformers MiniLM loaded
    ONCE via the L5 `WarmLocalSingleton` (so it participates in the same warm-once discipline as
    the LLM path). `SentenceTransformerEmbedder` implements this.
  * an HTTP-backed variant (`HttpEmbedder`), for an OpenAI-compatible `/embeddings` endpoint —
    e.g. the VM-hosted Ollama `all-minilm` service reached over the laptop's SSH tunnel
    (`infra/mu-vm/vm_reup.sh`). Selecting it is a `catalog.embedders` CONFIG change
    (`ModelCatalogSettings.http_embed_*`, `settings.py`), never a code-level swap: a developer
    with no VM must still get a complete, good, in-process FULL-LOCAL system (CLAUDE.md boundary
    rule) — `minilm_local` stays the default embed_backend in every composition root.

`build_embedder` is the backend registry — a dict factory keyed by `embed_backend`, the pattern
mem0 `LlmFactory.provider_to_class` (`other_repos/mem0/mem0/utils/factory.py:35`) and MemOS
`LLMFactory.backend_to_class` (`other_repos/MemOS/src/memos/llms/factory.py:18-27`) use, here for
embedders. Unknown backend → fail-loud (mirrors MemOS `factory.py:33-34`); it dispatches on the
registered config's TYPE (`WarmLocalConfig` -> in-process, `HttpEmbedConfig` -> HTTP) rather than
on a second parallel enum, so adding a config to `catalog.embedders` is the whole activation.

`HttpEmbedder` fails LOUD, never soft (D1 postmortem: a silent zero vector cost weeks and must
never repeat, including as a silent dimension change): an unreachable endpoint, a non-2xx
response, or a returned vector whose length does not match the configured `dimension` all raise
`HttpEmbedError`/`ProviderError` rather than substituting a zero vector, a different model, or a
truncated/padded vector. Connectivity failures are wrapped in `mu_contracts.domain.errors.
ProviderError` — already RETRYABLE in `mu_engine.platform.exceptions._RETRYABLE_TYPES` — so the
same `retry_io` bounded-retry-with-timeout discipline every store adapter uses
(`mu_engine.platform.decorators.retry_io`) applies here too; a dimension/shape mismatch is NOT
wrapped in `ProviderError` and is therefore TERMINAL (never retried) — retrying a wrong answer
would only get the same wrong answer back faster.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from mu_contracts.domain.errors import ProviderError
from mu_engine.platform.decorators import retry_io
from mu_engine.providers._contracts import EmbeddingPort, ModelLayerError, Vector
from mu_engine.providers.catalog import HttpEmbedConfig, ModelKind, WarmLocalConfig
from mu_engine.providers.settings import ModelCatalogSettings
from mu_engine.providers.warm_local import WarmLocalSingleton

__all__ = [
    "EmbedderConfigError",
    "HttpEmbedError",
    "HttpEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
]


class EmbedderConfigError(ModelLayerError):
    """The selected `embed_backend` has no registered embedder config (fail-loud, §6-P5)."""


class HttpEmbedError(ModelLayerError):
    """The HTTP embed backend answered, but the answer fails validation (§6-P5, D1 postmortem):
    a wrong item count or a vector whose length != the configured `dimension`. TERMINAL — never
    retried (`platform.exceptions._RETRYABLE_TYPES` does not include this type): the endpoint
    already responded, so retrying would not change a wrong shape, only repeat it. Connectivity
    failures (unreachable, timeout, non-2xx) are a DIFFERENT, retryable path — see `ProviderError`
    in this module's docstring.
    """


class SentenceTransformerEmbedder:
    """In-process sentence-transformers embedder implementing `EmbeddingPort` (§6-P5).

    `dimension`/`model_name` are read FROM the loaded model, never assumed (memory-layer §8).
    The heavy `encode` runs off the event loop via `asyncio.to_thread` — DEV-STANDARDS: no
    blocking/sync work in the loop.
    """

    def __init__(self, cfg: WarmLocalConfig) -> None:
        if cfg.kind is not ModelKind.EMBED:
            raise EmbedderConfigError(f"embedder config must be kind=EMBED, got {cfg.kind}")
        self._singleton = WarmLocalSingleton(cfg)  # loads MiniLM ONCE (L5)
        self.model_name: str = cfg.model_load_path
        self.dimension: int = self._singleton.embedding_dimension()

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Return one vector per input text, each of length `self.dimension` (§6-P5 contract)."""
        if not texts:
            return []
        return await asyncio.to_thread(self._singleton.embed, list(texts))


class HttpEmbedder:
    """`EmbeddingPort` over an OpenAI-compatible HTTP `/embeddings` endpoint (§6-P5).

    Built for the VM-hosted Ollama `all-minilm` service (`POST {api_base}/embeddings`, body
    `{"model": ..., "input": [...]}`, OpenAI response shape `{"data": [{"index",
    "embedding"}, ...]}`) but not tied to Ollama specifically — any endpoint speaking that wire
    shape works.

    Fail-loud contract (§6-P5, D1 postmortem — a silent zero vector cost weeks and must never
    repeat):
      * unreachable / timed out / non-2xx -> `ProviderError` (RETRYABLE, bounded by `retry_io`);
      * a response with a different item count than the request, ANY vector whose length !=
        `self.dimension`, or ANY all-zero vector -> `HttpEmbedError` (TERMINAL, never retried,
        never truncated/padded).
    Never a zero vector, never a silently different model, never a silently short vector. The
    all-zero check is not redundant with the width check: D1's vector had the CORRECT width and
    was all zeros, so width alone would have let D1 through again.

    Batches: `texts` is chunked into groups of `cfg.batch_size` and each batch is ONE HTTP call
    (never one call per item). Each batch call is wrapped by `retry_io(timeout_s=cfg.timeout_s)`
    — bounded retry + a per-attempt timeout, the SAME discipline every store adapter in this
    package uses (`mu_engine.platform.decorators.retry_io`, e.g. `QdrantMtmAdapter`).
    """

    def __init__(self, cfg: HttpEmbedConfig, *, client: httpx.AsyncClient | None = None) -> None:
        if cfg.kind is not ModelKind.EMBED:
            raise EmbedderConfigError(f"embedder config must be kind=EMBED, got {cfg.kind}")
        self._cfg = cfg
        # EmbeddingPort attrs — declared from CONFIG here (never inferred from a live probe: the
        # store's expected dimension is the thing this adapter must be held to, not discovered
        # from whatever the endpoint happens to answer today — §6-P5, memory-layer §8).
        self.model_name: str = cfg.model
        self.dimension: int = cfg.dimension
        self._client = client or httpx.AsyncClient(base_url=cfg.api_base, timeout=cfg.timeout_s)
        self._owns_client = client is None
        self._retry = retry_io(timeout_s=cfg.timeout_s)

    async def aclose(self) -> None:
        """Release the owned httpx client (DEV-STANDARDS resource management). A no-op when an
        external client was injected (the caller owns that one's lifecycle, not this adapter)."""
        if self._owns_client:
            await self._client.aclose()

    async def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Return one vector per input text, each of length `self.dimension` (§6-P5 contract)."""
        if not texts:
            return []
        all_texts = list(texts)
        out: list[Vector] = []
        for start in range(0, len(all_texts), self._cfg.batch_size):
            batch = all_texts[start : start + self._cfg.batch_size]
            out.extend(await self._retry(self._embed_batch)(batch))
        return out

    async def _embed_batch(self, batch: list[str]) -> list[Vector]:
        """ONE HTTP call for `batch`. Every failure mode is a typed, named exception — see the
        class docstring's fail-loud contract. Wrapped by `self._retry` in `embed`."""
        try:
            response = await self._client.post(
                "/embeddings", json={"model": self._cfg.model, "input": batch}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"embed backend {self._cfg.model_id!r} at {self._cfg.api_base!r} returned "
                f"{exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            # unreachable / connection refused / timed out — RETRYABLE (ProviderError), never a
            # silent zero vector (D1 postmortem: this is exactly the failure that must be loud).
            raise ProviderError(
                f"embed backend {self._cfg.model_id!r} at {self._cfg.api_base!r} is unreachable: "
                f"{exc!r}"
            ) from exc
        payload: Any = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(batch):
            got = len(data) if isinstance(data, list) else type(data).__name__
            raise HttpEmbedError(
                f"embed backend {self._cfg.model_id!r} returned {got} vectors for "
                f"{len(batch)} inputs — refusing rather than misaligning texts to vectors"
            )

        # OpenAI-format items carry their own "index" — sort defensively rather than assume the
        # endpoint preserved request order (never rely on network/serving-layer ordering).
        def _index_of(item: object) -> int:
            return item.get("index", 0) if isinstance(item, dict) else 0

        ordered = sorted(data, key=_index_of)
        vectors: list[Vector] = []
        for item in ordered:
            vec = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vec, list) or len(vec) != self.dimension:
                got_dim = len(vec) if isinstance(vec, list) else type(vec).__name__
                raise HttpEmbedError(
                    f"embed backend {self._cfg.model_id!r} returned dimension {got_dim}, "
                    f"expected {self.dimension} — refusing to write a vector that would be "
                    "incomparable to the existing corpus (the D1 failure mode, for a live corpus "
                    "rather than an empty one)"
                )
            floats = [float(x) for x in vec]
            # D1 ITSELF, not merely its shape. The width check above catches a DIFFERENT model;
            # it does NOT catch the original D1 vector, which was the RIGHT width and all zeros.
            # Measured 2026-08-31 against a stub returning 384 zeros: the write was accepted and a
            # dead point landed in a live collection, silently unretrievable forever, while this
            # class's own docstring promised "never a zero vector". A degraded/half-initialised
            # serving layer returning zeros is exactly the shape that costs weeks, because every
            # later search simply ranks it last instead of erroring. Terminal, like the width
            # mismatch: the endpoint answered, so a retry returns the same zeros.
            # A real embedding of ANY text (the empty string included — MiniLM still emits a
            # non-zero CLS-pooled vector) is never the zero vector, so this rejects no legitimate
            # answer; `embed([])` short-circuits in `embed` and never reaches here.
            if not any(floats):
                raise HttpEmbedError(
                    f"embed backend {self._cfg.model_id!r} returned an ALL-ZERO "
                    f"{self.dimension}-dim vector — refusing to write the D1 failure mode itself "
                    "(a right-width, meaningless vector that no search would ever surface and no "
                    "error would ever announce)"
                )
            vectors.append(floats)
        return vectors


def build_embedder(embed_backend: str, catalog: ModelCatalogSettings) -> EmbeddingPort:
    """Resolve `models.embed_backend` to a concrete `EmbeddingPort` (§6-P5 seam selection).

    Registry: `catalog.embedders[embed_backend]`'s CONFIG TYPE picks the adapter — a
    `WarmLocalConfig` resolves to the in-process `SentenceTransformerEmbedder` (the default),
    an `HttpEmbedConfig` resolves to `HttpEmbedder` (opt-in, e.g. the VM endpoint). Unknown
    backend key → `EmbedderConfigError` (fail-loud).
    """
    cfg = catalog.embedders.get(embed_backend)
    if cfg is None:
        raise EmbedderConfigError(
            f"embed_backend {embed_backend!r} is not registered in catalog.embedders "
            f"(have: {sorted(catalog.embedders)})"
        )
    if isinstance(cfg, HttpEmbedConfig):
        return HttpEmbedder(cfg)
    return SentenceTransformerEmbedder(cfg)
