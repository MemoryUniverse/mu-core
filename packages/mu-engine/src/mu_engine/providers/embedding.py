"""The dedicated `EmbeddingPort` seam (CANONICAL §6-P5, model-layer-spec §2.7/§7).

`EmbeddingPort` is the PRIMARY embedding path (R19) — a dedicated adapter selected by
`models.embed_backend`, NOT a litellm-routed deployment. This phase ships the offline
`minilm_local` backend: an in-process sentence-transformers MiniLM loaded ONCE via the L5
`WarmLocalSingleton` (so it participates in the same warm-once discipline as the LLM path).

`build_embedder` is the backend registry — a dict factory keyed by `embed_backend`, the pattern
mem0 `LlmFactory.provider_to_class` (`other_repos/mem0/mem0/utils/factory.py:35`) and MemOS
`LLMFactory.backend_to_class` (`other_repos/MemOS/src/memos/llms/factory.py:18-27`) use, here for
embedders. Unknown backend → fail-loud (mirrors MemOS `factory.py:33-34`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from mu_engine.providers._contracts import EmbeddingPort, ModelLayerError, Vector
from mu_engine.providers.catalog import ModelKind, WarmLocalConfig
from mu_engine.providers.settings import ModelCatalogSettings
from mu_engine.providers.warm_local import WarmLocalSingleton

__all__ = ["EmbedderConfigError", "SentenceTransformerEmbedder", "build_embedder"]


class EmbedderConfigError(ModelLayerError):
    """The selected `embed_backend` has no registered embedder config (fail-loud, §6-P5)."""


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


def build_embedder(embed_backend: str, catalog: ModelCatalogSettings) -> EmbeddingPort:
    """Resolve `models.embed_backend` to a concrete `EmbeddingPort` (§6-P5 seam selection).

    Registry: for now the only registered kind is a sentence-transformers backend, configured by
    `catalog.embedders[embed_backend]`. Unknown backend → `EmbedderConfigError` (fail-loud).
    """
    cfg = catalog.embedders.get(embed_backend)
    if cfg is None:
        raise EmbedderConfigError(
            f"embed_backend {embed_backend!r} is not registered in catalog.embedders "
            f"(have: {sorted(catalog.embedders)})"
        )
    return SentenceTransformerEmbedder(cfg)
