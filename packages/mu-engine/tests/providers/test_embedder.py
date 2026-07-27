"""EmbeddingPort seam on the REAL local MiniLM (CANONICAL §6-P5; spec §8 EmbeddingPort suite).

This is the mandatory REAL local-embedder test — NOT a mock. MiniLM is loaded offline from the HF
cache and actually embeds text.
"""

from __future__ import annotations

import pytest

from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.providers.catalog import ModelKind, WarmLocalConfig
from mu_engine.providers.embedding import (
    EmbedderConfigError,
    SentenceTransformerEmbedder,
    build_embedder,
)
from mu_engine.providers.settings import ModelCatalogSettings

pytestmark = pytest.mark.unit

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def _cfg() -> WarmLocalConfig:
    return WarmLocalConfig(model_id="minilm_local", kind=ModelKind.EMBED, model_load_path=MINILM)


async def test_embed_returns_one_vector_per_text_of_dimension() -> None:
    emb = SentenceTransformerEmbedder(_cfg())
    assert isinstance(emb, EmbeddingPort)  # runtime_checkable Protocol conformance
    assert emb.dimension == 384
    assert emb.model_name == MINILM
    texts = ["the cat sat on the mat", "quantum entanglement", "third"]
    vecs = await emb.embed(texts)
    assert len(vecs) == len(texts)
    assert all(len(v) == emb.dimension for v in vecs)
    assert all(isinstance(x, float) for x in vecs[0])


async def test_embed_is_semantically_meaningful() -> None:
    emb = SentenceTransformerEmbedder(_cfg())
    a, b, c = await emb.embed(["I love cats", "I adore kittens", "the stock market crashed"])

    def cos(u: list[float], v: list[float]) -> float:
        return sum(x * y for x, y in zip(u, v, strict=True))  # normalized → dot == cosine

    # semantically-close pair scores higher than the unrelated pair (real embeddings)
    assert cos(a, b) > cos(a, c)


async def test_empty_input_returns_empty() -> None:
    emb = SentenceTransformerEmbedder(_cfg())
    assert await emb.embed([]) == []


def test_build_embedder_resolves_registered_backend() -> None:
    cat = ModelCatalogSettings(embedders={"minilm_local": _cfg()})
    emb = build_embedder("minilm_local", cat)
    assert emb.dimension == 384


def test_build_embedder_unknown_backend_fails_loud() -> None:
    with pytest.raises(EmbedderConfigError, match="not registered"):
        build_embedder("ghost_backend", ModelCatalogSettings())


def test_non_embed_kind_rejected() -> None:
    with pytest.raises(EmbedderConfigError, match="kind=EMBED"):
        SentenceTransformerEmbedder(
            WarmLocalConfig(model_id="x", kind=ModelKind.LLM, model_load_path=MINILM)
        )
