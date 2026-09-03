"""``HttpEmbedder`` over the REAL VM-hosted `all-minilm` endpoint, ZERO mocks (DEV-STANDARDS
non-negotiable for `integration`) — the owner's ask end to end: embedding served by
`mu-dev-slm` (Ollama) on the VM, reached over real HTTP, through the SAME `HttpEmbedder` a
composition root builds via `build_embedder`/`default_local_catalog`.

Guard, not fake (mirrors `tests/pipelines/_slm_support.py`'s own doctrine): if the endpoint is
unreachable, or `all-minilm` was never pulled, the tests in this module SKIP — never a fabricated
response. On the VM (this repo's mandated test runner, `infra/mu-vm/vm_test.sh`) the container is
a same-host sibling reachable directly at `127.0.0.1:11435`, no SSH tunnel required.

What this file proves that the mocked unit tests (`test_http_embedder_unit.py`) cannot: the wire
shape `HttpEmbedder` assumes (OpenAI-format `/embeddings`, `{"data": [{"index", "embedding"}]}`)
is what the real Ollama server actually answers, and the dimension really is 384 — matching the
live Qdrant `__384` collections (the hard constraint this whole feature exists to respect).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from pydantic_settings import BaseSettings, SettingsConfigDict

from mu_engine.providers.catalog import HttpEmbedConfig, ModelKind
from mu_engine.providers.embedding import HttpEmbedder


class HttpEmbedTestSettings(BaseSettings):
    """The REAL VM embed endpoint (mu-dev-slm, `all-minilm`) as a test profile — same
    ``MU_TEST_*`` env-override convention as ``tests/pipelines/_slm_support.SlmTestSettings``, its
    own prefix so overriding one never touches the other."""

    model_config = SettingsConfigDict(
        env_prefix="MU_TEST_HTTP_EMBED__",
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_base: str = "http://127.0.0.1:11435/v1"
    probe_url: str = "http://127.0.0.1:11435"  # native Ollama root — the reachability probe
    probe_timeout_s: float = 2.0
    model: str = "all-minilm"
    dimension: int = 384


def _reachable(cfg: HttpEmbedTestSettings) -> bool:
    """A cheap env probe (real HTTP GET, short timeout) — guard, never fake."""
    try:
        with urllib.request.urlopen(cfg.probe_url, timeout=cfg.probe_timeout_s) as resp:  # noqa: S310
            return bool(200 <= int(resp.status) < 300)
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


_CFG = HttpEmbedTestSettings()
_UP = _reachable(_CFG)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _UP,
        reason=(
            f"mu-dev-slm unreachable at {_CFG.probe_url} (env probe) — "
            "start it: docker compose -f docker-compose.slm.yml up -d"
        ),
    ),
]


@pytest_asyncio.fixture
async def embedder() -> AsyncIterator[HttpEmbedder]:
    cfg = HttpEmbedConfig(
        model_id="minilm_vm_http_it",
        kind=ModelKind.EMBED,
        api_base=_CFG.api_base,
        model=_CFG.model,
        dimension=_CFG.dimension,
        timeout_s=15.0,  # real network + real model inference, not a mock's instant reply
    )
    emb = HttpEmbedder(cfg)
    yield emb
    await emb.aclose()


async def test_real_endpoint_returns_the_configured_dimension(embedder: HttpEmbedder) -> None:
    vecs = await embedder.embed(["hello world", "Memory Universe splits the engine into three."])

    assert len(vecs) == 2
    assert all(len(v) == 384 for v in vecs)
    assert all(isinstance(x, float) for x in vecs[0])


async def test_real_endpoint_is_semantically_meaningful(embedder: HttpEmbedder) -> None:
    """Not just SOME 384 floats — real embeddings, same sanity check as the in-process MiniLM
    test (`test_embedder.py::test_embed_is_semantically_meaningful`)."""
    a, b, c = await embedder.embed(["I love cats", "I adore kittens", "the stock market crashed"])

    def cosine(u: list[float], v: list[float]) -> float:
        dot = sum(x * y for x, y in zip(u, v, strict=True))
        norm_u = sum(x * x for x in u) ** 0.5
        norm_v = sum(y * y for y in v) ** 0.5
        return float(dot / (norm_u * norm_v))

    # semantically-close pair scores higher than the unrelated pair — real embeddings, not
    # normalization artifacts (cosine, not raw dot, since the server's own normalization is not
    # this test's concern).
    assert cosine(a, b) > cosine(a, c)


async def test_real_endpoint_batches_a_larger_request(embedder: HttpEmbedder) -> None:
    """A batch bigger than one item, against the REAL server — the wire round trip the mocked
    unit test's `test_batches_rather_than_one_call_per_item` cannot itself prove is real."""
    texts = [f"memory item number {i}" for i in range(5)]

    vecs = await embedder.embed(texts)

    assert len(vecs) == 5
    assert all(len(v) == 384 for v in vecs)
