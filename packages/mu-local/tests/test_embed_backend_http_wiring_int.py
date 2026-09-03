"""``StorageSettings.embedding.backend`` -> `LocalContainer._build_embedder`, ZERO mocks
(DEV-STANDARDS non-negotiable). Real ``LocalContainer`` over the live mu-dev-* stores.

Proves the SAME thing `test_config_wiring_int.py` proves for `MU_RECALL__WEIGHT_MTM`/
`MU_INGEST__IMPORTANCE_PROMOTE`, for the embedding seam this feature adds: `MU_MODEL_CATALOG__
HTTP_EMBED_API_BASE` (read through a FRESH `EngineSettings()`, never hand-injected) actually
reaches the composed `LocalContainer` and selects the REAL HTTP embedder over the REAL VM
endpoint — the "point the laptop daemon at it" claim, exercised at the actual composition root a
daemon/CLI process builds, not merely at `HttpEmbedder`'s own constructor.

Guard, not fake: if the VM's `mu-dev-slm` `all-minilm` endpoint is unreachable, the HTTP-backend
tests SKIP (env probe, module-scoped) — mirrors `tests/pipelines/_slm_support.py`'s doctrine. The
default-backend test has no such guard: it never leaves the process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from mu_engine.config import get_engine_settings
from mu_engine.providers._contracts import EmbeddingPort
from mu_engine.providers.embedding import HttpEmbedder, SentenceTransformerEmbedder
from mu_local.composition import LocalContainer
from mu_local.config import BackendChoice, StorageSettings

pytestmark = pytest.mark.integration

_PROBE_URL = "http://127.0.0.1:11435"
_API_BASE = "http://127.0.0.1:11435/v1"


def _slm_up() -> bool:
    try:
        with urllib.request.urlopen(_PROBE_URL, timeout=2.0) as resp:  # noqa: S310
            return bool(200 <= int(resp.status) < 300)
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


_UP = _slm_up()

_ENV_KEYS = (
    "MU_MODEL_CATALOG__HTTP_EMBED_API_BASE",
    "MU_MODEL_CATALOG__HTTP_EMBED_MODEL",
    "MU_MODEL_CATALOG__HTTP_EMBED_DIMENSION",
)


@pytest.fixture(autouse=True)
def _clean_engine_settings_env() -> Iterator[None]:
    """Same leak-proofing discipline as ``test_config_wiring_int.py`` — this module is the only
    other one touching ``get_engine_settings``'s process-global ``lru_cache``, and must leave it
    pointed back at the unmodified environment for every test that runs after it."""
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    get_engine_settings.cache_clear()


async def test_default_backend_stays_in_process_with_no_env_set() -> None:
    """CLAUDE.md boundary rule, proven at the real composition root: a box with NO
    ``MU_MODEL_CATALOG__HTTP_EMBED_*`` set — every developer's default — gets the in-process
    MiniLM singleton, never the HTTP backend, regardless of what else is configured."""
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    get_engine_settings.cache_clear()

    container = LocalContainer(StorageSettings())
    try:
        assert isinstance(container.embedder, SentenceTransformerEmbedder)
    finally:
        await container.close()


@pytest.mark.skipif(
    not _UP,
    reason=f"mu-dev-slm unreachable at {_PROBE_URL} (env probe) — "
    "start it: docker compose -f docker-compose.slm.yml up -d",
)
async def test_configured_backend_selects_the_real_http_embedder() -> None:
    """The genuine fix under test: setting the env vars a real config.env carries makes
    ``LocalContainer`` build an `HttpEmbedder` over the REAL VM endpoint — via `choice.backend`
    (`MU_EMBED_BACKEND` at mu-client's layer, `BackendChoice` here), never a code change."""
    os.environ["MU_MODEL_CATALOG__HTTP_EMBED_API_BASE"] = _API_BASE
    os.environ["MU_MODEL_CATALOG__HTTP_EMBED_MODEL"] = "all-minilm"
    os.environ["MU_MODEL_CATALOG__HTTP_EMBED_DIMENSION"] = "384"
    get_engine_settings.cache_clear()

    storage = StorageSettings(embedding=BackendChoice(backend="minilm_vm_http"))
    container = LocalContainer(storage)
    try:
        assert isinstance(container.embedder, HttpEmbedder)
        assert isinstance(container.embedder, EmbeddingPort)
        assert container.embedder.dimension == 384

        vecs = await container.embedder.embed(["hello from the real composition root"])

        assert len(vecs) == 1
        assert len(vecs[0]) == 384
    finally:
        await container.close()  # must close the HttpEmbedder's httpx client too (no leak)


# The probe a FRESH interpreter runs. It must be a subprocess, not an in-process assertion: the
# evidence is "`sentence_transformers`/`torch` were never imported", and those are process-global.
# The first version of this test asserted it in-process behind a skip-if-already-imported guard,
# and in the full mu-local suite it SKIPPED every time — an earlier module had already loaded the
# in-process embedder. A test that only runs when it is run alone is not a regression test.
_NO_MODEL_PROBE = """
import json, sys
from mu_engine.config import get_engine_settings
from mu_local.composition import LocalContainer
from mu_local.config import BackendChoice, StorageSettings

get_engine_settings.cache_clear()
container = LocalContainer(StorageSettings(embedding=BackendChoice(backend="minilm_vm_http")))
print("PROBE" + json.dumps({
    "embedder": type(container.embedder).__name__,
    "embedder_model": container.embedder.model_name,
    "router_model": container.model_router.model_name,
    "router_dim": container.model_router.dimension,
    "sentence_transformers": "sentence_transformers" in sys.modules,
    "torch": "torch" in sys.modules,
}))
"""


@pytest.mark.skipif(
    not _UP,
    reason=f"mu-dev-slm unreachable at {_PROBE_URL} (env probe) — "
    "start it: docker compose -f docker-compose.slm.yml up -d",
)
def test_http_backend_loads_no_in_process_model_anywhere_in_the_container() -> None:
    """THE owner's actual instruction — "embedding must run on the VPS, not on this laptop" — as
    an assertion about the PROCESS, not about a config field.

    A test that only checked ``isinstance(container.embedder, HttpEmbedder)`` passed while the
    laptop still loaded MiniLM: ``build_model_router`` resolved a SECOND ``EmbeddingPort`` from
    ``ModelSettings.embed_backend`` (which ``MU_EMBED_BACKEND`` does not reach), so the daemon
    imported torch + sentence-transformers anyway — measured 2026-08-31 through the real product
    path at **1139 MB RSS and two live SentenceTransformer instances**, for a ``ModelRouter.embed``
    with ZERO callers, while ``model_router_built`` logged ``embed_backend=minilm_local`` and every
    real embed went over HTTP. After the fix, the same path measures **344 MB** and zero. Only a
    process-level assertion catches that class of defect.

    ``sentence_transformers``/``torch`` are LAZY imports (``providers/warm_local.py``: "from
    sentence_transformers import SentenceTransformer  # lazy"), so their ABSENCE from a fresh
    interpreter's ``sys.modules`` after a full container build is direct evidence that no
    in-process model was loaded. Runs in a SUBPROCESS so the evidence survives suite ordering.
    """
    env = {
        **os.environ,
        "MU_MODEL_CATALOG__HTTP_EMBED_API_BASE": _API_BASE,
        "MU_MODEL_CATALOG__HTTP_EMBED_MODEL": "all-minilm",
        "MU_MODEL_CATALOG__HTTP_EMBED_DIMENSION": "384",
    }
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _NO_MODEL_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    line = next(x for x in proc.stdout.splitlines() if x.startswith("PROBE"))
    got = json.loads(line[len("PROBE") :])

    assert got["embedder"] == "HttpEmbedder"
    assert got["embedder_model"] == "all-minilm"
    # the model layer must have been HANDED this embedder, not have resolved a second one
    assert got["router_model"] == "all-minilm"
    assert got["router_dim"] == 384
    # …and nothing anywhere in the build pulled the in-process stack into the process
    assert got["sentence_transformers"] is False
    assert got["torch"] is False
