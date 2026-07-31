"""``LocalMemory`` over the REAL docker SLM — closes the composition-root LLM seam
(``mu_local.composition.LocalContainer.llm``, formerly a hardcoded ``None``) end-to-end at the
mu-local FACADE, ZERO mocks (DEV-STANDARDS non-negotiable).

Studied before writing (per project convention: read the model layer, don't invent a new shape):
  * ``mu_engine/tests/pipelines/test_distill_llm_slm_int.py`` — the engine-level acceptance this
    file mirrors at the facade layer: ``SlmTestSettings`` (env-probed, real Ollama ``mu-dev-slm``
    at ``http://127.0.0.1:11435``, model ``qwen2.5:0.5b``) + the LOCAL_HTTP catalog shape
    (``_build_slm_catalog``) that ``mu_local.composition._build_llm_catalog`` now ports verbatim
    behind ``mu_local.config.ModelProfileSettings``.
  * ``mu_local/composition.py`` (:210-222) — a configured ``storage.llm`` now builds a REAL
    ``ModelRouter`` and wires ``LlmFactExtractor`` into DISTILL, replacing ``HeuristicSpoExtractor``
    (the MVP no-LLM default).
  * ``mu_local/local_memory.py`` (``ask``) — with an LLM configured, ``ask()`` now recalls context
    and calls ``ModelRouter.generate(Task.ANSWER, ...)`` instead of always raising.
  * ``packages/mu-local/tests/test_local_roundtrip_int.py`` — the ``mem``/``_teardown``/
    ``_eventually`` fixture patterns this file reuses (real Valkey/Qdrant/FalkorDB isolation by
    unique ``uid``-scoped workspace/namespace).

Guard, not fake: if ``http://127.0.0.1:11435`` is unreachable the SLM test in this module SKIPS
(env probe, module-scoped) — never a fabricated response. If a mu-dev-* STORE container is down,
the ``mem`` fixture RAISES (BLOCKED, never faked) — same convention as every other integration test
in this package.
"""

from __future__ import annotations

import asyncio
import contextlib
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_engine.storage.domain.memory import MemoryTier
from mu_local import LlmNotConfiguredError, LocalMemory, MemoryListView
from mu_local.config import ModelProfileSettings, StorageSettings

_USER = "u1"
_SESSION = "s1"


# ---------------------------------------------------------------------------------------------
# Test-profile settings — central-config home for THIS test's SLM wiring (DEV-STANDARDS rule 3),
# mirrors ``mu_engine/tests/pipelines/test_distill_llm_slm_int.py::SlmTestSettings`` exactly, scoped
# to the mu-local facade test only.
# ---------------------------------------------------------------------------------------------
class SlmTestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MU_TEST_SLM__",
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    base_url: str = "http://127.0.0.1:11435/v1"  # Ollama's OpenAI-compat shim (docker-compose.slm)
    probe_url: str = "http://127.0.0.1:11435"  # native Ollama root — the reachability probe target
    probe_timeout_s: float = 2.0
    model: str = "qwen2.5:0.5b"
    max_tokens: int = 256
    temperature: float = 0.0


def _slm_reachable(cfg: SlmTestSettings) -> bool:
    """A cheap env probe (real HTTP GET, short timeout) — guard, never fake (module docstring)."""
    try:
        with urllib.request.urlopen(cfg.probe_url, timeout=cfg.probe_timeout_s) as resp:  # noqa: S310
            return bool(200 <= int(resp.status) < 300)
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


_SLM_CFG = SlmTestSettings()
_SLM_UP = _slm_reachable(_SLM_CFG)

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def slm_mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """A ``LocalMemory`` configured with the REAL SLM profile — the exact composition-root path
    under test (``LocalContainer`` builds a real ``ModelRouter`` + ``LlmFactExtractor``)."""
    profile = ModelProfileSettings(
        base_url=_SLM_CFG.base_url,
        model=_SLM_CFG.model,
        max_tokens=_SLM_CFG.max_tokens,
        temperature=_SLM_CFG.temperature,
    )
    memory = LocalMemory(
        StorageSettings(llm=profile),
        workspace=f"wsslm{uid}",
        namespace=f"orgslm{uid}",
        settings=settings,
    )
    try:
        yield memory
    finally:
        await _teardown(settings, f"slm{uid}")
        await memory.aclose()


@pytest_asyncio.fixture
async def heuristic_mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    """The unchanged default — ``StorageSettings()`` with ``llm=None`` — backward-compat control."""
    memory = LocalMemory(workspace=f"wsh{uid}", namespace=f"orgh{uid}", settings=settings)
    try:
        yield memory
    finally:
        await _teardown(settings, f"h{uid}")
        await memory.aclose()


async def _teardown(settings: Settings, uid: str) -> None:
    qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
    try:
        for coll in (await qdrant.get_collections()).collections:
            if uid in coll.name:
                with contextlib.suppress(Exception):
                    await qdrant.delete_collection(coll.name)
    finally:
        await qdrant.close()

    db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    try:
        for g in await db.list_graphs():
            name = g.decode() if isinstance(g, bytes) else g
            if uid in name:
                with contextlib.suppress(Exception):
                    await db.select_graph(name).delete()
    finally:
        with contextlib.suppress(Exception):
            await db.connection.aclose()

    redis: Redis = Redis.from_url(settings.storage.cache.url, decode_responses=False)
    try:
        keys = [k async for k in redis.scan_iter(match=f"*{uid}*".encode())]
        if keys:
            await redis.delete(*keys)
    finally:
        await redis.aclose()


async def _eventually(read: Callable[[], Awaitable[MemoryListView]]) -> MemoryListView:
    """Poll a recall until it returns hits (qdrant/falkordb apply writes asynchronously)."""
    last = await read()
    for _ in range(40):  # ~8s ceiling
        if last.items:
            return last
        await asyncio.sleep(0.2)
        last = await read()
    return last


@pytest.mark.skipif(
    not _SLM_UP,
    reason=(
        f"mu-dev-slm unreachable at {_SLM_CFG.probe_url} (env probe) — "
        "start it: docker compose -f docker-compose.slm.yml up -d"
    ),
)
async def test_add_distill_ask_over_real_slm_closes_the_llm_seam(slm_mem: LocalMemory) -> None:
    """The acceptance this file exists for: with an SLM profile configured, ``add`` then
    ``consolidate`` extracts SPO facts VIA THE REAL SLM (not the heuristic decomposer), and ``ask``
    returns a REAL generated answer instead of raising ``LlmNotConfiguredError``."""
    # (1) WRITE — durable STM (redis) + deterministic MTM (qdrant) promotion, same as heuristic
    # mode. REMEDIATION Rank 2 / conformance A6 fix: add() no longer hardcodes promote=True, so
    # this test (proving the MTM write pathway) earns it explicitly via importance_score (default
    # importance 0.5 sits below the 0.6 gate and would stay STM-only).
    r1 = await slm_mem.add(
        "Ada lives in Paris and works at Acme as a data engineer.",
        user=_USER,
        session=_SESSION,
        importance_score=0.9,
    )
    r2 = await slm_mem.add(
        "Ada uses Postgres for the analytics warehouse.",
        user=_USER,
        session=_SESSION,
        importance_score=0.9,
    )
    assert r1.promoted and r2.promoted, "STM->MTM deterministic promotion did not fire"

    # (2) DISTILL over the REAL SLM — LlmFactExtractor runs the mem0 Call-A prompt against
    #     qwen2.5:0.5b over real HTTP; facts land in the real FalkorDB graph.
    report = await slm_mem.consolidate(user=_USER, session=_SESSION)
    print(  # noqa: T201
        f"\n[SLM-DISTILL] facts_extracted={report.facts_extracted} added={report.added} "
        f"superseded={report.superseded}"
    )
    assert report.facts_extracted >= 1, (
        "the REAL SLM extracted zero facts from a two-turn conversation — check the mem0 Call-A "
        "prompt / JSON parsing against qwen2.5:0.5b's actual output"
    )

    ltm_hits = await _eventually(
        lambda: slm_mem.recall(
            "Where does Ada work?", user=_USER, session=_SESSION, tier=MemoryTier.LTM
        )
    )
    triples = [it.content for it in ltm_hits.items]
    print(f"[SLM-DISTILL] SPO facts written to FalkorDB by the REAL SLM: {triples}")  # noqa: T201
    assert ltm_hits.items, "DISTILL reported facts extracted but none landed in the real LTM graph"

    # (3) ASK — the configured ModelRouter's ANSWER task, fed REAL recalled context, produces a
    #     REAL generated answer (no more LlmNotConfiguredError now that an LLM is configured).
    question = "Where does Ada work?"
    answer = await slm_mem.ask(question, user=_USER, session=_SESSION)
    print(f"[SLM-ASK] question={question!r} answer={answer!r}")  # noqa: T201
    assert answer.strip(), "the REAL SLM returned an empty ANSWER completion"


async def test_none_profile_still_uses_heuristic_extraction(heuristic_mem: LocalMemory) -> None:
    """Backward-compat guard: ``StorageSettings(llm=None)`` (the untouched default) still yields
    deterministic heuristic SPO extraction and ``ask`` still refuses loudly — no behaviour change
    for existing no-LLM callers of this facade.

    A6 fix: earn the promotion explicitly (default importance 0.5 no longer promotes)."""
    result = await heuristic_mem.add(
        "Ada lives in Paris", user=_USER, session=_SESSION, importance_score=0.9
    )
    assert result.promoted

    report = await heuristic_mem.consolidate(user=_USER, session=_SESSION)
    assert report.facts_extracted >= 1, "heuristic extractor regressed on a plain SPO sentence"
    assert report.added >= 1

    with pytest.raises(LlmNotConfiguredError):
        await heuristic_mem.ask("Where does Ada live?", user=_USER, session=_SESSION)
