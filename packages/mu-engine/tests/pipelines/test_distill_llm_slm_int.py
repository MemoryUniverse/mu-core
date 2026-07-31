"""LLM path over the REAL docker SLM — ingest -> DISTILL(LlmFactExtractor) -> recall -> ask, ZERO
mocks (DEV-STANDARDS non-negotiable). The acceptance this file closes: the engine's LLM path
(``LlmFactExtractor`` SPO extraction + the model layer's ANSWER task) was only unit-tested with a
fake ``LLMProviderPort`` (``tests/services/test_extract_llm_unit.py``, ``tests/providers/
test_router_llm.py``) because Azure is unreachable from this box. This test points the SAME
production seams at a REAL local SLM — Ollama container ``mu-dev-slm`` serving ``qwen2.5:0.5b`` on
a GPU, reachable over real HTTP at ``http://127.0.0.1:11435`` — and runs the actual pipeline against
the real mu-dev-* containers (Valkey STM, Qdrant MTM, FalkorDB LTM).

Studied before writing (per project convention: read the model layer, don't invent a new shape):
  * ``mu_engine/providers/catalog.py`` — ``ProviderKind.LOCAL_HTTP`` (:71) is exactly "a localhost
    OpenAI-compatible endpoint (ollama/vllm/tgi/tei)"; ``ModelDeployment.extra_params`` (:94) is the
    documented passthrough seam for litellm_params a `ProviderRecord` doesn't model (e.g. a per-
    deployment `api_key` placeholder for a no-auth local server).
  * ``mu_engine/providers/settings.py`` — ``ModelSettings``/``ModelCatalogSettings`` are themselves
    the "central-config home" for the model layer (the real ``Settings`` tree does not carry this
    subtree yet, tracked seam, :7-14); ``default_local_catalog()`` (:117) is the base catalog every
    plane starts from (real MiniLM embedder wired, no LLM deployments) — this test layers an SLM
    deployment ON TOP of it via ``model_copy``, never touching ``ModelSettings()``'s Azure default.
  * ``mu_engine/providers/model_router.py`` — ``build_model_router`` (:302) is the ONE composition
    entry point; ``ModelRouter.generate(Task.ANSWER, ...)`` (:161) is the already-built "ask/answer"
    ergonomic surface (unit-tested with fakes in ``test_router_llm.py``) this test exercises for
    real.
  * ``mu_engine/providers/litellm_provider.py`` — confirms the router delegates ALL routing to one
    ``litellm.Router``; no custom health/failover code needed for a single-deployment local group.
  * ``mu_engine/services/extract.py`` — ``LlmFactExtractor`` (:404) already implements
    ``FactExtractorPort`` against any ``LLMProviderPort``; ``ModelRouter`` satisfies that Protocol
    structurally, so it drops in unchanged (the seam ``test_extract_llm_unit.py`` locked with a
    ``_FakeProvider``).
  * ``mu_engine/pipelines/distill.py`` + ``tests/pipelines/test_distill_int.py`` — the
    ``DistillPipeline(ltm=..., extractor=..., mtm=...)`` composition this file mirrors, swapping
    ``HeuristicSpoExtractor`` for the real-SLM ``LlmFactExtractor``.
  * ``mu_local/composition.py`` (:149-181) — confirms LOCAL's ``LocalMemory.ask()`` synthesis path
    is NOT implemented yet this phase (``self.llm = None`` unconditionally; ``ask()`` always raises
    ``LlmNotConfiguredError`` even hypothetically, ``local_memory.py:209-228``) — that facade-level
    feature is a separate, larger, out-of-scope gap. This test instead exercises the model layer's
    OWN "ask" primitive (``ModelRouter.generate(Task.ANSWER, ...)``) directly over recalled context,
    the same way ``LocalMemory.consolidate()`` composes ``stm.recent()`` + ``DistillPipeline``
    directly rather than through a not-yet-built orchestrator.
  * ``other_repos/mem0`` fact-retrieval Call-A prompt (already ported at ``extract.py:394-401``) —
    unchanged; this file only swaps the transport (real HTTP to a local model) behind it.

Docker reality (owner-confirmed, do not re-create): ``docker-compose.slm.yml`` (repo root) runs
``mu-dev-slm`` (Ollama, GPU) on host port 11435; the model ``qwen2.5:0.5b`` is already pulled and
serving both the OpenAI-compatible endpoint (``/v1/chat/completions``) and the native Ollama API.
This test wires litellm's ``openai/<model>`` provider against the ``/v1`` endpoint with a
placeholder (non-secret) API key — Ollama's OpenAI-compat shim does not validate it.

Guard, not fake: if ``http://127.0.0.1:11435`` is unreachable the SLM tests in this module SKIP
(env probe, module-scoped) — never a fabricated response. The mu-dev-* store fixtures (inherited
from ``tests/pipelines/conftest.py``) keep the repo's existing convention: if a STORE container is
down, the fixture RAISES (BLOCKED), because those are assumed-up dependencies, not an optional SLM.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from mu_contracts.config import Settings
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.pipelines.concrete.ingest import IngestActivity
from mu_engine.pipelines.distill import DistillPipeline
from mu_engine.pipelines.ledger import InMemoryStageLedger
from mu_engine.platform.adapters.bus_inproc import InprocBus
from mu_engine.platform.clock import SystemClock
from mu_engine.platform.tenancy import DefaultTenancyGuard
from mu_engine.providers._contracts import Message, MessageRole
from mu_engine.providers.catalog import (
    ModelDeployment,
    ModelKind,
    ProviderKind,
    ProviderRecord,
    Task,
    WarmLocalConfig,
)
from mu_engine.providers.embedding import SentenceTransformerEmbedder
from mu_engine.providers.model_router import ModelRouter, build_model_router
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings, default_local_catalog
from mu_engine.services.extract import ExtractionSettings, LlmFactExtractor
from mu_engine.services.ingest import IngestService
from mu_engine.services.recall import (
    InProcessSharedRecall,
    PrincipalAuthorizedIdsResolver,
    RecallAuthorizationFilter,
    RecallQuery,
    RecallService,
    RecallSettings,
    ReciprocalRankFusion,
    ThreeChannelRecallRanker,
)
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.namespace import Namespace

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------------------------
# Test-profile settings — central-config home for THIS test's SLM wiring (DEV-STANDARDS rule 3:
# no hardcoded model id / host / key lives in the test's LOGIC below, only as named field defaults
# here, env-overridable exactly like every other Settings subtree in this repo). Scoped to this
# test module only: it NEVER touches ``ModelSettings()``'s production Azure default (settings.py:53
# -66) — a plane composition root (``mu_local``/``mu_server``) still gets Azure unless it opts in.
# ---------------------------------------------------------------------------------------------
class SlmTestSettings(BaseSettings):
    """The REAL local Ollama SLM (``mu-dev-slm``, host port 11435, model ``qwen2.5:0.5b``) as an
    LLM-task-group test profile. ``env_prefix`` mirrors the repo's ``MU_`` convention so a CI runner
    can override any knob without touching this file (e.g. a different host port)."""

    model_config = SettingsConfigDict(
        env_prefix="MU_TEST_SLM__",
        env_file=(".env", ".env.test"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider_key: str = "slm_test_ollama"
    litellm_provider: str = "openai"  # litellm's OpenAI-compatible provider prefix (catalog.py:70)
    api_base: str = "http://127.0.0.1:11435/v1"  # Ollama's OpenAI-compat shim (docker-compose.slm)
    probe_url: str = "http://127.0.0.1:11435"  # native Ollama root — the reachability probe target
    probe_timeout_s: float = 2.0
    model_id: str = "qwen2.5:0.5b"
    model_group: str = "slm-test"  # the ONE logical group every LLM task routes to in this test
    api_key: str = "sk-mu-local-slm-placeholder"  # NOT a secret — Ollama's shim never validates it
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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _SLM_UP,
        reason=(
            f"mu-dev-slm unreachable at {_SLM_CFG.probe_url} (env probe) — "
            "start it: docker compose -f docker-compose.slm.yml up -d"
        ),
    ),
]


def _build_slm_catalog(cfg: SlmTestSettings) -> tuple[ModelSettings, ModelCatalogSettings]:
    """Layer ONE local-HTTP SLM deployment onto the real ``default_local_catalog()`` base (which
    already carries the real offline MiniLM embedder, untouched) — every LLM task field points at
    the SAME model-group so a single deployment satisfies the registry's per-task validation
    (``registry.py:79-88``)."""
    provider = ProviderRecord(
        key=cfg.provider_key,
        kind=ProviderKind.LOCAL_HTTP,
        litellm_provider=cfg.litellm_provider,
        api_base=cfg.api_base,
        is_local=True,
    )
    deployment = ModelDeployment(
        model_group=cfg.model_group,
        provider_key=cfg.provider_key,
        model_id=f"{cfg.litellm_provider}/{cfg.model_id}",
        kind=ModelKind.LLM,
        extra_params={"api_key": cfg.api_key},  # passthrough seam (catalog.py:94), not a secret
    )
    base = default_local_catalog()
    catalog = base.model_copy(update={"providers": [provider], "deployments": [deployment]})
    models = ModelSettings(
        provider=cfg.provider_key,
        answer_model=cfg.model_group,
        adjudicate_model=cfg.model_group,
        hard_extract_model=cfg.model_group,
        routine_extract_model=cfg.model_group,
        summarize_model=cfg.model_group,
        classify_model=cfg.model_group,
        rerank_model=cfg.model_group,
        max_output_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    return models, catalog


# ---------------------------------------------------------------------------------------------
# Fixtures — real Valkey (STM) + real Qdrant (MTM, via the folder conftest's ``mtm``/``ltm``) +
# real FalkorDB (LTM, via the folder conftest's ``ltm``) + the REAL local MiniLM embedder + the
# REAL SLM-backed ModelRouter. ``ltm``/``mtm``/``qdrant_client``/``make_ns``/``make_item``/
# ``settings``/``uid`` are inherited from ``tests/pipelines/conftest.py`` (test_distill_int.py's
# own fixtures) — unmodified, reused as-is.
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    # REAL MiniLM, loaded ONCE per module (heavy) — same config shape as tests/services/conftest.py.
    return SentenceTransformerEmbedder(
        WarmLocalConfig(model_id="minilm_local", kind=ModelKind.EMBED, model_load_path=_MINILM)
    )


@pytest_asyncio.fixture
async def valkey_client(settings: Settings, uid: str) -> AsyncIterator[Redis]:
    """The REAL Valkey container (``mu-dev-cache``, registered under its own ``(kv, "valkey")``
    key — ``ValkeySettings`` docstring) — the "Valkey" leg of the ingest -> distill -> recall path.
    Tears down every key this test's unique ``uid`` created (isolation by construction, same
    pattern as ``test_local_roundtrip_int.py``'s ``_teardown``)."""
    client: Redis = Redis.from_url(settings.storage.valkey.url, decode_responses=False)
    await client.ping()  # fail-loud if the container is down (BLOCKED, never faked)
    try:
        yield client
    finally:
        keys = [k async for k in client.scan_iter(match=f"*{uid}*".encode())]
        if keys:
            await client.delete(*keys)
        await client.aclose()


@pytest.fixture
def stm(valkey_client: Redis) -> ValkeyStmAdapter:
    return ValkeyStmAdapter(valkey_client)


@pytest.fixture
def mtm(
    qdrant_client: AsyncQdrantClient, embedder: SentenceTransformerEmbedder
) -> QdrantMtmAdapter:
    """OVERRIDES the folder conftest's ``mtm`` (bound to a tiny fixed 8-dim collection for the
    dimension-agnostic DISTILL cross-store tests, ``tests/pipelines/conftest.py:_MTM_TEST_DIM``).
    This test ingests through the REAL local MiniLM embedder, so its Qdrant collection MUST bind
    the embedder's REAL dimension — the same pattern as ``tests/services/conftest.py``'s
    ``ingest_service`` fixture."""
    return QdrantMtmAdapter(qdrant_client, dim=embedder.dimension)


@pytest_asyncio.fixture
async def ingest_service(
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    embedder: SentenceTransformerEmbedder,
) -> IngestService:
    return IngestService(
        stm=stm,
        mtm=mtm,
        embedder=embedder,
        bus=InprocBus(),
        ledger=InMemoryStageLedger(),
        clock=SystemClock(),
    )


@pytest.fixture
def slm_catalog() -> tuple[ModelSettings, ModelCatalogSettings]:
    return _build_slm_catalog(_SLM_CFG)


@pytest.fixture
def router(slm_catalog: tuple[ModelSettings, ModelCatalogSettings]) -> ModelRouter:
    models, catalog = slm_catalog
    return build_model_router(models=models, catalog=catalog)


@pytest.fixture
def extractor(router: ModelRouter) -> LlmFactExtractor:
    extraction_settings = ExtractionSettings(
        max_tokens=_SLM_CFG.max_tokens, temperature=_SLM_CFG.temperature
    )
    return LlmFactExtractor(router, model_group=_SLM_CFG.model_group, settings=extraction_settings)


@pytest.fixture
def recall_service(
    stm: ValkeyStmAdapter,
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    embedder: SentenceTransformerEmbedder,
) -> RecallService:
    fusion = ReciprocalRankFusion()
    recall_settings = RecallSettings()
    clock = SystemClock()
    ranker = ThreeChannelRecallRanker(
        stm=stm,
        mtm=mtm,
        ltm=ltm,
        fusion=fusion,
        settings=recall_settings,
        clock=clock,
        # D1: wire the REAL MiniLM embedder already available in this fixture so
        # `settings.stm_scoring="embed"` (the default) works exactly as it does in production.
        embedder=embedder,
    )
    authz = RecallAuthorizationFilter(
        tenancy=DefaultTenancyGuard(), authorized_ids=PrincipalAuthorizedIdsResolver()
    )
    shared = InProcessSharedRecall(ranker=ranker, embedder=embedder)  # no shared data; PRIVATE-only
    return RecallService(
        embedder=embedder,
        private_ranker=ranker,
        shared_recall=shared,
        authz=authz,
        fusion=fusion,
        settings=recall_settings,
        clock=clock,
    )


# ---------------------------------------------------------------------------------------------
# The acceptance test.
# ---------------------------------------------------------------------------------------------
async def test_ingest_distill_recall_ask_over_real_slm(
    ingest_service: IngestService,
    extractor: LlmFactExtractor,
    router: ModelRouter,
    recall_service: RecallService,
    ltm: FalkorLtmAdapter,
    mtm: QdrantMtmAdapter,
    stm: ValkeyStmAdapter,
    make_ns: Callable[..., Namespace],
) -> None:
    """The full REAL loop: ingest a short conversation -> DISTILL extracts SPO via the SLM over
    HTTP -> facts land in FalkorDB (+ Qdrant/Valkey already hold the raw turns) -> recall fuses all
    three real stores -> the model layer's ANSWER task (the same ``generate()`` ergonomic surface
    ``test_router_llm.py`` unit-tests with fakes) produces a REAL generated answer."""
    ns = make_ns()
    now = datetime(2026, 7, 27, tzinfo=UTC)

    # (1) INGEST a short, unstructured conversation — real Valkey (STM) + real Qdrant (MTM) writes.
    turns = [
        "Ada lives in Paris and works at Acme as a data engineer.",
        "Ada uses Postgres for the analytics warehouse.",
    ]
    for i, text in enumerate(turns):
        result = await ingest_service.remember(
            IngestActivity(
                namespace=ns,
                host="mu-core-int-test",
                session_offset=f"turn-{i}",
                kind="user_message",
                text=text,
                promote=True,
            )
        )
        assert result.promoted, f"turn {i} did not promote STM->MTM"

    # (2) DISTILL over the REAL SLM — the mem0 Call-A fact-retrieval prompt runs against
    #     qwen2.5:0.5b over real HTTP; the returned fact strings are structured via the SAME
    #     deterministic decomposer the heuristic extractor uses (extract.py:428-443).
    window = [scored.item for scored in await stm.recent(ns, limit=10)]
    assert window, "nothing landed in STM to distill"
    pipeline = DistillPipeline(ltm=ltm, extractor=extractor, clock=_FixedClock(now), mtm=mtm)
    report = await pipeline.distill(ns, window)

    print(f"\n[SLM-DISTILL] facts_extracted={report.facts_extracted} added={report.added}")  # noqa: T201
    facts = await ltm.graph_recall(ns, limit=20)
    triples = [(h.item.subject, h.item.predicate, h.item.object) for h in facts]
    print(f"[SLM-DISTILL] SPO triples written to FalkorDB by the REAL SLM: {triples}")  # noqa: T201

    assert report.facts_extracted >= 1, (
        "the REAL SLM extracted zero facts from a two-turn conversation — check the mem0 "
        "Call-A prompt / JSON parsing against qwen2.5:0.5b's actual output"
    )
    assert facts, "DISTILL reported facts but none landed in the real FalkorDB graph"

    # (3) RECALL — federate-live over the REAL Valkey/Qdrant/FalkorDB stores.
    scope = ClientScope(
        principal_id=ns.user,
        org_id=ns.org,
        workspace_id=ns.workspace,
        session_id=ns.session,
        agent_principal_id=ns.user,
    )
    recall_result = await recall_service.recall(
        scope, RecallQuery(namespace=ns, text="What do we know about Ada?", limit=10)
    )
    assert recall_result.items, "federated recall over the real stores returned nothing"
    context = "\n".join(f"- {it.content}" for it in recall_result.items)

    # (4) ASK — the model layer's own ANSWER task (ModelRouter.generate), fed the REAL recalled
    #     context, produces a REAL generated answer from the SLM (not a fabricated string).
    question = "Where does Ada work?"
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="Answer the question using ONLY the given facts. Be concise.",
        ),
        Message(role=MessageRole.USER, content=f"Facts:\n{context}\n\nQuestion: {question}"),
    ]
    completion = await router.generate(Task.ANSWER, messages)
    print(f"[SLM-ASK] question={question!r} answer={completion.text!r}")  # noqa: T201

    assert completion.text.strip(), "the REAL SLM returned an empty ANSWER completion"
    assert completion.model_group == _SLM_CFG.model_group
    # teardown of qdrant/falkordb/valkey is handled by the (folder + this module's) fixtures.


class _FixedClock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at
