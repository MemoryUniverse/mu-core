"""``StorageSettings`` / ``BackendChoice`` — the pluggable-backend selector for ``mu-local``.

Adopted from ``research-pluggable-infrastructure.md §3.2`` and the mem0 ``{provider, config}``
config shape (PORT of ``other_repos/mem0/mem0/configs/base.py:30-58`` — a provider key + a
per-backend config dict). One :class:`BackendChoice` per storage ROLE; the composition root
(:mod:`mu_local.composition`) resolves each to a concrete adapter, filling connection knobs from
the central ``Settings`` tree when the choice's ``config`` is empty (DEV-STANDARDS rule 3: no
hardcoded host/port — everything flows from the single env boundary).

This VO lives in ``mu-local`` (not ``mu-contracts``) for now: the spec files it under
``§Contract-changes 2`` as a PROPOSED additive contract (``StorageSettings``/``BackendChoice``/
``STORE_REGISTRY`` become canonical config surface), not yet pinned into CANONICAL. mu-local owns
the ONE default set until the owner pins it (spec §2.2, APPLY-PLAN B-4).

GRAPH IS MANDATORY (CANONICAL storage invariant, spec §3.1): the ``graph`` role must bind a real
graph engine; ``none``/``sqlfold`` are refused at build (``storage.registry`` mandatory-roles).

Phase-0 reality (honest, DEV-STANDARDS "no silent stubs"): the ``STORE_REGISTRY`` shipped so far
binds the mu-dev-container backends (``redis`` KV, ``qdrant`` vector, ``falkordb`` graph,
``sqlite`` or ``postgres`` relational) + the offline ``minilm_local`` embedder. The zero-infra
embedded floor named in the spec (in-proc KV / FAISS / embedded Kùzu) is NOT built yet — selecting
one is a NAMED fail-loud ``BackendUnavailableError`` at the composition root, never a silent
fallback. The default below therefore binds the backends that EXIST and resolves their host ports
from ``Settings`` (``.env.test`` -> the live mu-dev-* stack).

``ModelProfileSettings`` (added 2026-07-27, closes the ``self.llm = None`` composition-root seam):
a single configured LLM/SLM profile mu-local's composition root turns into a REAL
``mu_engine.providers.model_router.ModelRouter`` (the SAME LOCAL_HTTP/OpenAI-compatible catalog
shape the reference integration test builds, ``mu-engine/tests/pipelines/
test_distill_llm_slm_int.py:115-136,163-196``) — one deployment layered onto
``default_local_catalog()``, every task field pointed at the same model-group. ``StorageSettings
.llm=None`` (the default) keeps ``LocalMemory``'s LLM-dependent verbs in heuristic mode,
BYTE-FOR-BYTE the prior behaviour (backward compatible) — no field here is read unless a caller
opts in. Since 2026-08-28 (ENG-115a) ``llm=None`` no longer means the plane has NO model layer:
``LocalContainer.model_router`` is always built, from the SHIPPED multi-provider catalog narrowed
by a credential probe. This profile is the way to PIN every task to one specific deployment.

ENG-118 (the credential rule) is enforced by two defaults below: ``provider="hosted_vllm"`` — the
keyless local litellm prefix — and ``api_key=None`` in place of the former literal
``"sk-mu-local-placeholder"``. Both were measured against a header-recording listener, not
reasoned about; see the field comments.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

__all__ = [
    "BackendChoice",
    "ModelProfileSettings",
    "ObservabilitySettings",
    "StorageSettings",
]


class ObservabilitySettings(BaseModel):
    """Which of the three content-free sinks the embedded LOCAL engine wires (DEV-STANDARDS rule 4).

    Config-sourced (never hardcoded in the container): the composition root reads this and builds
    the real ``Tracer``/``MetricSink``/``AuditLog`` via ``mu_engine.platform.observability`` —
    tracer + metrics + a structured-log audit are ON by default so a real embedded run emits spans,
    latency/error metrics and content-free audit rows on every meaningful op. Flip any off (e.g. in
    a bare unit context) without touching wiring. Mirrors the SHARED-plane ``PlatformSelectors``
    observability flags; folds into ``settings.observability`` when that subtree lands.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    otel_enabled: bool = True
    metrics_enabled: bool = True
    audit_enabled: bool = True


class BackendChoice(BaseModel):
    """A ``{backend, config}`` selection for one storage role (mem0 ``configs/base.py:30-58``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str = Field(min_length=1)  # STORE_REGISTRY key: "redis"|"qdrant"|"falkordb"|"sqlite"…
    config: dict[str, Any] = Field(default_factory=dict)  # per-backend knobs (url/dsn/host/port…)


class ModelProfileSettings(BaseModel):
    """A configured LLM/SLM profile for ``LocalMemory``'s extraction (DISTILL) + ``ask`` synthesis.

    PORT of the reference SLM integration test's model-layer wiring (``mu-engine/tests/pipelines/
    test_distill_llm_slm_int.py`` — ``SlmTestSettings`` :115-136 + ``_build_slm_catalog`` :163-196):
    one ``ProviderKind.LOCAL_HTTP`` OpenAI-compatible deployment, reachable through litellm's
    ``openai/<model>`` provider prefix + ``api_base``, layered onto ``default_local_catalog()``.
    Every field is a NAMED default here (DEV-STANDARDS rule 3) — nothing is hardcoded at the
    composition root; ``base_url``/``model`` are the two knobs a caller MUST supply to point this
    at a real server (e.g. the dev SLM ``http://127.0.0.1:11435/v1`` + ``qwen2.5:0.5b``).

    ``StorageSettings.llm=None`` (the default) never constructs one of these — heuristic mode is
    unchanged (backward compatible).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # litellm's provider prefix. `hosted_vllm` — NOT `openai` — is the default, and the difference
    # was MEASURED against a listener that records headers, not reasoned about (ENG-118):
    #   openai + api_key="sk-mu-local-placeholder" -> Authorization: Bearer sk-mu-local-…
    #   openai + no api_key, OPENAI_API_KEY in env -> Authorization: Bearer <REAL cloud key>
    #   hosted_vllm                                -> Authorization: Bearer fake-api-key
    # i.e. the old default shipped a literal key from source to localhost, and merely deleting that
    # literal made it WORSE — litellm's openai branch falls back to `get_secret("OPENAI_API_KEY")`,
    # so an operator with a cloud key in their shell sent it to 127.0.0.1 on every extraction.
    # `hosted_vllm` reads only HOSTED_VLLM_API_KEY and substitutes "fake-api-key"
    # (litellm/llms/hosted_vllm/chat/transformation.py:125), so the local seam is credential-free
    # on the wire as well as in the table — the same prefix `shipped_catalog.py` uses, and its
    # chat/embedding/rerank handlers all exist.
    provider: str = "hosted_vllm"
    base_url: str  # e.g. "http://127.0.0.1:11435/v1" (Ollama's OpenAI-compat shim) — required
    model: str  # provider-native model id, e.g. "qwen2.5:0.5b" — required
    #: An endpoint that DOES check a key. `None` (the default) is the honest local posture — no
    #: placeholder, because a fake credential in source is a literal key on the wire, not an
    #: absence. When set, the value is handed to the composition root's `SecretSeamResolver` under
    #: `credential_ref` and reaches litellm through `ProviderRecord.credential_ref` — the ONE
    #: sanctioned seam (`catalog.py:77`) — never through `ModelDeployment.extra_params`, which
    #: `model-layer-spec §4` reserves for `api_version` and nothing else.
    api_key: SecretStr | None = None
    #: The NAME this profile's key is registered under in the secret seam. Never a value.
    credential_ref: str = "mu_local_llm_api_key"
    max_tokens: int = 512
    temperature: float = 0.0
    provider_key: str = "mu_local_llm"  # registry key stamped on the ProviderRecord/ModelDeployment
    model_group: str = "mu-local-llm"  # the ONE logical group every LLM task routes to


class StorageSettings(BaseModel):
    """One :class:`BackendChoice` per storage ROLE (research-pluggable-infrastructure §3.2).

    The default binds the backends the phase-0 registry actually ships; empty ``config`` dicts are
    filled from the central ``Settings`` tree at the composition root. ``llm=None`` (the default)
    ⇒ heuristic mode — every LLM-dependent verb refuses loudly (spec §7, T7); a configured
    :class:`ModelProfileSettings` ⇒ the composition root builds a REAL ``ModelRouter`` and those
    verbs run for real (extraction via ``LlmFactExtractor``, ``ask`` via the ANSWER task).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relational: BackendChoice = BackendChoice(backend="sqlite")  # control-plane + history
    kv: BackendChoice = BackendChoice(backend="redis")  # STM (shared with the pipeline ledger)
    vector: BackendChoice = BackendChoice(backend="qdrant")  # MTM dense
    graph: BackendChoice = BackendChoice(backend="falkordb")  # LTM — MANDATORY graph engine
    # ContextRepository (NEW — software-arch spec §5): the provenance-root artifact store
    # PersistRawArtifactStage writes through. NOT mandatory (StoreRegistry.MANDATORY_ROLES is
    # unchanged) — the filesystem floor (content_fs.py) is the only backend registered so far;
    # a future content_git.py (spec l.437) selects in by changing this ONE field.
    artifact: BackendChoice = BackendChoice(backend="filesystem")
    embedding: BackendChoice = BackendChoice(backend="minilm_local")  # REAL offline MiniLM
    llm: ModelProfileSettings | None = None  # None ⇒ heuristic mode (no synthesis); configured ⇒
    #                                          composition root builds a REAL ModelRouter (spec §7,
    #                                          T7 stays honoured — never a silent stub either way)
