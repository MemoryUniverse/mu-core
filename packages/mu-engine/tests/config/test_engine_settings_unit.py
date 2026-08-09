"""``EngineSettings`` unit tests (CONFIG-AND-DATA-FIX-PLAN.md PART 1 §1.2 stage C0).

Two obligations (plan's C0 row):
  (a) ``EngineSettings()`` defaults are field-equal to constructing each bare orphan class
      directly — NO behavior change from this aggregator landing.
  (b) an env override, through the ``MU_`` prefix + ``__`` nested delimiter, actually lands on
      ``get_engine_settings()`` — proving the knob is genuinely reachable from the environment,
      not just "the field exists" (the ``recency_floor_limit=10`` regression this plan starts
      from was exactly a field that existed but could never be reached).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mu_engine.config.engine_settings import EngineSettings, get_engine_settings
from mu_engine.lifecycle.settings import LifecycleSettings
from mu_engine.pipelines.distill import DistillSettings
from mu_engine.pipelines.ledger import LedgerSettings
from mu_engine.platform.settings import ObservabilitySettings, RetrySettings
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings
from mu_engine.services.extract import ExtractionSettings
from mu_engine.services.recall.dto import RecallSettings
from mu_engine.services.settings import IngestSettings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_engine_settings_cache() -> Iterator[None]:
    """``get_engine_settings`` is ``@lru_cache`` (one read per process, by design) — a test that
    mutates ``os.environ`` must clear it first/after or every subsequent test (in this file or
    any other) would keep observing the FIRST test's env, silently. Runs before AND after every
    test in this module (autouse), mirroring the same discipline ``get_settings`` already
    implies."""
    get_engine_settings.cache_clear()
    yield
    get_engine_settings.cache_clear()


# ------------------------------------------------------------------------------------ (a) no drift
def test_engine_settings_defaults_match_each_bare_subtree_exactly() -> None:
    """No behavior change (plan C0): every mounted field, constructed via the aggregator, is
    field-equal to constructing the SAME class bare — the exact "no re-shape" guarantee every
    subtree's own docstring promises ("the composition root wires ``settings.X`` when it lands —
    no re-shape")."""
    s = EngineSettings()

    assert s.recall == RecallSettings()
    assert s.distill == DistillSettings()
    assert s.extraction == ExtractionSettings()
    assert s.ingest == IngestSettings()
    assert s.lifecycle == LifecycleSettings()
    assert s.model == ModelSettings()
    assert s.model_catalog == ModelCatalogSettings()
    assert s.retry == RetrySettings()
    assert s.observability == ObservabilitySettings()
    assert s.ledger == LedgerSettings()


def test_engine_settings_defaults_are_independent_instances() -> None:
    """``default_factory`` (not a shared module-level constant) — two independent constructions
    never share a mutable subtree instance."""
    a, b = EngineSettings(), EngineSettings()
    assert a.recall is not b.recall
    assert a.recall == b.recall  # still field-equal, just not the same object


# ----------------------------------------------------------------------------------- (b) env lands
def test_env_override_lands_on_recall_weight_via_nested_delimiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MU_RECALL__WEIGHT_MTM", "1.4")

    settings = get_engine_settings()

    assert settings.recall.weight_mtm == 1.4
    # sibling fields on the SAME subtree stay at their bare default — the override is scoped to
    # the one field named, not a wholesale replacement of the subtree.
    assert settings.recall.weight_stm == RecallSettings().weight_stm
    assert settings.recall.rrf_k == RecallSettings().rrf_k


def test_env_override_lands_on_lifecycle_promotion_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MU_LIFECYCLE__PROMOTE_STM_MTM", "0.65")

    settings = get_engine_settings()

    assert settings.lifecycle.promote_stm_mtm == 0.65
    assert settings.lifecycle.promote_mtm_ltm == LifecycleSettings().promote_mtm_ltm


def test_env_override_lands_on_model_max_output_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MU_MODEL__MAX_OUTPUT_TOKENS", "2048")

    settings = get_engine_settings()

    assert settings.model.max_output_tokens == 2048
    assert settings.model.temperature == ModelSettings().temperature


def test_all_three_group_a_overrides_land_together_in_one_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed-together case the task asks for explicitly: three DIFFERENT subtrees,
    THREE-level-deep nesting for the deepest one (``lifecycle.salience.w_recency`` shape), all
    reachable from ONE flat env namespace in the SAME process."""
    monkeypatch.setenv("MU_RECALL__WEIGHT_MTM", "1.4")
    monkeypatch.setenv("MU_LIFECYCLE__PROMOTE_STM_MTM", "0.65")
    monkeypatch.setenv("MU_MODEL__MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("MU_LIFECYCLE__SALIENCE__W_RECENCY", "0.9")

    settings = get_engine_settings()

    assert settings.recall.weight_mtm == 1.4
    assert settings.lifecycle.promote_stm_mtm == 0.65
    assert settings.model.max_output_tokens == 2048
    assert settings.lifecycle.salience.w_recency == 0.9  # 3-deep nesting also resolves


def test_no_override_reproduces_bare_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt-and-suspenders: with NOTHING set, ``get_engine_settings()`` after a cache-clear is
    still field-equal to a bare aggregator construction (the cache itself introduces no drift)."""
    monkeypatch.delenv("MU_RECALL__WEIGHT_MTM", raising=False)

    settings = get_engine_settings()

    assert settings.recall == RecallSettings()
    assert settings.lifecycle == LifecycleSettings()


# --------------------------------------------------------------- C2/C3 (Group B/C) env overrides
def test_env_override_lands_on_extraction_min_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Group B: ``ExtractionSettings.min_tokens`` (services/extract.py) — was NEVER overridden
    (only ``max_tokens``/``temperature`` were threaded pre-C2)."""
    monkeypatch.setenv("MU_EXTRACTION__MIN_TOKENS", "7")

    settings = get_engine_settings()

    assert settings.extraction.min_tokens == 7
    assert settings.extraction.max_tokens == ExtractionSettings().max_tokens


def test_env_override_lands_on_model_embed_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Group B: ``ModelSettings.answer_model``/``embed_backend``/``embed_model``/``provider``
    (providers/settings.py) — mounted at C0, now proven env-overridable."""
    monkeypatch.setenv("MU_MODEL__ANSWER_MODEL", "gpt-custom")
    monkeypatch.setenv("MU_MODEL__EMBED_BACKEND", "custom_local")
    monkeypatch.setenv("MU_MODEL__EMBED_MODEL", "custom-embed-id")
    monkeypatch.setenv("MU_MODEL__PROVIDER", "custom_provider")

    settings = get_engine_settings()

    assert settings.model.answer_model == "gpt-custom"
    assert settings.model.embed_backend == "custom_local"
    assert settings.model.embed_model == "custom-embed-id"
    assert settings.model.provider == "custom_provider"


def test_env_override_lands_on_model_catalog_embedder_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group B: the FORMER stray module consts ``_DEFAULT_EMBED_BACKEND``/``_DEFAULT_MINILM_PATH``
    (providers/settings.py:39-40) — now ``ModelCatalogSettings.default_embed_backend``/
    ``default_minilm_path`` fields, env-overridable."""
    monkeypatch.setenv("MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND", "custom_backend")
    monkeypatch.setenv("MU_MODEL_CATALOG__DEFAULT_MINILM_PATH", "org/custom-minilm")

    settings = get_engine_settings()

    assert settings.model_catalog.default_embed_backend == "custom_backend"
    assert settings.model_catalog.default_minilm_path == "org/custom-minilm"


def test_env_override_lands_on_model_catalog_router_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group C: ``RouterSettings`` sub-fields (``num_retries``/``cooldown_s``/``allowed_fails``/
    ``health_interval_s``/``default_context_window``) under ``EngineSettings.model_catalog.router``
    — 3-deep nesting (``MU_MODEL_CATALOG__ROUTER__…``), never overridden before C3."""
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__NUM_RETRIES", "9")
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__COOLDOWN_S", "12.5")
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__ALLOWED_FAILS", "1")
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__HEALTH_INTERVAL_S", "60")
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__DEFAULT_CONTEXT_WINDOW", "4242")

    settings = get_engine_settings()

    router = settings.model_catalog.router
    assert router.num_retries == 9
    assert router.cooldown_s == pytest.approx(12.5)
    assert router.allowed_fails == 1
    assert router.health_interval_s == 60
    assert router.default_context_window == 4242


def test_env_override_lands_on_retry_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Group C: ``RetrySettings`` (platform/settings.py) — the module-level ``_DEFAULT_RETRY``
    singleton used to be frozen at IMPORT time; now reachable via ``EngineSettings.retry``."""
    monkeypatch.setenv("MU_RETRY__MAX_ATTEMPTS", "9")
    monkeypatch.setenv("MU_RETRY__BASE_DELAY_S", "0.01")
    monkeypatch.setenv("MU_RETRY__MAX_DELAY_S", "1.5")

    settings = get_engine_settings()

    assert settings.retry.max_attempts == 9
    assert settings.retry.base_delay_s == pytest.approx(0.01)
    assert settings.retry.max_delay_s == pytest.approx(1.5)


def test_env_override_lands_on_observability_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Group C: ``ObservabilitySettings.durable_audit_queue_max`` (platform/settings.py)."""
    monkeypatch.setenv("MU_OBSERVABILITY__DURABLE_AUDIT_QUEUE_MAX", "4096")

    settings = get_engine_settings()

    assert settings.observability.durable_audit_queue_max == 4096


def test_env_override_lands_on_ledger_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Group C: ``LedgerSettings.redis_io_timeout_s`` (pipelines/ledger.py)."""
    monkeypatch.setenv("MU_LEDGER__REDIS_IO_TIMEOUT_S", "12.0")

    settings = get_engine_settings()

    assert settings.ledger.redis_io_timeout_s == pytest.approx(12.0)


def test_env_override_lands_on_lifecycle_cadence_and_adjudicator_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group C: the FULL ``LifecycleSettings`` cadence (``maintenance_interval_s``/
    ``session_idle_s``/``batch_size``/…) + the adjudicator sub-knobs C3 added
    (``adjudication_budget_per_sweep``/``adjudication_degrade_threshold_s`` already existed;
    ``adjudicator_max_tokens``/``adjudicator_temperature`` are NEW — every
    ``ConflictAdjudicatorSettings`` field now has a home on this ONE ``MU_LIFECYCLE__…`` tree via
    ``conflict_adjudicator_settings_from_lifecycle``)."""
    monkeypatch.setenv("MU_LIFECYCLE__MAINTENANCE_INTERVAL_S", "120")
    monkeypatch.setenv("MU_LIFECYCLE__SESSION_IDLE_S", "30")
    monkeypatch.setenv("MU_LIFECYCLE__BATCH_SIZE", "5")
    monkeypatch.setenv("MU_LIFECYCLE__ADJUDICATION_BUDGET_PER_SWEEP", "17")
    monkeypatch.setenv("MU_LIFECYCLE__ADJUDICATION_DEGRADE_THRESHOLD_S", "9.5")
    monkeypatch.setenv("MU_LIFECYCLE__ADJUDICATOR_MAX_TOKENS", "999")
    monkeypatch.setenv("MU_LIFECYCLE__ADJUDICATOR_TEMPERATURE", "0.7")

    settings = get_engine_settings()

    assert settings.lifecycle.maintenance_interval_s == 120
    assert settings.lifecycle.session_idle_s == 30
    assert settings.lifecycle.batch_size == 5
    assert settings.lifecycle.adjudication_budget_per_sweep == 17
    assert settings.lifecycle.adjudication_degrade_threshold_s == pytest.approx(9.5)
    assert settings.lifecycle.adjudicator_max_tokens == 999
    assert settings.lifecycle.adjudicator_temperature == pytest.approx(0.7)


def test_c2_c3_overrides_all_land_together_in_one_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same composed-together proof as the Group-A test above, for every NEWLY-reachable
    Group B/C knob — one flat env namespace, several different subtrees, one process."""
    monkeypatch.setenv("MU_EXTRACTION__MIN_TOKENS", "7")
    monkeypatch.setenv("MU_MODEL__EMBED_MODEL", "custom-embed-id")
    monkeypatch.setenv("MU_MODEL_CATALOG__DEFAULT_EMBED_BACKEND", "custom_backend")
    monkeypatch.setenv("MU_MODEL_CATALOG__ROUTER__DEFAULT_CONTEXT_WINDOW", "4242")
    monkeypatch.setenv("MU_RETRY__MAX_ATTEMPTS", "9")
    monkeypatch.setenv("MU_OBSERVABILITY__DURABLE_AUDIT_QUEUE_MAX", "4096")
    monkeypatch.setenv("MU_LEDGER__REDIS_IO_TIMEOUT_S", "12.0")
    monkeypatch.setenv("MU_LIFECYCLE__ADJUDICATOR_MAX_TOKENS", "999")

    settings = get_engine_settings()

    assert settings.extraction.min_tokens == 7
    assert settings.model.embed_model == "custom-embed-id"
    assert settings.model_catalog.default_embed_backend == "custom_backend"
    assert settings.model_catalog.router.default_context_window == 4242
    assert settings.retry.max_attempts == 9
    assert settings.observability.durable_audit_queue_max == 4096
    assert settings.ledger.redis_io_timeout_s == pytest.approx(12.0)
    assert settings.lifecycle.adjudicator_max_tokens == 999
