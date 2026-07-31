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
