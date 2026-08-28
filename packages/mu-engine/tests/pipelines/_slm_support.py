"""Shared REAL-local-SLM wiring for the ``tests/pipelines`` integration modules.

**Why this module exists — it is import plumbing, and the plumbing was load-bearing.**
``test_conflict_adjudicator_int.py`` reused ``test_distill_llm_slm_int.py``'s proven SLM shape
(the right call: one wiring, not two) but reached for it with an ABSOLUTE cross-module import::

    from tests.pipelines.test_distill_llm_slm_int import _SLM_CFG, _SLM_UP, _build_slm_catalog

That import is only valid under ONE of the two module names pytest may legitimately give these
files, and which one it picks is a function of ``sys.path`` — not of this repo. With
``--import-mode=importlib`` + ``consider_namespace_packages = true`` (both set in the root
``pyproject.toml``), ``_pytest.pathlib.resolve_pkg_root_and_module_name`` resolves this very file
two different ways, MEASURED on a clean checkout at be70c3d:

* repo root NOT on ``sys.path`` (the ``pytest`` console script) ->
  pkg-root ``packages/mu-engine``, module ``tests.pipelines.test_conflict_adjudicator_int``.
  ``sys.modules["tests"]`` exists, so the absolute import resolves.
* repo root ON ``sys.path`` (``python -m pytest`` — pytest's own documented invocation, which
  prepends the CWD) -> pkg-root the REPO ROOT, module
  ``packages.mu-engine.tests.pipelines.test_conflict_adjudicator_int``. ``tests`` is never a
  top-level module, and collection dies with ``ModuleNotFoundError: No module named 'tests'``,
  which pytest reports as ``Interrupted: 1 error during collection`` — the WHOLE suite then runs
  ZERO tests while the summary line still reads like a healthy deselect count.

A RELATIVE import (``from ._slm_support import ...``) resolves through ``__package__``, so it is
correct under BOTH names and cannot be re-broken by a ``sys.path`` change. Hoisting the wiring
here also ends the "one test module imports another test module" coupling outright: neither test
file now imports the other, so neither can drag the other's module-level probe, ``pytestmark`` or
fixtures into its own import graph by accident.

Nothing about the wiring itself changed — ``SlmTestSettings`` / ``_slm_reachable`` / ``_SLM_CFG`` /
``_SLM_UP`` / ``_build_slm_catalog`` are moved here verbatim from
``test_distill_llm_slm_int.py``, names included, so both call sites read exactly as they did.

Not collected by pytest: the filename matches no ``python_files`` pattern. This is support code,
not a test module, which is the other half of why the coupling is gone.
"""

from __future__ import annotations

import urllib.error
import urllib.request

from pydantic_settings import BaseSettings, SettingsConfigDict

from mu_engine.providers.catalog import ModelDeployment, ModelKind, ProviderKind, ProviderRecord
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings, default_local_catalog

# ---------------------------------------------------------------------------------------------
# Test-profile settings — central-config home for the SLM wiring (DEV-STANDARDS rule 3: no
# hardcoded model id / host / key lives in the tests' LOGIC, only as named field defaults here,
# env-overridable exactly like every other Settings subtree in this repo). Scoped to the tests
# that import it: it NEVER touches ``ModelSettings()``'s production Azure default (settings.py:53
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
