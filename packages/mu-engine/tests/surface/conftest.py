"""Real-store fixtures for the surface suite's integration half.

Re-exported verbatim from ``tests/lifecycle/conftest.py`` rather than re-declared (DEV-STANDARDS
rule 6, DRY): that module is the one definition of "a real mu-dev-cache/qdrant/falkordb adapter,
fail-loud if the container is down", and a second copy would drift. Only the fixtures the surface
suite actually uses are re-exported, so an unused one does not silently become a dependency.

**Loaded BY PATH, not by the absolute name ``tests.lifecycle.conftest``.** The root
``pyproject.toml`` sets ``--import-mode=importlib`` + ``consider_namespace_packages = true``, which
makes the module NAME pytest gives a test file a function of ``sys.path``, not of the repo
(``.github/workflows/ci.yml``'s own "COLLECTION INTEGRITY" step documents the measurement):

* repo root NOT on ``sys.path`` (plain ``pytest``)   -> ``tests.lifecycle.conftest``
* repo root ON ``sys.path`` (``python -m pytest``)   ->
  ``packages.mu-engine.tests.lifecycle.conftest``

Under the second shape ``tests`` is not a top-level module, so a bare
``from tests.lifecycle.conftest import ...`` raises ``ModuleNotFoundError: No module named
'tests'``, pytest reports ``Interrupted: 1 error during collection`` and **ZERO tests run** — the
exact failure that CI step exists to catch, and which this file reintroduced (MEASURED on the VM
at ``dev/mlm-build@58385c5``: ``2255 tests collected, 1 error``, exit 2, while a plain ``pytest``
on the identical tree was 2269 green). ``packages/mu-engine`` cannot be imported as a package
either (the hyphen is not a valid identifier), so a relative import is not available. Loading the
sibling conftest by FILE PATH is independent of both ``sys.path`` shapes and of the directory's
name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_LIFECYCLE_CONFTEST = Path(__file__).resolve().parent.parent / "lifecycle" / "conftest.py"


def _load_lifecycle_conftest() -> ModuleType:
    """Import ``../lifecycle/conftest.py`` under a private, sys.path-independent module name.

    Pytest loads that same file separately as the lifecycle directory's own conftest plugin; this
    second load is for the RE-EXPORT below only. It is safe because that module has no top-level
    side effects — only imports and fixture definitions (no container connection is opened until a
    fixture is requested), and pytest binds a fixture by the name it finds in THIS module's
    namespace, not by the defining module's identity.
    """
    name = "mu_engine_tests_lifecycle_conftest"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, _LIFECYCLE_CONFTEST)
    if spec is None or spec.loader is None:  # pragma: no cover - a moved/renamed sibling
        raise ImportError(f"cannot load the lifecycle conftest from {_LIFECYCLE_CONFTEST}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_lifecycle = _load_lifecycle_conftest()

# Re-exported VERBATIM (the same objects, not copies) — only the fixtures the surface suite
# actually uses, so an unused one does not silently become a dependency.
falkor_db = _lifecycle.falkor_db
ltm = _lifecycle.ltm
make_item = _lifecycle.make_item
make_ns = _lifecycle.make_ns
make_stm = _lifecycle.make_stm
mtm = _lifecycle.mtm
qdrant_client = _lifecycle.qdrant_client
settings = _lifecycle.settings
uid = _lifecycle.uid
valkey_client = _lifecycle.valkey_client

__all__ = [
    "falkor_db",
    "ltm",
    "make_item",
    "make_ns",
    "make_stm",
    "mtm",
    "qdrant_client",
    "settings",
    "uid",
    "valkey_client",
]
