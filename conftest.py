"""Repo-root pytest conftest — one job: no test run ever writes ``__pycache__``.

**Why this is here and not an ``env`` entry under ``[tool.pytest.ini_options]``.** The obvious
fix — ``env = ["PYTHONDONTWRITEBYTECODE=1"]`` in ``pyproject.toml`` — does not work, in two
independent ways, and both were checked rather than assumed:

1. ``env`` is not a pytest ini key; it belongs to the ``pytest-env`` plugin, which this repo does
   not depend on. ``addopts`` carries ``--strict-config``, so an unknown key is a hard
   ``ERROR: Unknown config option: env`` and **zero tests run** — it would break every gate in
   the repo rather than harden one.
2. Even with the plugin installed it would be too late to do anything. CPython reads
   ``PYTHONDONTWRITEBYTECODE`` exactly once, at interpreter start, into
   ``sys.dont_write_bytecode``; ``importlib`` consults that flag, never the environment. Setting
   the variable from inside a running process leaves ``sys.dont_write_bytecode`` ``False``.

So the flag itself is set, at the earliest point this repo controls. pytest imports the rootdir
``conftest.py`` before it imports any test module — and therefore before any test module imports
``mu_engine``/``mu_contracts``/``mu_local`` — so every source file the suite touches is compiled
in memory and nothing is written next to it.

**What this prevents.** A stale ``.pyc`` left behind by an earlier tree (a since-renamed or
deleted module still shadowed by its cached bytecode) reported five tests RED against a source
tree that was correct. That failure mode costs a full investigation and reproduces on no other
machine, which is the worst combination a test signal can have; a run that writes no bytecode
cannot have it. Nothing about test SEMANTICS changes — the only cost is recompiling on each run.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
