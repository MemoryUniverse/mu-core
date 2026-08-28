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


# --- BEGIN VM-ONLY TEST GUARD (CLAUDE.md rule 13) ---
def pytest_cmdline_main(config: object) -> int | None:
    """Refuse to run a SUITE on the developer laptop. See root CLAUDE.md rule 13.

    Suites run on `mu-dev-vm` via `infra/mu-vm/vm_test.sh <repo>`. This is enforced here rather
    than merely written down, because the written rule was ignored repeatedly and a full suite
    costs ~2.5 GB of pytest RSS on a 15 GB laptop that is also running the agents — twice it
    exhausted swap and killed the session mid-run.

    It is not only about footprint. A local run reads the developer's working tree, its caches and
    its ``sys.path``: mu-core's own CI line passes here with 1568 tests and, under
    ``python -m pytest``, collects ZERO with an error that reads like a healthy deselect count.
    A local green does not tell you what a clean checkout or a CI runner sees.

    ESCAPE HATCHES, in order of preference:
      * run it on the VM: ``infra/mu-vm/vm_test.sh <repo> [args]``  (sets MU_ON_VM=1 there)
      * one file runs on the VM too: ``vm_test.sh <repo> tests/x/test_y.py``
      * a deliberate local full run: ``MU_ALLOW_LOCAL_TESTS=1 pytest ...`` — say why in your report.
    """
    import os
    import sys

    if os.environ.get("MU_ON_VM") or os.environ.get("MU_ALLOW_LOCAL_TESTS"):
        return None
    if os.environ.get("CI"):
        return None

    sys.stderr.write(
        "\n"
        "REFUSED: suites run on the VM, not on this laptop (root CLAUDE.md rule 13).\n"
        "  ->  infra/mu-vm/vm_test.sh <repo> [pytest args]\n"
        "      (~15x faster; the stores are localhost there)\n"
        "  ->  ONE file on the VM too: vm_test.sh <repo> tests/x/test_y.py\n"
        "  ->  deliberate local full run: MU_ALLOW_LOCAL_TESTS=1 (and say why)\n"
        "\n"
    )
    return 4


# --- END VM-ONLY TEST GUARD ---
