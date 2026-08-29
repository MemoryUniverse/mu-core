"""Put ``mu-core/eval`` on ``sys.path`` so ``mu_eval`` imports without being an installed package.

``mu_eval`` is deliberately not a distribution (see ``mu_eval/__init__.py`` for why), so nothing
in ``uv.lock`` installs it. This conftest is the one line that makes ``eval/tests`` runnable via
the sanctioned ``infra/mu-vm/vm_test.sh mu-core eval/tests/...`` path — the repo-root conftest's
VM guard still applies, unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_ROOT = str(Path(__file__).resolve().parent)
if _EVAL_ROOT not in sys.path:
    sys.path.insert(0, _EVAL_ROOT)
