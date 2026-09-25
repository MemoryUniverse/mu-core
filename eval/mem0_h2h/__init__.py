"""The mem0-OSS head-to-head arm of the LoCoMo retrieval harness.

Deliberately NOT part of `mu_eval`: it imports `mem0ai`, a third-party package that is not in
mu-core's uv lock and must never become a dependency of anything shipped. It reuses `mu_eval`'s
LoCoMo loader (pure pydantic, no engine imports) so that BOTH arms of the head-to-head read the
same rows, the same evidence labels and the same adversarial exclusion — one loader, never two
that could quietly disagree.
"""

import sys


def emit(line: str = "") -> None:
    """Progress/result output for these measurement scripts.

    A bare ``print`` is banned repo-wide (``ruff.toml`` selects ``T20``, DEV-STANDARDS "no
    print"), and structlog is the wrong tool for a CLI whose entire job is to print numbers a
    human reads. ``mu_eval/__main__.py:38`` already solved this the same way; this is that same
    ``_print``, shared across the three scripts in this package rather than copied into each.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()
