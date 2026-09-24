"""Real-store fixtures for the surface suite's integration half.

Re-exported verbatim from ``tests/lifecycle/conftest.py`` rather than re-declared (DEV-STANDARDS
rule 6, DRY): that module is the one definition of "a real mu-dev-cache/qdrant/falkordb adapter,
fail-loud if the container is down", and a second copy would drift. Only the fixtures the surface
suite actually uses are re-exported, so an unused one does not silently become a dependency.
"""

from __future__ import annotations

from tests.lifecycle.conftest import (
    falkor_db,
    ltm,
    make_item,
    make_ns,
    make_stm,
    mtm,
    qdrant_client,
    settings,
    uid,
    valkey_client,
)

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
