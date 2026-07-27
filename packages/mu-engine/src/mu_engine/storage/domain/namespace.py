"""η — the tenancy partition key.

SINGLE AUTHORITATIVE HOME (MAJOR-2, no split-brain): ``Namespace``/``Visibility`` are defined
ONCE in ``mu-contracts`` (``mu_contracts.domain.model.memory``), the shape pinned by
CANONICAL-CONTRACTS.md §1 (the un-collapsed η — ADR 0026 — with the stronger separator-injection
guard that also rejects ``\\x00``). This module is a thin RE-EXPORT so existing
``from mu_engine.storage.domain.namespace import Namespace, Visibility`` call sites keep working
while binding to the one canonical type — the engine never re-defines η.

Import direction is legal under the import-linter ``core-layers`` contract: engine sits ABOVE
contracts and may import down into it.
"""

from __future__ import annotations

from mu_contracts.domain.model.memory import Namespace, Visibility

__all__ = ["Namespace", "Visibility"]
