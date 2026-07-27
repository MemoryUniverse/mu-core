"""Relational store — the full SQLAlchemy 2.x schema + Alembic migrations (spec §2).

``schema.Base.metadata`` is the single source of truth for the content-free control-plane +
mirror tables; ``migrations/`` carries the forward+reversible Alembic revisions
(``alembic.ini`` sources its DSN from the central Settings tree, never a literal).
"""

from __future__ import annotations

from mu_engine.storage.relational.schema import Base

__all__ = ["Base"]
